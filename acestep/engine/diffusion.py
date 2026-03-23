from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Callable, Optional, Tuple, List, Union

import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from .conditions import PreparedCondition, ConditionSet
from .masking import LatentNoiseMask

logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for the DiffusionEngine loop.

    Attributes:
        infer_steps: Number of diffusion steps.
        infer_method: Solver type, "ode" (Euler) or "sde" (stochastic).
        shift: Timestep shift for flow matching. ACEStep turbo uses 3.0.
        seed: Random seed for noise generation.
        use_cache: Enable KV caching for cross-attention. Disabled by
            default to match ComfyUI behavior (no KV caching). Enable for
            faster inference when exact ComfyUI parity is not required.
        noise_on_cpu: Generate noise on CPU in [B,D,T] layout then
            transpose to [B,T,D], matching ComfyUI's RandomNoise node.
            When False, uses the HF model's prepare_noise (GPU, [B,T,D]).
        timesteps: Explicit timestep schedule (overrides infer_steps/shift).
        denoise: Denoising strength in [0, 1]. Controls how much of the
            source audio to preserve vs regenerate:
              1.0 = generate from pure noise (full generation)
              0.5 = start halfway through the noise schedule (style transfer)
              0.0 = no denoising (passthrough)
            When < 1.0, source_latents must be provided to generate().
            The schedule is computed as int(steps/denoise) full steps, then
            the last (steps+1) entries are used (matching ComfyUI behavior).
    """

    infer_steps: int = 8
    infer_method: str = "ode"
    shift: float = 3.0
    seed: Optional[Union[int, List[int]]] = None
    use_cache: bool = False
    noise_on_cpu: bool = True
    timesteps: Optional[List[float]] = None
    denoise: float = 1.0
    x0_target_gate: float = 0.0


class DiffusionEngine:
    """Enhanced diffusion loop supporting composable multi-condition generation.

    Four execution paths (selected automatically per step):
      - Fast path: single condition, no temporal weights. Full KV caching.
      - Switch path: two conditions with complementary step_ranges. KV caching
        with reset at the switch point.
      - Batched path: multiple conditions sharing the same hook_ref (model
        weights). Pads encoder_hidden_states, runs a single decoder call
        with batch_size=N, splits and blends velocities per-frame.
      - Sequential path: multiple conditions with different hook_refs
        (different LoRAs). Groups by hook_ref, batches within each group,
        runs sequentially across groups with weight switching via
        apply_hooks_fn.

    Supports partial denoising (denoise < 1.0) for audio-to-audio generation,
    and two-sided noise mask blending matching ComfyUI's KSamplerX0Inpaint.
    """

    def __init__(self, model, trt_engine_path=None):
        """
        Args:
            model: AceStepConditionGenerationModel instance.
            trt_engine_path: Optional path to a TRT decoder engine file.
                When provided, the engine is loaded via polygraphy and
                all decoder calls are routed through TensorRT. All engine
                modulations (temporal blending, velocity scaling, noise
                masks, etc.) continue to work because they operate on the
                velocity output, not inside the decoder.
        """
        self.model = model
        self.decoder = model.decoder
        self._compiled_loop: Optional[Callable] = None
        self._compiled_loop_sde: Optional[Callable] = None

        # TRT state (owned directly, no wrapper class).
        # Uses polygraphy engine loading and polygraphy CUDA stream to
        # avoid Blackwell multi-engine kernel slowdown.
        self._trt_engine = None
        self._trt_ctx = None
        self._trt_stream = None
        self._trt_buf_cache: dict[tuple, dict] = {}

        if trt_engine_path is not None:
            self.load_trt_engine(trt_engine_path)

    # ------------------------------------------------------------------
    # TRT engine management
    # ------------------------------------------------------------------

    def load_trt_engine(self, engine_path):
        """Load a TRT decoder engine via polygraphy.

        Uses polygraphy.backend.trt.engine_from_bytes (not
        trt.Runtime().deserialize_cuda_engine) to avoid process-global
        TRT state corruption on Blackwell GPUs with multiple engines.
        Shares the process-wide polygraphy CUDA stream with VAE engines.
        """
        from pathlib import Path
        from polygraphy.backend.common import bytes_from_path
        from polygraphy.backend.trt import engine_from_bytes
        from acestep.nodes.vae_nodes import _get_trt_stream

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        logger.info("Loading TRT decoder engine from %s ...", engine_path)
        self._trt_engine = engine_from_bytes(bytes_from_path(str(engine_path)))
        self._trt_ctx = self._trt_engine.create_execution_context()
        self._trt_stream = _get_trt_stream()
        self._trt_buf_cache = {}

        # Detect I/O dtypes from engine (fp16 for mixed-precision, fp32 for legacy)
        import tensorrt as trt
        _trt_dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
        }
        hs_trt_dtype = self._trt_engine.get_tensor_dtype("hidden_states")
        self._trt_io_dtype = _trt_dtype_map.get(hs_trt_dtype, torch.float32)
        logger.info("TRT decoder engine ready (io_dtype=%s)", self._trt_io_dtype)

    def _trt_decoder_step(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder step through TRT with pre-allocated buffers.

        Handles odd-T padding. Caches buffers by shape for reuse across
        steps. Calls execute_async_v3 on the shared polygraphy stream.
        """
        orig_T = hidden_states.shape[1]
        pad = orig_T % 2 == 1
        eff_T = orig_T + 1 if pad else orig_T

        key = (
            (hidden_states.shape[0], eff_T, 64),
            tuple(timestep.shape),
            tuple(encoder_hidden_states.shape),
            (context_latents.shape[0], eff_T, 128),
        )

        if key not in self._trt_buf_cache:
            ctx = self._trt_ctx
            dev = hidden_states.device
            hs_shape, ts_shape, enc_shape, cl_shape = key
            io_dtype = self._trt_io_dtype

            bufs = {
                "hidden_states": torch.empty(hs_shape, dtype=io_dtype, device=dev),
                "timestep": torch.empty(ts_shape, dtype=torch.float32, device=dev),
                "encoder_hidden_states": torch.empty(enc_shape, dtype=io_dtype, device=dev),
                "context_latents": torch.empty(cl_shape, dtype=io_dtype, device=dev),
            }
            for name, buf in bufs.items():
                ctx.set_input_shape(name, tuple(buf.shape))
                ctx.set_tensor_address(name, buf.data_ptr())

            out_shape = tuple(ctx.get_tensor_shape("velocity"))
            out_buf = torch.empty(out_shape, dtype=io_dtype, device=dev)
            ctx.set_tensor_address("velocity", out_buf.data_ptr())

            self._trt_buf_cache[key] = {"bufs": bufs, "output": out_buf}
            logger.info(
                "Allocated TRT buffers for shapes: hs=%s enc=%s",
                list(hs_shape), list(enc_shape),
            )

        entry = self._trt_buf_cache[key]
        bufs = entry["bufs"]

        if pad:
            bufs["hidden_states"][:, :orig_T, :].copy_(hidden_states)
            bufs["hidden_states"][:, orig_T:, :].zero_()
            bufs["context_latents"][:, :orig_T, :].copy_(context_latents)
            bufs["context_latents"][:, orig_T:, :].zero_()
        else:
            bufs["hidden_states"].copy_(hidden_states)
            bufs["context_latents"].copy_(context_latents)
        bufs["timestep"].copy_(timestep)
        bufs["encoder_hidden_states"].copy_(encoder_hidden_states)

        ctx = self._trt_ctx
        for name, buf in bufs.items():
            ctx.set_tensor_address(name, buf.data_ptr())
        ctx.set_tensor_address("velocity", entry["output"].data_ptr())

        ctx.execute_async_v3(self._trt_stream.ptr)
        self._trt_stream.synchronize()

        output = entry["output"]
        return output[:, :orig_T, :] if pad else output

    # ------------------------------------------------------------------
    # CUDA graph fast path
    # ------------------------------------------------------------------

    def _can_use_fast_path(
        self,
        condition_set: ConditionSet,
        config: DiffusionConfig,
        latent_mask: Optional[LatentNoiseMask],
        velocity_scale: Optional[torch.Tensor],
        sde_denoise_curve: Optional[torch.Tensor],
        x0_target: Optional[torch.Tensor],
        apply_hooks_fn: Optional[Callable],
        negative_condition_set: Optional[ConditionSet] = None,
        ode_noise_curve: Optional[torch.Tensor] = None,
    ) -> bool:
        """Check if a compiled fast path is viable.

        Supports both ODE and SDE with all per-frame curves
        (velocity_scale, ode_noise_curve, sde_denoise_curve,
        initial_noise_curve). These are baked into the compiled loop as
        always-present tensor args with no-op sentinel values when inactive.

        Still excluded: TRT decoder (standard path routes through
        _decoder_call which handles TRT; the compiled loop bypasses it),
        inpainting masks, x0_target blending, per-condition hooks,
        multi-condition, and CFG.
        """
        if self._trt_engine is not None:
            return False
        if latent_mask is not None:
            return False
        if x0_target is not None:
            return False
        if apply_hooks_fn is not None:
            return False
        if not condition_set.is_single_condition:
            return False
        if negative_condition_set is not None:
            return False
        return True

    def _generate_fast(
        self,
        condition_set: ConditionSet,
        config: DiffusionConfig,
        noise: torch.Tensor,
        t_schedule: torch.Tensor,
        source_latents: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        velocity_scale: Optional[torch.Tensor] = None,
        ode_noise_curve: Optional[torch.Tensor] = None,
        sde_denoise_curve: Optional[torch.Tensor] = None,
        initial_noise_curve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compiled fast path: single condition, ODE or SDE, with curves.

        Handles initial_noise_curve before the loop, then delegates to
        a compiled inner loop. ODE and SDE use separate compiled
        functions (cached in _compiled_loop and _compiled_loop_sde) to
        avoid branching inside the compiled graph.
        """
        device = noise.device
        dtype = noise.dtype
        bsz = noise.shape[0]
        infer_steps = len(t_schedule) - 1
        cond = condition_set.conditions[0]
        is_sde = config.infer_method == "sde"

        # Initial state (with optional per-frame noise/source mixing)
        t_start = t_schedule[0].item()
        if initial_noise_curve is not None and source_latents is not None:
            curve = self._normalize_curve(initial_noise_curve)
            xt = curve * noise + (1.0 - curve) * source_latents
        elif config.denoise < 1.0 and source_latents is not None:
            xt = t_start * noise + (1.0 - t_start) * source_latents
        else:
            xt = noise.clone()

        # Precompute schedule tensors
        t_curr_vec = t_schedule[:-1].unsqueeze(-1).expand(infer_steps, bsz)  # [steps, B]
        t_next_vec = t_schedule[1:].reshape(infer_steps, 1, 1)  # [steps, 1, 1]

        # Prepare curve tensors (always-present; no-op sentinels when inactive).
        if velocity_scale is not None:
            vs = self._normalize_curve(velocity_scale).to(device=device, dtype=dtype)
        else:
            vs = torch.ones(1, 1, 1, device=device, dtype=dtype)

        if is_sde:
            # SDE path
            # Source latents for sde_denoise_curve blending (zeros if absent)
            if source_latents is not None:
                src = source_latents.to(device=device, dtype=dtype)
            else:
                src = torch.zeros_like(xt)

            if sde_denoise_curve is not None:
                sdc = self._normalize_curve(sde_denoise_curve).to(device=device, dtype=dtype)
            else:
                # When no sde_denoise_curve: curve=1.0 means full x0_pred re-noise
                # (standard SDE behavior, no source blending)
                sdc = torch.ones(1, 1, 1, device=device, dtype=dtype)

            if self._compiled_loop_sde is None:
                self._compiled_loop_sde = torch.compile(
                    self._fast_loop_sde,
                    backend="inductor",
                    dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )

            return self._compiled_loop_sde(
                self.decoder, xt, t_curr_vec, t_next_vec,
                attention_mask,
                cond.encoder_hidden_states,
                cond.encoder_attention_mask,
                cond.context_latents,
                infer_steps,
                vs, sdc, src,
            )
        else:
            # ODE path
            dt_vec = (t_schedule[:-1] - t_schedule[1:]).clone()
            dt_vec[-1] = t_schedule[-2]
            dt_vec = dt_vec.reshape(infer_steps, 1, 1)

            if ode_noise_curve is not None:
                onc = self._normalize_curve(ode_noise_curve).to(device=device, dtype=dtype)
            else:
                onc = torch.zeros(1, 1, 1, device=device, dtype=dtype)

            if self._compiled_loop is None:
                self._compiled_loop = torch.compile(
                    self._fast_loop,
                    backend="inductor",
                    dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )

            return self._compiled_loop(
                self.decoder, xt, t_curr_vec, dt_vec, t_next_vec,
                attention_mask,
                cond.encoder_hidden_states,
                cond.encoder_attention_mask,
                cond.context_latents,
                infer_steps,
                vs, onc,
            )

    @staticmethod
    def _fast_loop(
        decoder,
        xt: torch.Tensor,
        t_curr_vec: torch.Tensor,
        dt_vec: torch.Tensor,
        t_next_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        enc_hs: torch.Tensor,
        enc_mask: torch.Tensor,
        ctx_lat: torch.Tensor,
        infer_steps: int,
        velocity_scale: torch.Tensor,
        ode_noise_curve: torch.Tensor,
    ) -> torch.Tensor:
        """ODE inner loop for torch.compile.

        velocity_scale and ode_noise_curve are always-present tensors.
        When inactive, they are ones/zeros respectively, making the
        operations mathematical no-ops that the compiler can optimize.
        """
        for i in range(infer_steps):
            vt = decoder(
                hidden_states=xt,
                timestep=t_curr_vec[i],
                timestep_r=t_curr_vec[i],
                attention_mask=attention_mask,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                context_latents=ctx_lat,
                use_cache=False,
                past_key_values=None,
            )[0]
            vt = vt * velocity_scale
            xt = xt - vt * dt_vec[i]
            # ODE noise injection (no-op when zeros; naturally zero on
            # final step since t_next_vec[-1] = 0)
            xt = xt + torch.randn_like(xt) * ode_noise_curve * t_next_vec[i]
        return xt

    @staticmethod
    def _fast_loop_sde(
        decoder,
        xt: torch.Tensor,
        t_curr_vec: torch.Tensor,
        t_next_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        enc_hs: torch.Tensor,
        enc_mask: torch.Tensor,
        ctx_lat: torch.Tensor,
        infer_steps: int,
        velocity_scale: torch.Tensor,
        sde_denoise_curve: torch.Tensor,
        source_latents: torch.Tensor,
    ) -> torch.Tensor:
        """SDE inner loop for torch.compile.

        At each step: predict x0 from velocity, then re-noise with
        per-frame sde_denoise_curve blending between full re-noise
        (toward x0_pred) and source-converging re-noise (toward
        source_latents).

        sde_denoise_curve=1.0 is standard SDE (full re-noise from x0).
        sde_denoise_curve=0.0 pulls toward source_latents (preserve).
        """
        for i in range(infer_steps):
            vt = decoder(
                hidden_states=xt,
                timestep=t_curr_vec[i],
                timestep_r=t_curr_vec[i],
                attention_mask=attention_mask,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                context_latents=ctx_lat,
                use_cache=False,
                past_key_values=None,
            )[0]
            vt = vt * velocity_scale
            # x0 prediction: x0 = xt - vt * t_curr
            t_curr_broad = t_curr_vec[i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            x0_pred = xt - vt * t_curr_broad
            # Re-noise with sde_denoise_curve blending.
            # t_next=0 on final step makes this return x0_pred directly.
            t_next = t_next_vec[i]  # [1, 1]
            sde_noise = torch.randn_like(xt)
            xt_full = t_next * sde_noise + (1.0 - t_next) * x0_pred
            xt_source = t_next * sde_noise + (1.0 - t_next) * source_latents
            xt = sde_denoise_curve * xt_full + (1.0 - sde_denoise_curve) * xt_source
        return xt

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _prepare_noise_cpu(
        self, ref_cond: PreparedCondition, seed: Optional[Union[int, List[int]]]
    ) -> torch.Tensor:
        """Generate noise on CPU in [B,D,T] layout, then transpose to [B,T,D].

        This matches ComfyUI's RandomNoise node which generates noise on CPU
        with torch.manual_seed, producing different values than GPU generation.
        """
        bsz = ref_cond.batch_size
        T = ref_cond.seq_len
        D = ref_cond.context_latents.shape[-1] // 2  # context_latents is [B,T,D*2]
        device = ref_cond.device
        dtype = ref_cond.dtype

        if seed is not None and not isinstance(seed, list):
            torch.manual_seed(int(seed))
            noise_bdt = torch.randn(bsz, D, T, device="cpu", dtype=torch.float32)
        elif isinstance(seed, list):
            noise_list = []
            for s in seed:
                if s is not None and s >= 0:
                    torch.manual_seed(int(s))
                noise_list.append(torch.randn(1, D, T, device="cpu", dtype=torch.float32))
            noise_bdt = torch.cat(noise_list, dim=0)
        else:
            noise_bdt = torch.randn(bsz, D, T, device="cpu", dtype=torch.float32)

        return noise_bdt.movedim(-1, -2).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Timestep schedule
    # ------------------------------------------------------------------

    def _build_timestep_schedule(
        self, config: DiffusionConfig, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build the timestep schedule, respecting denoise truncation.

        When denoise < 1.0, computes int(steps/denoise) full steps with
        shift applied, then takes the last (steps+1) entries. This matches
        ComfyUI's BasicScheduler.set_steps() behavior.
        """
        if config.timesteps is not None:
            return torch.tensor(config.timesteps, device=device, dtype=dtype)

        steps = config.infer_steps
        denoise = config.denoise

        if denoise <= 0.0:
            # No denoising: single-entry schedule (t=0 -> t=0)
            return torch.zeros(2, device=device, dtype=dtype)
        elif denoise >= 1.0:
            full_steps = steps
        else:
            # Compute extended schedule, then truncate
            full_steps = int(steps / denoise)

        t_schedule = torch.linspace(
            1.0, 0.0, full_steps + 1, device=device, dtype=dtype
        )
        if config.shift != 1.0:
            t_schedule = (
                config.shift * t_schedule
                / (1 + (config.shift - 1) * t_schedule)
            )

        if denoise < 1.0 and denoise > 0.0:
            # Take the last (steps+1) entries
            t_schedule = t_schedule[-(steps + 1):]

        return t_schedule

    # ------------------------------------------------------------------
    # Noise mask blending (two-sided, matching KSamplerX0Inpaint)
    # ------------------------------------------------------------------

    def _mask_pre_blend(
        self,
        xt: torch.Tensor,
        t_curr: float,
        latent_mask: LatentNoiseMask,
        step_idx: int,
        infer_steps: int,
    ) -> torch.Tensor:
        """Pre-decoder blend: preserved regions get properly-noised original.

        x_input = mask * xt + (1-mask) * (t * noise + (1-t) * original)

        This gives the model correct context at the current noise level
        in preserved regions, preventing boundary artifacts.
        """
        mask = latent_mask.get_mask(step_idx, infer_steps)
        noise = latent_mask.ensure_noise(xt.device, xt.dtype)

        # Flow matching noise scaling: sigma * noise + (1 - sigma) * x0
        noised_original = t_curr * noise + (1.0 - t_curr) * latent_mask.original_latents
        return mask * xt + (1.0 - mask) * noised_original

    def _mask_post_blend_x0(
        self,
        x0_pred: torch.Tensor,
        latent_mask: LatentNoiseMask,
        step_idx: int,
        infer_steps: int,
    ) -> torch.Tensor:
        """Post-decoder blend on x0 prediction: preserved regions get
        the clean original.

        x0_blended = mask * x0_pred + (1-mask) * original
        """
        mask = latent_mask.get_mask(step_idx, infer_steps)
        return mask * x0_pred + (1.0 - mask) * latent_mask.original_latents

    # ------------------------------------------------------------------
    # Decoder call
    # ------------------------------------------------------------------

    def _decoder_call(
        self,
        xt: torch.Tensor,
        t_curr_tensor: torch.Tensor,
        condition: PreparedCondition,
        attention_mask: torch.Tensor,
        use_cache: bool = True,
        past_key_values: Optional[EncoderDecoderCache] = None,
    ) -> Tuple[torch.Tensor, Optional[EncoderDecoderCache]]:
        """Single decoder forward pass.

        When a TRT engine is loaded, routes through TensorRT via
        execute_async_v3 on the shared polygraphy stream (no KV cache).
        All engine modulations continue to work unchanged because they
        operate on the velocity tensor returned here.
        """
        if self._trt_engine is not None:
            vt = self._trt_decoder_step(
                hidden_states=xt,
                timestep=t_curr_tensor,
                encoder_hidden_states=condition.encoder_hidden_states,
                context_latents=condition.context_latents,
            )
            return vt, None

        decoder_outputs = self.decoder(
            hidden_states=xt,
            timestep=t_curr_tensor,
            timestep_r=t_curr_tensor,
            attention_mask=attention_mask,
            encoder_hidden_states=condition.encoder_hidden_states,
            encoder_attention_mask=condition.encoder_attention_mask,
            context_latents=condition.context_latents,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        vt = decoder_outputs[0]
        new_cache = decoder_outputs[1] if use_cache else None
        return vt, new_cache

    # ------------------------------------------------------------------
    # Batched decoder call
    # ------------------------------------------------------------------

    def _batched_decoder_call(
        self,
        xt: torch.Tensor,
        t_curr_tensor: torch.Tensor,
        conditions: List[PreparedCondition],
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Batched decoder call for conditions sharing the same model weights.

        Pads encoder_hidden_states to the max sequence length across
        conditions, stacks along the batch dimension, and runs a single
        forward pass. Returns a list of velocity tensors, one per condition.
        """
        N = len(conditions)
        B = xt.shape[0]

        # Repeat inputs for N conditions
        xt_batched = xt.repeat(N, 1, 1)
        attn_batched = attention_mask.repeat(N, 1)
        t_batched = t_curr_tensor.repeat(N)

        # Find max encoder sequence length
        max_enc_len = max(c.encoder_hidden_states.shape[1] for c in conditions)

        enc_hidden_list = []
        enc_mask_list = []
        ctx_list = []

        for c in conditions:
            enc = c.encoder_hidden_states    # [B, L, D]
            mask = c.encoder_attention_mask   # [B, L]
            L = enc.shape[1]

            if L < max_enc_len:
                pad = max_enc_len - L
                enc = torch.nn.functional.pad(enc, (0, 0, 0, pad))
                mask = torch.nn.functional.pad(mask, (0, pad), value=0)

            enc_hidden_list.append(enc)
            enc_mask_list.append(mask)
            ctx_list.append(c.context_latents)

        batched_cond = PreparedCondition(
            encoder_hidden_states=torch.cat(enc_hidden_list, dim=0),
            encoder_attention_mask=torch.cat(enc_mask_list, dim=0),
            context_latents=torch.cat(ctx_list, dim=0),
        )

        vt_all, _ = self._decoder_call(
            xt_batched, t_batched, batched_cond, attn_batched,
            use_cache=False, past_key_values=None,
        )

        return list(vt_all.split(B, dim=0))

    # ------------------------------------------------------------------
    # Velocity blending
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_velocities(
        velocity_cond_pairs: List[Tuple[torch.Tensor, PreparedCondition]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Blend velocity outputs using per-condition temporal weights.

        vt_composite = sum(vt_i * w_i) / sum(w_i)

        Temporal weights are broadcast from [T] or [B, T] to [B, T, 1].
        Conditions with no temporal_weight use uniform weight of 1.
        """
        numerator = None
        denominator = None

        for vt, cond in velocity_cond_pairs:
            if cond.temporal_weight is not None:
                w = cond.temporal_weight
                if w.ndim == 1:       # [T]
                    w = w.unsqueeze(0).unsqueeze(-1)   # -> [1, T, 1]
                elif w.ndim == 2:     # [B, T]
                    w = w.unsqueeze(-1)                # -> [B, T, 1]
            else:
                w = torch.ones(1, 1, 1, device=device, dtype=dtype)

            if numerator is None:
                numerator = vt * w
                denominator = w
            else:
                numerator = numerator + vt * w
                denominator = denominator + w

        return numerator / denominator.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Per-step execution paths
    # ------------------------------------------------------------------

    def _single_condition_step(
        self,
        xt: torch.Tensor,
        t_curr_tensor: torch.Tensor,
        condition: PreparedCondition,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: Optional[EncoderDecoderCache],
    ) -> Tuple[torch.Tensor, Optional[EncoderDecoderCache]]:
        """Fast / switch path: one decoder call with optional KV cache."""
        return self._decoder_call(
            xt, t_curr_tensor, condition, attention_mask, use_cache, past_key_values
        )

    def _multi_condition_step(
        self,
        xt: torch.Tensor,
        t_curr_tensor: torch.Tensor,
        conditions: List[PreparedCondition],
        attention_mask: torch.Tensor,
        apply_hooks_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Multi-condition step with automatic batched/sequential routing.

        Groups conditions by hook_ref. Within each group, uses a single
        batched decoder call (padding encoder_hidden_states to match).
        Across groups with different hook_refs, runs sequentially with
        weight switching via apply_hooks_fn.

        Falls back to sequential single calls when only one condition
        exists in a group (no padding overhead).
        """
        # Group conditions by hook_ref identity
        groups: OrderedDict[Optional[int], List[PreparedCondition]] = OrderedDict()
        for cond in conditions:
            key = id(cond.hook_ref) if cond.hook_ref is not None else None
            groups.setdefault(key, []).append(cond)

        velocity_pairs: List[Tuple[torch.Tensor, PreparedCondition]] = []

        for hook_key, group_conds in groups.items():
            # Switch model weights for this hook group
            if apply_hooks_fn is not None:
                apply_hooks_fn(group_conds[0].hook_ref)

            if len(group_conds) == 1:
                vt, _ = self._decoder_call(
                    xt, t_curr_tensor, group_conds[0], attention_mask,
                    use_cache=False, past_key_values=None,
                )
                velocity_pairs.append((vt, group_conds[0]))
            else:
                vt_list = self._batched_decoder_call(
                    xt, t_curr_tensor, group_conds, attention_mask,
                )
                for vt, cond in zip(vt_list, group_conds):
                    velocity_pairs.append((vt, cond))

        return self._blend_velocities(velocity_pairs, xt.device, xt.dtype)

    # ------------------------------------------------------------------
    # ODE / SDE integration with mask-aware x0 blending
    # ------------------------------------------------------------------

    def _integrate_step(
        self,
        xt: torch.Tensor,
        vt: torch.Tensor,
        t_curr: float,
        t_next: float,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
        infer_method: str,
        latent_mask: Optional[LatentNoiseMask],
        step_idx: int,
        infer_steps: int,
        sde_denoise_curve: Optional[torch.Tensor] = None,
        source_latents: Optional[torch.Tensor] = None,
        x0_target: Optional[torch.Tensor] = None,
        x0_target_curve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the next xt from the current state and velocity.

        When a latent_mask is present, blends the x0 prediction before
        integrating. This matches ComfyUI's two-sided mask blending:
        the Euler/SDE step is computed from a blended x0 so that
        preserved regions naturally converge to the clean original.

        When sde_denoise_curve is provided (SDE mode only), modulates the
        re-noise amount per frame. High curve values get full re-noise
        (exploratory, more transformation), low values pull toward
        source_latents (converge faster, preserve source).

        When x0_target and x0_target_curve are provided, blends the x0
        prediction per-frame toward the target latent before integration.
        """
        t_curr_tensor = t_curr * torch.ones((bsz,), device=device, dtype=dtype)

        # Compute denoised x0 from velocity
        x0_pred = self.model.get_x0_from_noise(xt, vt, t_curr_tensor)

        # Post-blend x0: preserved regions get the clean original
        if latent_mask is not None:
            x0_pred = self._mask_post_blend_x0(
                x0_pred, latent_mask, step_idx, infer_steps
            )

        # Per-frame x0 target blending
        if x0_target is not None and x0_target_curve is not None:
            x0_pred = (1.0 - x0_target_curve) * x0_pred + x0_target_curve * x0_target

        # Final step: return blended x0 directly
        if t_next <= 0:
            return x0_pred

        # SDE: re-noise the blended x0
        if infer_method == "sde":
            sde_noise = torch.randn_like(xt)

            if sde_denoise_curve is not None and source_latents is not None:
                curve = self._normalize_curve(sde_denoise_curve)
                xt_full = t_next * sde_noise + (1.0 - t_next) * x0_pred
                xt_source = t_next * sde_noise + (1.0 - t_next) * source_latents
                return curve * xt_full + (1.0 - curve) * xt_source
            else:
                noise_arg = latent_mask.ensure_noise(device, dtype) if latent_mask else None
                return self.model.renoise(x0_pred, t_next, noise=noise_arg)

        # ODE: Euler step using velocity derived from blended x0
        v_blended = (xt - x0_pred) / t_curr
        dt = t_curr - t_next
        dt_tensor = (
            dt * torch.ones((bsz,), device=device, dtype=dtype)
            .unsqueeze(-1).unsqueeze(-1)
        )
        return xt - v_blended * dt_tensor

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_curve(curve: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-frame curve to [B, T, 1] for element-wise ops."""
        if curve.ndim == 1:       # [T]
            return curve.unsqueeze(0).unsqueeze(-1)
        elif curve.ndim == 2:     # [B, T]
            return curve.unsqueeze(-1)
        return curve              # already [B, T, 1]

    @torch.no_grad()
    def generate(
        self,
        condition_set: ConditionSet,
        config: DiffusionConfig,
        attention_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[LatentNoiseMask] = None,
        source_latents: Optional[torch.Tensor] = None,
        apply_hooks_fn: Optional[Callable] = None,
        sde_denoise_curve: Optional[torch.Tensor] = None,
        velocity_scale: Optional[torch.Tensor] = None,
        initial_noise_curve: Optional[torch.Tensor] = None,
        x0_target: Optional[torch.Tensor] = None,
        x0_target_curve: Optional[torch.Tensor] = None,
        negative_condition_set: Optional[ConditionSet] = None,
        guidance_curve: Optional[torch.Tensor] = None,
        ode_noise_curve: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run the diffusion loop.

        Args:
            condition_set: One or more conditions for generation.
            config: Diffusion loop configuration.
            attention_mask: Optional [B, T] mask. Defaults to all-ones.
            latent_mask: Optional noise mask for inpainting / blending.
                When present, two-sided blending is applied at each step
                (pre-blend on model input, post-blend on x0 prediction).
            source_latents: Source audio latents [B, T, D] for partial
                denoising (denoise < 1.0). When denoise < 1.0, the
                initial state is noise_scaling(t_start, noise, source)
                instead of pure noise. If None and denoise < 1.0, falls
                back to latent_mask.original_latents.
            apply_hooks_fn: Optional callback for per-condition model
                weight switching (LoRA, hooks). Called with hook_ref
                (from PreparedCondition) before each hook group's
                decoder call. Receives None to reset to base weights.
            sde_denoise_curve: Optional per-frame denoise modulation
                for SDE mode. Shape [T], [B, T], or [B, T, 1], values
                in [0, 1]. High values get full re-noise (more
                transformation), low values pull toward source_latents
                (preserve source). Requires infer_method="sde" and
                source_latents to be provided.
            velocity_scale: Optional per-frame velocity scaling. Shape
                [T], [B, T], or [B, T, 1]. Multiplies the decoder's
                velocity output before integration. Values > 1 increase
                transformation rate, < 1 decrease it.
            initial_noise_curve: Optional per-frame initial state mixing.
                Shape [T], [B, T], or [B, T, 1], values in [0, 1].
                Controls the noise/source mix at each frame in the
                initial state: 1.0 = pure noise, 0.0 = pure source.
                Requires source_latents. Uses full sigma schedule
                (denoise is ignored when this is set).
            x0_target: Optional target latent [B, T, D] for per-frame
                x0 blending. At each step, the x0 prediction is blended
                toward this target before integration.
            x0_target_curve: Per-frame blend strength toward x0_target.
                Shape [T], [B, T], or [B, T, 1], values in [0, 1].
                Required when x0_target is set. Blending is gated to
                the second half of diffusion steps (refinement phase)
                to avoid corrupting structural decisions.
            negative_condition_set: Optional negative (unconditional)
                conditions for classifier-free guidance. Used with
                guidance_curve for per-frame CFG.
            guidance_curve: Per-frame guidance scale. Shape [T], [B, T],
                or [B, T, 1]. Applied as:
                v = v_uncond + guidance * (v_cond - v_uncond).
                Requires negative_condition_set.
            ode_noise_curve: Per-frame noise injection after each ODE
                step (not final step). Shape [T], [B, T], or [B, T, 1].
                Injection is scaled by current sigma so it naturally
                decreases toward clean. Creates controlled creativity
                at specified frames.

        Returns:
            Dict with ``target_latents`` [B, T, D] and ``time_costs``.
        """
        time_costs = {}
        total_start_time = time.time()

        device = condition_set.device
        dtype = condition_set.dtype
        bsz = condition_set.batch_size

        # Reference condition (for shapes / defaults)
        ref_cond = condition_set.conditions[0]

        if attention_mask is None:
            attention_mask = torch.ones(
                bsz, ref_cond.seq_len, device=device, dtype=dtype
            )

        # Noise
        if config.noise_on_cpu:
            noise = self._prepare_noise_cpu(ref_cond, config.seed)
        else:
            noise = self.model.prepare_noise(ref_cond.context_latents, config.seed)

        # Timestep schedule (with denoise truncation).
        # Move to CPU so .item() in the loop doesn't trigger
        # cudaStreamSynchronize on the default stream, which causes
        # TRT performance degradation on Blackwell GPUs.
        t_schedule = self._build_timestep_schedule(config, device, dtype).cpu()
        infer_steps = len(t_schedule) - 1

        if infer_steps <= 0 or config.denoise <= 0.0:
            # No denoising: return source latents directly
            out = source_latents if source_latents is not None else noise
            return {"target_latents": out, "time_costs": {}}

        # ---- TRT fast path: tight loop, no torch stream interaction ----
        if (
            self._trt_engine is not None
            and condition_set.is_single_condition
            and latent_mask is None
            and x0_target is None
            and apply_hooks_fn is None
            and negative_condition_set is None
        ):
            diffusion_start = time.time()
            cond = condition_set.conditions[0]
            t_start = t_schedule[0].item()

            if initial_noise_curve is not None and source_latents is not None:
                curve = self._normalize_curve(initial_noise_curve)
                xt = curve * noise + (1.0 - curve) * source_latents
            elif config.denoise < 1.0 and source_latents is not None:
                xt = t_start * noise + (1.0 - t_start) * source_latents
            else:
                xt = noise.clone()

            trt_ctx = self._trt_ctx
            stream = self._trt_stream

            # Pre-allocate buffers and bind once (like StreamDiffusion).
            # Uses engine's native I/O dtype (fp16 for mixed-precision, fp32 for legacy).
            io_dtype = self._trt_io_dtype
            T = xt.shape[1]
            eff_T = T + 1 if T % 2 == 1 else T
            enc_hs = cond.encoder_hidden_states.to(io_dtype).contiguous()
            ctx_lat = cond.context_latents.to(io_dtype).contiguous()
            L = enc_hs.shape[1]

            bufs = {
                "hidden_states": torch.empty(bsz, eff_T, 64, dtype=io_dtype, device=device),
                "timestep": torch.empty(bsz, dtype=torch.float32, device=device),
                "encoder_hidden_states": torch.empty(bsz, L, 2048, dtype=io_dtype, device=device),
                "context_latents": torch.empty(bsz, eff_T, 128, dtype=io_dtype, device=device),
            }
            for name, buf in bufs.items():
                ok = trt_ctx.set_input_shape(name, tuple(buf.shape))
                trt_ctx.set_tensor_address(name, buf.data_ptr())
                if not ok:
                    logger.error("set_input_shape(%s, %s) failed", name, tuple(buf.shape))
            out_shape = tuple(trt_ctx.get_tensor_shape("velocity"))
            if any(d < 0 for d in out_shape):
                logger.error("TRT output shape unresolved: %s (T=%d, eff_T=%d, L=%d, bsz=%d)", out_shape, T, eff_T, L, bsz)
                raise RuntimeError(f"TRT output shape unresolved: {out_shape}. Inputs: T={T}, eff_T={eff_T}, L={L}, bsz={bsz}")
            out_buf = torch.empty(out_shape, dtype=io_dtype, device=device)
            trt_ctx.set_tensor_address("velocity", out_buf.data_ptr())

            # Pre-copy constant condition tensors
            bufs["encoder_hidden_states"].copy_(enc_hs)
            if T % 2 == 1:
                bufs["context_latents"][:, :T, :].copy_(ctx_lat)
                bufs["context_latents"][:, T:, :].zero_()
            else:
                bufs["context_latents"].copy_(ctx_lat)

            if config.infer_method == "ode":
                for i in range(infer_steps):
                    t_curr = t_schedule[i].item()
                    t_next = t_schedule[i + 1].item()
                    dt = t_next - t_curr

                    if T % 2 == 1:
                        bufs["hidden_states"][:, :T, :].copy_(xt)
                        bufs["hidden_states"][:, T:, :].zero_()
                    else:
                        bufs["hidden_states"].copy_(xt)
                    bufs["timestep"].fill_(t_curr)

                    for name, buf in bufs.items():
                        trt_ctx.set_tensor_address(name, buf.data_ptr())
                    trt_ctx.set_tensor_address("velocity", out_buf.data_ptr())

                    trt_ctx.execute_async_v3(stream.ptr)
                    stream.synchronize()

                    vt = out_buf[:, :T, :].to(dtype)
                    if velocity_scale is not None:
                        vt = vt * self._normalize_curve(velocity_scale)
                    xt = xt + dt * vt
                    if (
                        ode_noise_curve is not None
                        and i < infer_steps - 1
                        and t_next > 0
                    ):
                        xt = xt + torch.randn_like(xt) * self._normalize_curve(
                            ode_noise_curve
                        ) * t_next
            else:
                # SDE path
                if sde_denoise_curve is not None:
                    sdc = self._normalize_curve(sde_denoise_curve).to(
                        device=device, dtype=dtype
                    )
                else:
                    sdc = torch.ones(1, 1, 1, device=device, dtype=dtype)
                src = source_latents if source_latents is not None else torch.zeros_like(xt)

                for i in range(infer_steps):
                    t_curr = t_schedule[i].item()
                    t_next = t_schedule[i + 1].item()

                    if T % 2 == 1:
                        bufs["hidden_states"][:, :T, :].copy_(xt)
                        bufs["hidden_states"][:, T:, :].zero_()
                    else:
                        bufs["hidden_states"].copy_(xt)
                    bufs["timestep"].fill_(t_curr)

                    for name, buf in bufs.items():
                        trt_ctx.set_tensor_address(name, buf.data_ptr())
                    trt_ctx.set_tensor_address("velocity", out_buf.data_ptr())

                    trt_ctx.execute_async_v3(stream.ptr)
                    stream.synchronize()

                    vt = out_buf[:, :T, :].to(dtype)
                    if velocity_scale is not None:
                        vt = vt * self._normalize_curve(velocity_scale)
                    x0_pred = xt - vt * t_curr
                    sde_noise = torch.randn_like(xt)
                    xt_full = t_next * sde_noise + (1.0 - t_next) * x0_pred
                    xt_source = t_next * sde_noise + (1.0 - t_next) * src
                    xt = sdc * xt_full + (1.0 - sdc) * xt_source

            diffusion_end = time.time()
            time_costs["diffusion_time_cost"] = diffusion_end - diffusion_start
            time_costs["diffusion_per_step_time_cost"] = (
                time_costs["diffusion_time_cost"] / max(infer_steps, 1)
            )
            time_costs["total_time_cost"] = diffusion_end - total_start_time
            time_costs["trt_fast_path"] = True
            return {"target_latents": xt, "time_costs": time_costs}

        # ---- Precomputed fast path ----
        if self._can_use_fast_path(
            condition_set, config, latent_mask,
            velocity_scale, sde_denoise_curve, x0_target, apply_hooks_fn,
            negative_condition_set, ode_noise_curve,
        ):
            diffusion_start = time.time()
            xt = self._generate_fast(
                condition_set, config, noise, t_schedule,
                source_latents, attention_mask,
                velocity_scale=velocity_scale,
                ode_noise_curve=ode_noise_curve,
                sde_denoise_curve=sde_denoise_curve,
                initial_noise_curve=initial_noise_curve,
            )
            diffusion_end = time.time()
            time_costs["diffusion_time_cost"] = diffusion_end - diffusion_start
            time_costs["diffusion_per_step_time_cost"] = (
                time_costs["diffusion_time_cost"] / max(infer_steps, 1)
            )
            time_costs["total_time_cost"] = diffusion_end - total_start_time
            time_costs["fast_path"] = True
            return {"target_latents": xt, "time_costs": time_costs}

        # ---- Standard path ----

        # Initial state
        t_start = t_schedule[0].item()

        if initial_noise_curve is not None:
            # Per-frame noise/source mixing (Path B)
            if source_latents is None:
                raise ValueError(
                    "initial_noise_curve requires source_latents"
                )
            curve = self._normalize_curve(initial_noise_curve)
            xt = curve * noise + (1.0 - curve) * source_latents
        elif config.denoise < 1.0:
            # Partial denoise: start from noised source, not pure noise
            src = source_latents
            if src is None and latent_mask is not None:
                src = latent_mask.original_latents
            if src is None:
                raise ValueError(
                    "denoise < 1.0 requires source_latents or "
                    "latent_mask.original_latents"
                )
            # Flow matching noise scaling: xt = t * noise + (1-t) * x0
            xt = t_start * noise + (1.0 - t_start) * src
        else:
            xt = noise

        # KV cache state
        past_key_values = (
            EncoderDecoderCache(DynamicCache(), DynamicCache())
            if config.use_cache else None
        )
        prev_active_ids: Optional[frozenset] = None

        diffusion_start = time.time()

        for step_idx in range(infer_steps):
            t_curr = t_schedule[step_idx].item()
            t_next = t_schedule[step_idx + 1].item()
            t_curr_tensor = t_curr * torch.ones(
                (bsz,), device=device, dtype=dtype
            )

            # Active conditions at this step
            active = condition_set.active_conditions_at_step(
                step_idx, infer_steps
            )
            if not active:
                active = [ref_cond]

            # Detect condition switch -> reset KV cache
            active_ids = frozenset(id(c) for c in active)
            if prev_active_ids is not None and active_ids != prev_active_ids:
                past_key_values = (
                    EncoderDecoderCache(DynamicCache(), DynamicCache())
                    if config.use_cache else None
                )
            prev_active_ids = active_ids

            # --- Pre-blend: give model correct context in preserved regions ---
            xt_input = xt
            if latent_mask is not None:
                xt_input = self._mask_pre_blend(
                    xt, t_curr, latent_mask, step_idx, infer_steps
                )

            # --- Decoder call(s) ---
            if len(active) == 1 and active[0].temporal_weight is None:
                # Apply hooks for single condition if needed
                if apply_hooks_fn is not None:
                    apply_hooks_fn(active[0].hook_ref)
                vt, past_key_values = self._single_condition_step(
                    xt_input, t_curr_tensor, active[0], attention_mask,
                    config.use_cache, past_key_values,
                )
            else:
                vt = self._multi_condition_step(
                    xt_input, t_curr_tensor, active, attention_mask,
                    apply_hooks_fn=apply_hooks_fn,
                )
                past_key_values = (
                    EncoderDecoderCache(DynamicCache(), DynamicCache())
                    if config.use_cache else None
                )

            # --- Per-frame classifier-free guidance ---
            if negative_condition_set is not None and guidance_curve is not None:
                neg_active = negative_condition_set.active_conditions_at_step(
                    step_idx, infer_steps
                )
                if not neg_active:
                    neg_active = [negative_condition_set.conditions[0]]
                if len(neg_active) == 1 and neg_active[0].temporal_weight is None:
                    vt_uncond, _ = self._single_condition_step(
                        xt_input, t_curr_tensor, neg_active[0], attention_mask,
                        use_cache=False, past_key_values=None,
                    )
                else:
                    vt_uncond = self._multi_condition_step(
                        xt_input, t_curr_tensor, neg_active, attention_mask,
                    )
                gc = self._normalize_curve(guidance_curve)
                vt = vt_uncond + gc * (vt - vt_uncond)

            # --- Per-frame velocity scaling ---
            if velocity_scale is not None:
                vt = vt * self._normalize_curve(velocity_scale)

            # --- Per-frame x0 target blending (gated to refinement) ---
            x0_target_effective = None
            x0_target_curve_effective = None
            if x0_target is not None and x0_target_curve is not None:
                step_progress = step_idx / max(infer_steps - 1, 1)
                gate_start = config.x0_target_gate
                blend_gate = max(0.0, step_progress - gate_start) / max(1.0 - gate_start, 1e-6)
                if blend_gate > 0:
                    x0_target_effective = x0_target
                    x0_target_curve_effective = (
                        self._normalize_curve(x0_target_curve) * blend_gate
                    )

            # --- Integrate with mask-aware x0 blending ---
            xt = self._integrate_step(
                xt_input, vt, t_curr, t_next,
                bsz, device, dtype,
                config.infer_method,
                latent_mask, step_idx, infer_steps,
                sde_denoise_curve=sde_denoise_curve,
                source_latents=source_latents,
                x0_target=x0_target_effective,
                x0_target_curve=x0_target_curve_effective,
            )

            # --- Per-frame ODE noise injection (skip final step) ---
            if (
                ode_noise_curve is not None
                and step_idx < infer_steps - 1
                and t_next > 0
            ):
                injection_noise = torch.randn_like(xt)
                ode_curve = self._normalize_curve(ode_noise_curve)
                xt = xt + injection_noise * ode_curve * t_next

        diffusion_end = time.time()
        time_costs["diffusion_time_cost"] = diffusion_end - diffusion_start
        time_costs["diffusion_per_step_time_cost"] = (
            time_costs["diffusion_time_cost"] / max(infer_steps, 1)
        )
        time_costs["total_time_cost"] = diffusion_end - total_start_time

        return {
            "target_latents": xt,
            "time_costs": time_costs,
        }
