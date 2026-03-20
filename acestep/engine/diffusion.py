from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Callable, Optional, Tuple, List, Union

import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from .conditions import PreparedCondition, ConditionSet
from .masking import LatentNoiseMask


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

    def __init__(self, model, trt_decoder=None):
        """
        Args:
            model: AceStepConditionGenerationModel instance.
            trt_decoder: Optional TRTDecoder instance.  When provided, all
                decoder calls are routed through TensorRT instead of the
                PyTorch model.  All engine modulations (temporal blending,
                velocity scaling, noise masks, etc.) continue to work
                because they operate on the velocity output, not inside
                the decoder.
        """
        self.model = model
        self.decoder = model.decoder
        self.trt_decoder = trt_decoder
        self._compiled_loop: Optional[Callable] = None

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
    ) -> bool:
        """Check if the precomputed fast path is viable."""
        if config.infer_method != "ode":
            return False
        if latent_mask is not None:
            return False
        if velocity_scale is not None or sde_denoise_curve is not None:
            return False
        if x0_target is not None:
            return False
        if apply_hooks_fn is not None:
            return False
        if not condition_set.is_single_condition:
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
    ) -> torch.Tensor:
        """Precomputed fast path: single condition, ODE, no modulation.

        Delegates to a compiled inner loop that the torch compiler can
        see as a single graph (no Python overhead between steps).
        """
        device = noise.device
        dtype = noise.dtype
        bsz = noise.shape[0]
        infer_steps = len(t_schedule) - 1
        cond = condition_set.conditions[0]

        # Initial state
        t_start = t_schedule[0].item()
        if config.denoise < 1.0 and source_latents is not None:
            xt = t_start * noise + (1.0 - t_start) * source_latents
        else:
            xt = noise.clone()

        # Precompute schedule as stacked tensors (no per-step Python)
        t_curr_vec = t_schedule[:-1].unsqueeze(-1).expand(infer_steps, bsz)  # [steps, B]
        dt_vec = (t_schedule[:-1] - t_schedule[1:])  # [steps]
        # Final step: dt = t_curr (returns x0)
        dt_vec = dt_vec.clone()
        dt_vec[-1] = t_schedule[-2]
        # [steps] -> [steps, 1, 1] for broadcasting with [B, T, D]
        dt_vec = dt_vec.reshape(infer_steps, 1, 1)  # [steps, 1, 1]

        # Run compiled inner loop
        if self._compiled_loop is None:
            self._compiled_loop = torch.compile(
                self._fast_loop,
                backend="inductor",
                dynamic=False,
                mode="max-autotune-no-cudagraphs",
            )

        return self._compiled_loop(
            self.decoder, xt, t_curr_vec, dt_vec,
            attention_mask,
            cond.encoder_hidden_states,
            cond.encoder_attention_mask,
            cond.context_latents,
            infer_steps,
        )

    @staticmethod
    def _fast_loop(
        decoder,
        xt: torch.Tensor,
        t_curr_vec: torch.Tensor,
        dt_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        enc_hs: torch.Tensor,
        enc_mask: torch.Tensor,
        ctx_lat: torch.Tensor,
        infer_steps: int,
    ) -> torch.Tensor:
        """Inner loop as a standalone function for torch.compile.

        By compiling this separately, the compiler can see the full loop
        and optimize across step boundaries.
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
            xt = xt - vt * dt_vec[i]
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

        if denoise >= 1.0 or denoise <= 0.0:
            # Full schedule or no-op
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

        When trt_decoder is set, routes through TensorRT (no KV cache).
        All engine modulations continue to work unchanged because they
        operate on the velocity tensor returned here.
        """
        if self.trt_decoder is not None:
            vt = self.trt_decoder(
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

        # Timestep schedule (with denoise truncation)
        t_schedule = self._build_timestep_schedule(config, device, dtype)
        infer_steps = len(t_schedule) - 1

        if infer_steps <= 0 or config.denoise <= 0.0:
            # No denoising: return source latents directly
            out = source_latents if source_latents is not None else noise
            return {"target_latents": out, "time_costs": {}}

        # ---- Precomputed fast path ----
        if self._can_use_fast_path(
            condition_set, config, latent_mask,
            velocity_scale, sde_denoise_curve, x0_target, apply_hooks_fn,
        ) and initial_noise_curve is None:
            diffusion_start = time.time()
            xt = self._generate_fast(
                condition_set, config, noise, t_schedule,
                source_latents, attention_mask,
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

            # --- Per-frame velocity scaling ---
            if velocity_scale is not None:
                vt = vt * self._normalize_curve(velocity_scale)

            # --- Per-frame x0 target blending (gated to refinement) ---
            x0_target_effective = None
            x0_target_curve_effective = None
            if x0_target is not None and x0_target_curve is not None:
                step_progress = step_idx / max(infer_steps - 1, 1)
                blend_gate = max(0.0, step_progress - 0.5) * 2.0
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
