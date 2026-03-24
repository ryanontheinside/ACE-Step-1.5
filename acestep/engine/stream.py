"""StreamDiffusion-style pipeline for interactive ACE-Step generation.

Maintains a ring buffer of in-flight generations at different denoising
stages. Each tick(), one batched forward pass advances all slots. After
warmup, every tick produces a finished generation.

Supports per-slot denoise and source_latents for cover workflows where
the user adjusts the denoise knob in real time. When a TRT engine is
loaded on the DiffusionEngine, tick() routes through TensorRT
automatically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

import torch

from .diffusion import DiffusionConfig, DiffusionEngine

logger = logging.getLogger(__name__)


@dataclass
class SlotRequest:
    """A generation request to be fed into the pipeline.

    Holds the conditioning tensors and noise seed. All requests in a
    pipeline must share the same sequence length T (duration).
    """
    encoder_hidden_states: torch.Tensor  # [1, L, D]
    encoder_attention_mask: torch.Tensor  # [1, L]
    context_latents: torch.Tensor         # [1, T, D_ctx]
    seed: Optional[int] = None
    source_latents: Optional[torch.Tensor] = None  # [1, T, D] for cover
    denoise: float = 1.0  # per-request denoise strength
    sde_denoise_curve: Optional[torch.Tensor] = None  # [1, T, 1] per-frame denoise


@dataclass
class _Slot:
    """Internal state for one pipeline slot."""
    request: SlotRequest
    xt: torch.Tensor          # [1, T, D] current noisy latent
    t_schedule: torch.Tensor  # per-slot timestep schedule (on CPU)
    step_idx: int = 0         # which denoising step we're on (0-indexed)


class StreamPipeline:
    """StreamDiffusion-style batched denoising pipeline.

    Pipeline depth = number of denoising steps. After warmup (depth
    ticks), every tick() returns a finished latent.

    Each slot carries its own timestep schedule derived from its
    denoise value, so the user can change denoise between submissions
    and each in-flight generation uses the schedule it was born with.

    When the DiffusionEngine has a TRT engine loaded, tick() uses
    TensorRT for the batched forward pass automatically.

    Usage:
        pipe = StreamPipeline(engine, config)
        pipe.submit(request)     # enqueue a request
        result = pipe.tick()     # run one batched forward pass
        if result is not None:
            # result is [1, T, D] finished latent
            ...
    """

    def __init__(
        self,
        engine: DiffusionEngine,
        config: DiffusionConfig,
        noise_sharing: float = 0.0,
    ):
        self.engine = engine
        self.decoder = engine.decoder
        self.model = engine.model
        self.config = config
        self.noise_sharing = noise_sharing  # 0.0=off, 0.3-0.7 typical

        self._depth: int = config.infer_steps

        # Pipeline state
        self._slots: List[Optional[_Slot]] = [None] * self._depth
        self._queue: List[SlotRequest] = []

        # Shared noise: last noise tensor used, for blending into next gen
        self._last_noise: Optional[torch.Tensor] = None

        # Cached device/dtype (set on first submit)
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # Schedule cache: denoise -> cpu tensor
        self._schedule_cache: dict[float, torch.Tensor] = {}

        # TRT state (mirrors DiffusionEngine pattern)
        self._trt_ctx = engine._trt_ctx
        self._trt_stream = engine._trt_stream
        self._trt_engine = engine._trt_engine
        self._trt_io_dtype = getattr(engine, '_trt_io_dtype', torch.float32)
        self._trt_bufs: Optional[dict] = None
        self._trt_out_buf: Optional[torch.Tensor] = None

        # Stats
        self.ticks: int = 0
        self._last_tick_ms: float = 0.0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def active_slots(self) -> int:
        return sum(1 for s in self._slots if s is not None)

    @property
    def is_warmed_up(self) -> bool:
        """True when all slots are occupied (steady state)."""
        return all(s is not None for s in self._slots)

    @property
    def has_trt(self) -> bool:
        return self._trt_engine is not None

    def submit(self, request: SlotRequest) -> None:
        """Enqueue a generation request."""
        self._queue.append(request)

    def _get_schedule(self, denoise: float) -> torch.Tensor:
        """Get (cached) timestep schedule for a given denoise value."""
        if denoise not in self._schedule_cache:
            cfg = DiffusionConfig(
                infer_steps=self.config.infer_steps,
                shift=self.config.shift,
                denoise=denoise,
            )
            self._schedule_cache[denoise] = self.engine._build_timestep_schedule(
                cfg, self._device, self._dtype
            ).cpu()
        return self._schedule_cache[denoise]

    def _ensure_device(self, device: torch.device, dtype: torch.dtype):
        if self._device is None:
            self._device = device
            self._dtype = dtype

    def _make_noise(self, request: SlotRequest) -> torch.Tensor:
        """Generate initial noise for a request."""
        T = request.context_latents.shape[1]
        D = request.context_latents.shape[-1] // 2

        if request.seed is not None:
            torch.manual_seed(int(request.seed))

        if self.config.noise_on_cpu:
            noise_bdt = torch.randn(1, D, T, device="cpu", dtype=torch.float32)
            return noise_bdt.movedim(-1, -2).to(
                device=self._device, dtype=self._dtype
            )
        else:
            return torch.randn(
                1, T, D, device=self._device, dtype=self._dtype
            )

    def _init_slot(self, request: SlotRequest) -> _Slot:
        """Create a new slot from a request, initialized at step 0."""
        self._ensure_device(
            request.encoder_hidden_states.device,
            request.encoder_hidden_states.dtype,
        )

        t_schedule = self._get_schedule(request.denoise)
        noise = self._make_noise(request)

        # Noise sharing: blend with previous generation's noise
        alpha = self.noise_sharing
        if alpha > 0.0 and self._last_noise is not None:
            if self._last_noise.shape == noise.shape:
                noise = alpha * self._last_noise + (1.0 - alpha**2) ** 0.5 * noise
        self._last_noise = noise.clone()

        t_start = t_schedule[0].item()

        if request.source_latents is not None and request.denoise < 1.0:
            xt = t_start * noise + (1.0 - t_start) * request.source_latents
        else:
            xt = noise.clone()

        return _Slot(
            request=request, xt=xt,
            t_schedule=t_schedule, step_idx=0,
        )

    # ------------------------------------------------------------------
    # TRT buffer management
    # ------------------------------------------------------------------

    def _ensure_trt_bufs(self, B: int, T: int, max_L: int):
        """Allocate/resize TRT I/O buffers and bind shapes once.

        Reuses buffers when shapes haven't changed. Uses engine's native
        I/O dtype (fp16 for mixed-precision, fp32 for legacy engines).
        """
        eff_T = T + 1 if T % 2 == 1 else T
        key = (B, eff_T, max_L)

        if self._trt_bufs is not None and self._trt_bufs.get("_key") == key:
            return  # already allocated for this shape

        device = self._device
        io_dtype = self._trt_io_dtype
        bufs = {
            "hidden_states": torch.empty(B, eff_T, 64, dtype=io_dtype, device=device),
            "timestep": torch.empty(B, dtype=torch.float32, device=device),
            "encoder_hidden_states": torch.empty(B, max_L, 2048, dtype=io_dtype, device=device),
            "context_latents": torch.empty(B, eff_T, 128, dtype=io_dtype, device=device),
        }

        ctx = self._trt_ctx
        for name, buf in bufs.items():
            ctx.set_input_shape(name, tuple(buf.shape))
            ctx.set_tensor_address(name, buf.data_ptr())

        out_shape = tuple(ctx.get_tensor_shape("velocity"))
        if any(d < 0 for d in out_shape):
            raise RuntimeError(
                f"TRT output shape unresolved: {out_shape}. "
                f"B={B}, eff_T={eff_T}, L={max_L}"
            )
        out_buf = torch.empty(out_shape, dtype=io_dtype, device=device)
        ctx.set_tensor_address("velocity", out_buf.data_ptr())

        bufs["_key"] = key
        bufs["_eff_T"] = eff_T
        bufs["_T"] = T
        self._trt_bufs = bufs
        self._trt_out_buf = out_buf

        logger.info(
            "Stream TRT bufs allocated: B=%d eff_T=%d L=%d", B, eff_T, max_L
        )

    def _tick_trt(self, slots, indices) -> torch.Tensor:
        """Batched forward pass through TRT. Returns velocity [B, T, D]."""
        B = len(slots)
        T = slots[0].xt.shape[1]

        # Pad encoder_hidden_states to max L
        max_L = max(s.request.encoder_hidden_states.shape[1] for s in slots)

        self._ensure_trt_bufs(B, T, max_L)
        bufs = self._trt_bufs
        eff_T = bufs["_eff_T"]
        pad = T % 2 == 1

        # Fill hidden_states (cast to engine's I/O dtype) -- changes every tick
        xt_batch = torch.cat([s.xt for s in slots], dim=0).to(self._trt_io_dtype)
        if pad:
            bufs["hidden_states"][:, :T, :].copy_(xt_batch)
            bufs["hidden_states"][:, T:, :].zero_()
        else:
            bufs["hidden_states"].copy_(xt_batch)

        # Fill timesteps (per-slot from their schedules) -- changes every tick
        for i, s in enumerate(slots):
            bufs["timestep"][i] = s.t_schedule[s.step_idx].item()

        # Fill encoder_hidden_states (padded to max_L)
        io_dtype = self._trt_io_dtype
        for i, s in enumerate(slots):
            enc = s.request.encoder_hidden_states.to(io_dtype)
            L = enc.shape[1]
            bufs["encoder_hidden_states"][i, :L, :].copy_(enc[0])
            if L < max_L:
                bufs["encoder_hidden_states"][i, L:, :].zero_()

        # Fill context_latents (per-slot copy avoids cat+to temp allocations)
        for i, s in enumerate(slots):
            bufs["context_latents"][i, :T, :].copy_(s.request.context_latents[0, :T])
        if pad:
            bufs["context_latents"][:, T:, :].zero_()

        # Rebind addresses and execute
        ctx = self._trt_ctx
        for name, buf in bufs.items():
            if name.startswith("_"):
                continue
            ctx.set_tensor_address(name, buf.data_ptr())
        ctx.set_tensor_address("velocity", self._trt_out_buf.data_ptr())

        ctx.execute_async_v3(self._trt_stream.ptr)
        self._trt_stream.synchronize()

        # Slice off padding and convert to model dtype
        out = self._trt_out_buf
        if pad:
            return out[:, :T, :].to(self._dtype)
        return out.to(self._dtype)

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tick(self) -> Optional[torch.Tensor]:
        """Run one batched forward pass, advancing all active slots.

        Returns:
            Finished latent [1, T, D] if a slot completed, else None.
        """
        tick_start = time.time()

        # Check for finished slot (slot at final step of its schedule)
        finished = None
        for i, slot in enumerate(self._slots):
            if slot is not None and slot.step_idx >= len(slot.t_schedule) - 1:
                finished = slot.xt
                self._slots[i] = None
                break

        # Fill empty slots from queue
        for i, slot in enumerate(self._slots):
            if slot is None and self._queue:
                req = self._queue.pop(0)
                self._slots[i] = self._init_slot(req)

        # Collect active slots (exclude completed)
        active = [
            (i, s) for i, s in enumerate(self._slots)
            if s is not None and s.step_idx < len(s.t_schedule) - 1
        ]
        if not active:
            self._last_tick_ms = (time.time() - tick_start) * 1000
            self.ticks += 1
            return finished

        indices, slots = zip(*active)
        B = len(slots)

        # Forward pass: TRT or PyTorch
        if self._trt_engine is not None:
            vt_batch = self._tick_trt(slots, indices)
        else:
            # PyTorch path
            timesteps = torch.tensor(
                [s.t_schedule[s.step_idx].item() for s in slots],
                device=self._device, dtype=self._dtype,
            )
            xt_batch = torch.cat([s.xt for s in slots], dim=0)

            max_L = max(s.request.encoder_hidden_states.shape[1] for s in slots)
            enc_list = []
            enc_mask_list = []
            ctx_list = []

            for s in slots:
                enc = s.request.encoder_hidden_states
                mask = s.request.encoder_attention_mask
                L = enc.shape[1]
                if L < max_L:
                    pad = max_L - L
                    enc = torch.nn.functional.pad(enc, (0, 0, 0, pad))
                    mask = torch.nn.functional.pad(mask, (0, pad), value=0)
                enc_list.append(enc)
                enc_mask_list.append(mask)
                ctx_list.append(s.request.context_latents)

            enc_batch = torch.cat(enc_list, dim=0)
            enc_mask_batch = torch.cat(enc_mask_list, dim=0)
            ctx_batch = torch.cat(ctx_list, dim=0)
            attn_mask = torch.ones(B, xt_batch.shape[1],
                                   device=self._device, dtype=self._dtype)

            decoder_out = self.decoder(
                hidden_states=xt_batch,
                timestep=timesteps,
                timestep_r=timesteps,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_batch,
                encoder_attention_mask=enc_mask_batch,
                context_latents=ctx_batch,
                use_cache=False,
                past_key_values=None,
            )
            vt_batch = decoder_out[0]

        # Step: ODE (default) or SDE (when sde_denoise_curve is present)
        for batch_idx, (slot_idx, slot) in enumerate(zip(indices, slots)):
            t_curr = slot.t_schedule[slot.step_idx].item()
            t_next = slot.t_schedule[slot.step_idx + 1].item()

            vt = vt_batch[batch_idx:batch_idx+1]

            if slot.request.sde_denoise_curve is not None and slot.request.source_latents is not None:
                # SDE step: predict x0, re-noise, blend with source via curve
                x0_pred = slot.xt - vt * t_curr
                sde_noise = torch.randn_like(slot.xt)
                xt_full = t_next * sde_noise + (1.0 - t_next) * x0_pred
                xt_source = t_next * sde_noise + (1.0 - t_next) * slot.request.source_latents
                sdc = slot.request.sde_denoise_curve.to(
                    device=slot.xt.device, dtype=slot.xt.dtype
                )
                slot.xt = sdc * xt_full + (1.0 - sdc) * xt_source
            else:
                # ODE Euler step
                dt = t_next - t_curr
                slot.xt = slot.xt + dt * vt

            slot.step_idx += 1

        self._last_tick_ms = (time.time() - tick_start) * 1000
        self.ticks += 1

        return finished

    def flush(self) -> List[torch.Tensor]:
        """Drain the pipeline: keep ticking until all slots complete."""
        results = []
        max_iters = self._depth * 2
        for _ in range(max_iters):
            result = self.tick()
            if result is not None:
                results.append(result)
            if self.active_slots == 0 and not self._queue:
                break
        return results

    def stats(self) -> dict:
        return {
            "ticks": self.ticks,
            "active_slots": self.active_slots,
            "queue_depth": len(self._queue),
            "last_tick_ms": round(self._last_tick_ms, 2),
            "is_warmed_up": self.is_warmed_up,
            "backend": "trt" if self._trt_engine is not None else "pytorch",
        }
