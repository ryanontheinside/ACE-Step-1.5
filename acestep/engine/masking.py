from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import torch


@dataclass
class LatentNoiseMask:
    """Latent-space inpainting mask for the diffusion loop.

    Controls per-frame blending between generated and preserved content.
    The mask value is continuous: 1.0 = fully generate, 0.0 = fully
    preserve original, 0.5 = 50/50 blend at every step.

    The engine applies this mask on BOTH sides of the decoder call,
    matching ComfyUI's KSamplerX0Inpaint behavior:

      Pre-blend (model input):
        xt_input = mask * xt + (1-mask) * noise_scaling(t, noise, original)

      Post-blend (denoised x0 prediction):
        x0_blended = mask * x0_pred + (1-mask) * original

    This gives the model correct context in preserved regions (properly
    noised original at the current sigma) and ensures preserved regions
    converge to the clean original.

    Attributes:
        mask: Per-frame blend mask. 1.0 = generate, 0.0 = preserve.
              Shape: [B, T, 1], [B, T], or [T].
        original_latents: Clean (x0) latents for preserved regions.
              Shape: [B, T, D].
        step_strength_fn: Optional callable (step_idx, total_steps) -> float
              that modulates the mask per step. Enables progressive masking.
        noise: Fixed noise tensor for deterministic re-noising across steps.
              Shape: [B, T, D]. If None, generated once on first use and
              stored for consistency.
    """

    mask: torch.Tensor
    original_latents: torch.Tensor
    step_strength_fn: Optional[Callable[[int, int], float]] = None
    noise: Optional[torch.Tensor] = None

    def get_mask(
        self,
        step_idx: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the mask tensor, normalized to [B, T, 1] and optionally
        modulated by step_strength_fn."""
        mask = self.mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).unsqueeze(-1)  # [T] -> [1, T, 1]
        elif mask.ndim == 2:
            mask = mask.unsqueeze(-1)  # [B, T] -> [B, T, 1]

        if self.step_strength_fn is not None and step_idx is not None:
            strength = self.step_strength_fn(step_idx, total_steps)
            mask = mask * strength

        return mask

    def ensure_noise(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the stored noise tensor, creating it on first call.
        This ensures the same noise is used across all diffusion steps."""
        if self.noise is None:
            self.noise = torch.randn_like(self.original_latents)
        return self.noise.to(device=device, dtype=dtype)
