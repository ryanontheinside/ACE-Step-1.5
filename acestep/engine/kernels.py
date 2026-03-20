"""Fused Triton kernels for the diffusion loop.

These eliminate multiple small CUDA kernel launches in the integration
step by fusing x0 prediction + euler update into a single kernel.

Flow matching math:
  x0_pred = xt - vt * t_curr
  v_blended = (xt - x0_pred) / t_curr  = vt  (identity for vanilla euler)
  xt_next = xt - vt * dt

For the final step (t_next == 0):
  xt_next = x0_pred = xt - vt * t_curr

With masking (post-blend x0):
  x0_pred = mask * (xt - vt * t_curr) + (1 - mask) * original
  v_blended = (xt - x0_pred) / t_curr
  xt_next = xt - v_blended * dt
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _euler_step_kernel(
    xt_ptr,        # [B, T, D] current state (READ + WRITE in-place)
    vt_ptr,        # [B, T, D] velocity from decoder (READ)
    t_curr_ptr,    # scalar: current timestep
    dt_ptr,        # scalar: t_curr - t_next
    is_final_ptr,  # scalar: 1 if final step (return x0), 0 otherwise
    numel,         # total elements in xt
    BLOCK: tl.constexpr,
):
    """Fused euler step: xt = xt - vt * dt, or x0 = xt - vt * t on final step."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    xt = tl.load(xt_ptr + offsets, mask=mask)
    vt = tl.load(vt_ptr + offsets, mask=mask)
    t_curr = tl.load(t_curr_ptr)
    dt = tl.load(dt_ptr)
    is_final = tl.load(is_final_ptr)

    # Final step: return x0 = xt - vt * t_curr
    # Non-final: return xt - vt * dt
    scale = tl.where(is_final > 0, t_curr, dt)
    result = xt - vt * scale

    tl.store(xt_ptr + offsets, result, mask=mask)


@triton.jit
def _euler_step_masked_kernel(
    xt_ptr,        # [B, T, D] current state (READ + WRITE in-place)
    vt_ptr,        # [B, T, D] velocity from decoder (READ)
    orig_ptr,      # [B, T, D] original latents for mask blending (READ)
    mask_ptr,      # [B, T, 1] mask values (READ), broadcast over D
    t_curr_ptr,    # scalar
    dt_ptr,        # scalar
    is_final_ptr,  # scalar
    T,             # time dimension
    D,             # latent dimension
    numel,         # total elements
    BLOCK: tl.constexpr,
):
    """Fused euler step with post-blend x0 masking.

    x0_pred = xt - vt * t_curr
    x0_blended = mask * x0_pred + (1 - mask) * original
    if final: return x0_blended
    else: v_blended = (xt - x0_blended) / t_curr; return xt - v_blended * dt
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask_valid = offsets < numel

    xt = tl.load(xt_ptr + offsets, mask=mask_valid)
    vt = tl.load(vt_ptr + offsets, mask=mask_valid)
    orig = tl.load(orig_ptr + offsets, mask=mask_valid)

    # Compute mask index: mask is [B, T, 1], so index = (offset // D) for the T dim
    # offset = b * T * D + t * D + d => mask_idx = b * T + t = offset // D
    mask_idx = offsets // D
    m = tl.load(mask_ptr + mask_idx, mask=mask_valid)

    t_curr = tl.load(t_curr_ptr)
    dt = tl.load(dt_ptr)
    is_final = tl.load(is_final_ptr)

    # x0 prediction with mask blending
    x0_pred = xt - vt * t_curr
    x0_blended = m * x0_pred + (1.0 - m) * orig

    # Final step: return x0_blended
    # Non-final: euler from blended x0
    #   v_blended = (xt - x0_blended) / t_curr
    #   result = xt - v_blended * dt
    #          = xt - (xt - x0_blended) * dt / t_curr
    #          = xt * (1 - dt/t_curr) + x0_blended * (dt/t_curr)
    ratio = dt / t_curr
    result_nonfinal = xt * (1.0 - ratio) + x0_blended * ratio
    result = tl.where(is_final > 0, x0_blended, result_nonfinal)

    tl.store(xt_ptr + offsets, result, mask=mask_valid)


def fused_euler_step(
    xt: torch.Tensor,
    vt: torch.Tensor,
    t_curr: float,
    t_next: float,
    original: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> None:
    """In-place fused euler step.

    Modifies xt in-place. For the simple (no mask) case, this is a single
    kernel launch replacing 3-4 separate PyTorch ops. For the masked case,
    it fuses x0 prediction + mask blending + euler integration.

    Args:
        xt: [B, T, D] current noisy state (modified in-place).
        vt: [B, T, D] velocity output from decoder.
        t_curr: Current timestep value.
        t_next: Next timestep value (0 on final step).
        original: [B, T, D] original latents for mask blending (optional).
        mask: [B, T, 1] mask values (optional, requires original).
    """
    numel = xt.numel()
    dt = t_curr - t_next
    is_final = 1.0 if t_next <= 0 else 0.0

    # Scalar tensors on device for kernel access
    device = xt.device
    t_curr_t = torch.tensor(t_curr, dtype=xt.dtype, device=device)
    dt_t = torch.tensor(dt, dtype=xt.dtype, device=device)
    is_final_t = torch.tensor(is_final, dtype=xt.dtype, device=device)

    BLOCK = 1024
    grid = ((numel + BLOCK - 1) // BLOCK,)

    if mask is not None and original is not None:
        B, T, D = xt.shape
        # Ensure mask is [B, T, 1] contiguous
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).unsqueeze(-1).expand(B, T, 1).contiguous()
        elif mask.ndim == 2:
            mask = mask.unsqueeze(-1).contiguous()
        mask = mask.contiguous()

        _euler_step_masked_kernel[grid](
            xt, vt, original, mask,
            t_curr_t, dt_t, is_final_t,
            T, D, numel,
            BLOCK=BLOCK,
        )
    else:
        _euler_step_kernel[grid](
            xt, vt,
            t_curr_t, dt_t, is_final_t,
            numel,
            BLOCK=BLOCK,
        )
