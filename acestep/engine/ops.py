from __future__ import annotations

from typing import List, Optional, Union

import torch

from .conditions import PreparedCondition


def average_conditions(
    conditions: List[PreparedCondition],
    weights: Optional[List[float]] = None,
) -> PreparedCondition:
    """Weighted average of PreparedCondition tensors.

    Blends encoder_hidden_states and context_latents using the given weights.
    The encoder_attention_mask is taken from the first condition (all
    conditions must share the same mask geometry).

    This is a pre-loop operation: the result is a single PreparedCondition
    that can be used on the fast path (one decoder call per step).

    Args:
        conditions: Two or more PreparedConditions with identical tensor shapes.
        weights: Optional blend weights. If None, uniform averaging is used.
                 Weights are normalized to sum to 1.

    Returns:
        A new PreparedCondition with blended tensors.
    """
    n = len(conditions)
    if n == 0:
        raise ValueError("Need at least one condition to average")
    if n == 1:
        return conditions[0]

    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    enc_hs = sum(
        w * c.encoder_hidden_states for w, c in zip(weights, conditions)
    )
    ctx = sum(
        w * c.context_latents for w, c in zip(weights, conditions)
    )

    return PreparedCondition(
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=conditions[0].encoder_attention_mask,
        context_latents=ctx,
    )


def blend_semantic_hints(
    hints_a: torch.Tensor,
    hints_b: torch.Tensor,
    alpha: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Interpolate between two semantic hint tensors.

    This is a pre-loop operation: blend hint tensors before passing them
    to prepare_condition() as precomputed_lm_hints_25Hz.

    Supports both static (scalar) and per-frame (tensor) blending.

    Args:
        hints_a: First hint tensor [B, T, D].
        hints_b: Second hint tensor [B, T, D].
        alpha: Blend factor in [0, 1]. 0 = all hints_a, 1 = all hints_b.
            Scalar for uniform blending, or a tensor [T] or [B, T] for
            per-frame temporal blending. A [T] tensor is broadcast to
            [1, T, 1] to match the hint shape.

    Returns:
        Interpolated tensor of the same shape as inputs.
    """
    if isinstance(alpha, torch.Tensor):
        if alpha.ndim == 1:       # [T]
            alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        elif alpha.ndim == 2:     # [B, T]
            alpha = alpha.unsqueeze(-1)               # [B, T, 1]
    return (1.0 - alpha) * hints_a + alpha * hints_b


@torch.no_grad()
def extract_semantic_hints(
    model,
    source_latents: torch.Tensor,
) -> torch.Tensor:
    """Extract semantic hints from source audio latents.

    Pre-computes the tokenizer/detokenizer representation of the source
    audio. When passed as precomputed_lm_hints_25Hz to prepare_condition,
    this gives the model stable structural guidance throughout all
    diffusion steps instead of recomputing from noisy intermediates.

    This matches ComfyUI's ACEStep15SemanticExtractor node behavior.

    The alternative (stock ACE-Step behavior when precomputed hints are
    not provided) is to tokenize the noisy latent at each step, which
    acts like noise augmentation but gives weaker structural guidance
    for cover/extract tasks.

    Args:
        model: AceStepConditionGenerationModel instance.
        source_latents: Source audio latents [B, T, D].

    Returns:
        Semantic hints tensor [B, T, D], same shape as source_latents.
    """
    quantized, _indices = model.tokenizer.tokenize(source_latents)
    lm_hints = model.detokenizer(quantized)
    return lm_hints[:, :source_latents.shape[1], :]
