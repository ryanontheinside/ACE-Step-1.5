from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import torch


@dataclass
class PreparedCondition:
    """A single conditioning context ready for the decoder.

    Holds the three tensors produced by model.prepare_condition() plus
    compositing metadata that controls how and when this condition
    participates in the diffusion loop.
    """

    encoder_hidden_states: torch.Tensor  # [B, L_enc, D]
    encoder_attention_mask: torch.Tensor  # [B, L_enc]
    context_latents: torch.Tensor  # [B, T, D_ctx] (src_latents ++ chunk_masks)

    # Per-frame contribution weight for velocity compositing.
    # Shape: [T], [B, T], or [B, T, 1]. None means uniform weight of 1.
    temporal_weight: Optional[torch.Tensor] = None

    # Diffusion step range this condition is active for, as fractions of
    # total steps: (start_frac, end_frac) in [0.0, 1.0).
    # None means active for all steps.
    step_range: Optional[Tuple[float, float]] = None

    # Opaque reference to model patches (LoRA, hooks, etc.) associated
    # with this condition. The engine uses identity comparison to group
    # conditions that share the same model weights for batched execution.
    # None means base model weights (no patches).
    hook_ref: Optional[Any] = None

    def is_active_at_step(self, step_idx: int, total_steps: int) -> bool:
        if self.step_range is None:
            return True
        progress = step_idx / total_steps
        return self.step_range[0] <= progress < self.step_range[1]

    @property
    def batch_size(self) -> int:
        return self.encoder_hidden_states.shape[0]

    @property
    def seq_len(self) -> int:
        return self.context_latents.shape[1]

    @property
    def device(self) -> torch.device:
        return self.encoder_hidden_states.device

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder_hidden_states.dtype


@dataclass
class ConditionSet:
    """Groups multiple PreparedConditions for one generation.

    Carries guidance configuration and provides step-wise filtering.
    """

    conditions: List[PreparedCondition]

    # Guidance config (for base-model CFG / APG / ADG, future use)
    guidance_scale: float = 1.0
    use_adg: bool = False
    cfg_interval: Tuple[float, float] = (0.0, 1.0)

    @property
    def is_single_condition(self) -> bool:
        """True when the fast path can be used (zero overhead)."""
        if len(self.conditions) != 1:
            return False
        c = self.conditions[0]
        return c.temporal_weight is None and c.step_range is None

    def active_conditions_at_step(
        self, step_idx: int, total_steps: int
    ) -> List[PreparedCondition]:
        return [
            c for c in self.conditions
            if c.is_active_at_step(step_idx, total_steps)
        ]

    @property
    def batch_size(self) -> int:
        return self.conditions[0].batch_size

    @property
    def device(self) -> torch.device:
        return self.conditions[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.conditions[0].dtype


class ConditionBuilder:
    """Wraps model.prepare_condition() for convenient condition construction.

    Works with pre-encoded tensors (text_hidden_states, lyric_hidden_states,
    etc.). For building conditions from raw text, use
    AceStepHandler.build_condition() which handles encoding first.
    """

    def __init__(self, model):
        """
        Args:
            model: AceStepConditionGenerationModel instance.
        """
        self.model = model

    @torch.no_grad()
    def build(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
        src_latents: torch.Tensor,
        chunk_masks: torch.Tensor,
        is_covers: torch.Tensor,
        silence_latent: torch.Tensor,
        precomputed_lm_hints_25Hz: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        temporal_weight: Optional[torch.Tensor] = None,
        step_range: Optional[Tuple[float, float]] = None,
    ) -> PreparedCondition:
        """Build a PreparedCondition from pre-encoded tensors.

        Parameters mirror model.prepare_condition(), with the addition of
        temporal_weight and step_range for compositing metadata.
        """
        attention_mask = torch.ones(
            src_latents.shape[0],
            src_latents.shape[1],
            device=src_latents.device,
            dtype=src_latents.dtype,
        )

        encoder_hidden_states, encoder_attention_mask, context_latents = (
            self.model.prepare_condition(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
                hidden_states=src_latents,
                attention_mask=attention_mask,
                silence_latent=silence_latent,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
                audio_codes=audio_codes,
            )
        )

        return PreparedCondition(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            temporal_weight=temporal_weight,
            step_range=step_range,
        )
