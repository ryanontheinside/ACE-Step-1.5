"""Wire types for the ACE-Step node system.

Each type is a dataclass representing data that flows between nodes.
Types carry a TYPE_NAME class variable used for port validation:
connecting an output to an input requires matching TYPE_NAMEs.

Type categories:
  - Handle types (MODEL, VAE, CLIP): opaque references to loaded objects.
    All point to the same AceStepHandler instance but are distinct types
    so the port system prevents mis-wiring.
  - Tensor payload types (AUDIO, LATENT, CONDITIONING, etc.): carry the
    actual data produced/consumed by nodes.
  - Config types (CONFIG): wrap engine configuration dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Tuple, List

import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.masking import LatentNoiseMask

if TYPE_CHECKING:
    from acestep.handler import AceStepHandler


# -----------------------------------------------------------------------
# Type registry
# -----------------------------------------------------------------------

_TYPE_REGISTRY: dict[str, type] = {}


def _register(cls: type) -> type:
    """Register a wire type by its TYPE_NAME."""
    _TYPE_REGISTRY[cls.TYPE_NAME] = cls
    return cls


def get_type_class(type_name: str) -> type | None:
    """Look up a wire type class by name."""
    return _TYPE_REGISTRY.get(type_name)


def all_type_names() -> list[str]:
    """Return all registered type names."""
    return list(_TYPE_REGISTRY.keys())


def types_compatible(source_type: str, target_type: str) -> bool:
    """Check whether a source port type can connect to a target port type."""
    if source_type == target_type:
        return True
    # "ANY" accepts anything (for utility nodes like reroute)
    if target_type == "ANY" or source_type == "ANY":
        return True
    return False


# -----------------------------------------------------------------------
# Handle types (opaque references to loaded objects)
# -----------------------------------------------------------------------

@_register
@dataclass
class ModelHandle:
    """Reference to the loaded ACE-Step model via the handler."""
    TYPE_NAME: ClassVar[str] = "MODEL"
    handler: AceStepHandler


@_register
@dataclass
class VAEHandle:
    """Reference to the VAE via the handler."""
    TYPE_NAME: ClassVar[str] = "VAE"
    handler: AceStepHandler


@_register
@dataclass
class CLIPHandle:
    """Reference to the text encoder/tokenizer via the handler."""
    TYPE_NAME: ClassVar[str] = "CLIP"
    handler: AceStepHandler


# -----------------------------------------------------------------------
# Tensor payload types
# -----------------------------------------------------------------------

@_register
@dataclass
class Audio:
    """Waveform audio data."""
    TYPE_NAME: ClassVar[str] = "AUDIO"
    waveform: torch.Tensor  # [B, channels, samples]
    sample_rate: int = 48000
    start_sample: int = 0  # sample offset into the full signal (for windowed decode)


@_register
@dataclass
class Latent:
    """VAE-encoded audio latent, optionally carrying a noise mask."""
    TYPE_NAME: ClassVar[str] = "LATENT"
    tensor: torch.Tensor  # [B, T, D]
    mask: Optional[LatentNoiseMask] = None


@dataclass
class ConditioningEntry:
    """One condition within a combined set, with compositing metadata.

    Used internally by ConditioningCombine to attach temporal_weight,
    step_range, and hook_ref to individual conditions within a
    Conditioning payload. Not a wire type (no TYPE_NAME).
    """
    encoder_hidden_states: torch.Tensor  # [B, L_enc, D]
    encoder_attention_mask: torch.Tensor  # [B, L_enc]
    temporal_weight: Optional[torch.Tensor] = None  # [T], [B,T], or [B,T,1]
    step_range: Optional[Tuple[float, float]] = None
    hook_ref: Optional[Any] = None


@_register
@dataclass
class Conditioning:
    """Encoded cross-attention conditioning for the diffusion decoder.

    Contains encoder_hidden_states (packed text + lyrics + timbre) and
    the corresponding attention mask. Context latents (src_latents +
    chunk_mask) are built separately by Generate from explicit inputs.

    Can represent a single condition (from TextEncode) or a combined
    set (from ConditioningCombine). When entries is None, the top-level
    tensors represent a single condition. When entries is populated,
    those are the authoritative conditions (top-level tensors are ignored).
    """
    TYPE_NAME: ClassVar[str] = "CONDITIONING"

    # Single condition tensors (populated by TextEncode and similar)
    encoder_hidden_states: Optional[torch.Tensor] = None  # [B, L_enc, D]
    encoder_attention_mask: Optional[torch.Tensor] = None  # [B, L_enc]

    # Combined conditions (populated by ConditioningCombine)
    entries: Optional[List[ConditioningEntry]] = field(default=None, repr=False)

    @property
    def is_combined(self) -> bool:
        return self.entries is not None and len(self.entries) > 0

    def to_entries(self) -> List[ConditioningEntry]:
        """Return all conditions as a list of ConditioningEntry."""
        if self.entries is not None:
            return self.entries
        if self.encoder_hidden_states is None:
            return []
        return [
            ConditioningEntry(
                encoder_hidden_states=self.encoder_hidden_states,
                encoder_attention_mask=self.encoder_attention_mask,
            )
        ]


@_register
@dataclass
class SemanticHints:
    """Structural guidance extracted from source audio.

    Distinct from Latent for type safety: prevents accidentally
    wiring raw latents into a semantic_hints input.
    """
    TYPE_NAME: ClassVar[str] = "SEMANTIC_HINTS"
    tensor: torch.Tensor  # [B, T, D]


@_register
@dataclass
class Mask:
    """Per-frame spatial mask, values in [0, 1].

    Used for latent noise masking (which regions to preserve vs generate)
    and conditioning spatial blending.
    """
    TYPE_NAME: ClassVar[str] = "MASK"
    tensor: torch.Tensor  # [T] or [B, T]


@_register
@dataclass
class Curve:
    """Per-frame modulation signal, arbitrary range.

    Used for velocity scaling, SDE denoise curves, initial noise curves,
    x0 target blend curves, and any other per-frame parameter modulation.
    """
    TYPE_NAME: ClassVar[str] = "CURVE"
    tensor: torch.Tensor  # [T] or [B, T]


@_register
@dataclass
class Config:
    """Diffusion loop configuration."""
    TYPE_NAME: ClassVar[str] = "CONFIG"
    config: DiffusionConfig


@_register
@dataclass
class LoRA:
    """Loaded LoRA adapter weights."""
    TYPE_NAME: ClassVar[str] = "LORA"
    path: str
    scale: float = 1.0
