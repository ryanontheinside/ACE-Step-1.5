"""Semantic hint extraction and blending nodes."""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Curve, Latent, ModelHandle, SemanticHints


@NodeRegistry.register
class SemanticExtract(BaseNode):
    """Extract semantic hints from source audio latents.

    Pre-computes the tokenizer/detokenizer representation that provides
    stable structural guidance to the decoder. This is an alternative
    to letting the model recompute hints from noisy latents at each
    diffusion step.
    """

    node_type_id: ClassVar[str] = "acestep.SemanticExtract"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Semantic Extract",
            category="semantic",
            description="Extract semantic structural hints from source audio latents.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="latent", type="LATENT"),
            ),
            outputs=(
                NodePort(name="semantic_hints", type="SEMANTIC_HINTS"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        from acestep.engine.ops import extract_semantic_hints

        model_handle: ModelHandle = kwargs["model"]
        latent: Latent = kwargs["latent"]
        handler = model_handle.handler

        with handler._load_model_context("model"):
            hints = extract_semantic_hints(handler.model, latent.tensor)

        return {"semantic_hints": SemanticHints(tensor=hints)}


@NodeRegistry.register
class SemanticBlend(BaseNode):
    """Blend two semantic hint tensors.

    Interpolates between two sources of structural guidance. The blend
    factor can be a scalar (uniform) or a CURVE (per-frame).

    Node parameters:
        alpha: Scalar blend factor (0.0 = all A, 1.0 = all B).
               Ignored if a curve input is connected.
    """

    node_type_id: ClassVar[str] = "acestep.SemanticBlend"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Semantic Blend",
            category="semantic",
            description="Blend two semantic hint tensors per-frame.",
            inputs=(
                NodePort(name="hints_a", type="SEMANTIC_HINTS"),
                NodePort(name="hints_b", type="SEMANTIC_HINTS"),
                NodePort(
                    name="blend_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame blend factor (overrides scalar alpha).",
                ),
            ),
            outputs=(
                NodePort(name="semantic_hints", type="SEMANTIC_HINTS"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        from acestep.engine.ops import blend_semantic_hints

        hints_a: SemanticHints = kwargs["hints_a"]
        hints_b: SemanticHints = kwargs["hints_b"]
        blend_curve: Optional[Curve] = kwargs.get("blend_curve")

        alpha = kwargs.get("alpha", 0.5)
        if blend_curve is not None:
            alpha = blend_curve.tensor.to(
                device=hints_a.tensor.device, dtype=hints_a.tensor.dtype
            )

        blended = blend_semantic_hints(hints_a.tensor, hints_b.tensor, alpha)
        return {"semantic_hints": SemanticHints(tensor=blended)}
