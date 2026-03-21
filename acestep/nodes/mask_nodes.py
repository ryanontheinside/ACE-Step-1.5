"""Mask and latent noise mask nodes."""

from __future__ import annotations

import torch
from typing import Any, ClassVar, Optional

from acestep.engine.masking import LatentNoiseMask

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Curve, Latent, Mask


@NodeRegistry.register
class TemporalMask(BaseNode):
    """Create a per-frame temporal mask from a scalar or curve.

    Output mask shape matches the latent's time dimension.
    Values: 1.0 = generate (transform), 0.0 = preserve (keep original).

    Node parameters:
        value: Scalar mask value (used if no curve is connected).
    """

    node_type_id: ClassVar[str] = "acestep.TemporalMask"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Temporal Mask",
            category="mask",
            description="Create a per-frame temporal mask.",
            inputs=(
                NodePort(name="latent", type="LATENT"),
                NodePort(
                    name="curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame mask values (overrides scalar).",
                ),
            ),
            outputs=(
                NodePort(name="mask", type="MASK"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        latent: Latent = kwargs["latent"]
        curve: Optional[Curve] = kwargs.get("curve")
        value = kwargs.get("value", 0.5)

        T = latent.tensor.shape[1]

        if curve is not None:
            mask_tensor = curve.tensor.clamp(0.0, 1.0)
            # Ensure length matches latent
            if mask_tensor.shape[-1] != T:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=T,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
        else:
            mask_tensor = torch.full((T,), float(value))

        return {"mask": Mask(tensor=mask_tensor)}


@NodeRegistry.register
class SetLatentNoiseMask(BaseNode):
    """Attach a noise mask to a latent for inpainting/blending.

    The Generate node will extract this mask and apply two-sided
    blending during the diffusion loop.
    """

    node_type_id: ClassVar[str] = "acestep.SetLatentNoiseMask"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Set Latent Noise Mask",
            category="mask",
            description="Attach a noise mask to a latent for selective denoising.",
            inputs=(
                NodePort(name="latent", type="LATENT"),
                NodePort(name="mask", type="MASK"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        latent: Latent = kwargs["latent"]
        mask: Mask = kwargs["mask"]

        noise_mask = LatentNoiseMask(
            mask=mask.tensor.to(device=latent.tensor.device, dtype=latent.tensor.dtype),
            original_latents=latent.tensor,
        )

        return {
            "latent": Latent(
                tensor=latent.tensor,
                mask=noise_mask,
            )
        }


@NodeRegistry.register
class InvertMask(BaseNode):
    """Invert a mask (1.0 - mask)."""

    node_type_id: ClassVar[str] = "acestep.InvertMask"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Invert Mask",
            category="mask",
            description="Invert a mask (swap preserve/generate regions).",
            inputs=(
                NodePort(name="mask", type="MASK"),
            ),
            outputs=(
                NodePort(name="mask", type="MASK"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        mask: Mask = kwargs["mask"]
        return {"mask": Mask(tensor=1.0 - mask.tensor)}
