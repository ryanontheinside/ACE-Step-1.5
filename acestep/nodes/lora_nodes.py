"""LoRA loading and application nodes.

Supports two paths:
  - PyTorch path: modifies decoder parameters directly (original behavior).
  - TRT refit path: when a REFIT-enabled TRT engine is loaded, applies
    LoRA via weight refitting without touching PyTorch parameters.
    Detected automatically; no user configuration needed.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from safetensors.torch import load_file

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import LoRA, ModelHandle

logger = logging.getLogger(__name__)


def _has_trt_lora(handler) -> bool:
    """Check if handler has a TRT engine with LoRA refit support."""
    engine = getattr(handler, "_diffusion_engine", None)
    if engine is None:
        return False
    return getattr(engine, "trt_lora_available", False)


@NodeRegistry.register
class LoadLoRA(BaseNode):
    """Load a LoRA adapter from a safetensors file.

    Pre-computes the weight deltas (B @ A * scale) so they can be
    quickly applied/removed from the decoder.

    Node parameters:
        path: Path to the .safetensors LoRA file.
        scale: LoRA strength multiplier (default 1.0).
    """

    node_type_id: ClassVar[str] = "acestep.LoadLoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Load LoRA",
            category="model",
            description="Load a LoRA adapter from a safetensors file.",
            inputs=(),
            outputs=(
                NodePort(name="lora", type="LORA"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        path = kwargs["path"]
        scale = kwargs.get("scale", 1.0)
        return {"lora": LoRA(path=str(path), scale=float(scale))}


@NodeRegistry.register
class ApplyLoRA(BaseNode):
    """Apply a LoRA adapter to the model and return the modified handle.

    Automatically uses TRT weight refitting when a REFIT-enabled TRT
    engine is loaded, otherwise falls back to direct PyTorch parameter
    modification.
    """

    node_type_id: ClassVar[str] = "acestep.ApplyLoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Apply LoRA",
            category="model",
            description="Apply a LoRA adapter to the model for generation.",
            inputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="lora", type="LORA"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        lora: LoRA = kwargs["lora"]
        handler = model_handle.handler

        # TRT refit path: apply LoRA to the TRT engine directly
        if _has_trt_lora(handler):
            lora_id = handler._diffusion_engine.apply_trt_lora(
                lora.path, lora.scale,
            )
            # Store lora_id for RemoveLoRA
            if not hasattr(handler, '_active_trt_lora_ids'):
                handler._active_trt_lora_ids = []
            handler._active_trt_lora_ids.append(lora_id)
            return {"model": model_handle}

        # PyTorch path: modify decoder parameters directly
        deltas = _precompute_lora_deltas(
            lora.path, lora.scale,
            handler.device, handler.dtype,
        )

        with handler._load_model_context("model"):
            _apply_lora_deltas(handler.model.decoder, deltas, sign=1.0)
        logger.info(
            "Applied LoRA: %s (%d params, scale=%.2f)",
            lora.path, len(deltas), lora.scale,
        )

        if not hasattr(handler, '_active_lora_deltas'):
            handler._active_lora_deltas = []
        handler._active_lora_deltas.append(deltas)

        return {"model": model_handle}


@NodeRegistry.register
class RemoveLoRA(BaseNode):
    """Remove the most recently applied LoRA from the model.

    Handles both TRT refit and PyTorch parameter paths.
    """

    node_type_id: ClassVar[str] = "acestep.RemoveLoRA"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Remove LoRA",
            category="model",
            description="Remove the most recently applied LoRA adapter.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="model", type="MODEL"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model_handle: ModelHandle = kwargs["model"]
        handler = model_handle.handler

        # TRT refit path
        if _has_trt_lora(handler):
            ids = getattr(handler, '_active_trt_lora_ids', [])
            if ids:
                lora_id = ids.pop()
                handler._diffusion_engine.remove_trt_lora(lora_id)
            return {"model": model_handle}

        # PyTorch path
        if hasattr(handler, '_active_lora_deltas') and handler._active_lora_deltas:
            deltas = handler._active_lora_deltas.pop()
            with handler._load_model_context("model"):
                _apply_lora_deltas(handler.model.decoder, deltas, sign=-1.0)
            logger.info("Removed LoRA (%d params)", len(deltas))

        return {"model": model_handle}


# -----------------------------------------------------------------------
# Helpers (PyTorch path)
# -----------------------------------------------------------------------

def _precompute_lora_deltas(
    lora_path: str,
    strength: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Load LoRA weights and compute full-rank deltas: strength * (B @ A)."""
    raw = load_file(lora_path)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in raw.items():
        parts = key.replace("base_model.model.", "")
        if ".lora_A.weight" in parts:
            param_name = parts.replace(".lora_A.weight", ".weight")
            pairs.setdefault(param_name, {})["A"] = tensor
        elif ".lora_B.weight" in parts:
            param_name = parts.replace(".lora_B.weight", ".weight")
            pairs.setdefault(param_name, {})["B"] = tensor

    deltas = {}
    for param_name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].to(device=device, dtype=dtype)
        B = ab["B"].to(device=device, dtype=dtype)
        deltas[param_name] = strength * (B @ A)

    return deltas


def _apply_lora_deltas(
    decoder: torch.nn.Module,
    deltas: dict[str, torch.Tensor],
    sign: float = 1.0,
) -> None:
    """Add (sign=1) or remove (sign=-1) precomputed deltas from decoder params."""
    decoder_params = dict(decoder.named_parameters())
    applied = 0
    for param_name, delta in deltas.items():
        if param_name in decoder_params:
            decoder_params[param_name].data.add_(delta, alpha=sign)
            applied += 1
    logger.info("LoRA delta %s: %d/%d params", "applied" if sign > 0 else "removed", applied, len(deltas))
