"""Model management nodes."""

from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import CLIPHandle, ModelHandle, VAEHandle


@NodeRegistry.register
class LoadModel(BaseNode):
    """Load the ACE-Step checkpoint and return model/clip/vae handles.

    Node parameters (passed via execute kwargs):
        project_root: Path to project root or checkpoints directory.
        config_path: Model config directory name (e.g. "acestep-v15-turbo").
        device: Device string ("auto", "cuda", "cpu").
        use_flash_attention: Whether to use flash attention.
        compile_model: Whether to torch.compile the decoder.
        offload_to_cpu: Whether to offload models when not in use.
        quantization: Quantization type (None, "int8_weight_only", etc.).
    """

    node_type_id: ClassVar[str] = "acestep.LoadModel"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Load ACE-Step Model",
            category="model",
            description="Load checkpoint and return MODEL, CLIP, VAE handles.",
            inputs=(),
            outputs=(
                NodePort(name="model", type="MODEL"),
                NodePort(name="clip", type="CLIP"),
                NodePort(name="vae", type="VAE"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        from acestep.handler import AceStepHandler

        handler = AceStepHandler()
        handler.initialize_service(
            project_root=kwargs.get("project_root", "checkpoints"),
            config_path=kwargs.get("config_path", "acestep-v15-turbo"),
            device=kwargs.get("device", "auto"),
            use_flash_attention=kwargs.get("use_flash_attention", False),
            compile_model=kwargs.get("compile_model", False),
            offload_to_cpu=kwargs.get("offload_to_cpu", False),
            quantization=kwargs.get("quantization", None),
        )

        return {
            "model": ModelHandle(handler=handler),
            "clip": CLIPHandle(handler=handler),
            "vae": VAEHandle(handler=handler),
        }
