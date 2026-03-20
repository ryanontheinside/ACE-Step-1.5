"""VAE encode/decode nodes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import torch

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Audio, Latent, ModelHandle, VAEHandle

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# TRT VAE helpers (loaded once, reused across calls)
# -----------------------------------------------------------------------

_trt_vae_cache: dict[str, Any] = {}


def _trt_available() -> bool:
    """Check if TensorRT is importable."""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _get_trt_vae(engine_path: str, device: torch.device):
    """Load or return cached TRT VAE engine + context."""
    if engine_path in _trt_vae_cache:
        return _trt_vae_cache[engine_path]

    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TRT engine: {engine_path}")
    ctx = engine.create_execution_context()
    logger.info("Loaded TRT VAE engine: %s", engine_path)

    entry = {"engine": engine, "context": ctx}
    _trt_vae_cache[engine_path] = entry
    return entry


def _trt_vae_decode(
    latents_bdt: torch.Tensor, engine_path: str, device: torch.device
) -> torch.Tensor:
    """Decode latents [B, D, T] -> audio [B, 2, samples] via TRT."""
    entry = _get_trt_vae(engine_path, device)
    ctx = entry["context"]

    # Ensure input is on GPU, fp32, contiguous
    lat = latents_bdt.to(device=device, dtype=torch.float32).contiguous()

    # Sync before TRT to ensure all prior CUDA work is done
    torch.cuda.synchronize(device)

    ctx.set_input_shape("latents", tuple(lat.shape))
    ctx.set_tensor_address("latents", lat.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("audio"))
    audio_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("audio", audio_buf.data_ptr())

    # Use a dedicated stream for TRT execution
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT VAE decode failed")
    stream.synchronize()

    return audio_buf


def _trt_vae_encode(
    audio_bct: torch.Tensor, engine_path: str, device: torch.device
) -> torch.Tensor:
    """Encode audio [B, 2, samples] -> latents [B, D, T] via TRT.

    The ONNX export produces moments [B, 128, T] (mean+logvar concatenated).
    We split and sample: latent = mean + exp(0.5 * logvar) * noise,
    matching the VAE's latent_dist.sample() behavior.
    """
    entry = _get_trt_vae(engine_path, device)
    ctx = entry["context"]

    inp = audio_bct.float().contiguous().to(device)
    ctx.set_input_shape("audio", tuple(inp.shape))
    ctx.set_tensor_address("audio", inp.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("moments"))
    moments_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("moments", moments_buf.data_ptr())

    stream = torch.cuda.current_stream(device)
    if not ctx.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TRT VAE encode failed")
    stream.synchronize()

    # Split moments into mean and logvar, sample
    mean, logvar = moments_buf.chunk(2, dim=1)  # [B, 64, T] each
    std = torch.exp(0.5 * logvar)
    latent = mean + std * torch.randn_like(mean)
    return latent


def _find_trt_engine(name: str) -> Optional[str]:
    """Search for a TRT engine file in common locations."""
    candidates = [
        os.path.join("trt_engines", name),
        os.path.join(os.path.dirname(__file__), "..", "..", "trt_engines", name),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    return None


# -----------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------

@NodeRegistry.register
class VAEEncodeAudio(BaseNode):
    """Encode audio waveform to latent space.

    Uses TRT engine if available (vae_encode_fp16.engine), falls back
    to PyTorch VAE via the handler.
    """

    node_type_id: ClassVar[str] = "acestep.VAEEncodeAudio"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="VAE Encode Audio",
            category="vae",
            description="Encode audio waveform to latent representation.",
            inputs=(
                NodePort(name="vae", type="VAE"),
                NodePort(name="audio", type="AUDIO"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        vae: VAEHandle = kwargs["vae"]
        audio: Audio = kwargs["audio"]
        handler = vae.handler
        device = torch.device(handler.device)
        dtype = handler.dtype

        waveform = audio.waveform
        # Ensure [B, C, samples]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        trt_path = _find_trt_engine("vae_encode_fp16.engine") if _trt_available() else None
        if trt_path:
            logger.info("VAE encode via TRT")
            latents_bdt = _trt_vae_encode(waveform, trt_path, device)
            # [B, D, T] -> [B, T, D]
            latents = latents_bdt.transpose(1, 2).to(dtype)
        else:
            logger.info("VAE encode via PyTorch")
            with handler._load_model_context("vae"):
                latents = handler._encode_audio_to_latents(waveform)
            if latents.dim() == 2:
                latents = latents.unsqueeze(0)

        # Pad T to multiple of 5 (required by model tokenizer)
        T = latents.shape[1]
        pad_to = 5
        if T % pad_to != 0:
            pad_amount = pad_to - (T % pad_to)
            latents = torch.nn.functional.pad(latents, (0, 0, 0, pad_amount))

        return {"latent": Latent(tensor=latents)}


@NodeRegistry.register
class VAEDecodeAudio(BaseNode):
    """Decode latents back to audio waveform.

    Uses TRT engine if available (vae_decode_fp16.engine), falls back
    to PyTorch VAE via the handler.
    """

    node_type_id: ClassVar[str] = "acestep.VAEDecodeAudio"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="VAE Decode Audio",
            category="vae",
            description="Decode latent representation to audio waveform.",
            inputs=(
                NodePort(name="vae", type="VAE"),
                NodePort(name="latent", type="LATENT"),
            ),
            outputs=(
                NodePort(name="audio", type="AUDIO"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        vae: VAEHandle = kwargs["vae"]
        latent: Latent = kwargs["latent"]
        handler = vae.handler
        device = torch.device(handler.device)

        # [B, T, D] -> [B, D, T]
        lat_bdt = latent.tensor.transpose(1, 2)

        trt_path = _find_trt_engine("vae_decode_fp16.engine") if _trt_available() else None
        if trt_path:
            logger.info("VAE decode via TRT")
            waveform = _trt_vae_decode(lat_bdt, trt_path, device)
        else:
            logger.info("VAE decode via PyTorch (no TRT engine found)")
            with handler._load_model_context("vae"):
                waveform = handler.tiled_decode(lat_bdt)

        return {"audio": Audio(waveform=waveform, sample_rate=48000)}


@NodeRegistry.register
class EmptyLatent(BaseNode):
    """Create a silence-based latent of a given duration.

    Node parameters:
        duration: Duration in seconds.
    """

    node_type_id: ClassVar[str] = "acestep.EmptyLatent"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Empty ACE-Step Latent",
            category="vae",
            description="Create an empty (silence) latent for a given duration.",
            inputs=(
                NodePort(name="model", type="MODEL"),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        model: ModelHandle = kwargs["model"]
        handler = model.handler
        duration = kwargs.get("duration", 60.0)

        handler._ensure_silence_latent_on_device()
        silence = handler.silence_latent  # [1, D]

        T = int(duration * 25)  # 25 fps latent rate
        latent = silence.unsqueeze(0).expand(1, T, -1).clone()

        return {"latent": Latent(tensor=latent)}
