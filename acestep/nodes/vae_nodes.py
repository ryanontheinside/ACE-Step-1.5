"""VAE encode/decode nodes."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import torch

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Audio, Curve, Latent, ModelHandle, VAEHandle

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# TRT VAE helpers (loaded once, reused across calls)
# -----------------------------------------------------------------------

_trt_vae_cache: dict[str, Any] = {}

# Shared polygraphy CUDA stream for all TRT engines in this process.
# Using torch.cuda.Stream causes a 14x performance degradation on
# Blackwell GPUs when multiple TRT engines coexist. Polygraphy's
# cuda.Stream (a thin wrapper around cudaStreamCreate) avoids this.
_trt_stream = None

def _get_trt_stream():
    """Get or create the shared polygraphy CUDA stream."""
    global _trt_stream
    if _trt_stream is None:
        from polygraphy import cuda as pg_cuda
        _trt_stream = pg_cuda.Stream()
    return _trt_stream


def _trt_available() -> bool:
    """Check if TensorRT is importable."""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _get_trt_vae(engine_path: str, device: torch.device):
    """Load or return cached TRT VAE engine + context + stream."""
    engine_path = os.path.abspath(engine_path)
    if engine_path in _trt_vae_cache:
        return _trt_vae_cache[engine_path]

    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes

    engine = engine_from_bytes(bytes_from_path(engine_path))
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
    stream = _get_trt_stream()

    # Ensure input is on GPU, fp32, contiguous
    lat = latents_bdt.to(device=device, dtype=torch.float32).contiguous()

    ctx.set_input_shape("latents", tuple(lat.shape))
    ctx.set_tensor_address("latents", lat.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("audio"))

    cached = entry.get("_decode_buf")
    if cached is not None and cached.shape == out_shape:
        audio_buf = cached
    else:
        audio_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
        entry["_decode_buf"] = audio_buf

    ctx.set_tensor_address("audio", audio_buf.data_ptr())

    if not ctx.execute_async_v3(stream.ptr):
        raise RuntimeError("TRT VAE decode failed")
    stream.synchronize()

    return audio_buf.clone()


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
    stream = _get_trt_stream()

    inp = audio_bct.float().contiguous().to(device)

    # Release PyTorch's unused reserved VRAM before TRT encode.
    torch.cuda.empty_cache()

    ctx.set_input_shape("audio", tuple(inp.shape))
    ctx.set_tensor_address("audio", inp.data_ptr())

    out_shape = tuple(ctx.get_tensor_shape("moments"))
    moments_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
    ctx.set_tensor_address("moments", moments_buf.data_ptr())

    if not ctx.execute_async_v3(stream.ptr):
        raise RuntimeError("TRT VAE encode failed")
    stream.synchronize()

    # Split moments into mean and logvar, sample
    mean, logvar = moments_buf.chunk(2, dim=1)  # [B, 64, T] each
    std = torch.exp(0.5 * logvar)
    latent = mean + std * torch.randn_like(mean)
    return latent


def _find_trt_engine(name: str) -> Optional[str]:
    """Search for a TRT engine file in trt_engines/."""
    pkg_root = os.path.join(os.path.dirname(__file__), "..", "..")
    candidates = [
        os.path.join("trt_engines", name),
        os.path.join(pkg_root, "trt_engines", name),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    return None


def _find_best_vae_engine(component: str) -> Optional[str]:
    """Find the best available VAE TRT engine (FP16 only).

    Args:
        component: "vae_decode" or "vae_encode"
    """
    candidates = [
        f"{component}_fp16.engine",
        f"{component}_fp16_max6000.engine",
    ]
    for name in candidates:
        path = _find_trt_engine(name)
        if path:
            return path
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

        trt_path = _find_best_vae_engine("vae_encode") if _trt_available() else None
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

        trt_path = _find_best_vae_engine("vae_decode") if _trt_available() else None
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
        silence = handler.silence_latent  # [1, T_full, D]

        T = int(duration * 25)  # 25 fps latent rate
        # Take first T frames from the silence latent (tiled if needed)
        if silence.dim() == 3:
            latent = silence[:, :T, :].clone()
            if latent.shape[1] < T:
                reps = (T + latent.shape[1] - 1) // latent.shape[1]
                latent = latent.repeat(1, reps, 1)[:, :T, :]
        else:
            # Fallback: treat as [1, D] single frame
            latent = silence.unsqueeze(0).expand(1, T, -1).clone()

        return {"latent": Latent(tensor=latent)}


@NodeRegistry.register
class LatentBlend(BaseNode):
    """Blend two latents by weighted interpolation.

    Supports scalar or per-frame (CURVE) blend factor.
    Useful for timbre strength control (blend reference with silence)
    or mixing any two latent representations.

    Node parameters:
        alpha: Blend factor (0.0 = all A, 1.0 = all B).
               Ignored if a blend_curve input is connected.
    """

    node_type_id: ClassVar[str] = "acestep.LatentBlend"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Latent Blend",
            category="vae",
            description="Blend two latents with scalar or per-frame factor.",
            inputs=(
                NodePort(name="latent_a", type="LATENT"),
                NodePort(name="latent_b", type="LATENT"),
                NodePort(
                    name="blend_curve",
                    type="CURVE",
                    required=False,
                    description="Per-frame blend factor (overrides scalar alpha).",
                ),
            ),
            outputs=(
                NodePort(name="latent", type="LATENT"),
            ),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        latent_a: Latent = kwargs["latent_a"]
        latent_b: Latent = kwargs["latent_b"]
        blend_curve: Optional[Curve] = kwargs.get("blend_curve")

        a = latent_a.tensor
        b = latent_b.tensor

        alpha = kwargs.get("alpha", 0.5)
        if blend_curve is not None:
            alpha = blend_curve.tensor.to(device=a.device, dtype=a.dtype)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            elif alpha.ndim == 2:
                alpha = alpha.unsqueeze(-1)  # [B, T, 1]

        blended = (1.0 - alpha) * a + alpha * b
        return {"latent": Latent(tensor=blended)}
