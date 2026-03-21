"""Persistent session for ACE-Step generation.

Loads the model once and keeps handler, compiled decoder, and TRT engines
alive across multiple generation calls. Provides convenience methods that
delegate to the node system, so intermediate results (latents, hints,
conditioning) can be held by the caller and reused without recomputation.

Typical usage (cover iteration with different seeds):

    session = Session(project_root=".", compile_model=True)
    source = session.prepare_source(audio)
    cond = session.encode_text(
        tags="deathstep", instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent, bpm=136, duration=60.0, key="G# minor",
    )
    for seed in [1528, 9999, 42]:
        output = session.generate(
            conditioning=cond, context_latent=source.context_latent,
            source_latent=source.latent, seed=seed,
        )
        save_audio(session.decode(output), f"out_{seed}.wav")

When Daydream Scope integrates, its session management replaces this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from acestep.constants import TASK_INSTRUCTIONS
from acestep.nodes.types import (
    Audio,
    CLIPHandle,
    Conditioning,
    Curve,
    Latent,
    Mask,
    ModelHandle,
    SemanticHints,
    VAEHandle,
)


@dataclass
class PreparedSource:
    """Cached results from preparing a source audio."""
    latent: Latent
    hints: SemanticHints
    context_latent: Latent


class Session:
    """Persistent GPU state for ACE-Step generation.

    Loads the model, VAE, text encoder, and TRT engines once. Exposes
    node handle types and convenience methods that wrap node execution.

    Intermediate results are returned to the caller; the caller controls
    what gets reused between generations by holding references.
    """

    def __init__(
        self,
        *,
        project_root: str = "checkpoints",
        config_path: str = "acestep-v15-turbo",
        device: str = "cuda",
        compile_model: bool = True,
        use_flash_attention: bool = True,
        offload_to_cpu: bool = False,
        quantization: Optional[str] = None,
    ):
        from acestep.handler import AceStepHandler

        handler = AceStepHandler()
        handler.initialize_service(
            project_root=project_root,
            config_path=config_path,
            device=device,
            compile_model=compile_model,
            use_flash_attention=use_flash_attention,
            offload_to_cpu=offload_to_cpu,
            quantization=quantization,
        )

        self.model = ModelHandle(handler=handler)
        self.clip = CLIPHandle(handler=handler)
        self.vae = VAEHandle(handler=handler)

    @property
    def handler(self):
        return self.model.handler

    # ------------------------------------------------------------------
    # Source preparation
    # ------------------------------------------------------------------

    def encode_audio(self, audio: Audio) -> Latent:
        """VAE encode audio waveform to latent."""
        from acestep.nodes.vae_nodes import VAEEncodeAudio

        return VAEEncodeAudio().execute(vae=self.vae, audio=audio)["latent"]

    def extract_hints(self, latent: Latent) -> SemanticHints:
        """Extract semantic structural hints from a latent."""
        from acestep.nodes.semantic_nodes import SemanticExtract

        return SemanticExtract().execute(
            model=self.model, latent=latent
        )["semantic_hints"]

    def hints_to_latent(self, hints: SemanticHints) -> Latent:
        """Convert semantic hints to latent type for use as context."""
        from acestep.nodes.semantic_nodes import SemanticHintsToLatent

        return SemanticHintsToLatent().execute(
            semantic_hints=hints
        )["latent"]

    def prepare_source(self, audio: Audio) -> PreparedSource:
        """VAE encode + semantic extract + convert in one call.

        Returns a PreparedSource holding all three intermediate results.
        The caller holds this and reuses it across generations.
        """
        latent = self.encode_audio(audio)
        hints = self.extract_hints(latent)
        context_latent = self.hints_to_latent(hints)
        return PreparedSource(
            latent=latent, hints=hints, context_latent=context_latent,
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(
        self,
        *,
        tags: str = "",
        lyrics: str = "",
        instruction: Optional[str] = None,
        refer_latent: Optional[Latent] = None,
        bpm: int = 120,
        duration: float = 60.0,
        key: str = "C major",
        time_signature: str = "4",
        language: str = "en",
    ) -> Conditioning:
        """Encode text prompt into cross-attention conditioning."""
        from acestep.nodes.cond_nodes import TextEncode

        if instruction is None:
            instruction = TASK_INSTRUCTIONS["text2music"]

        return TextEncode().execute(
            clip=self.clip,
            model=self.model,
            refer_latent=refer_latent,
            tags=tags,
            lyrics=lyrics,
            instruction=instruction,
            bpm=bpm,
            duration=duration,
            key=key,
            time_signature=time_signature,
            language=language,
        )["conditioning"]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        conditioning: Conditioning,
        context_latent: Optional[Latent] = None,
        chunk_mask: Optional[Mask] = None,
        source_latent: Optional[Latent] = None,
        seed: Optional[int] = None,
        denoise: float = 1.0,
        steps: int = 8,
        shift: float = 3.0,
        **kwargs: Any,
    ) -> Latent:
        """Run the diffusion loop. Always executes (never cached)."""
        from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate

        config = DiffusionConfigNode().execute(
            steps=steps, shift=shift, seed=seed, denoise=denoise,
        )["config"]

        return Generate().execute(
            model=self.model,
            config=config,
            positive=conditioning,
            context_latent=context_latent,
            chunk_mask=chunk_mask,
            source_latent=source_latent,
            **kwargs,
        )["latent"]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, latent: Latent) -> Audio:
        """VAE decode latent to audio waveform."""
        from acestep.nodes.vae_nodes import VAEDecodeAudio

        return VAEDecodeAudio().execute(vae=self.vae, latent=latent)["audio"]
