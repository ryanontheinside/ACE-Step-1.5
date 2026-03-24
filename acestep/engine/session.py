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
from typing import Any, List, Optional

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

    When ``trt_engines`` is provided, decoder and/or VAE PyTorch weights
    are never loaded to GPU (or at all, for the VAE). The TRT engines
    are loaded via polygraphy and wired into the DiffusionEngine and
    VAE node cache directly.

    Example::

        s = Session(
            project_root=".",
            compile_model=False,
            trt_engines={
                "decoder": "trt_engines/decoder_mixed_v5.engine",
                "vae_encode": "trt_engines/vae_encode_fp16_max6000.engine",
                "vae_decode": "trt_engines/vae_decode_fp16_max6000.engine",
            },
        )
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
        trt_engines: Optional[dict[str, str]] = None,
        vae_window: float = 0.0,
        vae_overlap: float = 0.5,
    ):
        import torch
        from acestep.handler import AceStepHandler

        skip_decoder = bool(trt_engines and "decoder" in trt_engines)
        skip_vae = bool(
            trt_engines
            and "vae_encode" in trt_engines
            and "vae_decode" in trt_engines
        )

        handler = AceStepHandler()
        handler.initialize_service(
            project_root=project_root,
            config_path=config_path,
            device=device,
            compile_model=compile_model,
            use_flash_attention=use_flash_attention,
            offload_to_cpu=offload_to_cpu,
            quantization=quantization,
            skip_decoder=skip_decoder,
            skip_vae=skip_vae,
        )

        self.model = ModelHandle(handler=handler)
        self.clip = CLIPHandle(handler=handler)
        self.vae = VAEHandle(handler=handler)

        # Windowed VAE decode config (seconds; 0 = full decode)
        self._vae_window = vae_window
        self._vae_overlap = vae_overlap

        # Wire up TRT engines
        if trt_engines:
            if "decoder" in trt_engines:
                from acestep.engine.diffusion import DiffusionEngine
                handler._diffusion_engine = DiffusionEngine(
                    handler.model,
                    trt_engine_path=trt_engines["decoder"],
                )

            # Pre-load VAE TRT engines into the node cache
            from acestep.nodes.vae_nodes import _get_trt_vae
            dev = torch.device(device)
            if "vae_encode" in trt_engines:
                _get_trt_vae(trt_engines["vae_encode"], dev)
            if "vae_decode" in trt_engines:
                _get_trt_vae(trt_engines["vae_decode"], dev)

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
        method: str = "ode",
        **kwargs: Any,
    ) -> Latent:
        """Run the diffusion loop. Always executes (never cached)."""
        from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate

        config = DiffusionConfigNode().execute(
            steps=steps, shift=shift, seed=seed, denoise=denoise,
            method=method,
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

    def decode(self, latent: Latent, t_start: float = 0.0) -> Audio:
        """VAE decode latent to audio waveform.

        Args:
            latent: Latent to decode.
            t_start: When ``vae_window`` > 0, the start time (seconds) of
                the window to decode. The returned Audio contains only
                the interior of that window (overlap margins are used
                for context but trimmed from output). Ignored when
                ``vae_window`` is 0.

        Returns:
            Audio starting at ``t_start`` with duration ``vae_window``
            (or the full latent when windowing is off).
        """
        from acestep.nodes.vae_nodes import VAEDecodeAudio

        if self._vae_window <= 0:
            return VAEDecodeAudio().execute(vae=self.vae, latent=latent)["audio"]

        return self._decode_windowed(latent, t_start)

    def _decode_windowed(self, latent: Latent, t_start: float) -> Audio:
        """Decode a single window of the latent with overlap margins.

        Returns Audio with ``start_seconds`` set to the frame-quantized
        start time so callers know exactly where the window begins.
        """
        import torch
        from acestep.nodes.vae_nodes import VAEDecodeAudio

        FRAMES_PER_SEC = 25
        SAMPLES_PER_FRAME = 1920  # 48000 / 25

        tensor = latent.tensor  # [1, T, D]
        T = tensor.shape[1]

        win_frames = int(self._vae_window * FRAMES_PER_SEC)
        ovl_frames = int(self._vae_overlap * FRAMES_PER_SEC)

        # If the latent fits in one window, just decode the whole thing
        if T <= win_frames:
            return VAEDecodeAudio().execute(vae=self.vae, latent=latent)["audio"]

        # Clamp keep region to valid range (frame-quantized)
        keep_start = max(0, int(t_start * FRAMES_PER_SEC))
        keep_end = min(T, keep_start + win_frames)
        keep_start = max(0, keep_end - win_frames)  # adjust if clamped at end

        # Extend by overlap margins for VAE receptive field context
        decode_start = max(0, keep_start - ovl_frames)
        decode_end = min(T, keep_end + ovl_frames)

        # Decode the window
        chunk_lat = Latent(tensor=tensor[:, decode_start:decode_end, :].contiguous())
        chunk_audio = VAEDecodeAudio().execute(
            vae=self.vae, latent=chunk_lat
        )["audio"]

        # Trim overlap margins, return only the clean interior
        pre_margin = (keep_start - decode_start) * SAMPLES_PER_FRAME
        keep_samples = (keep_end - keep_start) * SAMPLES_PER_FRAME
        trimmed = chunk_audio.waveform[:, :, pre_margin:pre_margin + keep_samples]

        return Audio(waveform=trimmed, sample_rate=48000,
                     start_sample=keep_start * SAMPLES_PER_FRAME)

    # ------------------------------------------------------------------
    # Audio analysis
    # ------------------------------------------------------------------

    @staticmethod
    def audio_info(audio: Audio) -> dict:
        """Detect BPM, key, and duration from audio."""
        from acestep.nodes.audio_nodes import AudioInfo

        return AudioInfo().execute(audio=audio)

    # ------------------------------------------------------------------
    # Latent / LoRA utilities
    # ------------------------------------------------------------------

    def empty_latent(self, duration: float = 60.0) -> Latent:
        """Create a silence latent of a given duration."""
        from acestep.nodes.vae_nodes import EmptyLatent

        return EmptyLatent().execute(
            model=self.model, duration=duration,
        )["latent"]

    @staticmethod
    def blend_latents(
        a: Latent, b: Latent, alpha: float = 0.5,
    ) -> Latent:
        """Blend two latents. 0.0 = all A, 1.0 = all B."""
        from acestep.nodes.vae_nodes import LatentBlend

        return LatentBlend().execute(
            latent_a=a, latent_b=b, alpha=alpha,
        )["latent"]

    def apply_lora(self, path: str, scale: float = 1.0) -> None:
        """Load and apply a LoRA. Stackable (call multiple times)."""
        from acestep.nodes.lora_nodes import LoadLoRA, ApplyLoRA

        lora = LoadLoRA().execute(path=path, scale=scale)["lora"]
        ApplyLoRA().execute(model=self.model, lora=lora)
        if not hasattr(self, '_lora_stack'):
            self._lora_stack = []
        self._lora_stack.append(lora)

    def remove_loras(self) -> None:
        """Remove all applied LoRAs in reverse order."""
        from acestep.nodes.lora_nodes import RemoveLoRA

        if hasattr(self, '_lora_stack'):
            while self._lora_stack:
                RemoveLoRA().execute(
                    model=self.model, lora=self._lora_stack.pop(),
                )

    def remove_last_lora(self) -> None:
        """Remove the most recently applied LoRA."""
        from acestep.nodes.lora_nodes import RemoveLoRA

        if hasattr(self, '_lora_stack') and self._lora_stack:
            RemoveLoRA().execute(
                model=self.model, lora=self._lora_stack.pop(),
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def create_stream(
        self,
        *,
        source: PreparedSource,
        conditioning: Conditioning,
        steps: int = 8,
        shift: float = 3.0,
        noise_sharing: float = 0.0,
    ) -> "SessionStream":
        """Create a streaming pipeline for interactive generation.

        Returns a ``SessionStream`` that wraps the low-level
        ``StreamPipeline`` and ``SlotRequest`` construction so callers
        work with Session-level types only.
        """
        from .diffusion import DiffusionConfig
        from .stream import StreamPipeline

        engine = self.handler._diffusion_engine
        config = DiffusionConfig(
            infer_steps=steps, shift=shift, noise_on_cpu=True,
        )
        pipe = StreamPipeline(engine, config, noise_sharing=noise_sharing)

        return SessionStream(
            session=self,
            pipeline=pipe,
            config=config,
            source=source,
            conditioning=conditioning,
        )


class SessionStream:
    """Interactive streaming pipeline bound to a Session.

    Wraps ``StreamPipeline`` so callers work with ``Conditioning``,
    ``PreparedSource``, and ``Latent`` instead of raw tensors and
    ``SlotRequest``.

    Created via ``Session.create_stream()``.
    """

    def __init__(
        self,
        session: Session,
        pipeline: "StreamPipeline",
        config: Any,
        source: PreparedSource,
        conditioning: Conditioning,
    ):
        import torch
        self.session = session
        self.pipeline = pipeline
        self.config = config
        self.source = source
        self.conditioning = conditioning

        device = session.handler.device
        dtype = session.handler.dtype
        T = source.latent.tensor.shape[1]

        # Pre-build the tensors that stay constant across submissions
        entry = conditioning.to_entries()[0]
        self._encoder_hidden_states = entry.encoder_hidden_states
        self._encoder_attention_mask = entry.encoder_attention_mask

        ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
        D = ctx_lat.shape[2]
        cm = torch.ones(1, T, D, device=device, dtype=dtype)
        self._context_latents = torch.cat([ctx_lat, cm], dim=-1)
        self._source_latents = source.latent.tensor.to(device=device, dtype=dtype)

    def submit(
        self,
        *,
        denoise: float = 1.0,
        seed: Optional[int] = None,
        source_latents: Optional["torch.Tensor"] = None,
        sde_denoise_curve: Optional["torch.Tensor"] = None,
    ) -> None:
        """Enqueue a generation request.

        Args:
            denoise: Denoise strength for this request.
            seed: RNG seed (None = random).
            source_latents: Override source latents (e.g. for latent
                feedback). If None, uses the PreparedSource latents.
            sde_denoise_curve: Per-frame denoise curve [1, T, 1].
        """
        from .stream import SlotRequest

        self.pipeline.submit(SlotRequest(
            encoder_hidden_states=self._encoder_hidden_states,
            encoder_attention_mask=self._encoder_attention_mask,
            context_latents=self._context_latents,
            seed=seed,
            source_latents=(
                source_latents if source_latents is not None
                else self._source_latents
            ),
            denoise=denoise,
            sde_denoise_curve=sde_denoise_curve,
        ))

    def tick(self) -> Optional[Latent]:
        """Advance the pipeline by one step.

        Returns a finished ``Latent`` when available, otherwise None.
        """
        result = self.pipeline.tick()
        if result is None:
            return None
        return Latent(tensor=result)

    @property
    def active_slots(self) -> int:
        return self.pipeline.active_slots

    @property
    def source_latents(self) -> "torch.Tensor":
        """The source latent tensor on device (read-only)."""
        return self._source_latents

    def set_shift(self, shift: float) -> None:
        """Update the diffusion shift parameter, clearing cached schedules."""
        self.config.shift = shift
        self.pipeline._schedule_cache.clear()

    def stats(self) -> dict:
        return self.pipeline.stats()
