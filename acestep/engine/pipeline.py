"""Pipelined generation: overlaps VAE decode with diffusion on separate CUDA streams.

In continuous generation (user tweaking parameters, regenerating repeatedly),
the decode of generation N overlaps with the diffusion of generation N+1.
This hides ~11ms of decode latency per generation.

Usage:
    pipeline = GenerationPipeline(engine, trt_decode_fn)

    # Single generation (no overlap, same as calling engine.generate + decode):
    audio = pipeline.generate_single(condition_set, config, source_latents)

    # Continuous generation (pipelined):
    pipeline.start()
    for params in stream_of_params:
        audio = pipeline.generate_next(params.condition_set, params.config, ...)
        # audio is from the PREVIOUS generation (pipelined)
        play(audio)
    final_audio = pipeline.flush()  # get the last one
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from .conditions import ConditionSet
from .diffusion import DiffusionConfig, DiffusionEngine
from .masking import LatentNoiseMask

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Manages pipelined diffusion + decode with double-buffered CUDA streams.

    The diffusion runs on the default CUDA stream. VAE decode runs on a
    dedicated decode stream. When generating continuously, decode of the
    previous result overlaps with diffusion of the next.

    The decode function is injected (not hardcoded to TRT) so this works
    with any decode backend.
    """

    def __init__(
        self,
        engine: DiffusionEngine,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device | str = "cuda",
    ):
        """
        Args:
            engine: DiffusionEngine instance (should be persistent, not recreated).
            decode_fn: Function that takes latents [B, T, D] and returns
                audio [B, C, samples]. Must be GPU-resident (no CPU sync
                inside). Will be called on the decode stream.
            device: CUDA device.
        """
        self.engine = engine
        self.decode_fn = decode_fn
        self.device = torch.device(device)

        # Decode runs on its own stream
        self._decode_stream = torch.cuda.Stream(self.device)

        # Double-buffered: pending decode result from previous generation
        self._pending_audio: Optional[torch.Tensor] = None
        self._pending_latents: Optional[torch.Tensor] = None
        self._has_pending = False

    def generate_single(
        self,
        condition_set: ConditionSet,
        config: DiffusionConfig,
        source_latents: Optional[torch.Tensor] = None,
        latent_mask: Optional[LatentNoiseMask] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate and decode synchronously (no pipelining).

        Returns audio tensor [B, C, samples].
        """
        result = self.engine.generate(
            condition_set=condition_set,
            config=config,
            source_latents=source_latents,
            latent_mask=latent_mask,
            **kwargs,
        )
        return self.decode_fn(result["target_latents"])

    def generate_next(
        self,
        condition_set: ConditionSet,
        config: DiffusionConfig,
        source_latents: Optional[torch.Tensor] = None,
        latent_mask: Optional[LatentNoiseMask] = None,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Generate pipelined: returns audio from the PREVIOUS generation.

        First call returns None (no previous result yet).
        Subsequent calls return the decoded audio from the prior generation
        while the current generation's decode runs in the background.
        """
        # Wait for any pending decode to finish
        prev_audio = None
        if self._has_pending:
            self._decode_stream.synchronize()
            prev_audio = self._pending_audio
            self._has_pending = False

        # Run diffusion on default stream
        result = self.engine.generate(
            condition_set=condition_set,
            config=config,
            source_latents=source_latents,
            latent_mask=latent_mask,
            **kwargs,
        )
        latents = result["target_latents"]

        # Launch decode on decode stream (overlaps with next diffusion)
        self._decode_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._decode_stream):
            self._pending_audio = self.decode_fn(latents)
        self._has_pending = True

        return prev_audio

    def flush(self) -> Optional[torch.Tensor]:
        """Wait for and return the last pending decode result.

        Call this after the final generate_next() to get the last audio.
        """
        if self._has_pending:
            self._decode_stream.synchronize()
            self._has_pending = False
            return self._pending_audio
        return None

    def reset(self) -> None:
        """Discard any pending decode and reset pipeline state."""
        if self._has_pending:
            self._decode_stream.synchronize()
        self._pending_audio = None
        self._pending_latents = None
        self._has_pending = False
