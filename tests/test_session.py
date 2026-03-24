"""Tests for Session public API.

These validate the contract that Session exposes. If handler.py is
refactored, every test here should still pass.

Run:  uv run pytest tests/test_session.py -v
"""

import pytest
import torch

from acestep.engine.session import Session, PreparedSource
from acestep.nodes.types import Audio, Latent, Conditioning, SemanticHints

SAMPLE_RATE = 48000
FRAMES_PER_SEC = 25


# ---------------------------------------------------------------------------
# Source preparation
# ---------------------------------------------------------------------------

class TestSourcePreparation:

    def test_encode_audio_returns_latent(self, session, source_audio):
        latent = session.encode_audio(source_audio)
        assert isinstance(latent, Latent)

    def test_encode_audio_shape(self, session, source_audio):
        latent = session.encode_audio(source_audio)
        B, T, D = latent.tensor.shape
        assert B == 1
        assert D == 64
        expected_T = source_audio.waveform.shape[-1] // 1920
        assert T == expected_T

    def test_extract_hints_shape(self, session, prepared_source):
        hints = session.extract_hints(prepared_source.latent)
        assert isinstance(hints, SemanticHints)
        assert hints.tensor.shape[0] == 1
        assert hints.tensor.shape[1] == prepared_source.latent.tensor.shape[1]

    def test_hints_to_latent_shape(self, session, prepared_source):
        latent = session.hints_to_latent(prepared_source.hints)
        assert isinstance(latent, Latent)
        assert latent.tensor.shape[1] == prepared_source.latent.tensor.shape[1]

    def test_prepare_source_fields(self, prepared_source):
        assert isinstance(prepared_source, PreparedSource)
        assert isinstance(prepared_source.latent, Latent)
        assert isinstance(prepared_source.hints, SemanticHints)
        assert isinstance(prepared_source.context_latent, Latent)

    def test_prepare_source_shapes_consistent(self, prepared_source):
        T = prepared_source.latent.tensor.shape[1]
        assert prepared_source.hints.tensor.shape[1] == T
        assert prepared_source.context_latent.tensor.shape[1] == T


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

class TestTextEncoding:

    def test_returns_conditioning(self, conditioning):
        assert isinstance(conditioning, Conditioning)

    def test_conditioning_has_entries(self, conditioning):
        entries = conditioning.to_entries()
        assert len(entries) >= 1
        entry = entries[0]
        assert entry.encoder_hidden_states is not None
        assert entry.encoder_attention_mask is not None
        assert entry.encoder_hidden_states.ndim == 3
        assert entry.encoder_attention_mask.ndim == 2

    def test_different_tags_produce_different_embeddings(self, session):
        cond_a = session.encode_text(tags="heavy metal, distortion")
        cond_b = session.encode_text(tags="jazz piano, smooth")
        ea = cond_a.to_entries()[0].encoder_hidden_states
        eb = cond_b.to_entries()[0].encoder_hidden_states
        # Truncate to shorter length for comparison
        L = min(ea.shape[1], eb.shape[1])
        assert not torch.allclose(ea[:, :L], eb[:, :L])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGeneration:

    def test_returns_latent(self, generated_latent):
        assert isinstance(generated_latent, Latent)
        assert generated_latent.tensor.ndim == 3

    def test_output_shape_matches_source(self, generated_latent, prepared_source):
        assert generated_latent.tensor.shape == prepared_source.latent.tensor.shape

    def test_deterministic_with_same_seed(self, session, conditioning, prepared_source):
        kwargs = dict(
            conditioning=conditioning,
            context_latent=prepared_source.context_latent,
            source_latent=prepared_source.latent,
            seed=999,
            steps=4,
        )
        a = session.generate(**kwargs)
        b = session.generate(**kwargs)
        assert torch.allclose(a.tensor, b.tensor, atol=1e-4)

    def test_different_seeds_differ(self, session, conditioning, prepared_source):
        base = dict(
            conditioning=conditioning,
            context_latent=prepared_source.context_latent,
            source_latent=prepared_source.latent,
            steps=4,
        )
        a = session.generate(seed=1, **base)
        b = session.generate(seed=2, **base)
        assert not torch.allclose(a.tensor, b.tensor)

    def test_denoise_zero_preserves_source(self, session, conditioning, prepared_source):
        result = session.generate(
            conditioning=conditioning,
            source_latent=prepared_source.latent,
            denoise=0.0,
            seed=42,
            steps=4,
        )
        mse = (result.tensor.float() - prepared_source.latent.tensor.float()).pow(2).mean()
        assert mse.item() < 0.01

    def test_full_denoise_changes_source(self, session, conditioning, prepared_source):
        result = session.generate(
            conditioning=conditioning,
            context_latent=prepared_source.context_latent,
            denoise=1.0,
            seed=42,
            steps=4,
        )
        mse = (result.tensor.float() - prepared_source.latent.tensor.float()).pow(2).mean()
        assert mse.item() > 0.01

    def test_partial_denoise_between_extremes(self, session, conditioning, prepared_source):
        src = prepared_source.latent.tensor.float()
        full = session.generate(
            conditioning=conditioning,
            context_latent=prepared_source.context_latent,
            source_latent=prepared_source.latent,
            denoise=1.0, seed=42, steps=4,
        ).tensor.float()
        partial = session.generate(
            conditioning=conditioning,
            context_latent=prepared_source.context_latent,
            source_latent=prepared_source.latent,
            denoise=0.5, seed=42, steps=4,
        ).tensor.float()

        mse_full = (full - src).pow(2).mean().item()
        mse_partial = (partial - src).pow(2).mean().item()
        # Partial denoise should change less than full denoise
        assert mse_partial < mse_full


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class TestDecoding:

    def test_returns_audio(self, session, generated_latent):
        audio = session.decode(generated_latent)
        assert isinstance(audio, Audio)
        assert audio.sample_rate == SAMPLE_RATE

    def test_output_not_silent(self, session, generated_latent):
        audio = session.decode(generated_latent)
        rms = audio.waveform.float().pow(2).mean().sqrt().item()
        assert rms > 1e-4

    def test_output_not_clipping(self, session, generated_latent):
        audio = session.decode(generated_latent)
        peak = audio.waveform.float().abs().max().item()
        assert peak < 10.0  # not unreasonably loud

    def test_roundtrip_preserves_content(self, session, source_audio, prepared_source):
        """Encode then decode should produce audio similar to the original."""
        recon = session.decode(prepared_source.latent)
        # Trim to common length
        orig = source_audio.waveform.float().cpu()
        rec = recon.waveform.float().cpu().squeeze(0) if recon.waveform.ndim == 3 else recon.waveform.float().cpu()
        L = min(orig.shape[-1], rec.shape[-1])
        # Cosine similarity on the waveform (should be positive / correlated)
        cos = torch.nn.functional.cosine_similarity(
            orig[..., :L].reshape(1, -1),
            rec[..., :L].reshape(1, -1),
        ).item()
        assert cos > 0.3  # reasonable reconstruction


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_empty_latent_shape(self, session):
        lat = session.empty_latent(duration=30.0)
        assert isinstance(lat, Latent)
        assert lat.tensor.shape[1] == int(30.0 * FRAMES_PER_SEC)

    def test_empty_latent_different_durations(self, session):
        a = session.empty_latent(duration=10.0)
        b = session.empty_latent(duration=60.0)
        assert a.tensor.shape[1] == int(10.0 * FRAMES_PER_SEC)
        assert b.tensor.shape[1] == int(60.0 * FRAMES_PER_SEC)

    def test_blend_latents_endpoints(self, session, prepared_source):
        a = prepared_source.latent
        b = session.empty_latent(duration=a.tensor.shape[1] / FRAMES_PER_SEC)

        at_a = Session.blend_latents(a, b, alpha=0.0)
        at_b = Session.blend_latents(a, b, alpha=1.0)
        assert torch.allclose(at_a.tensor, a.tensor)
        assert torch.allclose(at_b.tensor, b.tensor)

    def test_blend_latents_midpoint(self, session, prepared_source):
        a = prepared_source.latent
        b = session.empty_latent(duration=a.tensor.shape[1] / FRAMES_PER_SEC)
        mid = Session.blend_latents(a, b, alpha=0.5)
        assert not torch.allclose(mid.tensor, a.tensor)
        assert not torch.allclose(mid.tensor, b.tensor)
