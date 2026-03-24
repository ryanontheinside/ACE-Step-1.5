"""Tests for SessionStream (interactive streaming API).

Validates the streaming contract: create_stream, submit, tick, feedback,
SDE curves, and shift control. Each test gets a fresh stream to avoid
cross-test state leakage.

Run:  uv run pytest tests/test_stream.py -v
"""

import pytest
import torch

from acestep.engine.session import SessionStream
from acestep.nodes.types import Latent

SAMPLE_RATE = 48000
FRAMES_PER_SEC = 25
MAX_TICKS = 200  # safety limit to avoid infinite loops


def drain(stream, count=1):
    """Submit and tick until `count` results are collected."""
    results = []
    for _ in range(MAX_TICKS):
        r = stream.tick()
        if r is not None:
            results.append(r)
            if len(results) >= count:
                break
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stream(session, prepared_source, conditioning):
    """Fresh stream per test."""
    return session.create_stream(
        source=prepared_source,
        conditioning=conditioning,
        steps=8,
        shift=3.0,
    )


# ---------------------------------------------------------------------------
# Creation and properties
# ---------------------------------------------------------------------------

class TestStreamCreation:

    def test_returns_session_stream(self, stream):
        assert isinstance(stream, SessionStream)

    def test_stats_has_backend(self, stream):
        s = stream.stats()
        assert isinstance(s, dict)
        assert "backend" in s

    def test_source_latents_shape(self, stream, prepared_source):
        sl = stream.source_latents
        assert isinstance(sl, torch.Tensor)
        T = prepared_source.latent.tensor.shape[1]
        assert sl.shape[1] == T
        assert sl.shape[2] == 64

    def test_source_latents_on_device(self, stream):
        assert stream.source_latents.is_cuda


# ---------------------------------------------------------------------------
# Submit + tick lifecycle
# ---------------------------------------------------------------------------

class TestStreamGeneration:

    def test_submit_and_tick_produces_latent(self, stream):
        stream.submit(denoise=0.5, seed=42)
        results = drain(stream)
        assert len(results) == 1
        assert isinstance(results[0], Latent)
        assert results[0].tensor.ndim == 3

    def test_output_shape_matches_source(self, stream, prepared_source):
        stream.submit(denoise=0.5, seed=42)
        results = drain(stream)
        assert results[0].tensor.shape == prepared_source.latent.tensor.shape

    def test_deterministic_seed(self, session, prepared_source, conditioning):
        outputs = []
        for _ in range(2):
            s = session.create_stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=3.0,
            )
            s.submit(denoise=0.5, seed=42)
            results = drain(s)
            outputs.append(results[0].tensor.clone())
        assert torch.allclose(outputs[0], outputs[1], atol=1e-4)

    def test_different_seeds_differ(self, stream):
        stream.submit(denoise=0.5, seed=1)
        stream.submit(denoise=0.5, seed=2)
        results = drain(stream, count=2)
        assert len(results) == 2
        assert not torch.allclose(results[0].tensor, results[1].tensor)

    def test_multiple_submissions_all_complete(self, stream):
        n = 4
        for i in range(n):
            stream.submit(denoise=0.5, seed=100 + i)
        results = drain(stream, count=n)
        assert len(results) == n


# ---------------------------------------------------------------------------
# Denoise control
# ---------------------------------------------------------------------------

class TestStreamDenoise:

    def test_low_denoise_close_to_source(self, stream, prepared_source):
        stream.submit(denoise=0.1, seed=42)
        results = drain(stream)
        src = prepared_source.latent.tensor.float()
        out = results[0].tensor.float()
        mse = (out - src.to(out.device)).pow(2).mean().item()
        assert mse < 0.5

    def test_full_denoise_far_from_source(self, stream, prepared_source):
        stream.submit(denoise=1.0, seed=42)
        results = drain(stream)
        src = prepared_source.latent.tensor.float()
        out = results[0].tensor.float()
        mse = (out - src.to(out.device)).pow(2).mean().item()
        assert mse > 0.01


# ---------------------------------------------------------------------------
# SDE denoise curves
# ---------------------------------------------------------------------------

class TestStreamSDE:

    def test_flat_curve_produces_result(self, stream):
        T = stream.source_latents.shape[1]
        curve = torch.full((1, T, 1), 0.5, dtype=torch.float32)
        stream.submit(denoise=1.0, seed=42, sde_denoise_curve=curve)
        results = drain(stream)
        assert len(results) == 1

    def test_sine_curve_produces_result(self, stream):
        T = stream.source_latents.shape[1]
        t = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1)
        curve = 0.5 * (0.5 + 0.5 * torch.sin(2 * 3.14159 * 4 * t))
        stream.submit(denoise=1.0, seed=42, sde_denoise_curve=curve)
        results = drain(stream)
        assert len(results) == 1

    def test_different_curves_produce_different_output(
        self, session, prepared_source, conditioning
    ):
        T = prepared_source.latent.tensor.shape[1]
        outputs = []
        for amp in [0.3, 0.9]:
            s = session.create_stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=3.0,
            )
            curve = torch.full((1, T, 1), amp, dtype=torch.float32)
            s.submit(denoise=1.0, seed=42, sde_denoise_curve=curve)
            results = drain(s)
            outputs.append(results[0].tensor.clone())
        assert not torch.allclose(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# Source latent override (feedback pattern)
# ---------------------------------------------------------------------------

class TestStreamFeedback:

    def test_custom_source_latents_accepted(self, stream):
        sl = stream.source_latents
        # Perturb the source latents
        noisy = sl + torch.randn_like(sl) * 0.1
        stream.submit(denoise=0.5, seed=42, source_latents=noisy)
        results = drain(stream)
        assert len(results) == 1

    def test_feedback_changes_output(self, session, prepared_source, conditioning):
        outputs = []
        for feedback_val in [0.0, 0.5]:
            s = session.create_stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=3.0,
            )
            # First generation
            s.submit(denoise=0.5, seed=42)
            first = drain(s)[0].tensor.clone()
            # Second generation with/without feedback
            if feedback_val > 0:
                blended = (1.0 - feedback_val) * s.source_latents + feedback_val * first
                s.submit(denoise=0.5, seed=42, source_latents=blended)
            else:
                s.submit(denoise=0.5, seed=42)
            second = drain(s)[0].tensor.clone()
            outputs.append(second)
        # Feedback should produce a different result
        assert not torch.allclose(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# Shift control
# ---------------------------------------------------------------------------

class TestStreamShift:

    def test_set_shift_updates_config(self, stream):
        stream.set_shift(5.0)
        assert abs(stream.config.shift - 5.0) < 1e-6

    def test_different_shift_changes_output(
        self, session, prepared_source, conditioning
    ):
        outputs = []
        for shift in [1.5, 5.0]:
            s = session.create_stream(
                source=prepared_source,
                conditioning=conditioning,
                steps=8, shift=shift,
            )
            s.submit(denoise=0.5, seed=42)
            results = drain(s)
            outputs.append(results[0].tensor.clone())
        assert not torch.allclose(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# Decode integration
# ---------------------------------------------------------------------------

class TestStreamDecode:

    def test_stream_output_decodable(self, session, stream):
        stream.submit(denoise=0.5, seed=42)
        results = drain(stream)
        audio = session.decode(results[0])
        assert audio.sample_rate == SAMPLE_RATE
        rms = audio.waveform.float().pow(2).mean().sqrt().item()
        assert rms > 1e-4
