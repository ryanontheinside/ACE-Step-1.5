"""Audio analysis nodes."""

from __future__ import annotations

import torch
import torchaudio
from typing import Any, ClassVar

from .base import BaseNode, NodeDefinition, NodePort, NodeRegistry
from .types import Audio


# Chroma-to-key mapping (Krumhansl-Schmuckler profiles)
_MAJOR_PROFILE = torch.tensor(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = torch.tensor(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _detect_key(waveform: torch.Tensor, sr: int) -> str:
    """Detect musical key using chroma features and Krumhansl-Schmuckler."""
    # Mono
    if waveform.dim() == 2:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform
    mono = mono.float()

    # Compute spectrogram
    n_fft = 4096
    hop = 2048
    spec = torch.stft(
        mono, n_fft=n_fft, hop_length=hop, return_complex=True,
        window=torch.hann_window(n_fft, device=mono.device),
    )
    power = spec.abs().pow(2)

    # Build chroma from power spectrum
    freqs = torch.linspace(0, sr / 2, power.shape[0])
    chroma = torch.zeros(12)
    for i in range(12):
        # Accumulate energy for this pitch class across octaves
        for octave in range(1, 9):
            f0 = 440.0 * (2.0 ** ((i - 9) / 12.0 + (octave - 4)))
            if f0 >= sr / 2:
                break
            bin_idx = int(f0 * n_fft / sr)
            if 0 <= bin_idx < power.shape[0]:
                lo = max(0, bin_idx - 1)
                hi = min(power.shape[0], bin_idx + 2)
                chroma[i] += power[lo:hi].sum().item()

    if chroma.sum() < 1e-10:
        return "C major"

    chroma = chroma / chroma.sum()

    # Correlate with all 24 key profiles
    best_corr = -2.0
    best_key = "C major"
    for shift in range(12):
        rotated = torch.roll(chroma, -shift)
        for profile, mode in [(_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")]:
            norm_p = profile / profile.sum()
            corr = torch.dot(rotated, norm_p).item()
            if corr > best_corr:
                best_corr = corr
                best_key = f"{_NOTE_NAMES[shift]} {mode}"

    return best_key


def _detect_bpm(waveform: torch.Tensor, sr: int) -> int:
    """Detect BPM using onset envelope autocorrelation."""
    # Mono, resample to 22050 for speed
    if waveform.dim() == 2:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform
    mono = mono.float()

    if sr != 22050:
        mono = torchaudio.functional.resample(mono, sr, 22050)
        sr = 22050

    # Compute onset envelope via spectral flux
    n_fft = 2048
    hop = 512
    spec = torch.stft(
        mono, n_fft=n_fft, hop_length=hop, return_complex=True,
        window=torch.hann_window(n_fft, device=mono.device),
    )
    mag = spec.abs()
    # Half-wave rectified spectral flux
    flux = torch.clamp(mag[:, 1:] - mag[:, :-1], min=0).sum(dim=0)

    # Autocorrelation
    # BPM range: 60-200 -> lag range in onset frames
    min_bpm, max_bpm = 60, 200
    fps = sr / hop
    min_lag = int(fps * 60.0 / max_bpm)
    max_lag = int(fps * 60.0 / min_bpm)
    max_lag = min(max_lag, len(flux) // 2)

    if max_lag <= min_lag:
        return 120

    flux = flux - flux.mean()
    autocorr = torch.zeros(max_lag - min_lag)
    for i, lag in enumerate(range(min_lag, max_lag)):
        autocorr[i] = torch.dot(flux[:len(flux) - lag], flux[lag:]).item()

    if autocorr.max() <= 0:
        return 120

    best_lag = autocorr.argmax().item() + min_lag
    bpm = 60.0 * fps / best_lag
    return int(round(bpm))


@NodeRegistry.register
class AudioInfo(BaseNode):
    """Detect BPM, key, and duration from audio.

    Uses signal processing (onset autocorrelation for BPM,
    chroma + Krumhansl-Schmuckler for key detection).
    Duration is computed from sample count and rate.
    """

    node_type_id: ClassVar[str] = "acestep.AudioInfo"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Audio Info",
            category="audio",
            description="Detect BPM, key, and duration from audio.",
            inputs=(
                NodePort(name="audio", type="AUDIO"),
            ),
            outputs=(),
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        audio: Audio = kwargs["audio"]
        waveform = audio.waveform
        sr = audio.sample_rate

        # Handle [B, C, samples] or [C, samples]
        if waveform.dim() == 3:
            waveform = waveform[0]

        samples = waveform.shape[-1]
        duration = samples / sr

        bpm = _detect_bpm(waveform, sr)
        key = _detect_key(waveform, sr)

        return {
            "bpm": bpm,
            "key": key,
            "duration": round(duration, 2),
        }
