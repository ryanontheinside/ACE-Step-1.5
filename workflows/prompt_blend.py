#!/usr/bin/env python3
"""Multi-prompt temporal blending workflow.

Replaces test_temporal_blend.py. Demonstrates:
  - Two TextEncode calls with different prompts
  - CurveWave -> temporal_weight for per-frame blending
  - ConditioningCombine to produce multi-condition set
  - Generate runs separate decoder calls, blends velocities per-frame
"""

import os
import sys
import time

import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.nodes import Audio, Mask
from acestep.nodes.model_nodes import LoadModel
from acestep.nodes.vae_nodes import VAEEncodeAudio, VAEDecodeAudio, EmptyLatent
from acestep.nodes.cond_nodes import TextEncode, ConditioningCombine
from acestep.nodes.curve_nodes import CurveWave
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate

SOURCE_AUDIO = os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output", "workflows")


def load_audio(path: str, duration: float = 60.0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2, :int(duration * 48000)]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, path: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    sf.write(path, wav.cpu().numpy().T, audio.sample_rate)
    print(f"Saved: {path}")


def main():
    print("=" * 70)
    print("WORKFLOW: Multi-Prompt Temporal Blend")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    handles = LoadModel().execute(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
    )
    model, clip, vae = handles["model"], handles["clip"], handles["vae"]

    # --- Create empty latent (pure generation, no source) ---
    empty = EmptyLatent().execute(model=model, duration=60.0)["latent"]
    T = empty.tensor.shape[1]

    # --- Encode two different prompts ---
    print("\n[TextEncode] Prompt A: daft punk style")
    cond_a = TextEncode().execute(
        clip=clip, model=model,
        tags="daft punk style electronic french house",
        lyrics="",
        task="generate",
        bpm=120, duration=60.0, key="E minor",
    )["conditioning"]

    print("[TextEncode] Prompt B: heavy demon techno")
    cond_b = TextEncode().execute(
        clip=clip, model=model,
        tags="heavy demon techno, growling bass, industrial",
        lyrics="",
        task="generate",
        bpm=120, duration=60.0, key="E minor",
    )["conditioning"]

    # --- Create temporal blend curve ---
    # Square wave: alternates between prompt A and prompt B every ~6 seconds
    blend_curve = CurveWave().execute(
        wave_type="square",
        frames_per_cycle=151,
        amplitude=0.5,
        offset=0.5,
        length=T,
    )["curve"]

    # Convert curve to mask (clamp to [0,1]) for temporal_weight
    temporal_mask = Mask(tensor=blend_curve.tensor.clamp(0.0, 1.0))

    # --- Combine conditions ---
    combined = ConditioningCombine().execute(
        conditioning_a=cond_a,
        conditioning_b=cond_b,
        temporal_weight_b=temporal_mask,
    )["conditioning"]

    print(f"Combined: {len(combined.to_entries())} entries")

    # --- Generate ---
    config = DiffusionConfigNode().execute(
        steps=8, shift=3.0, seed=1118, denoise=1.0,
    )["config"]

    t0 = time.time()
    output_latent = Generate().execute(
        model=model,
        config=config,
        positive=combined,
    )["latent"]
    print(f"Generated in {time.time() - t0:.2f}s")

    # --- Decode ---
    output_audio = VAEDecodeAudio().execute(vae=vae, latent=output_latent)["audio"]
    save_audio(output_audio, os.path.join(OUTPUT_DIR, "prompt_blend.wav"))

    print("\nDone.")


if __name__ == "__main__":
    main()
