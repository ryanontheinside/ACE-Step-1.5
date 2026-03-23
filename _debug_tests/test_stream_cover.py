"""Stream pipeline cover workflow: user turns the denoise knob in real time.

Loads source audio, encodes cover conditioning, then runs the stream
pipeline while sweeping the denoise value. Produces a single output WAV
that simulates what the user would hear in a real-time interactive session.

Each pipeline tick (~100ms wall clock) produces a finished 60s generation.
The output file splices consecutive generations at advancing playback
positions, so you hear the song progressing while the denoise character
shifts in real time.

Init pattern matches test_polygraphy_full.py (the known-working path):
  - torch.set_grad_enabled(False) + dynamo disabled
  - Session with compile_model=False, no flash attention
  - Decoder offloaded to CPU before TRT engine init
  - Text encoder offloaded to CPU after conditioning encoded
"""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import numpy as np
import soundfile as sf

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Audio, Latent

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_AUDIO = PROJECT_ROOT / "test_audio" / "new_order_confusion_60seconds.wav"
OUTPUT_DIR = PROJECT_ROOT / "_debug_tests" / "stream_output"
OUTPUT_FILE = OUTPUT_DIR / "stream_cover_denoise_sweep.wav"

SAMPLE_RATE = 48000
SEED = 1528  # fixed seed so output is stable when params don't change

TRT_ENGINE = PROJECT_ROOT / "trt_engines" / "decoder_mixed_b8_s1500.engine"


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
timings = {}  # label -> list of ms

def timed(label, quiet=False):
    """Context manager that records wall-clock time (with cuda sync)."""
    class _Timer:
        def __enter__(self_):
            torch.cuda.synchronize()
            self_.t0 = time.perf_counter()
            return self_
        def __exit__(self_, *exc):
            torch.cuda.synchronize()
            self_.ms = (time.perf_counter() - self_.t0) * 1000
            timings.setdefault(label, []).append(self_.ms)
            if not quiet:
                print(f"  [{label}] {self_.ms:.1f}ms")
    return _Timer()


def load_audio(path, duration=60.0):
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(duration * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Stream Cover - Denoise Knob Simulation (TIMED)")
print("=" * 60)

# ------------------------------------------------------------------
# Setup -- matches test_polygraphy_full.py init pattern
# ------------------------------------------------------------------
VAE_ENCODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_encode_fp16_max6000.engine"
VAE_DECODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_decode_fp16_max6000.engine"

with timed("model_load"):
    print("\n[Setup] Loading model (trt_engines, no compile)...")
    session = Session(
        project_root=str(PROJECT_ROOT / "checkpoints"),
        compile_model=False,
        trt_engines={
            "decoder": str(TRT_ENGINE),
            "vae_encode": str(VAE_ENCODE_ENGINE),
            "vae_decode": str(VAE_DECODE_ENGINE),
        },
    )

handler = session.handler
device = handler.device
dtype = handler.dtype

with timed("load_audio"):
    print("[Setup] Loading source audio...")
    audio = load_audio(SOURCE_AUDIO)

print("[Setup] Preparing source (VAE encode + semantic extract)...")
with timed("vae_encode"):
    latent = session.encode_audio(audio)
with timed("semantic_extract"):
    hints = session.extract_hints(latent)
with timed("hints_to_latent"):
    context_latent = session.hints_to_latent(hints)
source = PreparedSource(latent=latent, hints=hints, context_latent=context_latent)
T = source.latent.tensor.shape[1]
print(f"  Source: T={T} frames ({T/25:.1f}s)")

with timed("text_encode"):
    print("[Setup] Encoding cover conditioning...")
    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136,
        duration=60.0,
        key="G# minor",
    )
entry = cond.to_entries()[0]

# Build context latents from source
ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
D = ctx_lat.shape[2]
cm = torch.ones(1, T, D, device=device, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)

source_latents = source.latent.tensor.to(device=device, dtype=dtype)

# ------------------------------------------------------------------
# Denoise timeline: smooth sine-like sweep, two full cycles
# ------------------------------------------------------------------
num_ticks_per_cycle = 80
num_cycles = 2
total_submissions = num_ticks_per_cycle * num_cycles

denoise_per_tick = []
for i in range(total_submissions):
    t = i / total_submissions
    dn = 0.25 + 0.375 * (1.0 - np.cos(2 * np.pi * num_cycles * t))
    denoise_per_tick.append(round(dn, 3))

total_ticks = total_submissions + 8 + 8  # + warmup + drain

print(f"\n[Timeline] {total_submissions} submissions, ~{total_ticks} ticks")
print(f"  Denoise range: {min(denoise_per_tick):.3f} - {max(denoise_per_tick):.3f}")
print(f"  Two full cycles (0.25 -> 1.0 -> 0.25 -> 1.0 -> 0.25)")

# ------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------
engine = handler._diffusion_engine
config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)
pipe = StreamPipeline(engine, config)
print(f"  Backend: {pipe.stats()['backend']}, io_dtype: {pipe._trt_io_dtype}")

submit_idx = 0
num_completed = 0

# Decode inline: each tick that produces a result gets decoded
# immediately and a slice is extracted. No accumulation.
slice_duration = 0.3  # seconds per completion
slice_samples = int(slice_duration * SAMPLE_RATE)
playback_start = 5.0  # seconds into the song
playback_offset_samples = int(playback_start * SAMPLE_RATE)
output_chunks = []
prev_dn = None

print(f"\n[Run] Starting pipeline (decode inline, {slice_duration}s slices)...")
run_start = time.time()

for tick_num in range(total_ticks):
    # Submit next request with current denoise value
    if submit_idx < len(denoise_per_tick):
        dn = denoise_per_tick[submit_idx]
        with timed("submit", quiet=True):
            pipe.submit(SlotRequest(
                encoder_hidden_states=entry.encoder_hidden_states,
                encoder_attention_mask=entry.encoder_attention_mask,
                context_latents=context_latents,
                seed=SEED,
                source_latents=source_latents,
                denoise=dn,
            ))
        submit_idx += 1

    with timed("tick", quiet=True) as tick_t:
        result = pipe.tick()

    if result is not None:
        # Figure out which denoise this result was submitted with
        submit_tick = tick_num - config.infer_steps
        if 0 <= submit_tick < len(denoise_per_tick):
            dn_submitted = denoise_per_tick[submit_tick]
        else:
            dn_submitted = -1.0

        # Decode immediately and extract slice
        with timed("vae_decode", quiet=True) as dec_t:
            audio_out = session.decode(Latent(tensor=result))
        wav = audio_out.waveform.detach().cpu().float().squeeze(0)

        start = playback_offset_samples + num_completed * slice_samples
        end = start + slice_samples

        if end <= wav.shape[1]:
            chunk = wav[:, start:end]
        else:
            chunk = torch.zeros(wav.shape[0], slice_samples)
            available = wav.shape[1] - start
            if available > 0:
                chunk[:, :available] = wav[:, start:start+available]

        output_chunks.append(chunk)
        num_completed += 1

        if dn_submitted != prev_dn or num_completed % 20 == 0:
            print(f"  #{num_completed:3d} dn={dn_submitted:.2f}  "
                  f"tick={tick_t.ms:5.1f}ms  decode={dec_t.ms:5.1f}ms  "
                  f"(playback {start/SAMPLE_RATE:.1f}s-{end/SAMPLE_RATE:.1f}s)")
            prev_dn = dn_submitted

    if pipe.active_slots == 0 and submit_idx >= len(denoise_per_tick):
        break

run_ms = (time.time() - run_start) * 1000
print(f"\n[Run] {num_completed} generations in {run_ms:.0f}ms "
      f"({run_ms/max(num_completed,1):.1f}ms avg incl. decode)")

# Concatenate all chunks
output_wav = torch.cat(output_chunks, dim=1)
total_duration = output_wav.shape[1] / SAMPLE_RATE

print(f"\n[Save] Output: {total_duration:.1f}s, {output_wav.shape}")
sf.write(str(OUTPUT_FILE), output_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  Saved: {OUTPUT_FILE}")

# Also save the source for A/B comparison
source_out = OUTPUT_DIR / "source_reference.wav"
src_wav = audio.waveform
if src_wav.dim() == 3:
    src_wav = src_wav.squeeze(0)
sf.write(str(source_out), src_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  Source: {source_out}")

# ------------------------------------------------------------------
# Timing summary
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("TIMING SUMMARY")
print(f"{'=' * 60}")
for label in ["model_load", "load_audio", "vae_encode",
               "semantic_extract", "hints_to_latent", "text_encode",
               "submit", "tick", "vae_decode"]:
    vals = timings.get(label, [])
    if not vals:
        continue
    total = sum(vals)
    avg = total / len(vals)
    mn, mx = min(vals), max(vals)
    if len(vals) == 1:
        print(f"  {label:22s}  {total:8.1f}ms  (1 call)")
    else:
        print(f"  {label:22s}  {total:8.1f}ms total  "
              f"avg={avg:6.1f}ms  min={mn:6.1f}ms  max={mx:6.1f}ms  "
              f"({len(vals)} calls)")

# Per-generation cost breakdown (tick + decode for ticks that produced output)
tick_vals = timings.get("tick", [])
decode_vals = timings.get("vae_decode", [])
if tick_vals and decode_vals:
    avg_tick = sum(tick_vals) / len(tick_vals)
    avg_decode = sum(decode_vals) / len(decode_vals)
    print(f"\n  Per-generation avg:  tick={avg_tick:.1f}ms + decode={avg_decode:.1f}ms "
          f"= {avg_tick + avg_decode:.1f}ms")

print(f"\n[Summary]")
print(f"  Listen to {OUTPUT_FILE.name} to hear the denoise knob sweep.")
print(f"  Playback starts at {playback_start}s into the song.")
print(f"  Each {slice_duration}s slice is from a different generation.")
print(f"  Two smooth sine-wave cycles: 0.25 -> 1.0 -> 0.25 -> 1.0 -> 0.25")

print("\n" + "=" * 60)
