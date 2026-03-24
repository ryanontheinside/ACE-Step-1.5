"""Same as test_stream_cover.py but routes everything through Session API.

Uses Session.create_stream() / SessionStream.submit() / tick() instead
of manually constructing SlotRequests and accessing engine internals.

Measures whether the Session/node dispatch overhead adds measurable cost.
Compare timing output against test_stream_cover.py to quantify overhead.
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
from acestep.engine.session import Session
from acestep.nodes.types import Audio, Latent

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_AUDIO = PROJECT_ROOT / "tests/fixtures" / "new_order_confusion_60seconds.wav"
OUTPUT_DIR = PROJECT_ROOT / "_debug_tests" / "stream_output"
OUTPUT_FILE = OUTPUT_DIR / "stream_cover_graph_backend.wav"

SAMPLE_RATE = 48000
SEED = 1528

TRT_ENGINE = PROJECT_ROOT / "trt_engines" / "decoder_mixed_b8_s1500.engine"

# CLI flags
_args = sys.argv[1:]
vae_window = 0.0
if "--vae-window" in _args:
    _idx = _args.index("--vae-window")
    vae_window = float(_args[_idx + 1])


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
timings = {}

def timed(label, quiet=False):
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
print("Stream Cover - GRAPH BACKEND (overhead test)")
print("=" * 60)

# ------------------------------------------------------------------
# Setup -- all through Session API
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
        vae_window=vae_window,
    )

with timed("load_audio"):
    print("[Setup] Loading source audio...")
    audio = load_audio(SOURCE_AUDIO)

print("[Setup] Preparing source...")
with timed("prepare_source"):
    source = session.prepare_source(audio)
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

# ------------------------------------------------------------------
# Create stream -- all through Session API, no engine internals
# ------------------------------------------------------------------
with timed("create_stream"):
    stream = session.create_stream(
        source=source,
        conditioning=cond,
        steps=8,
        shift=3.0,
    )
print(f"  Backend: {stream.stats()['backend']}")

# ------------------------------------------------------------------
# Denoise timeline (identical to raw version)
# ------------------------------------------------------------------
num_ticks_per_cycle = 80
num_cycles = 2
total_submissions = num_ticks_per_cycle * num_cycles

denoise_per_tick = []
for i in range(total_submissions):
    t = i / total_submissions
    dn = 0.25 + 0.375 * (1.0 - np.cos(2 * np.pi * num_cycles * t))
    denoise_per_tick.append(round(dn, 3))

total_ticks = total_submissions + 8 + 8

print(f"\n[Timeline] {total_submissions} submissions, ~{total_ticks} ticks")
print(f"  Denoise range: {min(denoise_per_tick):.3f} - {max(denoise_per_tick):.3f}")

# ------------------------------------------------------------------
# Run pipeline -- submit/tick through SessionStream
# ------------------------------------------------------------------
submit_idx = 0
num_completed = 0

slice_duration = 0.3
slice_samples = int(slice_duration * SAMPLE_RATE)
playback_start = 5.0
playback_offset_samples = int(playback_start * SAMPLE_RATE)
output_chunks = []
prev_dn = None
last_latent = None
last_wav = None
skip_threshold = 1e-3
num_skipped = 0
mse_values = []
last_win_start_sample = 0

print(f"\n[Run] Starting pipeline (graph backend, {slice_duration}s slices, skip_threshold={skip_threshold})...")
run_start = time.time()

for tick_num in range(total_ticks):
    # Submit through SessionStream (no SlotRequest, no raw tensors)
    if submit_idx < len(denoise_per_tick):
        dn = denoise_per_tick[submit_idx]
        stream.submit(denoise=dn, seed=SEED)
        submit_idx += 1

    torch.cuda.synchronize()
    iter_t0 = time.perf_counter()

    result_latent = stream.tick()

    if result_latent is not None:
        result = result_latent.tensor
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - iter_t0) * 1000
        timings.setdefault("tick", []).append(tick_ms)

        submit_tick = tick_num - stream.config.infer_steps
        if 0 <= submit_tick < len(denoise_per_tick):
            dn_submitted = denoise_per_tick[submit_tick]
        else:
            dn_submitted = -1.0

        start = playback_offset_samples + num_completed * slice_samples
        end = start + slice_samples

        skipped = False
        if last_latent is not None:
            mse = (result - last_latent).pow(2).mean().item()
            mse_values.append(mse)
            if mse < skip_threshold and last_wav is not None:
                local_start = start - last_win_start_sample
                local_end = local_start + slice_samples
                if 0 <= local_start and local_end <= last_wav.shape[1]:
                    wav = last_wav
                    skipped = True
                    num_skipped += 1

        last_latent = result.clone()

        if not skipped:
            dec_t0 = time.perf_counter()
            if vae_window > 0:
                t_start = start / SAMPLE_RATE
                audio_out = session.decode(result_latent, t_start=t_start)
                wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                win_start_sample = audio_out.start_sample
            else:
                audio_out = session.decode(result_latent)
                wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                win_start_sample = 0
            torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - dec_t0) * 1000
            timings.setdefault("vae_decode", []).append(dec_ms)
            last_wav = wav
            last_win_start_sample = win_start_sample
            local_start = start - win_start_sample
            local_end = local_start + slice_samples

        if local_end <= wav.shape[1]:
            chunk = wav[:, local_start:local_end]
        else:
            chunk = torch.zeros(wav.shape[0], slice_samples)
            available = wav.shape[1] - local_start
            if available > 0:
                chunk[:, :available] = wav[:, local_start:local_start+available]

        output_chunks.append(chunk)
        num_completed += 1

        dec_str = "SKIP" if skipped else f"{dec_ms:5.1f}ms"
        mse_str = f"mse={mse:.2e}" if last_latent is not None and num_completed > 1 else ""
        if dn_submitted != prev_dn or num_completed % 20 == 0:
            print(f"  #{num_completed:3d} dn={dn_submitted:.2f}  "
                  f"tick={tick_ms:5.1f}ms  decode={dec_str:>7s}  {mse_str}  "
                  f"(playback {start/SAMPLE_RATE:.1f}s-{end/SAMPLE_RATE:.1f}s)")
            prev_dn = dn_submitted
    else:
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - iter_t0) * 1000
        timings.setdefault("tick", []).append(tick_ms)

    if stream.active_slots == 0 and submit_idx >= len(denoise_per_tick):
        break

run_ms = (time.time() - run_start) * 1000
num_decoded = num_completed - num_skipped
print(f"\n[Run] {num_completed} generations in {run_ms:.0f}ms "
      f"({run_ms/max(num_completed,1):.1f}ms avg incl. decode)")
print(f"  Decoded: {num_decoded}, Skipped: {num_skipped} "
      f"({100*num_skipped/max(num_completed,1):.0f}% skip rate)")
if mse_values:
    sorted_mse = sorted(mse_values)
    print(f"  MSE: min={sorted_mse[0]:.2e}  median={sorted_mse[len(sorted_mse)//2]:.2e}  "
          f"max={sorted_mse[-1]:.2e}")

# Concatenate all chunks
output_wav = torch.cat(output_chunks, dim=1)
total_duration = output_wav.shape[1] / SAMPLE_RATE

print(f"\n[Save] Output: {total_duration:.1f}s, {output_wav.shape}")
sf.write(str(OUTPUT_FILE), output_wav.numpy().T, SAMPLE_RATE, format="WAV")
print(f"  Saved: {OUTPUT_FILE}")

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
print("TIMING SUMMARY (graph backend)")
print(f"{'=' * 60}")
for label in ["model_load", "load_audio", "prepare_source",
               "text_encode", "create_stream",
               "tick", "vae_decode"]:
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

tick_vals = timings.get("tick", [])
decode_vals = timings.get("vae_decode", [])
if tick_vals:
    tick_total = sum(tick_vals)
    decode_total = sum(decode_vals) if decode_vals else 0
    avg_tick = tick_total / num_completed
    avg_decode_amortized = decode_total / num_completed
    print(f"\n  Per-generation (amortized over {num_completed} gens):")
    print(f"    tick={avg_tick:.1f}ms + decode={avg_decode_amortized:.1f}ms "
          f"({num_decoded} decoded, {num_skipped} skipped) "
          f"= {avg_tick + avg_decode_amortized:.1f}ms")

print(f"\n[Summary]")
print(f"  Compare timing with test_stream_cover.py to measure graph overhead.")
print(f"  Compare WAV output to verify identical results.")

print("\n" + "=" * 60)
