# Windowed VAE Decode: Benchmarks & Analysis

Date: 2026-03-23

## Context

The streaming pipeline (StreamPipeline) decodes a full 60s latent through the VAE TRT engine every tick. At ~107ms per decode, this is the single largest cost alongside the DiT forward pass (~73ms). Windowed decoding could cut this significantly by only decoding a portion of the latent each tick.

## VAE Decode Scaling (vae_decode_fp16_max6000.engine)

Engine profile: min=125 frames (5s), opt=1500 (60s), max=6000 (240s).

| Duration | Frames | Decode time | ms per second |
|----------|--------|-------------|---------------|
| 15s      | 375    | 30ms        | 2.0           |
| 30s      | 750    | 56ms        | 1.9           |
| 60s      | 1500   | 107ms       | 1.8           |
| 240s     | 6000   | 417ms       | 1.7           |

Scaling is almost perfectly linear with duration. No super-linear penalty at large sizes. The speedup from windowed decoding comes purely from doing less work.

## Chunking Quality: Receptive Field Measurement

Decoded the same 60s latent as 1x60s (reference) vs 4x15s chunks, compared waveforms sample-by-sample.

**Key findings:**
- Interior of each chunk is **bit-perfect** (0.000 error at center of chunk 0)
- Error is concentrated near chunk boundaries, falls off rapidly
- Max absolute error near boundary: ~0.73 (large, audibly significant)
- Error profile is consistent across all 3 boundaries:

| Distance from boundary | Max absolute error | Relative error |
|------------------------|-------------------|----------------|
| 20ms                   | 0.40 - 0.73       | 0.81 - 1.22    |
| 60ms                   | 0.16 - 0.51       | 0.41 - 0.73    |
| 100ms                  | 0.08 - 0.20       | 0.16 - 0.29    |
| 200ms                  | 0.02 - 0.05       | 0.03 - 0.08    |
| 300ms                  | 0.003 - 0.010     | 0.005 - 0.015  |
| 340ms                  | 0.001 - 0.002     | 0.002 - 0.003  |

- **Convergence threshold (error < 1e-5):** 333ms = 8.3 latent frames
- **Chosen safe overlap:** 500ms = 12.5 latent frames per side (rounded to 13)

## Windowed Decode Design

With 500ms overlap per side, each window decodes slightly more than its keep region:

| Keep window | Total decode (with 2x500ms overlap) | Frames | Estimated time | Speedup vs 60s |
|-------------|--------------------------------------|--------|----------------|----------------|
| 15s         | 16s                                  | 400    | ~32ms          | 3.3x           |
| 30s         | 31s                                  | 775    | ~58ms          | 1.8x           |

The overlap regions are crossfaded between adjacent windows to eliminate any boundary artifacts. The interior of each window is bit-identical to full decode.

## Potential Further Optimization: Tight-Profile Engine

The current engine has a wide dynamic range (125-6000 frames). A dedicated engine built with an exact profile (e.g., min=opt=max=400 for 16s windows) could allow TRT to select more aggressive kernel tactics since it doesn't need to handle variable shapes. This is an untested hypothesis; the gain (if any) would be on top of the linear scaling savings above.

## Tick Budget Impact

Current streaming tick (60s audio, 8-step DiT):
- DiT TRT execute: ~73ms
- VAE decode: ~107ms
- Overhead: ~5ms
- **Total: ~185ms**

With 15s windowed decode (500ms overlap):
- DiT TRT execute: ~73ms
- VAE decode: ~32ms
- Overhead: ~5ms
- **Total: ~110ms** (40% faster)

## Scripts

- `_debug_tests/bench_vae_decode_sizes.py` - Duration scaling benchmark
- `_debug_tests/bench_vae_chunk_quality.py` - Chunking quality / receptive field measurement
