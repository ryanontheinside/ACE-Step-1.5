# Working title candidates

**INSERT**
- Interactive Noise-Sequencer Engine for Real-Time
- Interactive Streaming Engine for Real-Time
- Interactive Noise-Sharing Engine for Real-Time

**DRIFT**
- Diffusion Real-time Interactive Flow Transform

**MIXER**
- Modulated Interactive X-attention Engine for Realtime
- Multi-condition Interactive X-frame Engine for Realtime
- Modulated Interactive eXecution Engine for Realtime

---

Real-time composable diffusion engine for interactive music generation, built on [ACE-Step](https://github.com/ace-step/ACE-Step).

## Live demo: MIDI-controlled generation

The realtime demo turns a MIDI controller into a diffusion synthesizer. Feed it source audio and a text prompt; twist physical knobs to control denoise strength, SDE curve shape, latent feedback, and diffusion shift while the engine generates and plays back audio continuously at ~110ms per tick.

```bash
# MIDI mode with SDE curves and windowed VAE decode (fastest)
uv run python demos/realtime_motion_graph.py --midi --sde --vae-window 15 path/to/source.wav

# Webcam motion mode (no MIDI controller needed)
uv run python demos/realtime_motion_graph.py --vae-window 15 path/to/source.wav
```

MIDI knob layout (CC#70-74 on any controller, tested with Akai MPK Mini):

| Knob | Regular mode | SDE mode (`--sde`) |
|---|---|---|
| K1 | Denoise strength | SDE curve amplitude |
| K2 | Seed | Seed |
| K3 | Latent feedback | Latent feedback |
| K4 | Diffusion shift | Diffusion shift |
| K5 | -- | SDE periodicity |

Requires `pygame`, `sounddevice`, and optionally `opencv-python` for webcam mode.

## Offline benchmark: stream pipeline stress test

Runs the streaming pipeline end-to-end without interactive I/O, sweeping denoise over many ticks and splicing the output into a single WAV. Use this to validate performance and listen to what the stream pipeline produces across a range of denoise values.

```bash
uv run python demos/test_stream_cover.py
```

Each tick produces a finished 60-second generation. The output file splices consecutive generations at advancing playback positions so you hear the song progress while the denoise character shifts.

## What it does

- Composable multi-condition diffusion with per-frame modulation curves (velocity scaling, SDE denoise, guidance, noise injection, x0 target blending)
- Automatic execution path selection (fast/switch/batched/sequential) based on active conditions per step
- StreamDiffusion-style ring buffer pipeline adapted for audio, with per-slot denoise, source latents, and SDE curves
- TensorRT acceleration for both the DiT decoder and VAE
- Fused Triton kernels for Euler/SDE integration
- Windowed VAE decode with empirically-sized overlap for streaming
- Typed node graph system (40+ nodes) for composable generation workflows
- Real-time interactive control via MIDI CC or webcam motion

## Requirements

- Python 3.11
- CUDA GPU (tested on RTX 5090, works on 8GB+ VRAM)
- ACE-Step v1.5 checkpoints in `checkpoints/` (auto-downloaded on first run)

## Setup

```bash
uv sync
uv run python tests/fixtures/download.py
```

The second command downloads test audio fixtures (~44MB) used by all demos and tests.

## Quick start

The Session API is the simplest path to generating audio programmatically:

```bash
uv run python workflows/session_demo.py
```

Loads the model once, then generates covers in ~310ms per iteration after warmup.

## Demos

All demos expect source audio in `test_audio/` and write output to `test_output/`.

| Script | What it does |
|---|---|
| `demos/realtime_motion_graph.py` | MIDI/webcam-driven real-time generation with audio playback |
| `demos/test_stream_cover.py` | StreamPipeline stress test with denoise sweep |
| `demos/test_noise_sharing.py` | Noise sharing for temporal continuity between generations |
| `workflows/session_demo.py` | Session API basics: load once, generate many |
| `workflows/realtime_cover.py` | Interactive cover generation with live parameter control |
| `workflows/session_test_all.py` | Exercises all node system features end-to-end |

## Workflow examples

The `workflows/covers/` directory contains standalone scripts demonstrating individual features. Each loads the model, runs one workflow, and saves output audio.

| Workflow | Feature |
|---|---|
| `cover_basic.py` | Standard cover pipeline (encode, condition, generate, decode) |
| `sde_denoise_curve.py` | Per-frame SDE re-noise modulation |
| `velocity_scaling.py` | Per-frame transformation rate control |
| `prompt_blend.py` | Two prompts blended with a temporal curve |
| `x0_target_blend.py` | Two-pass morphing toward a target latent |
| `guidance_curve.py` | Per-frame CFG scale via positive + zeroed-out negative |
| `conditioning_average.py` | Weighted average of two text conditionings |
| `cover_semantic_blend.py` | Blend structural hints from two source audios |
| `latent_noise_mask.py` | Temporal inpainting mask on source latent |
| `initial_noise_curve.py` | Per-frame source/noise mixing in initial latent |
| `ode_noise_injection.py` | Per-frame ODE solver noise injection |
| `lora_generation.py` | LoRA-conditioned generation |
| `x0_target_from_reference.py` | Reference audio as x0 target for blending |

## Running with TensorRT

Build TRT engines, then pass them to Session:

```bash
uv run python -m acestep.engine.trt.build --checkpoint acestep-v15-turbo
```

```python
session = Session(
    trt_engines={
        "decoder": "trt_engines/decoder.engine",
        "vae_encode": "trt_engines/vae_encode.engine",
        "vae_decode": "trt_engines/vae_decode.engine",
    },
)
```

## Tests

```bash
uv run pytest tests/ -v
```
