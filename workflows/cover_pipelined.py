#!/usr/bin/env python3
"""Pipelined continuous generation: decode overlaps with next diffusion.

Demonstrates GenerationPipeline for back-to-back generation where the
user is iterating (changing seed, denoise, prompt) and wants minimum
latency between results.
"""

import os, sys, time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import soundfile as sf
import torch

from acestep.nodes import Audio
from acestep.nodes.model_nodes import LoadModel
from acestep.nodes.vae_nodes import VAEEncodeAudio, _find_trt_engine, _get_trt_vae, _trt_available
from acestep.nodes.cond_nodes import TextEncode
from acestep.nodes.semantic_nodes import SemanticExtract
from acestep.engine import DiffusionEngine, DiffusionConfig, ConditionSet, PreparedCondition, GenerationPipeline

SOURCE_AUDIO = os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output", "workflows")


def make_trt_decode_fn(device):
    """Create a TRT VAE decode function for the pipeline."""
    trt_path = _find_trt_engine("vae_decode_fp16.engine")
    if not trt_path or not _trt_available():
        raise RuntimeError("TRT VAE decode engine not found")

    entry = _get_trt_vae(trt_path, device)
    ctx = entry["context"]

    # Pre-query output shape, double-buffered
    ctx.set_input_shape("latents", (1, 64, 1500))
    out_shape = tuple(ctx.get_tensor_shape("audio"))
    audio_bufs = [
        torch.empty(out_shape, dtype=torch.float32, device=device),
        torch.empty(out_shape, dtype=torch.float32, device=device),
    ]
    buf_idx = [0]

    def decode(latents_btd):
        lat = latents_btd.transpose(1, 2).float().contiguous()
        ctx.set_input_shape("latents", tuple(lat.shape))
        ctx.set_tensor_address("latents", lat.data_ptr())
        cur_shape = tuple(ctx.get_tensor_shape("audio"))
        idx = buf_idx[0]
        buf_idx[0] = 1 - idx
        if audio_bufs[idx].shape != cur_shape:
            audio_bufs[idx] = torch.empty(cur_shape, dtype=torch.float32, device=device)
        ctx.set_tensor_address("audio", audio_bufs[idx].data_ptr())
        stream = torch.cuda.current_stream()
        ctx.execute_async_v3(stream.cuda_stream)
        return audio_bufs[idx]

    return decode


def main():
    print("=" * 60)
    print("  PIPELINED CONTINUOUS GENERATION")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Setup (one-time) ---
    handles = LoadModel().execute(
        project_root=project_root,
        config_path="acestep-v15-turbo",
        device="cuda",
        use_flash_attention=True,
        compile_model=True,
    )
    model, clip, vae = handles["model"], handles["clip"], handles["vae"]
    handler = model.handler
    device = torch.device(handler.device)

    # Encode source
    data, sr = sf.read(SOURCE_AUDIO, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    waveform = waveform[:2, :60 * 48000]
    source_audio = Audio(waveform=waveform, sample_rate=48000)
    source_latent = VAEEncodeAudio().execute(vae=vae, audio=source_audio)["latent"]

    # Encode text + hints
    hints = SemanticExtract().execute(model=model, latent=source_latent)["semantic_hints"]
    conditioning = TextEncode().execute(
        clip=clip, model=model,
        source_latent=source_latent,
        semantic_hints=hints,
        tags="deathstep death deaht deaht",
        task="cover", bpm=136, duration=60.0, key="G# minor",
    )["conditioning"]

    # Build engine condition set
    cond = PreparedCondition(
        encoder_hidden_states=conditioning.encoder_hidden_states,
        encoder_attention_mask=conditioning.encoder_attention_mask,
        context_latents=conditioning.context_latents,
    )
    cs = ConditionSet(conditions=[cond])
    src = source_latent.tensor.to(device=device, dtype=handler.dtype)

    # Create pipeline
    if not hasattr(handler, "_diffusion_engine"):
        handler._diffusion_engine = DiffusionEngine(handler.model)
    engine = handler._diffusion_engine
    decode_fn = make_trt_decode_fn(device)
    pipeline = GenerationPipeline(engine, decode_fn, device)

    # --- Warmup ---
    print("\nWarming up (torch.compile)...")
    with handler._load_model_context("model"):
        for i in range(3):
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )
            pipeline.generate_single(cs, cfg, source_latents=src)
    print("Warm.")

    # --- Sequential baseline ---
    N = 10
    print(f"\n[Sequential] {N} generations...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with handler._load_model_context("model"):
        for i in range(N):
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=5000 + i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )
            audio = pipeline.generate_single(cs, cfg, source_latents=src)
    torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0

    # --- Pipelined ---
    print(f"[Pipelined]  {N} generations...")
    pipeline.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = []
    with handler._load_model_context("model"):
        for i in range(N):
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=6000 + i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )
            audio = pipeline.generate_next(cs, cfg, source_latents=src)
            if audio is not None:
                results.append(audio)
        final = pipeline.flush()
        if final is not None:
            results.append(final)
    torch.cuda.synchronize()
    t_pipe = time.perf_counter() - t0

    # --- Results ---
    seq_ms = t_seq / N * 1000
    pipe_ms = t_pipe / N * 1000

    print(f"\n{'=' * 60}")
    print(f"  Sequential: {seq_ms:.1f}ms/gen  ({N / t_seq:.1f} gen/sec)")
    print(f"  Pipelined:  {pipe_ms:.1f}ms/gen  ({N / t_pipe:.1f} gen/sec)")
    print(f"  Speedup:    {seq_ms / pipe_ms:.2f}x")
    print(f"  Saved:      {seq_ms - pipe_ms:.1f}ms per generation")
    print(f"{'=' * 60}")

    # Save last result
    if results:
        wav = results[-1]
        if wav.dim() == 3:
            wav = wav.squeeze(0)
        sf.write(
            os.path.join(OUTPUT_DIR, "cover_pipelined.wav"),
            wav.detach().cpu().float().numpy().T, 48000,
        )
        print(f"\nSaved: test_output/workflows/cover_pipelined.wav")


if __name__ == "__main__":
    main()
