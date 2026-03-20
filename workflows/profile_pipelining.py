#!/usr/bin/env python3
"""Test whether diffusion and VAE decode actually overlap on separate streams.

Three modes:
  1. Sequential: diffusion then decode, one at a time
  2. Pipelined: decode previous while diffusing next (separate streams)
  3. Theoretical: if perfect overlap, throughput = max(diffusion, decode)
"""

import os, sys, time
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
    os.path.expanduser("~"), ".cache", "torchinductor"
)

import torch
torch._dynamo.config.allow_unspec_int_on_nn_module = True

import soundfile as sf
import tensorrt as trt

from acestep.handler import AceStepHandler
from acestep.engine import DiffusionEngine, DiffusionConfig, ConditionSet, PreparedCondition


def main():
    # --- Setup ---
    handler = AceStepHandler()
    handler.initialize_service(
        project_root=project_root, config_path="acestep-v15-turbo",
        device="cuda", use_flash_attention=True, compile_model=True,
    )
    device, dtype = handler.device, handler.dtype

    data, sr = sf.read(
        os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav"),
        dtype="float32",
    )
    audio_raw = torch.from_numpy(data.T)[:2, :60 * 48000]

    with handler._load_model_context("vae"):
        src_lat = handler._encode_audio_to_latents(audio_raw).unsqueeze(0)
    T = src_lat.shape[1]
    pad = 5 - (T % 5) if T % 5 else 0
    if pad:
        src_lat = torch.nn.functional.pad(src_lat, (0, 0, 0, pad))
        T = src_lat.shape[1]

    with handler._load_model_context("model"):
        src = src_lat.to(device=device, dtype=dtype)
        q, _ = handler.model.tokenizer.tokenize(src)
        hints = handler.model.detokenizer(q)[:, :T, :]

    text_prompt = (
        "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"
        "# Caption\ndeathstep\n\n"
        "# Metas\n- bpm: 136\n- timesignature: 4\n- keyscale: G# minor\n- duration: 60\n"
        "<|endoftext|>\n"
    )
    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        tok = handler.text_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
        text_hs = handler.infer_text_embeddings(tok["input_ids"].to(device))
        text_mask = tok["attention_mask"].to(device).bool()
        lt = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
        lyric_hs = handler.infer_lyric_embeddings(lt["input_ids"].to(device))
        lyric_mask = torch.ones(lyric_hs.shape[:2], device=device, dtype=torch.bool)

    handler._ensure_silence_latent_on_device()
    with handler._load_model_context("model"):
        enc_hs, enc_mask, ctx_lat = handler.model.prepare_condition(
            text_hidden_states=text_hs.to(dtype), text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hs.to(dtype), lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=src.clone(),
            refer_audio_order_mask=torch.zeros(1, device=device, dtype=torch.long),
            hidden_states=src, attention_mask=torch.ones(1, T, device=device, dtype=dtype),
            silence_latent=handler.silence_latent, src_latents=src,
            chunk_masks=torch.ones(1, T, 64, device=device, dtype=dtype),
            is_covers=torch.tensor([True], device=device),
            precomputed_lm_hints_25Hz=hints.to(device=device, dtype=dtype),
        )

    cond = PreparedCondition(
        encoder_hidden_states=enc_hs, encoder_attention_mask=enc_mask,
        context_latents=ctx_lat,
    )
    cs = ConditionSet(conditions=[cond])

    # TRT VAE decode on a dedicated stream
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(os.path.join(project_root, "trt_engines", "vae_decode_fp16.engine"), "rb") as f:
        vae_eng = rt.deserialize_cuda_engine(f.read())
    vae_ctx = vae_eng.create_execution_context()

    decode_stream = torch.cuda.Stream()

    # Double-buffered decode: two output buffers
    vae_ctx.set_input_shape("latents", (1, 64, T))
    out_shape = tuple(vae_ctx.get_tensor_shape("audio"))
    audio_bufs = [
        torch.empty(out_shape, dtype=torch.float32, device="cuda"),
        torch.empty(out_shape, dtype=torch.float32, device="cuda"),
    ]

    def trt_decode_async(lat_btd, buf_idx):
        """Launch decode on decode_stream, return immediately."""
        lat = lat_btd.transpose(1, 2).float().contiguous()
        vae_ctx.set_tensor_address("latents", lat.data_ptr())
        vae_ctx.set_tensor_address("audio", audio_bufs[buf_idx].data_ptr())
        # Wait for diffusion stream to finish producing latents
        decode_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(decode_stream):
            vae_ctx.execute_async_v3(decode_stream.cuda_stream)

    def trt_decode_sync(lat_btd):
        """Synchronous decode on default stream."""
        lat = lat_btd.transpose(1, 2).float().contiguous()
        vae_ctx.set_tensor_address("latents", lat.data_ptr())
        vae_ctx.set_tensor_address("audio", audio_bufs[0].data_ptr())
        s = torch.cuda.current_stream()
        vae_ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
        return audio_bufs[0]

    if not hasattr(handler, "_diffusion_engine"):
        handler._diffusion_engine = DiffusionEngine(handler.model)
    engine = handler._diffusion_engine

    # --- Warmup ---
    print("Warming up...")
    with handler._load_model_context("model"):
        for i in range(5):
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )
            r = engine.generate(condition_set=cs, config=cfg, source_latents=src)
            trt_decode_sync(r["target_latents"])

    N_GENS = 20
    print(f"\n{'=' * 60}")
    print(f"  PIPELINING TEST: {N_GENS} generations")
    print(f"{'=' * 60}")

    # --- Mode 1: Sequential ---
    print("\n  [Sequential] diffusion -> decode -> diffusion -> decode ...")
    torch.cuda.synchronize()
    t0_seq = time.perf_counter()
    with handler._load_model_context("model"):
        for i in range(N_GENS):
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=2000 + i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )
            r = engine.generate(condition_set=cs, config=cfg, source_latents=src)
            trt_decode_sync(r["target_latents"])
    torch.cuda.synchronize()
    t_seq = time.perf_counter() - t0_seq

    # --- Mode 2: Pipelined ---
    print("  [Pipelined] decode overlaps with next diffusion ...")
    torch.cuda.synchronize()
    t0_pipe = time.perf_counter()
    with handler._load_model_context("model"):
        # First generation: diffuse
        cfg = DiffusionConfig(
            infer_steps=8, shift=3.0, seed=3000,
            use_cache=False, noise_on_cpu=True, denoise=1.0,
        )
        r = engine.generate(condition_set=cs, config=cfg, source_latents=src)
        prev_latents = r["target_latents"]

        for i in range(1, N_GENS):
            # Launch decode of previous on decode_stream
            buf_idx = i % 2
            trt_decode_async(prev_latents, buf_idx)

            # Diffuse next on default stream (overlaps with decode)
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=3000 + i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )
            r = engine.generate(condition_set=cs, config=cfg, source_latents=src)
            prev_latents = r["target_latents"]

            # Wait for decode to finish before overwriting its buffer
            decode_stream.synchronize()

        # Final decode
        trt_decode_async(prev_latents, N_GENS % 2)
        decode_stream.synchronize()
    torch.cuda.synchronize()
    t_pipe = time.perf_counter() - t0_pipe

    # --- Results ---
    seq_per = t_seq / N_GENS * 1000
    pipe_per = t_pipe / N_GENS * 1000
    seq_throughput = N_GENS / t_seq
    pipe_throughput = N_GENS / t_pipe

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({N_GENS} generations)")
    print(f"{'=' * 60}")
    print(f"  Sequential:")
    print(f"    Total:      {t_seq * 1000:.0f}ms")
    print(f"    Per-gen:    {seq_per:.1f}ms")
    print(f"    Throughput: {seq_throughput:.1f} gen/sec")
    print(f"  Pipelined:")
    print(f"    Total:      {t_pipe * 1000:.0f}ms")
    print(f"    Per-gen:    {pipe_per:.1f}ms")
    print(f"    Throughput: {pipe_throughput:.1f} gen/sec")
    print(f"  Speedup:      {seq_per / pipe_per:.2f}x")
    print(f"  Time saved:   {seq_per - pipe_per:.1f}ms per generation")


if __name__ == "__main__":
    main()
