#!/usr/bin/env python3
"""Profile the full generation pipeline to find remaining optimization targets."""

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
    # --- Setup (one-time costs, not profiled) ---
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

    # VAE encode
    with handler._load_model_context("vae"):
        src_lat = handler._encode_audio_to_latents(audio_raw).unsqueeze(0)
    T = src_lat.shape[1]
    # Pad to multiple of 5 for tokenizer
    pad_to = 5
    if T % pad_to != 0:
        pad_amount = pad_to - (T % pad_to)
        src_lat = torch.nn.functional.pad(src_lat, (0, 0, 0, pad_amount))
        T = src_lat.shape[1]

    # Semantic hints
    with handler._load_model_context("model"):
        src = src_lat.to(device=device, dtype=dtype)
        q, _ = handler.model.tokenizer.tokenize(src)
        hints = handler.model.detokenizer(q)[:, :T, :]

    # Text encode
    text_prompt = (
        "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"
        "# Caption\ndeathstep\n\n"
        "# Metas\n- bpm: 136\n- timesignature: 4\n- keyscale: G# minor\n- duration: 60\n"
        "<|endoftext|>\n"
    )
    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        tokens = handler.text_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
        text_hidden = handler.infer_text_embeddings(tokens["input_ids"].to(device))
        text_mask = tokens["attention_mask"].to(device).bool()
        lt = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
        lyric_hidden = handler.infer_lyric_embeddings(lt["input_ids"].to(device))
        lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)

    # Build condition
    handler._ensure_silence_latent_on_device()
    with handler._load_model_context("model"):
        enc_hs, enc_mask, ctx_lat = handler.model.prepare_condition(
            text_hidden_states=text_hidden.to(dtype),
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden.to(dtype),
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=src.clone(),
            refer_audio_order_mask=torch.zeros(1, device=device, dtype=torch.long),
            hidden_states=src,
            attention_mask=torch.ones(1, T, device=device, dtype=dtype),
            silence_latent=handler.silence_latent,
            src_latents=src,
            chunk_masks=torch.ones(1, T, 64, device=device, dtype=dtype),
            is_covers=torch.tensor([True], device=device),
            precomputed_lm_hints_25Hz=hints.to(device=device, dtype=dtype),
        )

    cond = PreparedCondition(
        encoder_hidden_states=enc_hs,
        encoder_attention_mask=enc_mask,
        context_latents=ctx_lat,
    )
    cs = ConditionSet(conditions=[cond])

    # TRT VAE decode
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(os.path.join(project_root, "trt_engines", "vae_decode_fp16.engine"), "rb") as f:
        vae_eng = rt.deserialize_cuda_engine(f.read())
    vae_ctx = vae_eng.create_execution_context()

    # Pre-allocate decode output buffer
    vae_ctx.set_input_shape("latents", (1, 64, 1500))
    out_shape = tuple(vae_ctx.get_tensor_shape("audio"))
    audio_buf = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    def trt_decode(lat_btd):
        lat = lat_btd.transpose(1, 2).float().contiguous()
        vae_ctx.set_tensor_address("latents", lat.data_ptr())
        vae_ctx.set_tensor_address("audio", audio_buf.data_ptr())
        s = torch.cuda.current_stream()
        vae_ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
        return audio_buf

    # Persistent engine
    if not hasattr(handler, "_diffusion_engine"):
        handler._diffusion_engine = DiffusionEngine(handler.model)
    engine = handler._diffusion_engine

    # --- Warmup ---
    print("Warming up...")
    config = DiffusionConfig(
        infer_steps=8, shift=3.0, seed=1528,
        use_cache=False, noise_on_cpu=True, denoise=1.0,
    )
    with handler._load_model_context("model"):
        for _ in range(5):
            r = engine.generate(condition_set=cs, config=config, source_latents=src)
            trt_decode(r["target_latents"])

    # --- Profile ---
    print("\n" + "=" * 50)
    print("  PROFILING: 30 runs, denoise=1.0, 8 steps")
    print("=" * 50)

    N = 30
    times_noise = []
    times_diff_only = []
    times_generate = []
    times_decode = []
    times_total = []

    with handler._load_model_context("model"):
        for i in range(N):
            cfg = DiffusionConfig(
                infer_steps=8, shift=3.0, seed=1528 + i,
                use_cache=False, noise_on_cpu=True, denoise=1.0,
            )

            # Noise generation
            torch.cuda.synchronize()
            tn0 = time.perf_counter()
            noise = engine._prepare_noise_cpu(cond, cfg.seed)
            torch.cuda.synchronize()
            tn1 = time.perf_counter()
            times_noise.append((tn1 - tn0) * 1000)

            # Full generate (includes noise + schedule + loop)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = engine.generate(condition_set=cs, config=cfg, source_latents=src)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # VAE decode
            audio = trt_decode(r["target_latents"])
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            times_generate.append((t1 - t0) * 1000)
            times_decode.append((t2 - t1) * 1000)
            times_total.append((t2 - t0) * 1000)

    def stats(times, label):
        s = sorted(times)
        print(f"  {label:20s}  mean={sum(s)/len(s):6.1f}ms  min={s[0]:6.1f}ms  p50={s[len(s)//2]:6.1f}ms  p95={s[int(len(s)*0.95)]:6.1f}ms")

    print()
    stats(times_noise, "Noise (CPU)")
    stats(times_generate, "generate() total")
    stats(times_decode, "VAE decode (TRT)")
    stats(times_total, "TOTAL (diff+dec)")

    gen_min = min(times_generate)
    dec_min = min(times_decode)
    noise_min = min(times_noise)
    overhead = gen_min - noise_min
    print(f"\n  Diffusion overhead (generate - noise): {overhead:.1f}ms")
    print(f"  Theoretical min (diff+dec): {gen_min + dec_min:.1f}ms")
    print(f"  Target: 300ms")
    print(f"  Gap: {gen_min + dec_min - 300:.1f}ms")


if __name__ == "__main__":
    main()
