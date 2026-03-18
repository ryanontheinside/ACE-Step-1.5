#!/usr/bin/env python3
"""End-to-end pipeline timing: compile_model=True + TRT VAE.

Uses the handler's compile_model=True (now compiles model.decoder directly
with dynamic=True) and TRT FP16 VAE engines for encode/decode.

Baseline numbers (eager PyTorch, no TRT, prior validated runs):
  vae_encode:      331ms
  semantic_hints:  125ms
  text_encode:      50ms
  prepare_cond:     54ms
  dit_8step:       331ms
  vae_decode:      162ms
"""

import importlib, importlib.util
_orig_find_spec = importlib.util.find_spec
def _p(name, *a, **k):
    if 'flash_attn' in str(name): return None
    return _orig_find_spec(name, *a, **k)
importlib.util.find_spec = _p

import os, sys, time
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import soundfile as sf
import torch
import tensorrt as trt

def _load_audio_sf(path, **kwargs):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1: data = data.reshape(1, -1)
    else: data = data.T
    return torch.from_numpy(data), sr
import torchaudio
torchaudio.load = _load_audio_sf

SOURCE_AUDIO = os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output")

BASELINE = {
    "vae_encode": 331,
    "semantic_hints": 125,
    "text_encode": 50,
    "prepare_cond": 54,
    "dit_8step": 331,
    "vae_decode": 162,
}


class TRTVAEEncoder:
    def __init__(self, engine_path, device="cuda"):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device(device)

    def __call__(self, audio):
        inp = audio.float().contiguous()
        self.ctx.set_input_shape("audio", tuple(inp.shape))
        self.ctx.set_tensor_address("audio", inp.data_ptr())
        out_shape = tuple(self.ctx.get_tensor_shape("moments"))
        out = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self.ctx.set_tensor_address("moments", out.data_ptr())
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        mean, logvar = out.chunk(2, dim=1)
        std = (0.5 * logvar).exp()
        return mean + std * torch.randn_like(mean)


class TRTVAEDecoder:
    def __init__(self, engine_path, device="cuda"):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device(device)

    def __call__(self, latents):
        inp = latents.float().contiguous()
        self.ctx.set_input_shape("latents", tuple(inp.shape))
        self.ctx.set_tensor_address("latents", inp.data_ptr())
        out_shape = tuple(self.ctx.get_tensor_shape("audio"))
        out = torch.empty(out_shape, dtype=torch.float32, device=self.device)
        self.ctx.set_tensor_address("audio", out.data_ptr())
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return result, ms


def run_pipeline(handler, device, dtype, audio_gpu, trt_enc, trt_dec):
    model = handler.model
    times = {}

    # 1. VAE encode (TRT)
    src_bdt, times["vae_encode"] = timed(lambda: trt_enc(audio_gpu))
    src_latent = src_bdt.transpose(1, 2).to(dtype)
    T = src_latent.shape[1]
    src_on_device = src_latent.to(device=device, dtype=dtype)

    # 2. Semantic hints
    def _hints():
        with handler._load_model_context("model"):
            with torch.no_grad():
                q, _ = model.tokenizer.tokenize(src_on_device)
                return model.detokenizer(q)[:, :T, :]
    lm_hints, times["semantic_hints"] = timed(_hints)

    # 3. Text encode (Qwen3 LLM)
    instruction = "Generate audio semantic tokens based on the given conditions:"
    meta_cap = "- bpm: 136\n- timesignature: 4\n- keyscale: G# minor\n- duration: 60\n"
    text_prompt = f"# Instruction\n{instruction}\n\n# Caption\ndeathstep\n# Metas\n{meta_cap}<|endoftext|>\n"
    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    def _text_encode():
        with handler._load_model_context("text_encoder"):
            tokens = handler.text_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
            text_hidden = handler.infer_text_embeddings(tokens["input_ids"].to(device))
            text_mask = tokens["attention_mask"].to(device).bool()
            lyric_tokens = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
            lyric_hidden = handler.infer_lyric_embeddings(lyric_tokens["input_ids"].to(device))
            lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)
            return text_hidden, text_mask, lyric_hidden, lyric_mask
    (text_hidden, text_mask, lyric_hidden, lyric_mask), times["text_encode"] = timed(_text_encode)

    # 4. Prepare condition
    handler._ensure_silence_latent_on_device()
    def _prepare():
        with handler._load_model_context("model"):
            return model.prepare_condition(
                text_hidden_states=text_hidden.to(dtype), text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden.to(dtype), lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=src_latent.clone().to(device),
                refer_audio_order_mask=torch.zeros(1, device=device, dtype=torch.long),
                hidden_states=src_on_device,
                attention_mask=torch.ones(1, T, device=device, dtype=dtype),
                silence_latent=handler.silence_latent, src_latents=src_on_device,
                chunk_masks=torch.ones(1, T, 64, device=device, dtype=dtype),
                is_covers=torch.BoolTensor([True]).to(device),
                precomputed_lm_hints_25Hz=lm_hints.to(device=device, dtype=dtype),
            )
    (enc_hidden, enc_mask, ctx_latents), times["prepare_cond"] = timed(_prepare)

    # 5. DiT diffusion (8 steps via DiffusionEngine)
    from acestep.engine import PreparedCondition, ConditionSet, DiffusionEngine, DiffusionConfig

    cond = PreparedCondition(
        encoder_hidden_states=enc_hidden,
        encoder_attention_mask=enc_mask,
        context_latents=ctx_latents,
    )
    cs = ConditionSet(conditions=[cond])
    config = DiffusionConfig(
        infer_steps=8, shift=3.0, seed=1528,
        use_cache=False, noise_on_cpu=True, denoise=0.75,
    )

    def _dit():
        with handler._load_model_context("model"):
            engine = DiffusionEngine(model)
            return engine.generate(
                condition_set=cs, config=config, source_latents=src_on_device,
            )
    result, times["dit_8step"] = timed(_dit)
    xt = result["target_latents"]

    # 6. VAE decode (TRT)
    audio_out, times["vae_decode"] = timed(lambda: trt_dec(xt.transpose(1, 2)))

    return audio_out, times


def main():
    from acestep.handler import AceStepHandler

    audio_data, sr = sf.read(SOURCE_AUDIO, dtype="float32")
    audio_raw = torch.from_numpy(audio_data.T if audio_data.ndim > 1 else audio_data.reshape(1, -1))
    if sr != 48000:
        audio_raw = torchaudio.transforms.Resample(sr, 48000)(audio_raw)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rerun_cached = ["vae_encode", "semantic_hints", "text_encode", "prepare_cond"]

    print("=" * 62)
    print("  END-TO-END TIMING: 60s cover, 8-step, denoise=0.75")
    print("  RTX 5090 / CUDA 12.8 / PyTorch 2.9 / TRT 10.16")
    print("  torch.compile(decoder, dynamic=True) + TRT VAE FP16")
    print("=" * 62)

    quantization = os.environ.get("QUANTIZATION", None) or None
    label = f"compile + TRT VAE" + (f" + {quantization}" if quantization else "")
    print(f"\n  Loading model (compile_model=True, quantization={quantization})...")
    handler = AceStepHandler()
    _, ok = handler.initialize_service(
        project_root=project_root, config_path="acestep-v15-turbo",
        device="cuda", use_flash_attention=False, compile_model=True,
        quantization=quantization,
    )
    assert ok
    device, dtype = handler.device, handler.dtype
    audio_gpu = audio_raw[:2, :60 * 48000].unsqueeze(0).to(device)

    trt_enc = TRTVAEEncoder("trt_engines/vae_encode_fp16.engine")
    trt_dec = TRTVAEDecoder("trt_engines/vae_decode_fp16.engine")

    # Warmup
    print("  Warmup 1 (triggers torch.compile, ~80s first time, cached after)...")
    t0 = time.perf_counter()
    run_pipeline(handler, device, dtype, audio_gpu, trt_enc, trt_dec)
    print(f"  Warmup 1: {(time.perf_counter()-t0)*1000:.0f}ms")

    print("  Warmup 2...")
    t0 = time.perf_counter()
    run_pipeline(handler, device, dtype, audio_gpu, trt_enc, trt_dec)
    print(f"  Warmup 2: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Timed run
    print("\n  Timed run...")
    audio_opt, times_opt = run_pipeline(handler, device, dtype, audio_gpu, trt_enc, trt_dec)

    sf.write(os.path.join(OUTPUT_DIR, "e2e_optimized.wav"),
             audio_opt.squeeze(0).cpu().float().numpy().T, 48000)

    # Results
    print(f"\n  {'Stage':<25s} {'Baseline':>10s} {'Optimized':>10s} {'Speedup':>8s}")
    print(f"  {'-'*55}")
    for stage in times_opt:
        b = BASELINE[stage]
        o = times_opt[stage]
        print(f"  {stage:<25s} {b:>8.0f}ms {o:>8.0f}ms {b/o:>7.1f}x")

    t_base_first = sum(BASELINE.values())
    t_opt_first = sum(times_opt.values())
    t_base_rerun = sum(v for k, v in BASELINE.items() if k not in rerun_cached)
    t_opt_rerun = sum(v for k, v in times_opt.items() if k not in rerun_cached)

    print(f"  {'-'*55}")
    print(f"  {'First run':<25s} {t_base_first:>8.0f}ms {t_opt_first:>8.0f}ms {t_base_first/t_opt_first:>7.1f}x")
    print(f"  {'Re-run (hot path)':<25s} {t_base_rerun:>8.0f}ms {t_opt_rerun:>8.0f}ms {t_base_rerun/t_opt_rerun:>7.1f}x")
    print(f"  {'='*55}")
    print(f"\n  Output: {os.path.join(OUTPUT_DIR, 'e2e_optimized.wav')}")


if __name__ == "__main__":
    main()
