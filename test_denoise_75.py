#!/usr/bin/env python3
"""
Test denoise=0.75 cover generation for comparison with ComfyUI.

Minimal delta from test_cond_blend.py: single condition, denoise=0.75.
"""

import importlib, importlib.util
_orig_find_spec = importlib.util.find_spec
def _p(name, *a, **k):
    if 'flash_attn' in str(name): return None
    return _orig_find_spec(name, *a, **k)
importlib.util.find_spec = _p

import os
import sys
import time

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import soundfile as sf
import torch

def _load_audio_sf(path, **kwargs):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1: data = data.reshape(1, -1)
    else: data = data.T
    return torch.from_numpy(data), sr
import torchaudio
torchaudio.load = _load_audio_sf

SOURCE_AUDIO = os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output")


def ts(t):
    if t is None: return "None"
    f = t.detach().float()
    return f"shape={list(t.shape)} mean={f.mean():.4f} std={f.std():.4f}"


def main():
    print("=" * 70)
    print("TEST: Denoise=0.75 cover (engine vs manual euler)")
    print("=" * 70)

    from acestep.handler import AceStepHandler
    from acestep.engine import (
        PreparedCondition, ConditionSet, DiffusionEngine, DiffusionConfig,
    )

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=project_root, config_path="acestep-v15-turbo",
        device="cuda", use_flash_attention=True,
    )
    assert success, f"Init failed: {status}"
    device = handler.device
    dtype = handler.dtype
    model = handler.model

    # ================================================================
    # 1. Load and encode audio
    # ================================================================
    print("\n1. VAE encode source audio...")
    audio_data, sr = sf.read(SOURCE_AUDIO, dtype="float32")
    audio_raw = torch.from_numpy(audio_data.T if audio_data.ndim > 1 else audio_data.reshape(1, -1))
    if sr != 48000:
        audio_raw = torchaudio.transforms.Resample(sr, 48000)(audio_raw)
    audio_raw = audio_raw[:2, :60 * 48000]

    with handler._load_model_context("vae"):
        vae_input = audio_raw.unsqueeze(0).to(device).to(handler.vae.dtype)
        with torch.no_grad():
            src_latent_bdt = handler.vae.encode(vae_input).latent_dist.sample()
    src_latent = src_latent_bdt.transpose(1, 2).to(dtype)
    T = src_latent.shape[1]
    print(f"   src_latent: {ts(src_latent)}")

    # ================================================================
    # 2. Extract semantic hints
    # ================================================================
    print("\n2. Extract semantic hints...")
    with handler._load_model_context("model"):
        src_on_device = src_latent.to(device=device, dtype=dtype)
        with torch.no_grad():
            quantized, _ = model.tokenizer.tokenize(src_on_device)
            lm_hints = model.detokenizer(quantized)
        lm_hints = lm_hints[:, :T, :]
    print(f"   semantic_hints: {ts(lm_hints)}")

    # ================================================================
    # 3. Encode text prompt
    # ================================================================
    print("\n3. Encode text prompt...")
    instruction = "Generate audio semantic tokens based on the given conditions:"
    bpm, duration, keyscale, timesig = 136, 60, "G# minor", 4
    meta_cap = f"- bpm: {bpm}\n- timesignature: {timesig}\n- keyscale: {keyscale}\n- duration: {duration}\n"

    caption = "deathstep death deaht deaht\n"
    text_prompt = f"# Instruction\n{instruction}\n\n# Caption\n{caption}\n# Metas\n{meta_cap}<|endoftext|>\n"
    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        tokens = handler.text_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
        text_hidden = handler.infer_text_embeddings(tokens["input_ids"].to(device))
        text_mask = tokens["attention_mask"].to(device).bool()

        lyric_tokens = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
        lyric_hidden = handler.infer_lyric_embeddings(lyric_tokens["input_ids"].to(device))
        lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)

    print(f"   text_hidden: {ts(text_hidden)}")

    # ================================================================
    # 4. Build condition
    # ================================================================
    print("\n4. prepare_condition...")
    refer_packed = src_latent.clone()
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)
    chunk_masks = torch.ones(1, T, 64, device=device, dtype=dtype)
    is_covers = torch.BoolTensor([True]).to(device)
    handler._ensure_silence_latent_on_device()

    with handler._load_model_context("model"):
        enc_hidden, enc_mask, context_latents = model.prepare_condition(
            text_hidden_states=text_hidden.to(dtype),
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden.to(dtype),
            lyric_attention_mask=lyric_mask,
            refer_audio_acoustic_hidden_states_packed=refer_packed.to(device),
            refer_audio_order_mask=refer_order_mask,
            hidden_states=src_on_device,
            attention_mask=torch.ones(1, T, device=device, dtype=dtype),
            silence_latent=handler.silence_latent,
            src_latents=src_on_device,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=lm_hints.to(device=device, dtype=dtype),
        )
    print(f"   encoder_hidden: {ts(enc_hidden)}")

    # ================================================================
    # 5. Engine generate at multiple denoise levels
    # ================================================================
    cond = PreparedCondition(
        encoder_hidden_states=enc_hidden,
        encoder_attention_mask=enc_mask,
        context_latents=context_latents,
    )
    cs = ConditionSet(conditions=[cond])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for denoise_val in [0.5, 0.75, 1.0]:
        print(f"\n5. Engine generate (denoise={denoise_val})...")
        config = DiffusionConfig(
            infer_steps=8, shift=3.0, seed=1528,
            use_cache=False, noise_on_cpu=True,
            denoise=denoise_val,
        )

        engine = DiffusionEngine(model)
        t0 = time.time()
        result = engine.generate(
            condition_set=cs,
            config=config,
            source_latents=src_on_device,
        )
        elapsed = time.time() - t0
        xt = result["target_latents"]
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Final: {ts(xt)}")

        del engine, result
        torch.cuda.empty_cache()

        print("   VAE decode...")
        with handler._load_model_context("vae"):
            audio_out = handler.vae.decode(xt.transpose(1, 2))
            if hasattr(audio_out, 'sample'):
                audio_out = audio_out.sample
            audio_np = audio_out.squeeze(0).detach().cpu().float().numpy().T

        name = f"denoise_{int(denoise_val * 100):03d}"
        path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        sf.write(path, audio_np, 48000)
        print(f"   Saved: {path}")
        del xt, audio_out, audio_np
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
