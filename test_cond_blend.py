#!/usr/bin/env python3
"""
Test conditioning blend: two text prompts averaged at 50/50,
matching the real-time-cover.ap-cond-blendi.json ComfyUI workflow.

ComfyUI's ConditioningAverage blends the cross_attn (text embeddings)
before they reach the model. Lyrics/extras come from the "to" side.
"""

import os
import sys
import time

import importlib, importlib.util
_orig_find_spec = importlib.util.find_spec
def _p(name, *a, **k):
    if 'flash_attn' in str(name): return None
    return _orig_find_spec(name, *a, **k)
importlib.util.find_spec = _p

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import soundfile as sf
import torch
from pathlib import Path

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
    print("TEST: Conditioning Blend (50/50 two prompts)")
    print("=" * 70)

    from acestep.handler import AceStepHandler
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
    # 1. Load and encode audio (same as baseline)
    # ================================================================
    print("\n1. VAE encode source audio...")
    audio_data, sr = sf.read(SOURCE_AUDIO, dtype="float32")
    audio_raw = torch.from_numpy(audio_data.T if audio_data.ndim > 1 else audio_data.reshape(1, -1))
    if sr != 48000:
        audio_raw = torchaudio.transforms.Resample(sr, 48000)(audio_raw)
    audio_raw = audio_raw[:2, :60 * 48000]  # stereo, truncate to 60s

    with handler._load_model_context("vae"):
        vae_input = audio_raw.unsqueeze(0).to(device).to(handler.vae.dtype)
        with torch.no_grad():
            src_latent_bdt = handler.vae.encode(vae_input).latent_dist.sample()
    src_latent = src_latent_bdt.transpose(1, 2).to(dtype)  # [B, T, D]
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
    # 3. Encode BOTH text prompts
    # ================================================================
    print("\n3. Encode text prompts...")
    instruction = "Generate audio semantic tokens based on the given conditions:"
    bpm, duration, keyscale, timesig = 136, 60, "G# minor", 4
    meta_cap = f"- bpm: {bpm}\n- timesignature: {timesig}\n- keyscale: {keyscale}\n- duration: {duration}\n"

    # Prompt A (conditioning_to): "deathstep death deaht deaht\n"
    caption_a = "deathstep death deaht deaht\n"
    text_prompt_a = f"# Instruction\n{instruction}\n\n# Caption\n{caption_a}\n# Metas\n{meta_cap}<|endoftext|>\n"

    # Prompt B (conditioning_from): "ambiet angelic synths a lot of synths"
    caption_b = "ambiet angelic synths a lot of synths"
    text_prompt_b = f"# Instruction\n{instruction}\n\n# Caption\n{caption_b}\n\n# Metas\n{meta_cap}<|endoftext|>\n"

    # Lyrics (same for both, empty)
    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        # Encode prompt A
        tokens_a = handler.text_tokenizer(text_prompt_a, return_tensors="pt", add_special_tokens=False)
        text_hidden_a = handler.infer_text_embeddings(tokens_a["input_ids"].to(device))
        text_mask_a = tokens_a["attention_mask"].to(device).bool()

        # Encode prompt B
        tokens_b = handler.text_tokenizer(text_prompt_b, return_tensors="pt", add_special_tokens=False)
        text_hidden_b = handler.infer_text_embeddings(tokens_b["input_ids"].to(device))
        text_mask_b = tokens_b["attention_mask"].to(device).bool()

        # Encode lyrics (shared)
        lyric_tokens = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
        lyric_hidden = handler.infer_lyric_embeddings(lyric_tokens["input_ids"].to(device))
        lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)

    print(f"   text_hidden_a: {ts(text_hidden_a)}")
    print(f"   text_hidden_b: {ts(text_hidden_b)}")
    print(f"   lyric_hidden: {ts(lyric_hidden)}")

    # ================================================================
    # 4. Blend text embeddings (ConditioningAverage, strength=0.5)
    # ================================================================
    print("\n4. Blend text embeddings (0.5 * A + 0.5 * B)...")

    # ComfyUI's ConditioningAverage:
    #   t1 = conditioning_to (prompt_a)
    #   t0 = conditioning_from[:, :t1.shape[1]] (prompt_b, truncated/padded to t1's length)
    #   out = t1 * strength + t0 * (1 - strength)
    # Extras dict comes entirely from conditioning_to.
    strength = 0.5

    len_to = text_hidden_a.shape[1]  # conditioning_to length is the reference
    t0 = text_hidden_b[:, :len_to]   # truncate from to match to
    if t0.shape[1] < len_to:
        # pad from with zeros if shorter
        t0 = torch.cat([t0, torch.zeros(1, len_to - t0.shape[1], t0.shape[2], device=t0.device, dtype=t0.dtype)], dim=1)

    text_hidden_blended = text_hidden_a * strength + t0 * (1.0 - strength)
    text_mask_blended = text_mask_a  # mask from conditioning_to
    print(f"   blended text: {ts(text_hidden_blended)}")

    # ================================================================
    # 5. Build condition with blended text
    # ================================================================
    print("\n5. prepare_condition with blended text...")
    refer_packed = src_latent.clone()
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)
    chunk_masks = torch.ones(1, T, 64, device=device, dtype=dtype)
    is_covers = torch.BoolTensor([True]).to(device)
    handler._ensure_silence_latent_on_device()

    with handler._load_model_context("model"):
        enc_hidden, enc_mask, context_latents = model.prepare_condition(
            text_hidden_states=text_hidden_blended.to(dtype),
            text_attention_mask=text_mask_blended,
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
    print(f"   context_latents: {ts(context_latents)}")

    # ================================================================
    # 6. Diffusion loop (euler, no KV cache, CPU noise)
    # ================================================================
    print("\n6. Diffusion loop...")
    steps, shift, seed = 8, 3.0, 1528
    t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    t_schedule = shift * t_schedule / (1 + (shift - 1) * t_schedule)

    torch.manual_seed(seed)
    noise = torch.randn(1, 64, T, device="cpu", dtype=torch.float32).movedim(-1, -2).to(device=device, dtype=dtype)
    attn_mask = torch.ones(1, T, device=device, dtype=dtype)

    xt = noise.clone()
    with handler._load_model_context("model"):
        for i in range(steps):
            tc, tn = t_schedule[i].item(), t_schedule[i + 1].item()
            tt = torch.tensor([tc], device=device, dtype=dtype)
            with torch.no_grad():
                out = model.decoder(
                    hidden_states=xt, timestep=tt, timestep_r=tt,
                    attention_mask=attn_mask,
                    encoder_hidden_states=enc_hidden,
                    encoder_attention_mask=enc_mask,
                    context_latents=context_latents,
                )
            vt = out[0] if isinstance(out, tuple) else out
            if i == steps - 1:
                xt = xt - tc * vt
            else:
                xt = xt - vt * (tc - tn)
            print(f"   Step {i}: t={tc:.4f}->{tn:.4f} vt_mean={vt.float().mean():.4f}")

    print(f"\n   Final: {ts(xt)}")

    # ================================================================
    # 7. VAE decode and save
    # ================================================================
    print("\n7. VAE decode...")
    with handler._load_model_context("vae"):
        audio_out = handler.vae.decode(xt.transpose(1, 2))
        if hasattr(audio_out, 'sample'):
            audio_out = audio_out.sample

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "cond_blend_test.wav")
    sf.write(out_path, audio_out.squeeze(0).detach().cpu().float().numpy().T, 48000)
    print(f"   Saved: {out_path}")

    print("\n" + "=" * 70)
    print("Compare with ComfyUI output from real-time-cover.ap-cond-blendi.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
