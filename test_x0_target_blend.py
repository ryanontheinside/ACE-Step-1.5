#!/usr/bin/env python3
"""
Spike: Per-frame x0 target blending.

At each step, computes x0_pred from the decoder velocity, then blends
it per-frame toward a target latent before integration. This creates
latent-space audio morphing: different parts of the song converge
toward different audio targets.

Setup: source is deathstep cover (no LoRA), target is daftpunk LoRA
cover (from test_lora_solo.py). Both share the same temporal structure
(same source audio, same seed) but differ in style/timbre.

Ramp curve: 0.0 at start (pure deathstep) -> 0.8 at end (mostly daftpunk).
Gated: only blends during refinement steps (second half of diffusion).
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
    print("SPIKE: Per-frame x0 target blending")
    print("  Morph from deathstep toward daftpunk LoRA cover")
    print("  Ramp 0.0 -> 0.8 across 60s (gated to refinement steps)")
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
    # 2. Load daftpunk LoRA cover latent as target
    # ================================================================
    print("\n2. Load target latent (daftpunk LoRA cover)...")
    target_path = os.path.join(OUTPUT_DIR, "daftpunk_solo_latent.pt")
    assert os.path.exists(target_path), f"Run test_lora_solo.py first to generate {target_path}"
    target_latent = torch.load(target_path, weights_only=True).to(device=device, dtype=dtype)
    print(f"   target_latent: {ts(target_latent)}")
    diff = (src_latent - target_latent).float().abs().mean().item()
    print(f"   src vs target diff: {diff:.4f}")

    # ================================================================
    # 3. Extract semantic hints
    # ================================================================
    print("\n3. Extract semantic hints...")
    with handler._load_model_context("model"):
        src_on_device = src_latent.to(device=device, dtype=dtype)
        with torch.no_grad():
            quantized, _ = model.tokenizer.tokenize(src_on_device)
            lm_hints = model.detokenizer(quantized)
        lm_hints = lm_hints[:, :T, :]

    # ================================================================
    # 4. Encode text prompt
    # ================================================================
    print("\n4. Encode text prompt...")
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

    # ================================================================
    # 5. Build condition
    # ================================================================
    print("\n5. prepare_condition...")
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

    # ================================================================
    # 6. Build blend curve and run diffusion with x0 target blending
    # ================================================================
    print("\n6. Diffusion loop with per-frame x0 target blending...")
    # Blend curve: how much to pull x0_pred toward target per frame
    # 0.0 = pure model prediction, 1.0 = fully replaced by target
    blend_curve = torch.linspace(0.0, 0.8, T, device=device, dtype=dtype)  # [T]
    blend_3d = blend_curve.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    print(f"   blend curve: {blend_curve[0]:.2f} (start) -> {blend_curve[-1]:.2f} (end)")

    steps, shift, seed = 8, 3.0, 1528
    t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    t_schedule = shift * t_schedule / (1 + (shift - 1) * t_schedule)

    torch.manual_seed(seed)
    noise = torch.randn(1, 64, T, device="cpu", dtype=torch.float32).movedim(-1, -2).to(device=device, dtype=dtype)
    attn_mask = torch.ones(1, T, device=device, dtype=dtype)

    xt = noise.clone()
    t0 = time.time()
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

            # Compute x0 prediction
            t_tensor = tc * torch.ones((1,), device=device, dtype=dtype)
            x0_pred = model.get_x0_from_noise(xt, vt, t_tensor)

            # Per-frame x0 target blending (gated: only during refinement steps)
            step_progress = i / max(steps - 1, 1)
            blend_gate = max(0.0, step_progress - 0.5) * 2.0  # 0 for first half, ramp 0->1 second half
            effective_blend = blend_3d * blend_gate
            x0_blended = (1.0 - effective_blend) * x0_pred + effective_blend * target_latent

            if tn <= 0:
                # Final step: return blended x0
                xt = x0_blended
            else:
                # Recompute velocity from blended x0 and integrate
                v_blended = (xt - x0_blended) / tc
                dt = tc - tn
                xt = xt - v_blended * dt

            diff_start = (xt[0, :25, :] - target_latent[0, :25, :]).float().abs().mean().item()
            diff_end = (xt[0, -25:, :] - target_latent[0, -25:, :]).float().abs().mean().item()
            print(f"   Step {i}: t={tc:.4f}->{tn:.4f} dist_to_target: start={diff_start:.3f} end={diff_end:.3f}")

    elapsed = time.time() - t0
    print(f"\n   Diffusion time: {elapsed:.2f}s ({steps/elapsed:.1f} steps/sec)")
    print(f"   Final: {ts(xt)}")

    # ================================================================
    # 7. VAE decode and save
    # ================================================================
    print("\n7. VAE decode...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with handler._load_model_context("vae"):
        audio_out = handler.vae.decode(xt.transpose(1, 2))
        if hasattr(audio_out, 'sample'):
            audio_out = audio_out.sample
        audio_np = audio_out.squeeze(0).detach().cpu().float().numpy().T

    path = os.path.join(OUTPUT_DIR, "x0_target_blend_timeshift.wav")
    sf.write(path, audio_np, 48000)
    print(f"   Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
