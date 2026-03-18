#!/usr/bin/env python3
"""
Test temporal conditioning blend with LoRA switching.

Replicates lora_blend_api.json: daftpunk LoRA vs deathstep LoRA
alternating via a pulse temporal mask (151 frames/cycle).

Minimal delta from test_cond_blend.py with LoRA weight patching.
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
from safetensors.torch import load_file

def _load_audio_sf(path, **kwargs):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1: data = data.reshape(1, -1)
    else: data = data.T
    return torch.from_numpy(data), sr
import torchaudio
torchaudio.load = _load_audio_sf

SOURCE_AUDIO = os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(project_root, "test_output")
LORA_DIR = r"C:\_dev\models\comfyui_models\loras\acestep1.5"
LORA_A_PATH = os.path.join(LORA_DIR, "daftpunkstyle1200.safetensors")
LORA_B_PATH = os.path.join(LORA_DIR, "deathsteap_1.safetensors")
LORA_A_STRENGTH = 2.0
LORA_B_STRENGTH = 2.0


def ts(t):
    if t is None: return "None"
    f = t.detach().float()
    return f"shape={list(t.shape)} mean={f.mean():.4f} std={f.std():.4f}"


def precompute_lora_deltas(lora_path, strength, decoder, device, dtype):
    """Load a LoRA safetensors and precompute merged weight deltas.

    Returns dict mapping decoder param name -> delta tensor (strength * B @ A).
    """
    raw = load_file(lora_path)
    # Group lora_A and lora_B by parameter
    pairs = {}
    for key, tensor in raw.items():
        # key format: base_model.model.layers.0.cross_attn.q_proj.lora_A.weight
        parts = key.replace("base_model.model.", "")
        if ".lora_A.weight" in parts:
            param_name = parts.replace(".lora_A.weight", ".weight")
            pairs.setdefault(param_name, {})["A"] = tensor
        elif ".lora_B.weight" in parts:
            param_name = parts.replace(".lora_B.weight", ".weight")
            pairs.setdefault(param_name, {})["B"] = tensor

    deltas = {}
    for param_name, ab in pairs.items():
        A = ab["A"].to(device=device, dtype=dtype)  # [rank, in]
        B = ab["B"].to(device=device, dtype=dtype)  # [out, rank]
        deltas[param_name] = strength * (B @ A)
    return deltas


def apply_lora_deltas(decoder, deltas, sign=1.0):
    """Add (sign=1) or remove (sign=-1) precomputed LoRA deltas."""
    decoder_params = dict(decoder.named_parameters())
    for param_name, delta in deltas.items():
        if param_name in decoder_params:
            decoder_params[param_name].data.add_(delta, alpha=sign)


def main():
    print("=" * 70)
    print("TEST: Temporal Blend with LoRA Switching")
    print("  Daftpunk LoRA vs Deathstep LoRA, pulse mask 151 frames/cycle")
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
    # 3. Encode text prompts (matching workflow)
    # ================================================================
    print("\n3. Encode text prompts...")
    instruction = "Generate audio semantic tokens based on the given conditions:"
    bpm, duration, keyscale, timesig = 136, 60, "G# minor", 4
    meta_cap = f"- bpm: {bpm}\n- timesignature: {timesig}\n- keyscale: {keyscale}\n- duration: {duration}\n"

    caption_a = "daft punk style"
    text_prompt_a = f"# Instruction\n{instruction}\n\n# Caption\n{caption_a}\n\n# Metas\n{meta_cap}<|endoftext|>\n"

    caption_b = "heavy demon techno, growling bass,  afxdump"
    text_prompt_b = f"# Instruction\n{instruction}\n\n# Caption\n{caption_b}\n\n# Metas\n{meta_cap}<|endoftext|>\n"

    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        tokens_a = handler.text_tokenizer(text_prompt_a, return_tensors="pt", add_special_tokens=False)
        text_hidden_a = handler.infer_text_embeddings(tokens_a["input_ids"].to(device))
        text_mask_a = tokens_a["attention_mask"].to(device).bool()

        tokens_b = handler.text_tokenizer(text_prompt_b, return_tensors="pt", add_special_tokens=False)
        text_hidden_b = handler.infer_text_embeddings(tokens_b["input_ids"].to(device))
        text_mask_b = tokens_b["attention_mask"].to(device).bool()

        lyric_tokens = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
        lyric_hidden = handler.infer_lyric_embeddings(lyric_tokens["input_ids"].to(device))
        lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)

    print(f"   text_hidden_a (daftpunk): {ts(text_hidden_a)}")
    print(f"   text_hidden_b (deathstep): {ts(text_hidden_b)}")

    # ================================================================
    # 4. Precompute LoRA deltas
    # ================================================================
    print("\n4. Precompute LoRA deltas...")
    deltas_a = precompute_lora_deltas(LORA_A_PATH, LORA_A_STRENGTH, model.decoder, device, dtype)
    deltas_b = precompute_lora_deltas(LORA_B_PATH, LORA_B_STRENGTH, model.decoder, device, dtype)
    print(f"   Daftpunk: {len(deltas_a)} params, strength={LORA_A_STRENGTH}")
    print(f"   Deathstep: {len(deltas_b)} params, strength={LORA_B_STRENGTH}")

    # ================================================================
    # 5. Build conditions (with LoRA applied during prepare_condition)
    # ================================================================
    print("\n5. prepare_condition for each prompt (with LoRA)...")
    refer_packed = src_latent.clone()
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)
    chunk_masks = torch.ones(1, T, 64, device=device, dtype=dtype)
    is_covers = torch.BoolTensor([True]).to(device)
    handler._ensure_silence_latent_on_device()

    with handler._load_model_context("model"):
        # Condition A with daftpunk LoRA
        apply_lora_deltas(model.decoder, deltas_a, sign=1.0)
        enc_hidden_a, enc_mask_a, ctx_latents_a = model.prepare_condition(
            text_hidden_states=text_hidden_a.to(dtype),
            text_attention_mask=text_mask_a,
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
        apply_lora_deltas(model.decoder, deltas_a, sign=-1.0)

        # Condition B with deathstep LoRA
        apply_lora_deltas(model.decoder, deltas_b, sign=1.0)
        enc_hidden_b, enc_mask_b, ctx_latents_b = model.prepare_condition(
            text_hidden_states=text_hidden_b.to(dtype),
            text_attention_mask=text_mask_b,
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
        apply_lora_deltas(model.decoder, deltas_b, sign=-1.0)

    print(f"   cond_a encoder_hidden: {ts(enc_hidden_a)}")
    print(f"   cond_b encoder_hidden: {ts(enc_hidden_b)}")

    # ================================================================
    # 6. Build temporal mask (pulse, 151 frames/cycle, matching workflow)
    # ================================================================
    print("\n6. Build temporal mask...")
    frames_per_cycle = 151
    frame_idx = torch.arange(T, device=device, dtype=dtype)
    # Pulse: first half of cycle = 1, second half = 0
    pulse = ((frame_idx % frames_per_cycle) < (frames_per_cycle / 2)).to(dtype)
    # Workflow: deathstep gets pulse, daftpunk gets inverted
    w_deathstep = pulse         # [T]
    w_daftpunk = 1.0 - pulse    # [T]
    # Reshape for broadcasting: [1, T, 1]
    w_a_3d = w_daftpunk.unsqueeze(0).unsqueeze(-1)
    w_b_3d = w_deathstep.unsqueeze(0).unsqueeze(-1)
    print(f"   Pulse: {frames_per_cycle} frames/cycle ({frames_per_cycle/25:.1f}s)")
    print(f"   w_daftpunk mean={w_daftpunk.mean():.2f}, w_deathstep mean={w_deathstep.mean():.2f}")

    # ================================================================
    # 7. Diffusion loop with LoRA switching per condition
    # ================================================================
    print("\n7. Diffusion loop (LoRA switching per step)...")
    steps, shift, seed = 8, 3.0, 1118  # seed from workflow
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
                # Decoder call with daftpunk LoRA
                apply_lora_deltas(model.decoder, deltas_a, sign=1.0)
                out_a = model.decoder(
                    hidden_states=xt, timestep=tt, timestep_r=tt,
                    attention_mask=attn_mask,
                    encoder_hidden_states=enc_hidden_a,
                    encoder_attention_mask=enc_mask_a,
                    context_latents=ctx_latents_a,
                )
                apply_lora_deltas(model.decoder, deltas_a, sign=-1.0)

                # Decoder call with deathstep LoRA
                apply_lora_deltas(model.decoder, deltas_b, sign=1.0)
                out_b = model.decoder(
                    hidden_states=xt, timestep=tt, timestep_r=tt,
                    attention_mask=attn_mask,
                    encoder_hidden_states=enc_hidden_b,
                    encoder_attention_mask=enc_mask_b,
                    context_latents=ctx_latents_b,
                )
                apply_lora_deltas(model.decoder, deltas_b, sign=-1.0)

            vt_a = out_a[0] if isinstance(out_a, tuple) else out_a
            vt_b = out_b[0] if isinstance(out_b, tuple) else out_b

            # Velocity difference diagnostic
            vdiff = (vt_a - vt_b).float().abs().mean().item()

            # Per-frame velocity blend
            vt = vt_a * w_a_3d + vt_b * w_b_3d

            if i == steps - 1:
                xt = xt - tc * vt
            else:
                xt = xt - vt * (tc - tn)
            print(f"   Step {i}: t={tc:.4f}->{tn:.4f} vt_diff={vdiff:.4f} vt_mean={vt.float().mean():.4f}")

    elapsed = time.time() - t0
    print(f"\n   Diffusion time: {elapsed:.2f}s ({steps/elapsed:.1f} steps/sec)")
    print(f"   Final: {ts(xt)}")

    # ================================================================
    # 8. VAE decode and save
    # ================================================================
    print("\n8. VAE decode...")
    with handler._load_model_context("vae"):
        audio_out = handler.vae.decode(xt.transpose(1, 2))
        if hasattr(audio_out, 'sample'):
            audio_out = audio_out.sample

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "temporal_blend_lora.wav")
    sf.write(out_path, audio_out.squeeze(0).detach().cpu().float().numpy().T, 48000)
    print(f"   Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
