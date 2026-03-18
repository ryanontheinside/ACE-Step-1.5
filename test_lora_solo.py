#!/usr/bin/env python3
"""
Generate a solo cover with daftpunk LoRA for use as x0 blend target.
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
LORA_PATH = r"C:\_dev\models\comfyui_models\loras\acestep1.5\daftpunkstyle1200.safetensors"
LORA_STRENGTH = 1.3


def ts(t):
    if t is None: return "None"
    f = t.detach().float()
    return f"shape={list(t.shape)} mean={f.mean():.4f} std={f.std():.4f}"


def precompute_lora_deltas(lora_path, strength, device, dtype):
    raw = load_file(lora_path)
    pairs = {}
    for key, tensor in raw.items():
        parts = key.replace("base_model.model.", "")
        if ".lora_A.weight" in parts:
            param_name = parts.replace(".lora_A.weight", ".weight")
            pairs.setdefault(param_name, {})["A"] = tensor
        elif ".lora_B.weight" in parts:
            param_name = parts.replace(".lora_B.weight", ".weight")
            pairs.setdefault(param_name, {})["B"] = tensor
    deltas = {}
    for param_name, ab in pairs.items():
        A = ab["A"].to(device=device, dtype=dtype)
        B = ab["B"].to(device=device, dtype=dtype)
        deltas[param_name] = strength * (B @ A)
    return deltas


def apply_lora_deltas(decoder, deltas, sign=1.0):
    decoder_params = dict(decoder.named_parameters())
    for param_name, delta in deltas.items():
        if param_name in decoder_params:
            decoder_params[param_name].data.add_(delta, alpha=sign)


def main():
    print("=" * 70)
    print("Solo daftpunk LoRA cover (for x0 blend target)")
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

    # 1. VAE encode
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

    # 2. Semantic hints
    print("\n2. Extract semantic hints...")
    with handler._load_model_context("model"):
        src_on_device = src_latent.to(device=device, dtype=dtype)
        with torch.no_grad():
            quantized, _ = model.tokenizer.tokenize(src_on_device)
            lm_hints = model.detokenizer(quantized)
        lm_hints = lm_hints[:, :T, :]

    # 3. Encode text + LoRA
    print("\n3. Encode text and load LoRA...")
    instruction = "Generate audio semantic tokens based on the given conditions:"
    bpm, duration, keyscale, timesig = 136, 60, "G# minor", 4
    meta_cap = f"- bpm: {bpm}\n- timesignature: {timesig}\n- keyscale: {keyscale}\n- duration: {duration}\n"
    caption = "daft punk style"
    text_prompt = f"# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{meta_cap}<|endoftext|>\n"
    lyrics_prompt = "# Languages\nen\n\n# Lyric\n<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        tokens = handler.text_tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
        text_hidden = handler.infer_text_embeddings(tokens["input_ids"].to(device))
        text_mask = tokens["attention_mask"].to(device).bool()
        lyric_tokens = handler.text_tokenizer(lyrics_prompt, return_tensors="pt", add_special_tokens=False)
        lyric_hidden = handler.infer_lyric_embeddings(lyric_tokens["input_ids"].to(device))
        lyric_mask = torch.ones(lyric_hidden.shape[:2], device=device, dtype=torch.bool)

    deltas = precompute_lora_deltas(LORA_PATH, LORA_STRENGTH, device, dtype)
    print(f"   LoRA: {len(deltas)} params, strength={LORA_STRENGTH}")

    # 4. Build condition with LoRA applied
    print("\n4. prepare_condition (with LoRA)...")
    refer_packed = src_latent.clone()
    refer_order_mask = torch.zeros(1, device=device, dtype=torch.long)
    chunk_masks = torch.ones(1, T, 64, device=device, dtype=dtype)
    is_covers = torch.BoolTensor([True]).to(device)
    handler._ensure_silence_latent_on_device()

    with handler._load_model_context("model"):
        apply_lora_deltas(model.decoder, deltas, sign=1.0)
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

    # 5. Diffusion with LoRA
    print("\n5. Diffusion loop (with LoRA)...")
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
            if i == steps - 1:
                xt = xt - tc * vt
            else:
                xt = xt - vt * (tc - tn)
            print(f"   Step {i}: t={tc:.4f}->{tn:.4f}")

        # Remove LoRA
        apply_lora_deltas(model.decoder, deltas, sign=-1.0)

    elapsed = time.time() - t0
    print(f"\n   Diffusion time: {elapsed:.2f}s")
    print(f"   Final: {ts(xt)}")

    # 6. Save latent for use as target
    latent_path = os.path.join(OUTPUT_DIR, "daftpunk_solo_latent.pt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(xt.cpu(), latent_path)
    print(f"   Saved latent: {latent_path}")

    # 7. VAE decode and save audio
    print("\n7. VAE decode...")
    with handler._load_model_context("vae"):
        audio_out = handler.vae.decode(xt.transpose(1, 2))
        if hasattr(audio_out, 'sample'):
            audio_out = audio_out.sample
        audio_np = audio_out.squeeze(0).detach().cpu().float().numpy().T

    path = os.path.join(OUTPUT_DIR, "daftpunk_solo.wav")
    sf.write(path, audio_np, 48000)
    print(f"   Saved audio: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
