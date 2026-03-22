"""Full pipeline test with polygraphy engine loading.

Tests ODE, SDE with sde_denoise_curve, velocity_scale, and initial_noise_curve.
Saves audio output and reports per-cycle timing.
"""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os, sys, time, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import soundfile as sf, torchaudio
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy import cuda as pg_cuda

from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionEngine, DiffusionConfig
from acestep.constants import TASK_INSTRUCTIONS
from acestep.nodes import Audio, Latent

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_polygraphy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared polygraphy stream
stream = pg_cuda.Stream()

# Load ALL engines via polygraphy
dec_engine = engine_from_bytes(bytes_from_path("trt_engines/decoder_mixed_v5.engine"))
dec_ctx = dec_engine.create_execution_context()
enc_engine = engine_from_bytes(bytes_from_path("trt_engines/vae_encode_fp16_max6000.engine"))
enc_ctx = enc_engine.create_execution_context()
vae_engine = engine_from_bytes(bytes_from_path("trt_engines/vae_decode_fp16_max6000.engine"))
vae_ctx = vae_engine.create_execution_context()

# Session (no TRT engines loaded through session)
s = Session(project_root=".", compile_model=False)
s.handler.model.decoder.to("cpu"); torch.cuda.empty_cache()

# Load and encode source audio
data, sr = sf.read("test_audio/new_order_confusion_60seconds.wav", dtype="float32")
waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
if sr != 48000: waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
waveform = waveform[:2, :int(60.0 * 48000)]
pool = 1920 * 5; rem = waveform.shape[-1] % pool
if rem: waveform = waveform[:, :waveform.shape[-1] - rem]

audio_inp = waveform.unsqueeze(0).float().cuda().contiguous()
enc_ctx.set_input_shape("audio", tuple(audio_inp.shape))
enc_ctx.set_tensor_address("audio", audio_inp.data_ptr())
moments_buf = torch.empty(tuple(enc_ctx.get_tensor_shape("moments")), dtype=torch.float32, device="cuda")
enc_ctx.set_tensor_address("moments", moments_buf.data_ptr())
enc_ctx.execute_async_v3(stream.ptr); stream.synchronize()
mean, logvar = moments_buf.chunk(2, dim=1)
source_lat_bdt = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
source_latent = Latent(tensor=source_lat_bdt.transpose(1, 2).to(torch.bfloat16))

hints = s.extract_hints(source_latent)
context_latent = s.hints_to_latent(hints)
source = PreparedSource(latent=source_latent, hints=hints, context_latent=context_latent)

cond = s.encode_text(
    tags="deathstep", instruction=TASK_INSTRUCTIONS["cover"],
    refer_latent=source.latent, bpm=136, duration=60.0, key="G# minor",
)

# Prepare decoder buffers
T = source_latent.tensor.shape[1]
entries = cond.to_entries()
enc_hs = entries[0].encoder_hidden_states.float().contiguous()
L = enc_hs.shape[1]
ctx_lat = source.context_latent.tensor.to(device="cuda", dtype=torch.bfloat16)
cm = torch.ones_like(ctx_lat)
context_latents_fp32 = torch.cat([ctx_lat, cm], dim=-1).float().contiguous()
src_lat = source.latent.tensor.to(device="cuda", dtype=torch.bfloat16)

dec_bufs = {
    "hidden_states": torch.empty(1, T, 64, dtype=torch.float32, device="cuda"),
    "timestep": torch.empty(1, dtype=torch.float32, device="cuda"),
    "encoder_hidden_states": torch.empty(1, L, 2048, dtype=torch.float32, device="cuda"),
    "context_latents": torch.empty(1, T, 128, dtype=torch.float32, device="cuda"),
}
for name, buf in dec_bufs.items():
    dec_ctx.set_input_shape(name, tuple(buf.shape))
    dec_ctx.set_tensor_address(name, buf.data_ptr())
dec_out = torch.empty(tuple(dec_ctx.get_tensor_shape("velocity")), dtype=torch.float32, device="cuda")
dec_ctx.set_tensor_address("velocity", dec_out.data_ptr())

# Pre-allocate VAE decode output
vae_lat_buf = torch.empty(1, 64, T, dtype=torch.float32, device="cuda")
vae_ctx.set_input_shape("latents", tuple(vae_lat_buf.shape))
vae_ctx.set_tensor_address("latents", vae_lat_buf.data_ptr())
vae_out = torch.empty(tuple(vae_ctx.get_tensor_shape("audio")), dtype=torch.float32, device="cuda")
vae_ctx.set_tensor_address("audio", vae_out.data_ptr())

dummy_eng = DiffusionEngine(s.handler.model)

def _normalize_curve(curve):
    if curve.ndim == 1: return curve.unsqueeze(0).unsqueeze(-1)
    if curve.ndim == 2: return curve.unsqueeze(-1)
    return curve

def _dec_step(xt, t_val):
    """One TRT decoder step."""
    dec_bufs["hidden_states"].copy_(xt.float())
    dec_bufs["timestep"].fill_(t_val)
    dec_bufs["encoder_hidden_states"].copy_(enc_hs)
    dec_bufs["context_latents"].copy_(context_latents_fp32)
    for name, buf in dec_bufs.items():
        dec_ctx.set_tensor_address(name, buf.data_ptr())
    dec_ctx.set_tensor_address("velocity", dec_out.data_ptr())
    dec_ctx.execute_async_v3(stream.ptr)
    stream.synchronize()
    return dec_out.bfloat16()


def trt_generate_ode(seed, denoise=0.75, velocity_scale=None, initial_noise_curve=None,
                     ode_noise_curve=None):
    config = DiffusionConfig(infer_steps=8, shift=3.0, seed=seed, denoise=denoise)
    t_schedule = dummy_eng._build_timestep_schedule(config, torch.device("cuda"), torch.bfloat16).cpu()
    infer_steps = len(t_schedule) - 1
    t_start = t_schedule[0].item()

    noise = s.handler.model.prepare_noise(context_latents_fp32.bfloat16(), seed)

    if initial_noise_curve is not None:
        curve = _normalize_curve(initial_noise_curve).to(device="cuda", dtype=torch.bfloat16)
        xt = curve * noise + (1.0 - curve) * src_lat
    elif denoise < 1.0:
        xt = t_start * noise + (1.0 - t_start) * src_lat
    else:
        xt = noise.clone()

    for i in range(infer_steps):
        t_curr = t_schedule[i].item()
        t_next = t_schedule[i + 1].item()
        dt = t_next - t_curr

        vt = _dec_step(xt, t_curr)
        if velocity_scale is not None:
            vt = vt * _normalize_curve(velocity_scale).to(device="cuda", dtype=torch.bfloat16)
        xt = xt + dt * vt
        if ode_noise_curve is not None and i < infer_steps - 1 and t_next > 0:
            xt = xt + torch.randn_like(xt) * _normalize_curve(ode_noise_curve).to(device="cuda", dtype=torch.bfloat16) * t_next

    return xt


def trt_generate_sde(seed, denoise=0.75, sde_denoise_curve=None):
    config = DiffusionConfig(infer_steps=8, shift=3.0, seed=seed, denoise=denoise, infer_method="sde")
    t_schedule = dummy_eng._build_timestep_schedule(config, torch.device("cuda"), torch.bfloat16).cpu()
    infer_steps = len(t_schedule) - 1
    t_start = t_schedule[0].item()

    noise = s.handler.model.prepare_noise(context_latents_fp32.bfloat16(), seed)
    xt = t_start * noise + (1.0 - t_start) * src_lat

    if sde_denoise_curve is not None:
        sdc = _normalize_curve(sde_denoise_curve).to(device="cuda", dtype=torch.bfloat16)
    else:
        sdc = torch.ones(1, 1, 1, device="cuda", dtype=torch.bfloat16)

    for i in range(infer_steps):
        t_curr = t_schedule[i].item()
        t_next = t_schedule[i + 1].item()

        vt = _dec_step(xt, t_curr)

        # x0 prediction
        x0_pred = xt - vt * t_curr
        # Re-noise with sde_denoise_curve blending
        sde_noise = torch.randn_like(xt)
        xt_full = t_next * sde_noise + (1.0 - t_next) * x0_pred
        xt_source = t_next * sde_noise + (1.0 - t_next) * src_lat
        xt = sdc * xt_full + (1.0 - sdc) * xt_source

    return xt


def trt_decode(latent_btd):
    lat_bdt = latent_btd.transpose(1, 2).float().contiguous()
    vae_ctx.set_input_shape("latents", tuple(lat_bdt.shape))
    vae_ctx.set_tensor_address("latents", lat_bdt.data_ptr())
    vae_ctx.set_tensor_address("audio", vae_out.data_ptr())
    vae_ctx.execute_async_v3(stream.ptr); stream.synchronize()
    return vae_out.clone()


def save(audio_tensor, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    sf.write(path, audio_tensor[0].cpu().numpy().T, 48000)


# Warmup
_ = trt_generate_ode(99)
_ = trt_decode(torch.randn(1, T, 64, device="cuda", dtype=torch.bfloat16))

# === ODE baseline (8 repeats) ===
print("=== ODE baseline (denoise=0.75) ===\n")
for i in range(8):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = trt_generate_ode(i, denoise=0.75)
    gen_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    audio = trt_decode(out)
    dec_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cycle {i+1}: generate={gen_ms:6.0f}ms  decode={dec_ms:6.0f}ms  total={gen_ms+dec_ms:6.0f}ms")
save(audio, "ode_baseline")

# === ODE + velocity_scale ramp ===
print("\n=== ODE + velocity_scale ramp (0.2 -> 1.5) ===\n")
vel_curve = torch.linspace(0.2, 1.5, T, device="cuda")
for i in range(4):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = trt_generate_ode(i, denoise=0.75, velocity_scale=vel_curve)
    gen_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    audio = trt_decode(out)
    dec_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cycle {i+1}: generate={gen_ms:6.0f}ms  decode={dec_ms:6.0f}ms  total={gen_ms+dec_ms:6.0f}ms")
save(audio, "ode_velocity_ramp")

# === SDE + sde_denoise_curve ramp (0.3 -> 1.0) ===
print("\n=== SDE + sde_denoise_curve ramp (0.3 -> 1.0) ===\n")
sde_curve = torch.linspace(0.3, 1.0, T, device="cuda")
for i in range(8):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = trt_generate_sde(i, denoise=0.75, sde_denoise_curve=sde_curve)
    gen_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    audio = trt_decode(out)
    dec_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cycle {i+1}: generate={gen_ms:6.0f}ms  decode={dec_ms:6.0f}ms  total={gen_ms+dec_ms:6.0f}ms")
save(audio, "sde_curve_ramp")

# === SDE + sde_denoise_curve constant 0.5 ===
print("\n=== SDE + sde_denoise_curve constant 0.5 ===\n")
sde_const = torch.full((T,), 0.5, device="cuda")
for i in range(4):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = trt_generate_sde(100 + i, denoise=0.75, sde_denoise_curve=sde_const)
    gen_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    audio = trt_decode(out)
    dec_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cycle {i+1}: generate={gen_ms:6.0f}ms  decode={dec_ms:6.0f}ms  total={gen_ms+dec_ms:6.0f}ms")
save(audio, "sde_curve_constant")

# === SDE no curve (standard stochastic) ===
print("\n=== SDE no curve (standard) ===\n")
for i in range(4):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = trt_generate_sde(200 + i, denoise=0.75)
    gen_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    audio = trt_decode(out)
    dec_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cycle {i+1}: generate={gen_ms:6.0f}ms  decode={dec_ms:6.0f}ms  total={gen_ms+dec_ms:6.0f}ms")
save(audio, "sde_standard")

# === ODE + initial_noise_curve ramp ===
print("\n=== ODE + initial_noise_curve ramp (0.3 -> 0.9) ===\n")
noise_curve = torch.linspace(0.3, 0.9, T, device="cuda")
for i in range(4):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = trt_generate_ode(300 + i, denoise=0.75, initial_noise_curve=noise_curve)
    gen_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    audio = trt_decode(out)
    dec_ms = (time.perf_counter() - t1) * 1000
    print(f"  Cycle {i+1}: generate={gen_ms:6.0f}ms  decode={dec_ms:6.0f}ms  total={gen_ms+dec_ms:6.0f}ms")
save(audio, "ode_initial_noise_ramp")

print(f"\nAudio saved to: {OUTPUT_DIR}")
