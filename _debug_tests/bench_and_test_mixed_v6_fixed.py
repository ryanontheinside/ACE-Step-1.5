"""Benchmark and quality test mixed-precision v6 engine with correct I/O dtypes."""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import soundfile as sf
torch.set_grad_enabled(False)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGINE_PATH = os.path.join(PROJECT_ROOT, "trt_engines", "decoder_mixed_b8_s1500.engine")
SOURCE_AUDIO = os.path.join(PROJECT_ROOT, "test_audio", "new_order_confusion_60seconds.wav")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "_debug_tests", "stream_output")

device = torch.device("cuda")
T = 1500
L = 200

# ---------------------------------------------------------------
# Benchmark with fp16 I/O
# ---------------------------------------------------------------
print("=" * 60)
print("Benchmark: mixed-precision v6 engine (fp16 I/O)")
print("=" * 60)

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from acestep.nodes.vae_nodes import _get_trt_stream

engine = engine_from_bytes(bytes_from_path(ENGINE_PATH))
ctx = engine.create_execution_context()
stream = _get_trt_stream()


def bench(B, warmup=5, iters=15):
    eff_T = T + 1 if T % 2 == 1 else T
    bufs = {
        "hidden_states": torch.randn(B, eff_T, 64, dtype=torch.float16, device=device),
        "timestep": torch.rand(B, dtype=torch.float32, device=device),
        "encoder_hidden_states": torch.randn(B, L, 2048, dtype=torch.float16, device=device),
        "context_latents": torch.randn(B, eff_T, 128, dtype=torch.float16, device=device),
    }
    for name, buf in bufs.items():
        ctx.set_input_shape(name, tuple(buf.shape))
        ctx.set_tensor_address(name, buf.data_ptr())
    out_shape = tuple(ctx.get_tensor_shape("velocity"))
    out_buf = torch.empty(out_shape, dtype=torch.float16, device=device)
    ctx.set_tensor_address("velocity", out_buf.data_ptr())

    for _ in range(warmup):
        for name, buf in bufs.items():
            ctx.set_tensor_address(name, buf.data_ptr())
        ctx.set_tensor_address("velocity", out_buf.data_ptr())
        ctx.execute_async_v3(stream.ptr)
        stream.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for name, buf in bufs.items():
            ctx.set_tensor_address(name, buf.data_ptr())
        ctx.set_tensor_address("velocity", out_buf.data_ptr())
        ctx.execute_async_v3(stream.ptr)
        stream.synchronize()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg = sum(times) / len(times)
    med = sorted(times)[len(times) // 2]
    return avg, med


print(f"\nT={T} (60s), L={L}")
print(f"{'Batch':>6s} {'Avg':>8s} {'Median':>8s}")
print("-" * 30)
results = {}
for B in [1, 4, 8]:
    avg, med = bench(B)
    results[B] = avg
    print(f"{B:6d} {avg:7.1f}ms {med:7.1f}ms")
print(f"\nB8/B1 ratio: {results[8]/results[1]:.2f}x")

del engine, ctx
torch.cuda.empty_cache()

# ---------------------------------------------------------------
# Quality test: need to update DiffusionEngine TRT path to use fp16 I/O
# We'll do it manually since the engine expects fp16
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Quality test: cover workflow (manual TRT integration)")
print("=" * 60)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.engine.diffusion import DiffusionEngine, DiffusionConfig
from acestep.engine.conditions import PreparedCondition, ConditionSet
from acestep.nodes.types import Audio, Latent

print("Loading model...")
session = Session(
    project_root=os.path.join(PROJECT_ROOT, "checkpoints"),
    compile_model=False,
    use_flash_attention=True,
)
handler = session.handler
dev = handler.device
dtype = handler.dtype

print("Loading source audio...")
data, sr = sf.read(SOURCE_AUDIO, dtype="float32")
waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
if sr != 48000:
    import torchaudio
    waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
waveform = waveform[:2, :int(60.0 * 48000)]
pool = 1920 * 5
rem = waveform.shape[-1] % pool
if rem:
    waveform = waveform[:, :waveform.shape[-1] - rem]
audio = Audio(waveform=waveform, sample_rate=48000)

print("Preparing source...")
source = session.prepare_source(audio)

print("Encoding conditioning...")
cond = session.encode_text(
    tags="deathstep, heavy bass, dark atmosphere",
    instruction=TASK_INSTRUCTIONS["cover"],
    refer_latent=source.latent,
    bpm=136, duration=60.0, key="G# minor",
)

# Manual TRT generation loop with correct fp16 I/O
engine2 = engine_from_bytes(bytes_from_path(ENGINE_PATH))
trt_ctx = engine2.create_execution_context()
trt_stream = _get_trt_stream()

entry = cond.to_entries()[0]
ctx_lat = source.context_latent.tensor.to(device=dev, dtype=dtype)
D = ctx_lat.shape[2]
T_src = ctx_lat.shape[1]
cm = torch.ones(1, T_src, D, device=dev, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)
source_latents = source.latent.tensor.to(device=dev, dtype=dtype)

diffusion_engine = DiffusionEngine(handler.model)
config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)

for dn in [0.5, 0.75, 1.0]:
    print(f"\nGenerating denoise={dn}...")
    cfg = DiffusionConfig(infer_steps=8, shift=3.0, seed=1528, noise_on_cpu=True, denoise=dn)
    t_schedule = diffusion_engine._build_timestep_schedule(cfg, dev, dtype)

    # Noise
    torch.manual_seed(1528)
    D_noise = context_latents.shape[-1] // 2
    noise_bdt = torch.randn(1, D_noise, T_src, device="cpu", dtype=torch.float32)
    noise = noise_bdt.movedim(-1, -2).to(device=dev, dtype=dtype)

    t_start = t_schedule[0].item()
    if dn < 1.0:
        xt = t_start * noise + (1.0 - t_start) * source_latents
    else:
        xt = noise.clone()

    # TRT buffers (fp16 for data, fp32 for timestep)
    eff_T = T_src + 1 if T_src % 2 == 1 else T_src
    pad = T_src % 2 == 1

    enc_hs = entry.encoder_hidden_states.half().contiguous()
    ctx_trt = context_latents.half().contiguous()
    L_enc = enc_hs.shape[1]

    bufs = {
        "hidden_states": torch.empty(1, eff_T, 64, dtype=torch.float16, device=dev),
        "timestep": torch.empty(1, dtype=torch.float32, device=dev),
        "encoder_hidden_states": torch.empty(1, L_enc, 2048, dtype=torch.float16, device=dev),
        "context_latents": torch.empty(1, eff_T, 128, dtype=torch.float16, device=dev),
    }
    for name, buf in bufs.items():
        trt_ctx.set_input_shape(name, tuple(buf.shape))
        trt_ctx.set_tensor_address(name, buf.data_ptr())
    out_shape = tuple(trt_ctx.get_tensor_shape("velocity"))
    out_buf = torch.empty(out_shape, dtype=torch.float16, device=dev)
    trt_ctx.set_tensor_address("velocity", out_buf.data_ptr())

    # Copy constant buffers
    bufs["encoder_hidden_states"].copy_(enc_hs)
    if pad:
        bufs["context_latents"][:, :T_src, :].copy_(ctx_trt)
        bufs["context_latents"][:, T_src:, :].zero_()
    else:
        bufs["context_latents"].copy_(ctx_trt)

    t0 = time.time()
    for i in range(cfg.infer_steps):
        t_curr = t_schedule[i].item()
        t_next = t_schedule[i + 1].item()
        dt = t_next - t_curr

        if pad:
            bufs["hidden_states"][:, :T_src, :].copy_(xt.half())
            bufs["hidden_states"][:, T_src:, :].zero_()
        else:
            bufs["hidden_states"].copy_(xt.half())
        bufs["timestep"].fill_(t_curr)

        for name, buf in bufs.items():
            trt_ctx.set_tensor_address(name, buf.data_ptr())
        trt_ctx.set_tensor_address("velocity", out_buf.data_ptr())
        trt_ctx.execute_async_v3(trt_stream.ptr)
        trt_stream.synchronize()

        vt = out_buf[:, :T_src, :].to(dtype)
        xt = xt + dt * vt

    gen_ms = (time.time() - t0) * 1000

    audio_out = session.decode(Latent(tensor=xt))
    wav = audio_out.waveform.detach().cpu().float().squeeze(0)
    path = os.path.join(OUTPUT_DIR, f"mixed_v6_fixed_d{int(dn*100)}.wav")
    sf.write(path, wav.numpy().T, 48000, format="WAV")
    print(f"  {gen_ms:.0f}ms -> {os.path.basename(path)}")

print(f"\nFiles saved. Compare mixed_v6_fixed_*.wav to test_output/workflows/cover_denoise_*.wav")
