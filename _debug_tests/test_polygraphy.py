"""Test: use polygraphy (StreamDiffusion's exact stack) for all TRT engines."""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os, sys, time, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy import cuda as pg_cuda

# ONE shared polygraphy stream (exactly like StreamDiffusion)
stream = pg_cuda.Stream()
print(f"Polygraphy stream: ptr={stream.ptr}")

# Load decoder engine via polygraphy
dec_engine = engine_from_bytes(bytes_from_path("trt_engines/decoder_mixed_v5.engine"))
dec_ctx = dec_engine.create_execution_context()

# Pre-allocate decoder buffers
T, L = 1500, 70
dec_bufs = {
    "hidden_states": torch.empty(1, T, 64, dtype=torch.float32, device="cuda"),
    "timestep": torch.empty(1, dtype=torch.float32, device="cuda"),
    "encoder_hidden_states": torch.empty(1, L, 2048, dtype=torch.float32, device="cuda"),
    "context_latents": torch.empty(1, T, 128, dtype=torch.float32, device="cuda"),
}
for name, buf in dec_bufs.items():
    dec_ctx.set_input_shape(name, tuple(buf.shape))
    dec_ctx.set_tensor_address(name, buf.data_ptr())
out_shape = tuple(dec_ctx.get_tensor_shape("velocity"))
dec_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
dec_ctx.set_tensor_address("velocity", dec_out.data_ptr())

# Bench decoder alone (no VAE loaded)
def bench_dec(label, n=8):
    hs = torch.randn(1, T, 64, device="cuda")
    ts = torch.full((1,), 0.5, device="cuda")
    enc = torch.randn(1, L, 2048, device="cuda")
    cl = torch.randn(1, T, 128, device="cuda")
    # warmup
    for _ in range(2):
        dec_bufs["hidden_states"].copy_(hs)
        dec_bufs["timestep"].copy_(ts)
        dec_bufs["encoder_hidden_states"].copy_(enc)
        dec_bufs["context_latents"].copy_(cl)
        for name, buf in dec_bufs.items():
            dec_ctx.set_tensor_address(name, buf.data_ptr())
        dec_ctx.set_tensor_address("velocity", dec_out.data_ptr())
        dec_ctx.execute_async_v3(stream.ptr)
        stream.synchronize()
    # timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        dec_bufs["hidden_states"].copy_(hs)
        dec_bufs["timestep"].copy_(ts)
        dec_bufs["encoder_hidden_states"].copy_(enc)
        dec_bufs["context_latents"].copy_(cl)
        for name, buf in dec_bufs.items():
            dec_ctx.set_tensor_address(name, buf.data_ptr())
        dec_ctx.set_tensor_address("velocity", dec_out.data_ptr())
        dec_ctx.execute_async_v3(stream.ptr)
        stream.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label}: {ms:.0f}ms total, {ms/n:.1f}ms/step")

print("\n=== Decoder alone (no VAE engine) ===")
bench_dec("baseline")

# Now load VAE decode engine via polygraphy (same stream)
print("\n=== Loading VAE decode engine via polygraphy ===")
vae_engine = engine_from_bytes(bytes_from_path("trt_engines/vae_decode_fp16_max6000.engine"))
vae_ctx = vae_engine.create_execution_context()
print("  VAE engine loaded")

bench_dec("after VAE engine load")

# Execute VAE once
print("\n=== Executing VAE decode once ===")
dummy_lat = torch.randn(1, 64, T, device="cuda", dtype=torch.float32)
vae_ctx.set_input_shape("latents", tuple(dummy_lat.shape))
vae_ctx.set_tensor_address("latents", dummy_lat.data_ptr())
vae_out_shape = tuple(vae_ctx.get_tensor_shape("audio"))
vae_out = torch.empty(vae_out_shape, dtype=torch.float32, device="cuda")
vae_ctx.set_tensor_address("audio", vae_out.data_ptr())
vae_ctx.execute_async_v3(stream.ptr)
stream.synchronize()
print(f"  VAE executed, output shape: {vae_out_shape}")

bench_dec("after VAE execution")
