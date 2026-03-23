"""Benchmark decoder: eager vs torch.compile at batch 1/4/8."""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from acestep.engine.session import Session

print("Loading model...")
session = Session(
    project_root=os.path.join(PROJECT_ROOT, "checkpoints"),
    compile_model=False,
    use_flash_attention=True,
)
handler = session.handler
device = torch.device(handler.device)
dtype = handler.dtype
decoder = handler.model.decoder

T = 1500
L = 200


def make_inputs(B):
    return dict(
        hidden_states=torch.randn(B, T, 64, device=device, dtype=dtype),
        timestep=torch.rand(B, device=device, dtype=dtype),
        timestep_r=torch.rand(B, device=device, dtype=dtype),
        attention_mask=torch.ones(B, T, device=device, dtype=dtype),
        encoder_hidden_states=torch.randn(B, L, 2048, device=device, dtype=dtype),
        encoder_attention_mask=torch.ones(B, L, device=device, dtype=dtype),
        context_latents=torch.randn(B, T, 128, device=device, dtype=dtype),
        use_cache=False,
        past_key_values=None,
    )


def bench(fn, B, warmup=3, iters=10):
    inputs = make_inputs(B)
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(**inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def report(label, times):
    avg = sum(times) / len(times)
    med = sorted(times)[len(times) // 2]
    mn = min(times)
    print(f"  {label:30s}  avg={avg:6.1f}ms  median={med:6.1f}ms  min={mn:6.1f}ms")
    return avg


# Compile the decoder
print("Compiling decoder (dynamic=True, max-autotune-no-cudagraphs)...")
t0 = time.time()
compiled_decoder = torch.compile(
    decoder,
    backend="inductor",
    dynamic=True,
    mode="max-autotune-no-cudagraphs",
)

# Warmup compile at each batch size
for B in [1, 4, 8]:
    print(f"  Warmup compile B={B}...")
    inputs = make_inputs(B)
    compiled_decoder(**inputs)
    torch.cuda.synchronize()
print(f"Compile done in {time.time() - t0:.1f}s\n")

print(f"T={T} (60s), L={L}")
print("=" * 70)

for B in [1, 4, 8]:
    print(f"\n--- Batch size = {B} ---")
    eager_avg = report("Eager (flash_attn)", bench(decoder, B))
    compiled_avg = report("torch.compile", bench(compiled_decoder, B))
    print(f"  {'speedup':30s}  {eager_avg/compiled_avg:.2f}x")

print()
