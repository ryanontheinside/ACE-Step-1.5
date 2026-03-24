"""Sub-millisecond breakdown of tick() and VAE decode.

Instruments every sub-operation inside _tick_trt and _trt_vae_decode
to find where the 124ms tick and 105ms decode actually go.
Sync between each step so timings are clean.
"""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

PROJECT_ROOT = Path(__file__).parent.parent

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Latent
from acestep.nodes.vae_nodes import (
    _get_trt_vae, _get_trt_stream, _find_trt_engine,
)

TRT_ENGINES = {
    "decoder": str(PROJECT_ROOT / "trt_engines" / "decoder_mixed_b8_s1500.engine"),
    "vae_encode": str(PROJECT_ROOT / "trt_engines" / "vae_encode_fp16_max6000.engine"),
    "vae_decode": str(PROJECT_ROOT / "trt_engines" / "vae_decode_fp16_max6000.engine"),
}

print("=" * 70)
print("PROFILE: tick() and VAE decode sub-operation breakdown")
print("=" * 70)

# -----------------------------------------------------------------------
# Setup (same as diag script)
# -----------------------------------------------------------------------
print("\n[Setup] Loading model...")
session = Session(
    project_root=str(PROJECT_ROOT / "checkpoints"),
    compile_model=False,
    trt_engines=TRT_ENGINES,
)
handler = session.handler
device = handler.device
dtype = handler.dtype

print("[Setup] Encoding text conditioning...")
cond = session.encode_text(
    tags="ambient, pad, minimal",
    instruction=TASK_INSTRUCTIONS["text2music"],
    bpm=120, duration=60.0, key="C major",
)
entry = cond.to_entries()[0]

T = 1500
D_ctx = 64
ctx_lat = torch.zeros(1, T, D_ctx, device=device, dtype=dtype)
cm = torch.ones(1, T, D_ctx, device=device, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)

engine = handler._diffusion_engine
config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)
pipe = StreamPipeline(engine, config)

def make_request(seed=42):
    return SlotRequest(
        encoder_hidden_states=entry.encoder_hidden_states,
        encoder_attention_mask=entry.encoder_attention_mask,
        context_latents=context_latents,
        seed=seed,
    )

# Warm up
print("[Setup] Warming up...")
for i in range(config.infer_steps + 4):
    pipe.submit(make_request(seed=42 + i))
    pipe.tick()

# Get a real latent for decode profiling
for i in range(config.infer_steps):
    pipe.submit(make_request(seed=100 + i))
    result = pipe.tick()
    if result is not None:
        warmup_latent = result

vae_decode_path = _find_trt_engine("vae_decode_fp16_max6000.engine")
vae_entry = _get_trt_vae(vae_decode_path, torch.device(device))
shared_stream = _get_trt_stream()

# Warm VAE decode
lat_bdt = warmup_latent.transpose(1, 2)
for _ in range(3):
    lat = lat_bdt.to(device=torch.device(device), dtype=torch.float32).contiguous()
    vae_ctx = vae_entry["context"]
    vae_ctx.set_input_shape("latents", tuple(lat.shape))
    vae_ctx.set_tensor_address("latents", lat.data_ptr())
    out_shape = tuple(vae_ctx.get_tensor_shape("audio"))
    audio_buf = torch.empty(out_shape, dtype=torch.float32, device=torch.device(device))
    vae_entry["_decode_buf"] = audio_buf
    vae_ctx.set_tensor_address("audio", audio_buf.data_ptr())
    vae_ctx.execute_async_v3(shared_stream.ptr)
    shared_stream.synchronize()

torch.cuda.synchronize()
print("[Setup] Ready.\n")


# -----------------------------------------------------------------------
# Profiling helper
# -----------------------------------------------------------------------
N = 15  # iterations

class Profiler:
    def __init__(self):
        self.data = defaultdict(list)
        self._t0 = None
        self._label = None

    def start(self, label):
        torch.cuda.synchronize()
        self._label = label
        self._t0 = time.perf_counter()

    def stop(self):
        torch.cuda.synchronize()
        ms = (time.perf_counter() - self._t0) * 1000
        self.data[self._label].append(ms)
        return ms

    def report(self, title, labels):
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        total_avg = 0
        for label in labels:
            vals = self.data.get(label, [])
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            total_avg += avg
            mn, mx = min(vals), max(vals)
            print(f"  {label:40s}  avg={avg:7.2f}ms  min={mn:6.2f}  max={mx:6.2f}")
        print(f"  {'─' * 60}")
        print(f"  {'TOTAL':40s}  avg={total_avg:7.2f}ms")
        return total_avg


# -----------------------------------------------------------------------
# TICK BREAKDOWN
# -----------------------------------------------------------------------
print("=" * 70)
print("PROFILING TICK (instrumented _tick_trt + ODE step)")
print("=" * 70)

prof = Profiler()

tick_labels = [
    "tick.slot_mgmt",
    "tick.init_slot",
    "tick.collect_active",
    "tick.cat_xt",
    "tick.to_io_dtype",
    "tick.copy_hidden_states",
    "tick.fill_timesteps",
    "tick.fill_encoder_hs",
    "tick.fill_context_lat",
    "tick.rebind_addresses",
    "tick.execute_async",
    "tick.stream_sync",
    "tick.output_to_dtype",
    "tick.ode_step",
]

for iteration in range(N):
    pipe.submit(make_request(seed=200 + iteration))

    # ---- slot management ----
    prof.start("tick.slot_mgmt")
    finished = None
    for i, slot in enumerate(pipe._slots):
        if slot is not None and slot.step_idx >= len(slot.t_schedule) - 1:
            finished = slot.xt
            pipe._slots[i] = None
            break
    prof.stop()

    # ---- init slot ----
    prof.start("tick.init_slot")
    for i, slot in enumerate(pipe._slots):
        if slot is None and pipe._queue:
            req = pipe._queue.pop(0)
            pipe._slots[i] = pipe._init_slot(req)
    prof.stop()

    # ---- collect active ----
    prof.start("tick.collect_active")
    active = [
        (i, s) for i, s in enumerate(pipe._slots)
        if s is not None and s.step_idx < len(s.t_schedule) - 1
    ]
    indices, slots = zip(*active)
    B = len(slots)
    prof.stop()

    # ---- _tick_trt breakdown ----
    T_val = slots[0].xt.shape[1]
    max_L = max(s.request.encoder_hidden_states.shape[1] for s in slots)
    pipe._ensure_trt_bufs(B, T_val, max_L)
    bufs = pipe._trt_bufs
    eff_T = bufs["_eff_T"]
    pad = T_val % 2 == 1
    io_dtype = pipe._trt_io_dtype

    # cat xt
    prof.start("tick.cat_xt")
    xt_batch = torch.cat([s.xt for s in slots], dim=0)
    prof.stop()

    # to io_dtype
    prof.start("tick.to_io_dtype")
    xt_batch = xt_batch.to(io_dtype)
    prof.stop()

    # copy hidden_states
    prof.start("tick.copy_hidden_states")
    if pad:
        bufs["hidden_states"][:, :T_val, :].copy_(xt_batch)
        bufs["hidden_states"][:, T_val:, :].zero_()
    else:
        bufs["hidden_states"].copy_(xt_batch)
    prof.stop()

    # fill timesteps
    prof.start("tick.fill_timesteps")
    for i, s in enumerate(slots):
        bufs["timestep"][i] = s.t_schedule[s.step_idx].item()
    prof.stop()

    # fill encoder_hidden_states
    prof.start("tick.fill_encoder_hs")
    for i, s in enumerate(slots):
        enc = s.request.encoder_hidden_states.to(io_dtype)
        L = enc.shape[1]
        bufs["encoder_hidden_states"][i, :L, :].copy_(enc[0])
        if L < max_L:
            bufs["encoder_hidden_states"][i, L:, :].zero_()
    prof.stop()

    # fill context_latents
    prof.start("tick.fill_context_lat")
    ctx_batch = torch.cat(
        [s.request.context_latents for s in slots], dim=0
    ).to(io_dtype)
    if pad:
        bufs["context_latents"][:, :T_val, :].copy_(ctx_batch)
        bufs["context_latents"][:, T_val:, :].zero_()
    else:
        bufs["context_latents"].copy_(ctx_batch)
    prof.stop()

    # rebind addresses
    prof.start("tick.rebind_addresses")
    ctx = pipe._trt_ctx
    for name, buf in bufs.items():
        if name.startswith("_"):
            continue
        ctx.set_tensor_address(name, buf.data_ptr())
    ctx.set_tensor_address("velocity", pipe._trt_out_buf.data_ptr())
    prof.stop()

    # execute_async
    prof.start("tick.execute_async")
    ctx.execute_async_v3(pipe._trt_stream.ptr)
    prof.stop()

    # stream sync
    prof.start("tick.stream_sync")
    pipe._trt_stream.synchronize()
    prof.stop()

    # output dtype conversion
    prof.start("tick.output_to_dtype")
    out = pipe._trt_out_buf
    if pad:
        vt_batch = out[:, :T_val, :].to(pipe._dtype)
    else:
        vt_batch = out.to(pipe._dtype)
    prof.stop()

    # ODE step
    prof.start("tick.ode_step")
    for batch_idx, (slot_idx, slot) in enumerate(zip(indices, slots)):
        t_curr = slot.t_schedule[slot.step_idx].item()
        t_next = slot.t_schedule[slot.step_idx + 1].item()
        vt = vt_batch[batch_idx:batch_idx+1]
        dt = t_next - t_curr
        slot.xt = slot.xt + dt * vt
        slot.step_idx += 1
    prof.stop()

    pipe.ticks += 1

tick_total = prof.report("TICK BREAKDOWN", tick_labels)


# -----------------------------------------------------------------------
# DECODE BREAKDOWN
# -----------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("PROFILING VAE DECODE (instrumented _trt_vae_decode)")
print("=" * 70)

dec_prof = Profiler()

dec_labels = [
    "dec.transpose_input",
    "dec.to_fp32_contiguous",
    "dec.set_input_shape",
    "dec.set_input_addr",
    "dec.get_output_shape",
    "dec.alloc_or_reuse_buf",
    "dec.set_output_addr",
    "dec.execute_async",
    "dec.stream_sync",
    "dec.clone_output",
    "dec.to_cpu_float",
]

# Use the same latent each iteration (shape doesn't change)
test_latent = warmup_latent.clone()

for iteration in range(N):
    vae_ctx = vae_entry["context"]

    # transpose
    dec_prof.start("dec.transpose_input")
    lat_bdt_i = test_latent.transpose(1, 2)
    dec_prof.stop()

    # to fp32 contiguous
    dec_prof.start("dec.to_fp32_contiguous")
    lat = lat_bdt_i.to(device=torch.device(device), dtype=torch.float32).contiguous()
    dec_prof.stop()

    # set input shape
    dec_prof.start("dec.set_input_shape")
    vae_ctx.set_input_shape("latents", tuple(lat.shape))
    dec_prof.stop()

    # set input address
    dec_prof.start("dec.set_input_addr")
    vae_ctx.set_tensor_address("latents", lat.data_ptr())
    dec_prof.stop()

    # get output shape
    dec_prof.start("dec.get_output_shape")
    out_shape = tuple(vae_ctx.get_tensor_shape("audio"))
    dec_prof.stop()

    # alloc or reuse buffer
    dec_prof.start("dec.alloc_or_reuse_buf")
    cached = vae_entry.get("_decode_buf")
    if cached is not None and cached.shape == out_shape:
        audio_buf = cached
    else:
        audio_buf = torch.empty(out_shape, dtype=torch.float32, device=torch.device(device))
        vae_entry["_decode_buf"] = audio_buf
    dec_prof.stop()

    # set output address
    dec_prof.start("dec.set_output_addr")
    vae_ctx.set_tensor_address("audio", audio_buf.data_ptr())
    dec_prof.stop()

    # execute async
    dec_prof.start("dec.execute_async")
    vae_ctx.execute_async_v3(shared_stream.ptr)
    dec_prof.stop()

    # stream sync
    dec_prof.start("dec.stream_sync")
    shared_stream.synchronize()
    dec_prof.stop()

    # clone output
    dec_prof.start("dec.clone_output")
    result_audio = audio_buf.clone()
    dec_prof.stop()

    # to cpu float (what session.decode does next: .detach().cpu().float())
    dec_prof.start("dec.to_cpu_float")
    wav = result_audio.detach().cpu().float()
    dec_prof.stop()

dec_total = dec_prof.report("DECODE BREAKDOWN", dec_labels)


# -----------------------------------------------------------------------
# RAW ENGINE ISOLATION: just execute + sync, nothing else
# -----------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("RAW ENGINE ISOLATION (execute_async + sync only, no prep)")
print("=" * 70)

# DiT engine: buffers are already filled from the last instrumented tick.
# Just slam execute + sync repeatedly.
print("\n  DiT decoder (buffers pre-filled, B=8, T=1500):")
dit_raw = []
for _ in range(N):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pipe._trt_ctx.execute_async_v3(pipe._trt_stream.ptr)
    pipe._trt_stream.synchronize()
    dit_raw.append((time.perf_counter() - t0) * 1000)

avg_dit_raw = sum(dit_raw) / len(dit_raw)
print(f"    avg={avg_dit_raw:.2f}ms  min={min(dit_raw):.2f}  max={max(dit_raw):.2f}")

# VAE decoder: buffers already filled from last instrumented decode.
print("\n  VAE decoder (buffers pre-filled, 60s audio):")
vae_raw = []
for _ in range(N):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vae_entry["context"].execute_async_v3(shared_stream.ptr)
    shared_stream.synchronize()
    vae_raw.append((time.perf_counter() - t0) * 1000)

avg_vae_raw = sum(vae_raw) / len(vae_raw)
print(f"    avg={avg_vae_raw:.2f}ms  min={min(vae_raw):.2f}  max={max(vae_raw):.2f}")

# Both sequential: raw ceiling
print(f"\n  Sequential raw ceiling: {avg_dit_raw + avg_vae_raw:.2f}ms")
print(f"  vs instrumented total:  {tick_total + dec_total:.1f}ms")
print(f"  Overhead outside TRT:   {tick_total + dec_total - avg_dit_raw - avg_vae_raw:.1f}ms")


# -----------------------------------------------------------------------
# OVERHEAD: measure sync cost itself
# -----------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("SYNC OVERHEAD (cost of torch.cuda.synchronize itself)")
print("=" * 70)

sync_times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    sync_times.append((time.perf_counter() - t0) * 1000)

avg_sync = sum(sync_times) / len(sync_times)
print(f"  avg={avg_sync:.4f}ms  min={min(sync_times):.4f}  max={max(sync_times):.4f}")
print(f"  With {len(tick_labels)} tick steps + {len(dec_labels)} decode steps = "
      f"{len(tick_labels) + len(dec_labels)} syncs")
print(f"  Estimated profiling overhead: {(len(tick_labels) + len(dec_labels)) * avg_sync:.2f}ms")


# -----------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------
print(f"\n\n{'=' * 70}")
print("FINAL SUMMARY")
print("=" * 70)

# Find top contributors
all_items = []
for label in tick_labels:
    vals = prof.data.get(label, [])
    if vals:
        all_items.append((label, sum(vals)/len(vals)))
for label in dec_labels:
    vals = dec_prof.data.get(label, [])
    if vals:
        all_items.append((label, sum(vals)/len(vals)))

all_items.sort(key=lambda x: x[1], reverse=True)

print(f"\n  Top contributors (tick + decode combined):")
cumulative = 0
for label, avg in all_items:
    cumulative += avg
    pct = avg / (tick_total + dec_total) * 100
    cum_pct = cumulative / (tick_total + dec_total) * 100
    bar = "#" * int(pct / 2)
    print(f"    {avg:7.2f}ms ({pct:4.1f}%) {bar:30s}  {label}")
    if cum_pct > 95:
        print(f"    ... ({100-cum_pct:.1f}% remaining in {len(all_items) - all_items.index((label, avg)) - 1} items)")
        break

print(f"\n  Tick total:   {tick_total:.1f}ms")
print(f"  Decode total: {dec_total:.1f}ms")
print(f"  Combined:     {tick_total + dec_total:.1f}ms")

combined = tick_total + dec_total
raw_combined = avg_dit_raw + avg_vae_raw
overhead_ms = combined - raw_combined

print(f"\n  Raw TRT execution:    {raw_combined:.1f}ms  (DiT={avg_dit_raw:.1f} + VAE={avg_vae_raw:.1f})")
print(f"  Instrumented total:   {combined:.1f}ms")
print(f"  Non-TRT overhead:     {overhead_ms:.1f}ms  ({overhead_ms/combined*100:.0f}%)")

print(f"\n{'=' * 70}")
