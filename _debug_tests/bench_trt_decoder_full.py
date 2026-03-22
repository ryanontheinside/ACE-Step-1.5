"""Full benchmark: TRT decoder through DiffusionEngine with all workflow variants."""

import os, sys, time
import torch, soundfile as sf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.engine.trt.runtime import TRTDecoder
from acestep.nodes import Audio
from acestep.nodes.curve_nodes import CurveRamp, CurveWave

SOURCE_AUDIO = os.path.join(project_root, "test_audio", "new_order_confusion_60seconds.wav")
ENGINE_PATH = os.path.join(project_root, "trt_engines", "decoder_mixed_v5.engine")
OUTPUT_DIR = os.path.join(project_root, "test_output", "trt_decoder_bench")


def load_audio(path, duration=60.0):
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2, :int(duration * 48000)]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio, name):
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)


class Timer:
    def __init__(self):
        self.steps = {}
        self._t0 = None
        self._name = None

    def start(self, name):
        if self._name:
            self.stop()
        torch.cuda.synchronize()
        self._name = name
        self._t0 = time.perf_counter()

    def stop(self):
        if self._name:
            torch.cuda.synchronize()
            self.steps[self._name] = time.perf_counter() - self._t0
            self._name = None

    def report(self):
        parts = [f"{k}={v*1000:.0f}ms" for k, v in self.steps.items()]
        total = sum(self.steps.values())
        return f"{' | '.join(parts)} | total={total*1000:.0f}ms"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load TRT decoder
    print("Loading TRT decoder...")
    trt_dec = TRTDecoder(ENGINE_PATH)

    # Create session WITHOUT torch.compile (TRT replaces it)
    print("Creating session (compile_model=False)...")
    t0 = time.perf_counter()
    s = Session(project_root=project_root, compile_model=False)
    print(f"Session ready: {time.perf_counter() - t0:.1f}s")

    # Wire TRT decoder into the diffusion engine
    from acestep.engine.diffusion import DiffusionEngine
    handler = s.handler
    if not hasattr(handler, '_diffusion_engine') or handler._diffusion_engine is None:
        handler._diffusion_engine = DiffusionEngine(handler.model, trt_decoder=trt_dec)
    else:
        handler._diffusion_engine.trt_decoder = trt_dec
    # Offload PyTorch decoder to CPU (TRT replaces it)
    handler.model.decoder.to("cpu")
    torch.cuda.empty_cache()
    print("TRT decoder wired into DiffusionEngine.\n")

    # Prepare source
    print("Preparing source...")
    audio = load_audio(SOURCE_AUDIO)
    source = s.prepare_source(audio)
    T = source.latent.tensor.shape[1]
    print(f"Source ready (T={T})\n")

    cond = s.encode_text(
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )

    # Warmup (no torch.compile, just TRT warmup)
    print("Warmup...")
    t0 = time.perf_counter()
    _ = s.generate(conditioning=cond, context_latent=source.context_latent,
                   source_latent=source.latent, seed=0)
    print(f"Warmup: {(time.perf_counter()-t0)*1000:.0f}ms\n")

    results = {}

    # 1. cover_basic (3 denoise levels)
    for dn in [0.5, 0.75, 1.0]:
        name = f"cover_basic_d{int(dn*100)}"
        tm = Timer()
        tm.start("generate")
        out = s.generate(conditioning=cond, context_latent=source.context_latent,
                         source_latent=source.latent, seed=1528, denoise=dn)
        tm.start("decode")
        audio_out = s.decode(out)
        tm.stop()
        save_audio(audio_out, name)
        results[name] = tm

    # 2. velocity_scaling
    name = "velocity_scaling"
    tm = Timer()
    vel_curve = CurveRamp().execute(start=0.2, end=1.5, length=T)["curve"]
    tm.start("generate")
    out = s.generate(conditioning=cond, context_latent=source.context_latent,
                     source_latent=source.latent, seed=1528, velocity_scale=vel_curve)
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = tm

    # 3. sde_denoise_curve
    name = "sde_denoise_curve"
    tm = Timer()
    sde_curve = CurveRamp().execute(start=0.3, end=1.0, length=T)["curve"]
    tm.start("generate")
    out = s.generate(conditioning=cond, context_latent=source.context_latent,
                     source_latent=source.latent, seed=1528, steps=8, shift=3.0,
                     method="sde", sde_denoise_curve=sde_curve)
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = tm

    # 4. initial_noise_curve
    name = "initial_noise_curve"
    tm = Timer()
    noise_curve = CurveRamp().execute(start=0.3, end=1.0, length=T)["curve"]
    tm.start("generate")
    out = s.generate(conditioning=cond, context_latent=source.context_latent,
                     source_latent=source.latent, seed=1528, initial_noise_curve=noise_curve)
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = tm

    # 5. ode_noise_injection
    name = "ode_noise_injection"
    tm = Timer()
    inject_curve = CurveWave().execute(wave_type="sine", frames_per_cycle=25,
                                        amplitude=0.25, offset=0.25, length=T)["curve"]
    tm.start("generate")
    out = s.generate(conditioning=cond, context_latent=source.context_latent,
                     source_latent=source.latent, seed=1528, ode_noise_curve=inject_curve)
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = tm

    # 6. rapid-fire (same params, different seeds)
    name = "rapid_fire_x5"
    tm = Timer()
    tm.start("5x generate")
    for i in range(5):
        out = s.generate(conditioning=cond, context_latent=source.context_latent,
                         source_latent=source.latent, seed=i * 100)
    tm.start("decode_last")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, "rapid_fire_last")
    results[name] = tm

    # 7. prepare_source again
    name = "re_prepare_source"
    tm = Timer()
    tm.start("prepare_source")
    source2 = s.prepare_source(audio)
    tm.start("generate")
    out = s.generate(conditioning=cond, context_latent=source2.context_latent,
                     source_latent=source2.latent, seed=42)
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, "re_prepare")
    results[name] = tm

    # Report
    print("=" * 80)
    print(f"{'WORKFLOW':<28} BREAKDOWN")
    print("=" * 80)
    for name, tm in results.items():
        print(f"{name:<28} {tm.report()}")
    print("=" * 80)
    print(f"\nAudio saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
