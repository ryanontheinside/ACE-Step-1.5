"""
Real-time input-to-music: webcam motion or MIDI mod wheel drives SDE
denoise curve in a StreamPipeline, generating and swapping audio in
near real-time (~175ms).

Usage:
    uv run python demos/realtime_motion.py [audio_file]            # webcam mode
    uv run python demos/realtime_motion.py --midi [audio_file]     # MIDI mod wheel

Controls:
    ESC = quit
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import cv2
import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Audio, Latent

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_AUDIO = PROJECT_ROOT / "test_audio" / "new_order_confusion_60seconds.wav"

SAMPLE_RATE = 48000
T = 1500  # 60s at 25fps
CROSSFADE_SECONDS = 0.05
LORA_PATH = r"C:\_dev\models\comfyui_models\loras\acestep1.5\daftpunkstyle1200.safetensors"


# ---------------------------------------------------------------------------
# Audio engine (from spike_motion, simplified)
# ---------------------------------------------------------------------------

class AudioEngine:
    def __init__(self, data, sr):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.sr = sr
        self.channels = data.shape[1]
        self.current = data.copy()
        self.position = 0
        self.crossfade_len = max(1, int(sr * CROSSFADE_SECONDS))
        self._old = None
        self._fading = False
        self._fade_pos = 0
        self._lock = threading.Lock()

    @property
    def duration(self):
        return len(self.current) / self.sr

    @property
    def playback_position(self):
        return self.position / self.sr

    def swap(self, new_data):
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)
        if new_data.shape[1] != self.channels:
            if self.channels == 2 and new_data.shape[1] == 1:
                new_data = np.column_stack([new_data, new_data])
            elif self.channels == 1 and new_data.shape[1] == 2:
                new_data = new_data.mean(axis=1, keepdims=True)
        with self._lock:
            self._old = self.current.copy()
            self.current = new_data
            self._fading = True
            self._fade_pos = 0

    def _fill(self, buf, src, pos, frames):
        n = len(src)
        written = 0
        p = pos % n
        while written < frames:
            chunk = min(frames - written, n - p)
            buf[written:written + chunk] = src[p:p + chunk]
            written += chunk
            p = (p + chunk) % n

    def _callback(self, outdata, frames, _time_info, _status):
        n = len(self.current)
        if n == 0:
            outdata[:] = 0
            return
        with self._lock:
            out = np.zeros((frames, self.channels), dtype="float32")
            self._fill(out, self.current, self.position, frames)
            if self._fading and self._old is not None:
                old_out = np.zeros((frames, self.channels), dtype="float32")
                self._fill(old_out, self._old, self.position, frames)
                fade_frames = min(frames, self.crossfade_len - self._fade_pos)
                t = np.linspace(
                    self._fade_pos / self.crossfade_len,
                    (self._fade_pos + fade_frames) / self.crossfade_len,
                    fade_frames,
                ).reshape(-1, 1)
                out[:fade_frames] = (
                    old_out[:fade_frames] * np.cos(t * np.pi / 2)
                    + out[:fade_frames] * np.sin(t * np.pi / 2)
                )
                self._fade_pos += fade_frames
                if self._fade_pos >= self.crossfade_len:
                    self._fading = False
                    self._old = None
            self.position = (self.position + frames) % n
            outdata[:] = out

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=self.channels,
            callback=self._callback, blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()


# ---------------------------------------------------------------------------
# Webcam motion tracker (from spike_motion)
# ---------------------------------------------------------------------------

class MotionTracker:
    def __init__(self, camera=0):
        self.cap = cv2.VideoCapture(camera)
        # Minimize buffer so we always get the latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.prev_gray = None
        self.smoothed = 0.0
        self.alpha = 0.3

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame, 0.0
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        raw = np.mean(diff) / 255.0
        scaled = min(raw * 40.0, 1.0)
        self.smoothed = self.alpha * scaled + (1 - self.alpha) * self.smoothed
        return frame, self.smoothed

    def read_latest(self):
        """Read the most recent frame. Buffer size is set to 1 so this
        returns the freshest available frame without blocking."""
        return self.read()

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# MIDI mod wheel reader
# ---------------------------------------------------------------------------

class MidiKnobs:
    """Reads MIDI CC from MPK Mini 3 endless encoders. Values are 0.0-1.0."""

    def __init__(self, port_name=None):
        import mido
        names = mido.get_input_names()
        if not names:
            raise RuntimeError("No MIDI input devices found")
        if port_name is None:
            port_name = names[0]
        print(f"  MIDI: available ports: {names}")
        print(f"  MIDI: opening '{port_name}'...")
        try:
            self._port = mido.open_input(port_name)
        except Exception as e:
            print(f"  MIDI: failed to open '{port_name}': {e}")
            print(f"  MIDI: trying virtual port...")
            import rtmidi
            mi = rtmidi.MidiIn()
            ports = mi.get_ports()
            print(f"  MIDI: rtmidi ports: {ports}")
            for i, p in enumerate(ports):
                if port_name.split(" ")[0].lower() in p.lower():
                    self._port = mido.open_input(p)
                    print(f"  MIDI: opened '{p}' via fallback")
                    break
            else:
                raise
        # K1=CC#70 (denoise/sde_curve), K2=CC#71 (seed), K3=CC#72 (lora), K4=CC#73 (feedback)
        self._values = {70: 0.0, 71: 0.0, 72: 0.0, 73: 0.0}
        self._sensitivity = {70: 2.0, 71: 0.5, 72: 2.0, 73: 2.0}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            for msg in self._port.iter_pending():
                if msg.type == "control_change" and msg.control in self._values:
                    # Endless encoder: 1-63 = clockwise, 65-127 = counter-clockwise
                    delta = msg.value if msg.value < 64 else msg.value - 128
                    sens = self._sensitivity[msg.control]
                    with self._lock:
                        v = self._values[msg.control] + delta * sens / 127.0
                        self._values[msg.control] = max(0.0, min(1.0, v))
            time.sleep(0.001)

    def get(self, cc):
        with self._lock:
            return self._values[cc]

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self._port.close()


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

def draw_hud(frame, audio_eng, motion, motion_history, num_gens, tick_ms, dec_ms, curve_val, denoise_val, seed, lora_val, feedback_val):
    h, w = frame.shape[:2]

    # Playback position bar (bottom)
    frac = audio_eng.playback_position / audio_eng.duration if audio_eng.duration else 0
    cv2.rectangle(frame, (0, h - 6), (int(w * frac), h), (0, 200, 255), -1)
    txt = f"{audio_eng.playback_position:.1f}s / {audio_eng.duration:.1f}s"
    cv2.putText(frame, txt, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Motion bar (right side)
    bar_h = int(motion * 200)
    cv2.rectangle(frame, (w - 20, 80), (w - 5, 280), (40, 40, 40), -1)
    cv2.rectangle(frame, (w - 20, 280 - bar_h), (w - 5, 280), (0, 255, 0), -1)

    # Motion history waveform (bottom-left)
    if len(motion_history) > 1:
        n = min(len(motion_history), 200)
        pts = []
        for i in range(n):
            x = 10 + int(i * (w * 0.4) / 200)
            y = h - 20 - int(motion_history[-(n - i)] * 60)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 255, 0), 1)

    # Stats (top-left)
    cv2.putText(frame, f"gen #{num_gens}  tick={tick_ms:.0f}ms  dec={dec_ms:.0f}ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"denoise={denoise_val:.2f}  seed={seed}  lora={lora_val:.2f}  fb={feedback_val:.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    audio_path = DEFAULT_AUDIO
    use_midi = False
    use_sde = False
    args = [a for a in sys.argv[1:]]
    if "--midi" in args:
        use_midi = True
        args.remove("--midi")
    if "--sde" in args:
        use_sde = True
        args.remove("--sde")
    if args:
        audio_path = Path(args[0])

    print("=" * 60)
    print("Real-Time Motion-to-Music")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load model + prepare source
    # ------------------------------------------------------------------
    trt_engines = {
        "decoder": str(PROJECT_ROOT / "trt_engines" / "decoder_mixed_b8_s1500.engine"),
        "vae_encode": str(PROJECT_ROOT / "trt_engines" / "vae_encode_fp16_max6000.engine"),
        "vae_decode": str(PROJECT_ROOT / "trt_engines" / "vae_decode_fp16_max6000.engine"),
    }

    print("[Setup] Loading model...")
    t0 = time.time()
    session = Session(
        project_root=str(PROJECT_ROOT / "checkpoints"),
        compile_model=False,
        trt_engines=trt_engines,
    )
    handler = session.handler
    device, dtype = handler.device, handler.dtype
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    print("[Setup] Loading source audio...")
    data, sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(60.0 * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)

    print("[Setup] VAE encode + semantic extract...")
    latent = session.encode_audio(audio_in)
    hints = session.extract_hints(latent)
    context_latent = session.hints_to_latent(hints)
    source = PreparedSource(latent=latent, hints=hints, context_latent=context_latent)

    print("[Setup] Text encode...")
    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    entry = cond.to_entries()[0]

    # Build context/source latents
    ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
    D = ctx_lat.shape[2]
    cm = torch.ones(1, T, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)
    source_latents = source.latent.tensor.to(device=device, dtype=dtype)

    # Pipeline
    engine = handler._diffusion_engine
    config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)
    pipe = StreamPipeline(engine, config)

    print(f"  Pipeline ready: {pipe.stats()['backend']}")

    # Precompute LoRA deltas at unit strength for real-time scaling
    from acestep.nodes.lora_nodes import _precompute_lora_deltas, _apply_lora_deltas
    print(f"[Setup] Precomputing LoRA deltas...")
    lora_deltas = _precompute_lora_deltas(LORA_PATH, strength=1.0, device=device, dtype=dtype)
    lora_applied_scale = 0.0  # current scale baked into weights
    print(f"  LoRA ready: {len(lora_deltas)} params")

    # ------------------------------------------------------------------
    # Start audio + input + display
    # ------------------------------------------------------------------
    src_np = waveform.numpy().T  # [samples, channels]
    audio_eng = AudioEngine(src_np, SAMPLE_RATE)
    audio_eng.start()
    print(f"[Audio] Playing ({audio_eng.duration:.1f}s, {SAMPLE_RATE}Hz)")

    # Input source
    tracker = None
    midi_knobs = None
    if use_midi:
        midi_knobs = MidiKnobs()
        disp_w, disp_h = 640, 480
    else:
        tracker = MotionTracker()
        test_ok, test_frame = tracker.cap.read()
        if not test_ok:
            print("Cannot open webcam")
            return
        disp_h, disp_w = test_frame.shape[:2]

    pygame.init()
    mode_str = "MIDI Mod Wheel" if use_midi else "Webcam Motion"
    screen = pygame.display.set_mode((disp_w, disp_h))
    pygame.display.set_caption(f"Real-Time {mode_str}")

    print(f"\n  Mode: {mode_str}")
    print(f"  {'Move mod wheel' if use_midi else 'Move'} to change the music. ESC to quit.\n")

    # ------------------------------------------------------------------
    # Shared state between display thread and pipeline thread
    # ------------------------------------------------------------------
    motion_val = [0.0]          # latest motion intensity (written by display, read by pipeline)
    motion_lock = threading.Lock()
    motion_history = []         # only touched by display thread
    running = [True]
    SEED = 1528
    skip_threshold = 1e-3
    stats = {                   # written by pipeline thread, read by display thread
        "num_gens": 0,
        "tick_ms": 0.0,
        "dec_ms": 0.0,
        "curve_val": 0.0,
        "denoise": 0.0,
        "seed": SEED,
        "lora": 0.0,
        "feedback": 0.0,
    }

    # ------------------------------------------------------------------
    # Pipeline thread: submit, tick, decode, swap audio
    # ------------------------------------------------------------------
    def pipeline_loop():
        nonlocal lora_applied_scale
        last_latent = None
        last_wav = None

        while running[0]:
            with motion_lock:
                cur_motion = motion_val[0]

            # Read knob values (MIDI mode) or derive from motion (webcam mode)
            if use_midi:
                k1_val = midi_knobs.get(70)  # K1: denoise or SDE curve
                seed = int(midi_knobs.get(71) * 1000)  # K2: seed (0-1000)
                lora_val = midi_knobs.get(72)  # K3: LoRA strength
                feedback_val = midi_knobs.get(73)  # K4: latent feedback
            else:
                k1_val = cur_motion
                seed = SEED
                lora_val = 0.0
                feedback_val = 0.0

            # Adjust LoRA weight deltas to match target scale
            if abs(lora_val - lora_applied_scale) > 1e-4:
                diff = lora_val - lora_applied_scale
                _apply_lora_deltas(engine.decoder, lora_deltas, sign=diff)
                lora_applied_scale = lora_val

            # Latent feedback: blend last output into source
            if feedback_val > 0.0 and last_latent is not None:
                effective_source = (1.0 - feedback_val) * source_latents + feedback_val * last_latent
            else:
                effective_source = source_latents

            sde_curve = None
            if use_sde:
                sde_curve = torch.full((1, T, 1), k1_val, dtype=torch.float32)
                denoise_val = 0.75
            else:
                denoise_val = k1_val

            pipe.submit(SlotRequest(
                encoder_hidden_states=entry.encoder_hidden_states,
                encoder_attention_mask=entry.encoder_attention_mask,
                context_latents=context_latents,
                seed=seed,
                source_latents=effective_source,
                denoise=denoise_val,
                sde_denoise_curve=sde_curve,
            ))

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = pipe.tick()
            torch.cuda.synchronize()
            tick_ms = (time.perf_counter() - t0) * 1000

            dec_ms = 0.0
            if result is not None:
                skipped = False
                if last_latent is not None:
                    mse = (result - last_latent).pow(2).mean().item()
                    if mse < skip_threshold and last_wav is not None:
                        skipped = True

                last_latent = result.clone()

                if not skipped:
                    t1 = time.perf_counter()
                    audio_out = session.decode(Latent(tensor=result))
                    torch.cuda.synchronize()
                    dec_ms = (time.perf_counter() - t1) * 1000

                    wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                    wav_np = wav.numpy().T
                    last_wav = wav_np
                    audio_eng.swap(wav_np)

                stats["num_gens"] += 1
                stats["tick_ms"] = tick_ms
                stats["dec_ms"] = dec_ms
                stats["curve_val"] = k1_val
                stats["denoise"] = denoise_val
                stats["seed"] = seed
                stats["lora"] = lora_val
                stats["feedback"] = feedback_val

    pipe_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipe_thread.start()

    # ------------------------------------------------------------------
    # Display loop: webcam + HUD at full frame rate
    # ------------------------------------------------------------------
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            if tracker is not None:
                frame, motion = tracker.read_latest()
                if frame is None:
                    break
            else:
                # MIDI mode: no webcam, build a simple visualization frame
                motion = midi_knobs.get(70)
                frame = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

            with motion_lock:
                motion_val[0] = motion

            motion_history.append(motion)
            if len(motion_history) > 400:
                motion_history = motion_history[-400:]

            draw_hud(frame, audio_eng, motion, motion_history,
                     stats["num_gens"], stats["tick_ms"],
                     stats["dec_ms"], stats["curve_val"],
                     stats["denoise"], stats["seed"],
                     stats["lora"], stats["feedback"])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Cap display loop to ~60fps to avoid burning CPU
            time.sleep(0.016)

    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        pipe_thread.join(timeout=2)
        audio_eng.stop()
        if tracker is not None:
            tracker.release()
        if midi_knobs is not None:
            midi_knobs.release()
        pygame.quit()
        print(f"\n{stats['num_gens']} generations completed.")


if __name__ == "__main__":
    main()
