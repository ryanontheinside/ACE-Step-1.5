"""
Real-time input-to-music using the Session graph API.

Routes the pipeline loop through Session.create_stream() / SessionStream.

Usage:
    uv run python demos/realtime_motion_graph.py [audio_file]            # webcam mode
    uv run python demos/realtime_motion_graph.py --midi [audio_file]     # MIDI knobs
    uv run python demos/realtime_motion_graph.py --midi --sde            # MIDI + SDE curves
    uv run python demos/realtime_motion_graph.py --vae-window 15         # windowed decode

Controls:
    ESC = quit

MIDI knob layout (K1-K5, CC#70-74):
    Regular:  denoise  seed  feedback  shift
    SDE:      sde_amp  seed  feedback  shift  periodicity
"""

import sys
import threading
import time
from dataclasses import dataclass
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
from acestep.engine.session import Session
from acestep.nodes.types import Audio

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_AUDIO = PROJECT_ROOT / "test_audio" / "new_order_confusion_60seconds.wav"

SAMPLE_RATE = 48000
T = 1500  # 60s at 25fps
CROSSFADE_SECONDS = 0.05


# ---------------------------------------------------------------------------
# MIDI knob configuration
# ---------------------------------------------------------------------------

@dataclass
class KnobDef:
    """A MIDI CC encoder mapping."""
    cc: int
    default: float = 0.0
    sensitivity: float = 2.0
    max_val: float = 1.0


def knob_layout(sde: bool) -> dict[str, KnobDef]:
    """Return the active knob layout for the given mode.

    K1=CC70  K2=CC71  K3=CC72  K4=CC73  K5=CC74
    Shared knobs first (seed, feedback, shift), mode-specific at edges.
    """
    knobs = {}
    # K1: primary control
    if sde:
        knobs["sde_amp"] = KnobDef(cc=70, sensitivity=2.0)
    else:
        knobs["denoise"] = KnobDef(cc=70, sensitivity=2.0)
    # K2-K4: shared
    knobs["seed"] = KnobDef(cc=71, sensitivity=0.5)
    knobs["feedback"] = KnobDef(cc=72, sensitivity=2.0)
    knobs["shift"] = KnobDef(cc=73, default=0.5, sensitivity=1.0)
    # K5: SDE only
    if sde:
        knobs["periodicity"] = KnobDef(cc=74, sensitivity=2.0)
    return knobs


# Colors for graph lines (auto-looked-up by parameter name)
GRAPH_COLORS = {
    "denoise":     (0, 255, 0),
    "sde_amp":     (0, 255, 0),
    "seed":        (255, 180, 0),
    "feedback":    (0, 200, 255),
    "shift":       (180, 0, 255),
    "periodicity": (255, 100, 100),
    "motion":      (0, 255, 0),
}


# ---------------------------------------------------------------------------
# Audio engine
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
                out[:fade_frames] = old_out[:fade_frames] * (1 - t) + out[:fade_frames] * t
                self._fade_pos += fade_frames
                if self._fade_pos >= self.crossfade_len:
                    self._fading = False
                    self._old = None
            self.position = (self.position + frames) % n
        outdata[:] = out

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            callback=self._callback,
            blocksize=2048,
        )
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()


# ---------------------------------------------------------------------------
# Input sources
# ---------------------------------------------------------------------------

class MotionTracker:
    def __init__(self, camera=0):
        self.cap = cv2.VideoCapture(camera)
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
        return self.read()

    def release(self):
        self.cap.release()


class MidiKnobs:
    """Reads MIDI CC values from endless encoders.

    Knobs are addressed by name, not CC number. The mapping is
    defined by the knobs dict passed at init.
    """

    def __init__(self, knobs: dict[str, KnobDef], port_name=None):
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

        self._knobs = knobs
        self._values = {name: k.default for name, k in knobs.items()}
        self._cc_map = {k.cc: name for name, k in knobs.items()}
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            for msg in self._port.iter_pending():
                if msg.type == "control_change" and msg.control in self._cc_map:
                    name = self._cc_map[msg.control]
                    knob = self._knobs[name]
                    delta = msg.value if msg.value < 64 else msg.value - 128
                    with self._lock:
                        v = self._values[name] + delta * knob.sensitivity / 127.0
                        self._values[name] = max(0.0, min(knob.max_val, v))
            time.sleep(0.001)

    def get(self, name: str) -> float:
        with self._lock:
            return self._values[name]

    def get_all(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self._port.close()


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

def draw_hud(frame, audio_eng, params, histories, motion=0.0, sde_curve_np=None):
    """Draw heads-up display. All active parameters render automatically."""
    h, w = frame.shape[:2]

    # Playback bar (bottom)
    frac = audio_eng.playback_position / audio_eng.duration if audio_eng.duration else 0
    cv2.rectangle(frame, (0, h - 6), (int(w * frac), h), (0, 200, 255), -1)
    cv2.putText(frame, f"{audio_eng.playback_position:.1f}s / {audio_eng.duration:.1f}s",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Motion bar (right side)
    bar_h = int(motion * 200)
    cv2.rectangle(frame, (w - 20, 80), (w - 5, 280), (40, 40, 40), -1)
    cv2.rectangle(frame, (w - 20, 280 - bar_h), (w - 5, 280), (0, 255, 0), -1)

    # Stats line (top)
    cv2.putText(frame,
                f"gen #{params.get('num_gens', 0)}  tick={params.get('tick_ms', 0):.0f}ms  dec={params.get('dec_ms', 0):.0f}ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Parameter line (auto-generated from all non-timing params)
    timing_keys = {"num_gens", "tick_ms", "dec_ms"}
    parts = []
    for k, v in params.items():
        if k in timing_keys:
            continue
        if isinstance(v, int):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
    if parts:
        cv2.putText(frame, "  ".join(parts),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Graph area: scrolling history lines + optional SDE curve overlay
    if not histories:
        return

    gx, gy = 10, 65
    gw, gh = w - 40, h - 90

    # History lines (one per tracked parameter)
    for name, hist in histories.items():
        if len(hist) < 2:
            continue
        color = GRAPH_COLORS.get(name, (255, 255, 255))
        n = min(len(hist), gw)
        pts = []
        for i in range(n):
            x = gx + i
            y = gy + gh - int(hist[-(n - i)] * gh)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color, 1)

    # SDE curve overlay: shows the actual per-frame denoise curve sent to the model
    if sde_curve_np is not None and len(sde_curve_np) > 1:
        color = (100, 255, 100)
        n_pts = min(len(sde_curve_np), gw)
        step = max(1, len(sde_curve_np) // n_pts)
        pts = []
        for i in range(n_pts):
            x = gx + int(i * gw / n_pts)
            y = gy + gh - int(float(sde_curve_np[i * step]) * gh)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, color, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    audio_path = DEFAULT_AUDIO
    use_midi = False
    use_sde = False
    vae_window = 0.0
    args = list(sys.argv[1:])
    if "--midi" in args:
        use_midi = True
        args.remove("--midi")
    if "--sde" in args:
        use_sde = True
        args.remove("--sde")
    if "--vae-window" in args:
        idx = args.index("--vae-window")
        vae_window = float(args[idx + 1])
        del args[idx:idx + 2]
    if args:
        audio_path = Path(args[0])

    knobs = knob_layout(use_sde)
    k1_name = "sde_amp" if use_sde else "denoise"

    print("=" * 60)
    print("Real-Time Motion-to-Music (GRAPH BACKEND)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Setup
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
        vae_window=vae_window,
    )
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

    print("[Setup] Preparing source...")
    source = session.prepare_source(audio_in)

    print("[Setup] Text encode...")
    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )

    print("[Setup] Creating stream...")
    stream = session.create_stream(
        source=source,
        conditioning=cond,
        steps=8,
        shift=3.0,
    )
    print(f"  Pipeline ready: {stream.stats()['backend']}")

    # ------------------------------------------------------------------
    # Audio + input
    # ------------------------------------------------------------------
    src_np = waveform.numpy().T
    audio_eng = AudioEngine(src_np, SAMPLE_RATE)
    audio_eng.start()
    print(f"[Audio] Playing ({audio_eng.duration:.1f}s, {SAMPLE_RATE}Hz)")

    tracker = None
    midi_knobs = None
    if use_midi:
        midi_knobs = MidiKnobs(knobs)
        disp_w, disp_h = 640, 480
    else:
        tracker = MotionTracker()
        test_ok, test_frame = tracker.cap.read()
        if not test_ok:
            print("Cannot open webcam")
            return
        disp_h, disp_w = test_frame.shape[:2]

    pygame.init()
    mode_str = "MIDI" + (" SDE" if use_sde else "") if use_midi else "Webcam"
    screen = pygame.display.set_mode((disp_w, disp_h))
    pygame.display.set_caption(f"Real-Time {mode_str} (Graph)")

    print(f"\n  Mode: {mode_str}")
    if use_midi:
        print(f"  Knobs: {', '.join(f'K{i+1}={name}' for i, name in enumerate(knobs))}")
    print(f"  {'Move knobs' if use_midi else 'Move'} to change the music. ESC to quit.\n")

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    motion_val = [0.0]
    motion_lock = threading.Lock()
    running = [True]
    SEED = 1528
    skip_threshold = 1e-3

    # Written by pipeline thread, read by display thread.
    # Pre-populate knob params in display order.
    params = {"num_gens": 0, "tick_ms": 0.0, "dec_ms": 0.0}
    params[k1_name] = 0.0
    params["seed"] = SEED
    params["feedback"] = 0.0
    params["shift"] = 3.0
    if use_sde:
        params["periodicity"] = 0.0

    # Current SDE curve for overlay (mutable container for thread sharing)
    sde_curve_display = [None]

    # ------------------------------------------------------------------
    # Pipeline thread
    # ------------------------------------------------------------------
    def pipeline_loop():
        last_latent = None
        last_wav = None

        while running[0]:
            # Read inputs
            if use_midi:
                raw = midi_knobs.get_all()
            else:
                with motion_lock:
                    m = motion_val[0]
                raw = {k1_name: m, "seed": 0.0, "feedback": 0.0, "shift": 0.5}
                if use_sde:
                    raw["periodicity"] = 0.0

            k1 = raw[k1_name]
            seed = int(raw["seed"] * 1000) if use_midi else SEED
            feedback = raw["feedback"]
            shift_raw = raw["shift"]

            # Shift: map 0-1 knob to 1.0-6.0
            shift_val = 1.0 + shift_raw * 5.0
            if abs(shift_val - stream.config.shift) > 0.05:
                stream.set_shift(shift_val)

            # Latent feedback: blend last output into source
            source_lat = None
            if feedback > 0.0 and last_latent is not None:
                source_lat = (
                    (1.0 - feedback) * stream.source_latents
                    + feedback * last_latent
                )

            # Build SDE curve or use direct denoise
            sde_curve = None
            if use_sde:
                denoise = 1.0
                amplitude = k1
                periodicity = raw.get("periodicity", 0.0)

                if periodicity > 0.01:
                    cycles = 0.5 + periodicity * 7.5
                    t = torch.linspace(0, 1, T).unsqueeze(0).unsqueeze(-1)
                    sde_curve = amplitude * (0.5 + 0.5 * torch.sin(2 * 3.14159 * cycles * t))
                else:
                    sde_curve = torch.full((1, T, 1), amplitude, dtype=torch.float32)

                sde_curve_display[0] = sde_curve.squeeze().numpy()
            else:
                denoise = k1
                sde_curve_display[0] = None

            stream.submit(
                denoise=denoise,
                seed=seed,
                source_latents=source_lat,
                sde_denoise_curve=sde_curve,
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result_latent = stream.tick()
            torch.cuda.synchronize()
            tick_ms = (time.perf_counter() - t0) * 1000

            dec_ms = 0.0
            if result_latent is not None:
                result = result_latent.tensor
                skipped = False
                if last_latent is not None:
                    mse = (result - last_latent).pow(2).mean().item()
                    if mse < skip_threshold and last_wav is not None:
                        skipped = True

                last_latent = result.clone()

                if not skipped:
                    t1 = time.perf_counter()
                    if vae_window > 0:
                        t_pos = audio_eng.position / SAMPLE_RATE
                        audio_out = session.decode(result_latent, t_start=t_pos)
                        torch.cuda.synchronize()
                        dec_ms = (time.perf_counter() - t1) * 1000
                        win_wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                        win_np = win_wav.numpy().T
                        win_start = audio_out.start_sample
                        win_end = win_start + win_np.shape[0]
                        buf = audio_eng.current.copy()
                        xfade = min(2400, win_np.shape[0] // 4)
                        if win_start > 0 and xfade > 0:
                            t_in = np.linspace(0.0, 1.0, xfade).reshape(-1, 1)
                            win_np[:xfade] = (
                                buf[win_start:win_start + xfade] * (1 - t_in)
                                + win_np[:xfade] * t_in
                            )
                        if win_end < buf.shape[0] and xfade > 0:
                            t_out = np.linspace(1.0, 0.0, xfade).reshape(-1, 1)
                            tail = min(xfade, buf.shape[0] - win_end + xfade)
                            s = win_np.shape[0] - tail
                            win_np[s:] = (
                                win_np[s:] * t_out[:tail]
                                + buf[win_start + s:win_start + s + tail] * (1 - t_out[:tail])
                            )
                        clamp_end = min(win_end, buf.shape[0])
                        buf[win_start:clamp_end] = win_np[:clamp_end - win_start]
                        audio_eng.swap(buf)
                        last_wav = buf
                    else:
                        audio_out = session.decode(result_latent)
                        torch.cuda.synchronize()
                        dec_ms = (time.perf_counter() - t1) * 1000
                        wav = audio_out.waveform.detach().cpu().float().squeeze(0)
                        wav_np = wav.numpy().T
                        last_wav = wav_np
                        audio_eng.swap(wav_np)

                # Update params for display (knob order)
                params["num_gens"] = params.get("num_gens", 0) + 1
                params["tick_ms"] = tick_ms
                params["dec_ms"] = dec_ms
                params[k1_name] = round(k1, 2)
                params["seed"] = seed
                params["feedback"] = round(feedback, 2)
                params["shift"] = round(shift_val, 2)
                if use_sde:
                    params["periodicity"] = round(raw.get("periodicity", 0.0), 2)

    pipe_thread = threading.Thread(target=pipeline_loop, daemon=True)
    pipe_thread.start()

    # ------------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------------
    if use_midi:
        histories = {name: [] for name in knobs}
    else:
        histories = {"motion": []}
    max_history = 600

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
                motion = midi_knobs.get(k1_name)
                frame = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

            with motion_lock:
                motion_val[0] = motion

            # Update histories (all tracked parameters, auto-discovered)
            if use_midi:
                raw = midi_knobs.get_all()
                for name in histories:
                    histories[name].append(raw.get(name, 0.0))
            else:
                histories["motion"].append(motion)
            for hist in histories.values():
                if len(hist) > max_history:
                    del hist[:-max_history]

            draw_hud(frame, audio_eng, params, histories,
                     motion=motion, sde_curve_np=sde_curve_display[0])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

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
        print(f"\n{params.get('num_gens', 0)} generations completed.")


if __name__ == "__main__":
    main()
