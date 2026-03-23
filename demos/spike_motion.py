"""
Motion-to-audio spike: Webcam motion -> float curve -> Scope confusion pipeline.

Usage: python spike_motion.py <audio_file> [--port 8000]

Controls:
    SPACE (hold) = capture motion
    ESC          = quit
"""

import datetime
import io
import os
import subprocess
import sys
import threading
import time

import cv2
import keyboard
import numpy as np
import pygame
import requests
import sounddevice as sd
import soundfile as sf

PIPELINE_ID = "comfyui/latent_noise_mask"
FLOAT_FPS = 25
CROSSFADE_SECONDS = 0.05  # 50ms equal-power crossfade
LATENCY_ESTIMATE = 1.0  # seconds to offset float placement ahead
DEFAULT_AUDIO = "notes/scope_plugins/scope-comfyui/scope_comfyui/assets/Vesuvius_v2_edit_60s.wav"

# -- Ghost overlay --
GHOST_ENABLED = True

# -- Pipeline overrides (None = use schema default) --
OVERRIDE_TEXT = None#"""Neo-Soul: A warm, organic neo-soul track dripping with live instrumentation and effortless groove. A live drummer plays a loose, hip-hop influenced pocket—soft kick drum with lazy swing, snare hits that sit just behind the beat, and brushed hi-hats that breathe and shuffle with human imperfection."""         # text_-1: prompt text
OVERRIDE_LYRICS = None  # lyrics_-1: lyrics text
OVERRIDE_LORA_STRENGTH = 0.0  # strength_model_32: lora weight (default 1.52)
OVERRIDE_DENOISE = 0.85      # denoise_51: 0.0-1.0 (default 1.0)


# ---------------------------------------------------------------------------
# Audio engine: looping playback with equal-power crossfade
# ---------------------------------------------------------------------------

class AudioEngine:
    def __init__(self, filepath):
        self.data, self.sr = sf.read(filepath, dtype="float32")
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.channels = self.data.shape[1]
        self.current = self.data.copy()
        self.position = 0
        self.crossfade_len = max(1, int(self.sr * CROSSFADE_SECONDS))

        # crossfade state
        self._old = None
        self._fading = False
        self._fade_pos = 0
        self._lock = threading.Lock()
        self._recorded = []

    @property
    def duration(self):
        return len(self.current) / self.sr

    @property
    def playback_position(self):
        return self.position / self.sr

    def swap(self, new_data):
        """Queue new audio for crossfade at current position."""
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

    # -- internal -----------------------------------------------------------

    def _fill(self, buf, src, pos, frames):
        """Fill *buf* from *src* starting at *pos*, looping."""
        n = len(src)
        written = 0
        p = pos % n
        while written < frames:
            chunk = min(frames - written, n - p)
            buf[written : written + chunk] = src[p : p + chunk]
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
            self._recorded.append(out.copy())

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def save_audio(self, path):
        if self._recorded:
            data = np.concatenate(self._recorded)
            sf.write(path, data, self.sr)
            print(f"  audio saved: {path} ({len(data)/self.sr:.1f}s)")


# ---------------------------------------------------------------------------
# Webcam motion tracker
# ---------------------------------------------------------------------------

class MotionTracker:
    def __init__(self, camera=0):
        self.cap = cv2.VideoCapture(camera)
        self.prev_gray = None
        self.smoothed = 0.0
        self.alpha = 0.3  # EMA smoothing

    def read(self):
        """Return (bgr_frame, motion_intensity_0_to_1)."""
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

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# Scope API client
# ---------------------------------------------------------------------------

class ScopeClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.server_path = None
        self.default_params = {}
        self._trigger = 0

    def fetch_defaults(self):
        """Fetch pipeline schema and build default params (like the React app)."""
        r = self.session.get(f"{self.base_url}/api/v1/pipelines/schemas")
        r.raise_for_status()
        pipelines = r.json().get("pipelines", {})
        schema = pipelines.get(PIPELINE_ID, {}).get("config_schema", {})
        props = schema.get("properties", {})
        self.all_keys = set(props.keys())
        self.default_params = {
            k: v["default"] for k, v in props.items() if "default" in v
        }
        print(f"  schema properties ({len(props)} total, {len(self.default_params)} with defaults):")
        for k, v in props.items():
            has_default = "default" in v
            default_preview = repr(v["default"])[:50] if has_default else "(no default)"
            print(f"    {k}: {default_preview}")

    def upload_audio(self, filepath):
        name = filepath.replace("\\", "/").split("/")[-1]
        with open(filepath, "rb") as f:
            r = self.session.post(
                f"{self.base_url}/api/v1/assets",
                params={"filename": name},
                data=f.read(),
            )
        r.raise_for_status()
        self.server_path = r.json()["path"]
        print(f"  uploaded -> {self.server_path}")

    def generate(self, float_values):
        self._trigger += 1
        params = dict(self.default_params)
        # Find audio and float keys (naming varies by workflow)
        audio_key = next(
            (k for k in self.all_keys if "audio" in k.lower()),
            "audio",
        )
        params[audio_key] = self.server_path
        float_key = next(
            (k for k in self.all_keys if "float" in k.lower()),
            "floats",
        )
        params[float_key] = float_values
        params["generation_trigger"] = self._trigger
        if OVERRIDE_TEXT is not None:
            params["text"] = OVERRIDE_TEXT
        if OVERRIDE_LYRICS is not None:
            params["lyrics"] = OVERRIDE_LYRICS
        if OVERRIDE_LORA_STRENGTH is not None:
            params["strength_model"] = OVERRIDE_LORA_STRENGTH
        if OVERRIDE_DENOISE is not None:
            params["denoise"] = OVERRIDE_DENOISE
        r = self.session.post(
            f"{self.base_url}/api/v1/pipeline/{PIPELINE_ID}/run",
            json=params,
            timeout=120,
        )
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "audio" in ct:
            data, sr = sf.read(io.BytesIO(r.content), dtype="float32")
            return data, sr
        print(f"  unexpected content-type: {ct}")
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_motion(samples, timestamps, duration):
    """Resample irregularly-timed samples to a fixed 25-fps grid."""
    n = max(1, int(duration * FLOAT_FPS))
    if len(samples) < 2:
        return [samples[0] if samples else 0.0] * n
    t_out = np.linspace(timestamps[0], timestamps[-1], n)
    return np.interp(t_out, timestamps, samples).tolist()


def build_curve(motion_floats, song_duration, offset_seconds):
    """Place motion data into a full-song float curve at *offset_seconds*."""
    total = max(1, int(song_duration * FLOAT_FPS))
    curve = [0.0] * total
    start = int(offset_seconds * FLOAT_FPS) % total
    for i, v in enumerate(motion_floats):
        curve[(start + i) % total] = v
    return curve


# ---------------------------------------------------------------------------
# Visualisation helpers (drawn on the OpenCV frame)
# ---------------------------------------------------------------------------

def draw_hud(frame, engine, motion, recording, motion_history, active_curve=None):
    h, w = frame.shape[:2]

    # playback position bar along the bottom
    frac = engine.playback_position / engine.duration if engine.duration else 0
    cv2.rectangle(frame, (0, h - 6), (int(w * frac), h), (0, 200, 255), -1)

    # position text
    txt = f"{engine.playback_position:.1f}s / {engine.duration:.1f}s"
    cv2.putText(frame, txt, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if recording:
        # red dot
        cv2.circle(frame, (w - 25, 25), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 55, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # motion bar (right side)
    bar_h = int(motion * 200)
    cv2.rectangle(frame, (w - 20, 80), (w - 5, 280), (40, 40, 40), -1)
    cv2.rectangle(frame, (w - 20, 280 - bar_h), (w - 5, 280), (0, 255, 0), -1)

    # motion history waveform (bottom-left)
    if len(motion_history) > 1:
        pts = []
        n = min(len(motion_history), 200)
        for i in range(n):
            x = 10 + int(i * (w * 0.4) / 200)
            y = h - 20 - int(motion_history[-(n - i)] * 60)
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 255, 0), 1)

    # Timeline curve strip (top of frame)
    if active_curve and len(active_curve) > 1:
        band_h = 40
        cv2.rectangle(frame, (0, 0), (w, band_h), (15, 15, 15), -1)
        n = len(active_curve)
        for i in range(w):
            ci = int(i * n / w)
            val = active_curve[ci]
            bar = int(val * (band_h - 6))
            if bar > 0:
                cv2.line(frame, (i, band_h - 3), (i, band_h - 3 - bar), (0, 140, 255), 1)
        # playhead
        px = int(frac * w)
        cv2.line(frame, (px, 0), (px, band_h), (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    port = 8000
    audio_path = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            audio_path = args[i]
            i += 1

    if not audio_path:
        audio_path = DEFAULT_AUDIO

    base_url = f"http://127.0.0.1:{port}"
    print(f"Scope @ {base_url}")
    print(f"Audio: {audio_path}")

    engine = AudioEngine(audio_path)
    tracker = MotionTracker()
    client = ScopeClient(base_url)

    print("Fetching pipeline schema...")
    client.fetch_defaults()

    print("Uploading audio...")
    client.upload_audio(audio_path)

    # Pre-generate with a flat curve to warm the pipeline
    print("Pre-generating (first run, may take a while)...")
    total_floats = max(1, int(engine.duration * FLOAT_FPS))
    flat_curve = [0.0] * total_floats
    pre_audio, pre_sr = client.generate(flat_curve)
    if pre_audio is not None:
        if pre_sr != engine.sr:
            ratio = engine.sr / pre_sr
            idx = (np.arange(int(len(pre_audio) * ratio)) / ratio).astype(int)
            idx = np.clip(idx, 0, len(pre_audio) - 1)
            pre_audio = pre_audio[idx]
        if pre_audio.ndim == 1:
            pre_audio = pre_audio.reshape(-1, 1)
        if pre_audio.shape[1] != engine.channels:
            if engine.channels == 2 and pre_audio.shape[1] == 1:
                pre_audio = np.column_stack([pre_audio, pre_audio])
            elif engine.channels == 1 and pre_audio.shape[1] == 2:
                pre_audio = pre_audio.mean(axis=1, keepdims=True)
        engine.current = pre_audio
        print(f"Pre-generation done, using generated audio ({len(pre_audio)/engine.sr:.1f}s)")
    else:
        print("Pre-generation failed, using original audio")

    print(f"Starting playback ({engine.duration:.1f}s, {engine.sr}Hz, {engine.channels}ch)")
    engine.start()

    gen_counter = 1  # 0 was used for pre-generation
    recording = False
    rec_samples = []
    rec_times = []
    rec_start_pos = 0.0
    motion_history = []
    rec_frames = []
    ghost_frames = []
    ghost_region_start = 0.0
    ghost_region_dur = 0.0
    ghost_ready = False
    active_curve = None
    gen_done_flag = [False]

    # Init pygame display using webcam frame size
    test_ok, test_frame = tracker.cap.read()
    if not test_ok:
        print("Cannot read webcam")
        return
    disp_h, disp_w = test_frame.shape[:2]
    cam_fps = tracker.cap.get(cv2.CAP_PROP_FPS) or 30

    # Video recording
    os.makedirs("spike_recordings", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path = f"spike_recordings/{stamp}_video.avi"
    out_path = f"spike_recordings/{stamp}.mp4"
    audio_path_rec = f"spike_recordings/{stamp}_audio.wav"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vid_writer = cv2.VideoWriter(vid_path, fourcc, 30, (disp_w, disp_h))
    vid_frame_count = 0
    vid_start_time = None
    if not vid_writer.isOpened():
        print(f"WARNING: VideoWriter failed to open ({vid_path})")
    else:
        print(f"  recording video to {vid_path}")
    pygame.init()
    screen = pygame.display.set_mode((disp_w, disp_h))
    pygame.display.set_caption("Scope Motion")

    print()
    print("  SPACE (hold) = capture motion")
    print("  ESC          = quit")
    print()

    try:
        while True:
            # Check pygame events for quit/ESC
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            frame, motion = tracker.read()
            if frame is None:
                break

            motion_history.append(motion)
            if len(motion_history) > 400:
                motion_history = motion_history[-400:]

            space = keyboard.is_pressed("space")

            # -- start recording --
            if space and not recording:
                recording = True
                rec_samples = []
                rec_times = []
                rec_frames = []
                rec_start_pos = engine.playback_position
                ghost_ready = False
                print(f"  recording @ {rec_start_pos:.1f}s ...")

            # -- accumulate --
            if recording:
                rec_samples.append(motion)
                rec_times.append(time.monotonic())
                if GHOST_ENABLED:
                    rec_frames.append(frame.copy())

            # -- stop recording, fire generation --
            if not space and recording:
                recording = False
                dur = rec_times[-1] - rec_times[0] if len(rec_times) > 1 else 0
                print(f"  captured {dur:.1f}s ({len(rec_samples)} samples)")

                resampled = resample_motion(rec_samples, rec_times, dur)
                offset = rec_start_pos + dur + LATENCY_ESTIMATE
                curve = build_curve(resampled, engine.duration, offset)
                active_curve = curve

                ghost_frames = rec_frames[:]
                ghost_region_start = offset % engine.duration
                ghost_region_dur = dur
                ghost_ready = False
                gen_done_flag = [False]

                gen_counter += 1
                _gen_id = gen_counter

                def _gen(c=curve, gid=_gen_id, done=gen_done_flag):
                    print(f"  generating (#{gid}) ...")
                    t0 = time.monotonic()
                    audio, sr = client.generate(c)
                    dt = time.monotonic() - t0
                    if audio is not None:
                        if sr != engine.sr:
                            ratio = engine.sr / sr
                            idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
                            idx = np.clip(idx, 0, len(audio) - 1)
                            audio = audio[idx]
                        print(f"  got audio in {dt:.1f}s, crossfading")
                        engine.swap(audio)
                    else:
                        print(f"  generation failed ({dt:.1f}s)")
                    done[0] = True

                threading.Thread(target=_gen, daemon=True).start()

            # Ghost becomes ready once generation completes (audio swapped in)
            if GHOST_ENABLED and not ghost_ready and gen_done_flag[0] and ghost_frames:
                ghost_ready = True

            # Ghost overlay: show recorded frames synced to playhead in affected region
            if GHOST_ENABLED and ghost_ready and ghost_frames and ghost_region_dur > 0:
                pos = engine.playback_position
                region_end = ghost_region_start + ghost_region_dur
                in_region = False
                progress = 0.0
                if region_end <= engine.duration:
                    if ghost_region_start <= pos < region_end:
                        in_region = True
                        progress = (pos - ghost_region_start) / ghost_region_dur
                else:
                    wrapped_end = region_end % engine.duration
                    if pos >= ghost_region_start:
                        in_region = True
                        progress = (pos - ghost_region_start) / ghost_region_dur
                    elif pos < wrapped_end:
                        in_region = True
                        progress = (pos + engine.duration - ghost_region_start) / ghost_region_dur
                if in_region:
                    idx = min(int(progress * len(ghost_frames)), len(ghost_frames) - 1)
                    gf = ghost_frames[idx]
                    tinted = gf.copy()
                    tinted[:, :, 2] = 0  # kill red channel (BGR) -> cyan ghost
                    frame = cv2.addWeighted(frame, 0.7, tinted, 0.3, 0)

            draw_hud(frame, engine, motion, recording, motion_history, active_curve)
            if vid_start_time is None:
                vid_start_time = time.monotonic()
            vid_writer.write(frame)
            vid_frame_count += 1

            # Convert BGR frame to RGB and blit to pygame surface
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        vid_writer.release()
        tracker.release()
        pygame.quit()

        # Save audio and mux with video
        elapsed = time.monotonic() - vid_start_time if vid_start_time else 1
        actual_fps = vid_frame_count / elapsed if elapsed > 0 else 30
        print(f"Saving recording... ({vid_frame_count} frames, {actual_fps:.1f} actual fps)")
        engine.save_audio(audio_path_rec)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-r", f"{actual_fps:.4f}",
                    "-i", vid_path,
                    "-i", audio_path_rec,
                    "-c:v", "libx264", "-crf", "23", "-c:a", "aac",
                    "-shortest",
                    out_path,
                ],
                capture_output=True,
            )
            os.remove(vid_path)
            os.remove(audio_path_rec)
            print(f"  recording saved: {out_path}")
        except FileNotFoundError:
            print(f"  ffmpeg not found, raw files kept: {vid_path}, {audio_path_rec}")
        print("done.")


if __name__ == "__main__":
    main()
