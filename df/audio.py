"""Audio capture: a permanently-open stream with pre-roll, plus speech detection.

THREADING. PortAudio calls `_callback` on its own high-priority thread. That
callback must never block, allocate unboundedly, or raise — a raised exception
there kills the stream silently. So it does exactly one thing: append to a
deque under a lock. Everything else happens on the caller's thread.

The stream is opened once at startup and never stopped, which is a deliberate
change. Starting the mic on keydown loses the first syllable (Wispr Flow lists
"missing first words" as a known issue of its own for exactly this reason) and
re-negotiating a CoreAudio device costs 100ms+, which on a brief tap consumed
the entire recording. Instead we run continuously and keep a rolling pre-roll
buffer, so a recording can begin slightly in the *past*.
"""
from __future__ import annotations

import collections
import threading
import time

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, CHANNELS

# Frames per PortAudio callback. 1024 @ 16kHz = 64ms — small enough that the
# pre-roll and the level meter feel immediate, large enough to keep callback
# overhead negligible.
BLOCKSIZE = 1024


def list_input_devices() -> list[dict]:
    """All devices that can record, for the settings picker."""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"],
                        "channels": d["max_input_channels"],
                        "default_samplerate": d.get("default_samplerate")})
    return out


def builtin_mic_device() -> int | None:
    """Prefer the Mac's built-in mic over the system default.

    A connected Bluetooth headset becomes the default input, and CoreAudio
    switching it from A2DP (output-only, high quality) to HFP (bidirectional,
    8kHz) the instant something records is a reliable source of PortAudio
    -9986 errors. Returns None if no built-in mic is found, in which case the
    system default is used.
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and "MacBook" in d["name"] and "Microphone" in d["name"]:
            return i
    return None


class Recorder:
    """Continuously captures audio; hands out slices on demand.

    `start()` / `stop()` bracket a dictation. Between them, audio accumulates
    in `_capture`. Outside them, the last `preroll_ms` are retained in
    `_preroll` so `start()` can prepend them.
    """

    def __init__(self, *, device: int | None = None, preroll_ms: int = 500,
                 max_seconds: int = 1200, on_overflow=None):
        self.device       = device if device is not None else builtin_mic_device()
        self.max_samples  = int(max_seconds * SAMPLE_RATE)
        self._on_overflow = on_overflow

        preroll_blocks = max(1, int(preroll_ms / 1000 * SAMPLE_RATE / BLOCKSIZE))
        self._preroll  = collections.deque(maxlen=preroll_blocks)
        self._capture: list[np.ndarray] = []
        self._captured_samples = 0

        self._lock      = threading.Lock()
        self._active    = False
        self._stream    = None
        self._level     = 0.0     # RMS of the most recent block, 0..1
        self._started_at = 0.0
        self._truncated = False
        self._overflowed = False
        # Set when the mic delivers pure digital zeros, which is what macOS
        # does when microphone permission is denied — as opposed to an error.
        self._all_zero_blocks = 0
        self._total_blocks    = 0
        # Set once a device open has hung, so later attempts fail fast with an
        # explanation instead of wedging another thread the same way.
        self._open_wedged = False

    # ── lifecycle ───────────────────────────────────────────────
    def open(self) -> None:
        """Open the input stream. Raises if the device is unavailable."""
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCKSIZE,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def reset(self) -> None:
        """Tear the stream down so the next open() rebuilds it.

        Needed because a failed start() used to leave a dead handle in place
        forever: every later press re-raised on the same broken stream and the
        only recovery was restarting the app.
        """
        self.close()
        with self._lock:
            self._active = False
            self._capture = []
            self._captured_samples = 0
            self._preroll.clear()

    def ensure_open(self) -> None:
        """Open, and on failure reset so the next attempt can retry cleanly."""
        if self._open_wedged:
            raise RuntimeError(
                "the microphone did not respond last time and CoreAudio is "
                "still wedged; see the startup message")
        try:
            self.open()
        except Exception:
            self.reset()
            raise

    def open_with_timeout(self, timeout: float = 8.0) -> tuple[bool, str]:
        """Open the stream, giving up after `timeout` seconds.

        CoreAudio does not always fail an open — it BLOCKS. Observed here with
        a stack sitting in Pa_OpenStream → OpenAndSetupOneAudioUnit →
        AudioUnitSetProperty and never returning, which wedged the whole
        startup: no models loaded, no keys handled, and no message explaining
        why. There is no way to cancel a PortAudio open, so the attempt runs
        on a daemon thread we can abandon; the process still exits cleanly
        because daemon threads don't hold it open.

        Returns (ok, detail).
        """
        result: dict = {}

        def attempt():
            try:
                self.open()
                result["ok"] = True
            except Exception as exc:
                result["ok"] = False
                result["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=attempt, daemon=True, name="df-mic-open")
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self._open_wedged = True
            return False, (
                f"the microphone did not respond within {timeout:.0f}s. "
                f"CoreAudio is blocked inside the device open, which usually "
                f"means this process has no working Microphone permission, or "
                f"the audio daemon is holding a stale client after a hard kill. "
                f"Try: sudo killall coreaudiod   (it restarts automatically), "
                f"then re-grant Microphone in System Settings if it persists.")
        if result.get("ok"):
            return True, ""
        return False, result.get("error", "unknown error")

    # ── PortAudio callback thread ───────────────────────────────
    def _callback(self, indata, frames, time_info, status) -> None:
        # Never let an exception escape: PortAudio silently kills the stream.
        try:
            if status and status.input_overflow:
                # Frames were dropped because we couldn't keep up — words go
                # missing from the middle of the recording with no other sign.
                self._overflowed = True
                if self._on_overflow:
                    self._on_overflow()
            block = indata.copy()
            self._total_blocks += 1
            if not block.any():
                self._all_zero_blocks += 1
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2))) / 32768.0
            with self._lock:
                self._level = rms
                if self._active:
                    if self._captured_samples < self.max_samples:
                        self._capture.append(block)
                        self._captured_samples += len(block)
                    else:
                        self._truncated = True
                else:
                    self._preroll.append(block)
        except Exception:
            pass

    # ── control ─────────────────────────────────────────────────
    def start(self) -> None:
        """Begin a recording, seeded with the pre-roll."""
        self.ensure_open()
        with self._lock:
            self._capture = list(self._preroll)
            self._captured_samples = sum(len(b) for b in self._capture)
            self._preroll.clear()
            self._active = True
            self._truncated = False
            self._overflowed = False
            self._started_at = time.monotonic()

    def stop(self) -> np.ndarray:
        """End the recording and return int16 mono samples (possibly empty)."""
        with self._lock:
            self._active = False
            chunks, self._capture = self._capture, []
            self._captured_samples = 0
        if not chunks:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(chunks, axis=0).reshape(-1)

    def snapshot(self) -> np.ndarray:
        """Copy of the audio captured so far, without ending the recording.

        Used by the streaming transcriber to work on completed phrases while
        you keep talking.
        """
        with self._lock:
            if not self._capture:
                return np.empty(0, dtype=np.int16)
            return np.concatenate(list(self._capture), axis=0).reshape(-1)

    def consume(self, n_samples: int) -> None:
        """Drop the first `n_samples` from the in-progress capture.

        The streaming transcriber calls this once a prefix has been
        transcribed, which is also what bounds memory during a long
        hands-free session: the buffer never holds more than the untranscribed
        tail.
        """
        with self._lock:
            remaining = n_samples
            while remaining > 0 and self._capture:
                head = self._capture[0]
                if len(head) <= remaining:
                    remaining -= len(head)
                    self._captured_samples -= len(head)
                    self._capture.pop(0)
                else:
                    self._capture[0] = head[remaining:]
                    self._captured_samples -= remaining
                    remaining = 0

    # ── introspection ───────────────────────────────────────────
    @property
    def level(self) -> float:
        with self._lock:
            return self._level

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at if self._active else 0.0

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    def looks_muted(self) -> bool:
        """True when the mic has been returning nothing but digital zeros.

        macOS does not raise when microphone permission is denied — it hands
        you a stream of silence. Without this check the app reports "no speech
        detected" forever and the real cause is invisible.
        """
        return self._total_blocks > 10 and self._all_zero_blocks == self._total_blocks


# ──────────────────────────────────────────────────────────────
# Speech detection
#
# Decided on spectral SHAPE, not loudness. Amplitude is the wrong axis: on this
# mic, room tone at peak 331 makes Whisper hallucinate "Thank you.", while
# genuine quiet speech at peak 90 transcribes perfectly. No amplitude threshold
# separates those.
#
# Spectral flatness — geometric mean over arithmetic mean of the power spectrum
# — measures how noise-like a signal is. Noise spreads energy evenly across
# frequency (→ 1.0); speech concentrates it into formants (→ 0.0). Measured
# separation is about three orders of magnitude:
#     noise (silence, mic hiss, room tone)        0.561 – 1.000
#     speech, incl. quiet AND whispered           0.001
# Whispering stays firmly on the speech side because it removes the voiced
# pitch harmonics but keeps the formant structure, which is what this measures.
# ──────────────────────────────────────────────────────────────
FLATNESS_THRESHOLD = 0.15
DEAD_MIC_PEAK      = 8      # below this there is no signal at all, not even noise
FRAME              = 512


def _frames(x: np.ndarray) -> np.ndarray | None:
    if len(x) < FRAME * 4:
        return None
    return np.lib.stride_tricks.sliding_window_view(x, FRAME)[:: FRAME // 2]


def spectral_flatness(samples: np.ndarray) -> float:
    """Median flatness over the loudest frames of the clip.

    Restricted to loud frames on purpose: every real dictation begins and ends
    with a moment of silence while you find the key, and silence is perfectly
    flat. Averaging the whole clip lets that lead-in drag a genuine utterance
    up over the threshold.
    """
    x = samples.astype(np.float32) / 32768.0
    frames = _frames(x)
    if frames is None:
        return 1.0                      # too short to judge — treat as noise
    energy = (frames ** 2).sum(axis=1)
    loud   = frames[energy >= np.percentile(energy, 70)]
    spec   = np.abs(np.fft.rfft(loud * np.hanning(FRAME), axis=1)) ** 2 + 1e-12
    flat   = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    return float(np.median(flat))


def has_speech(samples: np.ndarray) -> bool:
    """True if this clip contains something worth sending to Whisper."""
    if samples.size == 0:
        return False
    if int(np.abs(samples).max(initial=0)) < DEAD_MIC_PEAK:
        return False
    return spectral_flatness(samples) < FLATNESS_THRESHOLD


def find_pause(samples: np.ndarray, *, min_pause_secs: float = 0.35,
               search_from: float = 0.0) -> int | None:
    """Index of a good place to cut the audio, or None.

    "Good" means the middle of a silent stretch at least `min_pause_secs`
    long, so a streaming chunk boundary never lands mid-word. Searching only
    after `search_from` seconds keeps us from cutting off the very beginning.

    Returns a sample index, or None if there is no such pause.
    """
    x = samples.astype(np.float32) / 32768.0
    frames = _frames(x)
    if frames is None:
        return None
    hop    = FRAME // 2
    energy = np.sqrt((frames ** 2).mean(axis=1))
    if energy.size == 0:
        return None
    # Threshold relative to this clip's own loudness, so it adapts to a
    # whisper as readily as to a shout. The floor keeps pure digital silence
    # from producing a degenerate threshold of zero.
    speech_level = float(np.percentile(energy, 90))
    thresh = max(speech_level * 0.12, 1e-4)

    min_frames = max(1, int(min_pause_secs * SAMPLE_RATE / hop))
    start_frame = int(search_from * SAMPLE_RATE / hop)

    best = None
    run_start = None
    for i in range(start_frame, len(energy)):
        if energy[i] < thresh:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_frames:
                best = (run_start + i) // 2      # middle of the pause
            run_start = None
    if run_start is not None and len(energy) - run_start >= min_frames:
        best = (run_start + len(energy)) // 2
    if best is None:
        return None
    idx = best * hop
    return int(min(idx, len(samples)))
