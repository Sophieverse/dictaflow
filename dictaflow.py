#!/usr/bin/env python3
"""
DictaFlow — AI dictation powered by Groq Whisper.
Hold right-⌥ (Option) to record; release to transcribe and paste.
Transcripts are saved to ~/transcriptions/ as dated Markdown files.
"""

import os
import sys
import json
import wave
import time
import datetime
import threading
import tempfile
import subprocess
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Hard deps — fail fast with a helpful message
# ──────────────────────────────────────────────────────────────
def _require(pkg, install):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"Missing dependency: pip install {install}")
        sys.exit(1)

sd  = _require("sounddevice", "sounddevice")
np  = _require("numpy",       "numpy")
_pynput_kb = _require("pynput.keyboard", "pynput")
kb = _pynput_kb  # pynput.keyboard module — Key, Listener live here
# Backend-specific deps (groq, mlx_whisper) are imported lazily inside the
# transcribe/cleanup functions so you only need the ones your backend uses.

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CONFIG_DIR      = Path.home() / ".dictaflow"
CONFIG_FILE     = CONFIG_DIR / "config.json"
TRANSCRIPTS_DIR = Path.home() / "transcriptions"
TRANSCRIPT_FILE = TRANSCRIPTS_DIR / "transcripts.md"  # single rolling log
SAMPLE_RATE     = 16000
CHANNELS        = 1

# Local model folders, resolved next to this script so a fresh clone works
# without editing paths. mlx_whisper also accepts a Hugging Face repo id here
# (e.g. "mlx-community/whisper-large-v3-turbo") if you'd rather it download.
MODELS_DIR  = Path(__file__).resolve().parent / "models"
TURBO_MODEL = str(MODELS_DIR / "whisper-large-v3-turbo")
SMALL_MODEL = str(MODELS_DIR / "whisper-small-mlx")

# Hold-to-talk keys → which model transcribes. Hold a key, speak, release.
#   right-⌥ (Option)  → Turbo, most accurate (~1.8s)
#   right-⌘ (Command) → Small, fastest       (~0.5s)
# (MacBook keyboards have no right-Control, so right-Command is the 2nd key.)
KEY_MODELS = {
    kb.Key.alt_r: ("Turbo", TURBO_MODEL),
    kb.Key.cmd_r: ("Small", SMALL_MODEL),
}

DEFAULT_CONFIG: dict = {
    # "local" = open-weight models on your Mac (mlx-whisper + Ollama).
    # "groq"  = Groq cloud API (needs groq_api_key below).
    "backend": "local",

    # ── local backend ──────────────────────────────────────────
    # Whisper model run via Apple MLX. "turbo" is fast + accurate;
    # use ".../whisper-large-v3-mlx" for max accuracy, or
    # ".../whisper-small-mlx" / "...-tiny-mlx" for max speed.
    "local_whisper_model": "mlx-community/whisper-large-v3-turbo",
    # Cleanup via a local Ollama model. Empty string OR cleanup_enabled
    # = False skips cleanup entirely (Whisper already punctuates well).
    "ollama_host":  "http://localhost:11434",
    "ollama_model": "gpt-oss:20b",

    # ── groq backend ───────────────────────────────────────────
    "groq_api_key":     "",
    "transcribe_model": "whisper-large-v3",
    "cleanup_model":    "llama-3.3-70b-versatile",

    # ── shared ─────────────────────────────────────────────────
    "cleanup_enabled":  True,
    "cleanup_prompt": (
        "You are a transcription cleanup assistant. "
        "Fix punctuation, capitalization, and obvious speech-recognition errors. "
        "Preserve the speaker's exact words and meaning. "
        "Return only the cleaned text — no explanation, no quotes."
    ),
}


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def setup_wizard(cfg: dict) -> dict:
    print("\n─── DictaFlow First-Run Setup ─────────────────────")
    print("  Get a FREE Groq API key at: https://console.groq.com")
    key = input("  Groq API key: ").strip()
    if not key:
        print("No key entered — exiting.")
        sys.exit(1)
    cfg["groq_api_key"] = key
    save_config(cfg)
    print(f"  Config saved → {CONFIG_FILE}")
    print("───────────────────────────────────────────────────\n")
    return cfg


# ──────────────────────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────────────────────
def _builtin_mic_device() -> int | None:
    """Pin recording to the Mac's built-in mic instead of whatever the OS
    'default' input device is. Bluetooth headsets (AirPods, WH-1000XM4, …)
    become the default input the moment they're connected, and CoreAudio
    renegotiating their profile (output-only A2DP -> bidirectional HFP) the
    instant something tries to record from them is a common source of AUHAL
    'Invalid Property Value' / PortAudio -9986 errors. Falls back to the
    system default if no built-in mic is found (e.g. on a different Mac)."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "MacBook" in d["name"] and "Microphone" in d["name"]:
            return i
    return None


class AudioRecorder:
    def __init__(self):
        self._chunks: list = []
        self._stream = None
        self._active = False

    def start(self) -> None:
        self._chunks = []
        self._active = True
        if self._stream is None:
            # Opened once and reused across presses — negotiating a fresh
            # CoreAudio device on every keypress (~100ms+) was eating the
            # entire buffer on brief taps, so stop() saw zero frames.
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                device=_builtin_mic_device(),
                callback=self._cb,
            )
        self._stream.start()

    def _cb(self, indata, frames, time_info, status) -> None:
        if self._active:
            self._chunks.append(indata.copy())

    def stop(self) -> bytes | None:
        self._active = False
        if self._stream:
            self._stream.stop()
        if not self._chunks:
            return None
        return np.concatenate(self._chunks, axis=0).tobytes()

    @staticmethod
    def to_wav(audio_bytes: bytes) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        return tmp.name


# ──────────────────────────────────────────────────────────────
# Transcription — dispatches to the configured backend
# ──────────────────────────────────────────────────────────────
def transcribe(wav_path: str, cfg: dict) -> str:
    if cfg.get("backend") == "local":
        return _transcribe_local(wav_path, cfg)
    return _transcribe_groq(wav_path, cfg)


def _transcribe_local(wav_path: str, cfg: dict) -> str:
    import mlx_whisper  # lazy: only needed for the local backend
    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=cfg["local_whisper_model"],
        condition_on_previous_text=False,  # stops repetition loops carrying forward
    )
    text = result["text"].strip()
    return "" if _is_repetition_hallucination(text) else text


def _is_repetition_hallucination(text: str) -> bool:
    """Whisper on short/near-silent clips degenerates into one word repeated
    (e.g. 'Red Red Red…'). Detect that and discard so we never paste garbage."""
    words = text.split()
    if len(words) < 12:
        return False
    from collections import Counter
    _, top = Counter(w.lower() for w in words).most_common(1)[0]
    return top / len(words) > 0.5


SILENCE_PEAK_THRESHOLD = 300  # int16 peak; see check_audio.py (peak<50 = no mic signal)


def _is_near_silent(audio_bytes: bytes) -> bool:
    """A too-brief key-tap often captures only mic noise, not speech. Whisper
    doesn't reliably return empty text for that — it sometimes hallucinates a
    stock phrase from its training data instead (e.g. 'Thank you.',
    'Thanks for watching.'). Catch it on peak amplitude *before* calling
    Whisper so we never feed it near-silent audio in the first place."""
    peak = int(np.abs(np.frombuffer(audio_bytes, dtype="int16")).max(initial=0))
    return peak < SILENCE_PEAK_THRESHOLD


def _transcribe_groq(wav_path: str, cfg: dict) -> str:
    from groq import Groq  # lazy: only needed for the groq backend
    client = Groq(api_key=cfg["groq_api_key"])
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=cfg["transcribe_model"],
            file=("audio.wav", f, "audio/wav"),
            response_format="text",
        )
    return (result if isinstance(result, str) else result.text).strip()


# ──────────────────────────────────────────────────────────────
# Cleanup — dispatches to the configured backend
# ──────────────────────────────────────────────────────────────
def cleanup(text: str, cfg: dict) -> str:
    if not cfg.get("cleanup_enabled") or not text:
        return text
    if cfg.get("backend") == "local":
        if not cfg.get("ollama_model"):
            return text  # cleanup turned off by clearing the model name
        return _cleanup_ollama(text, cfg)
    return _cleanup_groq(text, cfg)


def _cleanup_ollama(text: str, cfg: dict) -> str:
    import urllib.request  # stdlib — no extra dependency for local cleanup
    payload = json.dumps({
        "model": cfg["ollama_model"],
        "messages": [
            {"role": "system", "content": cfg["cleanup_prompt"]},
            {"role": "user",   "content": text},
        ],
        "stream": False,
        "think": False,          # skip reasoning output on thinking models
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(
        cfg["ollama_host"].rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip() or text


def _cleanup_groq(text: str, cfg: dict) -> str:
    from groq import Groq  # lazy: only needed for the groq backend
    client = Groq(api_key=cfg["groq_api_key"])
    resp = client.chat.completions.create(
        model=cfg["cleanup_model"],
        messages=[
            {"role": "system", "content": cfg["cleanup_prompt"]},
            {"role": "user",   "content": text},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────
# Paste + save
# ──────────────────────────────────────────────────────────────
def paste_text(text: str) -> None:
    """Copy text to clipboard then simulate ⌘V into the active field."""
    # Save existing clipboard so we can restore it after pasting
    old = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
    subprocess.run(["pbcopy"], input=text, text=True, check=True)
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using command down'],
        capture_output=True,
    )
    # Brief delay so the paste lands before we restore the clipboard
    time.sleep(0.2)
    subprocess.run(["pbcopy"], input=old, text=True, check=True)


def save_transcript(raw: str, cleaned: str) -> Path:
    """Append one dated section to the single rolling transcripts.md."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now()
    with open(TRANSCRIPT_FILE, "a") as f:
        if f.tell() == 0:                       # brand-new file → add a title
            f.write("# Transcripts\n")
        f.write(f"\n## {ts.strftime('%Y-%m-%d %H:%M:%S')}\n\n{cleaned}\n")
        if cleaned != raw:                      # only when LLM cleanup altered it
            f.write(f"\n*Raw:* {raw}\n")
    return TRANSCRIPT_FILE


# ──────────────────────────────────────────────────────────────
# Bubble indicator — small floating pill, bottom-center of the screen.
# Visible whenever DictaFlow is doing something to your audio: recording
# (held key OR hands-free) and transcribing. If no pill, it isn't listening.
#
# Threading: every caller below (pynput's listener thread, the transcription
# worker thread) only ever writes a *desired state* under a lock. The actual
# AppKit calls happen exclusively in pump(), which runs on the main thread —
# AppKit is not thread-safe and calling it off-main silently misbehaves.
# ──────────────────────────────────────────────────────────────
RECORDING_COLOR    = (0.85, 0.20, 0.20)   # red   — capturing audio
TRANSCRIBING_COLOR = (0.95, 0.62, 0.10)   # amber — working on it


class BubbleWindow:
    def __init__(self):
        self._ok = False
        self._lock    = threading.Lock()
        self._desired = None   # (text, rgb) we want on screen, or None for hidden
        self._shown   = None   # what pump() has actually put on screen
        try:
            from Cocoa import (
                NSApplication, NSWindow, NSColor, NSMakeRect,
                NSBackingStoreBuffered, NSWindowStyleMaskBorderless,
                NSFloatingWindowLevel, NSScreen, NSTextField,
                NSApplicationActivationPolicyAccessory, NSAnyEventMask,
            )
            import Quartz
            self._Quartz = Quartz
            self._mask   = NSAnyEventMask

            self._NSColor = NSColor
            self._app = NSApplication.sharedApplication()
            self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

            w, h   = 220, 40
            screen = NSScreen.mainScreen().frame()
            x, y   = (screen.size.width - w) / 2, 50
            self._win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(x, y, w, h), NSWindowStyleMaskBorderless,
                NSBackingStoreBuffered, False,
            )
            self._win.setLevel_(NSFloatingWindowLevel)
            self._win.setOpaque_(False)
            self._win.setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.2, 0.2, 0.92)
            )
            self._win.setHasShadow_(True)
            cv = self._win.contentView()
            cv.setWantsLayer_(True)
            try:
                cv.layer().setCornerRadius_(h / 2)   # pill shape; cosmetic only
            except Exception:
                pass

            self._label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 11, w, 18))
            self._label.setBezeled_(False)
            self._label.setDrawsBackground_(False)
            self._label.setEditable_(False)
            self._label.setSelectable_(False)
            self._label.setAlignment_(2)  # NSTextAlignmentCenter
            self._label.setTextColor_(NSColor.whiteColor())
            cv.addSubview_(self._label)
            self._ok = True
        except Exception as exc:
            print(f"(bubble indicator unavailable, continuing without it: {exc})")

    def set(self, text: str | None, color=RECORDING_COLOR) -> None:
        """Ask for the pill to show `text` (or hide it, when text is None).
        Safe to call from ANY thread — nothing touches AppKit here, pump()
        picks the change up on the main thread within one tick (~50ms)."""
        with self._lock:
            self._desired = None if text is None else (text, color)

    # Kept for readability at the call sites.
    def show(self, text: str, color=RECORDING_COLOR) -> None:
        self.set(text, color)

    def hide(self) -> None:
        self.set(None)

    def _reconcile(self) -> None:
        """Main thread only: make the screen match whatever set() last asked for."""
        with self._lock:
            desired = self._desired
        if desired == self._shown:
            return
        if desired is None:
            self._win.orderOut_(None)
        else:
            text, (r, g, b) = desired
            self._label.setStringValue_(text)
            self._win.setBackgroundColor_(
                self._NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.92)
            )
            self._win.orderFrontRegardless()
            self._win.displayIfNeeded()
        self._shown = desired

    def pump(self, seconds: float = 0.05) -> None:
        """Drain pending AppKit events for a short window and apply any pending
        state change. Call this from the main-thread loop in place of a bare
        sleep — pynput's listener runs in its own thread, so this is the only
        thing keeping the bubble alive."""
        if not self._ok:
            time.sleep(seconds)
            return
        self._reconcile()
        until = self._Quartz.NSDate.dateWithTimeIntervalSinceNow_(seconds)
        event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
            self._mask, until, "kCFRunLoopDefaultMode", True,
        )
        if event is not None:
            self._app.sendEvent_(event)


def _key_label(key) -> str:
    return "right-⌥ (Option)" if key == kb.Key.alt_r else \
           "right-⌘ (Command)" if key == kb.Key.cmd_r else str(key)


# A press+release shorter than this counts as a "tap" (a click, not a real
# dictation hold); two taps within the window below count as a double-click.
TAP_MAX_HOLD      = 0.25
DOUBLE_TAP_WINDOW = 0.4


# ──────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────
class DictaFlow:
    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.recorder   = AudioRecorder()
        self.bubble     = BubbleWindow()
        self._held_key  = None    # which trigger key is currently down (or None)
        self._active    = None    # (name, model_path) for the in-flight recording
        self._busy      = False
        self._press_time      = {}   # key -> monotonic time of its current press
        self._last_tap_release = {}  # key -> monotonic time of its last quick tap
        self._handsfree_key    = None  # key currently in toggled hands-free mode
        self._pending_tap_timer = {}  # key -> Timer waiting to see if a 2nd tap follows

    # pynput calls these from its own thread, so we keep them lightweight.
    # Each trigger key maps to its own model — hold the key for the model you want.
    # Double-tapping a key (two quick taps within DOUBLE_TAP_WINDOW) instead
    # toggles hands-free mode: recording keeps going without holding anything
    # down, shows the bubble, and a single tap of the same key stops it.
    def _on_press(self, key) -> None:
        # NOTE: exceptions raised here propagate out of pynput and kill the
        # listener (crashing the whole app), so we catch everything.
        try:
            if key not in KEY_MODELS:
                return
            if self._handsfree_key is not None:
                if key == self._handsfree_key:
                    self._press_time[key] = time.monotonic()
                return  # already listening hands-free; ignore other keys too
            if self._held_key is None and not self._busy:
                self._held_key = key
                self._press_time[key] = time.monotonic()
                self._active   = KEY_MODELS[key]   # (name, path)
                print(f"\r🎙  Recording… [{self._active[0]}]\033[K", end="", flush=True)
                self.bubble.show(f"🎙 Recording… ({self._active[0]})", RECORDING_COLOR)
                self.recorder.start()
        except Exception as exc:
            self._held_key = None
            self._active   = None
            self.bubble.hide()
            # -9986 etc. = mic busy/unavailable (often another app holds it)
            print(f"\r✗  mic unavailable ({exc}); is another app using it?\033[K",
                  flush=True)

    def _on_release(self, key) -> None:
        try:
            if key not in KEY_MODELS:
                return
            now = time.monotonic()

            if self._handsfree_key == key:
                self._handsfree_key = None
                self._press_time.pop(key, None)
                self._last_tap_release.pop(key, None)
                self.bubble.hide()
                audio = self.recorder.stop()
                if audio:
                    threading.Thread(target=self._process,
                                     args=(audio, self._active), daemon=True).start()
                else:
                    print("\r(no audio captured)\033[K", end="", flush=True)
                return

            if key != self._held_key:
                return
            self._held_key   = None
            hold_duration    = now - self._press_time.get(key, now)

            if hold_duration < TAP_MAX_HOLD:
                last_tap = self._last_tap_release.get(key)
                if last_tap is not None and (now - last_tap) < DOUBLE_TAP_WINDOW:
                    # Double-click: cancel tap #1's pending finalize (its audio
                    # was just the click noise, not real speech — discard it).
                    # The recorder is already running (started by this second
                    # tap's press) — let it keep going into hands-free mode
                    # instead of stopping/restarting it.
                    timer = self._pending_tap_timer.pop(key, None)
                    if timer:
                        timer.cancel()
                    self._last_tap_release.pop(key, None)
                    self._handsfree_key = key
                    self.bubble.show(f"🎙 Listening… ({self._active[0]})")
                    print(f"\r🎙  Hands-free… [{self._active[0]}] "
                          f"(tap {_key_label(key)} again to stop)\033[K",
                          end="", flush=True)
                    return

                # Lone tap so far — don't transcribe yet. If it's actually
                # about to be a double-click, doing so immediately would set
                # self._busy while we're transcribing, silently swallowing
                # the second tap's press before we ever see it. Wait out the
                # double-click window first; _finalize_tap transcribes only
                # if no second tap arrives to cancel it.
                self._last_tap_release[key] = now
                audio = self.recorder.stop()
                self.bubble.hide()
                timer = threading.Timer(DOUBLE_TAP_WINDOW, self._finalize_tap,
                                        args=(key, audio, self._active))
                timer.daemon = True
                self._pending_tap_timer[key] = timer
                timer.start()
                return

            self._last_tap_release.pop(key, None)  # a real hold breaks any pending double-click
            audio = self.recorder.stop()
            self.bubble.hide()   # _process re-shows it in "transcribing" amber
            if audio:
                threading.Thread(target=self._process,
                                 args=(audio, self._active), daemon=True).start()
            else:
                print("\r(no audio captured)\033[K", end="", flush=True)
        except Exception as exc:
            self._held_key = None
            self.bubble.hide()
            print(f"\r✗  {exc}\033[K", flush=True)

    def _finalize_tap(self, key, audio, active) -> None:
        """Runs DOUBLE_TAP_WINDOW after a lone tap's release, unless a second
        tap cancelled it first (see the double-click branch in _on_release)."""
        self._pending_tap_timer.pop(key, None)
        if audio:
            threading.Thread(target=self._process,
                             args=(audio, active), daemon=True).start()
        else:
            print("\r(no audio captured)\033[K", end="", flush=True)

    def _process(self, audio_bytes: bytes, model) -> None:
        self._busy = True
        wav_path   = None
        name, path = model
        # transcribe with the model bound to the key that was held
        cfg = {**self.cfg, "local_whisper_model": path}
        try:
            self.bubble.show(f"⚙ Transcribing… ({name})", TRANSCRIBING_COLOR)
            if _is_near_silent(audio_bytes):
                print("\r✗  No speech detected (ignored)\033[K", flush=True)
                return

            print(f"\r⚙  Transcribing… [{name}]\033[K", end="", flush=True)
            wav_path = AudioRecorder.to_wav(audio_bytes)
            raw      = transcribe(wav_path, cfg)

            if not raw:
                print("\r✗  No speech detected (ignored)\033[K", flush=True)
                return

            cleaned = cleanup(raw, cfg)

            paste_text(cleaned)
            tpath = save_transcript(raw, cleaned)
            short = cleaned[:70] + ("…" if len(cleaned) > 70 else "")
            print(f"\r✓  [{name}] {short}\033[K", flush=True)
            print(f"   → {tpath}", flush=True)

        except Exception as exc:
            print(f"\r✗  {exc}\033[K", flush=True)
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self.bubble.hide()
            self._busy = False

    def _warmup(self) -> None:
        """Pre-load each model (on silence) so the first dictation with either
        key is fast. Runs off-thread; results discarded."""
        if self.cfg.get("backend") != "local":
            return
        for name, path in {v[0]: v[1] for v in KEY_MODELS.values()}.items():
            try:
                silence = np.zeros(SAMPLE_RATE // 2, dtype="int16").tobytes()
                wp = AudioRecorder.to_wav(silence)
                transcribe(wp, {**self.cfg, "local_whisper_model": path})
                os.unlink(wp)
                print(f"\r✓  {name} model ready.\033[K")
            except Exception:
                pass  # warmup is best-effort; real dictation will still work

    def run(self) -> None:
        print("DictaFlow is running.")
        for key, (name, _) in KEY_MODELS.items():
            print(f"  Hold {_key_label(key)} → {name}")
            print(f"  Double-tap {_key_label(key)} → hands-free {name} (tap once to stop)")
        print(f"  Transcripts → {TRANSCRIPT_FILE}")
        print("  Ctrl+C to quit.\n")
        threading.Thread(target=self._warmup, daemon=True).start()
        listener = kb.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        listener.start()
        try:
            # pynput's listener runs in its own thread; this thread instead
            # pumps AppKit's event loop so the bubble window can actually
            # draw/update/respond while dictation happens in the background.
            while listener.running:
                self.bubble.pump()
        except KeyboardInterrupt:
            listener.stop()
            print("\nBye!")


# ──────────────────────────────────────────────────────────────
def _check_ollama(cfg: dict) -> None:
    """Warn (don't abort) if local cleanup is on but Ollama isn't reachable."""
    if not (cfg.get("cleanup_enabled") and cfg.get("ollama_model")):
        return
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(
            cfg["ollama_host"].rstrip("/") + "/api/tags", timeout=2
        )
    except Exception:
        print("⚠  Ollama not reachable — cleanup will fail. Start it with: ollama serve")
        print("   (or set cleanup_enabled=false in ~/.dictaflow/config.json)\n")


def main() -> None:
    cfg = load_config()
    if cfg.get("backend") == "groq" and not cfg.get("groq_api_key"):
        cfg = setup_wizard(cfg)
    if cfg.get("backend") == "local":
        _check_ollama(cfg)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    DictaFlow(cfg).run()


if __name__ == "__main__":
    main()
