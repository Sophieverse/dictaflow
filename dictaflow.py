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
Groq = _require("groq", "groq").Groq

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CONFIG_DIR      = Path.home() / ".dictaflow"
CONFIG_FILE     = CONFIG_DIR / "config.json"
TRANSCRIPTS_DIR = Path.home() / "transcriptions"
SAMPLE_RATE     = 16000
CHANNELS        = 1

# The hold-to-talk key. right-⌥ (Option) is the default.
# Change to kb.Key.alt (left Option), kb.Key.ctrl_r, etc. if preferred.
TRIGGER_KEY = kb.Key.alt_r

DEFAULT_CONFIG: dict = {
    "groq_api_key":     "",
    "transcribe_model": "whisper-large-v3",
    "cleanup_model":    "llama-3.3-70b-versatile",
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
class AudioRecorder:
    def __init__(self):
        self._chunks: list = []
        self._stream = None
        self._active = False

    def start(self) -> None:
        self._chunks = []
        self._active = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
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
            self._stream.close()
            self._stream = None
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
# Groq pipeline
# ──────────────────────────────────────────────────────────────
def transcribe(wav_path: str, cfg: dict) -> str:
    client = Groq(api_key=cfg["groq_api_key"])
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=cfg["transcribe_model"],
            file=("audio.wav", f, "audio/wav"),
            response_format="text",
        )
    return (result if isinstance(result, str) else result.text).strip()


def cleanup(text: str, cfg: dict) -> str:
    if not cfg.get("cleanup_enabled") or not text:
        return text
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
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now()
    name = ts.strftime("%Y-%m-%d_%H%M%S") + ".md"
    path = TRANSCRIPTS_DIR / name
    lines = [f"# {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    if cleaned != raw:
        lines += [cleaned, "\n", "---\n", f"*Raw:* {raw}\n"]
    else:
        lines += [cleaned, "\n"]
    path.write_text("\n".join(lines))
    return path


# ──────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────
class DictaFlow:
    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.recorder = AudioRecorder()
        self._held    = False
        self._busy    = False

    # pynput calls these from its own thread, so we keep them lightweight
    def _on_press(self, key) -> None:
        if key == TRIGGER_KEY and not self._held and not self._busy:
            self._held = True
            print("\r🎙  Recording…          ", end="", flush=True)
            self.recorder.start()

    def _on_release(self, key) -> None:
        if key == TRIGGER_KEY and self._held:
            self._held = False
            audio = self.recorder.stop()
            if audio:
                threading.Thread(target=self._process, args=(audio,), daemon=True).start()
            else:
                print("\r(no audio captured)    ", end="", flush=True)

    def _process(self, audio_bytes: bytes) -> None:
        self._busy = True
        wav_path   = None
        try:
            print("\r⚙  Transcribing…       ", end="", flush=True)
            wav_path = AudioRecorder.to_wav(audio_bytes)
            raw      = transcribe(wav_path, self.cfg)

            if not raw:
                print("\r✗  No speech detected  ", flush=True)
                return

            print("\r✦  Cleaning up…        ", end="", flush=True)
            cleaned = cleanup(raw, self.cfg)

            paste_text(cleaned)
            path  = save_transcript(raw, cleaned)
            short = cleaned[:70] + ("…" if len(cleaned) > 70 else "")
            print(f"\r✓  {short}", flush=True)
            print(f"   → {path}", flush=True)

        except Exception as exc:
            print(f"\r✗  {exc}", flush=True)
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self._busy = False

    def run(self) -> None:
        print("DictaFlow is running.")
        print("  Hold right-⌥ (Option) to dictate; release to transcribe.")
        print(f"  Transcripts → {TRANSCRIPTS_DIR}/")
        print("  Ctrl+C to quit.\n")
        listener = kb.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        listener.start()
        try:
            listener.join()
        except KeyboardInterrupt:
            listener.stop()
            print("\nBye!")


# ──────────────────────────────────────────────────────────────
def main() -> None:
    cfg = load_config()
    if not cfg.get("groq_api_key"):
        cfg = setup_wizard(cfg)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    DictaFlow(cfg).run()


if __name__ == "__main__":
    main()
