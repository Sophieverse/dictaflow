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
SAMPLE_RATE     = 16000
CHANNELS        = 1

# The hold-to-talk key. right-⌥ (Option) is the default.
# Change to kb.Key.alt (left Option), kb.Key.ctrl_r, etc. if preferred.
TRIGGER_KEY = kb.Key.alt_r

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
    # "\033[K" clears from the cursor to end-of-line so a shorter status line
    # never leaves leftover characters from a longer previous one.
    def _on_press(self, key) -> None:
        if key == TRIGGER_KEY and not self._held and not self._busy:
            self._held = True
            print("\r🎙  Recording…\033[K", end="", flush=True)
            self.recorder.start()

    def _on_release(self, key) -> None:
        if key == TRIGGER_KEY and self._held:
            self._held = False
            audio = self.recorder.stop()
            if audio:
                threading.Thread(target=self._process, args=(audio,), daemon=True).start()
            else:
                print("\r(no audio captured)\033[K", end="", flush=True)

    def _process(self, audio_bytes: bytes) -> None:
        self._busy = True
        wav_path   = None
        try:
            print("\r⚙  Transcribing…\033[K", end="", flush=True)
            wav_path = AudioRecorder.to_wav(audio_bytes)
            raw      = transcribe(wav_path, self.cfg)

            if not raw:
                print("\r✗  No speech detected (ignored)\033[K", flush=True)
                return

            cleaned = cleanup(raw, self.cfg)

            paste_text(cleaned)
            path  = save_transcript(raw, cleaned)
            short = cleaned[:70] + ("…" if len(cleaned) > 70 else "")
            print(f"\r✓  {short}\033[K", flush=True)
            print(f"   → {path}", flush=True)

        except Exception as exc:
            print(f"\r✗  {exc}\033[K", flush=True)
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self._busy = False

    def _warmup(self) -> None:
        """Load the model once at startup (on silence) so the FIRST real
        dictation is ~2s instead of ~4s. Runs off-thread; result discarded."""
        if self.cfg.get("backend") != "local":
            return
        try:
            silence = np.zeros(SAMPLE_RATE // 2, dtype="int16").tobytes()
            wp = AudioRecorder.to_wav(silence)
            transcribe(wp, self.cfg)
            os.unlink(wp)
            print("\r✓  Model ready.\033[K")
        except Exception:
            pass  # warmup is best-effort; real dictation will still work

    def run(self) -> None:
        print("DictaFlow is running.")
        print("  Hold right-⌥ (Option) to dictate; release to transcribe.")
        print(f"  Transcripts → {TRANSCRIPTS_DIR}/")
        print("  Ctrl+C to quit.\n")
        threading.Thread(target=self._warmup, daemon=True).start()
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
