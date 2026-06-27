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

# Local model folders.
TURBO_MODEL = "/Users/melod/dictaflow/models/whisper-large-v3-turbo"
SMALL_MODEL = "/Users/melod/dictaflow/models/whisper-small-mlx"

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
# Main app
# ──────────────────────────────────────────────────────────────
class DictaFlow:
    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.recorder   = AudioRecorder()
        self._held_key  = None    # which trigger key is currently down (or None)
        self._active    = None    # (name, model_path) for the in-flight recording
        self._busy      = False

    # pynput calls these from its own thread, so we keep them lightweight.
    # Each trigger key maps to its own model — hold the key for the model you want.
    def _on_press(self, key) -> None:
        # NOTE: exceptions raised here propagate out of pynput and kill the
        # listener (crashing the whole app), so we catch everything.
        try:
            if key in KEY_MODELS and self._held_key is None and not self._busy:
                self._held_key = key
                self._active   = KEY_MODELS[key]   # (name, path)
                print(f"\r🎙  Recording… [{self._active[0]}]\033[K", end="", flush=True)
                self.recorder.start()
        except Exception as exc:
            self._held_key = None
            self._active   = None
            # -9986 etc. = mic busy/unavailable (often another app holds it)
            print(f"\r✗  mic unavailable ({exc}); is another app using it?\033[K",
                  flush=True)

    def _on_release(self, key) -> None:
        try:
            if key == self._held_key:
                self._held_key = None
                audio = self.recorder.stop()
                if audio:
                    threading.Thread(target=self._process,
                                     args=(audio, self._active), daemon=True).start()
                else:
                    print("\r(no audio captured)\033[K", end="", flush=True)
        except Exception as exc:
            self._held_key = None
            print(f"\r✗  {exc}\033[K", flush=True)

    def _process(self, audio_bytes: bytes, model) -> None:
        self._busy = True
        wav_path   = None
        name, path = model
        # transcribe with the model bound to the key that was held
        cfg = {**self.cfg, "local_whisper_model": path}
        try:
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
            label = "right-⌥ (Option)" if key == kb.Key.alt_r else \
                    "right-⌘ (Command)" if key == kb.Key.cmd_r else str(key)
            print(f"  Hold {label} → {name}")
        print(f"  Transcripts → {TRANSCRIPT_FILE}")
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
