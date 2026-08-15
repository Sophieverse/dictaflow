"""Configuration: defaults, load/save, validation, migration.

Two rules the rest of the code depends on:

1. `load()` NEVER raises and never returns a partial dict. A corrupt config
   must not stop you dictating. On a parse failure we keep the broken file
   (renamed) so nothing is silently destroyed, and carry on with defaults.

2. `save()` is atomic. The previous version used `write_text`, which truncates
   before it writes — a crash mid-write produced a truncated file, which the
   next read treated as corrupt, which the next save then finalised into total
   loss of the API keys. Temp file + os.replace makes the swap atomic.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

CONFIG_DIR      = Path.home() / ".dictaflow"
CONFIG_FILE     = CONFIG_DIR / "config.json"
TRANSCRIPTS_DIR = Path.home() / "transcriptions"
TRANSCRIPT_FILE = TRANSCRIPTS_DIR / "transcripts.md"
EVENTS_FILE     = TRANSCRIPTS_DIR / "events.jsonl"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SAMPLE_RATE = 16000
CHANNELS    = 1

SCHEMA_VERSION = 2

DEFAULTS: dict = {
    "version": SCHEMA_VERSION,

    # ── engine ──────────────────────────────────────────────────
    "backend": "local",                     # "local" | "groq"
    # Two models bound to two keys, so you can trade accuracy for speed
    # per-utterance without opening settings.
    "models": {
        "turbo": str(MODELS_DIR / "whisper-large-v3-turbo"),
        "small": str(MODELS_DIR / "whisper-small-mlx"),
    },
    # Which physical key triggers which model. MacBook keyboards have no
    # right-Control on the built-in keyboard, so right-⌘ is the second slot;
    # "command" is the rewrite-the-selection mode.
    "bindings": {"turbo": "alt_r", "small": "cmd_r", "command": "ctrl_r"},

    "language": "en",                       # null = auto-detect (costs ~0.9s)
    "languages": ["en"],                    # offered in the bar's picker
    "initial_prompt": "",

    # ── latency ─────────────────────────────────────────────────
    # Transcribe finished phrases WHILE you are still speaking, so on release
    # only the tail is left to process. See asr.py for why this matters so
    # much: cost is flat up to ~20s of audio and then explodes.
    "streaming": True,
    "chunk_target_secs": 12.0,              # cut here if a pause allows
    "chunk_max_secs": 20.0,                 # hard cut even mid-phrase
    # Whisper re-decodes a failing window at each temperature in this ladder.
    # The stock ladder has six rungs and measured 10.64s vs 1.59s greedy on a
    # 29s clip, with identical (still-bad) output. Two rungs keeps the escape
    # hatch for a genuine bad decode without the 6x worst case.
    "temperature_ladder": [0.0, 0.2],

    # ── capture ─────────────────────────────────────────────────
    # Wispr Flow's own docs list "missing first words" as a known issue. It is
    # caused by starting the mic on keydown: the first syllable is already in
    # the air. We keep the stream permanently open and retain this much audio
    # from BEFORE the key went down.
    "preroll_ms": 500,
    "max_session_seconds": 1200,            # 20 min, matching Wispr
    "warn_session_seconds": 1140,           # warn at 19 min
    "input_device": None,                   # None = built-in mic

    # ── formatting ──────────────────────────────────────────────
    "cleanup_level": "medium",              # none | light | medium | high
    "spoken_punctuation": True,
    "backtrack": True,
    "auto_lists": True,
    "context_awareness": True,
    "smart_capitalization": True,
    "strip_trailing_period_in_chat": True,
    "dictionary": [],                       # [{"from": ..., "to": ...}]
    "snippets": [],                         # [{"trigger": ..., "text": ...}]

    # ── output ──────────────────────────────────────────────────
    "paste_method": "clipboard",            # clipboard | keystroke
    "restore_clipboard": True,
    "sounds": True,
    "sound_start":  "/System/Library/Sounds/Tink.aiff",
    "sound_done":   "/System/Library/Sounds/Pop.aiff",
    "sound_cancel": "/System/Library/Sounds/Funk.aiff",

    # ── UI ──────────────────────────────────────────────────────
    "bar_dock": "bottom",                   # bottom | left | right
    "bar_offset": 60,
    "show_waveform": True,
    "show_word_count": True,
    "menu_bar_icon": True,

    # ── command mode (rewrite the selection by voice) ────────────
    "command_backend": "ollama",            # ollama | groq | none
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.2:3b",
    "command_prompt": (
        "You rewrite text according to an instruction. "
        "Return ONLY the rewritten text — no preamble, no explanation, "
        "no quotes around it. Preserve the author's voice."
    ),

    # ── groq backend ────────────────────────────────────────────
    "groq_api_key": "",
    "transcribe_model": "whisper-large-v3",

    # ── first run ───────────────────────────────────────────────
    "onboarded": False,
}

# Keys the dashboard is allowed to write. Everything absent from this set is
# unreachable from the browser. `models` is deliberately NOT here: it feeds
# mlx_whisper's `path_or_hf_repo`, which downloads and loads an arbitrary
# Hugging Face repo — and since we type the result as keystrokes into the
# focused app, letting a web page choose the model is keystroke injection.
EDITABLE = {
    "language", "languages", "initial_prompt", "cleanup_level",
    "spoken_punctuation", "backtrack", "auto_lists", "context_awareness",
    "smart_capitalization", "strip_trailing_period_in_chat",
    "dictionary", "snippets", "sounds", "streaming", "preroll_ms", "bindings",
    "bar_dock", "show_waveform", "show_word_count", "paste_method",
    "restore_clipboard", "max_session_seconds", "onboarded",
    "command_backend", "ollama_host", "ollama_model",
}

# Type/range validation for everything editable, so a malformed POST (or a
# hand-edited file) can't put the agent into a state where it throws on every
# dictation. Each entry: (predicate, coercion-or-None).
_VALIDATORS = {
    "cleanup_level":  lambda v: v in ("none", "light", "medium", "high"),
    "bar_dock":       lambda v: v in ("bottom", "left", "right"),
    "paste_method":   lambda v: v in ("clipboard", "keystroke"),
    "command_backend": lambda v: v in ("ollama", "groq", "none"),
    "language":       lambda v: v is None or (isinstance(v, str) and len(v) <= 8),
    "languages":      lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "initial_prompt": lambda v: isinstance(v, str) and len(v) <= 2000,
    "preroll_ms":     lambda v: _is_int(v) and 0 <= v <= 3000,
    "max_session_seconds": lambda v: _is_int(v) and 10 <= v <= 7200,
    "ollama_host":    lambda v: isinstance(v, str) and v.startswith(("http://", "https://")),
    "ollama_model":   lambda v: isinstance(v, str) and len(v) <= 120,
    "dictionary":     lambda v: _valid_pairs(v, "from", "to", 200),
    "snippets":       lambda v: _valid_pairs(v, "trigger", "text", 4000),
    "bindings":       lambda v: _valid_bindings(v),
}


def _valid_bindings(v) -> bool:
    """Bindings must name keys the router actually knows about.

    Checked here rather than at use time because an unknown key name would
    otherwise fail silently: the router would simply never fire for that slot
    and the key would appear dead with no explanation anywhere.
    """
    if not isinstance(v, dict) or not v:
        return False
    from .hotkeys import KEY_NAMES
    for slot, key in v.items():
        if slot not in ("turbo", "small", "command"):
            return False
        if not isinstance(key, str) or key not in KEY_NAMES:
            return False
    # Two slots on one key would make one of them unreachable.
    return len(set(v.values())) == len(v)


def _is_int(v) -> bool:
    """A real integer, not a bool.

    `isinstance(True, int)` is True in Python, so without this a POST of
    `{"preroll_ms": true}` sailed through validation and set the pre-roll to
    1ms — verified against the running server. bool is a subclass of int, so
    every int check in this file needs the explicit exclusion.
    """
    return isinstance(v, int) and not isinstance(v, bool)


def _valid_pairs(v, key_a: str, key_b: str, max_len: int) -> bool:
    if not isinstance(v, list) or len(v) > 5000:
        return False
    for item in v:
        if not isinstance(item, dict):
            return False
        a, b = item.get(key_a), item.get(key_b)
        if not isinstance(a, str) or not isinstance(b, str):
            return False
        if not a.strip() or len(a) > 200 or len(b) > max_len:
            return False
    return True


def validate(key: str, value) -> bool:
    """True if `value` is an acceptable setting for `key`."""
    if key not in DEFAULTS:
        return False
    check = _VALIDATORS.get(key)
    if check is not None:
        try:
            return bool(check(value))
        except Exception:
            return False
    # No explicit rule: require the same type as the default, which catches
    # the common case of a bool arriving as the string "false".
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return _is_int(value)           # again: bool is a subclass of int
    if default is None:
        return True
    return isinstance(value, type(default)) and not isinstance(value, bool)


def _migrate(raw: dict) -> dict:
    """Bring a v1 config (flat keys, different names) up to the v2 schema."""
    if raw.get("version") == SCHEMA_VERSION:
        return raw
    out = dict(raw)
    # v1 stored a single active model path under `local_whisper_model`; v2
    # keeps a name->path map so the two hotkeys are configurable separately.
    legacy = out.pop("local_whisper_model", None)
    if legacy and "models" not in out:
        models = dict(DEFAULTS["models"])
        if "small" in str(legacy):
            models["small"] = legacy
        else:
            models["turbo"] = legacy
        out["models"] = models
    # v1's cleanup was a boolean that ran an LLM over the text (~40s). v2's
    # levels are rule-based and instant, so the old flag maps to a level.
    if "cleanup_enabled" in out:
        enabled = out.pop("cleanup_enabled")
        out.setdefault("cleanup_level", "medium" if enabled else "light")
    out.pop("cleanup_prompt", None)
    out.pop("cleanup_model", None)
    out["version"] = SCHEMA_VERSION
    return out


def load() -> dict:
    """Read the config, falling back to defaults. Never raises."""
    cfg = copy.deepcopy(DEFAULTS)
    if not CONFIG_FILE.exists():
        return cfg
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(raw, dict):
            raise ValueError("config root is not an object")
    except Exception as exc:
        # Keep the damaged file rather than overwriting it — it may be the
        # only copy of an API key. Loud on stderr; the dashboard surfaces it.
        backup = CONFIG_FILE.with_suffix(".broken.json")
        try:
            shutil.copy2(CONFIG_FILE, backup)
        except Exception:
            pass
        print(f"⚠  config unreadable ({exc}); using defaults. "
              f"Your file was preserved at {backup}")
        cfg["_error"] = f"config unreadable: {exc}"
        return cfg
    raw = _migrate(raw)
    for key, value in raw.items():
        if key in DEFAULTS and validate(key, value):
            cfg[key] = value
        elif key in DEFAULTS:
            print(f"⚠  config: ignoring invalid value for {key!r}")
        # Unknown keys are dropped silently; they're from an older schema.
    return cfg


def save(cfg: dict) -> None:
    """Atomically write the config. Raises on genuine I/O failure."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    fd, tmp = tempfile.mkstemp(dir=str(CONFIG_DIR), prefix=".config-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CONFIG_FILE)        # atomic on the same filesystem
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def update(changes: dict) -> tuple[dict, list[str]]:
    """Apply validated changes on top of the current config and persist.

    Returns (new_config, rejected_keys). Rejected keys are reported rather
    than silently dropped — a settings form that appears to save but doesn't
    is worse than one that says no.
    """
    cfg = load()
    if cfg.get("_error"):
        # Writing now would rebuild the file from defaults and destroy
        # whatever the unparseable original still held.
        raise RuntimeError(
            "refusing to save over an unreadable config; fix or delete "
            f"{CONFIG_FILE} first"
        )
    rejected = []
    for key, value in changes.items():
        if key in EDITABLE and validate(key, value):
            cfg[key] = value
        else:
            rejected.append(key)
    save(cfg)
    return cfg, rejected
