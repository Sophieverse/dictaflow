"""Command Mode: select text, speak an instruction, get the text rewritten.

Wispr Flow's equivalent is cloud-backed. This one runs against a local Ollama
model, which means it is private and free but also that it is only as fast as
whatever model you point it at. The config default is a small one on purpose:
`gpt-oss:20b` measured around 40 seconds for a short rewrite, which is not a
usable interaction, while a 3B model answers in one to two.

If no model is available the feature reports that clearly and does nothing.
It never silently falls back to pasting the raw instruction, which would type
"make this more concise" into your document.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import inject


class CommandError(Exception):
    pass


def read_selection(ctx) -> str:
    """The currently selected text.

    Tries the Accessibility tree first because it is non-destructive. Falls
    back to ⌘C, which works nearly everywhere but costs the clipboard — so
    the clipboard is snapshotted and restored around it.
    """
    if ctx is not None and ctx.selection.strip():
        return ctx.selection

    saved = inject.snapshot_clipboard()
    try:
        import Quartz
        src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        for down in (True, False):
            ev = Quartz.CGEventCreateKeyboardEvent(src, 8, down)  # kVK_ANSI_C
            Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            time.sleep(0.01)
        time.sleep(0.12)                # let the target app service the copy
        text = inject.get_clipboard_text()
    except Exception:
        text = ""
    finally:
        inject.restore_clipboard(saved)
    return text


def _ollama(instruction: str, selection: str, cfg: dict) -> str:
    payload = json.dumps({
        "model": cfg.get("ollama_model", "llama3.2:3b"),
        "messages": [
            {"role": "system", "content": cfg.get("command_prompt", "")},
            {"role": "user",
             "content": f"Instruction: {instruction}\n\nText:\n{selection}"},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode()
    req = urllib.request.Request(
        cfg.get("ollama_host", "http://localhost:11434").rstrip("/") + "/api/chat",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise CommandError(
            f"Ollama is not reachable at {cfg.get('ollama_host')} — start it "
            f"with `ollama serve`. ({exc.reason})") from exc
    except Exception as exc:
        raise CommandError(f"Ollama request failed: {exc}") from exc

    # Ollama reports a missing model as a 200 with an `error` key rather than
    # an HTTP error, and the old code indexed straight into ["message"],
    # turning that into a KeyError that destroyed the whole dictation.
    if isinstance(data, dict) and data.get("error"):
        raise CommandError(str(data["error"]))
    try:
        text = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise CommandError(f"unexpected Ollama response: {str(data)[:200]}") from exc
    return (text or "").strip()


def available(cfg: dict) -> tuple[bool, str]:
    """Whether Command Mode can run right now, and why not if it can't."""
    backend = cfg.get("command_backend", "ollama")
    if backend == "none":
        return False, "command mode is disabled in settings"
    if backend != "ollama":
        return False, f"unsupported command backend {backend!r}"
    host = cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
    want = cfg.get("ollama_model", "")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=2) as resp:
            tags = json.loads(resp.read())
    except Exception:
        return False, f"Ollama is not running at {host} — start it with `ollama serve`"
    names = {m.get("name", "") for m in tags.get("models", [])}
    if want and want not in names and not any(n.startswith(want.split(":")[0]) for n in names):
        listed = ", ".join(sorted(names)) or "none"
        return False, (f"Ollama has no model {want!r} — pull it with "
                       f"`ollama pull {want}` (installed: {listed})")
    return True, ""


def run(instruction: str, selection: str, cfg: dict) -> str:
    """Apply `instruction` to `selection`. Raises CommandError on failure."""
    if not instruction.strip():
        raise CommandError("no instruction heard")
    if not selection.strip():
        raise CommandError("no text selected — select something first")
    ok, why = available(cfg)
    if not ok:
        raise CommandError(why)
    result = _ollama(instruction, selection, cfg)
    if not result:
        raise CommandError("the model returned nothing")
    # Small models like to wrap answers in quotes or prefix them with
    # "Here's the rewritten text:". Strip the most common shapes.
    for prefix in ("here's the rewritten text:", "here is the rewritten text:",
                   "rewritten text:", "sure!", "certainly!"):
        if result.lower().startswith(prefix):
            result = result[len(prefix):].strip()
    if len(result) > 1 and result[0] == result[-1] == '"':
        result = result[1:-1]
    return result.strip()
