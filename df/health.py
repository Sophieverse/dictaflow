"""A heartbeat file, because the failure that mattered most was invisible.

DictaFlow once spent three days refusing every dictation while looking
perfectly healthy from the outside: the process was up, the menu bar item was
there, the hotkeys responded and printed "Recording…". The only evidence was a
warning that had scrolled off the top of a log file nobody reads.

So the agent now publishes what it actually knows about itself, once every few
seconds, to one small file. `dictaflow.py --status` and the dashboard both read
it. The point is not the file — it is that "is it working?" has an answer you
can get in one second without reading source.

STALENESS IS THE POINT. A heartbeat that is merely present proves nothing; a
crashed or hung agent leaves its last one lying there looking fine. Readers
must compare `updated` against now, which is why `read()` returns the age and
`summarise()` refuses to trust anything older than STALE_AFTER.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import CONFIG_DIR

STATUS_PATH = CONFIG_DIR / "status.json"

# The agent writes every HEARTBEAT_S; anything older than STALE_AFTER means it
# is wedged, killed, or was never running. The gap between them is slack for a
# loaded machine, not a grace period for a sick one.
HEARTBEAT_S = 2.0
STALE_AFTER = 15.0


def write(state: dict) -> None:
    """Publish the current state. Never raises — this is diagnostics."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(state)
        payload["pid"] = os.getpid()
        payload["updated"] = time.time()
        tmp = STATUS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, STATUS_PATH)          # atomic; no half-written reads
    except Exception:
        pass


def clear() -> None:
    """Remove the heartbeat on a clean exit, so 'stopped' reads as stopped."""
    try:
        STATUS_PATH.unlink()
    except Exception:
        pass


def read() -> tuple[dict | None, float]:
    """Return (state, age_seconds). state is None if there is no heartbeat."""
    try:
        state = json.loads(STATUS_PATH.read_text())
    except Exception:
        return None, float("inf")
    age = max(0.0, time.time() - float(state.get("updated") or 0))
    return state, age


def summarise() -> tuple[bool, str, dict]:
    """(healthy, one-line verdict, state). Safe to call from anywhere."""
    state, age = read()
    if state is None:
        return False, "not running — no heartbeat found", {}
    if age > STALE_AFTER:
        return False, (f"not responding — last heartbeat {age:.0f}s ago "
                       f"(pid {state.get('pid')})"), state
    if not state.get("models_ready"):
        return False, "starting up — models still loading", state
    if not state.get("mic_open"):
        err = state.get("last_error") or "the microphone is not open"
        return False, f"microphone unavailable — {err}", state
    # Open is not the same as working. An open handle whose callbacks have
    # stopped is the failure that looks healthiest from every other angle.
    if not state.get("mic_flowing", True):
        return False, ("the microphone is open but no audio is arriving — "
                       "another app may have taken the device"), state
    if state.get("mic_silent") or state.get("mic_muted"):
        return False, ("the microphone is delivering pure silence — the agent "
                       "is resetting the audio device; if it persists, run "
                       "dictaflow.py --fix-audio"), state
    return True, "ready", state
