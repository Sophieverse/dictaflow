"""Audio cues. Fire-and-forget so nothing ever blocks on a sound playing."""
from __future__ import annotations

import os
import subprocess
import threading


def play(path: str | None, enabled: bool = True) -> None:
    if not enabled or not path or not os.path.exists(path):
        return

    def _run():
        try:
            subprocess.run(["afplay", "-v", "0.35", path],
                           capture_output=True, timeout=5)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
