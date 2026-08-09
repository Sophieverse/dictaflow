#!/usr/bin/env python3
"""
DictaFlow dashboard — local web view of your dictation history and stats.

Two data sources, deliberately not joined:

  transcripts.md  — the full prose history, going back further than the stats
                    log. Source of truth for *what you said*.
  events.jsonl    — one structured record per dictation attempt, including the
                    ones that produced no text. Source of truth for *how well
                    it worked*: latency, model, rejection rate.

Joining them on timestamp would be fragile (they're written a beat apart) and
buys nothing — no view needs both at once.

Serves http://localhost:7755. Pure standard library, no dependencies.
"""
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, date, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT            = 7755
TRANSCRIPTS_DIR = Path.home() / "transcriptions"
TRANSCRIPT_FILE = TRANSCRIPTS_DIR / "transcripts.md"
EVENTS_FILE     = TRANSCRIPTS_DIR / "events.jsonl"
CONFIG_FILE     = Path.home() / ".dictaflow" / "config.json"
HTML_FILE       = Path(__file__).parent / "dashboard.html"

# For the "time saved" figure. Sustained prose typing for most people is around
# 40 wpm; conversational speech is around 150. Both are population averages, so
# treat the number as indicative rather than a measurement of you specifically.
TYPING_WPM   = 40
SPEAKING_WPM = 150

# A transcript section looks like:  "## 2026-06-26 00:40:14\n\n<body>\n"
ENTRY_RE = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\n(.*?)(?=^##\s|\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_transcripts() -> list[dict]:
    """Return entries newest-first: [{ts, text, raw, words}]."""
    if not TRANSCRIPT_FILE.exists():
        return []
    entries = []
    for ts, body in ENTRY_RE.findall(TRANSCRIPT_FILE.read_text()):
        body = body.strip()
        raw = None
        m = re.search(r"\n\*Raw:\*\s*(.*)$", body, re.DOTALL)
        if m:                                   # split off the optional Raw: line
            raw = m.group(1).strip()
            body = body[: m.start()].strip()
        if not body:
            continue
        entries.append({"ts": ts, "text": body, "raw": raw,
                        "words": len(body.split())})
    entries.reverse()                            # newest first
    return entries


def load_events() -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    out = []
    for line in EVENTS_FILE.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                             # tolerate a torn final write
    return out


def streak(days: set[str]) -> int:
    """Consecutive days ending today (or yesterday, so it survives until you've
    dictated today — otherwise the number would read 0 every morning)."""
    if not days:
        return 0
    today = date.today()
    start = today if today.isoformat() in days else today - timedelta(days=1)
    n, cur = 0, start
    while cur.isoformat() in days:
        n += 1
        cur -= timedelta(days=1)
    return n


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(int(len(s) * p), len(s) - 1)], 2)


def compute(entries: list[dict], events: list[dict]) -> dict:
    total_words = sum(e["words"] for e in entries)
    days        = {e["ts"][:10] for e in entries}

    per_day = defaultdict(int)
    for e in entries:
        per_day[e["ts"][:10]] += e["words"]
    span = [(date.today() - timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
    daily = [{"day": d, "words": per_day.get(d, 0)} for d in span]

    week_ago  = (date.today() - timedelta(days=6)).isoformat()
    week_words = sum(v for d, v in per_day.items() if d >= week_ago)

    # Time saved = how long these words would have taken to type, minus how
    # long they actually took to say. Prefer measured audio length where we
    # have it; fall back to the speaking-rate estimate for older entries.
    measured = sum(ev.get("audio_secs", 0) for ev in events if ev.get("outcome") == "ok")
    measured_words = sum(ev.get("words", 0) for ev in events if ev.get("outcome") == "ok")
    est_secs = (total_words - measured_words) / SPEAKING_WPM * 60
    spoken_secs = measured + max(est_secs, 0)
    saved_secs  = max(total_words / TYPING_WPM * 60 - spoken_secs, 0)

    ok  = [ev for ev in events if ev.get("outcome") == "ok"]
    lat = defaultdict(list)
    for ev in ok:
        if ev.get("latency"):
            lat[ev.get("model", "?")].append(ev["latency"])
    latency = {m: {"median": percentile(v, .5), "p90": percentile(v, .9), "n": len(v)}
               for m, v in lat.items()}

    wpm = None
    if measured > 5:
        wpm = round(measured_words / (measured / 60))

    rejected = sum(1 for ev in events if ev.get("outcome") in ("no_speech", "rejected"))
    return {
        "count":        len(entries),
        "total_words":  total_words,
        "week_words":   week_words,
        "days":         len(days),
        "streak":       streak(days),
        "saved_min":    round(saved_secs / 60),
        "wpm":          wpm,
        "latency":      latency,
        "attempts":     len(events),
        "rejected":     rejected,
        "reject_pct":   round(rejected / len(events) * 100) if events else None,
        "last":         entries[0]["ts"] if entries else None,
        "daily":        daily,
    }


def agent_running() -> bool:
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=3).stdout
    except Exception:
        return False
    for line in out.splitlines():
        if line.endswith("com.dictaflow.agent"):
            return not line.startswith("-")      # a PID in col 1 means running
    return False


# Mirrors DEFAULT_CONFIG in dictaflow.py for the keys this UI touches. The
# agent merges its own defaults over the file at load, so a config written
# before a key existed still behaves correctly there — but the dashboard reads
# the file directly, and without these it would render "language: auto-detect"
# for a config that is really running "en", then persist that misreading the
# moment you pressed Save. Duplicated rather than imported to keep this file
# dependency-free (importing dictaflow drags in sounddevice, numpy, pynput).
CONFIG_DEFAULTS = {"language": "en", "initial_prompt": "", "cleanup_enabled": False}


def read_config() -> dict:
    try:
        return {**CONFIG_DEFAULTS, **json.loads(CONFIG_FILE.read_text())}
    except Exception:
        return dict(CONFIG_DEFAULTS)


# Only these are exposed to the web UI. The config file also holds API keys,
# and a settings endpoint that could rewrite arbitrary keys would be a way to
# repoint the backend at a remote service from a page in the browser.
EDITABLE = {"language", "initial_prompt", "cleanup_enabled", "local_whisper_model"}


def write_config(updates: dict) -> dict:
    cfg = read_config()
    cfg.update({k: v for k, v in updates.items() if k in EDITABLE})
    CONFIG_FILE.parent.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    return cfg


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):                # quiet: no per-request spam
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self) -> None:
        if self.path.startswith("/api/data"):
            entries = parse_transcripts()
            events  = load_events()
            cfg     = read_config()
            self._json({
                "entries":  entries[:500],       # page is virtualised at 500
                "stats":    compute(entries, events),
                "running":  agent_running(),
                "settings": {k: cfg.get(k) for k in EDITABLE},
            })
        elif self.path == "/" or self.path.startswith("/index"):
            if HTML_FILE.exists():
                self._send(HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(b"dashboard.html missing", "text/plain", 500)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if self.path != "/api/settings":
            return self._send(b"not found", "text/plain", 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            updates = json.loads(self.rfile.read(n) or b"{}")
            cfg = write_config(updates)
            self._json({"ok": True, "settings": {k: cfg.get(k) for k in EDITABLE}})
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 400)


if __name__ == "__main__":
    print(f"DictaFlow dashboard → http://localhost:{PORT}")
    print(f"  transcripts {TRANSCRIPT_FILE}")
    print(f"  events      {EVENTS_FILE}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
