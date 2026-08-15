#!/usr/bin/env python3
"""DictaFlow dashboard — local web UI for history, stats and settings.

SECURITY. This binds to loopback, which is *not* the same as being private:
every page in your browser can also reach 127.0.0.1. An audit of the previous
version confirmed a working cross-origin attack — a `text/plain` POST (a
"simple request", so no CORS preflight to block it) from any website could
rewrite the config. Because one of the writable keys was the Whisper model
path, and because DictaFlow types its output as keystrokes into whatever app
you have focused, that was keystroke injection reachable from a web page.

Three defences, all required:
  1. Origin/Referer must be this server, when present.
  2. Host must be localhost — otherwise DNS rebinding lets an attacker's
     domain resolve to 127.0.0.1 and read your entire transcript history.
  3. A per-process CSRF token, embedded in the page and required as a header
     on every mutating request. A cross-origin page cannot read the token.
And the model path is no longer writable at all (see config.EDITABLE).
"""
from __future__ import annotations

import json
import math
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from df import config, store          # noqa: E402

PORT      = 7755
HTML_FILE = Path(__file__).resolve().parent / "dashboard.html"

# Regenerated every time the server starts, so a token can never outlive the
# process it belongs to.
CSRF_TOKEN = secrets.token_urlsafe(32)

ALLOWED_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}",
                 f"[::1]:{PORT}", "localhost", "127.0.0.1"}
ALLOWED_ORIGINS = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"}

# For "time saved". Sustained prose typing is ~40wpm for most people;
# conversational speech ~150. Both are population averages, so the number is
# indicative rather than a measurement of you.
TYPING_WPM = 40

_STATUS_CACHE = {"at": 0.0, "value": False}
_STATUS_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────
def _word_count(entry: dict) -> int:
    """Words in an entry, robust to text with no whitespace.

    `.split()` counted an 889-character repetition hallucination as one word,
    which silently distorted every word-derived metric on this page. Counting
    `\\w+` runs is correct for the CJK/no-space case too.
    """
    text = entry.get("text") or ""
    if not text:
        return 0
    n = len(re.findall(r"\w+", text))
    return n if n else (1 if text.strip() else 0)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile.

    The previous `int(len(s) * p)` overshot by about one rank, which made p90
    literally equal to max() for any sample of ten or fewer — the page showed
    a single measurement as a "p90".
    """
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(len(s) * p) - 1))
    return round(s[idx], 2)


def streak(days: set[str]) -> int:
    """Consecutive days ending today, or yesterday so the number doesn't read
    zero every morning before you've dictated."""
    if not days:
        return 0
    today = date.today()
    start = today if today.isoformat() in days else today - timedelta(days=1)
    n, cur = 0, start
    while cur.isoformat() in days:
        n += 1
        cur -= timedelta(days=1)
    return n


def compute(entries: list[dict]) -> dict:
    ok = [e for e in entries if e.get("outcome") == "ok" and (e.get("text") or "")]
    rejected = [e for e in entries if e.get("outcome") == "rejected"]

    words_by_entry = {id(e): _word_count(e) for e in ok}
    total_words = sum(words_by_entry.values())
    days = {e["ts"][:10] for e in ok if e.get("ts")}

    per_day = defaultdict(int)
    for e in ok:
        if e.get("ts"):
            per_day[e["ts"][:10]] += words_by_entry[id(e)]
    span = [(date.today() - timedelta(days=i)).isoformat() for i in range(29, -1, -1)]
    daily = [{"day": d, "words": per_day.get(d, 0)} for d in span]

    week_ago = (date.today() - timedelta(days=6)).isoformat()
    week_words = sum(v for d, v in per_day.items() if d >= week_ago)

    # Time saved, computed only over entries where we actually measured the
    # audio length. Mixing measured and estimated durations made the figure
    # collapse to zero whenever the two data sources disagreed; restricting it
    # to real measurements means it reports on less, but reports honestly.
    measured = [e for e in ok if e.get("audio_secs", 0) > 0]
    spoken_secs = sum(e["audio_secs"] for e in measured)
    measured_words = sum(words_by_entry[id(e)] for e in measured)
    saved_secs = max(measured_words / TYPING_WPM * 60 - spoken_secs, 0)

    wpm = round(measured_words / (spoken_secs / 60)) if spoken_secs > 5 else None

    lat = defaultdict(list)
    for e in ok:
        if e.get("latency"):
            lat[e.get("model") or "?"].append(e["latency"])
    latency = {m: {"median": percentile(v, .5), "p90": percentile(v, .9),
                   "n": len(v)}
               for m, v in sorted(lat.items())}

    by_app = Counter()
    for e in ok:
        if e.get("app"):
            by_app[e["app"]] += words_by_entry[id(e)]

    reasons = Counter(e.get("rejected", "") for e in rejected if e.get("rejected"))
    failed_paste = [e for e in ok if e.get("pasted") is False]

    attempts = len(ok) + len(rejected)
    return {
        "count":       len(ok),
        "total_words": total_words,
        "week_words":  week_words,
        "days":        len(days),
        "streak":      streak(days),
        "saved_min":   round(saved_secs / 60),
        "saved_basis": len(measured),
        "wpm":         wpm,
        "latency":     latency,
        "attempts":    attempts,
        "rejected":    len(rejected),
        "reject_pct":  round(len(rejected) / attempts * 100) if attempts else None,
        "reject_reasons": reasons.most_common(6),
        "failed_paste": len(failed_paste),
        "by_app":      by_app.most_common(8),
        "last":        ok[0]["ts"] if ok else None,
        "daily":       daily,
    }


def agent_running() -> bool:
    """Is the LaunchAgent up? Cached — this forks a process, and the page
    polls, so without the cache it ran every few seconds per open tab."""
    with _STATUS_LOCK:
        if time.monotonic() - _STATUS_CACHE["at"] < 10:
            return _STATUS_CACHE["value"]
    value = False
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=3).stdout
        for line in out.splitlines():
            if line.endswith("com.dictaflow.agent"):
                value = not line.startswith("-")
                break
    except Exception:
        value = False
    with _STATUS_LOCK:
        _STATUS_CACHE.update(at=time.monotonic(), value=value)
    return value


# ──────────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "DictaFlow"

    def log_message(self, *args):
        pass                                    # no per-request spam

    # ── security gates ──────────────────────────────────────────
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in ALLOWED_HOSTS

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return False
        referer = self.headers.get("Referer")
        if referer:
            parsed = urlparse(referer)
            if f"{parsed.scheme}://{parsed.netloc}" not in ALLOWED_ORIGINS:
                return False
        return True

    def _csrf_ok(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-DictaFlow-Token") or "", CSRF_TOKEN)

    def _guard(self, *, mutating: bool) -> bool:
        if not self._host_ok():
            self._send(b"bad host", "text/plain", 403)
            return False
        if not self._origin_ok():
            self._send(b"bad origin", "text/plain", 403)
            return False
        if mutating and not self._csrf_ok():
            self._json({"ok": False, "error": "missing or bad CSRF token"}, 403)
            return False
        return True

    # ── plumbing ────────────────────────────────────────────────
    def _send(self, body: bytes, ctype: str, code: int = 200,
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        # This page never needs to be framed or to load anything remote.
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'unsafe-inline'; "
                         "script-src 'unsafe-inline'; img-src data:; "
                         "connect-src 'self'; frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    class BadBody(Exception):
        """Raised rather than returning {} so a malformed request reports a
        failure instead of a cheerful 200 that changed nothing — the same
        silent-success pattern being removed everywhere else in this app."""

    MAX_BODY = 8 * 1024 * 1024

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > self.MAX_BODY:
            raise self.BadBody(f"body too large ({length} bytes, max "
                               f"{self.MAX_BODY})")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise self.BadBody(f"body is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise self.BadBody("body must be a JSON object")
        return parsed

    # ── routes ──────────────────────────────────────────────────
    def do_GET(self) -> None:
        try:
            if not self._guard(mutating=False):
                return
            path = urlparse(self.path).path
            if path == "/api/data":
                return self._api_data()
            if path == "/api/export":
                return self._api_export()
            if path in ("/", "/index.html"):
                return self._page()
            self._send(b"not found", "text/plain", 404)
        except Exception as exc:
            # Without this, any handler exception closed the socket with no
            # response and the page silently froze on stale data forever.
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:
        try:
            if not self._guard(mutating=True):
                return
            path = urlparse(self.path).path
            if path == "/api/settings":
                return self._api_settings()
            if path == "/api/entry":
                return self._api_entry()
            if path == "/api/compact":
                return self._json({"ok": True, "entries": store.compact()})
            self._send(b"not found", "text/plain", 404)
        except self.BadBody as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    def _page(self) -> None:
        if not HTML_FILE.exists():
            return self._send(b"dashboard.html is missing", "text/plain", 500)
        html = HTML_FILE.read_text(encoding="utf-8", errors="replace")
        html = html.replace("__CSRF_TOKEN__", CSRF_TOKEN)
        self._send(html.encode("utf-8"), "text/html; charset=utf-8")

    def _api_data(self) -> None:
        entries = store.load()
        cfg = config.load()
        query = parse_qs(urlparse(self.path).query)
        limit = min(int((query.get("limit") or ["300"])[0] or 300), 2000)
        self._json({
            "entries":  entries[:limit],
            "total":    len(entries),
            "stats":    compute(entries),
            "running":  agent_running(),
            "settings": {k: cfg.get(k) for k in sorted(config.EDITABLE)},
            "config_error": cfg.get("_error"),
        })

    def _api_settings(self) -> None:
        try:
            cfg, rejected = config.update(self._body())
        except RuntimeError as exc:
            return self._json({"ok": False, "error": str(exc)}, 409)
        self._json({
            "ok": True,
            "rejected": rejected,
            "settings": {k: cfg.get(k) for k in sorted(config.EDITABLE)},
        })

    def _api_entry(self) -> None:
        body = self._body()
        entry_id = str(body.get("id") or "")
        if not re.fullmatch(r"[0-9a-f]{6,32}", entry_id):
            return self._json({"ok": False, "error": "bad id"}, 400)
        fields = {}
        if "pinned" in body:
            fields["pinned"] = bool(body["pinned"])
        if "deleted" in body:
            fields["deleted"] = bool(body["deleted"])
        if "text" in body:
            text = body["text"]
            if not isinstance(text, str) or len(text) > 100000:
                return self._json({"ok": False, "error": "bad text"}, 400)
            fields["text"] = text
        if not fields:
            return self._json({"ok": False, "error": "nothing to change"}, 400)
        store.patch(entry_id, **fields)
        self._json({"ok": True})

    def _api_export(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        fmt = (query.get("format") or ["md"])[0]
        entries = store.load()
        if fmt == "json":
            body = json.dumps(entries, indent=2, ensure_ascii=False).encode("utf-8")
            ctype, name = "application/json", "dictaflow-history.json"
        elif fmt == "txt":
            body = store.export_text(entries).encode("utf-8")
            ctype, name = "text/plain; charset=utf-8", "dictaflow-history.txt"
        else:
            body = store.export_markdown(entries).encode("utf-8")
            ctype, name = "text/markdown; charset=utf-8", "dictaflow-history.md"
        self._send(body, ctype, 200,
                   {"Content-Disposition": f'attachment; filename="{name}"'})


def main() -> int:
    store.migrate_legacy()
    print(f"DictaFlow dashboard → http://localhost:{PORT}")
    print(f"  history {store.HISTORY_FILE}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
