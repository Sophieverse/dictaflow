#!/usr/bin/env python3
"""
DictaFlow dashboard — a tiny local web view of your dictation transcripts.

Reads ~/transcriptions/transcripts.md (the single rolling log the dictation
agent appends to) and serves a searchable web page at http://localhost:7755.
Pure standard library — no dependencies.
"""
import json
import re
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT            = 7755
TRANSCRIPT_FILE = Path.home() / "transcriptions" / "transcripts.md"
HTML_FILE       = Path(__file__).parent / "dashboard.html"

# A transcript section looks like:  "## 2026-06-26 00:40:14\n\n<body>\n"
ENTRY_RE = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\n(.*?)(?=^##\s|\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_transcripts() -> list[dict]:
    """Return entries newest-first: [{ts, text, raw, words}]."""
    if not TRANSCRIPT_FILE.exists():
        return []
    md = TRANSCRIPT_FILE.read_text()
    entries = []
    for ts, body in ENTRY_RE.findall(md):
        body = body.strip()
        raw = None
        m = re.search(r"\n\*Raw:\*\s*(.*)$", body, re.DOTALL)
        if m:                                   # split off the optional Raw: line
            raw = m.group(1).strip()
            body = body[: m.start()].strip()
        entries.append({
            "ts":    ts,
            "text":  body,
            "raw":   raw,
            "words": len(body.split()),
        })
    entries.reverse()                            # newest first
    return entries


def compute_stats(entries: list[dict]) -> dict:
    total_words = sum(e["words"] for e in entries)
    days = {e["ts"][:10] for e in entries}
    return {
        "count":       len(entries),
        "total_words": total_words,
        "days":        len(days),
        "last":        entries[0]["ts"] if entries else None,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):                # quiet: no per-request console spam
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/transcripts"):
            entries = parse_transcripts()
            payload = {"entries": entries, "stats": compute_stats(entries)}
            self._send(json.dumps(payload).encode(), "application/json")
        elif self.path == "/" or self.path.startswith("/index"):
            if HTML_FILE.exists():
                self._send(HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(b"dashboard.html missing", "text/plain")
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"DictaFlow dashboard → http://localhost:{PORT}")
    print(f"  reading {TRANSCRIPT_FILE}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
