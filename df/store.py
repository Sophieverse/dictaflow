"""History and stats storage.

The previous design made `transcripts.md` the source of truth and re-parsed it
with a regex. That regex looked for lines beginning `## `, which meant that
*dictating* a line beginning `## ` silently deleted the rest of that entry —
verified: an 11-word tail vanished with no error. Prose is a bad database.

So: `history.jsonl` is now the source of truth, one JSON object per line, and
`transcripts.md` is a human-readable mirror that nothing ever parses. A newline
inside a transcript is escaped by json.dumps, so no utterance can corrupt the
store no matter what you say.

Edits, pins and deletes are appended as `patch` records rather than rewriting
the file, so every write stays a single atomic append. Reading folds the
patches over the entries. `compact()` collapses them when the file grows.
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

from .config import TRANSCRIPTS_DIR, TRANSCRIPT_FILE

HISTORY_FILE = TRANSCRIPTS_DIR / "history.jsonl"
LEGACY_EVENTS = TRANSCRIPTS_DIR / "events.jsonl"

_WRITE_LOCK = threading.Lock()

# Above this many lines, folding patches on every read starts to show. Well
# beyond any realistic amount of dictation, but the store shouldn't degrade
# silently if it is ever reached.
COMPACT_THRESHOLD = 20000


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _append(record: dict) -> None:
    """Append one JSON line. Best-effort: a stats write must never be able to
    lose a transcript, so failures are reported but not raised."""
    with _WRITE_LOCK:
        try:
            TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            print(f"⚠  could not write history: {exc}")


def add_entry(*, text: str, raw: str = "", model: str = "", app: str = "",
              bundle_id: str = "", category: str = "", latency: float = 0.0,
              audio_secs: float = 0.0, chunks: int = 1,
              outcome: str = "ok", rejected: str = "",
              pasted: bool = True, paste_detail: str = "") -> str:
    """Record one dictation attempt. Returns its id."""
    entry_id = uuid.uuid4().hex[:12]
    _append({
        "type": "entry", "id": entry_id, "ts": _now(),
        "text": text, "raw": raw, "model": model,
        "app": app, "bundle_id": bundle_id, "category": category,
        "latency": round(latency, 2), "audio_secs": round(audio_secs, 2),
        "chunks": chunks, "words": len(text.split()) if text else 0,
        "chars": len(text), "outcome": outcome, "rejected": rejected,
        "pasted": pasted, "paste_detail": paste_detail,
        "pinned": False, "deleted": False,
    })
    if outcome == "ok" and text:
        _mirror_markdown(text, raw)
    return entry_id


def patch(entry_id: str, **fields) -> None:
    """Record a change to an existing entry (pin, delete, edit)."""
    allowed = {"pinned", "deleted", "text", "pasted", "paste_detail"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    if not payload:
        return
    _append({"type": "patch", "id": entry_id, "ts": _now(), **payload})


def _mirror_markdown(text: str, raw: str) -> None:
    """Append to the human-readable log. Nothing ever parses this file."""
    try:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("# Transcripts\n")
            f.write(f"\n## {stamp}\n\n{text}\n")
            if raw and raw != text:
                f.write(f"\n*Raw:* {raw}\n")
    except Exception as exc:
        print(f"⚠  could not write transcripts.md: {exc}")


def load(include_deleted: bool = False) -> list[dict]:
    """All entries, newest first, with patches folded in. Never raises."""
    if not HISTORY_FILE.exists():
        return []
    entries: dict[str, dict] = {}
    order: list[str] = []
    try:
        raw = HISTORY_FILE.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"⚠  could not read history: {exc}")
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                    # tolerate a torn final write
        if not isinstance(rec, dict) or "id" not in rec:
            continue
        if rec.get("type") == "patch":
            target = entries.get(rec["id"])
            if target is not None:
                for key in ("pinned", "deleted", "text", "pasted",
                            "paste_detail"):
                    if key in rec:
                        target[key] = rec[key]
                if "text" in rec:
                    target["edited"] = True
                    target["words"] = len(str(rec["text"]).split())
                    target["chars"] = len(str(rec["text"]))
        else:
            if rec["id"] not in entries:
                order.append(rec["id"])
            entries[rec["id"]] = rec
    out = [entries[i] for i in order if i in entries]
    if not include_deleted:
        out = [e for e in out if not e.get("deleted")]
    out.reverse()                        # newest first
    return out


def compact() -> int:
    """Rewrite the file with patches applied. Returns the new line count."""
    entries = load(include_deleted=True)
    entries.reverse()                    # back to chronological for storage
    fd, tmp = tempfile.mkstemp(dir=str(TRANSCRIPTS_DIR), prefix=".history-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for entry in entries:
                entry.pop("edited", None)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with _WRITE_LOCK:
            os.replace(tmp, HISTORY_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return len(entries)


def maybe_compact() -> None:
    try:
        if not HISTORY_FILE.exists():
            return
        with open(HISTORY_FILE, "rb") as f:
            lines = sum(1 for _ in f)
        if lines > COMPACT_THRESHOLD:
            compact()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# Migration from the v1 files
# ──────────────────────────────────────────────────────────────
def migrate_legacy() -> int:
    """Import transcripts.md + events.jsonl into history.jsonl once.

    Uses the old regex deliberately — it is the only way to read the old
    format, and any entry it truncated was already truncated on disk. Better
    to carry the history forward imperfectly than to drop it.
    """
    if HISTORY_FILE.exists():
        return 0
    import re
    imported = 0
    records: list[dict] = []

    if TRANSCRIPT_FILE.exists():
        try:
            md = TRANSCRIPT_FILE.read_text(encoding="utf-8", errors="replace")
        except Exception:
            md = ""
        pattern = re.compile(
            r"^##\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\n(.*?)(?=^##\s+\d{4}-\d{2}-\d{2}|\Z)",
            re.DOTALL | re.MULTILINE)
        for ts, body in pattern.findall(md):
            body = body.strip()
            raw = ""
            m = list(re.finditer(r"^\*Raw:\*\s*", body, re.MULTILINE))
            if m:
                last = m[-1]
                raw = body[last.end():].strip()
                body = body[: last.start()].strip()
            if not body:
                continue
            records.append({
                "type": "entry", "id": uuid.uuid4().hex[:12],
                "ts": ts.replace(" ", "T"), "text": body, "raw": raw,
                "model": "", "app": "", "bundle_id": "", "category": "",
                "latency": 0.0, "audio_secs": 0.0, "chunks": 1,
                "words": len(body.split()), "chars": len(body),
                "outcome": "ok", "rejected": "", "pasted": True,
                "paste_detail": "", "pinned": False, "deleted": False,
                "imported": True,
            })
            imported += 1

    if LEGACY_EVENTS.exists():
        try:
            for line in LEGACY_EVENTS.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("outcome") == "ok":
                    continue             # already covered by transcripts.md
                records.append({
                    "type": "entry", "id": uuid.uuid4().hex[:12],
                    "ts": ev.get("ts", ""), "text": "", "raw": "",
                    "model": ev.get("model", ""), "app": "", "bundle_id": "",
                    "category": "", "latency": ev.get("latency", 0.0),
                    "audio_secs": ev.get("audio_secs", 0.0), "chunks": 1,
                    "words": 0, "chars": 0,
                    "outcome": ev.get("outcome", "rejected"), "rejected": "",
                    "pasted": False, "paste_detail": "",
                    "pinned": False, "deleted": False, "imported": True,
                })
                imported += 1
        except Exception:
            pass

    records.sort(key=lambda r: r.get("ts", ""))
    if records:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return imported


def export_markdown(entries: list[dict]) -> str:
    # `or ""` rather than a .get() default throughout: a record can hold an
    # explicit null (JSON round-trips it faithfully), and .get("text", "")
    # returns that null rather than the default — which crashed the join.
    lines = ["# DictaFlow transcripts", ""]
    for e in entries:
        lines.append(f"## {e.get('ts') or ''}")
        meta = " · ".join(str(x) for x in (e.get("app"), e.get("model")) if x)
        if meta:
            lines.append(f"*{meta}*")
        lines.append("")
        lines.append(str(e.get("text") or ""))
        lines.append("")
    return "\n".join(lines)


def export_text(entries: list[dict]) -> str:
    return "\n\n".join(str(e.get("text")) for e in entries if e.get("text"))
