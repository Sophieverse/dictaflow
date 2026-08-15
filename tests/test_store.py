"""Tests for the JSONL history store.

Every test points the store's module-level paths at a scratch directory, the
same way test_session.py does, so nothing here can ever read or write the
user's real `~/transcriptions`. tearDown asserts the real paths are byte-for-
byte untouched, because a store test that quietly appends to a year of real
history is worse than no test at all.

The headline case is `TestUtteranceCannotCorruptTheStore`: the old store made
`transcripts.md` the source of truth and re-parsed it with a `^## ` regex, so
*dictating* a line that began `## ` silently deleted the rest of that entry.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from df import store  # noqa: E402


def _fingerprint(path: Path):
    """Enough of a path's state to notice any write to it."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_size, st.st_mtime_ns)


class StoreTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Captured before any patch is active, so these are the real ones.
        cls.real_paths = [store.HISTORY_FILE, store.TRANSCRIPT_FILE,
                          store.LEGACY_EVENTS, store.TRANSCRIPTS_DIR]

    def setUp(self):
        self.real_before = [_fingerprint(p) for p in self.real_paths]

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.history = self.dir / "history.jsonl"
        self.md = self.dir / "transcripts.md"
        self.events = self.dir / "events.jsonl"
        self._patches = [
            mock.patch.object(store, "HISTORY_FILE", self.history),
            mock.patch.object(store, "TRANSCRIPTS_DIR", self.dir),
            mock.patch.object(store, "TRANSCRIPT_FILE", self.md),
            mock.patch.object(store, "LEGACY_EVENTS", self.events),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

        after = [_fingerprint(p) for p in self.real_paths]
        for path, before, now in zip(self.real_paths, self.real_before, after):
            self.assertEqual(before, now,
                             f"the test touched the real store at {path}")

    # ── helpers ─────────────────────────────────────────────────
    def add(self, text="hello world", **kw):
        return store.add_entry(text=text, **kw)

    def raw_lines(self):
        return [l for l in self.history.read_text(encoding="utf-8",
                                                  errors="replace").splitlines() if l]

    def append_raw(self, line: str):
        with open(self.history, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class TestAddAndLoad(StoreTestCase):
    def test_one_entry_is_one_line_and_round_trips(self):
        eid = store.add_entry(text="send the report on Friday", raw="send teh report",
                              model="turbo", app="Notes", bundle_id="com.apple.Notes",
                              category="writing", latency=1.234, audio_secs=2.567,
                              chunks=2, outcome="ok", pasted=True,
                              paste_detail="cmd-v")

        self.assertEqual(len(self.raw_lines()), 1)
        entries = store.load()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["id"], eid, "add_entry returned an id load() disagrees with")
        self.assertEqual(e["type"], "entry")
        self.assertEqual(e["text"], "send the report on Friday")
        self.assertEqual(e["raw"], "send teh report")
        self.assertEqual(e["model"], "turbo")
        self.assertEqual(e["app"], "Notes")
        self.assertEqual(e["bundle_id"], "com.apple.Notes")
        self.assertEqual(e["category"], "writing")
        self.assertEqual(e["latency"], 1.23)
        self.assertEqual(e["audio_secs"], 2.57)
        self.assertEqual(e["chunks"], 2)
        self.assertEqual(e["words"], 5)
        self.assertEqual(e["chars"], len("send the report on Friday"))
        self.assertEqual(e["outcome"], "ok")
        self.assertFalse(e["pinned"])
        self.assertFalse(e["deleted"])
        self.assertTrue(e["ts"])

    def test_load_on_a_missing_file_is_empty_not_an_error(self):
        self.assertFalse(self.history.exists())
        self.assertEqual(store.load(), [])

    def test_ids_are_unique(self):
        ids = {self.add(text=f"line {i}") for i in range(50)}
        self.assertEqual(len(ids), 50)

    def test_newest_first(self):
        ids = [self.add(text=f"entry {i}") for i in range(5)]
        got = [e["id"] for e in store.load()]
        self.assertEqual(got, list(reversed(ids)))
        self.assertEqual(store.load()[0]["text"], "entry 4")

    def test_empty_text_counts_zero_words(self):
        self.add(text="", outcome="rejected", rejected="silence")
        e = store.load()[0]
        self.assertEqual(e["words"], 0)
        self.assertEqual(e["chars"], 0)
        self.assertEqual(e["outcome"], "rejected")

    def test_a_rejected_entry_is_not_mirrored_to_markdown(self):
        self.add(text="", outcome="rejected", rejected="silence")
        self.assertFalse(self.md.exists())


class TestPatches(StoreTestCase):
    def test_pin_and_unpin(self):
        eid = self.add()
        store.patch(eid, pinned=True)
        self.assertTrue(store.load()[0]["pinned"])
        store.patch(eid, pinned=False)
        self.assertFalse(store.load()[0]["pinned"])

    def test_delete_hides_the_entry_unless_asked_for(self):
        keep = self.add(text="keep me")
        gone = self.add(text="delete me")
        store.patch(gone, deleted=True)

        visible = store.load()
        self.assertEqual([e["id"] for e in visible], [keep])
        both = store.load(include_deleted=True)
        self.assertEqual({e["id"] for e in both}, {keep, gone})
        self.assertTrue(next(e for e in both if e["id"] == gone)["deleted"])

    def test_an_edit_updates_counts_and_marks_the_entry_edited(self):
        eid = self.add(text="one two three")
        self.assertNotIn("edited", store.load()[0])

        store.patch(eid, text="one two three four five")
        e = store.load()[0]
        self.assertEqual(e["text"], "one two three four five")
        self.assertEqual(e["words"], 5)
        self.assertEqual(e["chars"], len("one two three four five"))
        self.assertTrue(e["edited"])

    def test_the_last_patch_wins(self):
        eid = self.add()
        store.patch(eid, text="first edit")
        store.patch(eid, text="second edit")
        self.assertEqual(store.load()[0]["text"], "second edit")

    def test_paste_state_can_be_patched(self):
        eid = self.add(pasted=False, paste_detail="no ⌘V")
        store.patch(eid, pasted=True, paste_detail="retried")
        e = store.load()[0]
        self.assertTrue(e["pasted"])
        self.assertEqual(e["paste_detail"], "retried")

    def test_a_patch_for_an_unknown_id_is_ignored(self):
        eid = self.add(text="untouched")
        store.patch("nosuchid00", text="injected", deleted=True)
        entries = store.load()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], eid)
        self.assertEqual(entries[0]["text"], "untouched")

    def test_a_patch_that_arrives_before_its_entry_is_ignored(self):
        """Patches fold forward only; an out-of-order line must not resurrect
        or half-create an entry."""
        self.append_raw(json.dumps({"type": "patch", "id": "early0000000",
                                    "text": "ghost"}))
        eid = self.add(text="real")
        self.assertEqual([e["id"] for e in store.load()], [eid])

    def test_only_allowed_fields_can_be_patched(self):
        eid = self.add(text="original", outcome="ok", latency=1.0)
        store.patch(eid, outcome="rejected", latency=99.0, id="hijacked",
                    type="patchy", words=999, ts="1999-01-01T00:00:00")

        e = store.load()[0]
        self.assertEqual(e["id"], eid)
        self.assertEqual(e["type"], "entry")
        self.assertEqual(e["outcome"], "ok")
        self.assertEqual(e["latency"], 1.0)
        self.assertEqual(e["words"], 1)
        self.assertNotEqual(e["ts"], "1999-01-01T00:00:00")

    def test_a_patch_with_nothing_allowed_writes_no_line(self):
        self.add()
        before = self.raw_lines()
        store.patch("whatever0000", outcome="rejected")
        self.assertEqual(self.raw_lines(), before)

    def test_a_mixed_patch_applies_only_the_allowed_part(self):
        eid = self.add(text="original", outcome="ok")
        store.patch(eid, pinned=True, outcome="rejected")
        e = store.load()[0]
        self.assertTrue(e["pinned"])
        self.assertEqual(e["outcome"], "ok")


class TestMalformedLines(StoreTestCase):
    def test_torn_blank_and_non_object_lines_are_skipped(self):
        first = self.add(text="first good")
        self.append_raw('{"type": "entry", "id": "torn0000000"')   # truncated
        self.append_raw("")                                        # blank
        self.append_raw("   ")                                     # whitespace
        self.append_raw("[1, 2, 3]")                               # not an object
        self.append_raw('"just a string"')
        self.append_raw("42")
        self.append_raw('{"type": "entry", "text": "no id at all"}')
        self.append_raw("not json at all")
        second = self.add(text="second good")

        entries = store.load()
        self.assertEqual([e["id"] for e in entries], [second, first])
        self.assertEqual([e["text"] for e in entries],
                         ["second good", "first good"])

    def test_a_torn_final_line_does_not_lose_earlier_entries(self):
        eid = self.add(text="survivor")
        with open(self.history, "a", encoding="utf-8") as f:
            f.write('{"type": "entry", "id": "half", "text": "tru')
        self.assertEqual([e["id"] for e in store.load()], [eid])

    def test_invalid_utf8_does_not_crash_load(self):
        good = self.add(text="valid entry")
        with open(self.history, "ab") as f:
            f.write(b'{"type": "entry", "id": "bad", "text": "\xff\xfe\x00"}\n')
            f.write(b"\xff\xfe raw garbage bytes\n")
        also = self.add(text="after the garbage")

        entries = store.load()
        self.assertEqual([e["id"] for e in entries], [also, good])

    def test_a_directory_where_the_file_should_be_is_survivable(self):
        """load() promises never to raise; an unreadable path is the case
        that promise exists for."""
        self.history.mkdir()
        self.assertEqual(store.load(), [])


class TestUtteranceCannotCorruptTheStore(StoreTestCase):
    """The regression the whole JSONL design exists to prevent."""

    HOSTILE = ("intro line\n"
               "## 2026-01-01 10:00:00\n"
               "this tail used to vanish without any error\n"
               "*Raw:* and this looked like a raw-text marker\n"
               '{"type":"entry","id":"deadbeef","text":"forged"}\n'
               "final line")

    def test_a_dictated_markdown_heading_cannot_truncate_the_entry(self):
        eid = self.add(text=self.HOSTILE, raw="raw version")

        entries = store.load()
        self.assertEqual(len(entries), 1,
                         f"the utterance split into several records: {entries}")
        self.assertEqual(entries[0]["id"], eid)
        self.assertEqual(entries[0]["text"], self.HOSTILE,
                         "part of the transcript was eaten")
        self.assertIn("final line", entries[0]["text"])
        self.assertEqual(entries[0]["words"], len(self.HOSTILE.split()))

    def test_a_dictated_json_record_cannot_forge_an_entry(self):
        self.add(text=self.HOSTILE)
        self.assertEqual(len(self.raw_lines()), 1,
                         "the utterance's newlines were written literally")
        self.assertNotIn("deadbeef", [e["id"] for e in store.load(include_deleted=True)])

    def test_a_dictated_patch_record_cannot_delete_another_entry(self):
        victim = self.add(text="please keep me")
        self.add(text='{"type":"patch","id":"%s","deleted":true}' % victim)

        ids = [e["id"] for e in store.load()]
        self.assertIn(victim, ids, "an utterance deleted another entry")
        self.assertEqual(len(ids), 2)

    def test_the_markdown_mirror_is_written_but_never_read_back(self):
        self.add(text=self.HOSTILE, raw="raw version")
        mirrored = self.md.read_text(encoding="utf-8")
        self.assertIn("## 2026-01-01 10:00:00", mirrored)
        # …and load() still doesn't care what is in there.
        self.md.write_text("## 2020-01-01 00:00:00\n\ncompletely made up\n",
                           encoding="utf-8")
        self.assertEqual(len(store.load()), 1)
        self.assertEqual(store.load()[0]["text"], self.HOSTILE)


class TestCompaction(StoreTestCase):
    def _strip_edited(self, entries):
        return [{k: v for k, v in e.items() if k != "edited"} for e in entries]

    def test_compact_collapses_patches_and_preserves_order(self):
        a = self.add(text="first")
        b = self.add(text="second")
        c = self.add(text="third")
        store.patch(a, pinned=True)
        store.patch(b, deleted=True)
        store.patch(c, text="third, edited a bit")

        before = store.load(include_deleted=True)
        self.assertGreater(len(self.raw_lines()), 3)

        n = store.compact()
        self.assertEqual(n, 3, "compact() miscounted the surviving entries")
        self.assertEqual(len(self.raw_lines()), 3,
                         "the patch lines were not collapsed")

        after = store.load(include_deleted=True)
        self.assertEqual([e["id"] for e in after], [e["id"] for e in before])
        # `edited` is dropped by compact(); everything else must be identical.
        self.assertEqual(self._strip_edited(after), self._strip_edited(before))

        by_id = {e["id"]: e for e in after}
        self.assertTrue(by_id[a]["pinned"])
        self.assertTrue(by_id[b]["deleted"])
        self.assertEqual(by_id[c]["text"], "third, edited a bit")
        self.assertEqual(by_id[c]["words"], 4)

    def test_compact_keeps_the_visible_view_identical(self):
        for i in range(4):
            self.add(text=f"entry {i}")
        store.patch(store.load()[0]["id"], pinned=True)
        before = store.load()
        store.compact()
        self.assertEqual(self._strip_edited(store.load()),
                         self._strip_edited(before))

    def test_every_compacted_line_is_a_valid_entry(self):
        for i in range(3):
            self.add(text=f"entry {i}")
        store.compact()
        for line in self.raw_lines():
            rec = json.loads(line)
            self.assertEqual(rec["type"], "entry")
            self.assertIn("id", rec)

    def test_compact_drops_torn_lines_rather_than_carrying_them_forward(self):
        good = self.add(text="good")
        self.append_raw('{"type": "entry", "id": "torn')
        store.compact()
        self.assertEqual(self.raw_lines() and
                         [json.loads(l)["id"] for l in self.raw_lines()], [good])

    def test_maybe_compact_does_nothing_below_the_threshold(self):
        for i in range(5):
            self.add(text=f"entry {i}")
        store.patch(store.load()[0]["id"], pinned=True)
        before_bytes = self.history.read_bytes()
        self.assertLess(len(self.raw_lines()), store.COMPACT_THRESHOLD)

        store.maybe_compact()
        self.assertEqual(self.history.read_bytes(), before_bytes,
                         "maybe_compact rewrote the file below the threshold")

    def test_maybe_compact_on_a_missing_file_is_a_no_op(self):
        store.maybe_compact()
        self.assertFalse(self.history.exists())


class TestExports(StoreTestCase):
    def test_export_markdown_has_a_section_per_entry(self):
        self.add(text="first one", model="turbo", app="Notes")
        self.add(text="second one", model="small", app="Mail")
        md = store.export_markdown(store.load())

        self.assertTrue(md.startswith("# DictaFlow transcripts"))
        self.assertIn("first one", md)
        self.assertIn("second one", md)
        self.assertIn("Mail · small", md)
        self.assertEqual(md.count("\n## "), 2)

    def test_export_text_joins_only_non_empty_entries(self):
        entries = [{"text": "alpha"}, {"text": ""}, {}, {"text": "beta"}]
        self.assertEqual(store.export_text(entries), "alpha\n\nbeta")

    def test_exports_tolerate_missing_fields(self):
        entries = [{}, {"text": None}, {"ts": "2026-01-01T00:00:00"},
                   {"app": "Notes"}, {"model": "turbo", "text": "ok"}]
        md = store.export_markdown(entries)
        self.assertIsInstance(md, str)
        self.assertIn("ok", md)
        self.assertEqual(store.export_text(entries), "ok")

    def test_exports_of_nothing_are_empty_not_broken(self):
        self.assertEqual(store.export_text([]), "")
        self.assertIn("# DictaFlow transcripts", store.export_markdown([]))


class TestMigration(StoreTestCase):
    MD = ("# Transcripts\n"
          "\n## 2026-01-02 03:04:05\n"
          "\nthe first legacy transcript\n"
          "\n## 2026-01-03 09:10:11\n"
          "\nthe second legacy transcript\n"
          "\n*Raw:* the second legacy transkript\n")

    EVENTS = "\n".join([
        json.dumps({"ts": "2026-01-02T03:04:05", "outcome": "ok",
                    "model": "turbo", "latency": 1.0}),
        json.dumps({"ts": "2026-01-04T05:06:07", "outcome": "rejected",
                    "model": "small", "latency": 0.5, "audio_secs": 0.2}),
        "{not json}",
    ])

    def _write_legacy(self):
        self.md.write_text(self.MD, encoding="utf-8")
        self.events.write_text(self.EVENTS, encoding="utf-8")

    def test_legacy_files_are_imported_once(self):
        self._write_legacy()
        n = store.migrate_legacy()
        self.assertEqual(n, 3, "expected 2 transcripts + 1 non-ok event")

        entries = store.load()            # newest first
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(e["imported"] for e in entries))
        self.assertEqual([e["ts"] for e in entries],
                         ["2026-01-04T05:06:07",
                          "2026-01-03T09:10:11",
                          "2026-01-02T03:04:05"])

        second = entries[1]
        self.assertEqual(second["text"], "the second legacy transcript")
        self.assertEqual(second["raw"], "the second legacy transkript")
        self.assertEqual(second["words"], 4)
        self.assertEqual(entries[2]["text"], "the first legacy transcript")

        event = entries[0]
        self.assertEqual(event["outcome"], "rejected")
        self.assertEqual(event["model"], "small")
        self.assertEqual(event["text"], "")
        self.assertFalse(event["pasted"])

    def test_a_second_migration_is_a_no_op(self):
        self._write_legacy()
        store.migrate_legacy()
        before = store.load()

        self.assertEqual(store.migrate_legacy(), 0,
                         "migration ran twice and would duplicate history")
        self.assertEqual(store.load(), before)
        self.assertEqual(len(self.raw_lines()), 3)

    def test_migration_with_no_legacy_files_imports_nothing(self):
        self.assertEqual(store.migrate_legacy(), 0)
        self.assertEqual(store.load(), [])

    def test_migration_never_runs_once_history_exists(self):
        self.add(text="modern entry")
        self._write_legacy()
        self.assertEqual(store.migrate_legacy(), 0)
        self.assertEqual(len(store.load()), 1)


class TestConcurrency(StoreTestCase):
    def test_parallel_writes_never_interleave(self):
        """Every append is one locked, fsynced write; if that ever regresses
        the symptom is a half-written line, i.e. a lost transcript."""
        threads, per_thread = 8, 10
        errors: list[BaseException] = []

        def worker(n: int):
            try:
                for i in range(per_thread):
                    store.add_entry(text=f"thread {n} line {i}", model="turbo")
            except BaseException as exc:      # noqa: BLE001
                errors.append(exc)

        ts = [threading.Thread(target=worker, args=(n,)) for n in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
        self.assertEqual(errors, [])
        self.assertFalse(any(t.is_alive() for t in ts))

        lines = self.raw_lines()
        self.assertEqual(len(lines), threads * per_thread)
        texts = set()
        for line in lines:
            rec = json.loads(line)            # raises if a line was torn
            self.assertIsInstance(rec, dict)
            self.assertEqual(rec["type"], "entry")
            texts.add(rec["text"])

        expected = {f"thread {n} line {i}"
                    for n in range(threads) for i in range(per_thread)}
        self.assertEqual(texts, expected)
        self.assertEqual(len(store.load()), threads * per_thread)

    def test_parallel_writes_and_reads(self):
        stop = threading.Event()
        seen: list[int] = []
        errors: list[BaseException] = []

        def reader():
            try:
                while not stop.is_set():
                    seen.append(len(store.load()))
            except BaseException as exc:       # noqa: BLE001
                errors.append(exc)

        r = threading.Thread(target=reader, daemon=True)
        r.start()
        for i in range(40):
            self.add(text=f"entry {i}")
        stop.set()
        r.join(timeout=30)

        self.assertEqual(errors, [], "a concurrent read blew up")
        self.assertEqual(len(store.load()), 40)
        self.assertEqual(seen, sorted(seen),
                         "a concurrent read saw fewer entries than an earlier one")


if __name__ == "__main__":
    unittest.main(verbosity=2)
