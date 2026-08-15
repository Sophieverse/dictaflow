"""End-to-end test of the dictation pipeline, with the paste stubbed out.

Covers the path a real dictation takes — audio in, transcribe, format, persist,
insert — plus the concurrency cases from the audit that used to corrupt state:
overlapping dictations, a press during a transcription, cancel mid-recording,
and a paste failure. The paste is stubbed because a real one would type into
whatever window happens to be focused while the tests run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from df import config, session as session_mod, store  # noqa: E402
from df.config import SAMPLE_RATE                     # noqa: E402
from df.inject import PasteResult                     # noqa: E402

SPEECH = Path("/tmp/df_test_speech.wav")
MODELS = Path(__file__).resolve().parent.parent / "models"


def _speech() -> np.ndarray:
    if not SPEECH.exists():
        subprocess.run(
            ["say", "-o", str(SPEECH), "--file-format=WAVE",
             "--data-format=LEI16@16000",
             "Please send the report to the team before Friday."],
            check=True)
    with wave.open(str(SPEECH)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


class StubRecorder:
    """Stands in for the real Recorder; hands back a fixed clip."""

    def __init__(self, samples):
        self.samples = samples
        self.level = 0.1
        self.elapsed = 0.0
        self.truncated = False
        self.overflowed = False
        self.started = 0
        self.reset_calls = 0
        self.fail_start = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("mic unavailable (-9986)")
        self.started += 1

    def stop(self):
        return self.samples

    def reset(self):
        self.reset_calls += 1

    def snapshot(self):
        return np.empty(0, dtype=np.int16)

    def consume(self, n):
        pass

    def looks_muted(self):
        return False


class StubBar:
    def __init__(self):
        self.states = []

    def set_state(self, state, label="", **kw):
        self.states.append(state)

    def push_level(self, level):
        pass

    def set_words(self, n):
        pass

    def hide(self):
        self.states.append("hidden")


class SessionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODELS / "whisper-small-mlx").exists():
            raise unittest.SkipTest("small model not present")
        cls.samples = _speech()

    def setUp(self):
        # Point the store at a scratch directory so tests never touch the
        # user's real transcript history.
        self.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self.tmp.name)
        self._patches = [
            mock.patch.object(store, "HISTORY_FILE", tmpdir / "history.jsonl"),
            mock.patch.object(store, "TRANSCRIPTS_DIR", tmpdir),
            mock.patch.object(store, "TRANSCRIPT_FILE", tmpdir / "transcripts.md"),
        ]
        for p in self._patches:
            p.start()

        self.cfg = dict(config.DEFAULTS)
        self.cfg.update(streaming=False, sounds=False, context_awareness=False,
                        language="en", cleanup_level="medium")
        self.bar = StubBar()
        self.recorder = StubRecorder(self.samples)
        self.logs = []
        self.pastes = []

        def fake_insert(text, cfg, ctx=None):
            self.pastes.append(text)
            return PasteResult(True, "stub")

        self.insert_patch = mock.patch.object(session_mod.inject, "insert",
                                              side_effect=fake_insert)
        self.insert_patch.start()

        self.session = session_mod.Session(
            self.cfg, bar=self.bar, recorder=self.recorder,
            model_paths={"turbo": str(MODELS / "whisper-small-mlx"),
                         "small": str(MODELS / "whisper-small-mlx"),
                         "command": str(MODELS / "whisper-small-mlx")},
            log=self.logs.append)

    def tearDown(self):
        self.insert_patch.stop()
        for p in self._patches:
            p.stop()
        self.tmp.cleanup()

    def _drain(self, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.session.pending == 0 and self.session._queue.empty():
                time.sleep(0.2)
                if self.session.pending == 0:
                    return True
            time.sleep(0.05)
        return False


class TestHappyPath(SessionTestCase):
    def test_dictation_transcribes_persists_and_pastes(self):
        self.session.begin("small")
        self.session.end()
        self.assertTrue(self._drain(), "job did not finish")

        self.assertEqual(len(self.pastes), 1, f"logs: {self.logs}")
        text = self.pastes[0]
        self.assertIn("report", text.lower())
        self.assertIn("friday", text.lower())

        entries = store.load()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["outcome"], "ok")
        self.assertEqual(entries[0]["text"], text)
        self.assertGreater(entries[0]["audio_secs"], 1.0)
        self.assertGreater(entries[0]["latency"], 0.0)

    def test_transcript_is_persisted_before_the_paste(self):
        """A paste failure must never take the transcript down with it.

        The old ordering pasted first; because pasting can raise, a paste
        failure meant the audio was gone AND nothing was written down.
        """
        seen = {}

        def failing_insert(text, cfg, ctx=None):
            seen["history_at_paste_time"] = len(store.load())
            raise RuntimeError("pasteboard exploded")

        self.insert_patch.stop()
        with mock.patch.object(session_mod.inject, "insert",
                               side_effect=failing_insert):
            self.session.begin("small")
            self.session.end()
            self.assertTrue(self._drain())
        self.insert_patch.start()

        self.assertEqual(seen.get("history_at_paste_time"), 1,
                         "the transcript was not written before pasting")
        self.assertEqual(len(store.load()), 1, "the transcript was lost")

    def test_paste_failure_is_reported_and_recorded(self):
        self.insert_patch.stop()
        with mock.patch.object(
                session_mod.inject, "insert",
                return_value=PasteResult(False, "clipboard-only", "no ⌘V")):
            self.session.begin("small")
            self.session.end()
            self.assertTrue(self._drain())
        self.insert_patch.start()

        self.assertTrue(any("could not insert" in m for m in self.logs),
                        f"paste failure was silent. logs={self.logs}")
        entry = store.load()[0]
        self.assertFalse(entry["pasted"])
        self.assertIn("⌘V", entry["paste_detail"])


class TestRejection(SessionTestCase):
    def test_silence_is_rejected_and_logged_not_pasted(self):
        self.recorder.samples = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
        self.session.begin("small")
        self.session.end()
        self.assertTrue(self._drain())

        self.assertEqual(self.pastes, [])
        entries = store.load()
        self.assertEqual(entries[0]["outcome"], "rejected")
        self.assertTrue(entries[0]["rejected"])
        self.assertTrue(any("discarded" in m for m in self.logs),
                        f"rejection was silent. logs={self.logs}")


class TestConcurrency(SessionTestCase):
    def test_a_press_during_transcription_is_not_dropped(self):
        """The old `if not self._busy` guard silently swallowed the entire
        press: you'd speak a full sentence and nothing at all would happen."""
        self.session.begin("small")
        self.session.end()
        # Immediately start another while the first is still in the queue.
        self.session.begin("small")
        self.session.end()
        self.assertTrue(self._drain(timeout=90))

        self.assertEqual(len(self.pastes), 2,
                         f"a dictation was dropped. logs={self.logs}")
        self.assertEqual(len(store.load()), 2)

    def test_pastes_never_overlap(self):
        """Two concurrent pastes interleaved their clipboard save/restore and
        destroyed it. One worker means that cannot happen."""
        active = []
        overlaps = []

        def slow_insert(text, cfg, ctx=None):
            active.append(1)
            if len(active) > 1:
                overlaps.append(len(active))
            time.sleep(0.25)
            active.pop()
            return PasteResult(True, "stub")

        self.insert_patch.stop()
        with mock.patch.object(session_mod.inject, "insert",
                               side_effect=slow_insert):
            for _ in range(3):
                self.session.begin("small")
                self.session.end()
            self.assertTrue(self._drain(timeout=120))
        self.insert_patch.start()
        self.assertEqual(overlaps, [], "two pastes ran at the same time")

    def test_cancel_discards_without_pasting(self):
        self.session.begin("small")
        self.session.cancel()
        time.sleep(0.5)
        self.assertEqual(self.pastes, [])
        self.assertEqual(store.load(), [])

    def test_short_tap_is_discarded_silently(self):
        self.session.begin("small")
        self.session.discard_tap("small")
        time.sleep(0.5)
        self.assertEqual(self.pastes, [])
        self.assertEqual(store.load(), [])

    def test_begin_twice_does_not_double_start(self):
        self.session.begin("small")
        self.session.begin("small")
        self.assertEqual(self.recorder.started, 1)
        self.session.end()
        self.assertTrue(self._drain())


class TestMicFailure(SessionTestCase):
    def test_mic_failure_resets_so_the_next_press_can_retry(self):
        """One failure used to poison every later press: the dead stream was
        left in place and re-raised forever until the app was restarted."""
        self.recorder.fail_start = True
        self.session.begin("small")
        self.assertEqual(self.recorder.reset_calls, 1,
                         "the recorder was not reset after a failure")
        self.assertTrue(any("microphone unavailable" in m for m in self.logs))
        self.assertEqual(self.session.state, "idle")

        self.recorder.fail_start = False
        self.session.begin("small")
        self.session.end()
        self.assertTrue(self._drain())
        self.assertEqual(len(self.pastes), 1, "could not recover after a mic failure")


if __name__ == "__main__":
    unittest.main(verbosity=2)
