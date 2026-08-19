"""The microphone must recover on its own once the machine does.

This file exists because of a specific three-day failure. A device open hung
at startup, the Recorder set a boolean saying so, and nothing ever cleared it.
CoreAudio recovered the same day; the app went on refusing every dictation
with "CoreAudio is still wedged" until it was restarted. The bug was not the
hang — hangs happen — it was making a transient condition permanent.

So the rule under test is: a failure may cost you a cooldown, never a session.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from df import audio  # noqa: E402


class FakeStream:
    """Stands in for sounddevice.InputStream."""

    def __init__(self, **kw):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class HangingOnce:
    """An InputStream factory that blocks the first time, then behaves.

    Models the real thing: CoreAudio does not fail an open, it stops
    returning. The block is released by `release` so the abandoned thread
    doesn't outlive the test.
    """

    def __init__(self, hangs: int = 1):
        self.remaining_hangs = hangs
        self.release = threading.Event()
        self.calls = 0
        self.streams: list[FakeStream] = []

    def __call__(self, **kw):
        self.calls += 1
        if self.remaining_hangs > 0:
            self.remaining_hangs -= 1
            self.release.wait(30)
        s = FakeStream(**kw)
        self.streams.append(s)
        return s


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self._real_stream = audio.sd.InputStream
        self._real_timeout = audio.OPEN_TIMEOUT_S
        self._real_cooldown = audio.WEDGE_COOLDOWN_S
        audio.OPEN_TIMEOUT_S = 0.2
        audio.WEDGE_COOLDOWN_S = 0.4

    def tearDown(self):
        audio.sd.InputStream = self._real_stream
        audio.OPEN_TIMEOUT_S = self._real_timeout
        audio.WEDGE_COOLDOWN_S = self._real_cooldown

    def _recorder(self):
        r = audio.Recorder(device=0, preroll_ms=100, max_seconds=10)
        return r

    def test_hung_open_does_not_disable_the_microphone_forever(self):
        """The regression. After a hang, a later attempt must be allowed."""
        factory = HangingOnce(hangs=1)
        audio.sd.InputStream = factory
        r = self._recorder()

        ok, detail = r.open_with_timeout(audio.OPEN_TIMEOUT_S)
        self.assertFalse(ok)
        self.assertIn("did not respond", detail)

        # During the cooldown we fail fast rather than stacking blocked opens.
        with self.assertRaises(RuntimeError) as cm:
            r.ensure_open()
        self.assertIn("retrying", str(cm.exception))

        factory.release.set()
        time.sleep(audio.WEDGE_COOLDOWN_S + 0.1)

        # ...and once it expires, we try again and succeed.
        r.ensure_open()
        self.assertIsNotNone(r._stream)
        r.close()

    def test_cooldown_message_names_a_retry_not_a_dead_end(self):
        """Wording is load-bearing: the old text told her to reboot audio."""
        factory = HangingOnce(hangs=1)
        audio.sd.InputStream = factory
        r = self._recorder()
        r.open_with_timeout(audio.OPEN_TIMEOUT_S)
        with self.assertRaises(RuntimeError) as cm:
            r.ensure_open()
        msg = str(cm.exception)
        self.assertNotIn("still wedged", msg)
        self.assertIn("automatically", msg)
        factory.release.set()

    def test_abandoned_open_that_finishes_late_installs_itself(self):
        """Self-healing: if CoreAudio returns, the next press just works."""
        factory = HangingOnce(hangs=1)
        audio.sd.InputStream = factory
        r = self._recorder()
        r.open_with_timeout(audio.OPEN_TIMEOUT_S)
        self.assertIsNone(r._stream)
        factory.release.set()
        deadline = time.monotonic() + 5
        while r._stream is None and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertIsNotNone(r._stream, "late open should install its stream")
        r.close()

    def test_reopen_clears_the_muted_verdict(self):
        """looks_muted() must describe the current stream, not history.

        Same latching mistake in a second place: the all-zero counters were
        never reset, so a mic that started denied and was later granted still
        reported itself muted.
        """
        audio.sd.InputStream = lambda **kw: FakeStream(**kw)
        r = self._recorder()
        r.open()
        r._total_blocks = 50
        r._all_zero_blocks = 50
        self.assertTrue(r.looks_muted())
        r.close()
        r._stream = None
        r.open()
        self.assertFalse(r.looks_muted(), "counters must reset on reopen")
        r.close()

    def test_failed_start_does_not_leak_the_device(self):
        """A stream we built but couldn't start must be closed, not dropped."""
        built: list[FakeStream] = []

        class FailingStart(FakeStream):
            def start(self):
                raise RuntimeError("device busy")

        def factory(**kw):
            s = FailingStart(**kw)
            built.append(s)
            return s

        audio.sd.InputStream = factory
        r = self._recorder()
        with self.assertRaises(RuntimeError):
            r.open()
        self.assertTrue(built and built[0].closed,
                        "half-built stream should be closed on failure")
        self.assertIsNone(r._stream)


class StallTests(unittest.TestCase):
    """An open stream whose callbacks stopped is the nastiest failure here.

    Everything else still looks right: the process is up, the handle is
    non-None, close() succeeds, looks_muted() says no because the first few
    blocks were real. The only evidence is that blocks stopped arriving.
    """

    def setUp(self):
        self._real_stream = audio.sd.InputStream
        self._real_stall = audio.STALL_AFTER_S
        self._real_reopen = audio.REOPEN_EVERY_S
        audio.STALL_AFTER_S = 0.2
        audio.REOPEN_EVERY_S = 0.1
        audio.sd.InputStream = lambda **kw: FakeStream(**kw)

    def tearDown(self):
        audio.sd.InputStream = self._real_stream
        audio.STALL_AFTER_S = self._real_stall
        audio.REOPEN_EVERY_S = self._real_reopen

    def _open(self):
        r = audio.Recorder(device=0, preroll_ms=100, max_seconds=10)
        r.open()
        return r

    def test_fresh_stream_counts_as_flowing(self):
        """Don't declare a stream dead before it has had a chance to speak."""
        r = self._open()
        self.assertTrue(r.is_flowing())
        r.close()

    def test_stream_with_no_callbacks_is_not_flowing(self):
        r = self._open()
        time.sleep(audio.STALL_AFTER_S + 0.05)
        self.assertFalse(r.is_flowing())
        self.assertIsNotNone(r._stream, "still 'open' — that is the trap")
        r.close()

    def test_stalled_stream_is_reopened(self):
        r = self._open()
        first = r._stream
        time.sleep(audio.STALL_AFTER_S + 0.05)
        note = r.recover_if_broken()
        self.assertIsNotNone(note)
        self.assertIn("reopened", note)
        self.assertIsNot(r._stream, first, "should be a new stream")
        self.assertTrue(r.is_flowing())
        r.close()

    def test_recovery_does_not_interrupt_a_recording(self):
        """Pulling the device mid-dictation loses words she already said."""
        r = self._open()
        r.start()
        time.sleep(audio.STALL_AFTER_S + 0.05)
        self.assertIsNone(r.recover_if_broken())
        r.stop()

    def test_recovery_is_rate_limited(self):
        """A device that stays dead must not be reopened every loop pass."""
        r = self._open()
        time.sleep(audio.STALL_AFTER_S + 0.05)
        self.assertIsNotNone(r.recover_if_broken())
        # Immediately stalled again, but inside the reopen interval.
        r._last_block_at = 0.0
        self.assertIsNone(r.recover_if_broken())
        r.close()

    def test_callback_marks_the_stream_alive(self):
        import numpy as np
        r = self._open()
        r._last_block_at = 0.0
        self.assertFalse(r.is_flowing())
        r._callback(np.zeros((audio.BLOCKSIZE, 1), dtype="int16"),
                    audio.BLOCKSIZE, None, None)
        self.assertTrue(r.is_flowing())
        r.close()


class SilenceTests(unittest.TestCase):
    """Blocks arriving on time, every sample exactly zero.

    The worst of the three failures: the stream is open, callbacks fire at
    the right rate, no error is raised, and the audio is nothing. Reopening
    does not help — the fault is below PortAudio — so the recorder has to
    reach past it and reset the device.
    """

    def setUp(self):
        self._real_stream = audio.sd.InputStream
        self._real_run = audio.SILENT_RUN_BLOCKS
        self._real_every = audio.RESET_EVERY_S
        self._real_reset = audio.coreaudio.reset_input_device
        self._real_muted = audio.coreaudio.input_is_muted
        audio.SILENT_RUN_BLOCKS = 3
        audio.RESET_EVERY_S = 0.0
        audio.sd.InputStream = lambda **kw: FakeStream(**kw)
        self.resets = []
        audio.coreaudio.reset_input_device = lambda *a, **k: (
            self.resets.append(1) or (True, "reset the input device (48000Hz)"))
        audio.coreaudio.input_is_muted = lambda: False

    def tearDown(self):
        audio.sd.InputStream = self._real_stream
        audio.SILENT_RUN_BLOCKS = self._real_run
        audio.RESET_EVERY_S = self._real_every
        audio.coreaudio.reset_input_device = self._real_reset
        audio.coreaudio.input_is_muted = self._real_muted

    def _rec(self):
        r = audio.Recorder(device=0, preroll_ms=100, max_seconds=10)
        r.open()
        return r

    def _feed(self, r, n, zero=True):
        import numpy as np
        block = np.zeros((audio.BLOCKSIZE, 1), dtype="int16")
        if not zero:
            block[0, 0] = 500
        for _ in range(n):
            r._callback(block, audio.BLOCKSIZE, None, None)

    def test_zero_run_triggers_a_device_reset(self):
        r = self._rec()
        self._feed(r, audio.SILENT_RUN_BLOCKS)
        self.assertTrue(r.is_silent())
        note = r.recover_if_broken()
        self.assertIsNotNone(note)
        self.assertIn("silence", note)
        self.assertEqual(len(self.resets), 1)
        r.close()

    def test_one_real_block_clears_the_verdict(self):
        """A run, not a tally. Recovery must be observable."""
        r = self._rec()
        self._feed(r, audio.SILENT_RUN_BLOCKS)
        self.assertTrue(r.is_silent())
        self._feed(r, 1, zero=False)
        self.assertFalse(r.is_silent())
        self.assertIsNone(r.recover_if_broken())
        self.assertEqual(self.resets, [])
        r.close()

    def test_reset_is_abandoned_after_a_few_tries(self):
        """If it is still silent after N resets, stop resetting her audio."""
        r = self._rec()
        for _ in range(audio.MAX_RESETS + 3):
            self._feed(r, audio.SILENT_RUN_BLOCKS)
            r.recover_if_broken()
        self.assertEqual(len(self.resets), audio.MAX_RESETS)
        r.close()

    def test_a_deliberately_muted_mic_is_not_reset(self):
        """Resetting the device would not unmute it, and is rude."""
        audio.coreaudio.input_is_muted = lambda: True
        r = self._rec()
        self._feed(r, audio.SILENT_RUN_BLOCKS)
        note = r.recover_if_broken()
        self.assertIn("muted", note)
        self.assertEqual(self.resets, [])
        r.close()

    def test_silence_during_a_recording_is_left_alone(self):
        """Never pull the device out from under audio she already spoke."""
        r = self._rec()
        r.start()
        self._feed(r, audio.SILENT_RUN_BLOCKS)
        self.assertIsNone(r.recover_if_broken())
        self.assertEqual(self.resets, [])
        r.stop()
        r.close()

    def test_reopen_clears_the_zero_run(self):
        r = self._rec()
        self._feed(r, audio.SILENT_RUN_BLOCKS)
        r.close()
        r._stream = None
        r.open()
        self.assertFalse(r.is_silent())
        r.close()


if __name__ == "__main__":
    unittest.main()
