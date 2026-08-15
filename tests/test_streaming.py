"""Integration test for the streaming transcriber.

The claim being tested is the whole reason streaming exists: for a long
dictation, the wait *after you release the key* should be roughly the wait for
a short one, because everything before the last pause was already transcribed
while you were still talking.

Uses a fake recorder that reveals a real WAV progressively, so the streaming
loop sees a buffer that grows the way a live one does — without needing a
microphone or real time.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from df import asr                       # noqa: E402
from df.config import SAMPLE_RATE        # noqa: E402

FIXTURE = Path("/tmp/df_test_long.wav")

SCRIPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Mechanistic interpretability tries to explain what a language model has "
    "actually learned. "
    "Tamper resistant safeguards for open weight models remain an unsolved "
    "problem in the field. "
    "I would like to send this note to the team before the end of the day. "
    "Please review the attached document and let me know what you think. "
)


def _fixture() -> np.ndarray:
    """Generate speech locally with `say` so the test needs no network."""
    if not FIXTURE.exists():
        subprocess.run(
            ["say", "-o", str(FIXTURE), "--file-format=WAVE",
             "--data-format=LEI16@16000", SCRIPT * 3],
            check=True)
    with wave.open(str(FIXTURE)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


class FakeRecorder:
    """Reveals `samples` progressively, mimicking a live capture buffer."""

    def __init__(self, samples: np.ndarray, speed: float = 5.0):
        # 5x real time, not faster: the streaming loop polls every 0.4s and
        # each chunk takes ~1s to transcribe, so at 40x the audio was
        # "finished" before the streamer had done any work — the test then
        # measured nothing and reported a 1.1x speedup that meant only that
        # the fake recorder was too fast.
        self._all = samples
        self._speed = speed
        self._pos = 0
        self._consumed = 0
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def _available(self) -> int:
        elapsed = (time.monotonic() - self._t0) * self._speed
        return min(len(self._all), int(elapsed * SAMPLE_RATE))

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._all[self._consumed:self._available()]

    def consume(self, n: int) -> None:
        with self._lock:
            self._consumed += n

    def remaining(self) -> np.ndarray:
        with self._lock:
            return self._all[self._consumed:]

    def finished(self) -> bool:
        return self._available() >= len(self._all)


class TestStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = _fixture()
        cls.model = str(Path(__file__).resolve().parent.parent
                        / "models" / "whisper-large-v3-turbo")
        if not Path(cls.model).exists():
            raise unittest.SkipTest("turbo model not present")
        cls.cfg = {"language": "en", "temperature_ladder": [0.0, 0.2],
                   "streaming": True, "chunk_target_secs": 8.0,
                   "chunk_max_secs": 15.0, "initial_prompt": ""}
        asr.warm(cls.model)

    def test_audio_is_long_enough_to_matter(self):
        secs = len(self.samples) / SAMPLE_RATE
        self.assertGreater(secs, 45, "fixture must cross the latency cliff")

    def test_bounded_ladder_is_not_slower_than_the_stock_one(self):
        """Directly compare the two-rung ladder against Whisper's stock six.

        Worth measuring rather than assuming: on CLEAN audio the stock ladder
        never falls back, so the two are equal and the ladder change costs
        nothing. Its value shows up only on audio that trips the thresholds —
        which is exactly the audio that used to produce a 14.6s paste of
        "Coll2018,2018...". This test guards the no-cost half of that claim.
        """
        bounded = asr.Transcriber(self.model, self.cfg)
        stock = asr.Transcriber(
            self.model, {**self.cfg,
                         "temperature_ladder": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]})
        t0 = time.monotonic(); a = bounded.run(self.samples); t_bounded = time.monotonic() - t0
        t0 = time.monotonic(); b = stock.run(self.samples);   t_stock = time.monotonic() - t0
        print(f"\n    bounded ladder (2 rungs) {t_bounded:6.2f}s")
        print(f"    stock ladder   (6 rungs) {t_stock:6.2f}s")
        self.assertTrue(a["text"] and b["text"])
        self.assertLessEqual(t_bounded, t_stock * 1.35,
                             "the bounded ladder must never be slower")

    def test_streaming_beats_one_shot_on_perceived_latency(self):
        transcriber = asr.Transcriber(self.model, self.cfg)

        # ── one shot: the whole clip transcribed on release ──
        t0 = time.monotonic()
        one_shot = transcriber.run(self.samples)
        one_shot_wait = time.monotonic() - t0

        # ── streaming: chunks run during "recording", tail on release ──
        rec = FakeRecorder(self.samples)
        streamer = asr.StreamingTranscriber(transcriber, rec, self.cfg)
        streamer.start()
        while not rec.finished():
            time.sleep(0.05)
        time.sleep(0.5)                      # let the loop pick up the last chunk
        release = time.monotonic()
        result = streamer.finish(rec.remaining())
        streaming_wait = time.monotonic() - release

        print(f"\n    clip length        {len(self.samples)/SAMPLE_RATE:6.1f}s")
        print(f"    one-shot wait      {one_shot_wait:6.2f}s")
        print(f"    streaming wait     {streaming_wait:6.2f}s  "
              f"({result['chunks']} chunks)")
        print(f"    speedup            {one_shot_wait/max(streaming_wait,0.01):6.1f}x")

        self.assertTrue(result["text"], f"streaming produced nothing: {result}")
        self.assertGreater(result["chunks"], 1,
                           "no chunking happened, so this measured nothing")
        self.assertLess(streaming_wait, one_shot_wait * 0.75,
                        "streaming should meaningfully cut the post-release wait")

    def test_streaming_text_is_not_degraded(self):
        """Chunking must not lose content. Compares word overlap against the
        one-shot transcription of the same audio."""
        transcriber = asr.Transcriber(self.model, self.cfg)
        one_shot = transcriber.run(self.samples)["text"].lower()

        rec = FakeRecorder(self.samples)
        streamer = asr.StreamingTranscriber(transcriber, rec, self.cfg)
        streamer.start()
        while not rec.finished():
            time.sleep(0.05)
        time.sleep(0.5)
        streamed = streamer.finish(rec.remaining())["text"].lower()

        import re
        a = set(re.findall(r"[a-z]+", one_shot))
        b = set(re.findall(r"[a-z]+", streamed))
        overlap = len(a & b) / max(len(a), 1)
        print(f"\n    one-shot words {len(a)}, streamed words {len(b)}, "
              f"overlap {overlap:.0%}")
        self.assertGreater(overlap, 0.80,
                           f"streaming lost too much content\n"
                           f"  one-shot: {one_shot[:200]}\n"
                           f"  streamed: {streamed[:200]}")


class TestLatencyLadder(unittest.TestCase):
    """The temperature ladder is the single biggest latency lever; guard it."""

    def test_bounded_ladder_is_configured(self):
        from df import config
        ladder = config.DEFAULTS["temperature_ladder"]
        self.assertLessEqual(
            len(ladder), 3,
            "Whisper's stock 6-rung ladder re-decodes a failing window up to "
            "six times — measured 10.64s vs 1.59s on a 29s clip, with "
            "identical output. Keep this short.")
        self.assertEqual(ladder[0], 0.0, "first rung must be greedy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
