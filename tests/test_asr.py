"""Tests for hallucination rejection.

The asymmetry that shapes every test here: a FALSE NEGATIVE pastes garbage,
which is annoying but visible and undoable. A FALSE POSITIVE silently eats a
dictation you already spoke, which is invisible and unrecoverable. So the
false-positive cases are the ones that matter most, and there are more of them.
"""
from __future__ import annotations

import random
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from df.asr import (COMPRESSION_LIMIT, _compression_ratio,  # noqa: E402
                    _repeated_ngram_fraction, _word_repetition_fraction,
                    rejection_reason)

REAL_TRANSCRIPTS = Path.home() / "transcriptions" / "transcripts.md"


def _real_entries() -> list[str]:
    if not REAL_TRANSCRIPTS.exists():
        return []
    md = REAL_TRANSCRIPTS.read_text(encoding="utf-8", errors="replace")
    entries = re.findall(
        r"^##\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s*\n(.*?)(?=^##\s+\d{4}|\Z)",
        md, re.S | re.M)
    return [e.strip() for e in entries if e.strip()]


class TestMustKeep(unittest.TestCase):
    """Real speech must never be rejected."""

    KEEP = [
        "Send that to Ernest today.",
        "Yes.",
        "Thank you for sending the draft, I'll review it tonight.",
        "The mechanistic interpretability results were inconclusive.",
        "No no no, that is not what I meant at all.",
        "The meeting is at 2:30 on March 14th in room 2018.",
        "One, finish the report. Two, send the presentation.",
        "Very very very important.",
        "I want to make it more like Wispr Flow with a dashboard, and I want "
        "it to be accurate when I am whispering.",
        "def transcribe(audio, model): return model.decode(audio)",
        "arielwalters12@gmail.com",
        "ha ha ha that's funny",
    ]

    def test_short_real_speech(self):
        for text in self.KEEP:
            with self.subTest(text=text[:40]):
                self.assertIsNone(rejection_reason(text))

    def test_long_real_speech_at_every_length(self):
        """The regression this file exists for.

        The first version used the compression ratio of the WHOLE string.
        That ratio rises with length for ordinary English, so a genuine
        1200-character transcription scored 4.3 and was thrown away. Any
        length must pass.
        """
        entries = _real_entries()
        if not entries:
            self.skipTest("no real transcripts available")
        clean = [e for e in entries
                 if len(e) >= 80 and rejection_reason(e) is None]
        self.assertTrue(clean, "no clean real transcripts to build from")
        random.seed(7)
        for target in (300, 800, 1500, 3000, 6000, 12000, 25000):
            text = ""
            while len(text) < target:
                text += random.choice(clean) + " "
            text = text[:target]
            with self.subTest(chars=target):
                self.assertIsNone(
                    rejection_reason(text),
                    f"{target} chars of real speech was rejected "
                    f"(windowed ratio {_compression_ratio(text):.2f})")

    def test_no_false_positives_across_the_whole_history(self):
        entries = _real_entries()
        if not entries:
            self.skipTest("no real transcripts available")
        rejected = [(e, rejection_reason(e)) for e in entries
                    if rejection_reason(e)]
        # Everything rejected must be a known hallucination shape, not prose.
        for text, reason in rejected:
            with self.subTest(text=text[:40]):
                looks_like_garbage = (
                    text.strip().lower().strip(" .,!?") in
                    ("thank you", "you", "bye", "okay")
                    or _compression_ratio(text) > COMPRESSION_LIMIT
                    or len(text) / max(len(text.split()), 1) > 25
                    or _word_repetition_fraction(text) > 0.55
                )
                self.assertTrue(
                    looks_like_garbage,
                    f"real prose rejected as {reason!r}: {text[:120]!r}")


class TestMustReject(unittest.TestCase):
    REJECT = {
        "observed 889-char loop": "Coll2018," + "2018" * 220,
        "stock phrase":           "Thank you.",
        "stock bare you":         " you ",
        "word loop":              "Red " * 30,
        "two-token alternation":  "Red Blue " * 20,
        "cjk no-space loop":      "こんにちは" * 60,
        "sentence then loop":     ("This is a real sentence about the weather "
                                   "today. " + "ha" * 300),
        "empty":                  "   ",
    }

    def test_rejects_known_garbage(self):
        for label, text in self.REJECT.items():
            with self.subTest(label):
                self.assertIsNotNone(rejection_reason(text),
                                     f"{label} was NOT rejected")


class TestMetrics(unittest.TestCase):
    def test_compression_ratio_is_length_stable(self):
        """A metric that drifts with length cannot have a fixed threshold.

        Built by cycling through DISTINCT clean transcripts rather than
        sampling with replacement — sampling could draw the same entry twice
        in a row, which really is a repetition loop, and the metric correctly
        flagging that would look like a failure of this test.
        """
        clean = sorted({e for e in _real_entries()
                        if len(e) >= 80 and rejection_reason(e) is None})
        if len(clean) < 5:
            self.skipTest("not enough distinct clean transcripts")
        ratios = []
        for target in (500, 2000, 8000):
            parts, i = [], 0
            while sum(len(p) for p in parts) < target:
                parts.append(clean[i % len(clean)])
                i += 1
            ratios.append(_compression_ratio(" ".join(parts)[:target]))
        self.assertLess(max(ratios) - min(ratios), 1.0,
                        f"windowed ratio drifted with length: {ratios}")
        self.assertLess(max(ratios), COMPRESSION_LIMIT,
                        "real speech must stay well under the threshold")

    def test_ngram_fraction(self):
        self.assertGreater(_repeated_ngram_fraction("abcdefghijkl" * 20), 0.9)
        self.assertLess(
            _repeated_ngram_fraction(
                "The quick brown fox jumps over the lazy dog near the river."),
            0.5)

    def test_word_repetition_catches_alternation(self):
        # Single-word counting tops out at exactly 0.5 here, which a strict
        # `> 0.5` test misses — that is why pairs are counted too.
        self.assertGreater(_word_repetition_fraction("Red Blue " * 20), 0.55)

    def test_segments_all_non_speech_are_rejected(self):
        segs = [{"no_speech_prob": 0.95, "compression_ratio": 1.2}]
        self.assertIsNotNone(rejection_reason("Some words here", segs))

    def test_healthy_segments_are_kept(self):
        segs = [{"no_speech_prob": 0.01, "compression_ratio": 1.4}]
        self.assertIsNone(rejection_reason("Some real words here", segs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
