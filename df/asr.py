"""Speech recognition: the Whisper wrapper, hallucination rejection, streaming.

Three measured facts drive every decision in this file.

1. WHISPER IS ENCODER-BOUND AND PADS TO 30s. Every clip, however short, is
   padded to a 30-second mel spectrogram and run through the full encoder.
   Quantization therefore buys nothing, and clip length barely matters — until
   you cross the window boundary.

2. THERE IS A LATENCY CLIFF AT ~20 SECONDS. Measured end-to-end on Turbo:

        clip     2s     5s    10s    20s    29s    45s    60s
        time   1.07   1.20   1.23   1.47  18.49  18.42  36.00

   The cliff is not the extra encoder pass — that would be linear. It is
   Whisper's temperature fallback: when a decoded window trips
   compression_ratio_threshold or logprob_threshold, the whole window is
   re-decoded at the next temperature, up to six times. Isolating it:

        29s clip, default ladder (6 rungs)   10.64s
        29s clip, temperature=0.0            1.59s

   ...and all six rungs produced the same bad output (compression ratio stayed
   at 2.98). The retries bought nothing and cost 6.7x. Hence the two-rung
   ladder in config: one escape hatch for a genuinely unlucky decode, without
   the pathological worst case.

   The real fix for long dictations is not to send long clips at all — see
   StreamingTranscriber below.

3. WHISPER'S OWN ANTI-HALLUCINATION GUARDS CANCEL EACH OTHER OUT. In
   mlx_whisper/transcribe.py the repetition check is skipped whenever
   no_speech_prob is high (transcribe.py:237-241), and the resulting
   "skip this window" decision is then reversed whenever avg_logprob is
   healthy (transcribe.py:304-309). Degenerate loops are *confidently*
   predicted, so both guards disarm and the garbage is returned. Worse, after
   exhausting the ladder the library returns the LAST result — the most
   randomly-sampled one — with no best-of selection and no reject path.
   So rejection has to happen here, after the fact.
"""
from __future__ import annotations

import collections
import re
import threading
import time
import zlib

import numpy as np

from .config import SAMPLE_RATE

# mlx_whisper's model cache is a single slot keyed on path, so alternating
# between two models reloads weights every call. Serialising all transcription
# through one lock avoids that thrash and also stops two threads driving MLX
# concurrently, which doubles peak memory on a cold model.
_ASR_LOCK = threading.Lock()


def to_float(samples: np.ndarray) -> np.ndarray:
    """int16 PCM → the float32 mono in [-1, 1] that mlx_whisper expects.

    Passing the array straight in skips writing a temp WAV and skips the
    ffmpeg subprocess mlx_whisper would otherwise shell out to for a path.
    Nothing in the library validates this contract, so we honour it here:
    mono, 1-D, float32, already at 16 kHz.
    """
    if samples.dtype == np.float32:
        arr = samples
    else:
        arr = samples.astype(np.float32) / 32768.0
    return np.ascontiguousarray(arr.reshape(-1))


def warm(model_path: str) -> None:
    """Load weights and compile kernels so the first real dictation is fast.

    Loading alone isn't enough: mlx_whisper's ModelHolder only calls
    mx.eval on the parameters, which doesn't compile the kernels. A throwaway
    transcription of silence does.
    """
    import mlx_whisper
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    with _ASR_LOCK:
        mlx_whisper.transcribe(silence, path_or_hf_repo=model_path, language="en")


# ──────────────────────────────────────────────────────────────
# Hallucination rejection
# ──────────────────────────────────────────────────────────────

# Whisper's training data is largely YouTube captions, so on audio with no
# intelligible speech it emits the phrase that most often captions such a
# moment. Matched only against the WHOLE output, so saying "thank you" inside
# a real sentence is unaffected.
STOCK_PHRASES = {
    "thank you", "thanks for watching", "thank you for watching",
    "thanks for watching!", "you", "bye", "okay", "beep", "please subscribe",
    "subscribe to my channel", "the end", "silence", "music", "applause",
    "foreign", "yeah", "mm-hmm", "uh", "um", "so", "oh",
    "transcription by castingwords", "subtitles by the amara.org community",
    "www.mooji.org", "thanks for listening",
}

# A degenerate loop is far more compressible than speech — but the ratio has
# to be measured over a FIXED WINDOW, not the whole string.
#
# Measured here on real transcripts concatenated to various lengths, because
# the first version of this check got it wrong and rejected a perfectly good
# 1200-character transcription:
#
#     chars    whole-string ratio    windowed max (200 chars)
#       100          1.11                    1.11
#      1500          2.01                    1.65
#      6000          2.47                    1.59
#     12000          3.10                    1.64
#
# The whole-string ratio drifts upward with length — English is redundant and
# zlib finds more back-references the more text you give it — so any fixed
# threshold against it is guaranteed to eventually reject long legitimate
# dictation. The windowed maximum is flat at ~1.65 at every length, while
# actual loops score 8–15 in some window. So 4.0 sits in a wide empty gap
# that does not move as a dictation gets longer.
COMPRESSION_LIMIT = 4.0
COMPRESSION_MIN_CHARS = 80
COMPRESSION_WINDOW = 200
COMPRESSION_STEP = 100

# Real English averages ~5 characters per word. The observed garbage was 889
# characters in a single whitespace-delimited token.
CHARS_PER_WORD_LIMIT = 25


def _zlib_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def _compression_ratio(text: str) -> float:
    """Worst (highest) compression ratio over a sliding fixed-size window.

    Fixed-size so the result does not depend on how long you spoke — see the
    table above. Taking the max rather than the mean is what catches a loop
    that begins after a legitimate opening sentence, which is the shape the
    observed failure actually had.
    """
    if len(text) <= COMPRESSION_WINDOW:
        return _zlib_ratio(text)
    return max(_zlib_ratio(text[i:i + COMPRESSION_WINDOW])
               for i in range(0, len(text) - COMPRESSION_WINDOW + 1,
                              COMPRESSION_STEP))


def _repeated_ngram_fraction(text: str, n: int = 12) -> float:
    """Fraction of the string covered by its single most frequent n-gram.

    Catches loops that whitespace-based counting misses entirely — the
    previous filter split on spaces, so an 889-character run of "2018" with no
    spaces counted as one word and short-circuited before any test ran.
    """
    if len(text) < n * 3:
        return 0.0
    counts = collections.Counter(text[i:i + n] for i in range(len(text) - n + 1))
    gram, hits = counts.most_common(1)[0]
    return (hits * n) / len(text)


def _word_repetition_fraction(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return 0.0
    # Look at both single words and adjacent pairs: "Red Red Red" and the
    # two-token alternation "Red Blue Red Blue" are both loops, and the latter
    # tops out at exactly 0.5 on single-word counting, which a `> 0.5` test
    # misses by design.
    top_word = collections.Counter(words).most_common(1)[0][1] / len(words)
    if len(words) < 10:
        return top_word
    pairs = [f"{a} {b}" for a, b in zip(words, words[1:])]
    top_pair = collections.Counter(pairs).most_common(1)[0][1] / len(pairs)
    return max(top_word, top_pair * 2)


def rejection_reason(text: str, segments: list[dict] | None = None) -> str | None:
    """Why this transcript should be discarded, or None to keep it.

    Returning a reason rather than a bool because every rejection is logged:
    a filter that silently eats real dictation is worse than no filter, and
    the only way to tell the difference later is to have recorded which test
    fired.
    """
    stripped = text.strip()
    if not stripped:
        return "empty"

    normalised = stripped.lower().strip(" .,!?\"'…-[]()")
    if normalised in STOCK_PHRASES:
        return f"stock phrase: {normalised!r}"

    words = stripped.split()
    if words and len(stripped) / len(words) > CHARS_PER_WORD_LIMIT:
        return (f"chars-per-word {len(stripped) / len(words):.0f} "
                f"exceeds {CHARS_PER_WORD_LIMIT}")

    if len(stripped) >= COMPRESSION_MIN_CHARS:
        ratio = _compression_ratio(stripped)
        if ratio > COMPRESSION_LIMIT:
            return f"compression ratio {ratio:.1f} exceeds {COMPRESSION_LIMIT}"

    if _repeated_ngram_fraction(stripped) > 0.5:
        return "repeated n-gram covers over half the text"

    if _word_repetition_fraction(stripped) > 0.55:
        return "one word or pair dominates the text"

    # Segment-level metrics. mlx_whisper computes these and then declines to
    # act on them (see the module docstring), so we act on them here.
    if segments:
        confident = [s for s in segments if s.get("no_speech_prob", 0) < 0.8]
        if segments and not confident:
            return "every segment scored as non-speech"
        bad = [s for s in segments if s.get("compression_ratio", 0) > 3.0]
        if bad and len(bad) == len(segments):
            return "every segment is degenerately repetitive"
    return None


# ──────────────────────────────────────────────────────────────
# Transcription
# ──────────────────────────────────────────────────────────────
class Transcriber:
    """Wraps one Whisper model. Thread-safe; calls are serialised."""

    def __init__(self, model_path: str, cfg: dict):
        self.model_path = model_path
        self.cfg = cfg

    def run(self, samples: np.ndarray, *, prompt: str | None = None) -> dict:
        """Transcribe int16 samples. Returns a result dict; never raises for
        ordinary bad audio, only for genuine failures (missing model, etc.)."""
        import mlx_whisper

        audio = to_float(samples)
        if audio.size < SAMPLE_RATE // 20:          # under 50ms is a keystroke
            return {"text": "", "segments": [], "rejected": "clip too short",
                    "duration": 0.0}

        opts: dict = {"condition_on_previous_text": False}
        ladder = self.cfg.get("temperature_ladder") or [0.0]
        opts["temperature"] = tuple(ladder) if len(ladder) > 1 else float(ladder[0])
        if self.cfg.get("language"):
            # Without this mlx_whisper runs detect_language() first — an entire
            # extra encoder pass to conclude you are speaking English. Measured
            # 1.93s → 0.99s on Turbo.
            opts["language"] = self.cfg["language"]
        seed = prompt if prompt is not None else self.cfg.get("initial_prompt")
        if seed:
            opts["initial_prompt"] = seed

        started = time.monotonic()
        with _ASR_LOCK:
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.model_path, **opts,
            )
        duration = time.monotonic() - started

        text     = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        reason   = rejection_reason(text, segments)
        return {
            "text":     "" if reason else text,
            "raw_text": text,
            "segments": segments,
            "rejected": reason,
            "duration": duration,
            "audio_secs": len(audio) / SAMPLE_RATE,
        }


# ──────────────────────────────────────────────────────────────
# Streaming
# ──────────────────────────────────────────────────────────────
class StreamingTranscriber:
    """Transcribes completed phrases while you are still speaking.

    This is what makes a long dictation feel as fast as a short one. Whisper's
    cost is essentially flat up to ~20s and then explodes (see the module
    docstring), so the strategy is never to hand it a long clip: whenever the
    buffer has grown past `chunk_target_secs` AND there is a natural pause to
    cut on, that prefix is transcribed on a worker thread and dropped from the
    buffer. On release only the untranscribed tail remains, which is short by
    construction.

    Cutting only at pauses matters — a boundary mid-word costs accuracy at the
    seam, and Whisper has no cross-chunk context here because
    condition_on_previous_text is off.

    Trade-off worth stating plainly: chunking loses whole-utterance context, so
    a very long sentence spanning a cut can be punctuated slightly differently
    than it would have been in one pass. That is the price of not waiting 18
    seconds, and it only applies past the target length.
    """

    def __init__(self, transcriber: Transcriber, recorder, cfg: dict,
                 on_partial=None):
        self.transcriber = transcriber
        self.recorder    = recorder
        self.cfg         = cfg
        self.on_partial  = on_partial

        self._parts: list[str] = []
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread  = None
        self._errors: list[str] = []
        # Audio the streamer has already consumed from the recorder. The
        # recorder's own buffer only holds the untranscribed tail, so without
        # this the logged duration of a long dictation would be the tail only.
        self._consumed_samples = 0

    def start(self) -> None:
        if not self.cfg.get("streaming", True):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="df-stream")
        self._thread.start()

    def _loop(self) -> None:
        target = float(self.cfg.get("chunk_target_secs", 12.0))
        hard   = float(self.cfg.get("chunk_max_secs", 20.0))
        while not self._stop.wait(0.4):
            try:
                buf = self.recorder.snapshot()
                secs = len(buf) / SAMPLE_RATE
                if secs < target:
                    continue
                from .audio import find_pause
                # Look for a pause in the region that would make a
                # well-sized chunk; never cut in the first two seconds.
                cut = find_pause(buf[: int(hard * SAMPLE_RATE)], search_from=2.0)
                if cut is None:
                    if secs < hard:
                        continue          # still room to wait for a pause
                    cut = int(hard * SAMPLE_RATE)   # forced cut, mid-phrase
                chunk = buf[:cut]
                self.recorder.consume(cut)
                with self._lock:
                    self._consumed_samples += cut
                self._transcribe_chunk(chunk)
            except Exception as exc:
                # A streaming failure must never lose the dictation: the tail
                # transcription on release still covers whatever is left in
                # the buffer. Record it so it is not silent.
                with self._lock:
                    self._errors.append(str(exc))

    def _transcribe_chunk(self, chunk: np.ndarray) -> None:
        from .audio import has_speech
        if not has_speech(chunk):
            return
        result = self.transcriber.run(chunk)
        if result["text"]:
            with self._lock:
                self._parts.append(result["text"])
                joined = " ".join(self._parts)
            if self.on_partial:
                self.on_partial(joined)

    def finish(self, tail: np.ndarray) -> dict:
        """Stop streaming, transcribe the remaining tail, return everything."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

        from .audio import has_speech
        tail_result = {"text": "", "rejected": None, "segments": [],
                       "duration": 0.0}
        if tail.size and has_speech(tail):
            tail_result = self.transcriber.run(tail)

        with self._lock:
            parts  = list(self._parts)
            errors = list(self._errors)
            consumed = self._consumed_samples
        if tail_result["text"]:
            parts.append(tail_result["text"])

        text = " ".join(p.strip() for p in parts if p.strip())
        # The joined text can be degenerate even when each chunk was fine —
        # e.g. the same phrase repeated across chunks — so re-check the whole.
        reason = rejection_reason(text) if text else (
            tail_result.get("rejected") or "no speech")
        return {
            "text": "" if reason else text,
            "raw_text": text,
            "chunks": len(parts),
            "rejected": reason,
            "errors": errors,
            "segments": tail_result.get("segments") or [],
            "audio_secs": (consumed + tail.size) / SAMPLE_RATE,
        }

    @property
    def partial(self) -> str:
        with self._lock:
            return " ".join(self._parts)
