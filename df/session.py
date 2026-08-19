"""The dictation state machine.

CONCURRENCY. Everything that was a bare attribute mutated from four different
threads now lives behind `self._lock`, and finished recordings go onto a
`queue.Queue` drained by exactly ONE worker thread. That single change fixes a
cluster of bugs at once:

  * Two transcriptions can no longer paste at the same time, so the clipboard
    save/restore can't interleave and destroy itself.
  * A key press during a transcription is no longer silently swallowed. The
    old `if not self._busy` guard dropped the entire press — you'd hold the
    key, speak a sentence, release, and nothing at all would happen, with no
    message. Now it records normally and queues.
  * Worker A's `finally: hide the indicator` can't hide the pill while
    worker B is still running, because there is only ever one worker.

ORDERING. The transcript is written to history BEFORE the paste is attempted.
Previously the paste ran first, and because pasting can raise, a paste failure
took the transcript down with it — the audio was gone and there was no record
of what had been said. Transcription is the expensive, unrepeatable part; it
gets persisted the moment it exists.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import appctx, asr, command, inject, sounds, store, textproc
from .audio import has_speech
from .config import SAMPLE_RATE
from .hud import STATE_DONE, STATE_ERROR, STATE_LISTEN, STATE_WORKING

IDLE        = "idle"
RECORDING   = "recording"
CANCELLING  = "cancelling"


@dataclass
class Job:
    samples: np.ndarray
    slot: str
    ctx: object
    streamer: object | None
    started: float
    handsfree: bool = False
    mode: str = "dictate"        # "dictate" | "command"


class Session:
    def __init__(self, cfg: dict, *, bar, recorder, model_paths: dict,
                 log=print):
        self.cfg = cfg
        self.bar = bar
        self.recorder = recorder
        self.model_paths = model_paths
        self.log = log

        self._lock = threading.RLock()
        self._state = IDLE
        self._slot = None
        self._ctx = None
        self._streamer = None
        self._started = 0.0
        self._handsfree = False
        self._mode = "dictate"
        self._cancelled = False
        self._warned_long = False

        self._queue: queue.Queue[Job] = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True,
                                        name="df-worker")
        self._worker.start()

        self.last_text = ""
        self.pending = 0             # queued or in-flight jobs, for the UI

    # ── recording ───────────────────────────────────────────────
    def begin(self, slot: str, *, handsfree: bool = False,
              mode: str = "dictate") -> None:
        with self._lock:
            if self._state == RECORDING:
                return
            try:
                self.recorder.start()
            except Exception as exc:
                # Reset so the NEXT press can rebuild the stream. The old code
                # left a dead handle in place, so one failure meant every
                # later press failed identically until the app was restarted.
                self.recorder.reset()
                self.bar.set_state(STATE_ERROR, "Mic", auto_hide=3.0)
                self.log(f"✗  microphone unavailable: {exc}. "
                         f"Is another app using it?")
                return

            self._state = RECORDING
            self._slot = slot
            self._mode = mode
            self._handsfree = handsfree
            self._cancelled = False
            self._warned_long = False
            self._started = time.monotonic()
            # Capture where the text has to go NOW, not when transcription
            # finishes — by then you may well have switched windows.
            self._ctx = appctx.capture(read_text=self.cfg.get("context_awareness", True))

            label = "Cmd" if mode == "command" else slot.title()
            self.bar.set_state(STATE_LISTEN, label)

            if mode == "dictate" and self.cfg.get("streaming", True):
                transcriber = asr.Transcriber(self.model_paths[slot], self.cfg)
                self._streamer = asr.StreamingTranscriber(
                    transcriber, self.recorder, self.cfg,
                    on_partial=self._on_partial)
                self._streamer.start()
            else:
                self._streamer = None

        sounds.play(self.cfg.get("sound_start"), self.cfg.get("sounds", True))

    def _on_partial(self, text: str) -> None:
        """Streaming produced more text — show a live word count, as the Flow
        Bar does. We deliberately don't display the partial text itself:
        chunk boundaries make it flicker and rewrite, which reads as an error."""
        self.bar.set_words(len(text.split()))

    def end(self) -> None:
        with self._lock:
            if self._state != RECORDING:
                return
            self._state = IDLE
            samples = self.recorder.stop()
            job = Job(samples=samples, slot=self._slot, ctx=self._ctx,
                      streamer=self._streamer, started=self._started,
                      handsfree=self._handsfree, mode=self._mode)
            self._streamer = None
            self.pending += 1

        if self.recorder.overflowed:
            self.log("⚠  the audio buffer overflowed — some frames were "
                     "dropped, so words may be missing from this transcript.")
        if self.recorder.truncated:
            self.log("⚠  recording hit the session length limit and was cut short.")

        self.bar.set_state(STATE_WORKING, "Cmd" if job.mode == "command"
                           else job.slot.title())
        self._queue.put(job)

    def cancel(self) -> None:
        """Esc — throw away whatever is being recorded."""
        with self._lock:
            if self._state != RECORDING:
                return
            self._state = IDLE
            self._cancelled = True
            self.recorder.stop()
            streamer, self._streamer = self._streamer, None
        if streamer is not None:
            streamer._stop.set()
        self.bar.set_state(STATE_ERROR, "Cancelled", auto_hide=1.2)
        sounds.play(self.cfg.get("sound_cancel"), self.cfg.get("sounds", True))
        self.log("✗  cancelled.")

    def discard_tap(self, slot: str) -> None:
        """A press too short to be speech. Drop the audio without a sound."""
        with self._lock:
            if self._state != RECORDING:
                return
            self._state = IDLE
            self.recorder.stop()
            streamer, self._streamer = self._streamer, None
        if streamer is not None:
            streamer._stop.set()
        self.bar.hide()

    def is_recording(self) -> bool:
        """For the heartbeat. Reads shared state, so it takes the lock."""
        with self._lock:
            return self._state == RECORDING

    # ── periodic, called from the main loop ─────────────────────
    def tick(self) -> None:
        with self._lock:
            recording = self._state == RECORDING
        if not recording:
            return
        self.bar.push_level(self.recorder.level)
        elapsed = self.recorder.elapsed
        warn_at = float(self.cfg.get("warn_session_seconds", 1140))
        max_at  = float(self.cfg.get("max_session_seconds", 1200))
        if not self._warned_long and elapsed > warn_at:
            self._warned_long = True
            self.log(f"⚠  {int(elapsed)}s recorded — this session stops "
                     f"automatically at {int(max_at)}s.")
        if elapsed > max_at:
            self.log("⚠  session length limit reached; finishing up.")
            self.end()

    # ── the single worker ───────────────────────────────────────
    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job.mode == "command":
                    self._run_command(job)
                else:
                    self._run_dictation(job)
            except Exception as exc:
                self.log(f"✗  {type(exc).__name__}: {exc}")
                self.bar.set_state(STATE_ERROR, "Error", auto_hide=3.0)
            finally:
                with self._lock:
                    self.pending = max(0, self.pending - 1)
                self._queue.task_done()

    # ── dictation ───────────────────────────────────────────────
    def _run_dictation(self, job: Job) -> None:
        audio_secs = len(job.samples) / SAMPLE_RATE
        transcriber = asr.Transcriber(self.model_paths[job.slot], self.cfg)

        if job.streamer is not None:
            result = job.streamer.finish(job.samples)
            for err in result.get("errors", []):
                self.log(f"⚠  streaming chunk failed: {err}")
            # The streamer consumed the transcribed prefix out of the
            # recorder's buffer, so `job.samples` is only the tail. Take the
            # true length from the streamer or the stats under-report badly.
            audio_secs = result.get("audio_secs", audio_secs)
        else:
            if not has_speech(job.samples):
                self._reject(job, "no speech detected", audio_secs)
                return
            result = transcriber.run(job.samples)

        latency = time.monotonic() - job.started
        raw = result.get("raw_text", "") or ""
        if not result.get("text"):
            self._reject(job, result.get("rejected") or "no speech", audio_secs,
                         raw=raw)
            return

        processed = self._format(result["text"], job.ctx)
        text = processed["text"]
        if not text.strip():
            self._reject(job, "empty after formatting", audio_secs, raw=raw)
            return

        # Persist FIRST. Everything below here can fail; the transcript
        # must survive all of it.
        entry_id = store.add_entry(
            text=text, raw=raw, model=job.slot,
            app=getattr(job.ctx, "name", ""),
            bundle_id=getattr(job.ctx, "bundle_id", ""),
            category=getattr(job.ctx, "category", ""),
            latency=latency, audio_secs=audio_secs,
            chunks=result.get("chunks", 1), outcome="ok",
        )
        self.last_text = text

        target_ok = True
        if self.cfg.get("context_awareness", True) and job.ctx is not None:
            target_ok = appctx.activate(job.ctx)
            if not target_ok:
                self.log(f"⚠  could not return focus to "
                         f"{getattr(job.ctx, 'name', 'the original app')}; "
                         f"pasting into whatever is frontmost.")

        paste = inject.insert(text, self.cfg, ctx=job.ctx)
        if paste.ok:
            if processed.get("command") == "enter":
                time.sleep(0.05)
                inject.press_enter()
            sounds.play(self.cfg.get("sound_done"), self.cfg.get("sounds", True))
            self.bar.set_state(STATE_DONE, "✓", auto_hide=0.8)
            preview = text[:70] + ("…" if len(text) > 70 else "")
            where = getattr(job.ctx, "name", "") or "?"
            self.log(f"✓  [{job.slot}] {preview}  ({latency:.1f}s → {where})")
        else:
            # Never silent: the text exists and is recoverable, and the user
            # is told exactly how. Record the failure against the entry so the
            # dashboard can show which dictations never reached their target.
            store.patch(entry_id, pasted=False, paste_detail=paste.detail)
            self.bar.set_state(STATE_ERROR, "⌘V", auto_hide=6.0)
            self.log(f"✗  could not insert the text: {paste.detail}")

    def _reject(self, job: Job, reason: str, audio_secs: float,
                raw: str = "") -> None:
        store.add_entry(text="", raw=raw, model=job.slot,
                        app=getattr(job.ctx, "name", ""),
                        bundle_id=getattr(job.ctx, "bundle_id", ""),
                        category=getattr(job.ctx, "category", ""),
                        audio_secs=audio_secs, outcome="rejected",
                        rejected=reason, pasted=False)
        self.bar.set_state(STATE_ERROR, "—", auto_hide=1.5)
        detail = f" ({raw[:60]}…)" if raw else ""
        self.log(f"✗  discarded: {reason}{detail}")
        if self.recorder.looks_muted():
            self.log("   The microphone is returning pure silence. Check "
                     "System Settings → Privacy & Security → Microphone.")

    # ── formatting ──────────────────────────────────────────────
    def _format(self, text: str, ctx) -> dict:
        chat = bool(ctx is not None and getattr(ctx, "is_chat", False))
        mid  = bool(ctx is not None and getattr(ctx, "mid_sentence", False))
        return textproc.process(
            text,
            level=self.cfg.get("cleanup_level", "medium"),
            dictionary=self.cfg.get("dictionary") or [],
            snippets=self.cfg.get("snippets") or [],
            spoken_punctuation=self.cfg.get("spoken_punctuation", True),
            backtrack=self.cfg.get("backtrack", True),
            auto_lists=self.cfg.get("auto_lists", True),
            strip_trailing_period=(
                chat and self.cfg.get("strip_trailing_period_in_chat", True)),
            lowercase_first=(
                mid and self.cfg.get("smart_capitalization", True)),
        )

    # ── command mode ────────────────────────────────────────────
    def _run_command(self, job: Job) -> None:
        if not has_speech(job.samples):
            self.bar.set_state(STATE_ERROR, "—", auto_hide=1.5)
            self.log("✗  command mode: no instruction heard.")
            return
        transcriber = asr.Transcriber(self.model_paths.get("turbo"), self.cfg)
        result = transcriber.run(job.samples)
        instruction = result.get("text", "").strip()
        if not instruction:
            self.bar.set_state(STATE_ERROR, "—", auto_hide=1.5)
            self.log(f"✗  command mode: {result.get('rejected') or 'nothing heard'}")
            return

        selection = command.read_selection(job.ctx)
        try:
            rewritten = command.run(instruction, selection, self.cfg)
        except command.CommandError as exc:
            self.bar.set_state(STATE_ERROR, "Cmd", auto_hide=5.0)
            self.log(f"✗  command mode: {exc}")
            return

        store.add_entry(text=rewritten, raw=f"[command] {instruction}",
                        model="command", app=getattr(job.ctx, "name", ""),
                        bundle_id=getattr(job.ctx, "bundle_id", ""),
                        category=getattr(job.ctx, "category", ""),
                        latency=time.monotonic() - job.started,
                        audio_secs=len(job.samples) / SAMPLE_RATE,
                        outcome="ok")
        if self.cfg.get("context_awareness", True) and job.ctx is not None:
            appctx.activate(job.ctx)
        paste = inject.insert(rewritten, self.cfg, ctx=job.ctx)
        if paste.ok:
            self.bar.set_state(STATE_DONE, "✓", auto_hide=0.8)
            self.log(f"✓  [command] {instruction} → {rewritten[:60]}…")
        else:
            self.bar.set_state(STATE_ERROR, "⌘V", auto_hide=6.0)
            self.log(f"✗  could not insert the rewrite: {paste.detail}")

    # ── introspection ───────────────────────────────────────────
    @property
    def state(self) -> str:
        with self._lock:
            return self._state
