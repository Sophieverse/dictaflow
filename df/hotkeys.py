"""Keyboard routing: turns raw key events into semantic dictation intents.

Kept separate from the session state machine because the previous version
interleaved the two, and the timing logic is subtle enough to deserve its own
tests:

  * A lone tap can't be finalised immediately — it might be the first half of
    a double-tap. So it arms a timer, and EVERY other transition has to cancel
    that timer. Missing one cancel is what produced the observed bug where an
    accidental brush of the key was transcribed and pasted 400ms into the
    next real dictation.

  * The double-tap window was measured release-to-release, so the second tap's
    own hold time (up to TAP_MAX_HOLD = 250ms) ate most of the 400ms budget,
    leaving ~150ms to double-tap in. That is far tighter than a system
    double-click and it explains the "Recording… Recording…" pairs in the log
    that never became hands-free. Now measured tap-1-release → tap-2-PRESS.

This module owns no audio and no UI; it just calls the handlers it was given.
"""
from __future__ import annotations

import threading
import time

from pynput import keyboard as kb

# A press+release shorter than this is a tap (a click), not a dictation hold.
TAP_MAX_HOLD = 0.25
# Gap allowed between tap 1's release and tap 2's press.
DOUBLE_TAP_WINDOW = 0.45

# Name → pynput key, for configurable bindings.
KEY_NAMES = {
    "alt_r": kb.Key.alt_r, "alt_l": kb.Key.alt_l,
    "cmd_r": kb.Key.cmd_r, "cmd_l": kb.Key.cmd_l,
    "ctrl_r": kb.Key.ctrl_r, "ctrl_l": kb.Key.ctrl_l,
    "shift_r": kb.Key.shift_r, "shift_l": kb.Key.shift_l,
    "f13": kb.Key.f13, "f14": kb.Key.f14, "f15": kb.Key.f15,
    "f16": kb.Key.f16, "f17": kb.Key.f17, "f18": kb.Key.f18,
    "f19": kb.Key.f19,
    "esc": kb.Key.esc,
}

LABELS = {
    "alt_r": "right ⌥", "alt_l": "left ⌥",
    "cmd_r": "right ⌘", "cmd_l": "left ⌘",
    "ctrl_r": "right ⌃", "ctrl_l": "left ⌃",
    "shift_r": "right ⇧", "shift_l": "left ⇧",
}


def label(name: str) -> str:
    return LABELS.get(name, name)


class KeyRouter:
    """Drives dictation intents from key events.

    Handlers (all called from pynput's listener thread, so they must be
    quick and must not raise):
        on_hold_start(slot)      a real hold began
        on_hold_end(slot)        that hold ended — transcribe it
        on_handsfree_on(slot)    double-tap toggled hands-free on
        on_handsfree_off(slot)   tap toggled it back off
        on_cancel()              Esc — discard whatever is in flight
        on_tap_discarded(slot)   a lone tap that produced nothing usable
        on_command_start()/on_command_end()   command-mode key
    """

    def __init__(self, bindings: dict[str, str], handlers: dict):
        # bindings: {"turbo": "alt_r", "small": "cmd_r", "command": "ctrl_r"}
        self._slot_for = {}
        for slot, key_name in bindings.items():
            key = KEY_NAMES.get(key_name)
            if key is not None:
                self._slot_for[key] = slot
        self._h = handlers
        self._lock = threading.RLock()

        self._held_slot = None          # slot of the key physically down
        self._held_key = None
        self._press_at = 0.0
        self._handsfree_slot = None
        self._last_tap_release = {}     # key -> monotonic time
        self._tap_timer = None
        self._tap_key = None
        self._listener = None
        self._down: set = set()         # modifier keys currently held, for chords

    # ── lifecycle ───────────────────────────────────────────────
    def start(self) -> None:
        self._listener = kb.Listener(on_press=self._on_press,
                                     on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        self._cancel_tap_timer()
        if self._listener is not None:
            self._listener.stop()

    @property
    def running(self) -> bool:
        return self._listener is not None and self._listener.running

    @property
    def handsfree_slot(self) -> str | None:
        return self._handsfree_slot

    # ── internals ───────────────────────────────────────────────
    def _call(self, name: str, *args) -> None:
        fn = self._h.get(name)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as exc:
            # An exception escaping into pynput kills the listener thread and
            # the whole app stops responding to keys, with no message.
            print(f"⚠  handler {name} failed: {exc}")

    def _cancel_tap_timer(self) -> None:
        timer, self._tap_timer = self._tap_timer, None
        self._tap_key = None
        if timer is not None:
            timer.cancel()

    # Chords are handled separately from the hold/tap machine because they are
    # instantaneous actions, not recordings — there is no press/release
    # duration to interpret. Matches Wispr Flow's ⌘⌃V / ⌘⌃C.
    _CHORDS = {
        "v": "on_paste_last",
        "c": "on_copy_last",
    }
    _CHORD_MODS = {kb.Key.cmd, kb.Key.cmd_l, kb.Key.cmd_r,
                   kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r}

    def _chord_active(self) -> bool:
        cmd = any(k in self._down for k in (kb.Key.cmd, kb.Key.cmd_l, kb.Key.cmd_r))
        ctrl = any(k in self._down for k in (kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r))
        return cmd and ctrl

    def _on_press(self, key) -> None:
        try:
            with self._lock:
                if key in self._CHORD_MODS:
                    self._down.add(key)
                char = getattr(key, "char", None)
                # Not while recording: right-⌘ and right-⌃ double as trigger
                # keys, so ⌘⌃V during a hold would both dictate and paste.
                if (char and self._chord_active()
                        and self._held_slot is None
                        and self._handsfree_slot is None):
                    handler = self._CHORDS.get(char.lower())
                    if handler:
                        self._call(handler)
                        return

                if key == kb.Key.esc:
                    if self._held_slot or self._handsfree_slot:
                        self._cancel_tap_timer()
                        self._held_slot = self._held_key = None
                        self._handsfree_slot = None
                        self._call("on_cancel")
                    return

                slot = self._slot_for.get(key)
                if slot is None:
                    return

                # Any recognised key press invalidates a pending lone tap.
                # Missing this cancel is what let stale click-noise audio get
                # transcribed in the middle of the next dictation.
                if self._tap_key is not None and self._tap_key != key:
                    self._cancel_tap_timer()

                if self._handsfree_slot is not None:
                    # Already recording hands-free; remember the press so the
                    # release can decide whether it was a stop-tap.
                    if slot == self._handsfree_slot:
                        self._press_at = time.monotonic()
                    return

                if self._held_slot is not None:
                    return                      # another key already down

                now = time.monotonic()
                # Double-tap detection: tap-1 release → tap-2 PRESS.
                last = self._last_tap_release.get(key)
                if last is not None and (now - last) < DOUBLE_TAP_WINDOW:
                    self._cancel_tap_timer()
                    self._last_tap_release.pop(key, None)
                    self._handsfree_slot = slot
                    self._held_slot = self._held_key = None
                    self._call("on_handsfree_on", slot)
                    return

                self._held_slot = slot
                self._held_key = key
                self._press_at = now
                if slot == "command":
                    self._call("on_command_start")
                else:
                    self._call("on_hold_start", slot)
        except Exception as exc:
            print(f"⚠  key press routing failed: {exc}")

    def _on_release(self, key) -> None:
        try:
            with self._lock:
                self._down.discard(key)
                slot = self._slot_for.get(key)
                if slot is None:
                    return
                now = time.monotonic()

                if self._handsfree_slot == slot:
                    held = now - self._press_at if self._press_at else 0.0
                    if held < TAP_MAX_HOLD:
                        self._handsfree_slot = None
                        self._press_at = 0.0
                        self._call("on_handsfree_off", slot)
                    return

                if key != self._held_key:
                    return
                held = now - self._press_at
                self._held_slot = self._held_key = None

                if slot == "command":
                    self._call("on_command_end")
                    return

                if held < TAP_MAX_HOLD:
                    # Too short to be speech. Don't finalise yet — it may be
                    # the first half of a double-tap. Arm a timer that only
                    # fires if no second press arrives.
                    self._last_tap_release[key] = now
                    self._tap_key = key
                    timer = threading.Timer(DOUBLE_TAP_WINDOW,
                                            self._finalize_tap, args=(key, slot))
                    timer.daemon = True
                    self._tap_timer = timer
                    timer.start()
                    return

                self._last_tap_release.pop(key, None)
                self._call("on_hold_end", slot)
        except Exception as exc:
            print(f"⚠  key release routing failed: {exc}")

    def _finalize_tap(self, key, slot) -> None:
        with self._lock:
            if self._tap_key != key:
                return                  # superseded by a newer transition
            self._tap_timer = None
            self._tap_key = None
            self._last_tap_release.pop(key, None)
        self._call("on_tap_discarded", slot)
