"""Tests for the KeyRouter state machine.

No real pynput listener is ever started — a listener would swallow the
keyboard of whoever is running the tests. Instead the router's private
`_on_press` / `_on_release` are driven directly with `pynput.keyboard.Key`
values, which is exactly what the listener thread does.

Timing is real (no fake clock) because `_finalize_tap` fires from a
`threading.Timer`, which uses the wall clock; a monkeypatched
`time.monotonic` would disagree with the timer and make the tests lie.
Waits are done by polling for the expected event with a generous deadline
rather than by sleeping a fixed amount, so a slow machine is slow, not flaky.
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pynput import keyboard as kb                                # noqa: E402

from df import hotkeys                                           # noqa: E402
from df.hotkeys import (DOUBLE_TAP_WINDOW, TAP_MAX_HOLD,         # noqa: E402
                        KeyRouter, label)

HANDLER_NAMES = ("on_hold_start", "on_hold_end", "on_handsfree_on",
                 "on_handsfree_off", "on_cancel", "on_tap_discarded",
                 "on_command_start", "on_command_end")

# A press+release this short is unambiguously a tap.
TAP = 0.02
# A hold this long is unambiguously a hold, with margin over TAP_MAX_HOLD.
HOLD = TAP_MAX_HOLD + 0.15
# How long to wait for a threading.Timer-driven callback before giving up.
TIMER_GRACE = DOUBLE_TAP_WINDOW + 1.5
# How long to wait before concluding a callback is never coming. Only has to
# clear the tap timer's own deadline, so it can be much shorter.
NEG_WAIT = DOUBLE_TAP_WINDOW + 0.35


class Recorder:
    """Collects every handler call, in order, with its arguments."""

    def __init__(self, raising: set[str] | None = None):
        self.events: list[tuple] = []
        self._lock = threading.Lock()
        self._raising = raising or set()
        self.handlers = {name: self._make(name) for name in HANDLER_NAMES}

    def _make(self, name):
        def fn(*args):
            with self._lock:
                self.events.append((name,) + args)
            if name in self._raising:
                raise RuntimeError(f"handler {name} blew up")
        return fn

    # ── inspection ──────────────────────────────────────────────
    def names(self) -> list[str]:
        with self._lock:
            return [e[0] for e in self.events]

    def count(self, name: str) -> int:
        return self.names().count(name)

    def of(self, name: str) -> list[tuple]:
        with self._lock:
            return [e for e in self.events if e[0] == name]

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def wait_for(self, name: str, n: int = 1, timeout: float = TIMER_GRACE) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.count(name) >= n:
                return True
            time.sleep(0.01)
        return False


class RouterTestCase(unittest.TestCase):
    TURBO = kb.Key.alt_r
    SMALL = kb.Key.cmd_r
    COMMAND = kb.Key.ctrl_r

    def setUp(self):
        self.rec = Recorder()
        self.router = self._router(self.rec)

    def tearDown(self):
        # Never leave a live timer behind to fire during a later test.
        self.router.stop()

    def _router(self, rec: Recorder) -> KeyRouter:
        return KeyRouter({"turbo": "alt_r", "small": "cmd_r",
                          "command": "ctrl_r"}, rec.handlers)

    # ── gesture helpers ─────────────────────────────────────────
    def tap(self, key, hold: float = TAP) -> None:
        self.router._on_press(key)
        time.sleep(hold)
        self.router._on_release(key)

    def hold(self, key, hold: float = HOLD) -> None:
        self.tap(key, hold)

    def engage_handsfree(self, key) -> None:
        """Double-tap `key` into hands-free and assert it took.

        The second press is deliberately held past TAP_MAX_HOLD; see
        `TestHandsFree.test_a_long_hold_does_not_stop_handsfree` for why the
        length of that second press is not a neutral detail.
        """
        self.tap(key)
        time.sleep(0.15)
        self.router._on_press(key)
        time.sleep(TAP_MAX_HOLD + 0.05)
        self.router._on_release(key)
        self.assertEqual(self.rec.count("on_handsfree_on"), 1,
                         f"hands-free did not engage: {self.rec.names()}")
        self.assertIsNotNone(self.router.handsfree_slot)

    def assert_never(self, name: str, within: float = NEG_WAIT) -> None:
        """Assert `name` has not fired and still hasn't after the tap timer
        would have had time to fire."""
        self.assertEqual(self.rec.count(name), 0,
                         f"{name} fired: {self.rec.names()}")
        time.sleep(within)
        self.assertEqual(self.rec.count(name), 0,
                         f"{name} fired late: {self.rec.names()}")


class TestHold(RouterTestCase):
    def test_hold_starts_and_ends_one_dictation(self):
        self.hold(self.TURBO)
        self.assertEqual(self.rec.events,
                         [("on_hold_start", "turbo"), ("on_hold_end", "turbo")])
        self.assert_never("on_tap_discarded")

    def test_the_bound_slot_is_reported_not_a_default(self):
        self.hold(self.SMALL)
        self.assertEqual(self.rec.events,
                         [("on_hold_start", "small"), ("on_hold_end", "small")])

    def test_holds_are_repeatable(self):
        for _ in range(3):
            self.hold(self.TURBO)
        self.assertEqual(self.rec.count("on_hold_start"), 3)
        self.assertEqual(self.rec.count("on_hold_end"), 3)
        self.assertEqual(self.rec.count("on_tap_discarded"), 0)

    def test_a_second_key_while_one_is_held_starts_nothing(self):
        """Two keys down at once must not open a second recording — the old
        code started one and then lost track of which release ended what."""
        self.router._on_press(self.TURBO)
        self.router._on_press(self.SMALL)
        time.sleep(HOLD)
        self.router._on_release(self.SMALL)
        self.router._on_release(self.TURBO)

        self.assertEqual(self.rec.count("on_hold_start"), 1)
        self.assertEqual(self.rec.of("on_hold_start")[0], ("on_hold_start", "turbo"))
        self.assertEqual(self.rec.of("on_hold_end"), [("on_hold_end", "turbo")])


class TestTap(RouterTestCase):
    def test_a_lone_tap_is_started_then_discarded(self):
        """A tap can't be rejected on the spot — it may be half a double-tap —
        so it starts a recording and a timer discards it once the double-tap
        window closes with no second press."""
        self.tap(self.TURBO)
        self.assertEqual(self.rec.events, [("on_hold_start", "turbo")])

        self.assertTrue(self.rec.wait_for("on_tap_discarded"),
                        f"the tap was never finalised: {self.rec.names()}")
        self.assertEqual(self.rec.of("on_tap_discarded"),
                         [("on_tap_discarded", "turbo")])
        self.assertEqual(self.rec.count("on_hold_end"), 0,
                         "a tap must not be transcribed as a hold")

    def test_the_tap_timer_waits_the_full_window(self):
        self.tap(self.TURBO)
        time.sleep(DOUBLE_TAP_WINDOW * 0.5)
        self.assertEqual(self.rec.count("on_tap_discarded"), 0,
                         "the tap was discarded before the double-tap window closed")
        self.assertTrue(self.rec.wait_for("on_tap_discarded"))


class TestDoubleTap(RouterTestCase):
    def test_double_tap_turns_on_handsfree_without_discarding_the_first_tap(self):
        """The pending tap timer must be cancelled by the second press.

        If it isn't, the first tap's audio is discarded *after* hands-free has
        already begun recording, which tears down the live session.
        """
        self.tap(self.TURBO)
        time.sleep(0.15)
        self.router._on_press(self.TURBO)

        self.assertEqual(self.rec.of("on_handsfree_on"),
                         [("on_handsfree_on", "turbo")])
        self.assertEqual(self.router.handsfree_slot, "turbo")
        self.assert_never("on_tap_discarded")

    def test_the_window_is_measured_release_to_press_not_release_to_release(self):
        """Regression: the window used to be tap-1-release → tap-2-release, so
        the second tap's own hold time (up to TAP_MAX_HOLD) ate most of the
        budget and left ~150ms to double-tap in.

        Here the second press lands 300ms after the first release — inside the
        450ms window — but is held for 200ms, putting the second *release* at
        500ms, outside it. Hands-free must still engage.
        """
        gap, second_hold = 0.30, 0.20
        self.assertLess(gap, DOUBLE_TAP_WINDOW)
        self.assertGreater(gap + second_hold, DOUBLE_TAP_WINDOW,
                           "the timings no longer distinguish the two rules")

        self.tap(self.TURBO)
        time.sleep(gap)
        self.router._on_press(self.TURBO)
        self.assertEqual(self.rec.count("on_handsfree_on"), 1,
                         f"the double-tap window is too tight: {self.rec.names()}")
        time.sleep(second_hold)
        self.router._on_release(self.TURBO)
        self.assertEqual(self.router.handsfree_slot, "turbo")

    def test_a_slow_second_tap_is_two_separate_taps(self):
        self.tap(self.TURBO)
        self.assertTrue(self.rec.wait_for("on_tap_discarded"))
        self.rec.clear()

        self.router._on_press(self.TURBO)
        self.assertEqual(self.rec.count("on_handsfree_on"), 0,
                         "a tap outside the window engaged hands-free")
        self.assertEqual(self.rec.of("on_hold_start"), [("on_hold_start", "turbo")])
        time.sleep(HOLD)
        self.router._on_release(self.TURBO)
        self.assertEqual(self.rec.of("on_hold_end"), [("on_hold_end", "turbo")])

    def test_a_second_tap_on_a_different_key_is_not_a_double_tap(self):
        self.tap(self.TURBO)
        time.sleep(0.15)
        self.router._on_press(self.SMALL)
        self.assertEqual(self.rec.count("on_handsfree_on"), 0)
        self.assertEqual(self.rec.of("on_hold_start")[-1], ("on_hold_start", "small"))


class TestHandsFree(RouterTestCase):
    def test_a_tap_stops_handsfree(self):
        self.engage_handsfree(self.TURBO)
        self.tap(self.TURBO)
        self.assertEqual(self.rec.of("on_handsfree_off"),
                         [("on_handsfree_off", "turbo")])
        self.assertIsNone(self.router.handsfree_slot)

    def test_a_long_hold_does_not_stop_handsfree(self):
        """Current behaviour, asserted deliberately: while hands-free is on,
        only a *short* tap stops it. A long press of the same key is ignored
        entirely — it neither stops hands-free nor starts a second recording.
        """
        self.engage_handsfree(self.TURBO)
        self.rec.clear()

        self.router._on_press(self.TURBO)
        time.sleep(HOLD)
        self.router._on_release(self.TURBO)

        self.assertEqual(self.rec.events, [],
                         f"a long hold changed hands-free state: {self.rec.names()}")
        self.assertEqual(self.router.handsfree_slot, "turbo")

    def test_the_other_key_does_not_disturb_handsfree(self):
        self.engage_handsfree(self.TURBO)
        self.rec.clear()

        self.hold(self.SMALL)
        self.assertEqual(self.rec.events, [],
                         f"another key leaked through hands-free: {self.rec.names()}")
        self.assertEqual(self.router.handsfree_slot, "turbo")

    def test_holds_work_again_after_handsfree_is_stopped(self):
        self.engage_handsfree(self.TURBO)
        self.tap(self.TURBO)
        self.assertTrue(self.rec.wait_for("on_handsfree_off"))
        self.rec.clear()

        # The stop-tap must not itself arm a discard timer.
        self.hold(self.SMALL)
        self.assertEqual(self.rec.events,
                         [("on_hold_start", "small"), ("on_hold_end", "small")])


class TestPendingTapCancellation(RouterTestCase):
    def test_an_unrelated_key_press_cancels_a_pending_tap(self):
        """Audit finding #2, which pasted real garbage into a real document.

        An accidental brush of one trigger key armed a 450ms discard timer.
        The user then started a genuine dictation with a different key, and
        450ms later the stale click-noise audio was finalised and pasted into
        the middle of what they were writing. The pending timer has to be
        cancelled by *any* recognised key press, not just by the same key.
        """
        self.tap(self.TURBO)                       # accidental brush
        self.assertEqual(self.rec.of("on_hold_start"), [("on_hold_start", "turbo")])

        time.sleep(0.15)
        self.router._on_press(self.SMALL)          # the real dictation begins
        self.assertEqual(self.rec.of("on_hold_start")[-1], ("on_hold_start", "small"),
                         "the real dictation never started")
        self.assertEqual(self.rec.count("on_handsfree_on"), 0)

        # Speak for a while — well past when the stale timer would have fired.
        time.sleep(NEG_WAIT)
        self.assertLessEqual(self.rec.count("on_tap_discarded"), 1,
                             "the stale tap fired more than once")
        self.assertEqual(self.rec.count("on_tap_discarded"), 0,
                         f"a stale tap was finalised mid-dictation: {self.rec.names()}")

        # And the state machine is still coherent: the hold completes normally.
        self.router._on_release(self.SMALL)
        self.assertEqual(self.rec.of("on_hold_end"), [("on_hold_end", "small")])

        # …and the next hold on that key still works.
        self.rec.clear()
        self.hold(self.SMALL)
        self.assertEqual(self.rec.events,
                         [("on_hold_start", "small"), ("on_hold_end", "small")])

    def test_a_cancelled_tap_never_fires_even_much_later(self):
        self.tap(self.TURBO)
        time.sleep(0.10)
        self.hold(self.SMALL)
        time.sleep(NEG_WAIT)
        self.assertEqual(self.rec.count("on_tap_discarded"), 0,
                         f"the cancelled timer still fired: {self.rec.names()}")


class TestEscape(RouterTestCase):
    def test_esc_cancels_a_hold_and_leaves_a_clean_state(self):
        self.router._on_press(self.TURBO)
        time.sleep(0.05)
        self.router._on_press(kb.Key.esc)
        self.assertEqual(self.rec.of("on_cancel"), [("on_cancel",)])

        time.sleep(HOLD)
        self.router._on_release(self.TURBO)        # the key comes up eventually
        self.assertEqual(self.rec.count("on_hold_end"), 0,
                         "a cancelled hold was still transcribed")
        self.assert_never("on_tap_discarded")

        self.rec.clear()
        self.hold(self.SMALL)
        self.assertEqual(self.rec.events,
                         [("on_hold_start", "small"), ("on_hold_end", "small")])

    def test_esc_cancels_handsfree(self):
        self.engage_handsfree(self.TURBO)
        self.rec.clear()

        self.router._on_press(kb.Key.esc)
        self.assertEqual(self.rec.of("on_cancel"), [("on_cancel",)])
        self.assertIsNone(self.router.handsfree_slot)
        self.assertEqual(self.rec.count("on_handsfree_off"), 0)

        self.rec.clear()
        self.hold(self.TURBO)
        self.assertEqual(self.rec.events,
                         [("on_hold_start", "turbo"), ("on_hold_end", "turbo")])

    def test_esc_while_idle_cancels_nothing(self):
        self.router._on_press(kb.Key.esc)
        self.router._on_release(kb.Key.esc)
        self.assertEqual(self.rec.events, [],
                         f"Esc fired handlers while idle: {self.rec.names()}")

    def test_esc_cancels_a_pending_tap_timer(self):
        self.router._on_press(self.TURBO)
        self.router._on_press(kb.Key.esc)
        self.router._on_release(self.TURBO)
        self.assertEqual(self.rec.of("on_cancel"), [("on_cancel",)])
        self.assert_never("on_tap_discarded")


class TestUnboundKeys(RouterTestCase):
    def test_an_unbound_function_key_is_ignored(self):
        self.router._on_press(kb.Key.f5)
        time.sleep(TAP)
        self.router._on_release(kb.Key.f5)
        self.assertEqual(self.rec.events, [])

    def test_a_character_key_is_ignored(self):
        key = kb.KeyCode.from_char("a")
        self.router._on_press(key)
        self.router._on_release(key)
        self.assertEqual(self.rec.events, [])

    def test_an_unbound_key_does_not_cancel_a_pending_tap(self):
        """Typing while a tap is pending is normal; only *trigger* keys are
        transitions of this state machine."""
        self.tap(self.TURBO)
        self.router._on_press(kb.KeyCode.from_char("x"))
        self.router._on_release(kb.KeyCode.from_char("x"))
        self.assertTrue(self.rec.wait_for("on_tap_discarded"))

    def test_a_key_bound_to_nothing_in_the_config_is_dropped(self):
        rec = Recorder()
        router = KeyRouter({"turbo": "no_such_key"}, rec.handlers)
        self.addCleanup(router.stop)
        router._on_press(kb.Key.alt_r)
        router._on_release(kb.Key.alt_r)
        self.assertEqual(rec.events, [])


class TestCommandSlot(RouterTestCase):
    def test_command_key_uses_the_command_handlers(self):
        self.router._on_press(self.COMMAND)
        time.sleep(HOLD)
        self.router._on_release(self.COMMAND)
        self.assertEqual(self.rec.events,
                         [("on_command_start",), ("on_command_end",)])
        self.assertEqual(self.rec.count("on_hold_start"), 0)
        self.assertEqual(self.rec.count("on_hold_end"), 0)

    def test_a_short_command_press_still_completes(self):
        """Command mode has no tap/double-tap semantics: a quick press is a
        command, not a discardable tap."""
        self.tap(self.COMMAND)
        self.assertEqual(self.rec.events,
                         [("on_command_start",), ("on_command_end",)])
        self.assert_never("on_tap_discarded")


class TestHandlerFailures(RouterTestCase):
    def test_a_raising_handler_does_not_escape_or_wedge_the_router(self):
        """An exception escaping into pynput kills the listener thread: the
        app keeps running but silently stops responding to every key, with no
        message. `_call` has to swallow and report instead.
        """
        rec = Recorder(raising={"on_hold_start"})
        router = self._router(rec)
        self.addCleanup(router.stop)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            router._on_press(self.TURBO)           # must not raise
            time.sleep(HOLD)
            router._on_release(self.TURBO)
            # …and the router still routes afterwards.
            router._on_press(self.SMALL)
            time.sleep(HOLD)
            router._on_release(self.SMALL)

        self.assertIn("on_hold_start", buf.getvalue(),
                      "the failure was swallowed with no report at all")
        self.assertEqual(rec.names(),
                         ["on_hold_start", "on_hold_end",
                          "on_hold_start", "on_hold_end"])

    def test_a_raising_tap_handler_does_not_wedge_the_router(self):
        rec = Recorder(raising={"on_tap_discarded"})
        router = self._router(rec)
        self.addCleanup(router.stop)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            router._on_press(self.TURBO)
            time.sleep(TAP)
            router._on_release(self.TURBO)
            deadline = time.monotonic() + TIMER_GRACE
            while time.monotonic() < deadline and rec.count("on_tap_discarded") < 1:
                time.sleep(0.01)
            router._on_press(self.SMALL)
            time.sleep(HOLD)
            router._on_release(self.SMALL)

        self.assertEqual(rec.count("on_tap_discarded"), 1)
        self.assertEqual(rec.of("on_hold_end"), [("on_hold_end", "small")])

    def test_missing_handlers_are_simply_not_called(self):
        router = KeyRouter({"turbo": "alt_r"}, {})
        self.addCleanup(router.stop)
        router._on_press(kb.Key.alt_r)
        time.sleep(TAP)
        router._on_release(kb.Key.alt_r)
        time.sleep(NEG_WAIT)                       # the tap timer also fires


class TestLabels(unittest.TestCase):
    def test_known_names_are_readable(self):
        self.assertEqual(label("alt_r"), "right ⌥")
        self.assertEqual(label("cmd_l"), "left ⌘")
        self.assertEqual(label("shift_r"), "right ⇧")

    def test_unknown_names_fall_back_to_the_raw_name(self):
        self.assertEqual(label("f13"), "f13")
        self.assertEqual(label("nonsense"), "nonsense")
        self.assertEqual(label(""), "")

    def test_every_bindable_key_has_some_label(self):
        for name in hotkeys.KEY_NAMES:
            self.assertTrue(label(name), f"{name} has no label")


if __name__ == "__main__":
    unittest.main(verbosity=2)
