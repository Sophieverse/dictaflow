"""The Flow Bar: a floating pill with a live waveform, plus the menu bar item.

THREADING CONTRACT — the single most important thing in this file. AppKit is
not thread-safe and misbehaves *silently* when called off the main thread: no
exception, just a window that never redraws. So every public method here only
writes a desired state under a lock, and `pump()` — which must be called from
the main thread — reconciles the screen to it. Nothing else touches AppKit.

The window is a borderless NSPanel rather than an NSStatusItem-hosted popover
so it can float above full-screen apps. (The previous code's comment claimed
NSStatusItem won't render from an unsigned bundle; that was tested and is
false, so the menu bar item below is a real one.)
"""
from __future__ import annotations

import collections
import threading
import time

import objc
from Cocoa import (
    NSApplication, NSApplicationActivationPolicyAccessory, NSAnyEventMask,
    NSBackingStoreBuffered, NSBezierPath, NSColor, NSFont,
    NSMakeRect, NSMakePoint, NSMenu, NSMenuItem, NSPanel, NSScreen,
    NSStatusBar, NSTextField, NSView, NSVariableStatusItemLength,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
)
import Quartz

# Window levels. Above the screen saver so the pill stays visible over
# full-screen apps, which is where dictation is most often used.
_OVERLAY_LEVEL = 25

STATE_HIDDEN   = "hidden"
STATE_LISTEN   = "listening"
STATE_WORKING  = "transcribing"
STATE_DONE     = "done"
STATE_ERROR    = "error"

_COLORS = {
    STATE_LISTEN:  (0.87, 0.19, 0.28),   # red    — the mic is live
    STATE_WORKING: (0.96, 0.62, 0.10),   # amber  — thinking
    STATE_DONE:    (0.16, 0.70, 0.42),   # green  — inserted
    STATE_ERROR:   (0.75, 0.20, 0.60),   # purple — needs your attention
}

WAVE_POINTS = 44          # bars in the waveform
PILL_W, PILL_H = 260, 44
VERT_W, VERT_H = 44, 260  # when docked to a side edge


class WaveView(NSView):
    """Draws the level history as a symmetric bar waveform.

    Levels are pushed from the audio thread; drawing happens on the main
    thread during pump(). A deque with maxlen is the whole synchronisation
    story — appends and iteration are both atomic enough under the GIL for a
    display that redraws 20 times a second.
    """

    def initWithFrame_(self, frame):
        self = objc.super(WaveView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._levels = collections.deque([0.0] * WAVE_POINTS, maxlen=WAVE_POINTS)
        self._color = (1.0, 1.0, 1.0)
        self._vertical = False
        return self

    def pushLevel_(self, level):
        # Speech RMS sits well below 1.0, so scale before clamping or the
        # waveform is a flat line for anything but a shout. sqrt gives the
        # quiet end more visual range, which matters when you're whispering.
        scaled = min(1.0, (max(0.0, float(level)) ** 0.5) * 3.2)
        self._levels.append(scaled)

    def setVertical_(self, vertical):
        self._vertical = bool(vertical)

    def clearLevels(self):
        self._levels.extend([0.0] * WAVE_POINTS)

    def drawRect_(self, rect):
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height
        n = len(self._levels)
        if n == 0:
            return
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.95).set()
        if self._vertical:
            slot = h / n
            bar = max(1.5, slot * 0.55)
            for i, lvl in enumerate(self._levels):
                length = max(2.0, lvl * (w * 0.8))
                y = i * slot + (slot - bar) / 2
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect((w - length) / 2, y, length, bar), bar / 2, bar / 2)
                path.fill()
        else:
            slot = w / n
            bar = max(1.5, slot * 0.55)
            for i, lvl in enumerate(self._levels):
                length = max(2.0, lvl * (h * 0.8))
                x = i * slot + (slot - bar) / 2
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, (h - length) / 2, bar, length), bar / 2, bar / 2)
                path.fill()


class FlowBar:
    """The floating pill. All AppKit work happens in pump()."""

    def __init__(self, cfg: dict, on_stop=None):
        self.cfg = cfg
        self.on_stop = on_stop
        self.ok = False

        self._lock = threading.Lock()
        self._state = STATE_HIDDEN
        self._label = ""
        self._words = 0
        self._shown = None
        self._pending_levels: list[float] = []
        self._auto_hide_at = 0.0
        self._dock = cfg.get("bar_dock", "bottom")
        # Per-instance, not class-level: NSMenuItem holds only a weak
        # reference to its target, so these have to stay alive here or the
        # menu actions crash when clicked.
        self._menu_targets: list = []
        self._pending_menu = None

        try:
            self._build()
            self.ok = True
        except Exception as exc:
            print(f"(Flow Bar unavailable, continuing without it: {exc})")

    # ── construction (main thread, at startup) ──────────────────
    def _build(self) -> None:
        self._app = NSApplication.sharedApplication()
        self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

        vertical = self._dock in ("left", "right")
        w, h = (VERT_W, VERT_H) if vertical else (PILL_W, PILL_H)
        x, y = self._origin(w, h)

        # NonactivatingPanel so showing the bar never steals focus from the
        # app you're dictating into — stealing focus would send the paste to
        # the wrong window.
        self._win = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, w, h),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False,
        )
        self._win.setLevel_(_OVERLAY_LEVEL)
        self._win.setOpaque_(False)
        self._win.setHasShadow_(True)
        self._win.setIgnoresMouseEvents_(True)   # clicks pass through to your app
        self._win.setBackgroundColor_(NSColor.clearColor())
        # Show over full-screen spaces, and don't appear in Mission Control /
        # the window cycle.
        self._win.setCollectionBehavior_(1 << 0 | 1 << 6 | 1 << 8)

        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
        content.setWantsLayer_(True)
        layer = content.layer()
        layer.setCornerRadius_(min(w, h) / 2)
        layer.setMasksToBounds_(True)
        self._win.setContentView_(content)
        self._content = content

        if vertical:
            wave_frame  = NSMakeRect(4, 34, w - 8, h - 62)
            label_frame = NSMakeRect(2, 8, w - 4, 20)
        else:
            wave_frame  = NSMakeRect(46, 8, w - 108, h - 16)
            label_frame = NSMakeRect(8, h / 2 - 9, 40, 18)

        self._wave = WaveView.alloc().initWithFrame_(wave_frame)
        self._wave.setVertical_(vertical)
        content.addSubview_(self._wave)

        self._text = self._make_label(label_frame, 11, bold=True)
        content.addSubview_(self._text)

        count_frame = (NSMakeRect(2, h - 26, w - 4, 18) if vertical
                       else NSMakeRect(w - 58, h / 2 - 9, 50, 18))
        self._count = self._make_label(count_frame, 11, align=2 if vertical else 1)
        content.addSubview_(self._count)

        self._status_item = None
        if self.cfg.get("menu_bar_icon", True):
            self._build_menu_bar()

    def _make_label(self, frame, size, bold: bool = False, align: int = 0):
        field = NSTextField.alloc().initWithFrame_(frame)
        field.setBezeled_(False)
        field.setDrawsBackground_(False)
        field.setEditable_(False)
        field.setSelectable_(False)
        field.setAlignment_(align)
        field.setTextColor_(NSColor.whiteColor())
        field.setFont_(NSFont.boldSystemFontOfSize_(size) if bold
                       else NSFont.systemFontOfSize_(size))
        field.setStringValue_("")
        return field

    def _build_menu_bar(self) -> None:
        bar = NSStatusBar.systemStatusBar()
        self._status_item = bar.statusItemWithLength_(NSVariableStatusItemLength)
        button = self._status_item.button()
        if button is not None:
            button.setTitle_("◉")
            button.setToolTip_("DictaFlow")
        menu = NSMenu.alloc().init()
        self._menu = menu
        self._status_item.setMenu_(menu)
        self._menu_items: dict[str, object] = {}
        self.set_menu([("DictaFlow — idle", None, False)])

    def set_menu(self, rows: list[tuple]) -> None:
        """Rebuild the menu bar dropdown. `rows` are (title, action, enabled)
        or ("-", None, False) for a separator. Main thread only."""
        if self._status_item is None:
            return
        with self._lock:
            self._pending_menu = rows

    def _apply_menu(self, rows) -> None:
        self._menu.removeAllItems()
        self._menu_targets = []
        for title, action, enabled in rows:
            if title == "-":
                self._menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, None, "")
            item.setEnabled_(bool(enabled))
            if action is not None:
                handler = _MenuTarget.alloc().initWithCallback_(action)
                item.setTarget_(handler)
                item.setAction_(objc.selector(handler.fire_, signature=b"v@:@"))
                self._menu_targets.append(handler)   # keep it alive
            self._menu.addItem_(item)

    def _origin(self, w: float, h: float) -> tuple[float, float]:
        """Where the pill sits, recomputed each time so it survives a
        resolution change or an external display being plugged in."""
        try:
            screen = NSScreen.mainScreen().visibleFrame()
        except Exception:
            return 100.0, 100.0
        offset = float(self.cfg.get("bar_offset", 60))
        if self._dock == "left":
            return screen.origin.x + offset, screen.origin.y + (screen.size.height - h) / 2
        if self._dock == "right":
            return (screen.origin.x + screen.size.width - w - offset,
                    screen.origin.y + (screen.size.height - h) / 2)
        return (screen.origin.x + (screen.size.width - w) / 2,
                screen.origin.y + offset)

    # ── public API (any thread) ─────────────────────────────────
    def set_state(self, state: str, label: str = "", *, auto_hide: float = 0.0) -> None:
        with self._lock:
            self._state = state
            self._label = label
            self._auto_hide_at = time.monotonic() + auto_hide if auto_hide else 0.0
            if state == STATE_LISTEN:
                self._words = 0

    def push_level(self, level: float) -> None:
        with self._lock:
            self._pending_levels.append(level)

    def set_words(self, n: int) -> None:
        with self._lock:
            self._words = n

    def hide(self) -> None:
        self.set_state(STATE_HIDDEN)

    # ── main thread only ────────────────────────────────────────
    def _reconcile(self) -> None:
        with self._lock:
            state, label, words = self._state, self._label, self._words
            levels, self._pending_levels = self._pending_levels, []
            auto_hide_at = self._auto_hide_at
            pending_menu, self._pending_menu = self._pending_menu, None

        if pending_menu is not None:
            try:
                self._apply_menu(pending_menu)
            except Exception:
                pass

        if auto_hide_at and time.monotonic() > auto_hide_at:
            with self._lock:
                if self._auto_hide_at == auto_hide_at:
                    self._state = STATE_HIDDEN
                    self._auto_hide_at = 0.0
            state = STATE_HIDDEN

        if state == STATE_HIDDEN:
            if self._shown is not None:
                self._win.orderOut_(None)
                self._wave.clearLevels()
                self._shown = None
            self._set_menu_icon("◉")
            return

        for level in levels:
            self._wave.pushLevel_(level)

        signature = (state, label, words)
        if signature != self._shown:
            r, g, b = _COLORS.get(state, _COLORS[STATE_LISTEN])
            self._content.layer().setBackgroundColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.94).CGColor())
            self._text.setStringValue_(label)
            if self.cfg.get("show_word_count", True) and words:
                self._count.setStringValue_(f"{words}w")
            else:
                self._count.setStringValue_("")
            if self._shown is None:
                w = self._win.frame().size.width
                h = self._win.frame().size.height
                x, y = self._origin(w, h)
                self._win.setFrameOrigin_(NSMakePoint(x, y))
                self._win.orderFrontRegardless()
            self._shown = signature
            self._set_menu_icon("●" if state == STATE_LISTEN else "◍")

        if state == STATE_LISTEN and self.cfg.get("show_waveform", True):
            self._wave.setNeedsDisplay_(True)

    def _set_menu_icon(self, glyph: str) -> None:
        if self._status_item is None:
            return
        button = self._status_item.button()
        if button is not None and button.title() != glyph:
            button.setTitle_(glyph)

    def pump(self, seconds: float = 0.04) -> None:
        """Drain AppKit events and apply pending state. MAIN THREAD ONLY."""
        if not self.ok:
            time.sleep(seconds)
            return
        try:
            self._reconcile()
        except Exception:
            pass
        until = Quartz.NSDate.dateWithTimeIntervalSinceNow_(seconds)
        event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
            NSAnyEventMask, until, "kCFRunLoopDefaultMode", True)
        if event is not None:
            self._app.sendEvent_(event)


class _MenuTarget(objc.lookUpClass("NSObject")):
    """Holds a Python callback so an NSMenuItem can call back into us."""

    def initWithCallback_(self, callback):
        self = objc.super(_MenuTarget, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def fire_(self, sender):
        try:
            self._callback()
        except Exception as exc:
            print(f"⚠  menu action failed: {exc}")
