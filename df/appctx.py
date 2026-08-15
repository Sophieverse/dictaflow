"""Context awareness: which app you're dictating into, and what's around the cursor.

Wispr Flow's equivalent feature collects "app info, textbox contents, on-screen
text, ... and a screenshot". The screenshot is the part that caused their 2025
privacy incident, and it is deliberately not implemented here. Everything below
uses only the Accessibility tree of the focused element, stays on this machine,
and can be turned off entirely with `context_awareness: false`.

What the context is actually used for:
  - picking a formatting style (chat apps drop the trailing period, prose
    apps keep it)
  - deciding whether you're mid-sentence, so an insertion isn't wrongly
    capitalised
  - reading the selection for Command Mode
  - re-focusing the right app before pasting, because transcription is async
    and you may have switched windows since
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# Bundle-id prefixes → style category. Matched by prefix so
# "com.tinyspeck.slackmacgap.helper" lands with Slack.
_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("chat", (
        "com.tinyspeck.slackmacgap", "com.apple.MobileSMS", "com.apple.iChat",
        "net.whatsapp.WhatsApp", "com.hnc.Discord", "ru.keepcoder.Telegram",
        "org.whispersystems.signal-desktop", "com.microsoft.teams",
        "com.tencent.xinWeChat", "com.electron.lark", "com.google.Chat",
        "com.automattic.beeper", "com.texts.desktop", "com.readdle.smartemail",
    )),
    ("email", (
        "com.apple.mail", "com.superhuman", "com.readdle.spark",
        "com.microsoft.Outlook", "com.airmail",
    )),
    ("code", (
        "com.microsoft.VSCode", "com.todesktop.230313mzl4w4u92",   # Cursor
        "com.apple.dt.Xcode", "com.jetbrains", "com.sublimetext",
        "dev.zed.Zed", "com.exafunction.windsurf", "com.neovide",
    )),
    ("terminal", (
        "com.apple.Terminal", "com.googlecode.iterm2", "co.zeit.hyper",
        "net.kovidgoyal.kitty", "com.github.wez.wezterm", "dev.warp.Warp-Stable",
        "org.alacritty",
    )),
    ("notes", (
        "notion.id", "md.obsidian", "net.shinyfrog.bear", "com.apple.Notes",
        "com.agiletortoise.Drafts", "com.evernote", "com.roamresearch",
    )),
    ("browser", (
        "com.google.Chrome", "com.apple.Safari", "company.thebrowser.Browser",
        "org.mozilla.firefox", "com.microsoft.edgemac", "com.brave.Browser",
    )),
]

# Categories where a trailing full stop reads as stiff or passive-aggressive.
CHATTY = {"chat"}


@dataclass
class AppContext:
    """Everything we know about where the text is going."""
    name: str = ""
    bundle_id: str = ""
    pid: int = 0
    category: str = "other"
    text_before: str = ""
    selection: str = ""
    has_selection: bool = False
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def is_chat(self) -> bool:
        return self.category in CHATTY

    @property
    def mid_sentence(self) -> bool:
        """True when the cursor sits inside an unfinished sentence.

        Used to avoid capitalising an insertion that continues a line you
        already started typing. Conservative: an empty or unreadable field
        counts as the start of a sentence.
        """
        tail = self.text_before.rstrip()
        if not tail:
            return False
        return tail[-1] not in ".!?:\n"


def _categorise(bundle_id: str) -> str:
    bid = (bundle_id or "").lower()
    for category, prefixes in _CATEGORIES:
        for prefix in prefixes:
            if bid.startswith(prefix.lower()):
                return category
    return "other"


def frontmost() -> AppContext:
    """The app currently receiving keystrokes. Cheap — ~12µs measured."""
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return AppContext()
        bundle_id = app.bundleIdentifier() or ""
        return AppContext(
            name=app.localizedName() or "",
            bundle_id=bundle_id,
            pid=int(app.processIdentifier()),
            category=_categorise(bundle_id),
        )
    except Exception:
        return AppContext()


def read_focused_text(ctx: AppContext, *, max_chars: int = 400) -> AppContext:
    """Fill in `text_before` and `selection` from the Accessibility tree.

    Best-effort by design. Many apps (anything Electron without accessibility
    enabled, most canvas-based editors) expose nothing useful, and a password
    field exposes nothing at all — which is correct and we do not work around
    it. On any failure the fields stay empty and the caller falls back to
    context-free formatting.
    """
    try:
        from ApplicationServices import (
            AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
            kAXFocusedUIElementAttribute, kAXValueAttribute,
            kAXSelectedTextAttribute, kAXSelectedTextRangeAttribute,
            kAXSubroleAttribute,
        )
    except Exception:
        return ctx
    if not ctx.pid:
        return ctx
    try:
        app_el = AXUIElementCreateApplication(ctx.pid)
        err, focused = AXUIElementCopyAttributeValue(
            app_el, kAXFocusedUIElementAttribute, None)
        if err != 0 or focused is None:
            return ctx

        # Never read a secure text field. macOS reports these as
        # AXSecureTextField; reading them would be a straightforward way to
        # end up with a password in a transcript log.
        err, subrole = AXUIElementCopyAttributeValue(focused, kAXSubroleAttribute, None)
        if err == 0 and subrole and "Secure" in str(subrole):
            return ctx

        err, selection = AXUIElementCopyAttributeValue(
            focused, kAXSelectedTextAttribute, None)
        if err == 0 and selection:
            ctx.selection = str(selection)
            ctx.has_selection = bool(ctx.selection.strip())

        err, value = AXUIElementCopyAttributeValue(focused, kAXValueAttribute, None)
        if err == 0 and value is not None:
            full = str(value)
            err_r, rng = AXUIElementCopyAttributeValue(
                focused, kAXSelectedTextRangeAttribute, None)
            cursor = len(full)
            if err_r == 0 and rng is not None:
                # AXValue CFRange — location is where the cursor sits.
                try:
                    from ApplicationServices import AXValueGetValue, kAXValueCFRangeType
                    ok, parsed = AXValueGetValue(rng, kAXValueCFRangeType, None)
                    if ok and parsed is not None:
                        cursor = int(parsed.location)
                except Exception:
                    pass
            ctx.text_before = full[:cursor][-max_chars:]
    except Exception:
        pass
    return ctx


def capture(read_text: bool = True) -> AppContext:
    """Full context capture, as taken at the moment recording starts."""
    ctx = frontmost()
    if read_text:
        ctx = read_focused_text(ctx)
    return ctx


def activate(ctx: AppContext) -> bool:
    """Bring `ctx`'s app back to the front before pasting.

    Transcription is asynchronous, so between releasing the key and the text
    being ready you may well have switched windows — and the paste goes to
    whatever is frontmost at that instant, not to where you were dictating.
    Returns True if the right app is now frontmost.
    """
    if not ctx.pid:
        return False
    current = frontmost()
    if current.pid == ctx.pid:
        return True
    try:
        from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(ctx.pid)
        if app is None:
            return False
        app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        # Activation is asynchronous; give it a moment and confirm rather than
        # assuming it worked.
        for _ in range(20):
            time.sleep(0.02)
            if frontmost().pid == ctx.pid:
                return True
        return False
    except Exception:
        return False
