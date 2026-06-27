#!/usr/bin/env python3
"""
DictaFlow Transcripts — a native Dock app that shows your transcript dashboard
in its own window (a WKWebView pointed at the local dashboard server on :7755).

This is a regular windowed app, so it gets a Dock icon and a normal window —
unlike a menu-bar status item, this code path renders reliably on macOS.
"""
import urllib.request

import Cocoa
import WebKit
from Foundation import NSURL, NSURLRequest
from PyObjCTools import AppHelper

DASHBOARD_URL = "http://localhost:7755/"


def _server_up() -> bool:
    try:
        urllib.request.urlopen(DASHBOARD_URL, timeout=1)
        return True
    except Exception:
        return False


class AppDelegate(Cocoa.NSObject):
    def applicationDidFinishLaunching_(self, notification):
        rect = Cocoa.NSMakeRect(0, 0, 900, 720)
        style = (Cocoa.NSWindowStyleMaskTitled
                 | Cocoa.NSWindowStyleMaskClosable
                 | Cocoa.NSWindowStyleMaskMiniaturizable
                 | Cocoa.NSWindowStyleMaskResizable)
        self.window = Cocoa.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, Cocoa.NSBackingStoreBuffered, False)
        self.window.setTitle_("DictaFlow Transcripts")
        self.window.center()
        self.window.setMinSize_(Cocoa.NSMakeSize(420, 400))

        self.web = WebKit.WKWebView.alloc().initWithFrame_(rect)
        self.web.setAutoresizingMask_(
            Cocoa.NSViewWidthSizable | Cocoa.NSViewHeightSizable)
        self.window.contentView().addSubview_(self.web)
        self._load()

        self.window.makeKeyAndOrderFront_(None)
        Cocoa.NSApp.activateIgnoringOtherApps_(True)

    def _load(self):
        if _server_up():
            url = NSURL.URLWithString_(DASHBOARD_URL)
            self.web.loadRequest_(NSURLRequest.requestWithURL_(url))
        else:
            html = ("<body style='background:#0d0f12;color:#f4f6f8;"
                    "font:16px -apple-system;text-align:center;padding-top:120px'>"
                    "<h2>Dashboard server not running</h2>"
                    "<p>Start it with:<br><code>launchctl bootstrap gui/$(id -u) "
                    "~/Library/LaunchAgents/com.dictaflow.dashboard.plist</code></p></body>")
            self.web.loadHTMLString_baseURL_(html, None)

    # reload when the window is re-focused, so new dictations show up
    def applicationDidBecomeActive_(self, notification):
        if hasattr(self, "web"):
            self.web.reload_(None)

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True


if __name__ == "__main__":
    app = Cocoa.NSApplication.sharedApplication()
    app.setActivationPolicy_(Cocoa.NSApplicationActivationPolicyRegular)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    globals()["_delegate"] = delegate   # keep a ref so it isn't GC'd
    AppHelper.runEventLoop()
