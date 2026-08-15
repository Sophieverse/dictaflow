"""DictaFlow — local, open-weight voice dictation for macOS.

Split into modules because the single-file version had grown a threading model
it could no longer document: pynput callbacks, a PortAudio callback, worker
threads and an AppKit main loop all mutating the same flat attributes. The
boundaries here are drawn along thread ownership, not along topic:

    config    pure data, no threads
    audio     PortAudio callback thread owns the ring buffer
    asr       worker threads only; never touches UI state
    textproc  pure functions, no I/O at all
    appctx    main/worker; cheap Cocoa reads
    inject    worker thread; owns the clipboard
    store     append-only, any thread
    hud       MAIN THREAD ONLY for AppKit calls; others post intents
    session   the state machine that ties them together, with one lock
"""

__version__ = "2.0.0"
