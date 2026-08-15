#!/usr/bin/env python3
"""DictaFlow — local, open-weight voice dictation for macOS.

Hold a key, talk, release. The audio is transcribed on your Mac and inserted
wherever you were typing. Nothing leaves the machine.

    hold right-⌥      dictate with Turbo (large-v3-turbo, most accurate)
    hold right-⌘      dictate with Small (fastest)
    double-tap either hands-free — keeps recording; tap again to stop
    hold right-⌃      command mode — rewrites the selected text
    Esc               cancel whatever is recording

Run `python dictaflow.py --check` for a preflight report, or `--doctor` for a
deeper diagnosis when something isn't working.
"""
from __future__ import annotations

import argparse
import atexit
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from df import asr, config, hud, hotkeys, store          # noqa: E402
from df.audio import Recorder, list_input_devices         # noqa: E402
from df.session import Session                            # noqa: E402


BANNER = "DictaFlow"


LOCK_FILE = config.CONFIG_DIR / "dictaflow.lock"
_lock_handle = None


def acquire_single_instance() -> bool:
    """Refuse to start if another DictaFlow is already running.

    Two instances fight over the microphone: the loser's CoreAudio open can
    block indefinitely, so the second copy starts, prints nothing, and looks
    dead. That happened here with an instance orphaned by a launchd restart.
    An flock is released automatically when the process dies, however it dies,
    which a pidfile alone would not guarantee.
    """
    global _lock_handle
    import fcntl
    try:
        config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        handle = open(LOCK_FILE, "w")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(str(os.getpid()))
        handle.flush()
        _lock_handle = handle          # held for the process lifetime
        return True
    except BlockingIOError:
        return False
    except Exception:
        return True                    # never let the guard itself block startup


def log(message: str) -> None:
    """Single-line status output. `\\r` + clear-to-EOL keeps the transient
    'Recording…' line from leaving debris behind the final result."""
    print(f"\r{message}\033[K", flush=True)


# ──────────────────────────────────────────────────────────────
# Preflight
# ──────────────────────────────────────────────────────────────
def check_models(cfg: dict) -> list[str]:
    problems = []
    for name, path in cfg["models"].items():
        if "/" in path and not Path(path).is_absolute():
            continue                    # a Hugging Face repo id; fetched lazily
        if not Path(path).exists():
            problems.append(
                f"model {name!r} is missing at {path}\n"
                f"    fetch it with:  .venv/bin/huggingface-cli download "
                f"mlx-community/{Path(path).name} --local-dir {path}")
    return problems


def check_permissions() -> list[str]:
    problems = []
    try:
        from ApplicationServices import AXIsProcessTrusted
        if not AXIsProcessTrusted():
            problems.append(
                "Accessibility permission is not granted, so DictaFlow cannot "
                "paste.\n    System Settings → Privacy & Security → "
                f"Accessibility → add {sys.executable}")
    except Exception:
        problems.append("could not query Accessibility permission")
    return problems


def check_audio() -> list[str]:
    try:
        devices = list_input_devices()
    except Exception as exc:
        return [f"no audio system available: {exc}"]
    if not devices:
        return ["no input devices found"]
    return []


def preflight(cfg: dict, *, verbose: bool = True) -> list[str]:
    problems = check_models(cfg) + check_permissions() + check_audio()
    if verbose:
        if problems:
            print("\n⚠  Preflight found problems:\n")
            for p in problems:
                print(f"  • {p}")
            print()
        else:
            print("✓  Preflight OK — models, permissions and audio all good.\n")
    return problems


def setup(cfg: dict) -> int:
    """Guided first-run: permissions, mic check, a practice dictation.

    Deliberately a terminal wizard rather than a GUI, and deliberately not
    reachable from the LaunchAgent — stdin is /dev/null under launchd, so an
    input() prompt there would be an EOFError in a KeepAlive restart loop.
    `main()` checks for a tty before ever calling this.
    """
    import numpy as np
    from df.audio import has_speech, spectral_flatness

    print(f"\n─── {BANNER} setup ─────────────────────────────\n")
    print("  Everything runs on this Mac. No account, no API key, no audio")
    print("  leaves the machine.\n")

    print("  1. Permissions")
    problems = check_permissions()
    if problems:
        for p in problems:
            print(f"     ✗ {p}")
        print("\n     Grant it, then run setup again.")
    else:
        print("     ✓ Accessibility granted")

    print("\n  2. Models")
    model_problems = check_models(cfg)
    if model_problems:
        for p in model_problems:
            print(f"     ✗ {p}")
        return 1
    print("     ✓ both models present")

    print("\n  3. Microphone")
    for d in list_input_devices():
        print(f"     [{d['index']}] {d['name']}")
    input("\n     Press Return, then speak normally for 3 seconds… ")
    rec = Recorder(preroll_ms=0)
    ok, detail = rec.open_with_timeout(8.0)
    if not ok:
        print(f"     \u2717 {detail}")
        return 1
    try:
        rec.start()
        time.sleep(3.0)
        samples = rec.stop()
    finally:
        rec.close()
    peak = int(np.abs(samples).max(initial=0)) if samples.size else 0
    flat = spectral_flatness(samples) if samples.size else 1.0
    print(f"     captured {samples.size / config.SAMPLE_RATE:.1f}s, "
          f"peak {peak}, flatness {flat:.3f}")
    if peak == 0:
        print("     ✗ pure silence — microphone permission is denied.")
        print("       System Settings → Privacy & Security → Microphone")
        return 1
    if not has_speech(samples):
        print("     ✗ that didn't register as speech. Try again closer to the mic.")
    else:
        print("     ✓ speech detected")

    print("\n  4. Now try whispering, 3 seconds…")
    input("     Press Return, then whisper… ")
    ok, detail = rec.open_with_timeout(8.0)
    if not ok:
        print(f"     \u2717 {detail}")
        return 1
    try:
        rec.start()
        time.sleep(3.0)
        whisper_samples = rec.stop()
    finally:
        rec.close()
    wflat = spectral_flatness(whisper_samples) if whisper_samples.size else 1.0
    wpeak = int(np.abs(whisper_samples).max(initial=0)) if whisper_samples.size else 0
    print(f"     peak {wpeak}, flatness {wflat:.3f} "
          f"(threshold {0.15}; lower is more speech-like)")
    if has_speech(whisper_samples):
        print("     ✓ whispering registers — the gate measures spectral shape,")
        print("       not loudness, so a whisper is as valid as a shout.")
    else:
        print("     ✗ that whisper didn't register. Get closer to the mic —")
        print("       roughly a hand's width. A headset mic helps a lot.")

    print("\n  5. Transcribing what you just said…")
    try:
        transcriber = asr.Transcriber(cfg["models"]["turbo"], cfg)
        result = transcriber.run(samples)
        if result["text"]:
            print(f"     ✓ “{result['text']}”")
        else:
            print(f"     ✗ discarded: {result['rejected']}")
    except Exception as exc:
        print(f"     ✗ transcription failed: {exc}")
        return 1

    cfg["onboarded"] = True
    try:
        config.save(cfg)
    except Exception as exc:
        print(f"\n  ⚠ could not save config: {exc}")

    bindings = cfg.get("bindings") or DEFAULT_BINDINGS
    print("\n  You're set up.\n")
    for slot, key in bindings.items():
        verb = "command mode" if slot == "command" else f"dictate ({slot})"
        print(f"     hold {hotkeys.label(key):<10} → {verb}")
    print(f"     {'Esc':<15} → cancel")
    print("\n  Settings and history: http://localhost:7755")
    print("──────────────────────────────────────────────────\n")
    return 0


def doctor(cfg: dict) -> int:
    """A deeper report, for when dictation isn't working and it isn't obvious."""
    print(f"\n─── {BANNER} doctor ───────────────────────────────\n")
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")
    print(f"  config        {config.CONFIG_FILE}"
          f"{'  ⚠ ' + cfg['_error'] if cfg.get('_error') else ''}")
    print(f"  history       {store.HISTORY_FILE}")

    print("\n  models:")
    for name, path in cfg["models"].items():
        exists = Path(path).exists() if Path(path).is_absolute() else None
        mark = "✓" if exists else ("?" if exists is None else "✗")
        print(f"    {mark} {name:<6} {path}")

    print("\n  input devices:")
    try:
        for d in list_input_devices():
            print(f"    [{d['index']}] {d['name']}")
    except Exception as exc:
        print(f"    ✗ {exc}")

    print("\n  permissions:")
    try:
        from ApplicationServices import AXIsProcessTrusted
        print(f"    {'✓' if AXIsProcessTrusted() else '✗'} Accessibility")
    except Exception as exc:
        print(f"    ? Accessibility ({exc})")

    print("\n  live mic test (2s) — say something:")
    try:
        import numpy as np
        from df.audio import has_speech, spectral_flatness
        rec = Recorder(preroll_ms=0)
        ok, detail = rec.open_with_timeout(8.0)
        if not ok:
            print(f"    ✗ {detail}")
            raise RuntimeError("microphone unavailable")
        rec.start()
        time.sleep(2.0)
        samples = rec.stop()
        rec.close()
        peak = int(np.abs(samples).max(initial=0)) if samples.size else 0
        flat = spectral_flatness(samples) if samples.size else 1.0
        print(f"    captured {samples.size / config.SAMPLE_RATE:.1f}s, "
              f"peak {peak}, flatness {flat:.3f}")
        if peak == 0:
            print("    ✗ pure digital silence — microphone permission is "
                  "probably denied.")
        elif has_speech(samples):
            print("    ✓ speech detected")
        else:
            print("    ✗ no speech detected (flatness above "
                  f"{0.15}) — this clip would be discarded")
    except Exception as exc:
        print(f"    ✗ mic test failed: {exc}")

    print("\n  command mode:")
    try:
        from df import command
        ok, why = command.available(cfg)
        print(f"    {'✓ ready' if ok else '✗ ' + why}")
    except Exception as exc:
        print(f"    ✗ {exc}")

    entries = store.load()
    ok = [e for e in entries if e.get("outcome") == "ok"]
    rejected = [e for e in entries if e.get("outcome") == "rejected"]
    print(f"\n  history: {len(entries)} entries, {len(ok)} ok, "
          f"{len(rejected)} rejected")
    if rejected:
        print("  most recent rejections:")
        for e in rejected[:5]:
            print(f"    {e.get('ts', '')}  {e.get('rejected', '')}")
    print("\n──────────────────────────────────────────────────\n")
    return 0


# ──────────────────────────────────────────────────────────────
# The app
# ──────────────────────────────────────────────────────────────
class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bar = hud.FlowBar(cfg)
        self.recorder = Recorder(
            device=cfg.get("input_device"),
            preroll_ms=int(cfg.get("preroll_ms", 500)),
            max_seconds=int(cfg.get("max_session_seconds", 1200)),
        )
        self.model_paths = {
            "turbo": cfg["models"]["turbo"],
            "small": cfg["models"]["small"],
            "command": cfg["models"]["turbo"],
        }
        self.session = Session(cfg, bar=self.bar, recorder=self.recorder,
                               model_paths=self.model_paths, log=log)
        self.router = None
        self._ready = threading.Event()
        self._stopping = threading.Event()

    # ── warmup ──────────────────────────────────────────────────
    def _open_mic(self) -> None:
        """Open the input stream — never fatally, never indefinitely."""
        ok, detail = self.recorder.open_with_timeout(8.0)
        if not ok:
            log(f"⚠  microphone: {detail}")
            log("   Dictation will not capture audio until this is resolved.")
            log("   Run `dictaflow.py --doctor` for a live check.")

    def _warmup(self) -> None:
        """Load both models and open the mic before the first dictation.

        Models first, deliberately: opening a CoreAudio device can block for
        a long time or forever, and when that ran first it starved model
        loading, so `_ready` never got set and every keypress was answered
        with "still loading models". The models are the thing dictation
        cannot proceed without.

        Failures are printed. The previous version swallowed them, so a
        missing or corrupt model was invisible until the first real dictation
        — at which point the audio was already gone.
        """
        for name in ("turbo", "small"):
            path = self.model_paths[name]
            try:
                started = time.monotonic()
                asr.warm(path)
                log(f"✓  {name} ready ({time.monotonic() - started:.1f}s)")
            except Exception as exc:
                log(f"✗  {name} FAILED to load: {exc}")
                log(f"   dictation with this model will not work. Path: {path}")
        self._ready.set()
        self._open_mic()
        self._check_mic_is_live()

    def _check_mic_is_live(self) -> None:
        """Confirm the mic is delivering signal, not just digital zeros.

        macOS does not raise when microphone permission is denied — it hands
        you a stream of silence. Without this the app looks perfectly healthy
        and every dictation is quietly discarded as "no speech", which is
        exactly what the old logs were full of. Better to say so at startup.
        """
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.recorder.level > 0:
                return
            time.sleep(0.2)
        if self.recorder.looks_muted():
            log("⚠  the microphone is returning pure silence. Dictation will")
            log("   be discarded until this is fixed: System Settings →")
            log(f"   Privacy & Security → Microphone → enable {sys.executable}")
        # A genuinely silent room also reads as level 0, so anything short of
        # looks_muted() (which needs an all-zero history) is not worth a
        # warning — a false alarm here would train you to ignore real ones.

    # ── key handlers ────────────────────────────────────────────
    def on_hold_start(self, slot: str) -> None:
        if not self._ready.is_set():
            log("⏳  still loading models — one moment…")
            return
        log(f"🎙  Recording… [{slot}]")
        self.session.begin(slot)

    def on_hold_end(self, slot: str) -> None:
        self.session.end()

    def on_handsfree_on(self, slot: str) -> None:
        if not self._ready.is_set():
            log("⏳  still loading models — one moment…")
            return
        limit = int(self.cfg.get("max_session_seconds", 1200))
        log(f"🎙  Hands-free [{slot}] — tap {hotkeys.label(self._binding(slot))} "
            f"again to stop (auto-stops after {limit // 60} min)")
        self.session.begin(slot, handsfree=True)

    def on_handsfree_off(self, slot: str) -> None:
        self.session.end()

    def on_command_start(self) -> None:
        if not self._ready.is_set():
            return
        log("🎙  Command mode — say what to do with the selected text")
        self.session.begin("command", mode="command")

    def on_command_end(self) -> None:
        self.session.end()

    def on_cancel(self) -> None:
        self.session.cancel()

    def on_tap_discarded(self, slot: str) -> None:
        self.session.discard_tap(slot)

    def on_paste_last(self) -> None:
        """⌘⌃V — re-insert the last transcript.

        The recovery path when a paste didn't land: the text is already in
        history, so this doesn't need the audio or the model.
        """
        text = self.session.last_text or self._last_from_history()
        if not text:
            log("✗  nothing to paste — no transcripts yet.")
            return
        from df import inject
        result = inject.insert(text, self.cfg)
        if result.ok:
            log(f"✓  re-inserted: {text[:60]}…")
        else:
            log(f"✗  {result.detail}")

    def on_copy_last(self) -> None:
        """⌘⌃C — put the last transcript on the clipboard without pasting."""
        text = self.session.last_text or self._last_from_history()
        if not text:
            log("✗  nothing to copy — no transcripts yet.")
            return
        from df import inject
        if inject.set_clipboard(text):
            log(f"✓  copied to clipboard: {text[:60]}…")
        else:
            log("✗  could not write to the clipboard.")

    def _last_from_history(self) -> str:
        """Survives a restart — session.last_text is only this process's."""
        for entry in store.load():
            if entry.get("outcome") == "ok" and entry.get("text"):
                return entry["text"]
        return ""

    def _binding(self, slot: str) -> str:
        for name, key in self.cfg.get("bindings", DEFAULT_BINDINGS).items():
            if name == slot:
                return key
        return slot

    # ── run ─────────────────────────────────────────────────────
    def run(self) -> int:
        bindings = self.cfg.get("bindings") or DEFAULT_BINDINGS
        self.router = hotkeys.KeyRouter(bindings, {
            "on_hold_start":    self.on_hold_start,
            "on_hold_end":      self.on_hold_end,
            "on_handsfree_on":  self.on_handsfree_on,
            "on_handsfree_off": self.on_handsfree_off,
            "on_command_start": self.on_command_start,
            "on_command_end":   self.on_command_end,
            "on_cancel":        self.on_cancel,
            "on_tap_discarded": self.on_tap_discarded,
            "on_paste_last":    self.on_paste_last,
            "on_copy_last":     self.on_copy_last,
        })

        # The help text prints BEFORE the microphone is touched. Opening a
        # CoreAudio input device can block indefinitely when another process
        # holds it — observed here with an orphaned instance still attached —
        # and doing it inline meant the app started, printed nothing, and
        # looked dead. The open now happens on the warmup thread.
        self._print_help(bindings)
        self.bar.set_menu([
            (f"{BANNER} — ready", None, False),
            ("-", None, False),
            ("Open dashboard", self._open_dashboard, True),
            ("Quit", self._quit, True),
        ])

        threading.Thread(target=self._warmup, daemon=True, name="df-warm").start()
        self.router.start()
        atexit.register(self.shutdown)

        try:
            while not self._stopping.is_set():
                if not self.router.running:
                    log("✗  the keyboard listener stopped — Input Monitoring "
                        "permission may have been revoked. Exiting so launchd "
                        "can restart me.")
                    return 3
                self.session.tick()
                self.bar.pump()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    def _print_help(self, bindings: dict) -> None:
        print(f"\n{BANNER} is running.\n")
        for slot, key in bindings.items():
            if slot == "command":
                print(f"  hold {hotkeys.label(key):<10} → command mode "
                      f"(rewrite the selection)")
            else:
                print(f"  hold {hotkeys.label(key):<10} → dictate with {slot}")
                print(f"  double-tap {hotkeys.label(key):<4} → hands-free {slot}")
        print(f"  {'Esc':<15} → cancel")
        print(f"\n  history   {store.HISTORY_FILE}")
        print(f"  dashboard http://localhost:7755")
        print("  Ctrl+C to quit.\n")

    def _open_dashboard(self) -> None:
        import subprocess
        subprocess.run(["open", "http://localhost:7755"], capture_output=True)

    def _quit(self) -> None:
        self._stopping.set()

    def shutdown(self) -> None:
        if self._stopping.is_set() and self.router is None:
            return
        self._stopping.set()
        try:
            if self.router is not None:
                self.router.stop()
        except Exception:
            pass
        try:
            self.recorder.close()
        except Exception:
            pass


DEFAULT_BINDINGS = {"turbo": "alt_r", "small": "cmd_r", "command": "ctrl_r"}


def main() -> int:
    parser = argparse.ArgumentParser(description=BANNER)
    parser.add_argument("--check", action="store_true",
                        help="run preflight checks and exit")
    parser.add_argument("--doctor", action="store_true",
                        help="detailed diagnosis, including a live mic test")
    parser.add_argument("--compact", action="store_true",
                        help="compact the history file and exit")
    parser.add_argument("--setup", action="store_true",
                        help="guided first-run setup with a mic and whisper test")
    args = parser.parse_args()

    cfg = config.load()
    cfg.setdefault("bindings", DEFAULT_BINDINGS)

    if args.compact:
        print(f"compacted to {store.compact()} entries")
        return 0
    if args.setup:
        return setup(cfg)
    if args.doctor:
        return doctor(cfg)
    if args.check:
        return 1 if preflight(cfg) else 0

    # First run: point at the dashboard rather than prompting. An input()
    # here would be an EOFError under launchd, where stdin is /dev/null —
    # and with KeepAlive that becomes a silent restart loop.
    if not cfg.get("onboarded"):
        if sys.stdin.isatty():
            log("First run — run `dictaflow.py --setup` for a guided check, "
                "or just hold right-⌥ and talk.")
        cfg["onboarded"] = True
        try:
            config.save(cfg)
        except Exception:
            pass

    if cfg.get("_error"):
        log(f"⚠  {cfg['_error']} — running with defaults.")

    if not acquire_single_instance():
        log("✗  another DictaFlow is already running; exiting so the two "
            "don't fight over the microphone.")
        log(f"   (lock: {LOCK_FILE})")
        return 0                       # 0, so launchd's KeepAlive doesn't loop

    imported = store.migrate_legacy()
    if imported:
        log(f"✓  imported {imported} entries from the previous format.")
    store.maybe_compact()

    problems = preflight(cfg, verbose=False)
    for p in problems:
        log(f"⚠  {p}")

    # Exit cleanly on SIGTERM so launchd's stop is not a kill.
    app = App(cfg)
    signal.signal(signal.SIGTERM, lambda *_: app._quit())
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
