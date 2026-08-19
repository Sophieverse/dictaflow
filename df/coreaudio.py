"""Direct CoreAudio access, for the one thing PortAudio cannot do: reset the device.

WHY THIS EXISTS. The built-in microphone gets into a state where it is, by
every observable measure, working: the device is alive and unmuted, its input
volume is normal, blocks arrive at exactly the expected rate, no error is
raised anywhere, and `AudioObjectGetPropertyData` reports the device running.
Every sample in every block is nevertheless exactly zero, for every process on
the machine — ffmpeg included. Nothing in the API surfaces it, so an app that
trusts its own stream handle reports itself healthy while recording silence.

The documented remedy is `sudo killall coreaudiod`, which is useless inside an
app: it needs a password, it kills audio for everything, and it is a thing a
user has to be told to do — which in practice means the app stays broken until
someone reads a log. Measured here on macOS 26.5.1, a much smaller lever does
the same job from user space: setting the device's nominal sample rate to a
different value and back forces the HAL to tear down and rebuild its IO
context. Immediately afterwards the microphone delivers real audio again.

    before flip: ffmpeg peak = 0     (digital silence, ~30 minutes of it)
    after  flip: ffmpeg peak = 1059  (room tone)

Kept in its own module because ctypes against a C API is worth quarantining;
everything here is best-effort and returns rather than raises.
"""
from __future__ import annotations

import ctypes
import struct
import time

_ca = None


def _lib():
    global _ca
    if _ca is None:
        _ca = ctypes.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
    return _ca


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32),
                ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


_GLOBAL = _fourcc("glob")
_SYSTEM = 1                       # kAudioObjectSystemObject


def _get(obj: int, selector: str, ctype):
    addr = _Address(_fourcc(selector), _GLOBAL, 0)
    size = ctypes.c_uint32(ctypes.sizeof(ctype))
    value = ctype()
    status = _lib().AudioObjectGetPropertyData(
        ctypes.c_uint32(obj), ctypes.byref(addr), 0, None,
        ctypes.byref(size), ctypes.byref(value))
    return status, value.value


def _set(obj: int, selector: str, ctype, value) -> int:
    addr = _Address(_fourcc(selector), _GLOBAL, 0)
    boxed = ctype(value)
    return _lib().AudioObjectSetPropertyData(
        ctypes.c_uint32(obj), ctypes.byref(addr), 0, None,
        ctypes.c_uint32(ctypes.sizeof(ctype)), ctypes.byref(boxed))


def default_input_device() -> int | None:
    """kAudioHardwarePropertyDefaultInputDevice, or None."""
    try:
        status, dev = _get(_SYSTEM, "dIn ", ctypes.c_uint32)
    except Exception:
        return None
    return None if status or not dev else dev


def input_is_muted() -> bool | None:
    """True/False from the HAL, or None when the device has no mute control.

    Worth checking before a reset: if she muted the microphone on purpose,
    resetting the device is both futile and rude.
    """
    dev = default_input_device()
    if dev is None:
        return None
    try:
        addr = _Address(_fourcc("mute"), _fourcc("inpt"), 0)
        size = ctypes.c_uint32(4)
        value = ctypes.c_uint32()
        status = _lib().AudioObjectGetPropertyData(
            ctypes.c_uint32(dev), ctypes.byref(addr), 0, None,
            ctypes.byref(size), ctypes.byref(value))
        return None if status else bool(value.value)
    except Exception:
        return None


def reset_input_device(settle: float = 0.6) -> tuple[bool, str]:
    """Force the HAL to rebuild the input device's IO context.

    Flips the nominal sample rate away and back. Returns (ok, detail).
    Restoring the original rate is not optional and not best-effort: leaving
    her microphone at 44.1kHz to save a line of code would degrade every
    other app on the machine.
    """
    dev = default_input_device()
    if dev is None:
        return False, "no default input device"
    try:
        status, original = _get(dev, "nsrt", ctypes.c_double)
        if status or not original:
            return False, f"could not read sample rate (status {status})"
        # Any rate the device supports will do; these two cover every Mac
        # built-in microphone, and we only need it to differ from `original`.
        other = 44100.0 if abs(original - 44100.0) > 1 else 48000.0
        if _set(dev, "nsrt", ctypes.c_double, other):
            return False, "device refused the sample-rate change"
        time.sleep(settle)
        restore = _set(dev, "nsrt", ctypes.c_double, original)
        time.sleep(settle)
        if restore:
            return False, (f"could not restore {original:.0f}Hz — your input "
                           f"device may be left at {other:.0f}Hz")
        return True, f"reset the input device ({original:.0f}Hz)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
