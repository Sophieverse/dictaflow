#!/usr/bin/env python3
"""Diagnostic: record 3s from the mic and report whether real audio came through.
Run in your Terminal:  .venv/bin/python check_audio.py
"""
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
SECONDS = 3

print("Default input device:", sd.query_devices(kind="input")["name"])
print(f"\nRecording {SECONDS}s — say something now…")
audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
               channels=1, dtype="int16")
sd.wait()
print("…done.\n")

peak = int(np.abs(audio).max())
rms  = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
print(f"Peak amplitude: {peak:>6}  (int16 max is 32767)")
print(f"RMS level:      {rms:>8.1f}")

if peak < 50:
    print("\n✗ SILENT — the app is not getting microphone audio.")
    print("  Fix: System Settings → Privacy & Security → Microphone →")
    print("       enable your terminal app, then quit & reopen it.")
else:
    print("\n✓ Audio captured fine — the mic is working.")
