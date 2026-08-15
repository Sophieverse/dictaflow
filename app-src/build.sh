#!/bin/sh
# Build DictaFlow.app — an optional wrapper that gives the LaunchAgent a
# stable TCC identity.
#
# Only needed if the plain LaunchAgent cannot get a Microphone grant. A bare
# launchd job is its own responsible process with no bundle identifier, so
# macOS has nothing to attach the grant to; the input stream then either
# returns digital zeros or blocks inside CoreAudio, in both cases with no
# error. If you switch to this, you must re-grant BOTH Microphone and
# Accessibility to "DictaFlow" — until you do, the agent cannot even see
# keypresses. See com.dictaflow.agent.plist.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
APP="$HERE/DictaFlow.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$HERE/app-src/Info.plist" "$APP/Contents/Info.plist"
swiftc -O -o "$APP/Contents/MacOS/DictaFlow" "$HERE/app-src/main.swift"
# Ad-hoc signature is enough for TCC to have something stable to key on.
# Note: after re-signing, `launchctl kickstart -k` silently fails to pick up
# the new binary — you must bootout and bootstrap.
codesign --force --deep --sign - "$APP"
codesign -dv "$APP" 2>&1 | head -3
echo "built $APP"
