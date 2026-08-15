// DictaFlow.app launcher.
//
// This bundle exists for exactly one reason: to give the agent a stable TCC
// identity. A bare launchd job is its own responsible process and carries no
// bundle identifier, so macOS has nothing to attach a Microphone grant to —
// it opens the input stream successfully and then hands back pure digital
// zeros, forever, with no error anywhere. Downstream that reads as "no speech
// detected" on every dictation, which is exactly what the old logs were full
// of. Verified: a probe run as a launchd job captured 49,152 samples with a
// peak of 0 and not one non-zero value.
//
// It must be a compiled Mach-O, not a shell script: a script that `exec`s
// python replaces the process image and the bundle identity goes with it.
// Here python is spawned as a CHILD, so it inherits this signed bundle as its
// responsible process and the grants apply to it.

import Foundation
import AVFoundation
import ApplicationServices

let venvPython = "/Users/melod/dictaflow/.venv/bin/python"
let script     = "/Users/melod/dictaflow/dictaflow.py"

func warn(_ message: String) {
    FileHandle.standardError.write(Data("DictaFlow: \(message)\n".utf8))
}

// Ask for the microphone. Deliberately NON-blocking: the callback fires when
// the user answers, which may be minutes away, and blocking startup on it
// would leave the agent dead in the meantime. Requesting is what makes the
// prompt appear at all — a background job can otherwise never ask, so you end
// up denied without having been asked.
switch AVCaptureDevice.authorizationStatus(for: .audio) {
case .authorized:
    break
case .notDetermined:
    AVCaptureDevice.requestAccess(for: .audio) { ok in
        if !ok { warn("microphone access was declined; dictation cannot hear you.") }
    }
default:
    warn("microphone access is denied. System Settings → Privacy & Security "
         + "→ Microphone → enable DictaFlow, then restart the agent with: "
         + "launchctl kickstart -k gui/$(id -u)/com.dictaflow.agent")
}

// Accessibility, needed to paste. `kAXTrustedCheckOptionPrompt` opens the
// pane rather than failing mutely.
let axOptions = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true]
if !AXIsProcessTrustedWithOptions(axOptions as CFDictionary) {
    warn("Accessibility is not granted yet; DictaFlow cannot paste until it is. "
         + "System Settings → Privacy & Security → Accessibility → enable DictaFlow.")
}

let task = Process()
task.executableURL = URL(fileURLWithPath: venvPython)
task.arguments = ["-u", script] + Array(CommandLine.arguments.dropFirst())
task.currentDirectoryURL = URL(fileURLWithPath: "/Users/melod/dictaflow")

// Forward SIGTERM so `launchctl bootout` stops the agent cleanly instead of
// orphaning a python that still holds the microphone.
let term = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
term.setEventHandler { task.terminate(); exit(0) }
term.resume()
signal(SIGTERM, SIG_IGN)

do {
    try task.run()
} catch {
    warn("could not start \(venvPython): \(error)")
    exit(1)
}
task.waitUntilExit()
exit(task.terminationStatus)
