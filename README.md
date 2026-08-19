# DictaFlow

Hold-to-talk voice dictation for macOS that runs **entirely on your own
machine**. Hold a key, talk, release — the text is transcribed locally and
inserted wherever you were typing. No account, no API key, no audio leaving
your Mac.

An open-weight alternative to [Wispr Flow](https://wisprflow.ai) /
Superwhisper, built on
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
(Apple Silicon / Metal).

---

## Using it

| Gesture | What happens |
| --- | --- |
| **Hold right-⌥** | Records while held → transcribes with **large-v3-turbo** (most accurate) |
| **Hold right-⌘** | Same, with **small** (fastest) |
| **Double-tap** either | Hands-free — keeps recording without holding. Tap again to stop. |
| **Hold right-⌃** | **Command mode** — speak an instruction to rewrite the selected text |
| **Esc** | Cancel — discards the recording, transcribes nothing |

A floating pill (the **Flow Bar**) appears whenever DictaFlow is listening,
showing a live waveform and a running word count. Red = recording, amber =
transcribing, green = inserted, purple = something needs your attention. No
pill means the mic is not hot. There's also a menu bar item showing the same
state.

Recording begins slightly *before* you press the key. The mic runs
continuously and keeps a rolling half-second buffer, so the first syllable
isn't clipped — "missing first words" is a known issue in Wispr Flow itself,
and it's caused by starting the mic on keydown.

Hands-free sessions stop automatically after 20 minutes, with a warning at 19.

---

## What it does to your words

Transcription is only half of it. The raw Whisper output goes through a
**rule-based** editing pass — no LLM, so it is instant and, more importantly,
predictable. (Over-editing is the single most common complaint about
cloud dictation tools: they rewrite what you said instead of transcribing it.
Everything here is a rule you can read, test, and turn off.)

- **Filler removal**, at four levels you choose:
  - `none` — no filler removal at all
  - `light` — only unambiguous non-words: um, uh, erm, hmm
  - `medium` *(default)* — plus "you know", "sort of", "basically", and
    "like" **only** when it's a filler, never in "I like coffee"
  - `high` — plus sentence-initial "So"/"Well"/"Right" and trailing tag
    questions
- **Backtrack** — self-corrections are applied, not transcribed.
  `"Let's do coffee at 2 actually 3"` → `"Let's do coffee at 3."`
  Also handles "scratch that", "no wait", "I mean", and stutters
  (`"the the report"` → `"the report"`).
- **Spoken punctuation** — say "comma", "question mark", "new paragraph".
  Deliberately conservative: "a period of great change" is left alone. A
  missed conversion is invisible; a wrong one corrupts the sentence.
- **List formatting** —
  `"my top goals are one finish the report two send the deck"` becomes a
  numbered list. Won't fire on "I have one thing to say".
- **Dictionary** — find-and-replace for words Whisper gets wrong
  (`anthropic` → `Anthropic`). Whole-word, case-aware, and safe with
  metacharacters (`C++` works).
- **Snippets** — a spoken trigger expands to canned text ("my email" →
  your address). Expanded last, so a snippet body is inserted exactly as you
  wrote it, formatting and all.
- **Context awareness** — DictaFlow knows which app you're dictating into.
  In chat apps (Slack, Messages, WhatsApp, Discord, …) it drops the trailing
  full stop, because a period in a one-line message reads as terse. If the
  cursor is mid-sentence it doesn't capitalise the insertion.
  It reads the focused text field via the Accessibility API, skips secure
  password fields entirely, and **never takes a screenshot** — which is what
  caused Wispr Flow's 2025 privacy incident.
- **"press enter"** at the end of an utterance inserts the text and hits
  Return.

Everything above is individually switchable in Settings. The raw, unedited
transcript is always kept alongside the edited one, so nothing is lost.

---

## Dashboard

`dashboard.py` serves <http://localhost:7755> — standard library only.

- **Overview** — words dictated, time saved, streak, speaking pace, a 30-day
  chart, and median/p90 latency per model. Also an **accuracy panel**: how
  many attempts were discarded and *why*, which is how you'd notice if the
  speech filter started eating real dictation.
- **History** — every transcript, searchable, editable, pinnable, deletable,
  exportable (Markdown / text / JSON). Discarded attempts are shown too, with
  their rejection reason.
- **Dictionary & Snippets** — editable tables.
- **Settings** — everything above. Changes apply on your next dictation; no
  restart.

The dashboard binds to loopback, which is **not** the same as being private —
every page in your browser can also reach `127.0.0.1`. It therefore checks the
`Host` and `Origin` headers and requires a per-process CSRF token on every
write. The Whisper model path is deliberately **not** editable from the web UI:
it feeds `path_or_hf_repo`, which will download and load an arbitrary Hugging
Face repo, and since DictaFlow types its output as keystrokes, letting a web
page choose the model would be keystroke injection.

---

## Install

Requires an Apple Silicon Mac and **Python 3.10+**.

```bash
git clone https://github.com/Sophieverse/dictaflow.git
cd dictaflow
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Fetch the weights into `models/` (gitignored; 1.6GB for turbo):

```bash
.venv/bin/huggingface-cli download mlx-community/whisper-large-v3-turbo \
    --local-dir models/whisper-large-v3-turbo
.venv/bin/huggingface-cli download mlx-community/whisper-small-mlx \
    --local-dir models/whisper-small-mlx
```

Or put the Hugging Face repo ids directly in the config and let mlx_whisper
download them on first use.

```bash
.venv/bin/python dictaflow.py --check     # preflight
.venv/bin/python dictaflow.py             # run
```

macOS will ask for **Microphone**, **Accessibility** and **Input Monitoring**.
Grant them to the interpreter that actually runs — `.venv/bin/python` is a
symlink, so check `readlink -f .venv/bin/python` and grant that path.

### When the microphone goes silent

Three distinct failures look identical from the outside — every dictation
comes back "no speech detected" — and none of them raises an error. DictaFlow
now detects and repairs all three by itself; this section is what it is doing
and why.

Check first:

```bash
dictaflow.py --status      # the agent's own account of itself
```

1. **The device open blocks.** CoreAudio does not fail the open, it stops
   returning — the stack sits in `AudioUnitSetProperty` forever. DictaFlow
   abandons the attempt after 8 seconds, keeps loading models and handling
   keys, and **retries on the next keypress after a short cooldown**.

   That retry is the whole point. An earlier version set a flag when the
   open hung and never cleared it, so a machine that recovered in an hour
   met an app that refused every dictation for three days.

2. **The stream is open but stopped.** Callbacks simply cease; the handle
   stays valid and `close()` still succeeds. Detected by watching whether
   blocks arrive (`is_flowing()`), repaired by reopening.

3. **Blocks arrive and every sample is exactly zero.** The nastiest one: the
   device is alive, unmuted, at normal input volume, delivering blocks at
   precisely the right rate, and carrying nothing. It affects *every* app on
   the machine, `ffmpeg` included, so reopening the stream cannot help.

   The usual advice is `sudo killall coreaudiod`. Measured here on macOS
   26.5.1, a much smaller lever does the same job from user space: setting
   the input device's nominal sample rate to a different value and back
   forces the HAL to rebuild its IO context.

   ```
   before flip: ffmpeg peak = 0     (digital silence, ~30 minutes of it)
   after  flip: ffmpeg peak = 1059  (room tone)
   ```

   The agent does this automatically after ~3 seconds of unbroken zero
   blocks, at most three times, and never while you are recording. To do it
   by hand: `dictaflow.py --fix-audio`. If the device reports itself *muted*
   it is left alone — a reset would not unmute it.

Common to all three: a fault must be able to expire. Every check here is a
rolling window or a deadline, never a flag that only goes one way, because
the app cannot see the machine get better — it can only stop assuming it
hasn't.

**If it is genuinely a permissions problem.** A bare launchd job is its own
responsible process with no bundle identifier, so macOS has nothing to attach
a Microphone grant to and hands back digital zeros. Granting via Terminal does
not help — that grants *Terminal*. Fix: build the app bundle
(`app-src/build.sh`), point `com.dictaflow.agent.plist` at
`DictaFlow.app/Contents/MacOS/DictaFlow`, and grant Microphone **and**
Accessibility to "DictaFlow". Both, or the agent won't even see keypresses.

Only one DictaFlow may run at a time; a second copy exits immediately rather
than fighting for the microphone, which is its own way of producing an
apparently-dead app.

**Something not working?** `dictaflow.py --status` answers "is it working
right now?" in one line from the agent's heartbeat — the dashboard's dot uses
the same source, and turns amber for *running but not working*, which is a
different problem from *stopped* and needs a different fix. For more depth,
`dictaflow.py --doctor` reports models,
permissions, devices, a live 2-second mic test with the measured speech
metrics, command-mode readiness, and your most recent rejections with reasons.

### Always-on

```bash
cp com.dictaflow.agent.plist com.dictaflow.dashboard.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dictaflow.agent.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dictaflow.dashboard.plist
```

After re-granting permissions use `bootout` + `bootstrap`, not `kickstart -k` —
launchd caches the old denial.

---

## Configuration

`~/.dictaflow/config.json`, created on first run. Most of it is reachable from
the dashboard. Notable keys:

- `language` — `"en"` by default. **Leave it set** unless you dictate in more
  than one language; see the speed note below.
- `cleanup_level` — `none` / `light` / `medium` / `high`.
- `dictionary`, `snippets` — see above.
- `streaming` — transcribe while you speak. On by default.
- `temperature_ladder` — see below. Don't lengthen it without reading why.
- `bindings` — which key drives which model.
- `preroll_ms` — how much audio to keep from before the keypress.

The config is written atomically and validated on read: an invalid value is
rejected with a message rather than silently accepted, and an unparseable file
is preserved (not overwritten) while the app carries on with defaults.

---

## Notes from building it

Measured findings, including the approaches that turned out to be wrong.

### Speed

- **Whisper is encoder-bound and pads every clip to 30 seconds.** Quantizing
  the models (8/4-bit) gives **no speedup** — it's compute-bound, not
  memory-bound.
- **Pinning `language` nearly halves latency.** With no language set,
  mlx-whisper runs `detect_language()` first, which is an entire extra encoder
  pass just to conclude you're speaking English. Removing it takes Turbo from
  **1.93s to 0.99s**.
- **There is a latency cliff at ~20 seconds**, and it is not the extra encoder
  pass:

  | clip | 2s | 5s | 10s | 20s | 29s | 45s | 60s |
  |---|---|---|---|---|---|---|---|
  | Turbo | 1.07 | 1.20 | 1.23 | 1.47 | **18.49** | 18.42 | **36.00** |

  The cause is **temperature fallback**. Whisper's default `temperature` is a
  six-value tuple; when a decoded window trips `compression_ratio_threshold`
  or `logprob_threshold`, the entire window is re-decoded at the next
  temperature — up to six times. Isolating it on a 29s clip:
  **10.64s with the stock ladder, 1.59s at `temperature=0.0`** — and all six
  rungs produced identical (still-bad) output. The retries bought nothing and
  cost 6.7×. DictaFlow ships a two-rung ladder: one escape hatch, no
  pathological case.
- **Streaming is what makes long dictations feel fast.** Because cost is flat
  up to ~20s, the strategy is to never send a long clip: finished phrases are
  transcribed *while you keep talking*, cut only at natural pauses so a
  boundary never lands mid-word. On a 57-second dictation this cut the wait
  after releasing the key from 3.2s to **1.45s**, with 100% word overlap
  against the one-shot transcription.
- Audio is passed to mlx_whisper as a numpy array, which skips writing a temp
  WAV and skips the ffmpeg subprocess it would otherwise shell out to.

### Accuracy

- **Loudness is the wrong axis for detecting speech.** This began as a peak
  amplitude gate and failed in both directions at once: room tone at peak 331
  was transcribed as `"Thank you."`, while genuine quiet speech at peak 90
  transcribed *perfectly* but was discarded as silence. No threshold separates
  those, because the difference isn't level.

  What does separate them is **spectral flatness** — the geometric over the
  arithmetic mean of the power spectrum. Noise spreads energy evenly across
  frequency (→ 1.0); speech concentrates it into formants (→ 0.0). Measured:
  noise **0.561–1.000**, speech **0.001**, including quiet *and whispered*
  speech. Whispering removes the voiced pitch harmonics but keeps the formant
  structure, which is exactly what this measures — so it stays firmly on the
  speech side.

- **Whisper's own anti-hallucination guards cancel each other out.** The
  built-in repetition check is skipped whenever `no_speech_prob` is high, and
  the resulting "skip this window" decision is then *reversed* whenever
  `avg_logprob` looks healthy. Degenerate loops are confidently predicted, so
  both guards disarm and the garbage is returned. After exhausting the
  temperature ladder the library returns the **last** result — the most
  randomly-sampled one — with no best-of selection and no reject path. So
  rejection has to happen afterwards, here.

- **Gate on a windowed compression ratio, not the whole string.** The first
  version of the rejection filter measured zlib compressibility over the
  entire transcript, and threw away a perfectly good 1,200-character
  transcription. The reason:

  | chars | whole-string ratio | windowed max (200 chars) |
  |---|---|---|
  | 100 | 1.11 | 1.11 |
  | 1,500 | 2.01 | 1.65 |
  | 6,000 | 2.47 | 1.59 |
  | 12,000 | 3.10 | 1.64 |

  English is redundant, and zlib finds more back-references the more text you
  give it — so *any* fixed threshold on the whole-string ratio is guaranteed
  to eventually reject long legitimate dictation. The windowed maximum is flat
  at ~1.65 at every length, while real loops score 8–15 in some window.

- **Counting words with `.split()` misses whole classes of hallucination.**
  An observed failure was 889 characters of `"2018"` with no spaces at all —
  one "word", so a word-frequency filter short-circuited before any test ran,
  and it was pasted into the document. Detection now uses characters-per-word,
  a windowed compression ratio, and repeated-n-gram coverage.

- Validated against 246 real transcripts: **zero false positives**. It flags
  29 entries, all of which are genuine historical hallucinations — 27 of them
  the phrase "Thank you."

### Approaches that did *not* work

- **Gain-normalising quiet audio does nothing.** Word error rate is identical
  from peak 23,149 down to peak 199; Whisper's log-mel front-end already
  normalises per clip.
- **Whisper's confidence signals can't gate silence.** `no_speech_prob` reports
  **0.00 for pure digital silence**, and `avg_logprob` doesn't separate either
  — silence scores −0.21 while real quiet speech scores −0.24. The model is
  *more* confident about its hallucination than about your actual words.
- **An LLM cleanup pass isn't worth it.** A local Ollama model added ~40
  seconds for no real gain over Whisper's own punctuation. The rule-based pass
  replaced it entirely and runs in microseconds. The LLM is now used only for
  Command Mode, where you explicitly asked for a rewrite.

### Platform

- **Menu bar status items work fine from an unsigned bundle.** An earlier note
  in this file claimed otherwise; it was tested and is false.
- The mic is pinned to the built-in microphone rather than the system default.
  Recording from Bluetooth headphones forces macOS to renegotiate A2DP → HFP
  mid-stream, which throws PortAudio `-9986`.
- Text is inserted by clipboard paste, with the **whole pasteboard** saved and
  restored via `NSPasteboard` — `pbpaste`/`pbcopy` can only round-trip plain
  text, so copying a screenshot and then dictating used to destroy the
  screenshot. The synthesised ⌘V waits for modifier keys to be released first,
  because a held ⌥ turns it into ⌥⌘V — "Move Item Here" in Finder.
- If insertion fails for any reason (Accessibility revoked, macOS Secure Input
  active because a password field has focus, an app that won't take ⌘V), the
  text is **left on the clipboard**, the old clipboard is not restored, and you
  are told to press ⌘V. Losing a transcript is the one outcome worth trading
  anything to avoid — which is also why the transcript is written to history
  *before* the paste is attempted.

---

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
```

The suite pins the specific bugs above: the rejection filter against 246 real
transcripts, the streaming latency claim end-to-end, the double-tap timing, and
the concurrency cases (overlapping dictations, a press during transcription,
paste failures, mic failure recovery).

## License

MIT
