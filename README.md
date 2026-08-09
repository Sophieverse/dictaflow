# DictaFlow

Hold-to-talk voice dictation for macOS that runs **entirely on your own machine**.
Hold a key, talk, release — the text is transcribed locally and pasted into
whatever field you were typing in. No account, no API key, no audio leaving
your Mac.

An open-weight alternative to Wispr Flow / Superwhisper, built on
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
(Apple Silicon / Metal).

## How it works

| Gesture | What happens |
| --- | --- |
| **Hold right-⌥ (Option)** | Records while held → transcribes with Whisper **large-v3-turbo** (most accurate, ~1.8s) |
| **Hold right-⌘ (Command)** | Same, but Whisper **small** (fastest, ~0.5s) |
| **Double-tap** either key | Hands-free mode — keeps recording without holding. Tap the same key again to stop. |

A floating pill appears at the bottom of the screen whenever DictaFlow is
listening (red) or transcribing (amber), so you always know whether the mic is
hot. No pill = not recording.

Every transcript is appended to `~/transcriptions/transcripts.md`.

## Dashboard

`dashboard.py` serves <http://localhost:7755> — standard library only, no
dependencies:

- **Overview** — words dictated, time saved vs typing, daily streak, speaking
  pace, words-per-day chart, and median/p90 transcription latency per model.
- **History** — every transcript, searchable, with one-click copy.
- **Settings** — language, vocabulary hint, and cleanup toggle. Changes take
  effect on your next dictation; no restart needed.

It reads two files: `transcripts.md` for what you said, and `events.jsonl` for
how well it worked (including the attempts that produced no text, so the
rejection rate is visible rather than silent).

## Requirements

- Apple Silicon Mac (mlx-whisper needs Metal)
- **Python 3.10+** — the code uses `X | None` syntax, so the system 3.9 will not run it

## Setup

```bash
git clone https://github.com/Sophieverse/dictaflow.git
cd dictaflow
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Fetch the model weights into `models/` (they're gitignored — 1.6GB for turbo):

```bash
.venv/bin/pip install huggingface_hub
.venv/bin/huggingface-cli download mlx-community/whisper-large-v3-turbo \
    --local-dir models/whisper-large-v3-turbo
.venv/bin/huggingface-cli download mlx-community/whisper-small-mlx \
    --local-dir models/whisper-small-mlx
```

Or skip the download entirely and put the Hugging Face repo ids directly in
`TURBO_MODEL` / `SMALL_MODEL` in `dictaflow.py` — mlx_whisper accepts either.

Run it:

```bash
.venv/bin/python dictaflow.py
```

macOS will ask for **Microphone**, **Accessibility**, and **Input Monitoring**
permissions on first use. Grant them to the Python binary that's actually
running (`.venv/bin/python` resolves to a real interpreter elsewhere — check
`readlink -f .venv/bin/python` and grant that path).

## Running it always-on

`com.dictaflow.agent.plist` is a LaunchAgent template that starts DictaFlow at
login and keeps it alive. Edit the paths inside it to match your checkout, then:

```bash
cp com.dictaflow.agent.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dictaflow.agent.plist
```

If you re-grant permissions afterwards, use `bootout` + `bootstrap` rather than
`kickstart -k` — launchd caches the old permission denial.

## Configuration

`~/.dictaflow/config.json` is created on first run.

- `backend`: `"local"` (default, fully offline) or `"groq"` (cloud Whisper +
  llama-3.3-70b, needs `groq_api_key`)
- `language`: `"en"` by default. Leave it set unless you dictate in more than
  one language — see the speed note below.
- `initial_prompt`: a short list of names, acronyms and jargon you use often.
  It's fed to the decoder as if it were preceding text, which biases it away
  from spelling your vocabulary phonetically. Keep it short; long prompts can
  bleed into the output.
- `cleanup_enabled`: run a local Ollama model over the transcript to tidy it up.
  Off by default — Whisper already punctuates well, and it added ~40s for no
  real gain.

## Notes from building it

- Quantizing the models (8/4-bit) gives **no speedup** on Apple Silicon. Whisper
  pads every clip to 30s of mel spectrogram and runs the full encoder regardless
  of how long you actually spoke, so it's compute-bound, not memory-bound.
- **Pinning `language` nearly halves latency.** Because the encoder is the whole
  cost, and because mlx-whisper runs `detect_language()` — an entire extra
  encoder pass — whenever no language is set, removing that pass takes Turbo
  from **1.93s to 0.99s**. This is by far the biggest speed lever available;
  quantization and smaller models both matter less.
- **Loudness is the wrong axis for detecting speech.** This started as a peak
  amplitude gate, and it failed in both directions: room tone at peak 331 got
  transcribed as `"Thank you."`, while genuine quiet speech at peak 90
  transcribed *perfectly* but was thrown away as silence. No threshold can
  separate those, because the difference isn't level.
  What does separate them is **spectral flatness** — the ratio of the geometric
  to the arithmetic mean of the power spectrum. Noise spreads energy evenly
  across frequency (→ 1.0); speech concentrates it into formants (→ 0.0).
  Measured here: noise 0.561–1.000, speech 0.001, including quiet *and*
  whispered speech. Whispering removes the voiced pitch harmonics but keeps the
  formant structure, so it stays firmly on the speech side of the gap.
- **Whisper's own confidence signals are useless for this.** The obvious fix is
  to gate on `no_speech_prob`, but it reports **0.00 for pure digital silence**,
  and `avg_logprob` doesn't separate either — silence scores −0.21 while real
  quiet speech scores −0.24. The model is *more* confident about its
  hallucination than about your actual words.
- Whisper's training data is largely YouTube captions, so on unintelligible
  audio it emits the phrase that most often captions such a moment. Anything
  that survives the gate is checked against a small blocklist (`"Thank you."`,
  `"Thanks for watching"`, …), matched only against the **whole** output so
  saying "thank you" inside a real sentence is unaffected.
- Gain-normalising quiet audio before transcription does **nothing** — a
  plausible idea worth recording as a dead end. Whisper's log-mel front-end
  already normalises per clip, so word error rate is identical from peak 23149
  all the way down to peak 199.
- The mic is pinned to the built-in microphone rather than the system default.
  Recording from Bluetooth headphones forces macOS to renegotiate A2DP → HFP
  mid-stream, which throws PortAudio `-9986` errors.
- The status pill is a plain borderless `NSWindow`, not an `NSStatusItem`.
  Menu-bar status items refuse to render from an unsigned, un-notarized bundle
  on recent macOS; an ordinary floating window has no such restriction.

## License

MIT
