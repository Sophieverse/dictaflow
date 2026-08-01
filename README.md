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

Every transcript is appended to `~/transcriptions/transcripts.md`. There's also
a small web dashboard (`dashboard.py`, <http://localhost:7755>) to search them.

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
- `cleanup_enabled`: run a local Ollama model over the transcript to tidy it up.
  Off by default — Whisper already punctuates well, and it added ~40s for no
  real gain.

## Notes from building it

- Quantizing the models (8/4-bit) gives **no speedup** on Apple Silicon. Whisper
  pads every clip to 30s of mel spectrogram and runs the full encoder regardless
  of how long you actually spoke, so it's compute-bound, not memory-bound.
- Near-silent clips are dropped before they reach Whisper. Feeding it silence
  reliably produces hallucinated `"Thank you."` — an artifact of training on
  YouTube captions, where silence is often captioned "thanks for watching".
- The mic is pinned to the built-in microphone rather than the system default.
  Recording from Bluetooth headphones forces macOS to renegotiate A2DP → HFP
  mid-stream, which throws PortAudio `-9986` errors.
- The status pill is a plain borderless `NSWindow`, not an `NSStatusItem`.
  Menu-bar status items refuse to render from an unsigned, un-notarized bundle
  on recent macOS; an ordinary floating window has no such restriction.

## License

MIT
