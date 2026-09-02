# Indic Voice Assistant

A multilingual, voice-in / voice-out AI assistant for India that listens to a
spoken question in any of the 22 scheduled languages, transcribes it,
generates an answer, and speaks the answer back — all in the same language
the user spoke.

## How it works

```
Microphone
    ↓
WAV (in-memory, no temp files)
    ↓
Sarvam STT (saaras:v3)      — auto-detects the spoken language and transcribes
    ↓
Groq LLM (openai/gpt-oss-20b) — generates a reply in the same language
    ↓
Sarvam TTS (bulbul:v3)      — synthesizes the reply as speech
    ↓
Speaker
```

Both scripts run as an interactive loop in the terminal: press **Enter** to
start recording, speak, press **Ctrl+C** to stop recording, and the assistant
transcribes, thinks, and speaks back before prompting for the next turn.

### `app.py` — core pipeline

- **Language auto-detection.** Sarvam's `saaras:v3` model detects the
  spoken language and transcribes it in a single call — no separate
  classifier needed.
- **Reply in the same language.** The LLM is instructed to answer in
  whatever language the user spoke, in a natural, TTS-friendly style (no
  markdown, code blocks, or hard-to-speak symbols).
- **Short-term memory.** The assistant's own last 5 responses are fed back
  into the system prompt so it stays consistent across turns without
  repeating itself.

## Requirements

- Python 3.10+
- A working microphone and speakers
- API keys for [Sarvam AI](https://www.sarvam.ai/) and [Groq](https://groq.com/)
- `afplay` for audio playback (macOS default). On other platforms, the
  script falls back to printing the path of the generated WAV file so you
  can play it manually.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys**

   Create a `.env` file in the project root:

   ```
   SARVAM_API_KEY=your_sarvam_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

3. **Run the assistant**

   ```bash
   python app.py
   ```

## Usage

1. Press **Enter** when prompted.
2. Speak naturally in any supported language.
3. Press **Ctrl+C** to stop recording.
4. The assistant prints the transcript, its detected language, the LLM's
   response, and then plays the spoken reply back.
5. Press **Enter** to continue, or type `q` to end the session.

## Configuration

Key constants at the top of `app.py`:

| Constant | Purpose |
|---|---|
| `STT_MODEL` / `TTS_MODEL` | Sarvam model versions used |
| `GROQ_MODEL` | Groq LLM model |
| `MEMORY_SIZE` | Number of past responses kept in context |
| `TTS_SPEAKERS` / `DEFAULT_SPEAKER` | Voice used per language (Sarvam `bulbul:v3` speaker names) |

## Supported languages

Speech-to-text auto-detects across all 22 scheduled Indian languages via
Sarvam's `saaras:v3` model. `LANGUAGE_NAMES` in the code maps the most
common codes (Hindi, Bengali, Marathi, Kannada, Tamil, Telugu, English) to
display names for logging — this is a display convenience, not a
restriction on which languages are recognized.

## Project structure

```
indic-voice-assistant/
├── app.py                          # Core voice pipeline
├── requirements.txt
└── .env                            # API keys (not committed going forward)
```

## Known limitations

- Single-user, terminal-based, single-turn-at-a-time — not built for
  concurrent sessions or a web/mobile front end.
- Recording length is manual (stop with Ctrl+C) rather than automatic
  silence detection.