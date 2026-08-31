import os
import io
import sys
import time
import wave
import subprocess
import traceback
from collections import deque

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from sarvamai import SarvamAI
from sarvamai.play import save as save_tts_audio
from groq import Groq


# ===============================================================
# CONFIGURATION
# ===============================================================

load_dotenv()  # reads a .env file in the current directory, if present

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

STT_MODEL = "saaras:v3"          # auto-detects language
TTS_MODEL = "bulbul:v3"

GROQ_MODEL = "openai/gpt-oss-20b"

# Short-term memory: how many of the assistant's own past responses get
# fed back into the system prompt on the next turn.
MEMORY_SIZE = 5
response_history = deque(maxlen=MEMORY_SIZE)

SAMPLE_RATE = 16000

TTS_OUTPUT_FILE = "response.wav"

# Sarvam language codes -> display names.
# Only used for logging; STT auto-detects, so this isn't a fixed list
# we classify into anymore -- Sarvam covers all 22 scheduled languages.
LANGUAGE_NAMES = {
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "kn-IN": "Kannada",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "en-IN": "English",
}

# Sarvam bulbul:v3 speakers -- pick one per language if you want a
# consistent voice per language, or just use one speaker for all.
# Valid bulbul:v3 speakers (per Sarvam API): aditya, ritu, ashutosh, priya,
# neha, rahul, pooja, rohan, simran, kavya, amit, dev, ishita, shreya, ratan,
# varun, manan, sumit, roopa, kabir, aayan, shubh, advait, anand, tanya,
# tarun, sunny, mani, gokul, vijay, shruti, suhani, mohit, kavitha, rehan,
# soham, rupali, niharika
TTS_SPEAKERS = {
    "hi-IN": "priya",
    "bn-IN": "priya",
    "mr-IN": "priya",
    "kn-IN": "priya",
    "ta-IN": "priya",
    "te-IN": "priya",
    "en-IN": "priya",
}
DEFAULT_SPEAKER = "priya"


# ===============================================================
# CLIENTS
# ===============================================================

sarvam_client = None
groq_client = None


def init_clients():

    global sarvam_client, groq_client

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY environment variable is not set."
        )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set."
        )

    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)


# ===============================================================
# UTILITIES
# ===============================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_float32(audio):
    audio = np.asarray(audio)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


# ===============================================================
# MICROPHONE
# ===============================================================

def record_audio():

    section("STEP 1 — MICROPHONE")

    print()
    print("Press ENTER when ready...")
    input()

    print()
    print("🎙️  RECORDING")
    print()
    print("Speak naturally.")
    print("Press Ctrl+C to stop recording.")
    print()

    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    stream.start()

    try:
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print()
        print("Recording stopped.")

    finally:
        stream.stop()
        stream.close()

    if not chunks:
        raise RuntimeError("No audio was recorded.")

    audio = np.concatenate(chunks, axis=0)
    audio = audio[:, 0]
    audio = safe_float32(audio)
    audio = np.clip(audio, -1.0, 1.0)

    duration = len(audio) / SAMPLE_RATE

    audio_int16 = (audio * 32767).astype(np.int16)

    # Build the WAV entirely in memory -- no file written to disk.
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    buffer.seek(0)

    print()
    print("Recording complete.")
    print()
    print(f"Duration:       {duration:.2f}s")
    print(f"Samples:        {len(audio):,}")

    return buffer


# ===============================================================
# SARVAM — SPEECH TO TEXT (auto language detection + transcription
# in a single API call; replaces the local Whisper classifier and
# IndicConformer entirely)
# ===============================================================

def transcribe(audio_buffer):

    section("STEP 2 — SARVAM SPEECH-TO-TEXT")

    print()
    print(f"Model: {STT_MODEL}")
    print("Sending audio to Sarvam AI...")

    start = time.time()

    response = sarvam_client.speech_to_text.transcribe(
        file=("input.wav", audio_buffer, "audio/wav"),
        model=STT_MODEL,
        language_code="unknown",  # auto-detect
    )

    elapsed = time.time() - start

    transcript = (response.transcript or "").strip()
    language_code = getattr(response, "language_code", None) or "hi-IN"

    if not transcript:
        raise RuntimeError("Sarvam STT returned an empty transcript.")

    language_name = LANGUAGE_NAMES.get(language_code, language_code)

    print()
    print(f"STT time: {elapsed:.2f}s")
    print(f"Detected language: {language_name} ({language_code})")
    print()
    print("TRANSCRIPT:")
    print()
    print(transcript)

    return transcript, language_code


# ===============================================================
# GROQ — LLM
# ===============================================================

def ask_groq(transcript, language_code):

    section("STEP 3 — GROQ LLM")

    language_name = LANGUAGE_NAMES.get(language_code, language_code)

    print()
    print(f"Model: {GROQ_MODEL}")
    print("Sending transcript to Groq...")

    system_prompt = (
        f"You are a helpful voice assistant. The user is speaking in "
        f"{language_name}. Reply in the same language as the user. "
        f"Keep the answer natural and conversational. Do not mention "
        f"language detection. Do not translate unless the user asks. "
        f"The response will be spoken using text-to-speech, so avoid "
        f"markdown, code blocks, or symbols that are difficult to speak."
    )

    if response_history:

        memory_block = "\n".join(
            f"{i}. {past_response}"
            for i, past_response in enumerate(response_history, start=1)
        )

        system_prompt += (
            "\n\nFor context, here are your own last "
            f"{len(response_history)} response(s) in this conversation "
            "(most recent last). Use them only to stay consistent and "
            "avoid repeating yourself -- do not mention this list "
            "explicitly:\n"
            f"{memory_block}"
        )

    start = time.time()

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )

    elapsed = time.time() - start

    response = completion.choices[0].message.content.strip()

    if not response:
        raise RuntimeError("Groq returned an empty response.")

    response_history.append(response)

    print()
    print(f"LLM time: {elapsed:.2f}s")
    print(f"Memory:   {len(response_history)}/{MEMORY_SIZE} past responses in context")
    print()
    print("LLM RESPONSE:")
    print()
    print(response)

    return response


# ===============================================================
# SARVAM — TEXT TO SPEECH
# ===============================================================

def generate_tts(text, language_code):

    section("STEP 4 — SARVAM TEXT-TO-SPEECH")

    speaker = TTS_SPEAKERS.get(language_code, DEFAULT_SPEAKER)

    print()
    print(f"Model:    {TTS_MODEL}")
    print(f"Language: {LANGUAGE_NAMES.get(language_code, language_code)}")
    print(f"Speaker:  {speaker}")
    print()
    print("Generating speech...")

    start = time.time()

    audio_response = sarvam_client.text_to_speech.convert(
        text=text,
        language_code=language_code,
        model=TTS_MODEL,
        speaker=speaker,
    )

    elapsed = time.time() - start

    save_tts_audio(audio_response, TTS_OUTPUT_FILE)

    print()
    print(f"TTS time: {elapsed:.2f}s")
    print(f"Saved: {os.path.abspath(TTS_OUTPUT_FILE)}")

    return TTS_OUTPUT_FILE


# ===============================================================
# PLAY AUDIO
# ===============================================================

def play_audio(filename):

    section("STEP 5 — SPEAKER")

    print()
    print(f"Playing: {filename}")

    try:
        subprocess.run(["afplay", filename], check=True)

    except FileNotFoundError:
        print("afplay not found.")
        print("Audio file is available at:")
        print(os.path.abspath(filename))


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def main():

    section("INDIC MULTILINGUAL ONLINE VOICE AI")

    print()
    print("Architecture:")
    print()
    print(
        """
  Microphone
      ↓
  WAV
      ↓
  Sarvam STT (saaras:v3) — auto language detection + transcription
      ↓
  Groq LLM (openai/gpt-oss-20b)
      ↓
  Sarvam TTS (bulbul:v3)
      ↓
  Speaker
"""
    )

    print(f"STT:  {STT_MODEL} (Sarvam AI)")
    print(f"LLM:  {GROQ_MODEL} (Groq)")
    print(f"TTS:  {TTS_MODEL} (Sarvam AI)")

    try:
        init_clients()

        turn = 1

        while True:

            section(f"TURN {turn}")

            audio_buffer = record_audio()

            transcript, language_code = transcribe(audio_buffer)

            llm_response = ask_groq(transcript, language_code)

            tts_file = generate_tts(llm_response, language_code)

            play_audio(tts_file)

            section("TURN COMPLETE")

            print()
            print(f"Language:    {LANGUAGE_NAMES.get(language_code, language_code)}")
            print(f"Code:        {language_code}")
            print()
            print("Transcript:")
            print(transcript)
            print()
            print("LLM response:")
            print(llm_response)
            print()
            print("TTS:")
            print(os.path.abspath(tts_file))

            print()
            choice = input(
                "Press ENTER for another turn, or type 'q' then ENTER to quit: "
            ).strip().lower()

            if choice == "q":
                print()
                print("Session ended.")
                break

            turn += 1

    except KeyboardInterrupt:
        print()
        print("Pipeline interrupted.")
        sys.exit(1)

    except Exception as e:
        section("ERROR")
        print()
        print(type(e).__name__)
        print()
        print(str(e))
        print()
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()