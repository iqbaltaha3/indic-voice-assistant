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

from rag import LandRecordsRAG


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
rag = None


def init_clients():

    global sarvam_client, groq_client, rag

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

    print()
    print("Building land-records knowledge base index...")
    rag = LandRecordsRAG()
    print(f"Indexed {len(rag.chunks)} chunks from the knowledge base.")


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
# GROQ — AGENTIC LLM
#
# openai/gpt-oss-20b supports Groq's built-in server-side tools:
# browser_search (live web search) and code_interpreter (sandboxed
# Python execution). The model decides on its own whether a turn
# needs a tool -- Groq executes it server-side and returns the final
# answer, so there's no tool-calling loop to hand-roll here.
# ===============================================================

AGENT_TOOLS = [
    {"type": "browser_search"},
    {"type": "code_interpreter"},
]


def ask_groq(transcript, language_code):

    section("STEP 3 — GROQ AGENT")

    language_name = LANGUAGE_NAMES.get(language_code, language_code)

    print()
    print(f"Model: {GROQ_MODEL}")
    print("Tools: browser_search, code_interpreter")
    print("Sending transcript to Groq...")

    system_prompt = (
        f"You are a helpful voice assistant that specializes in Indian "
        f"land records and property due diligence -- explaining terms "
        f"like RTC, 7/12 extract, Khatauni, Patta, Khata, encumbrance "
        f"certificates, mutation, and helping people understand what to "
        f"check before buying land, in plain language. The user is "
        f"speaking in {language_name}. Reply in the same language as the "
        f"user. Keep the answer natural and conversational. Do not "
        f"mention language detection. Do not translate unless the user "
        f"asks. The response will be spoken using text-to-speech, so "
        f"avoid markdown, code blocks, or symbols that are difficult to "
        f"speak. You have access to a live web search tool and a Python "
        f"code execution tool for anything outside the land-records "
        f"knowledge base -- current events, facts you're unsure of, or "
        f"calculations. Don't mention that you used a tool -- just "
        f"answer naturally with the result."
    )

    retrieved_context = rag.build_context_block(transcript)

    if retrieved_context:

        system_prompt += (
            "\n\nHere is relevant, verified information from a curated "
            "land-records knowledge base. Base your answer on this "
            "information where it applies, rather than guessing -- land "
            "record details are state-specific and getting them wrong "
            "can genuinely mislead someone about their legal rights. If "
            "the knowledge base doesn't cover something the user asked, "
            "say so plainly rather than inventing details, and suggest "
            "they confirm with a local revenue office or lawyer. Do not "
            "read out the source labels -- just use the information:\n\n"
            f"{retrieved_context}"
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

    if retrieved_context:
        retrieved_sources = [
            source for source, _, _ in rag.retrieve(transcript)
        ]
        print(f"RAG:   grounded in {retrieved_sources}")
    else:
        print("RAG:   no relevant knowledge base match -- answering ungrounded")

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        tools=AGENT_TOOLS,
        tool_choice="auto",
    )

    elapsed = time.time() - start

    message = completion.choices[0].message

    response = (message.content or "").strip()

    if not response:
        raise RuntimeError("Groq returned an empty response.")

    response_history.append(response)

    # executed_tools tells us which built-in tools the model actually
    # invoked server-side this turn (may be absent/empty if none were
    # used) -- useful for showing agentic behavior, not just logging.
    executed_tools = getattr(message, "executed_tools", None) or []

    print()
    print(f"LLM time: {elapsed:.2f}s")
    print(f"Memory:   {len(response_history)}/{MEMORY_SIZE} past responses in context")

    if executed_tools:

        print(f"Tools used this turn: {len(executed_tools)}")

        for tool_call in executed_tools:

            tool_type = getattr(tool_call, "type", "unknown")
            print(f"  - {tool_type}")

    else:
        print("Tools used this turn: none (answered directly)")

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

    section("INDIC LAND-RECORDS VOICE AGENT")

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
  BM25 retrieval over land-records knowledge base
      ↓
  Groq Agent (openai/gpt-oss-20b)
      + grounded context (RAG)
      + browser_search (live web lookup, server-side)
      + code_interpreter (Python execution, server-side)
      + short-term memory (last 5 responses)
      ↓
  Sarvam TTS (bulbul:v3)
      ↓
  Speaker
"""
    )

    print(f"STT:   {STT_MODEL} (Sarvam AI)")
    print(f"Agent: {GROQ_MODEL} (Groq, tools: browser_search, code_interpreter)")
    print(f"TTS:   {TTS_MODEL} (Sarvam AI)")

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