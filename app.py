import os
import time
from collections import deque

import streamlit as st
from dotenv import load_dotenv

from sarvamai import SarvamAI
from sarvamai.play import save as save_tts_audio
from groq import Groq


# ===============================================================
# CONFIGURATION
# ===============================================================

load_dotenv()  # reads a .env file locally; on Streamlit Cloud, use st.secrets instead

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY") or st.secrets.get("SARVAM_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

STT_MODEL = "saaras:v3"          # auto-detects language
TTS_MODEL = "bulbul:v3"

GROQ_MODEL = "openai/gpt-oss-20b"

# Short-term memory: how many of the assistant's own past responses get
# fed back into the system prompt on the next turn.
MEMORY_SIZE = 5

TTS_OUTPUT_FILE = "response.wav"

# Sarvam language codes -> display names.
# Only used for display; STT auto-detects, so this isn't a fixed list
# we classify into -- Sarvam covers all 22 scheduled languages.
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
# CLIENTS (cached so we don't reconnect on every Streamlit rerun)
# ===============================================================

@st.cache_resource
def get_clients():

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Add it in Streamlit Cloud under "
            "Settings -> Secrets, or in a local .env file."
        )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it in Streamlit Cloud under "
            "Settings -> Secrets, or in a local .env file."
        )

    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)

    return sarvam_client, groq_client


# ===============================================================
# SARVAM — SPEECH TO TEXT (auto language detection + transcription
# in a single API call)
# ===============================================================

def transcribe(sarvam_client, audio_bytes):

    start = time.time()

    response = sarvam_client.speech_to_text.transcribe(
        file=("input.wav", audio_bytes, "audio/wav"),
        model=STT_MODEL,
        language_code="unknown",  # auto-detect
    )

    elapsed = time.time() - start

    transcript = (response.transcript or "").strip()
    language_code = getattr(response, "language_code", None) or "hi-IN"

    if not transcript:
        raise RuntimeError("Sarvam STT returned an empty transcript.")

    return transcript, language_code, elapsed


# ===============================================================
# GROQ — LLM
# ===============================================================

def ask_groq(groq_client, transcript, language_code, response_history):

    language_name = LANGUAGE_NAMES.get(language_code, language_code)

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

    return response, elapsed


# ===============================================================
# SARVAM — TEXT TO SPEECH
# ===============================================================

def generate_tts(sarvam_client, text, language_code):

    speaker = TTS_SPEAKERS.get(language_code, DEFAULT_SPEAKER)

    start = time.time()

    audio_response = sarvam_client.text_to_speech.convert(
        text=text,
        language_code=language_code,
        model=TTS_MODEL,
        speaker=speaker,
    )

    elapsed = time.time() - start

    save_tts_audio(audio_response, TTS_OUTPUT_FILE)

    return TTS_OUTPUT_FILE, speaker, elapsed


# ===============================================================
# STREAMLIT APP
# ===============================================================

def main():

    st.set_page_config(page_title="Indic Voice Assistant", page_icon="🎙️")

    st.title("🎙️ Indic Voice Assistant")
    st.caption(
        "Speak in any of the 22 scheduled Indian languages. The assistant "
        "detects the language, replies in it, and speaks the reply back."
    )

    if "response_history" not in st.session_state:
        st.session_state.response_history = deque(maxlen=MEMORY_SIZE)

    if "turn_log" not in st.session_state:
        st.session_state.turn_log = []

    try:
        sarvam_client, groq_client = get_clients()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Architecture")
    st.code(
        "Browser mic\n"
        "    ↓\n"
        "Sarvam STT (saaras:v3) — auto language detection + transcription\n"
        "    ↓\n"
        "Groq LLM (openai/gpt-oss-20b)\n"
        "    ↓\n"
        "Sarvam TTS (bulbul:v3)\n"
        "    ↓\n"
        "Browser speaker",
        language=None,
    )

    st.subheader("Speak")
    audio_value = st.audio_input("Record your question")

    if audio_value is not None:

        if st.button("Process this recording", type="primary"):

            with st.status("Running the pipeline...", expanded=True) as status:

                try:
                    status.write("Transcribing with Sarvam STT...")
                    transcript, language_code, stt_time = transcribe(
                        sarvam_client, audio_value.getvalue()
                    )
                    language_name = LANGUAGE_NAMES.get(language_code, language_code)
                    status.write(
                        f"Detected language: {language_name} ({language_code}) "
                        f"in {stt_time:.2f}s"
                    )

                    status.write("Asking Groq LLM...")
                    llm_response, llm_time = ask_groq(
                        groq_client,
                        transcript,
                        language_code,
                        st.session_state.response_history,
                    )
                    st.session_state.response_history.append(llm_response)
                    status.write(f"LLM responded in {llm_time:.2f}s")

                    status.write("Generating speech with Sarvam TTS...")
                    tts_file, speaker, tts_time = generate_tts(
                        sarvam_client, llm_response, language_code
                    )
                    status.write(f"TTS ({speaker}) generated in {tts_time:.2f}s")

                    with open(tts_file, "rb") as f:
                        tts_bytes = f.read()

                    st.session_state.turn_log.insert(
                        0,
                        {
                            "language_name": language_name,
                            "language_code": language_code,
                            "transcript": transcript,
                            "response": llm_response,
                            "audio": tts_bytes,
                        },
                    )

                    status.update(label="Done", state="complete")

                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.exception(e)

    if st.session_state.turn_log:

        st.subheader("Conversation")

        for i, turn in enumerate(st.session_state.turn_log):

            n = len(st.session_state.turn_log) - i

            with st.container(border=True):
                st.markdown(
                    f"**Turn {n} — {turn['language_name']} ({turn['language_code']})**"
                )
                st.markdown(f"**You said:** {turn['transcript']}")
                st.markdown(f"**Assistant:** {turn['response']}")
                st.audio(turn["audio"], format="audio/wav")

    with st.sidebar:
        st.header("Settings")
        st.write(f"STT: `{STT_MODEL}` (Sarvam AI)")
        st.write(f"LLM: `{GROQ_MODEL}` (Groq)")
        st.write(f"TTS: `{TTS_MODEL}` (Sarvam AI)")
        st.write(
            f"Memory: {len(st.session_state.response_history)}/{MEMORY_SIZE} "
            "past responses in context"
        )
        if st.button("Clear conversation"):
            st.session_state.response_history.clear()
            st.session_state.turn_log = []
            st.rerun()


if __name__ == "__main__":
    main()