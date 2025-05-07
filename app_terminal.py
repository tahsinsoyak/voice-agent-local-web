import os
import sys
import time
import queue
import tempfile
import threading
import numpy as np
import sounddevice as sd
import wave
from faster_whisper import WhisperModel
from llama_cpp import Llama
import pyttsx3

# --- Configuration ---
STT_MODEL_SIZE     = "base"
STT_COMPUTE_TYPE   = "int8"

# GGUF model path & settings
LLM_GGUF_PATH      = "C:/Users/tahsinsoyak/.lmstudio/models/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf"
LLM_N_CTX          = 512
LLM_MAX_NEW_TOKENS = 60
LLM_TEMPERATURE    = 0.7
LLM_TOP_P          = 0.9
LLM_REPEAT_PENALTY = 1.1

TTS_RATE = 180

# --- Load STT (whisper) ---
try:
    print(f"Loading STT model: {STT_MODEL_SIZE} ({STT_COMPUTE_TYPE})...")
    stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type=STT_COMPUTE_TYPE)
    print("STT model loaded.")
except Exception as e:
    print(f"Error loading STT model: {e}")
    sys.exit(1)

# --- Load LLM via llama-cpp-python ---
try:
    print(f"Loading Llama.cpp model from: {LLM_GGUF_PATH}...")
    if not os.path.isfile(LLM_GGUF_PATH):
        raise FileNotFoundError(f"No such GGUF file: {LLM_GGUF_PATH}")
    llm = Llama(
        model_path=LLM_GGUF_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=4,        # adjust to your CPU cores
        seed=1337
    )
    print("LLM model loaded.")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    print("Make sure you’ve installed llama-cpp-python and that the GGUF path is correct.") 
    sys.exit(1)

# Fallbacks & history
fallback_responses = {
    "greeting":      "Merhaba! İstanbul tur rehberinizim. Size nasıl yardımcı olabilirim?",
    "not_understood":"Üzgünüm, sizi anlayamadım. Lütfen tekrar eder misiniz?",
    "default":       "İstanbul hakkında başka ne öğrenmek istersiniz?"
}
conversation_history = []
audio_queue = queue.Queue()

# TTS init
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', TTS_RATE)
except Exception as e:
    print(f"Warning: TTS init error: {e}")
    tts_engine = None

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def record_audio(duration=5, samplerate=16000):
    audio_data = []
    while not audio_queue.empty():
        audio_queue.get()
    with sd.RawInputStream(samplerate=samplerate, blocksize=4000, dtype='int16',
                           channels=1, callback=audio_callback):
        print(f"Recording for {duration}s…")
        t0 = time.time()
        while time.time() - t0 < duration:
            if not audio_queue.empty():
                audio_data.append(audio_queue.get())
            time.sleep(0.05)
    print("Recording finished.")
    audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name,'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(samplerate)
            w.writeframes(audio_np.tobytes())
        return tmp.name

def transcribe_audio(audio_file):
    try:
        print("Transcribing…")
        segments, info = stt_model.transcribe(
            audio_file, beam_size=5, language="tr",
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
        )
        print(f"Detected language {info.language} (p={info.language_probability})")
        text = "".join(seg.text for seg in segments).strip()
        os.remove(audio_file)
        return text or None
    except Exception as e:
        print(f"STT error: {e}")
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return None

def truncate_response(text, max_words=35):
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    p = cut.rfind(".")
    return (cut[:p+1] if p>len(cut)//2 else cut + "...")

def generate_response(user_text):
    if not user_text:
        return fallback_responses["not_understood"]
    conversation_history.append({"role":"user","content":user_text})
    if len(conversation_history)>6:
        conversation_history[:] = conversation_history[-6:]
    # build a short system+chat prompt
    prompt = "Sen İstanbul'da profesyonel bir tur rehberisin. Kısa ve net yanıt ver.\n"
    for m in conversation_history[:-1]:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
    prompt += f"User: {user_text}\nAssistant:"
    try:
        print("Generating LLM response…")
        out = llm(
            prompt,
            max_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            repeat_penalty=LLM_REPEAT_PENALTY,
            stop=["\nUser:","\nSystem:"]
        )
        text = out["choices"][0]["text"].strip()
        text = truncate_response(text.split("\n")[0])
        conversation_history.append({"role":"assistant","content":text})
        return text if len(text.split())>=2 else fallback_responses["default"]
    except Exception as e:
        print(f"LLM error: {e}")
        if conversation_history and conversation_history[-1]["role"]=="user":
            conversation_history.pop()
        return fallback_responses["default"]

def speak_text(text):
    if not tts_engine:
        print("TTS unavailable.")
        return
    def _s():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except: pass
    t = threading.Thread(target=_s)
    t.start()
    t.join(timeout=20)
    if t.is_alive():
        print("Warning: TTS timed out.")
        try: tts_engine.stop()
        except: pass

def main():
    speak_text(fallback_responses["greeting"])
    while True:
        print("\nPress Enter to record…")
        input()
        wav = record_audio(5)
        user = transcribe_audio(wav)
        if user:
            print("User:", user)
            if user.lower() in ["çıkış","kapat","exit","quit"]:
                speak_text("Görüşmek üzere! İyi günler.")
                break
            resp = generate_response(user)
            print("Guide:", resp)
            speak_text(resp)
        else:
            print("No speech detected.")
            speak_text(fallback_responses["not_understood"])
    if tts_engine:
        try: tts_engine.stop()
        except: pass
    print("Program terminated.")

if __name__=="__main__":
    main()
