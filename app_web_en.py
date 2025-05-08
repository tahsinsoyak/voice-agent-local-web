import os
import tempfile
from flask import Flask, render_template, request, jsonify
from faster_whisper import WhisperModel
from llama_cpp import Llama
from pydub import AudioSegment
import soundfile as sf

app = Flask(__name__)

# —— Configuration ——
STT_MODEL_SIZE   = "base"
STT_COMPUTE_TYPE = "int8"
WHISPER          = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type=STT_COMPUTE_TYPE)

LLM = Llama(
    model_path=(
      "C:/Users/tahsinsoyak/.lmstudio/models/hugging-quants/"
      "Llama-3.2-1B-Instruct-Q8_0-GGUF/"
      "llama-3.2-1b-instruct-q8_0.gguf"
    ),
    n_ctx=512,
    n_threads=4
)

SYSTEM_PROMPT = "You are a professional tour guide in Istanbul. Provide concise and clear answers."

# —— Routes ——
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/voice", methods=["POST"])
def api_voice():
    # 1) Receive WebM audio blob from browser
    f = request.files["audio_data"]

    # 2) Save WebM to a temp file
    webm_tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    f.save(webm_tmp.name)
    webm_tmp.close()

    # 3) Convert WebM → WAV (16 kHz, mono, 16‑bit PCM) via pydub
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()
    audio = AudioSegment.from_file(webm_tmp.name) \
                       .set_frame_rate(16000) \
                       .set_channels(1) \
                       .set_sample_width(2)
    audio.export(wav_tmp.name, format="wav")
    os.unlink(webm_tmp.name)

    # 4) Run Whisper STT on the WAV
    segments, info = WHISPER.transcribe(
        wav_tmp.name,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}
    )
    os.unlink(wav_tmp.name)

    user_text = "".join(seg.text for seg in segments).strip() or None
    if not user_text:
        return jsonify(error="Could not understand audio"), 400

    # 5) Run Llama to generate guide response
    prompt = SYSTEM_PROMPT + "\nUser: " + user_text + "\nAssistant:"
    resp = LLM(
        prompt,
        max_tokens=60,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nUser:", "\nSystem:"]
    )
    answer = resp["choices"][0]["text"].strip()

    return jsonify(user=user_text, guide=answer)

if __name__ == "__main__":
    app.run(debug=True)
