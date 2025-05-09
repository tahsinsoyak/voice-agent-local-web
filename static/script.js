let recorder, audioChunks;
const btn       = document.getElementById("record");
const userSpan  = document.getElementById("user");
const guideSpan = document.getElementById("guide");

btn.onclick = async () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    btn.textContent = "🎤 Record";
  } else {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => audioChunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/wav" });
      const fd   = new FormData();
      fd.append("audio_data", blob, "rec.wav");

      const res = await fetch("/api/voice", { method: "POST", body: fd });
      const j   = await res.json();
      if (res.ok) {
        userSpan.textContent  = j.user;
        guideSpan.textContent = j.guide;
        // speak via browser TTS
        const u = new SpeechSynthesisUtterance(j.guide);
        u.lang = "tr-TR";
        speechSynthesis.speak(u);
      } else {
        guideSpan.textContent = j.error || "Error";
      }
    };
    recorder.start();
    btn.textContent = "◼️ Stop";
  }
};
