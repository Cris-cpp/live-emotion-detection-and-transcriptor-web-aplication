import streamlit as st
import numpy as np
import torch
import wave
import joblib
import librosa as lb
import soundfile as sf
import noisereduce as nr
from scipy.io import wavfile as wav
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

try:
    from streamlit_mic_recorder import mic_recorder  # External dependency
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False  # Handle case where mic_recorder is not installed

# Load models
@st.cache_resource
def load_model():
    return joblib.load("random_forest.pkl")

@st.cache_resource
def load_whisper():
    processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-small.en", torch_dtype=torch.float16, device_map="auto"
    )
    return processor, model

# Transcription
def transcribe(file_path):
    processor, model = load_whisper()
    audio_input, sample_rate = lb.load(file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=448, num_beams=5)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Extract features
def extract_features(audio_path):
    data, sr = lb.load(audio_path, sr=16000, mono=True, duration=7)
    mfcc = lb.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=256, n_mels=40)
    return np.mean(mfcc, axis=1)

# Emotion detection
def emotion(file_path):
    rate, audio_data = wav.read(file_path)

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.8)

    cleaned_path = "cleaned_audio.wav"
    wav.write(cleaned_path, rate, (reduced_noise * 32768).astype(np.int16))

    X_new = extract_features(cleaned_path).reshape(1, -1)

    model = load_model()
    emotion_pred = model.predict(X_new)[0]

    emotion_map = {
        1: "Neutral", 2: "Calm", 3: "Happy", 4: "Angry",
        5: "Excited", 6: "Cheerful", 7: "Disgust", 8: "Surprised"
    }
    return emotion_map.get(emotion_pred, "Unknown")

# Convert raw audio bytes to WAV
def save_audio_bytes(audio_bytes, file_path):
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)  
        wf.setsampwidth(2)  
        wf.setframerate(16000)  
        wf.writeframes(audio_bytes)  

# UI
def main():
    st.title("🎙️ Speech Transcription & Emotion Detection")

    if not MIC_AVAILABLE:
        st.error("⚠️ streamlit-mic-recorder is not installed. Please install it with `pip install streamlit-mic-recorder`.")
        return

    audio_dict = mic_recorder(
        start_prompt="🎤 Click to Record (7s)", 
        stop_prompt="🛑 Recording Stopped",
        key="recorder",
        sampling_rate=16000,  
        time_limit=7  
    )

    if isinstance(audio_dict, dict) and "bytes" in audio_dict:
        audio_bytes = audio_dict["bytes"]
    else:
        audio_bytes = None

    if audio_bytes:
        st.write(f"📏 Audio size: {len(audio_bytes)} bytes")

        file_path = "recorded_audio.wav"
        save_audio_bytes(audio_bytes, file_path)

        st.audio(file_path, format="audio/wav")

        transcription = transcribe(file_path)
        predicted_emotion = emotion(file_path)

        st.write(f"📝 **Transcription:** {transcription}")
        st.write(f"😊 **Predicted Emotion:** {predicted_emotion}")

if __name__ == "__main__":
    main()
