import streamlit as st
import numpy as np
import os
import torch
import joblib
import librosa as lb
import noisereduce as nr
import sounddevice as sd
from scipy.io import wavfile as wav
from scipy.io.wavfile import write
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Load emotion classification model
@st.cache_resource
def load_model():
    return joblib.load("random_forest.pkl")

# Load Whisper model for speech-to-text
@st.cache_resource
def load_whisper():
    processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-small.en", torch_dtype=torch.float16, device_map="auto"
    )
    return processor, model

# Record audio for 7 seconds at 16kHz
def listen():
    samplerate = 16000  
    duration = 7  
    st.write("🎤 Recording...")
    
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    
    st.write("✅ Recording finished.")
    file_path = "recorded_audio.wav"
    write(file_path, samplerate, audio_data)  # Save recorded audio

    return file_path

# Transcribe speech using Whisper
def transcribe(file_path):
    processor, model = load_whisper()
    audio_input, sample_rate = lb.load(file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=448, num_beams=5)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Extract MFCC features for emotion classification
def extract_features(audio_path):
    data, sr = lb.load(audio_path, sr=16000, mono=True, duration=7)
    mfcc = lb.feature.mfcc(y=data, sr=sr, n_mfcc=20, hop_length=256, n_mels=40)
    return np.mean(mfcc, axis=1)

# Predict emotion from audio
def emotion(file_path):
    rate, audio_data = wav.read(file_path)

    # Normalize if needed
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    # Noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=rate, prop_decrease=0.8)

    # Save cleaned audio
    cleaned_path = "cleaned_audio.wav"
    wav.write(cleaned_path, rate, (reduced_noise * 32768).astype(np.int16))

    # Extract features
    X_new = extract_features(cleaned_path).reshape(1, -1)

    # Load model and predict
    model = load_model()
    emotion_pred = model.predict(X_new)[0]

    # Emotion mapping
    emotion_map = {
        1: "Neutral", 2: "Calm", 3: "Happy", 4: "Angry",
        5: "Excited", 6: "Cheerful", 7: "Disgust", 8: "Surprised"
    }
    return emotion_map.get(emotion_pred, "Unknown")

# Streamlit UI
def main():
    st.title("🎙️ Live Speech Transcription & Emotion Recognition")

    if st.button("🎤 Record (7s) and Analyze"):
        file_path = listen()
        
        st.audio(file_path, format="audio/wav")  # Play the recorded audio
        
        st.write("📝 **Processing...**")
        
        transcription = transcribe(file_path)
        predicted_emotion = emotion(file_path)
        
        st.write(f"📝 **Transcription:** {transcription}")
        st.write(f"😊 **Predicted Emotion:** {predicted_emotion}")

if __name__ == "__main__":
    main()
