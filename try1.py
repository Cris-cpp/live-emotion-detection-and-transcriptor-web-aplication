import streamlit as st
import numpy as np
from scipy.io import wavfile as wav
import noisereduce as nr
import librosa as lb
import joblib
import os
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from streamlit_mic_recorder import mic_recorder  # NEW IMPORT

# Load the emotion classification model
def load_model():
    return joblib.load("random_forest.pkl")

# Transcribe audio using Whisper
def transcribe(file_path):
    processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small.en")

    audio_input, sample_rate = lb.load(file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=448, num_beams=5)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Extract MFCC features for emotion classification
def extract_features(audio_path):
    data, sr = lb.load(audio_path, sr=None, mono=True, offset=1.0, duration=10)
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

# Streamlit App
def main():
    st.title("🎙️ Emotion Recognition App")

    # Record or upload audio
    audio_bytes = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="🛑 Stop Recording", key="recorder")

    # Debugging: Check audio_bytes type
    st.write(f"🔍 Audio type: {type(audio_bytes)}")
    if audio_bytes:
        st.write(f"📏 Audio length: {len(audio_bytes)} bytes")

        # Save recorded audio
        file_path = "recorded_audio.wav"
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        
        # Ensure correct format
        try:
            audio_data, sample_rate = sf.read(file_path)
            sf.write(file_path, audio_data, sample_rate, format="WAV")
        except Exception as e:
            st.error(f"⚠️ Audio processing error: {e}")
            return
        
        # Play recorded audio
        st.audio(file_path, format="audio/wav")

        # Process and predict
        transcription = transcribe(file_path)
        predicted_emotion = emotion(file_path)

        # Display results
        st.write(f"📝 Transcription: **{transcription}**")
        st.write(f"😊 Predicted Emotion: **{predicted_emotion}**")

# Run the app
if __name__ == "__main__":
    main()
