import librosa
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
import tempfile

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

try:
    loaded_model = load_model("C:/Users/sanme/Downloads/Model.h5")
except Exception as e:
    st.write("Ошибка при загрузке модели:", e)


def check_audio_file(audio_path):
    st.write("Checking file:", os.path.basename(audio_path))  # Отображаем только имя файла, а не его полный путь

    audio, sr = librosa.load(audio_path, sr=None)  # Загрузка аудиофайла

    segment_duration = 1  # Длительность сегмента в секундах
    segment_length = int(sr * segment_duration)
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]

    data = []
    for segment in segments:  # Вычисление аудио-параметров для каждого сегмента
        chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
        rms = librosa.feature.rms(y=segment)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)

        features = [np.mean(chroma_stft), np.mean(rms), np.mean(spectral_centroid), np.mean(spectral_bandwidth),
                    np.mean(rolloff), np.mean(zero_crossing_rate)]
        features.extend(mfcc_mean)

        data.append(features)

    columns = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'rolloff',
               'zero_crossing_rate']  # Создание DataFrame
    columns.extend([f'mfcc{i}' for i in range(1, 21)])
    df = pd.DataFrame(data, columns=columns)

    features = df

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = features_scaled.reshape(
        (features_scaled.shape[0], features_scaled.shape[1], 1))  # стандартизации

    predictions = loaded_model.predict(features_scaled)  # Предсказание

    df['PREDICTION'] = predictions

    threshold = 0.4
    positive_count = sum(1 for p in df['PREDICTION'] if p >= threshold)
    negative_count = sum(1 for p in df['PREDICTION'] if p < threshold)

    if positive_count > negative_count:
        st.write("+++Позитивных+++ предсказаний больше")
    elif positive_count < negative_count:
        st.write("--Негативных-- предсказаний больше")
    else:
        st.write("?????Позитивных и негативных предсказаний одинаковое количество")

    st.dataframe(df)


st.title("Анализ аудиофайлов")

uploaded_files = st.file_uploader("Выберите аудиофайлы", type=["mp3", "wav", "ogg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith((".mp3", ".wav", ".ogg")):
            uploaded_file_path = save_uploaded_file(uploaded_file)
            check_audio_file(uploaded_file_path)
