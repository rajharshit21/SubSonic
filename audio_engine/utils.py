import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display


def generate_spectrogram(audio_path: str, output_path: str, title: str = "Spectrogram"):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_waveform(audio_path: str, output_path: str, title: str = "Waveform"):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # Example usage
    raw_audio = "data/raw/sample.wav"
    processed_audio = "data/processed/sample_transformed.wav"
    
    os.makedirs("data/processed/analytics", exist_ok=True)

    generate_spectrogram(raw_audio, "data/processed/analytics/raw_spectrogram.png", "Raw Audio Spectrogram")
    generate_spectrogram(processed_audio, "data/processed/analytics/transformed_spectrogram.png", "Transformed Audio Spectrogram")

    generate_waveform(raw_audio, "data/processed/analytics/raw_waveform.png", "Raw Audio Waveform")
    generate_waveform(processed_audio, "data/processed/analytics/transformed_waveform.png", "Transformed Audio Waveform")