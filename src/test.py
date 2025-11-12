import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

# ============================
# Configuration
# ============================
MODEL_PATH = "cnn_mask_model.h5"
TEST_AUDIO_DIR = "./dataset/test/noisy"
OUTPUT_DIR = "./dataset/output/test_output/"
SR = 16000          # sampling rate
N_FFT = 1024
HOP_LENGTH = 256

# ============================
# Load model
# ============================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Preprocessing function
# ============================
def audio_to_spectrogram(filepath):
    """Load audio file and compute magnitude spectrogram."""
    y, _ = librosa.load(filepath, sr=SR)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(S), np.angle(S)
    mag = mag[np.newaxis, ..., np.newaxis]  # shape (1, freq, time, 1)
    return mag, phase, y.shape[0]

def spectrogram_to_audio(mag, phase):
    """Reconstruct waveform from magnitude and phase."""
    S = mag * np.exp(1j * phase)
    y = librosa.istft(S, hop_length=HOP_LENGTH)
    return y

# ============================
# Test loop
# ============================
test_files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.lower().endswith('.flac')]

print(f"Processing {len(test_files)} test files...")

for filename in tqdm(test_files, desc="Processing audio files"):
    path = os.path.join(TEST_AUDIO_DIR, filename)
    # ∂print(f"Processing {filename}...")

    # Convert to spectrogram
    mag, phase, orig_len = audio_to_spectrogram(path)

    # Predict mask
    pred_mask = model.predict(mag)[0, ..., 0]  # remove batch and channel dims

    # Apply mask to magnitude
    enhanced_mag = np.abs(mag[0, ..., 0]) * pred_mask

    # Reconstruct waveform
    enhanced = spectrogram_to_audio(enhanced_mag, phase)
    enhanced = enhanced[:orig_len]  # ensure same length

    # Save enhanced audio
    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_enhanced.wav")
    sf.write(out_path, enhanced, SR)
    # print(f"Saved enhanced audio to {out_path}")

    # # Optional: visualize
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.amplitude_to_db(enhanced_mag, ref=np.max),
    #                          sr=SR, hop_length=HOP_LENGTH, y_axis='log', x_axis='time')
    # plt.title(f"Enhanced Spectrogram - {filename}")
    # plt.colorbar(format="%+2.0f dB")
    # plt.tight_layout()
    # plt.show()

print("All test files processed.")