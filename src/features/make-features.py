import os
import numpy as np
import librosa
from tqdm import tqdm

# --------------- Configuration ---------------
NOISY_DIR = './data/dataset/noisy'
FEATURES_DIR = './data/train/features'
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
EPSILON = 1e-10 # Small constant to avoid log(0)
# ---------------------------------------------

os.makedirs(FEATURES_DIR, exist_ok=True)

# Iterate through noisy signals
for fname in tqdm(os.listdir(NOISY_DIR), desc="Extracting features"):
    if not fname.lower().endswith(('.flac', '.wav')):
        continue

    path = os.path.join(NOISY_DIR, fname)

    # Load audio
    y, _ = librosa.load(path, sr=SAMPLE_RATE)

    # Normalize audio
    y = y / (np.max(np.abs(y)) + EPSILON)

    # Compute STFT
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    # Take power spectrogram
    S_power = np.abs(S) ** 2

    # Log-power spectrogram
    log_S = np.log(S_power + EPSILON)

    # Save as .npy
    base = os.path.splitext(fname)[0]
    out_path = os.path.join(FEATURES_DIR, base + '.npy')
    np.save(out_path, log_S)

    # print(f"Saved features for {fname} to {out_path}")

print("Feature extraction completed.")