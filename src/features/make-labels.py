import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
SPEECH_DIR = "./data/dataset/speech/"
NOISE_DIR = "./data/dataset/noise/"
NOISY_DIR = "./data/dataset/noisy/"
IBM_OUTPUT_DIR = "./data/train/labels/"

os.makedirs(IBM_OUTPUT_DIR, exist_ok=True)

def compute_IBM(signal, noise, n_fft=1024, hop_length=None, win_length=None):
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Normalize inputs
    EPSILON = 1e-10
    signal = signal / (np.max(np.abs(signal)) + EPSILON)
    noise = noise / (np.max(np.abs(noise)) + EPSILON)

    # Compute STFTs
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    N = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Power spectrograms
    S_power = np.abs(S) ** 2
    N_power = np.abs(N) ** 2

    # IBM
    ibm = (S_power > N_power).astype(np.float32)

    return ibm

# List all noisy files
noisy_files = os.listdir(NOISY_DIR)

for fname in tqdm(os.listdir(SPEECH_DIR), desc="Processing files"):
    if not fname.lower().endswith('.flac'):
        continue

    # Get the ID
    speech_id = os.path.splitext(os.path.basename(fname))[0]

    # Find the corresponding noisy file
    noisy_file = None
    for f in noisy_files:
        if speech_id in f:
            noisy_file = f
            break

    if noisy_file is None:
        raise FileNotFoundError(f"No noisy file found for speech ID: {speech_id}")

    # Format for the noisy file: {speech_id}_noisy_{start_idx}_{noise_type}_{noise_file}.flac (no _ in noise_file)
    # Extract start index from noisy filename
    start_idx = int(noisy_file.split('_')[2])
    noise_type = noisy_file.split('_')[3]
    noise_file_name = noisy_file.split('_')[4].replace('.flac', '.wav')

    fpath = f"{NOISE_DIR}{noise_type}/{noise_file_name}"

    # print(f"Processing {speech_id}: noise_type={noise_type}, noise_file={noise_file_name}, start_idx={start_idx} | path={fpath}")

    # Load speech signal
    speech_signal, _ = librosa.load(os.path.join(SPEECH_DIR, fname), sr=SAMPLE_RATE)

    # Load noise signal
    noise_signal, _ = librosa.load(fpath, sr=SAMPLE_RATE)

    # Extract noise segment
    noise_segment = noise_signal[start_idx:start_idx + len(speech_signal)]

    # Compute IBM
    ibm = compute_IBM(speech_signal, noise_segment)

    # Save IBM as a numpy file to ./dataset/labels/
    ibm_output_path = os.path.join(IBM_OUTPUT_DIR, f"{speech_id}_ibm.npy")
    np.save(ibm_output_path, ibm)

    # print(f"Saving IBM for {speech_id} to {ibm_output_path}")