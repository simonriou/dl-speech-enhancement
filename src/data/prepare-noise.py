import os
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# --------------- Configuration ---------------
SPEECH_DIR = './data/dataset/speech/'
NOISE_PATH = './data/dataset/noise/babble_16k.wav'
OUTPUT_NOISE_DIR = './data/dataset/noisy/'
SAMPLE_RATE = 16000
DURATION = 3  # seconds
TARGET_SNR = 10 # dB
# ----------- End of Configuration ------------

os.makedirs(OUTPUT_NOISE_DIR, exist_ok=True)

# Load cafeteria noise file
noise, _ = librosa.load(NOISE_PATH, sr=SAMPLE_RATE)
noise_len = len(noise)

# Compute target length in samples
target_len = int(SAMPLE_RATE * DURATION)

# For all speech samples
for fname in tqdm(os.listdir(SPEECH_DIR)):
    if not fname.lower().endswith('.flac'):
        continue

    speech_path = os.path.join(SPEECH_DIR, fname)

    # Load speech file
    speech, _ = librosa.load(os.path.join(SPEECH_DIR, fname), sr=SAMPLE_RATE)
    L = len(speech)

    # ---------- 1: Adjust duration ----------
    if L > target_len:
        # Crop randomly (to avoid always taking the start)
        start = random.randint(0, L - target_len)
        speech_trunc = speech[start:start + target_len]
    else:
        # Pad with zeros
        pad = target_len - L
        speech_trunc = np.pad(speech, (pad // 2, pad - pad // 2))

    # Replace the original speech signal by the adjusted one
    # Temporary path
    tmp_path = speech_path + ".tmp"
    sf.write(tmp_path, speech_trunc, SAMPLE_RATE, format='FLAC')
    os.replace(tmp_path, speech_path)

    # ---------- 2: Select random noise segment ----------
    start_idx = random.randint(0, noise_len - target_len)
    noise_seg = noise[start_idx:start_idx + target_len]

    # ---------- 3: Add noise at fixed SNR ----------
    alpha = 10 ** ( - TARGET_SNR / 20)

    noisy = speech_trunc + alpha * noise_seg

    # Normalize to avoid clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    # ---------- 4: Save noisy file ----------
    # Encode noise start index in file name for traceability
    base = os.path.splitext(fname)[0] # e.g., "xx-xxxxxx-xxxx"
    out_name = f"{base}_noisy_{start_idx}.flac"
    out_path = os.path.join(OUTPUT_NOISE_DIR, out_name)
    sf.write(out_path, noisy, SAMPLE_RATE)

    # print(f"Saved: {out_name} (noise start idx {start_idx})")

print("All files processed.")