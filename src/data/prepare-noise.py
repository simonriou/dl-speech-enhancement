import os
import random
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# --------------- Configuration ---------------
SPEECH_DIR = './data/test/clean/'
NOISE_TYPE = 'babble' # 'babble', 'ten-places' or 'ambient-hospital'
NOISE_DIR = f"./data/dataset/noise/{NOISE_TYPE}/"
OUTPUT_NOISE_DIR = './data/test/noisy/'
SAMPLE_RATE = 16000
DURATION = 3  # seconds
TARGET_SNR = 5 # dB
MAX_NOISE_FILES = 100
EPSILON = 1e-10
# ----------- End of Configuration ------------

os.makedirs(OUTPUT_NOISE_DIR, exist_ok=True)

# Get the list of noise files in NOISE_DIR
noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith('.wav')]
if MAX_NOISE_FILES is not None:
    noise_files = random.sample(noise_files, min(MAX_NOISE_FILES, len(noise_files)))

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

    # ---------- 2: Select random file & segment ----------
    noise_file = random.choice(noise_files)
    noise, _ = librosa.load(NOISE_DIR + noise_file, sr=SAMPLE_RATE)
    noise_len = len(noise)
    start_idx = random.randint(0, noise_len - target_len)
    noise_seg = noise[start_idx:start_idx + target_len]

    # ---------- 3: Add noise at fixed SNR ----------
    signal_power = np.mean(speech_trunc ** 2)
    noise_power = np.mean(noise_seg ** 2)

    alpha = np.sqrt(signal_power / ((noise_power + EPSILON) * (10 ** (TARGET_SNR / 10))))

    noisy = speech_trunc + alpha * noise_seg

    # Normalize to avoid clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    # ---------- 4: Save noisy file ----------
    # Encode noise start index in file name for traceability
    base = os.path.splitext(fname)[0] # e.g., "xx-xxxxxx-xxxx"
    out_name = f"{base}_noisy_{start_idx}_{NOISE_TYPE}_{noise_file.replace('.wav', '')}.flac"
    out_path = os.path.join(OUTPUT_NOISE_DIR, out_name)
    sf.write(out_path, noisy, SAMPLE_RATE)

    # print(f"Saved: {out_name} (noise start idx {start_idx})")

print("All files processed.")