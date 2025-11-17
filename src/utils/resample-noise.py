import os
import librosa
import tqdm
import soundfile as sf

# --------------- Configuration ---------------
NOISE_DIR = './data/dataset/noise/babble/'
TARGET_SR = 16000
# ----------- End of Configuration ------------

# Recursively, for all files in all subfolders of NOISE_DIR, resample if needed
for root, _, files in tqdm.tqdm(os.walk(NOISE_DIR), desc="Processing folders"):
    for fname in tqdm.tqdm(files, desc="Processing files", leave=False):
        if not fname.lower().endswith('.wav'):
            continue

        path = os.path.join(root, fname)
        y, sr = librosa.load(path, sr=None)

        if sr != TARGET_SR:
            print(f"Resampling {path} from {sr} Hz to {TARGET_SR} Hz...")
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sf.write(path, y_resampled, TARGET_SR)
            print(f"Resampled and saved {path}.")
        else:
            print(f"{path} is already at {TARGET_SR} Hz, skipping.")
        
