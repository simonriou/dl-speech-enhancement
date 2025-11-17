import os
import torch
import numpy as np
import librosa
import soundfile as sf
from cnn import build_cnn_mask_model
from tqdm import tqdm

MODEL_PATH = "./models/model_final.pt"
TEST_AUDIO_DIR = "./data/test/noisy"
OUTPUT_DIR = "./data/output/test_output/"
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_TEST = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = build_cnn_mask_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
os.makedirs(OUTPUT_DIR, exist_ok=True)

def audio_to_spectrogram(filepath):
    y, _ = librosa.load(filepath, sr=SR)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = np.abs(S), np.angle(S)
    mag = mag[np.newaxis, np.newaxis, :, :]  # (1, C, F, T)
    return mag, phase, y.shape[0]

def spectrogram_to_audio(mag, phase):
    S = mag * np.exp(1j * phase)
    y = librosa.istft(S, hop_length=HOP_LENGTH)
    return y

test_files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.lower().endswith('.flac')]
if SAMPLE_TEST:
    test_files = test_files[:max(1, len(test_files) // 10)]

for filename in tqdm(test_files, desc="Processing audio files"):
    path = os.path.join(TEST_AUDIO_DIR, filename)
    mag, phase, orig_len = audio_to_spectrogram(path)

    # Predict mask
    with torch.no_grad():
        mag_tensor = torch.from_numpy(mag).float().to(device)
        pred_mask = model(mag_tensor)[0, 0].cpu().numpy()  # (F, T)

    # Apply mask
    enhanced_mag = np.abs(mag[0, 0]) * pred_mask

    # Reconstruct waveform
    enhanced = spectrogram_to_audio(enhanced_mag, phase)
    enhanced = enhanced[:orig_len]

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_enhanced.wav")
    sf.write(out_path, enhanced, SR)

print("All test files processed.")