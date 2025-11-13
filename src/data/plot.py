import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

SPEECH_DIR = "data/dataset/speech/"
NOISY_DIR = "data/dataset/noisy/"
LABELS_DIR = "data/train/labels/"

FILE_ID = '2035-147961-0001'

# Get the filenames
speech_file = f"{SPEECH_DIR}{FILE_ID}.flac"
noisy_file = None
for fname in os.listdir(NOISY_DIR):
    if FILE_ID in fname:
        noisy_file = f"{NOISY_DIR}{fname}"
        break
ibm = f"{LABELS_DIR}{FILE_ID}_ibm.npy"

# Load the files
speech, sr = librosa.load(speech_file, sr=None)
noisy, _ = librosa.load(noisy_file, sr=None)
label = np.load(ibm)

# Plot the 3 spectrograms
def plot_spectrogram(data, title, sr, hop_length=512):
    plt.figure(figsize=(10, 4))
    S = librosa.stft(data)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_spectrogram(speech, 'Clean Speech Spectrogram', sr)
plot_spectrogram(noisy, 'Noisy Speech Spectrogram', sr)
# Display label as an image
plt.figure(figsize=(10, 4))
plt.imshow(label.T, aspect='auto', origin='lower', cmap='gray_r')
plt.title('Ideal Binary Mask (IBM)')
plt.xlabel('Time Frames')
plt.ylabel('Frequency Bins')
plt.colorbar(label='Mask Value')
plt.tight_layout()
plt.show()