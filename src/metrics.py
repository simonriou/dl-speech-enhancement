import numpy as np
import os
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from pystoi import stoi

TEST_OUTPUT_DIR = "./data/output/test_output/"
TEST_SPEECH_DIR = "./data/test/clean/"
TEST_NOISY_DIR = "./data/test/noisy/"
SAMPLE_RATE = 16000

# Plot training history
acc_history = np.load('./history/accuracy.npy')
val_acc_history = np.load('./history/val_accuracy.npy')
loss_history = np.load('./history/loss.npy')
val_loss_history = np.load('./history/val_loss.npy')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def si_sdr(reference, estimation, eps=1e-8):
    reference = np.asarray(reference)
    estimation = np.asarray(estimation)

    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)

    scale = np.sum(reference * estimation) / (np.sum(reference ** 2) + eps)

    projection = scale * reference
    noise = estimation - projection

    ratio = np.sum(projection ** 2) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)

def compute_stoi(clean, enhanced, sr):
    return stoi(clean, enhanced, sr, extended=False)

def compute_estoi(clean, enhanced, sr):
    return stoi(clean, enhanced, sr, extended=True) 

sdr_scores = []
noisy_sdr_scores = []
stoi_scores = []
noisy_stoi_scores = []

stoi_scores = []
noisy_stoi_scores = []
estoi_scores = []
noisy_estoi_scores = []

# For every test output file
for f in tqdm(os.listdir(TEST_OUTPUT_DIR), desc="Computing metrics"):
    # Get ID from filename
    file_id = f.split("_")[0]

    # Find the speech file that contains this ID
    speech_file = None
    for sf in os.listdir(TEST_SPEECH_DIR):
        if file_id in sf:
            speech_file = sf
            break

    if speech_file is None:
        print(f"No matching speech file found for ID {file_id}")
        continue

    noisy_file = None
    for nf in os.listdir(TEST_NOISY_DIR):
        if file_id in nf:
            noisy_file = nf
            break

    if noisy_file is None:
        print(f"No matching noisy file found for ID {file_id}")
        continue

    # Load the reference and estimated signals
    reference_path = os.path.join(TEST_SPEECH_DIR, speech_file)
    estimation_path = os.path.join(TEST_OUTPUT_DIR, f)
    noisy_path = os.path.join(TEST_NOISY_DIR, noisy_file)

    reference, _ = librosa.load(reference_path, sr=SAMPLE_RATE)
    estimation, _ = librosa.load(estimation_path, sr=SAMPLE_RATE)
    noisy, _ = librosa.load(noisy_path, sr=SAMPLE_RATE)

    normalized_reference = reference / (np.max(np.abs(reference)) + 1e-8)
    normalized_estimation = estimation / (np.max(np.abs(estimation)) + 1e-8)
    normalized_noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)

    min_len = min(len(normalized_reference), len(normalized_estimation), len(normalized_noisy))
    normalized_reference = normalized_reference[:min_len]
    normalized_estimation = normalized_estimation[:min_len]
    normalized_noisy = normalized_noisy[:min_len]

    sdr_score = si_sdr(normalized_reference, normalized_estimation)
    noisy_sdr_score = si_sdr(normalized_reference, normalized_noisy)
    sdr_scores.append(sdr_score)
    noisy_sdr_scores.append(noisy_sdr_score)

    stoi_score = compute_stoi(normalized_reference, normalized_estimation, sr=SAMPLE_RATE)
    noisy_stoi_score = compute_stoi(normalized_reference, normalized_noisy, sr=SAMPLE_RATE)
    stoi_scores.append(stoi_score)
    noisy_stoi_scores.append(noisy_stoi_score)

    estoi_score = compute_estoi(normalized_reference, normalized_estimation, sr=SAMPLE_RATE)
    noisy_estoi_score = compute_estoi(normalized_reference, normalized_noisy, sr=SAMPLE_RATE)
    estoi_scores.append(estoi_score)
    noisy_estoi_scores.append(noisy_estoi_score)

# Print average score (SDR)
average_score = np.mean(sdr_scores)
median_score = np.median(sdr_scores)
print(f"Average SI-SDR: {average_score:.2f} dB")
print(f"Median SI-SDR: {median_score:.2f} dB")
average_noisy_score = np.mean(noisy_sdr_scores)
median_noisy_score = np.median(noisy_sdr_scores)
print(f"Average Noisy SI-SDR: {average_noisy_score:.2f} dB")
print(f"Median Noisy SI-SDR: {median_noisy_score:.2f} dB")

# Histogram of scores (SDR)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(sdr_scores, bins=20, edgecolor='black')
ax1.set_title("SI-SDR Scores Distribution")
ax1.set_xlabel("SI-SDR (dB)")
ax1.set_ylabel("Frequency")
ax1.grid(axis='y', alpha=0.75)

ax2.hist(noisy_sdr_scores, bins=20, edgecolor='black')
ax2.set_title("Noisy SI-SDR Scores Distribution")
ax2.set_xlabel("SI-SDR (dB)")
ax2.set_ylabel("Frequency")
ax2.grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.show()

# Print average score (STOI)
average_stoi = np.mean(stoi_scores)
median_stoi = np.median(stoi_scores)
print(f"Average STOI: {average_stoi:.4f}")
print(f"Median STOI: {median_stoi:.4f}")
average_noisy_stoi = np.mean(noisy_stoi_scores)
median_noisy_stoi = np.median(noisy_stoi_scores)
print(f"Average Noisy STOI: {average_noisy_stoi:.4f}")
print(f"Median Noisy STOI: {median_noisy_stoi:.4f}")

# Histogram of scores (STOI)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(stoi_scores, bins=20, edgecolor='black')
ax1.set_title("STOI Scores Distribution")
ax1.set_xlabel("STOI")
ax1.set_ylabel("Frequency")
ax1.grid(axis='y', alpha=0.75)

ax2.hist(noisy_stoi_scores, bins=20, edgecolor='black')
ax2.set_title("Noisy STOI Scores Distribution")
ax2.set_xlabel("STOI")
ax2.set_ylabel("Frequency")
ax2.grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.show()