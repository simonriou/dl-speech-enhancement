import os
import shutil
from tqdm import tqdm

# --------------------------
# CONFIG
# --------------------------
source_dir = "/Users/simonriou/Documents/Phelma/3A/DL/Project/dl-speech-enhance/dataset/LibriSpeech"  # directory to search recursively
target_dir = "/Users/simonriou/Documents/Phelma/3A/DL/Project/dl-speech-enhance/dataset/speech"  # directory where .flac files will be moved

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)


# --------------------------
# FIND ALL FLAC FILES
# --------------------------
flac_files = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(".flac"):
            flac_files.append(os.path.join(root, file))

print(f"Found {len(flac_files)} .flac files.")

# --------------------------
# MOVE FILES WITH PROGRESS BAR
# --------------------------
for src_path in tqdm(flac_files, desc="Moving .flac files"):
    file = os.path.basename(src_path)
    dst_path = os.path.join(target_dir, file)

    # Handle filename conflicts
    counter = 1
    original_dst_path = dst_path
    while os.path.exists(dst_path):
        name, ext = os.path.splitext(file)
        dst_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
        counter += 1

    # Move the file
    shutil.move(src_path, dst_path)