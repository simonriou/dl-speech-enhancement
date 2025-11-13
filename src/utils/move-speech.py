import os
import shutil
import re

# Directories
dir_a = './data/dataset/speech'
dir_b = './data/dataset/noisy'
dir_c = './data/dataset/temp_speech'

# Make sure dir C exists
os.makedirs(dir_c, exist_ok=True)

# Get all IDs from dir B
ids_in_b = set()
pattern_b = re.compile(r'^(.*)_noisy_\d+\.flac$')  # captures the id part

for filename in os.listdir(dir_b):
    match = pattern_b.match(filename)
    if match:
        ids_in_b.add(match.group(1))

# Move files from dir A to dir C if their ID is in dir B
for filename in os.listdir(dir_a):
    if filename.endswith('.flac'):
        file_id = filename[:-5]  # remove '.flac'
        if file_id in ids_in_b:
            src_path = os.path.join(dir_a, filename)
            dst_path = os.path.join(dir_c, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {dir_c}")