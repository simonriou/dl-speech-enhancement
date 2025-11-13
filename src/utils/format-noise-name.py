import os

directory = './data/dataset/noise/ten-places/'

# List all files (excluding subdirectories)
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Sort to ensure consistent ordering
files.sort()

# Rename files
for i, filename in enumerate(files, start=1):
    name, ext = os.path.splitext(filename)
    new_name = f"{i}{ext}"
    src = os.path.join(directory, filename)
    dst = os.path.join(directory, new_name)
    os.rename(src, dst)
    print(f"Renamed {filename} → {new_name}")