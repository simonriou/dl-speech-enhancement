import os
import numpy as np

FEATURES_DIR = "./data/train/features"
LABELS_DIR = "./data/train/labels"

feature_shapes = []
label_shapes = []

# Check features
for fname in os.listdir(FEATURES_DIR):
    if not fname.lower().endswith('.npy'):
        continue
    path = os.path.join(FEATURES_DIR, fname)
    arr = np.load(path)
    feature_shapes.append(arr.shape)

# Check labels
for fname in os.listdir(LABELS_DIR):
    if not fname.lower().endswith('.npy'):
        continue
    path = os.path.join(LABELS_DIR, fname)
    arr = np.load(path)
    label_shapes.append(arr.shape)

# Convert to sets to detect unique shapes
unique_feature_shapes = set(feature_shapes)
unique_label_shapes = set(label_shapes)

print("Unique feature shapes:", unique_feature_shapes)
print("Unique label shapes:", unique_label_shapes)

# Check consistency between features and labels
if unique_feature_shapes == unique_label_shapes:
    print("All features and labels have matching shapes.")
else:
    print("Shapes mismatch between features and labels!")