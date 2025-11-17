import os
import numpy as np
from tensorflow.keras.utils import Sequence

class SpectrogramDataGenerator(Sequence):
    def __init__(
        self,
        features_dir,
        labels_dir,
        batch_size=8,
        shuffle=True,
        validation_split=0.0,
        subset=None
    ):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        # List all feature files
        self.files = sorted([
            f for f in os.listdir(features_dir)
            if f.endswith('.npy')
        ])

        # --- validation split ---
        if validation_split > 0.0:
            if subset not in {"training", "validation"}:
                raise ValueError(
                    "subset must be 'training' or 'validation' "
                    "when validation_split > 0."
                )

            split_idx = int(len(self.files) * (1 - validation_split))

            if subset == "training":
                self.files = self.files[:split_idx]
            else:
                self.files = self.files[split_idx:]

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.files[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = [], []

        for fname in batch_files:
            feat_path = os.path.join(self.features_dir, fname)
            label_fname = fname.split('_')[0] + '_ibm.npy'
            label_path = os.path.join(self.labels_dir, label_fname)

            feat = np.load(feat_path)
            label = np.load(label_path)

            # Add channel dimension
            feat = np.expand_dims(feat, axis=-1)
            label = np.expand_dims(label, axis=-1)

            X.append(feat)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)