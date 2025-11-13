from cnn import build_cnn_mask_model
import os
from data import SpectrogramDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

os.makedirs('./checkpoints/', exist_ok=True)
os.makedirs('./models/', exist_ok=True)

freq_bins = 513  # NFFT/2 + 1
n_frames = 188
input_shape = (freq_bins, n_frames, 1)
batch_size = 8
epochs = 15

# Build model
model = build_cnn_mask_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Create data generators
train_gen = SpectrogramDataGenerator(
    features_dir='./data/train/features/',
    labels_dir='./data/train/labels/',
    batch_size=batch_size,
    shuffle=True
)

# Callback to save a checkpoint at each epoch
checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/model_epoch_{epoch:02d}.h5',  # Save each epoch separately
    save_weights_only=False,  # Save the full model
    save_freq='epoch',
    verbose=1
)

# Train
model.fit(
    train_gen,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

# Save final model
model.save('./models/model3.keras')

# Model 1: (Trained on development set, using only babble_16k.wav as noise)
# - 2x Conv2D 32 + MaxPool(1,2)
# - 2x Conv2D 64 + MaxPool(1,2)
# - 2x Conv2D 128
# - Upsample(1,2) + Conv2D 64
# - Upsample(1,2) + Conv2D 32
# - Conv2D 1 (sigmoid)
# Trained for 15 epochs, batch size 8
# No normalization before making features, nor before testing
# No BatchNorm layers

# Model 2: (Trained on development set, using only babble_16k.wav as noise)
# - Same as Model 1, except
# - MaxPools are (2, 2) instead of (1, 2)
# - Upsamples are (2, 2) instead of (1, 2)
# - Added crop to match dimensions
# - Adding BatchNorm after each Conv2D
# - Added normalization before making features, and before testing
# Trained for 15 epochs, batch size 8

# Model 3: (Trained on development set, with different types of noises - from ambient-hospital)
# Same as Model 2