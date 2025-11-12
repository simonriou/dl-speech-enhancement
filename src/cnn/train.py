from cnn import build_cnn_mask_model
from data import SpectrogramDataGenerator

freq_bins = 513 # NFFT/2 + 1
n_frames = 188 # 
input_shape = (freq_bins, n_frames, 1)
batch_size = 8
epochs = 15

# Build model
model = build_cnn_mask_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Create data generators
train_gen = SpectrogramDataGenerator(
    features_dir='./data/train/features',
    labels_dir='./data/train/labels',
    batch_size=batch_size,
    shuffle=True
)

# Train
model.fit(
    train_gen,
    epochs=epochs
)

# Save model
model.save('.models/model2.h5')
model.save('./model/model2.keras')

# Model 1: (Trained on development set)
# - 2x Conv2D 32 + MaxPool(1,2)
# - 2x Conv2D 64 + MaxPool(1,2)
# - 2x Conv2D 128
# - Upsample(1,2) + Conv2D 64
# - Upsample(1,2) + Conv2D 32
# - Conv2D 1 (sigmoid)
# Trained for 15 epochs, batch size 8
# No normalization before making features, nor before testing
# No BatchNorm layers

# Model 2: (Trained on development set)
# - Same as Model 1, except
# - MaxPools are (2, 2) instead of (1, 2)
# - Upsamples are (2, 2) instead of (1, 2)
# - Added crop to match dimensions
# - Adding BatchNorm after each Conv2D
# - Added normalization before making features, and before testing
# Trained for 15 epochs, batch size 8