from cnn import build_cnn_mask_model
from data import SpectrogramDataGenerator

freq_bins = 513
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
    features_dir='./dataset/train/features',
    labels_dir='./dataset/train/labels',
    batch_size=batch_size,
    shuffle=True
)

# Train
model.fit(
    train_gen,
    epochs=epochs
)

# Save model
model.save('cnn_mask_model.h5')
model.save('cnn_mask_model.keras')