from cnn import build_cnn_mask_model
import os
from data import SpectrogramDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np

os.makedirs('./checkpoints/', exist_ok=True)
os.makedirs('./models/', exist_ok=True)
os.makedirs('./history/', exist_ok=True)

# GPU configuration (Metal)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

freq_bins = 513
n_frames = 188
input_shape = (freq_bins, n_frames, 1)
batch_size = 8
epochs = 15
validation_split = 0.2  # 20% val

# Build model
model = build_cnn_mask_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Data generators with proper split ---
train_gen = SpectrogramDataGenerator(
    features_dir='./data/train/features/',
    labels_dir='./data/train/labels/',
    batch_size=batch_size,
    shuffle=True,
    validation_split=validation_split,
    subset='training'
)

val_gen = SpectrogramDataGenerator(
    features_dir='./data/train/features/',
    labels_dir='./data/train/labels/',
    batch_size=batch_size,
    shuffle=False,
    validation_split=validation_split,
    subset='validation'
)

# Save a checkpoint after each epoch
checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/model_epoch_{epoch:02d}.h5',
    save_weights_only=False,
    save_freq='epoch',
    verbose=1
)

# --- Train and record history ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint_callback]
)

# Save final model
model.save('./models/model3.keras')

# --- Save training curves ---
np.save('./history/loss.npy', history.history['loss'])
np.save('./history/val_loss.npy', history.history['val_loss'])

np.save('./history/accuracy.npy', history.history['accuracy'])
np.save('./history/val_accuracy.npy', history.history['val_accuracy'])

print("Saved training history in ./history/")