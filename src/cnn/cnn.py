import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_mask_model(input_shape):
    inputs = layers.Input(shape=input_shape) # (freq_bins, time_frames, 1)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(1,2), padding='same')(x) # only pool time, keep frequency unchanged

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(1,2), padding='same')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    # Upsample back to original size
    x = layers.UpSampling2D(size=(1, 2))(x)  # only upsample time, keep frequency unchanged
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(1, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model