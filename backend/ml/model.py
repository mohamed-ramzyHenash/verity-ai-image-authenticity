"""Model definition: frozen ResNet50 backbone + small classification head."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

from config import IMAGE_SIZE, LEARNING_RATE


def build_model(weights="imagenet") -> keras.Model:
    base = ResNet50(weights=weights, include_top=False,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(Real Images)
    return keras.Model(inputs, outputs)


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
    )
    return model
