"""Image preprocessing shared by training and inference (no train/serve skew)."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input

from config import IMAGE_SIZE

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="data_augmentation",
)


def decode_image(raw_bytes: tf.Tensor) -> tf.Tensor:
    """Decode encoded image bytes into an RGB tensor of unknown size."""
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return image


def prepare_image(image: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Resize -> (augment) -> ResNet50 preprocessing."""
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    if training:
        image = data_augmentation(image, training=True)
    return preprocess_input(image)
