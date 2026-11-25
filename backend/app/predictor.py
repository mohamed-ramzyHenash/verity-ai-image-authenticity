
"""Loads the trained model once and runs single-image inference."""

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow import keras # type: ignore

from config import LABEL_TO_NAME, MODEL_PATH
from ml.preprocessing import decode_image, prepare_image

log = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: keras.Model | None = None
        self.threshold = 0.5

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            log.warning(
                "Model file not found at %s - run `python -m ml.train` first.",
                self.model_path,
            )
            return

        self.model = keras.models.load_model(self.model_path)
        log.info("Model loaded from %s", self.model_path)

    def predict(self, image_bytes: bytes) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # Decode and preprocess the uploaded image
        image = decode_image(tf.constant(image_bytes))
        image = prepare_image(image)

        # Add batch dimension
        batch = tf.expand_dims(image, axis=0)

        # Predict probability of REAL class
        p_real = float(self.model.predict(batch, verbose=0)[0][0])

        # Apply classification threshold
        label_id = int(p_real >= self.threshold)

        return {
            "label": LABEL_TO_NAME[label_id],
            "is_real": bool(label_id),
            "confidence": p_real if label_id else 1.0 - p_real,
            "probability_real": p_real,
            "probability_ai": 1.0 - p_real,
        }


predictor = Predictor()
