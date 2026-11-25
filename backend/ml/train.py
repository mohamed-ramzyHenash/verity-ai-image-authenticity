
"""Train the AI-generated image classifier.

Usage (from backend/):
    python -m ml.train
"""

import argparse
import logging

from tensorflow import keras # type: ignore

from config import DATA_DIR, EPOCHS, MODEL_PATH, MODELS_DIR
from ml.data import build_datasets
from ml.model import build_model, compile_model

log = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Root folder containing train and test directories"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    args = parser.parse_args()

    # Create the models directory.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training, validation and test datasets.
    train_ds, val_ds, test_ds, test_labels = build_datasets(
        args.data_dir
    )

    # Build and compile the model.
    model = compile_model(build_model())

    model.summary()

    # Training callbacks.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Train the model.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    log.info("Training completed.")

    # Load the best checkpoint.
    best_model = keras.models.load_model(MODEL_PATH)

    # Evaluate on the untouched test dataset.
    test_results = best_model.evaluate(
        test_ds,
        return_dict=True,
        verbose=1
    )

    log.info("Test Results:")

    for metric, value in test_results.items():
        log.info("%s: %.4f", metric, value)

    log.info("Best model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    main()
