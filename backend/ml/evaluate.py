
"""Evaluate the saved model on the test split.
Usage: python -m ml.evaluate
"""
import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras  # type: ignore

from config import (
    CONFUSION_MATRIX_PATH,
    DATA_DIR,
    LABEL_TO_NAME,
    METRICS_PATH,
    MODEL_PATH,
)
from ml.data import build_datasets

log = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    # Load the original test dataset
    _, _, test_ds, y_test = build_datasets(args.data_dir)

    # Load the best saved model
    model = keras.models.load_model(MODEL_PATH)

    # Evaluate overall performance
    results = model.evaluate(test_ds, verbose=1)

    loss, accuracy, precision, recall = results

    # Generate predictions
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = (probabilities >= 0.5).astype(int).ravel()

    # Keep class order consistent with label mapping
    labels = sorted(LABEL_TO_NAME)
    names = [LABEL_TO_NAME[i] for i in labels]

    # Generate detailed classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=names,
            zero_division=0,
        )
    )

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("Confusion Matrix:")
    print(cm)

    # Save all metrics
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "threshold": 0.5,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    # Plot and save confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=names,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("ResNet50 - Test Confusion Matrix")
    plt.tight_layout()
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    log.info("Evaluation completed.")
    log.info("Metrics saved to: %s", METRICS_PATH)
    log.info("Confusion matrix saved to: %s", CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
