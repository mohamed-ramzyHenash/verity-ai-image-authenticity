
"""Dataset discovery, splitting and tf.data pipelines."""

import logging
import os
from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_EXTENSIONS,
    RANDOM_STATE,
    SHUFFLE_BUFFER,
    VAL_FRACTION,
)
from ml.preprocessing import decode_image, prepare_image

log = logging.getLogger(__name__)
AUTOTUNE = tf.data.AUTOTUNE


def collect_image_paths(data_dir, split):
    """Collect image paths and labels from train or test folders."""

    paths, labels = [], []

    split_dir = os.path.join(data_dir, split)

    for folder, label in CLASS_NAMES.items():
        folder_path = os.path.join(split_dir, folder)

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Missing class folder: {folder_path}"
            )

        files = [
            f
            for ext in IMAGE_EXTENSIONS
            for f in glob(os.path.join(folder_path, ext))
        ]

        log.info("%s/%s: %d images", split, folder, len(files))

        paths.extend(files)
        labels.extend([label] * len(files))

    return np.array(paths), np.array(labels, dtype=np.int64)


def split_data(paths, labels):
    """Split training data into training and validation sets."""

    return train_test_split(
        paths,
        labels,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def _load(path, label, training):
    image = decode_image(tf.io.read_file(path))
    return prepare_image(image, training=training), label


def make_dataset(paths, labels, training=False):
    """Build an optimized TensorFlow dataset."""

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            SHUFFLE_BUFFER,
            seed=RANDOM_STATE
        )

    ds = ds.map(
        lambda p, l: _load(p, l, training),
        num_parallel_calls=AUTOTUNE
    )

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_datasets(data_dir=DATA_DIR):
    """Build train, validation and test datasets."""

    # Load the original training and test folders.
    train_paths, train_labels = collect_image_paths(
        data_dir, "train"
    )

    test_paths, test_labels = collect_image_paths(
        data_dir, "test"
    )

    # Split training data into train and validation.
    xtr, xv, ytr, yv = split_data(
        train_paths, train_labels
    )

    log.info(
        "Train: %d | Validation: %d | Test: %d",
        len(xtr), len(xv), len(test_paths)
    )

    return (
        make_dataset(xtr, ytr, training=True),
        make_dataset(xv, yv),
        make_dataset(test_paths, test_labels),
        test_labels,
    )
