"""Dataset sanity check (image sizes, modes, corrupted files).   Usage:  python -m ml.explore"""
import os
from collections import Counter

from PIL import Image

from config import CLASS_NAMES, DATA_DIR


def main():
    stats = {"dimensions": Counter(), "modes": Counter(), "formats": Counter()}
    corrupted = []
    for folder in CLASS_NAMES:
        folder_path = os.path.join(DATA_DIR, folder)
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            try:
                with Image.open(path) as img:
                    stats["dimensions"][(folder, img.size)] += 1
                    stats["modes"][img.mode] += 1
                    stats["formats"][img.format] += 1
                    img.verify()
            except Exception:
                corrupted.append(path)

    print("Dimensions per class:", stats["dimensions"].most_common(5))
    print("Modes:", dict(stats["modes"]))
    print("Formats:", dict(stats["formats"]))
    print("Corrupted images:", len(corrupted))


if __name__ == "__main__":
    main()
