"""Synthetic stand-in for the Kaggle chest X-ray dataset.

It reproduces the same folder layout and the same real-world quirks
(imbalanced classes, varying image sizes, mixed grayscale/RGB files, a tiny
val/ folder) so the whole pipeline can be developed, tested and run in CI
without downloading 1.2 GB. It is NOT medical data and the resulting numbers
say nothing about real-world performance.
"""

import os

import numpy as np
from PIL import Image


def _box_blur(arr, k=5):
    """Fast box blur using an integral image (no scipy needed)."""
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    h, w = arr.shape
    return (
        csum[k:k + h, k:k + w]
        - csum[0:h, k:k + w]
        - csum[k:k + h, 0:w]
        + csum[0:h, 0:w]
    ) / (k * k)


def _make_xray(rng, label, size):
    img = _box_blur(rng.normal(90, 20, size=(size, size)), k=5)
    yy, xx = np.ogrid[:size, :size]

    # Some NORMAL images get a faint bright blob too, so the task is not trivial.
    if rng.random() < 0.2:
        cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
        r = rng.integers(size // 12, size // 6)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] += rng.normal(15, 6)

    if label == "PNEUMONIA":
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
            r = rng.integers(size // 12, size // 6)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] += rng.normal(22, 8)

    return np.clip(img, 0, 255).astype(np.uint8)


def _build_split(data_root, rng, split, n_normal, n_pneumonia):
    for label, n in [("NORMAL", n_normal), ("PNEUMONIA", n_pneumonia)]:
        out_dir = os.path.join(data_root, split, label)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n):
            size = int(rng.integers(120, 220))
            arr = _make_xray(rng, label, size)
            as_rgb = rng.random() < 0.35
            im = Image.fromarray(arr).convert("RGB" if as_rgb else "L")
            im.save(os.path.join(out_dir, f"{label.lower()}_{i:04d}.png"))


def generate_synthetic_dataset(data_root, scale=1.0, seed=42):
    """Create the synthetic dataset under `data_root` (skipped if it exists).

    scale: multiplies the image counts (use e.g. 0.3 for a very fast smoke test).
    """
    if os.path.isdir(data_root):
        return data_root
    rng = np.random.default_rng(seed)

    def n(x):
        return max(2, int(round(x * scale)))

    _build_split(data_root, rng, "train", n(110), n(330))   # ~3x imbalance
    _build_split(data_root, rng, "test", n(45), n(75))
    _build_split(data_root, rng, "val", 8, 8)               # tiny, like the real one
    return data_root
