"""Plotting helpers. All functions save to a file (if `out_path` is given) and return the figure."""

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks still display figures inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from . import config
from .data import list_images, load_and_preprocess


def _finish(fig, out_path):
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def plot_samples(data_root, n=5, split="train", out_path=None):
    """Task 1c: 2 x n grid, top row NORMAL, bottom row PNEUMONIA (raw, un-resized images)."""
    from PIL import Image

    fig, axes = plt.subplots(2, n, figsize=(14, 6))
    for row, cls in enumerate(config.CLASS_NAMES):
        for col, path in enumerate(list_images(data_root, split, cls)[:n]):
            with Image.open(path) as im:
                ax = axes[row, col]
                ax.imshow(im, cmap="gray" if im.mode == "L" else None)
                ax.set_title(f"{cls}\n{im.size[0]}x{im.size[1]} {im.mode}", fontsize=9)
                ax.axis("off")
    return _finish(fig, out_path)


def plot_class_distribution(counts, out_path=None):
    """Bar chart of class counts per split."""
    fig, axes = plt.subplots(1, len(counts), figsize=(4 * len(counts), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (split, per_class) in zip(axes, counts.items()):
        bars = ax.bar(per_class.keys(), per_class.values(), color=["#4c78a8", "#e45756"])
        ax.bar_label(bars)
        ax.set_title(f"{split}/ class counts")
    return _finish(fig, out_path)


def plot_augmentation(train_paths, val_paths, n=5, out_path=None):
    """Task 3d: augmented training images (top) vs un-augmented validation images (bottom)."""
    rng = np.random.default_rng(config.SEED)
    fig, axes = plt.subplots(2, n, figsize=(12, 5.2))
    for col in range(n):
        aug = load_and_preprocess(train_paths[col], augment=True, rng=rng)[..., 0]
        axes[0, col].imshow(aug, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title("train (augmented)", fontsize=9)
        axes[0, col].axis("off")
        val = load_and_preprocess(val_paths[col], augment=False)[..., 0]
        axes[1, col].imshow(val, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("val (not augmented)", fontsize=9)
        axes[1, col].axis("off")
    return _finish(fig, out_path)


def plot_history(history, out_path=None):
    """Task 5c: train vs validation loss and accuracy side by side."""
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(h["loss"], label="train")
    ax1.plot(h["val_loss"], label="validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(h["accuracy"], label="train")
    ax2.plot(h["val_accuracy"], label="validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    return _finish(fig, out_path)


def plot_confusion_matrix(cm, title="Confusion matrix (test set)", out_path=None):
    """Task 6c."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=config.CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    return _finish(fig, out_path)
