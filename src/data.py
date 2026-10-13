"""Data handling: inspection (Task 1), splitting (Task 2), preprocessing (Task 3)."""

import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from . import config

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------
# Task 1 - inspect before you touch anything
# --------------------------------------------------------------------------
def list_images(data_root, split, cls):
    """Sorted full paths of every image in <data_root>/<split>/<cls>/."""
    folder = os.path.join(data_root, split, cls)
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def count_images(data_root, splits=("train", "test")):
    """Return {split: {class_name: count}} (Task 1a)."""
    return {
        split: {cls: len(list_images(data_root, split, cls)) for cls in config.CLASS_NAMES}
        for split in splits
    }


def sample_image_info(data_root, n=5, split="train"):
    """Return [(class, filename, (w, h), mode)] for n images per class (Task 1b)."""
    rows = []
    for cls in config.CLASS_NAMES:
        for path in list_images(data_root, split, cls)[:n]:
            with Image.open(path) as im:
                rows.append((cls, os.path.basename(path), im.size, im.mode))
    return rows


def dataset_diversity(data_root, split="train"):
    """Scan every train image's size and mode -> (set of sizes, Counter of modes)."""
    from collections import Counter

    sizes, modes = set(), Counter()
    for cls in config.CLASS_NAMES:
        for path in list_images(data_root, split, cls):
            with Image.open(path) as im:
                sizes.add(im.size)
                modes[im.mode] += 1
    return sizes, modes


def imbalance_ratio(counts):
    """PNEUMONIA / NORMAL for the train split (Task 1d)."""
    n_normal = counts["train"]["NORMAL"]
    n_pneumonia = counts["train"]["PNEUMONIA"]
    return n_normal, n_pneumonia, n_pneumonia / n_normal


# --------------------------------------------------------------------------
# Task 2 - proper stratified train / validation split
# --------------------------------------------------------------------------
def collect_paths_and_labels(data_root, split):
    """Parallel lists of file paths and labels (0=NORMAL, 1=PNEUMONIA)."""
    paths, labels = [], []
    for label, cls in enumerate(config.CLASS_NAMES):
        for path in list_images(data_root, split, cls):
            paths.append(path)
            labels.append(label)
    return paths, labels


def stratified_split(paths, labels, val_fraction=config.VAL_FRACTION, seed=config.SEED):
    """85/15 split that preserves the class ratio in both parts.

    The provided val/ folder (16 images) is deliberately ignored.
    """
    return train_test_split(
        paths, labels, test_size=val_fraction, stratify=labels, random_state=seed
    )


# --------------------------------------------------------------------------
# Task 3 - preprocessing pipeline
# --------------------------------------------------------------------------
def load_and_preprocess(path, img_size=config.IMG_SIZE, augment=False, rng=None):
    """Load one image -> float32 array of shape (img_size, img_size, 1) in [0, 1].

    Order of operations:
      1. convert to grayscale ("L")           -> one consistent colour mode
      2. resize to (img_size, img_size)       -> constant tensor shape
      3. (training only) augment              -> flip, rotation, zoom, brightness
      4. scale to [0, 1]                      -> well-behaved gradients
    """
    with Image.open(path) as im:
        im = im.convert("L").resize((img_size, img_size), Image.BILINEAR)

    if augment:
        if rng is None:
            rng = np.random.default_rng()
        fill = int(np.asarray(im).mean())  # fill empty corners with mean grey

        # random horizontal flip (50 %)
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        # random rotation in [-15, +15] degrees
        im = im.rotate(float(rng.uniform(-15, 15)), resample=Image.BILINEAR, fillcolor=fill)

        # random zoom in [0.9, 1.1] about the image centre
        z = float(rng.uniform(0.9, 1.1))
        c = img_size / 2.0
        im = im.transform(
            (img_size, img_size),
            Image.AFFINE,
            (1 / z, 0, c - c / z, 0, 1 / z, c - c / z),
            resample=Image.BILINEAR,
            fillcolor=fill,
        )

    arr = np.asarray(im, dtype=np.float32) / 255.0

    if augment:
        # slight brightness variation, 0.9x - 1.1x, clipped back into [0, 1]
        arr = np.clip(arr * float(rng.uniform(0.9, 1.1)), 0.0, 1.0)

    return arr[..., np.newaxis].astype(np.float32)


def build_dataset(
    paths,
    labels,
    augment,
    img_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    seed=config.SEED,
):
    """tf.data.Dataset of (image batch, label batch).

    Augmentation and shuffling are applied to the training set ONLY. Val/test
    datasets keep their original order, so predictions line up with labels.
    """
    import tensorflow as tf  # imported lazily so Task 1 works without TensorFlow

    paths = list(paths)
    labels = list(labels)

    def generator():
        rng = np.random.default_rng(seed) if not augment else np.random.default_rng()
        order = np.arange(len(paths))
        if augment:
            rng.shuffle(order)  # fresh shuffle every epoch
        for i in order:
            yield load_and_preprocess(paths[i], img_size, augment, rng), np.int32(labels[i])

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_labels):
    """Balanced class weights from the ORIGINAL (un-augmented) label counts.

    Augmentation creates random variants on the fly but does not change how
    common each disease state really is, so the weights must reflect the true
    distribution - never counts that include augmented copies.
    """
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=np.array(train_labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}
