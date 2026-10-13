"""Fast tests for the preprocessing / splitting / modelling logic (run on a tiny synthetic dataset)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data  # noqa: E402
from src.synthetic import generate_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("xray") / "chest_xray")
    generate_synthetic_dataset(root, scale=0.25, seed=0)
    return root


def test_counts_show_class_imbalance(data_root):
    counts = data.count_images(data_root)
    n_normal, n_pneumonia, ratio = data.imbalance_ratio(counts)
    assert n_normal > 0 and n_pneumonia > n_normal
    assert 2.0 < ratio < 4.0


def test_raw_images_are_inconsistent(data_root):
    sizes, modes = data.dataset_diversity(data_root)
    assert len(sizes) > 1          # not pre-resized
    assert set(modes) == {"L", "RGB"}  # mixed colour modes


def test_stratified_split_preserves_ratio(data_root):
    paths, labels = data.collect_paths_and_labels(data_root, "train")
    tr_p, va_p, tr_l, va_l = data.stratified_split(paths, labels)
    assert len(tr_p) + len(va_p) == len(paths)
    assert abs(len(va_p) / len(paths) - config.VAL_FRACTION) < 0.02
    assert abs(sum(tr_l) / len(tr_l) - sum(va_l) / len(va_l)) < 0.03
    assert not set(tr_p) & set(va_p)  # no leakage between train and validation


def test_preprocess_shape_range_and_mode_handling(data_root):
    paths, _ = data.collect_paths_and_labels(data_root, "train")
    for p in paths[:20]:  # covers both RGB and grayscale files, many sizes
        arr = data.load_and_preprocess(p)
        assert arr.shape == (config.IMG_SIZE, config.IMG_SIZE, 1)
        assert arr.dtype == np.float32
        assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_augmentation_changes_image_but_stays_valid(data_root):
    paths, _ = data.collect_paths_and_labels(data_root, "train")
    rng = np.random.default_rng(1)
    plain = data.load_and_preprocess(paths[0], augment=False)
    aug = data.load_and_preprocess(paths[0], augment=True, rng=rng)
    assert aug.shape == plain.shape
    assert 0.0 <= aug.min() and aug.max() <= 1.0
    assert not np.allclose(aug, plain)
    # no augmentation => deterministic
    assert np.array_equal(plain, data.load_and_preprocess(paths[0], augment=False))


def test_class_weights_favour_minority_class():
    labels = [0] * 100 + [1] * 300
    w = data.compute_class_weights(labels)
    assert w[0] > w[1]
    assert w[0] == pytest.approx(2.0)
    assert w[1] == pytest.approx(2 / 3)


def test_dataset_batches_and_model_output(data_root):
    from src.model import build_cnn

    paths, labels = data.collect_paths_and_labels(data_root, "test")
    ds = data.build_dataset(paths, labels, augment=False, batch_size=4)
    imgs, lbls = next(iter(ds))
    assert imgs.shape == (4, config.IMG_SIZE, config.IMG_SIZE, 1)

    model = build_cnn()
    out = model.predict(imgs, verbose=0)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()
    n_blocks = sum(1 for layer in model.layers if layer.__class__.__name__ == "Conv2D")
    assert n_blocks >= 3
