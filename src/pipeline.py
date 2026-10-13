"""End-to-end pipeline: Tasks 1-6 (+ optional bonus experiments).

Usage examples (from the project root):
    python run_pipeline.py                      # real Kaggle data in data/chest_xray_data/
    python run_pipeline.py --synthetic          # fast demo on generated fake data
    python run_pipeline.py --ablation           # also train WITHOUT class weights (Bonus 1)
    python run_pipeline.py --mobilenet          # also try frozen MobileNetV2 (Bonus 2)
"""

import argparse
import json
import os

import numpy as np

from . import config, data, evaluate as ev, visualize as viz
from .synthetic import generate_synthetic_dataset


def _banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def _fmt_report(metrics):
    r = metrics["report"]
    lines = [f"| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls in config.CLASS_NAMES:
        c = r[cls]
        lines.append(
            f"| {cls} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1-score']:.3f} | {int(c['support'])} |"
        )
    lines.append(f"\nOverall accuracy: **{metrics['accuracy']:.3f}**")
    return "\n".join(lines)


def run(data_root, synthetic=False, scale=1.0, epochs=config.MAX_EPOCHS,
        ablation=False, mobilenet=False, results_dir=config.RESULTS_DIR):
    import tensorflow as tf

    from . import model as model_lib
    from .train import train

    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)
    os.makedirs(results_dir, exist_ok=True)
    fig_dir = os.path.join(results_dir, "figures")
    summary = {"dataset": "synthetic" if synthetic else "kaggle"}

    if synthetic:
        generate_synthetic_dataset(data_root, scale=scale, seed=config.SEED)
    if not os.path.isdir(os.path.join(data_root, "train")):
        raise SystemExit(
            f"Dataset not found at {data_root}\n"
            "Download it (see data/README.md) or run with --synthetic for a demo."
        )
    print("TensorFlow:", tf.__version__, "| data root:", data_root)

    # ------------------------------------------------------------------ Task 1
    _banner("Task 1 - Inspect the raw data")
    counts = data.count_images(data_root)
    for split, per_class in counts.items():
        print(f"--- {split} ---")
        for cls, n in per_class.items():
            print(f"  {cls}: {n} images")
    n_normal, n_pneumonia, ratio = data.imbalance_ratio(counts)
    print(f"\nImbalance ratio (PNEUMONIA / NORMAL) in train: {ratio:.2f}x")

    print(f"\n{'Class':<10} {'File':<22} {'Size':<12} Mode")
    for cls, name, size, mode in data.sample_image_info(data_root, n=5):
        print(f"{cls:<10} {name:<22} {str(size):<12} {mode}")
    sizes, modes = data.dataset_diversity(data_root)
    print(f"\nDistinct image sizes in train: {len(sizes)} | colour modes: {dict(modes)}")

    viz.plot_samples(data_root, out_path=os.path.join(fig_dir, "01_samples.png"))
    viz.plot_class_distribution(counts, out_path=os.path.join(fig_dir, "02_class_counts.png"))
    summary["counts"] = counts
    summary["imbalance_ratio"] = ratio
    summary["distinct_sizes"] = len(sizes)
    summary["color_modes"] = dict(modes)

    # ------------------------------------------------------------------ Task 2
    _banner("Task 2 - Stratified 85/15 train/validation split (val/ folder ignored)")
    all_paths, all_labels = data.collect_paths_and_labels(data_root, "train")
    train_paths, val_paths, train_labels, val_labels = data.stratified_split(all_paths, all_labels)
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
    tr_ratio = sum(train_labels) / len(train_labels)
    va_ratio = sum(val_labels) / len(val_labels)
    print(f"PNEUMONIA share  train: {tr_ratio:.3f} | val: {va_ratio:.3f}  (close = stratification works)")

    # ------------------------------------------------------------------ Task 3
    _banner("Task 3 - Preprocessing pipeline and class weights")
    train_ds = data.build_dataset(train_paths, train_labels, augment=True)
    val_ds = data.build_dataset(val_paths, val_labels, augment=False)
    for imgs, lbls in train_ds.take(1):
        print("Batch shape:", imgs.shape,
              "| value range:", round(float(tf.reduce_min(imgs)), 3), "-", round(float(tf.reduce_max(imgs)), 3))
    viz.plot_augmentation(train_paths, val_paths, out_path=os.path.join(fig_dir, "03_augmentation.png"))
    class_weights = data.compute_class_weights(train_labels)
    print("Class weights (NORMAL=0, PNEUMONIA=1):", class_weights)
    summary["class_weights"] = class_weights

    # ------------------------------------------------------------------ Task 4
    _banner("Task 4 - Build the CNN")
    model = model_lib.build_cnn()
    model.summary()

    # ------------------------------------------------------------------ Task 5
    _banner("Task 5 - Train (class weights + early stopping on val_loss)")
    history, seconds = train(model, train_ds, val_ds, class_weights, epochs=epochs)
    epochs_run = len(history.history["loss"])
    best_val = float(min(history.history["val_loss"]))
    print(f"\nStopped after {epochs_run} epochs ({seconds:.0f}s); best val_loss = {best_val:.4f}")
    viz.plot_history(history, out_path=os.path.join(fig_dir, "04_training_curves.png"))
    summary["training"] = {"epochs_run": epochs_run, "best_val_loss": best_val, "seconds": seconds}

    # ------------------------------------------------------------------ Task 6
    _banner("Task 6 - Evaluate on the untouched test set")
    test_paths, test_labels = data.collect_paths_and_labels(data_root, "test")
    test_ds = data.build_dataset(test_paths, test_labels, augment=False)
    _, y_pred = ev.predict(model, test_ds)
    metrics = ev.evaluate(test_labels, y_pred)
    print(metrics["report_text"])
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    baseline = ev.always_pneumonia_baseline(test_labels)
    print(f"(A useless 'always PNEUMONIA' model would already score {baseline:.3f} accuracy.)\n")
    analysis = ev.error_analysis(metrics)
    print(analysis)
    viz.plot_confusion_matrix(metrics["confusion_matrix"], "Weighted CNN - test set",
                              os.path.join(fig_dir, "05_confusion_matrix.png"))
    summary["test_weighted"] = {k: v for k, v in metrics.items() if k != "report_text"}
    summary["always_pneumonia_accuracy"] = baseline

    model.save(os.path.join(results_dir, "chest_xray_cnn.keras"))

    # ------------------------------------------------------------------ Bonus 1
    ablation_metrics = None
    if ablation:
        _banner("Bonus 1 - Same architecture WITHOUT class weights")
        tf.keras.backend.clear_session()
        plain = model_lib.compile_model(tf.keras.models.clone_model(model))  # fresh random weights
        _, secs = train(plain, train_ds, val_ds, class_weight=None, epochs=epochs)
        _, plain_pred = ev.predict(plain, test_ds)
        ablation_metrics = ev.evaluate(test_labels, plain_pred)
        print(ablation_metrics["report_text"])
        viz.plot_confusion_matrix(ablation_metrics["confusion_matrix"], "No class weights - test set",
                                  os.path.join(fig_dir, "06_confusion_matrix_unweighted.png"))
        summary["test_unweighted"] = {k: v for k, v in ablation_metrics.items() if k != "report_text"}

    # ------------------------------------------------------------------ Bonus 2
    mobile_metrics = None
    if mobilenet:
        _banner("Bonus 2 - Frozen MobileNetV2 base + small head")
        try:
            tf.keras.backend.clear_session()
            mobile = model_lib.build_mobilenet()
            _, m_secs = train(mobile, train_ds, val_ds, class_weights, epochs=epochs)
            _, m_pred = ev.predict(mobile, test_ds)
            mobile_metrics = ev.evaluate(test_labels, m_pred)
            print(mobile_metrics["report_text"])
            print(f"Training time - scratch CNN: {seconds:.0f}s | MobileNetV2: {m_secs:.0f}s")
            summary["test_mobilenet"] = {k: v for k, v in mobile_metrics.items() if k != "report_text"}
            summary["mobilenet_seconds"] = m_secs
        except Exception as exc:  # most commonly: ImageNet weights could not be downloaded
            print(f"Skipping MobileNetV2 bonus: {exc}")

    # ------------------------------------------------------------------ Outputs
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results_dir, "writeup.md"), "w") as f:
        f.write(make_writeup(summary, metrics, ablation_metrics, mobile_metrics, analysis))
    print(f"\nDone. Figures, metrics.json, writeup.md and the model are in: {results_dir}")
    return summary


def make_writeup(summary, metrics, ablation_metrics, mobile_metrics, analysis):
    synthetic = summary["dataset"] == "synthetic"
    s = summary
    lines = ["# Writeup", ""]
    if synthetic:
        lines += [
            "> **These numbers come from the synthetic demo dataset, not real X-rays.** "
            "Re-run on the Kaggle data (`python run_pipeline.py`) for results that mean anything.",
            "",
        ]
    lines += [
        "## 1. Final test-set precision / recall / F1 per class",
        "",
        _fmt_report(metrics),
        "",
        f"A model that always predicted PNEUMONIA would already reach "
        f"{s['always_pneumonia_accuracy']:.3f} accuracy, which is why per-class numbers matter.",
        "",
        "```",
        analysis,
        "```",
        "",
        "## 2. Did class weighting change the confusion matrix?",
        "",
    ]
    if ablation_metrics:
        w, u = metrics, ablation_metrics
        lines += [
            "| Model | NORMAL recall | PNEUMONIA recall | False negatives | False positives | Accuracy |",
            "|---|---|---|---|---|---|",
            f"| With class weights | {w['normal_recall']:.3f} | {w['pneumonia_recall']:.3f} | "
            f"{w['false_negatives']} | {w['false_positives']} | {w['accuracy']:.3f} |",
            f"| Without class weights | {u['normal_recall']:.3f} | {u['pneumonia_recall']:.3f} | "
            f"{u['false_negatives']} | {u['false_positives']} | {u['accuracy']:.3f} |",
            "",
            f"Change in NORMAL recall from weighting: {w['normal_recall'] - u['normal_recall']:+.3f}.",
        ]
    else:
        lines.append("Run with `--ablation` to train the unweighted model and fill in this comparison.")
    if mobile_metrics:
        lines += [
            "",
            "**Bonus - MobileNetV2 (frozen) vs from-scratch CNN:** "
            f"accuracy {mobile_metrics['accuracy']:.3f} vs {metrics['accuracy']:.3f}; "
            f"NORMAL recall {mobile_metrics['normal_recall']:.3f} vs {metrics['normal_recall']:.3f}; "
            f"training time {s.get('mobilenet_seconds', 0):.0f}s vs {s['training']['seconds']:.0f}s.",
        ]
    lines += [
        "",
        "## 3. Something surprising about the preprocessing step",
        "",
        f"The raw train images come in {s['distinct_sizes']} different sizes and mixed colour modes "
        f"({s['color_modes']}), so nothing can be batched until every image is resized and converted "
        "to grayscale. Order also matters: class weights must come from the original label counts "
        "(imbalance ratio "
        f"{s['imbalance_ratio']:.2f}x), and augmentation may only touch the training set - otherwise "
        "the validation and test numbers stop describing real-world performance.",
        "",
        "*(Draft - replace with your own observation after running on the real dataset.)*",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=config.DATA_ROOT, help="folder containing train/ test/ val/")
    p.add_argument("--synthetic", action="store_true", help="generate and use a fake demo dataset")
    p.add_argument("--scale", type=float, default=1.0, help="synthetic dataset size multiplier")
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS, help="max epochs (early stopping may stop sooner)")
    p.add_argument("--ablation", action="store_true", help="Bonus 1: also train without class weights")
    p.add_argument("--mobilenet", action="store_true", help="Bonus 2: also try frozen MobileNetV2")
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    args = p.parse_args()

    data_root = args.data_root
    if args.synthetic and args.data_root == config.DATA_ROOT:
        # keep the fake data out of the real data location
        data_root = os.path.join(config.PROJECT_ROOT, "data", "synthetic_chest_xray")
    run(data_root, args.synthetic, args.scale, args.epochs, args.ablation, args.mobilenet, args.results_dir)


if __name__ == "__main__":
    main()
