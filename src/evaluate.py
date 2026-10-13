"""Task 6 - evaluate properly on the untouched test set."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config


def predict(model, test_ds, threshold=0.5):
    """Return (probabilities, hard 0/1 predictions) for a test dataset (kept in file order)."""
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def evaluate(y_true, y_pred):
    """Accuracy, per-class precision/recall/F1, confusion matrix and the key error counts.

    Confusion matrix layout (rows = truth, columns = prediction):
        [[ TN (NORMAL ok)        FP (NORMAL -> PNEUMONIA) ],
         [ FN (PNEUMONIA->NORMAL) TP (PNEUMONIA ok)       ]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": report,
        "report_text": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=config.CLASS_NAMES, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "normal_recall": float(report["NORMAL"]["recall"]),
        "pneumonia_recall": float(report["PNEUMONIA"]["recall"]),
        "false_negatives": fn,   # real PNEUMONIA predicted NORMAL (missed disease)
        "false_positives": fp,   # real NORMAL predicted PNEUMONIA (false alarm)
        "true_negatives": tn,
        "true_positives": tp,
    }


def always_pneumonia_baseline(y_true):
    """Accuracy of a useless model that always says PNEUMONIA (shows why accuracy lies)."""
    y_true = np.asarray(y_true)
    return float((y_true == 1).mean())


def error_analysis(metrics):
    """Plain-language answers to the three Task 6 questions."""
    fn, fp = metrics["false_negatives"], metrics["false_positives"]
    if fn > fp:
        more = "false negatives"
    elif fp > fn:
        more = "false positives"
    else:
        more = "the same number of each"
    return "\n".join(
        [
            f"Recall on the NORMAL class: {metrics['normal_recall']:.3f}",
            f"False negatives (real PNEUMONIA predicted NORMAL): {fn}",
            f"False positives (real NORMAL predicted PNEUMONIA): {fp}",
            f"The model makes more: {more}",
            "In a medical setting a false negative is usually worse: a real pneumonia case "
            "is missed and goes untreated, while a false positive normally costs only a "
            "follow-up check. That is why NORMAL recall alone is not enough - PNEUMONIA "
            "recall (how many sick patients are caught) must be checked too.",
        ]
    )
