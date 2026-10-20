"""Nested participant-held-out evaluation for Cerqueira ordinal self-reports.

The outcome is self-perceived upper-limb fatigue (labels 0/1/2). This script
compares signal-feature models with a label-prior baseline and a time-only
diagnostic. Time, window index, trial order, and subject ID are never inputs to
the EMG classifiers. All outputs should remain in the restricted data area.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from output_safety import PublicCheckoutOutputError, ensure_outside_public_checkout  # noqa: E402


CLASSES = np.array([0, 1, 2], dtype=np.int64)
DEFAULT_REGULARIZATIONS = (0.01, 0.1, 1.0)


class CerqueiraModelError(ValueError):
    """Raised when data, fitting, or model provenance violates the contract."""


def write_precision_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write numeric outputs at enough precision for a binary64 round trip."""
    frame.to_csv(Path(path), index=False, float_format="%.17g")


def fit_standardizer(x: np.ndarray) -> dict[str, np.ndarray]:
    """Fit median imputation and z-scoring on training rows only."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise CerqueiraModelError("Training feature matrix must be nonempty and two-dimensional")
    values = values.copy()
    values[~np.isfinite(values)] = np.nan
    with np.errstate(all="ignore"):
        medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    imputed = np.where(np.isfinite(values), values, medians)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales[~np.isfinite(scales) | (scales <= np.finfo(float).eps)] = 1.0
    return {"median": medians, "mean": means, "scale": scales}


def transform_standardizer(x: np.ndarray, transformer: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != len(transformer["median"]):
        raise CerqueiraModelError("Feature matrix does not match the fitted transformer")
    values[~np.isfinite(values)] = np.nan
    values = np.where(np.isfinite(values), values, transformer["median"])
    return (values - transformer["mean"]) / transformer["scale"]


def group_trial_weights(subjects: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Give equal total fitting weight to each subject, then each of their trials."""
    subjects = np.asarray(subjects)
    trials = np.asarray(trials)
    if subjects.ndim != 1 or trials.ndim != 1 or subjects.size != trials.size or not subjects.size:
        raise CerqueiraModelError("Subject and trial vectors must be nonempty and have equal length")
    keys = list(zip(subjects.tolist(), trials.tolist()))
    trial_counts: dict[tuple[Any, Any], int] = {}
    subject_trials: dict[Any, set[Any]] = {}
    for subject, trial in keys:
        trial_counts[(subject, trial)] = trial_counts.get((subject, trial), 0) + 1
        subject_trials.setdefault(subject, set()).add(trial)
    n_subjects = len(subject_trials)
    weights = np.asarray([
        1.0 / (n_subjects * len(subject_trials[subject]) * trial_counts[(subject, trial)])
        for subject, trial in keys
    ], dtype=np.float64)
    return weights / weights.sum()


def make_loso_folds(subjects: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(subjects)
    unique = np.unique(groups)
    if unique.size < 3:
        raise CerqueiraModelError("At least three participants are required for grouped evaluation")
    folds = []
    for subject in unique:
        test = np.flatnonzero(groups == subject)
        train = np.flatnonzero(groups != subject)
        folds.append((train, test))
    return folds


def make_group_folds(subjects: np.ndarray, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(subjects)
    unique = np.unique(groups)
    if n_splits < 2 or unique.size < n_splits:
        raise CerqueiraModelError("Grouped inner folds require at least n_splits distinct participants")
    folds = []
    for held_groups in np.array_split(unique, n_splits):
        validation = np.flatnonzero(np.isin(groups, held_groups))
        training = np.flatnonzero(~np.isin(groups, held_groups))
        if training.size == 0 or validation.size == 0:
            raise CerqueiraModelError("Empty grouped train or validation fold")
        folds.append((training, validation))
    return folds


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -700.0, 50.0))
    return exp / exp.sum(axis=1, keepdims=True)


def fit_multinomial_logit(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    regularization: float = 0.1,
    max_iter: int = 500,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Fit L2-regularized three-class softmax regression with FISTA."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or weight.shape != (len(x),) or len(x) == 0:
        raise CerqueiraModelError("Inconsistent logistic-regression input shapes")
    if not np.isin(y, CLASSES).all() or regularization < 0 or max_iter < 1:
        raise CerqueiraModelError("Invalid labels, regularization, or iteration limit")
    if not np.isfinite(x).all() or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise CerqueiraModelError("Logistic inputs and positive sample weights must be finite")
    weight = weight / weight.sum()
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    target = np.eye(3, dtype=np.float64)[y]
    penalty = np.ones((design.shape[1], 1), dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = design.T @ (weight[:, None] * design)
    lipschitz = 0.5 * float(np.linalg.eigvalsh(gram)[-1]) + regularization
    step = 1.0 / max(lipschitz, 1e-12)
    current = np.zeros((design.shape[1], 3), dtype=np.float64)
    accelerated = current.copy()
    momentum = 1.0
    converged = False
    previous_loss = math.inf
    for iteration in range(1, max_iter + 1):
        probs = _softmax(design @ accelerated)
        gradient = design.T @ (weight[:, None] * (probs - target)) + regularization * penalty * accelerated
        updated = accelerated - step * gradient
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - current)
        delta = float(np.max(np.abs(updated - current)))
        current = updated
        momentum = next_momentum
        if iteration % 10 == 0 or iteration == 1:
            logits = design @ current
            log_partition = np.logaddexp.reduce(logits, axis=1)
            chosen = logits[np.arange(len(y)), y]
            loss = float(np.sum(weight * (log_partition - chosen)) + 0.5 * regularization * np.sum((penalty * current) ** 2))
            if (math.isfinite(previous_loss) and abs(previous_loss - loss) <= tolerance * max(1.0, abs(previous_loss))) or delta <= tolerance:
                converged = True
                previous_loss = loss
                break
            previous_loss = loss
    final_logits = design @ current
    final_loss = float(np.sum(weight * (np.logaddexp.reduce(final_logits, axis=1) - final_logits[np.arange(len(y)), y]))
                       + 0.5 * regularization * np.sum((penalty * current) ** 2))
    return {"coef": current[1:, :], "intercept": current[0, :], "classes": CLASSES.copy(),
            "converged": converged, "iterations": iteration, "loss": final_loss,
            "regularization": float(regularization)}


def predict_multinomial_proba(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return _softmax(x @ model["coef"] + model["intercept"])


def fit_gaussian_nb(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or weight.shape != (len(x),) or not np.isfinite(x).all():
        raise CerqueiraModelError("Invalid Gaussian-NB inputs")
    weight = weight / weight.sum()
    means = np.zeros((3, x.shape[1]), dtype=np.float64)
    variances = np.zeros_like(means)
    priors = np.zeros(3, dtype=np.float64)
    global_variance = np.var(x, axis=0)
    floor = np.maximum(global_variance * 1e-8, 1e-10)
    for label in CLASSES:
        selected = y == label
        if not selected.any():
            raise CerqueiraModelError(f"Training fold has no class {label}")
        local_weight = weight[selected]
        local_weight = local_weight / local_weight.sum()
        means[label] = np.sum(x[selected] * local_weight[:, None], axis=0)
        variances[label] = np.sum((x[selected] - means[label]) ** 2 * local_weight[:, None], axis=0)
        variances[label] = np.maximum(variances[label], floor)
        priors[label] = weight[selected].sum()
    priors = priors / priors.sum()
    return {"means": means, "variances": variances, "priors": priors, "classes": CLASSES.copy()}


def predict_gaussian_nb_proba(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    log_probs = []
    for label in CLASSES:
        variance = model["variances"][label]
        score = np.log(model["priors"][label]) - 0.5 * np.sum(
            np.log(2.0 * np.pi * variance) + (x - model["means"][label]) ** 2 / variance, axis=1
        )
        log_probs.append(score)
    return _softmax(np.column_stack(log_probs))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values, sorted_weights = values[order], weights[order]
    threshold = sorted_weights.sum() / 2.0
    return float(sorted_values[min(int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left")), len(values)-1)])


def fit_time_only(train_progress: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, np.ndarray]:
    progress = np.asarray(train_progress, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    centers = np.zeros(3, dtype=np.float64)
    scales = np.zeros(3, dtype=np.float64)
    priors = np.zeros(3, dtype=np.float64)
    for label in CLASSES:
        selected = y == label
        if not selected.any():
            raise CerqueiraModelError(f"Training fold has no class {label} for time-only baseline")
        local_w = weight[selected]
        centers[label] = _weighted_median(progress[selected], local_w)
        deviations = np.abs(progress[selected] - centers[label])
        scales[label] = max(1.4826 * _weighted_median(deviations, local_w), 0.03)
        priors[label] = local_w.sum()
    return {"centers": centers, "scales": scales, "priors": priors / priors.sum()}


def predict_time_only_proba(model: dict[str, np.ndarray], progress: np.ndarray) -> np.ndarray:
    x = np.asarray(progress, dtype=np.float64).reshape(-1, 1)
    logits = np.empty((len(x), 3), dtype=np.float64)
    for label in CLASSES:
        logits[:, label] = np.log(model["priors"][label]) - np.log(model["scales"][label]) - 0.5 * (
            (x[:, 0] - model["centers"][label]) / model["scales"][label]
        ) ** 2
    return _softmax(logits)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def _binary_auc(y_binary: np.ndarray, scores: np.ndarray) -> float | None:
    positive = y_binary.astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_average_precision(y_binary: np.ndarray, scores: np.ndarray) -> float | None:
    positive = y_binary.astype(bool)
    n_pos = int(positive.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    s, y_sorted = scores[order], positive[order]
    cumulative_pos = np.cumsum(y_sorted)
    group_ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    precision_at_group_end = cumulative_pos[group_ends] / (group_ends + 1)
    positives_in_group = np.diff(np.r_[0, cumulative_pos[group_ends]])
    return float(np.sum(precision_at_group_end * positives_in_group) / n_pos)


def _quadratic_kappa(y: np.ndarray, prediction: np.ndarray) -> float | None:
    matrix = np.zeros((3, 3), dtype=np.float64)
    for truth, pred in zip(y, prediction):
        matrix[int(truth), int(pred)] += 1
    n = matrix.sum()
    if n == 0:
        return None
    observed = matrix / n
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2 / 4.0
    denominator = float(np.sum(weights * expected))
    return 1.0 - float(np.sum(weights * observed)) / denominator if denominator > 0 else None


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (len(y), 3) or len(y) == 0 or not np.isfinite(probs).all():
        raise CerqueiraModelError("Metrics require finite n×3 probabilities and nonempty labels")
    probs = np.clip(probs, 1e-15, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    predicted = np.argmax(probs, axis=1)
    class_metrics: dict[str, dict[str, float | None]] = {}
    f1_values, recalls, auc_values, ap_values = [], [], [], []
    for label in CLASSES:
        tp = int(np.sum((y == label) & (predicted == label)))
        fp = int(np.sum((y != label) & (predicted == label)))
        fn = int(np.sum((y == label) & (predicted != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = _binary_auc(y == label, probs[:, label])
        ap = _binary_average_precision(y == label, probs[:, label])
        class_metrics[str(int(label))] = {"precision": precision, "recall": recall, "f1": f1,
                                          "roc_auc_ovr": auc, "average_precision_ovr": ap}
        f1_values.append(f1)
        recalls.append(recall)
        if auc is not None:
            auc_values.append(auc)
        if ap is not None:
            ap_values.append(ap)
    accuracy = float(np.mean(predicted == y))
    return {
        "n": int(len(y)),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "ordinal_mae": float(np.mean(np.abs(predicted - y))),
        "quadratic_weighted_kappa": _quadratic_kappa(y, predicted),
        "macro_roc_auc_ovr": float(np.mean(auc_values)) if auc_values else None,
        "macro_average_precision_ovr": float(np.mean(ap_values)) if ap_values else None,
        "multiclass_brier_score": float(np.mean(np.sum((probs - np.eye(3)[y]) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(probs[np.arange(len(y)), y]))),
        "classes": class_metrics,
    }


def build_roc_pr_curve_rows(y_true: np.ndarray, probabilities: np.ndarray, model_name: str,
                            thresholds: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Create reproducible one-vs-rest ROC and precision-recall points."""
    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (len(y), 3) or not np.isfinite(probs).all() or not np.isin(y, CLASSES).all():
        raise CerqueiraModelError("Curve generation requires valid labels and finite n×3 probabilities")
    cutoffs = np.linspace(0.0, 1.0, 101) if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    if cutoffs.ndim != 1 or not np.isfinite(cutoffs).all():
        raise CerqueiraModelError("Curve thresholds must be finite one-dimensional values")
    rows: list[dict[str, Any]] = []
    for label in CLASSES:
        positive = y == label
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        for threshold in cutoffs:
            predicted = probs[:, label] >= threshold
            tp = int(np.sum(predicted & positive))
            fp = int(np.sum(predicted & ~positive))
            fn = n_positive - tp
            tn = n_negative - fp
            rows.append({
                "model": model_name,
                "class_ovr": int(label),
                "threshold": float(threshold),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "tpr_recall": tp / n_positive if n_positive else None,
                "fpr": fp / n_negative if n_negative else None,
                "precision": tp / (tp + fp) if tp + fp else 1.0,
            })
    return rows


def derive_trial_progress(
    subjects: np.ndarray,
    trials: np.ndarray,
    window_indices: np.ndarray,
    *,
    window_starts: np.ndarray | None = None,
    window_ends: np.ndarray | None = None,
    trial_support: dict[tuple[int, int], tuple[float, float]] | None = None,
) -> np.ndarray:
    subjects, trials, windows = map(np.asarray, (subjects, trials, window_indices))
    if not (subjects.ndim == trials.ndim == windows.ndim == 1 and subjects.size == trials.size == windows.size):
        raise CerqueiraModelError("Progress inputs must be aligned one-dimensional vectors")
    if trial_support is not None:
        if window_starts is None or window_ends is None:
            raise CerqueiraModelError("Full-support progress requires window start and end times")
        starts, ends = np.asarray(window_starts, dtype=np.float64), np.asarray(window_ends, dtype=np.float64)
        if starts.shape != windows.shape or ends.shape != windows.shape or not np.isfinite(starts).all() or not np.isfinite(ends).all():
            raise CerqueiraModelError("Window times are missing or invalid for full-support progress")
        keys = np.column_stack((subjects, trials))
        progress = np.zeros(len(windows), dtype=np.float64)
        for key in np.unique(keys, axis=0):
            lookup_key = (int(key[0]), int(key[1]))
            if lookup_key not in trial_support:
                raise CerqueiraModelError(f"Trial support is absent for {lookup_key}")
            support_start, support_end = trial_support[lookup_key]
            if not math.isfinite(support_start) or not math.isfinite(support_end) or support_end <= support_start:
                raise CerqueiraModelError(f"Invalid full signal time support for {lookup_key}")
            selected = np.all(keys == key, axis=1)
            centers = (starts[selected] + ends[selected]) / 2.0
            local = (centers - support_start) / (support_end - support_start)
            if np.min(local) < -1e-3 or np.max(local) > 1.001:
                raise CerqueiraModelError(f"Window lies outside audited signal support for {lookup_key}")
            progress[selected] = np.clip(local, 0.0, 1.0)
        return progress
    progress = np.zeros(len(windows), dtype=np.float64)
    keys = np.column_stack((subjects, trials))
    for key in np.unique(keys, axis=0):
        selected = np.all(keys == key, axis=1)
        last = float(np.max(windows[selected]))
        progress[selected] = (windows[selected].astype(np.float64) + 0.5) / (last + 1.0)
    return progress


def load_trial_signal_support(alignment_receipt_path: str | Path, feature_summary: dict[str, Any]) -> dict[tuple[int, int], tuple[float, float]]:
    """Build label-independent full-signal time bounds from the verified alignment receipt."""
    path = Path(alignment_receipt_path)
    if not path.is_file():
        raise CerqueiraModelError("The alignment receipt is required for the time-only diagnostic")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    summary = receipt.get("summary", {})
    if summary.get("emg_archive_md5") != feature_summary.get("source_emg_md5"):
        raise CerqueiraModelError("Alignment receipt EMG hash differs from the feature extraction source")
    if summary.get("label_archive_md5") != feature_summary.get("source_labels_md5"):
        raise CerqueiraModelError("Alignment receipt label hash differs from the feature extraction source")
    if not summary.get("pairing_complete"):
        raise CerqueiraModelError("Alignment receipt does not confirm complete subject/trial pairing")
    support: dict[tuple[int, int], tuple[float, float]] = {}
    for row in receipt.get("trials", []):
        subject, trial = int(row["subject_index"]), int(row["trial_index"])
        start = float(row["signal_observed_time_start_s"])
        sample_count = int(row["signal_monotonic_prefix_sample_count"])
        fs = float(row["signal_sampling_hz_median"])
        mismatch = row.get("signal_channel_time_mismatch_details", [])
        if mismatch:
            sample_count = min(sample_count, min(int(item["row_index"]) for item in mismatch))
        if sample_count <= 0 or not math.isfinite(start) or not math.isfinite(fs) or fs <= 0:
            raise CerqueiraModelError(f"Invalid audited signal support for subject/trial {(subject, trial)}")
        # End is the boundary after the last included sample, matching the window-end convention.
        end = start + sample_count / fs
        support[(subject, trial)] = (start, end)
    return support


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(classification_metrics(y, p)["macro_f1"])


def _subject_macro_f1(y: np.ndarray, p: np.ndarray, subjects: np.ndarray) -> tuple[float, float]:
    scores = [classification_metrics(y[subjects == subject], p[subjects == subject])["macro_f1"]
              for subject in np.unique(subjects)]
    errors = [classification_metrics(y[subjects == subject], p[subjects == subject])["ordinal_mae"]
              for subject in np.unique(subjects)]
    return float(np.mean(scores)), float(np.mean(errors))


def select_family_inner(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray,
    progress: np.ndarray,
    regularizations: tuple[float, ...],
    *,
    n_splits: int = 3,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Select Gaussian NB or softmax regression using grouped inner validation."""
    fold_results: dict[str, list[tuple[float, float]]] = {"gaussian_nb": [], **{f"softmax_l2_{r:g}": [] for r in regularizations}}
    for train, validation in make_group_folds(subjects, n_splits):
        transformer = fit_standardizer(x[train])
        x_train = transform_standardizer(x[train], transformer)
        x_validation = transform_standardizer(x[validation], transformer)
        weight = group_trial_weights(subjects[train], trials[train])
        gnb = fit_gaussian_nb(x_train, y[train], weight)
        p = predict_gaussian_nb_proba(gnb, x_validation)
        fold_results["gaussian_nb"].append(_subject_macro_f1(y[validation], p, subjects[validation]))
        for regularization in regularizations:
            model = fit_multinomial_logit(x_train, y[train], weight, regularization=regularization, max_iter=max_iter)
            if not model["converged"]:
                fold_results[f"softmax_l2_{regularization:g}"].append((float("nan"), float("nan")))
                continue
            p = predict_multinomial_proba(model, x_validation)
            fold_results[f"softmax_l2_{regularization:g}"].append(_subject_macro_f1(y[validation], p, subjects[validation]))
    scores: dict[str, dict[str, float | None]] = {}
    for name, folds in fold_results.items():
        valid = [(f1, mae) for f1, mae in folds if math.isfinite(f1) and math.isfinite(mae)]
        scores[name] = {
            "mean_subject_macro_f1": float(np.mean([v[0] for v in valid])) if valid else None,
            "mean_subject_ordinal_mae": float(np.mean([v[1] for v in valid])) if valid else None,
            "valid_inner_folds": len(valid),
            "excluded_nonconverged_or_invalid_inner_folds": len(folds) - len(valid),
        }
    candidates = [(name, score) for name, score in scores.items() if score["valid_inner_folds"] == n_splits and score["mean_subject_macro_f1"] is not None]
    if not candidates:
        raise CerqueiraModelError("All candidate models failed inner grouped validation")
    selected_name, selected_score = max(candidates, key=lambda item: (
        float(item[1]["mean_subject_macro_f1"]),
        -float(item[1]["mean_subject_ordinal_mae"]),
        item[0] == "gaussian_nb",
    ))
    if selected_name == "gaussian_nb":
        family, regularization = "gaussian_nb", None
    else:
        family = "softmax_l2"
        regularization = float(selected_name.removeprefix("softmax_l2_"))
    logistic_candidates = [(name, score) for name, score in candidates if name.startswith("softmax_l2_")]
    best_logistic = max(logistic_candidates, key=lambda item: (
        float(item[1]["mean_subject_macro_f1"]),
        -float(item[1]["mean_subject_ordinal_mae"]),
    )) if logistic_candidates else None
    best_logistic_regularization = (
        float(best_logistic[0].removeprefix("softmax_l2_")) if best_logistic is not None else None
    )
    return {"family": family, "regularization": regularization,
            "best_softmax_regularization": best_logistic_regularization,
            "selected_score": selected_score,
            "inner_scores": scores, "inner_fold_count": n_splits}


def _fit_and_predict(family: str, regularization: float | None, x_train: np.ndarray, y_train: np.ndarray,
                     subjects_train: np.ndarray, trials_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    transformer = fit_standardizer(x_train)
    train_z = transform_standardizer(x_train, transformer)
    test_z = transform_standardizer(x_test, transformer)
    weight = group_trial_weights(subjects_train, trials_train)
    if family == "gaussian_nb":
        model = fit_gaussian_nb(train_z, y_train, weight)
        return predict_gaussian_nb_proba(model, test_z), model, transformer
    if family == "softmax_l2":
        model = fit_multinomial_logit(train_z, y_train, weight, regularization=float(regularization), max_iter=800)
        if not model["converged"]:
            raise CerqueiraModelError(f"Softmax failed to converge in {model['iterations']} iterations")
        return predict_multinomial_proba(model, test_z), model, transformer
    raise CerqueiraModelError(f"Unknown model family: {family}")


def _weighted_class_prior(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight = weight / weight.sum()
    return np.asarray([weight[y == label].sum() for label in CLASSES], dtype=np.float64)


def _save_fold_model(
    path: Path,
    family: str,
    model: dict[str, Any],
    transformer: dict[str, np.ndarray],
    regularization: float | None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "model_family": np.asarray(family),
        "regularization": np.asarray(-1.0 if regularization is None else regularization),
        "median": transformer["median"],
        "mean": transformer["mean"],
        "scale": transformer["scale"],
        "classes": CLASSES,
    }
    if family == "gaussian_nb":
        arrays.update({"means": model["means"], "variances": model["variances"], "priors": model["priors"]})
    elif family == "softmax_l2":
        arrays.update({"coef": model["coef"], "intercept": model["intercept"]})
    else:
        raise CerqueiraModelError(f"Cannot serialize unknown model family: {family}")
    np.savez_compressed(path, **arrays)


def load_fold_model_bundle(model_path: str | Path, fold_manifest_path: str | Path) -> dict[str, Any]:
    """Load one LOSO fold model only after manifest and SHA-256 verification."""
    path, manifest_path = Path(model_path), Path(fold_manifest_path)
    if not path.is_file() or not manifest_path.is_file():
        raise CerqueiraModelError("Fold model and its manifest are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((row for row in manifest.get("fold_models", []) if row.get("model_file") == path.name), None)
    if entry is None:
        raise CerqueiraModelError("Fold model is not listed in the supplied manifest")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.get("model_sha256"):
        raise CerqueiraModelError("Fold model checksum differs from its manifest")
    with np.load(path, allow_pickle=False) as payload:
        family = str(payload["model_family"].item())
        regularization = float(payload["regularization"].item())
        transformer = {name: payload[name].astype(np.float64) for name in ("median", "mean", "scale")}
        if family == "gaussian_nb":
            model = {name: payload[name].astype(np.float64) for name in ("means", "variances", "priors")}
        elif family == "softmax_l2":
            model = {"coef": payload["coef"].astype(np.float64), "intercept": payload["intercept"].astype(np.float64)}
        else:
            raise CerqueiraModelError(f"Unknown serialized model family: {family}")
        classes = payload["classes"].astype(np.int64)
    if not np.array_equal(classes, CLASSES):
        raise CerqueiraModelError("Serialized class order differs from the fixed 0/1/2 order")
    return {"family": family, "regularization": None if regularization < 0 else regularization,
            "transformer": transformer, "model": model,
            "feature_columns": manifest["feature_columns_ordered"],
            "feature_extraction": manifest["feature_extraction"], "manifest_entry": entry}


def predict_model_bundle(bundle: dict[str, Any], raw_features: np.ndarray) -> np.ndarray:
    """Run inference from the frozen feature matrix using a verified fold bundle."""
    x = transform_standardizer(np.asarray(raw_features, dtype=np.float64), bundle["transformer"])
    if bundle["family"] == "gaussian_nb":
        return predict_gaussian_nb_proba(bundle["model"], x)
    if bundle["family"] == "softmax_l2":
        return predict_multinomial_proba(bundle["model"], x)
    raise CerqueiraModelError(f"Unknown model family in bundle: {bundle['family']}")


def _metrics_by_subject(y: np.ndarray, probabilities: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    results = {str(int(subject)): classification_metrics(y[subjects == subject], probabilities[subjects == subject])
               for subject in np.unique(subjects)}
    names = ("accuracy", "macro_f1", "balanced_accuracy", "ordinal_mae", "quadratic_weighted_kappa",
             "macro_roc_auc_ovr", "macro_average_precision_ovr", "multiclass_brier_score", "log_loss")
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(4021)
    for metric in names:
        values = np.asarray([row[metric] for row in results.values() if row[metric] is not None], dtype=np.float64)
        if not values.size:
            summary[metric] = {"n_subjects": 0, "mean": None, "sd": None, "ci95_percentile": None}
            continue
        bootstrap = np.mean(values[rng.integers(0, len(values), size=(5000, len(values)))], axis=1)
        summary[metric] = {"n_subjects": int(values.size), "mean": float(values.mean()),
                           "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                           "ci95_percentile": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]}
    return {"by_subject": results, "subject_macro_summary": summary}


def evaluate_loso(
    feature_csv: str | Path,
    feature_summary_json: str | Path,
    output_directory: str | Path,
    *,
    regularizations: tuple[float, ...] = DEFAULT_REGULARIZATIONS,
    inner_splits: int = 3,
    max_iter: int = 500,
    alignment_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    feature_path, summary_path, out_dir = map(Path, (feature_csv, feature_summary_json, output_directory))
    try:
        ensure_outside_public_checkout((out_dir,))
    except PublicCheckoutOutputError as error:
        raise CerqueiraModelError(str(error)) from error
    if not feature_path.is_file() or not summary_path.is_file():
        raise CerqueiraModelError("Feature CSV and its extraction summary are required")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise CerqueiraModelError("Refusing to overwrite nonempty model output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    if csv_hash != source_summary.get("feature_csv_sha256"):
        raise CerqueiraModelError("Feature CSV hash differs from its extraction summary")
    frame = pd.read_csv(feature_path)
    if len(frame) != int(source_summary.get("feature_csv_rows", -1)):
        raise CerqueiraModelError("Feature CSV row count differs from extraction summary")
    feature_columns = list(source_summary["feature_columns"])
    required = {"subject_index", "trial_index", "window_index", "label", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CerqueiraModelError(f"Missing required feature columns: {missing}")
    if frame.duplicated(["subject_index", "trial_index", "window_index"]).any():
        raise CerqueiraModelError("Duplicate subject/trial/window keys")
    y = frame["label"].to_numpy(dtype=np.int64)
    subjects = frame["subject_index"].to_numpy(dtype=np.int64)
    trials = frame["trial_index"].to_numpy(dtype=np.int64)
    window_indices = frame["window_index"].to_numpy(dtype=np.int64)
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    if alignment_receipt_path is not None:
        trial_support = load_trial_signal_support(alignment_receipt_path, source_summary)
        progress = derive_trial_progress(subjects, trials, window_indices,
                                         window_starts=frame["window_start_s"].to_numpy(dtype=np.float64),
                                         window_ends=frame["window_end_s"].to_numpy(dtype=np.float64),
                                         trial_support=trial_support)
        time_progress_source = "window-center time divided by full audited signal support, independent of label-window retention"
    else:
        trial_support = None
        progress = derive_trial_progress(subjects, trials, window_indices)
        time_progress_source = "retained stable-window indices only; diagnostic approximation, not full recording support"
    if not np.isin(y, CLASSES).all() or np.unique(subjects).size < 3:
        raise CerqueiraModelError("Labels or participant count violate the ordinal evaluation contract")

    model_names = ["majority", "time_only_diagnostic", "gaussian_nb", "softmax_l2", "nested_selector"]
    probability_buffers = {name: np.full((len(y), 3), np.nan, dtype=np.float64) for name in model_names}
    selected_by_outer: dict[str, str] = {}
    selection_details: dict[str, Any] = {}
    fold_models_directory = out_dir / "fold_models"
    fold_models_directory.mkdir(parents=True, exist_ok=True)
    fold_model_entries: list[dict[str, Any]] = []
    folds = make_loso_folds(subjects)
    for outer_index, (train, test) in enumerate(folds, start=1):
        selection = select_family_inner(x[train], y[train], subjects[train], trials[train], progress[train],
                                        regularizations, n_splits=inner_splits, max_iter=max_iter)
        selected_by_outer[str(int(subjects[test[0]]))] = selection["family"] if selection["family"] == "gaussian_nb" else f"softmax_l2_{selection['regularization']:g}"
        selection_details[str(int(subjects[test[0]]))] = selection
        train_weight = group_trial_weights(subjects[train], trials[train])
        prior = _weighted_class_prior(y[train], train_weight)
        probability_buffers["majority"][test] = np.tile(prior, (len(test), 1))
        time_model = fit_time_only(progress[train], y[train], train_weight)
        probability_buffers["time_only_diagnostic"][test] = predict_time_only_proba(time_model, progress[test])
        outer_fitted: dict[str, tuple[dict[str, Any], dict[str, np.ndarray], float | None]] = {}
        for family in ("gaussian_nb", "softmax_l2"):
            regularization = selection["best_softmax_regularization"] if family == "softmax_l2" else None
            if family == "softmax_l2" and regularization is None:
                raise CerqueiraModelError("Softmax had no converged candidate in inner grouped validation")
            probabilities, model, transformer = _fit_and_predict(family, regularization, x[train], y[train],
                                                                  subjects[train], trials[train], x[test])
            probability_buffers[family][test] = probabilities
            outer_fitted[family] = (model, transformer, regularization)
        chosen = selection["family"]
        probability_buffers["nested_selector"][test] = probability_buffers[chosen][test]
        heldout_subject = int(subjects[test[0]])
        chosen_model, chosen_transformer, chosen_regularization = outer_fitted[chosen]
        fold_model_path = fold_models_directory / f"heldout_subject_{heldout_subject:02d}.npz"
        _save_fold_model(fold_model_path, chosen, chosen_model, chosen_transformer, chosen_regularization)
        fold_model_entries.append({
            "heldout_subject_index": heldout_subject,
            "model_file": fold_model_path.name,
            "model_sha256": hashlib.sha256(fold_model_path.read_bytes()).hexdigest(),
            "model_family": chosen,
            "regularization": chosen_regularization,
            "selected_by_inner_grouped_validation": selection,
        })
        print(f"LOSO {outer_index}/{len(folds)} participant={int(subjects[test[0]])} selected={selected_by_outer[str(int(subjects[test[0]]))]}")

    if any(not np.isfinite(values).all() for values in probability_buffers.values()):
        raise CerqueiraModelError("Incomplete/non-finite out-of-fold predictions")
    metrics = {name: {"pooled_window_metrics": classification_metrics(y, probs),
                      **_metrics_by_subject(y, probs, subjects)}
               for name, probs in probability_buffers.items()}
    metrics["metadata"] = {
        "task": source_summary["task"],
        "feature_csv_sha256": csv_hash,
        "subject_count": int(np.unique(subjects).size),
        "window_count": int(len(y)),
        "trial_count": int(frame.groupby(["subject_index", "trial_index"]).ngroups),
        "feature_columns": feature_columns,
        "time_predictors_excluded_from_emg_models": ["window_index", "window_start_s", "window_end_s", "trial_index", "subject_index", "progress"],
        "time_only_diagnostic": "Uses normalized within-trial window position and training-fold class-distribution estimates; requires complete trial duration and is not a deployable online model.",
        "time_progress_source": time_progress_source,
        "outer_validation": "leave-one-participant-out (LOSO); all windows from the held participant are excluded from fitting",
        "inner_validation": f"{inner_splits}-fold grouped participant CV for model family and regularization selection",
        "regularization_grid": list(regularizations),
        "selection_frequency_by_outer_participant": selected_by_outer,
        "interpretation_limit": "Predicts self-perceived upper-limb fatigue labels; cannot establish functional fatigue or quadriceps fatigue.",
    }
    oof_path = out_dir / "oof_predictions.csv"
    output = frame[["subject_index", "trial_index", "window_index", "window_start_s", "window_end_s", "label"]].copy()
    for name, probs in probability_buffers.items():
        output[f"{name}_pred"] = np.argmax(probs, axis=1)
        for label in CLASSES:
            output[f"{name}_p{label}"] = probs[:, label]
    write_precision_csv(output, oof_path)
    curve_rows = [row for name, probs in probability_buffers.items()
                  for row in build_roc_pr_curve_rows(y, probs, name)]
    curve_path = out_dir / "roc_pr_curves_ovr.csv"
    write_precision_csv(pd.DataFrame(curve_rows), curve_path)
    metrics_path = out_dir / "nested_loso_metrics.json"
    metrics["oof_predictions_csv"] = oof_path.name
    metrics["oof_predictions_sha256"] = hashlib.sha256(oof_path.read_bytes()).hexdigest()
    metrics["roc_pr_curves_csv"] = curve_path.name
    metrics["roc_pr_curves_sha256"] = hashlib.sha256(curve_path.read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fold_manifest_path = out_dir / "fold_models_manifest.json"
    fold_manifest_path.write_text(json.dumps({
        "task": source_summary["task"],
        "outer_validation": "leave-one-participant-out",
        "feature_columns_ordered": feature_columns,
        "feature_extraction": {
            "window_seconds": source_summary["window_seconds"],
            "window_overlap_fraction": source_summary["window_overlap_fraction"],
            "frequency_band_hz": source_summary["frequency_band_hz"],
            "fir_design": source_summary["fir_design"]["type"],
            "fir_taps": source_summary["fir_design"]["taps"],
            "sampling_hz_median": source_summary["fir_design"]["sampling_hz_median"],
            "sampling_rate_relative_tolerance": 0.005,
            "edge_padding": source_summary["fir_design"]["edge_padding"],
            "zero_phase_offline": source_summary["fir_design"]["zero_phase_offline"],
        },
        "training_feature_csv_sha256": csv_hash,
        "fold_models": fold_model_entries,
        "privacy": "Private validation artifacts; one held-out participant per fold, not a deployable public model.",
    }, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    final_selection = select_family_inner(x, y, subjects, trials, progress, regularizations,
                                          n_splits=inner_splits, max_iter=max_iter)
    transformer = fit_standardizer(x)
    x_scaled = transform_standardizer(x, transformer)
    weights = group_trial_weights(subjects, trials)
    if final_selection["family"] == "gaussian_nb":
        final_model = fit_gaussian_nb(x_scaled, y, weights)
        model_arrays = {"means": final_model["means"], "variances": final_model["variances"], "priors": final_model["priors"]}
    else:
        final_model = fit_multinomial_logit(x_scaled, y, weights, regularization=float(final_selection["regularization"]), max_iter=1000)
        if not final_model["converged"]:
            raise CerqueiraModelError("Final model did not converge; no final artifact was written")
        model_arrays = {"coef": final_model["coef"], "intercept": final_model["intercept"]}
    model_path = out_dir / "final_model_cerqueira_ordinal_20261020.npz"
    np.savez_compressed(model_path, **model_arrays, median=transformer["median"], mean=transformer["mean"],
                        scale=transformer["scale"], classes=CLASSES)
    manifest = {
        "task": source_summary["task"],
        "model_family": final_selection["family"],
        "regularization": final_selection["regularization"],
        "inner_selection": final_selection,
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "training_feature_csv_sha256": csv_hash,
        "feature_columns_ordered": feature_columns,
        "preprocessing": "training-fold median imputation, then training-fold mean/std scaling; final artifact stores these statistics",
        "feature_extraction": {
            "window_seconds": source_summary["window_seconds"],
            "window_overlap_fraction": source_summary["window_overlap_fraction"],
            "frequency_band_hz": source_summary["frequency_band_hz"],
            "fir_design": source_summary["fir_design"]["type"],
            "fir_taps": source_summary["fir_design"]["taps"],
            "sampling_hz_median": source_summary["fir_design"]["sampling_hz_median"],
        },
        "not_a_fatigue_function_model": True,
        "privacy": "Private research output; do not publish participant-level predictions or trained artifact without a separate distribution review.",
    }
    manifest_path = out_dir / "final_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"metrics_path": str(metrics_path), "oof_path": str(oof_path), "model_path": str(model_path),
            "manifest_path": str(manifest_path), "fold_models_directory": str(fold_models_directory),
            "fold_manifest_path": str(fold_manifest_path), "model_family": final_selection["family"],
            "regularization": final_selection["regularization"], "outer_selection": selected_by_outer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_csv", type=Path)
    parser.add_argument("feature_summary_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--regularizations", type=float, nargs="+", default=list(DEFAULT_REGULARIZATIONS))
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--alignment-receipt", type=Path, help="verified full-signal time support used only for the diagnostic time baseline")
    args = parser.parse_args()
    result = evaluate_loso(args.feature_csv, args.feature_summary_json, args.output_directory,
                           regularizations=tuple(args.regularizations), inner_splits=args.inner_splits,
                           max_iter=args.max_iter, alignment_receipt_path=args.alignment_receipt)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
