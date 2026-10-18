"""Nested participant-held-out baselines for the low-confidence time-position proxy.

The target is within-recording temporal position only. It is not fatigue,
repetition count, active time, observed technical failure, or a physiological
ground truth. Private pseudonyms and per-participant predictions stay in memory;
only participant-aggregated metrics are written to the project.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from extraer_caracteristicas_proxy_temporal import (
    PREDICTOR_COLUMNS,
    _contiguous_pair_indices,
    _features,
)
from comparar_filtros_emg import augment_signal


LABEL_TO_INT = {"temprana": 0, "transicion": 1, "tardia": 2}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
FILTER_BRANCHES = ("control", "hp20", "hp40", "bp20_400", "fir_bp20_400")
MODEL_CANDIDATES = {
    "majority": (None,),
    "nearest_centroid": (None,),
    "knn": (3, 5, 9),
    "ridge": (0.1, 1.0, 10.0),
    "logistic": (0.01, 0.1, 1.0),
}
MODEL_COMPLEXITY = {"majority": 0, "nearest_centroid": 1, "knn": 2, "ridge": 3, "logistic": 4}
MODEL_SCORE_TYPES = {
    "majority": "training_class_prior_score",
    "nearest_centroid": "negative_squared_centroid_distance",
    "knn": "neighbor_vote_fraction",
    "ridge": "ridge_decision_score",
    "logistic": "multinomial_logistic_probability",
}
KEY_FIELDS = ("participant_id", "recording_id", "leg", "set_index")
LOGISTIC_INITIAL_ITERATION_BUDGET = 20000
LOGISTIC_GRADIENT_TOLERANCE = 1e-6
LOGISTIC_MAX_RETRIES = 4
LOGISTIC_RETRY_MULTIPLIER = 4


def synthetic_noise_scenarios(fs_hz: float) -> list[tuple[str, str, dict[str, float | int]]]:
    """Return predeclared simulated scenarios; parameters are not measured noise."""
    sample_rate = float(fs_hz)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("estimated sampling rate must be finite and positive")
    scenarios: list[tuple[str, str, dict[str, float | int]]] = [
        ("gaussian_25db", "gaussian", {"snr_db": 25.0}),
        ("gaussian_30db", "gaussian", {"snr_db": 30.0}),
        ("gaussian_35db", "gaussian", {"snr_db": 35.0}),
    ]
    for frequency in (50.0, 60.0):
        if frequency >= sample_rate / 2.0:
            raise ValueError("estimated sampling rate is too low for the predeclared line-tone scenario")
        scenarios.append((f"line_{int(frequency)}", f"line_{int(frequency)}", {
            "fs_hz": sample_rate, "amplitude_ratio": 0.1,
        }))
    if 2.0 >= sample_rate / 2.0:
        raise ValueError("estimated sampling rate is too low for the predeclared 2 Hz scenario")
    scenarios.extend([
        ("motion_2hz", "motion_low_frequency", {
            "fs_hz": sample_rate, "frequency_hz": 2.0, "amplitude_ratio": 0.25,
        }),
        ("transient_5samples", "transient", {
            "count": 1, "width_samples": 5, "amplitude_ratio": 3.0,
        }),
    ])
    return scenarios


def augment_training_segments(
    signal_segments: list[dict[str, np.ndarray | float]],
    train_indices: np.ndarray,
    branch: str,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create one balanced-mix synthetic copy for each supplied train index only."""
    indices = np.asarray(train_indices, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("training segment indices must be a nonempty unique vector")
    if np.any(indices < 0) or np.any(indices >= len(signal_segments)) or y.ndim != 1 or y.size != len(signal_segments):
        raise ValueError("training indices or labels are inconsistent with private signal segments")
    scenario_names = [item[0] for item in synthetic_noise_scenarios(float(signal_segments[int(indices[0])]["fs_hz"]))]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    assigned = {int(index): scenario_names[position % len(scenario_names)] for position, index in enumerate(shuffled)}
    counts: Counter[str] = Counter()
    feature_rows: list[list[float]] = []
    for position, index_value in enumerate(indices):
        index = int(index_value)
        item = signal_segments[index]
        if branch not in item:
            raise ValueError("the selected filter branch is missing from a training signal segment")
        scenarios = {name: (kind, params) for name, kind, params in synthetic_noise_scenarios(float(item["fs_hz"]))}
        scenario_name = assigned[index]
        kind, params = scenarios[scenario_name]
        original = np.asarray(item[branch], dtype=np.float64)
        if original.ndim != 1 or original.size < 2 or not np.isfinite(original).all():
            raise ValueError("training signal segment must be finite and contain at least two samples")
        pair_indices = np.asarray(item.get("contiguous_pair_indices", []), dtype=np.int64)
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or pair_indices.shape[0] == 0:
            raise ValueError("training signal segment must retain at least one contiguous sample pair")
        augmented = augment_signal(original, kind, params, seed=int(seed + 7919 * position + 104729 * index))
        feature = _features(
            augmented.tolist(), [tuple(map(int, pair)) for pair in pair_indices]
        )
        feature_rows.append([float(feature[column]) for column in PREDICTOR_COLUMNS])
        counts[scenario_name] += 1
    matrix = np.asarray(feature_rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("synthetic training features contain nonfinite values")
    return matrix, y[indices].copy(), dict(counts)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize_train_test(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature centering/scaling on train rows only."""
    train = np.asarray(train_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test features must be compatible two-dimensional arrays")
    if train.shape[0] == 0 or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("feature arrays must be nonempty and finite")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError("truth and prediction must be nonempty one-dimensional arrays of equal length")
    scores: list[float] = []
    for label in range(3):
        true_positive = int(np.sum((truth == label) & (predicted == label)))
        false_positive = int(np.sum((truth != label) & (predicted == label)))
        false_negative = int(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(sum(scores) / 3.0)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError("truth and prediction must be nonempty one-dimensional arrays of equal length")
    recalls: list[float] = []
    for label in range(3):
        support = int(np.sum(truth == label))
        if support == 0:
            raise ValueError("balanced accuracy requires all three classes in each scored group")
        recalls.append(float(np.sum((truth == label) & (predicted == label)) / support))
    return float(sum(recalls) / 3.0)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | str]:
    """Return complete hard-class metrics for the three temporal proxy classes."""
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError("truth and prediction must be nonempty one-dimensional arrays of equal length")
    if np.any((truth < 0) | (truth >= 3)) or np.any((predicted < 0) | (predicted >= 3)):
        raise ValueError("truth and prediction must contain only the three proxy class indices")
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    if np.any(support == 0):
        raise ValueError("fold metrics require all three true proxy classes")

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    specificity: list[float] = []
    negative_predictive_value: list[float] = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    result: dict[str, float | int | str] = {
        "accuracy": float(np.trace(confusion) / truth.size),
        "roc_auc": "",
        "pr_auc": "",
        "auc_status": "not_computed_hard_class_predictions_only",
    }
    for index, label in sorted(INT_TO_LABEL.items()):
        true_positive = int(confusion[index, index])
        class_precision = true_positive / int(predicted_count[index]) if predicted_count[index] else 0.0
        class_recall = true_positive / int(support[index])
        class_f1 = (
            2.0 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        false_positive = int(predicted_count[index] - true_positive)
        false_negative = int(support[index] - true_positive)
        true_negative = int(truth.size - true_positive - false_positive - false_negative)
        class_specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        class_npv = true_negative / (true_negative + false_negative) if true_negative + false_negative else 0.0
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
        specificity.append(class_specificity)
        negative_predictive_value.append(class_npv)
        weighted_precision += class_precision * int(support[index])
        weighted_recall += class_recall * int(support[index])
        weighted_f1 += class_f1 * int(support[index])
        result[f"precision_{label}"] = class_precision
        result[f"recall_{label}"] = class_recall
        result[f"f1_{label}"] = class_f1
        result[f"specificity_{label}"] = class_specificity
        result[f"npv_{label}"] = class_npv
    result["macro_precision"] = float(np.mean(precision))
    result["macro_recall"] = float(np.mean(recall))
    result["macro_f1"] = float(np.mean(f1))
    result["macro_specificity"] = float(np.mean(specificity))
    result["macro_npv"] = float(np.mean(negative_predictive_value))
    result["weighted_precision"] = weighted_precision / truth.size
    result["weighted_recall"] = weighted_recall / truth.size
    result["weighted_f1"] = weighted_f1 / truth.size
    result["micro_precision"] = result["accuracy"]
    result["micro_recall"] = result["accuracy"]
    result["micro_f1"] = result["accuracy"]
    result["balanced_accuracy"] = result["macro_recall"]
    row_totals = confusion.sum(axis=1).astype(np.float64)
    column_totals = confusion.sum(axis=0).astype(np.float64)
    total = float(truth.size)
    numerator = float(np.trace(confusion) * total - np.dot(row_totals, column_totals))
    denominator = math.sqrt(
        max(total * total - float(np.dot(column_totals, column_totals)), 0.0)
        * max(total * total - float(np.dot(row_totals, row_totals)), 0.0)
    )
    result["matthews_corrcoef"] = numerator / denominator if denominator else 0.0
    expected_accuracy = float(np.dot(row_totals, column_totals) / (total * total))
    result["cohen_kappa"] = (
        (float(result["accuracy"]) - expected_accuracy) / (1.0 - expected_accuracy)
        if expected_accuracy < 1.0 else 0.0
    )
    for true_index, true_label in sorted(INT_TO_LABEL.items()):
        for predicted_index, predicted_label in sorted(INT_TO_LABEL.items()):
            result[f"confusion_{true_index}{predicted_index}"] = int(
                confusion[true_index, predicted_index]
            )
    return result


def binary_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, np.ndarray | float]:
    """Compute tie-aware ROC and step average precision for one-vs-rest scores."""
    truth = np.asarray(y_true, dtype=bool)
    score = np.asarray(y_score, dtype=np.float64)
    if truth.ndim != 1 or score.shape != truth.shape or truth.size == 0:
        raise ValueError("binary truth and score must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite(score).all():
        raise ValueError("binary ranking scores must be finite")
    positives = int(np.sum(truth))
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC and average precision require positive and negative examples")

    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    sorted_truth = truth[order].astype(np.int64)
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), truth.size - 1]
    true_positives = np.cumsum(sorted_truth, dtype=np.int64)[threshold_ends]
    false_positives = 1 + threshold_ends - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    recall = true_positives / positives
    precision = true_positives / (true_positives + false_positives)
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    roc_auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))
    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "average_precision": average_precision}


def classification_score_metrics(
    y_true: np.ndarray, class_scores: np.ndarray, *, score_type: str
) -> dict[str, float | str]:
    """Return OvR ROC/AP per class and probability scores when the output is probabilistic."""
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(class_scores, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != (truth.size, 3) or truth.size == 0:
        raise ValueError("class scores must have shape (n_samples, 3) matching nonempty truth")
    if np.any((truth < 0) | (truth >= 3)) or not np.isfinite(scores).all():
        raise ValueError("truth labels and class scores must be finite and in the three-class schema")
    if not score_type:
        raise ValueError("score_type must describe the model output")

    roc_values: list[float] = []
    ap_values: list[float] = []
    result: dict[str, float | str] = {"score_type": score_type}
    one_hot = np.eye(3, dtype=np.int64)[truth]
    for index, label in sorted(INT_TO_LABEL.items()):
        ranked = binary_ranking_metrics(truth == index, scores[:, index])
        roc = float(ranked["roc_auc"])
        average_precision = float(ranked["average_precision"])
        roc_values.append(roc)
        ap_values.append(average_precision)
        result[f"roc_auc_{label}"] = roc
        result[f"average_precision_{label}"] = average_precision
    result["macro_roc_auc_ovr"] = float(np.mean(roc_values))
    result["macro_average_precision_ovr"] = float(np.mean(ap_values))
    micro = binary_ranking_metrics(one_hot.reshape(-1).astype(bool), scores.reshape(-1))
    result["micro_roc_auc_ovr"] = float(micro["roc_auc"])
    result["micro_average_precision_ovr"] = float(micro["average_precision"])

    probability_types = {
        "probability", "training_class_prior_score", "neighbor_vote_fraction",
        "multinomial_logistic_probability",
    }
    result["probability_metrics_status"] = "computed_not_calibration_validation" if score_type in probability_types else "not_probability_output"
    result["multiclass_brier"] = ""
    result["log_loss"] = ""
    if score_type in probability_types:
        if np.any(scores < -1e-12) or np.any(scores > 1.0 + 1e-12):
            raise ValueError("probability-like class scores must be in [0, 1]")
        row_sums = scores.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("probability-like class scores must sum to one per sample")
        result["multiclass_brier"] = float(np.mean(np.sum((scores - one_hot) ** 2, axis=1)))
        clipped = np.clip(scores[np.arange(truth.size), truth], 1e-15, 1.0)
        result["log_loss"] = float(-np.mean(np.log(clipped)))
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(centered)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _predict_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
    *,
    diagnostics: dict[str, object] | None = None,
    max_iterations: int = LOGISTIC_INITIAL_ITERATION_BUDGET,
    gradient_tolerance: float = LOGISTIC_GRADIENT_TOLERANCE,
    max_retries: int = LOGISTIC_MAX_RETRIES,
    retry_multiplier: int = LOGISTIC_RETRY_MULTIPLIER,
    return_training_predictions: bool = False,
    return_scores: bool = False,
) -> np.ndarray:
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("logistic L2 strength must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("logistic maximum iterations must be a positive integer")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("logistic retry count must be a nonnegative integer")
    if not isinstance(retry_multiplier, int) or retry_multiplier < 2:
        raise ValueError("logistic retry multiplier must be an integer of at least two")
    if not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("logistic gradient tolerance must be finite and positive")
    n_rows, n_features = train_x.shape
    x_augmented = np.column_stack((np.ones(n_rows), train_x))
    prediction_x = np.vstack((test_x, train_x)) if return_training_predictions else test_x
    prediction_augmented = np.column_stack((np.ones(prediction_x.shape[0]), prediction_x))
    targets = np.eye(3, dtype=np.float64)[train_y]
    weights = np.zeros((n_features + 1, 3), dtype=np.float64)
    penalty = np.full(n_features + 1, alpha, dtype=np.float64)
    penalty[0] = 1e-8  # tiny bias stabilization; feature weights use the selected L2 strength
    spectral = float(np.linalg.norm(x_augmented, ord=2) ** 2 / n_rows)
    # The multinomial-softmax Hessian is bounded by 0.5 * X'X / n;
    # adding alpha gives a deterministic Lipschitz upper bound.
    step_size = 1.0 / max(0.5 * spectral + alpha, 1e-12)
    converged = False
    gradient_inf_norm = float("inf")
    total_iterations = 0
    retry_count = 0
    resumed_from_finite_checkpoint = False
    attempt_diagnostics: list[dict[str, object]] = []
    current_step_size = step_size
    for attempt in range(max_retries + 1):
        attempt_budget = max_iterations * (retry_multiplier ** attempt)
        extrapolated = weights.copy()
        momentum = 1.0
        last_finite_weights = weights.copy()
        attempt_iterations = 0
        failure: str | None = None
        for attempt_iterations in range(1, attempt_budget + 1):
            probabilities = _softmax(x_augmented @ extrapolated)
            gradient = (x_augmented.T @ (probabilities - targets)) / n_rows
            gradient += penalty[:, None] * extrapolated
            updated = extrapolated - current_step_size * gradient
            if not np.isfinite(updated).all():
                failure = "nonfinite_weights"
                break
            updated_probabilities = _softmax(x_augmented @ updated)
            updated_gradient = (x_augmented.T @ (updated_probabilities - targets)) / n_rows
            updated_gradient += penalty[:, None] * updated
            gradient_inf_norm = float(np.max(np.abs(updated_gradient)))
            if not math.isfinite(gradient_inf_norm):
                failure = "nonfinite_gradient"
                break
            previous = weights
            weights = updated
            last_finite_weights = weights.copy()
            if gradient_inf_norm <= gradient_tolerance:
                converged = True
                break
            next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
            next_extrapolated = weights + ((momentum - 1.0) / next_momentum) * (weights - previous)
            # Adaptive restart limits oscillation while retaining FISTA acceleration.
            if float(np.sum((extrapolated - weights) * (weights - previous))) > 0.0:
                next_momentum = 1.0
                next_extrapolated = weights.copy()
            extrapolated = next_extrapolated
            momentum = next_momentum
        total_iterations += attempt_iterations
        attempt_diagnostics.append({
            "attempt": attempt + 1,
            "iteration_budget": attempt_budget,
            "iterations_used": attempt_iterations,
            "step_size_used": current_step_size,
            "gradient_inf_norm": gradient_inf_norm,
            "converged": converged,
            "failure": failure or (None if converged else "iteration_budget_exhausted"),
        })
        if converged:
            break
        weights = last_finite_weights
        if attempt < max_retries:
            retry_count += 1
            resumed_from_finite_checkpoint = True
            if failure in {"nonfinite_weights", "nonfinite_gradient"}:
                current_step_size *= 0.5
    if diagnostics is not None:
        diagnostics.update({
            "solver": "fista_adaptive_restart",
            "iterations": total_iterations,
            "total_iterations": total_iterations,
            "max_iterations": max_iterations,
            "gradient_inf_norm": gradient_inf_norm,
            "gradient_tolerance": gradient_tolerance,
            "converged": converged,
            "retry_count": retry_count,
            "resumed_from_finite_checkpoint": resumed_from_finite_checkpoint,
            "attempts": attempt_diagnostics,
        })
    if not converged:
        raise RuntimeError(
            f"multinomial logistic fit did not converge after {total_iterations} iterations "
            f"(gradient infinity norm {gradient_inf_norm:.6g}; tolerance {gradient_tolerance:.6g})"
        )
    logits = prediction_augmented @ weights
    if return_scores:
        return _softmax(logits)
    return np.argmax(logits, axis=1).astype(np.int64)


def predict_model(
    model: str,
    parameter: float | int | None,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    fit_diagnostics: list[dict[str, object]] | None = None,
    fit_context: str = "unspecified",
    training_predictions: list[np.ndarray] | None = None,
    return_scores: bool = False,
) -> np.ndarray:
    """Fit one classifier; optionally return an n-sample by 3 class-score matrix."""
    x_train = np.asarray(train_x, dtype=np.float64)
    x_test = np.asarray(test_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.int64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("training and test features must be compatible matrices")
    if y_train.shape != (x_train.shape[0],) or x_train.shape[0] == 0:
        raise ValueError("training labels must match nonempty training rows")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("model features must be finite")
    if not set(np.unique(y_train)).issuperset({0, 1, 2}):
        raise ValueError("all three classes must be represented in each training split")

    def record_training(prediction: np.ndarray) -> None:
        if training_predictions is not None:
            training_predictions.append(np.asarray(prediction, dtype=np.int64))

    if model == "majority":
        counts = np.bincount(y_train, minlength=3)
        predicted_class = int(np.argmax(counts))
        record_training(np.full(x_train.shape[0], predicted_class, dtype=np.int64))
        if return_scores:
            prior = counts.astype(np.float64) / y_train.size
            return np.repeat(prior[None, :], x_test.shape[0], axis=0)
        return np.full(x_test.shape[0], predicted_class, dtype=np.int64)
    if model == "nearest_centroid":
        centroids = np.vstack([x_train[y_train == label].mean(axis=0) for label in range(3)])
        train_distances = ((x_train[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        test_distances = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        record_training(np.argmin(train_distances, axis=1).astype(np.int64))
        if return_scores:
            return -test_distances
        return np.argmin(test_distances, axis=1).astype(np.int64)
    if model == "knn":
        neighbors = int(parameter) if parameter is not None else 0
        if neighbors < 1:
            raise ValueError("kNN requires a positive neighbor count")
        neighbors = min(neighbors, x_train.shape[0])
        def nearest_neighbor_scores(rows: np.ndarray) -> np.ndarray:
            class_fractions = []
            for row in rows:
                distances = ((x_train - row) ** 2).sum(axis=1)
                nearest = np.argsort(distances, kind="stable")[:neighbors]
                class_fractions.append(np.bincount(y_train[nearest], minlength=3) / neighbors)
            return np.asarray(class_fractions, dtype=np.float64)

        # This is an apparent in-sample score; each training row can be its own neighbor.
        training_scores = nearest_neighbor_scores(x_train)
        record_training(np.argmax(training_scores, axis=1).astype(np.int64))
        test_scores = nearest_neighbor_scores(x_test)
        return test_scores if return_scores else np.argmax(test_scores, axis=1).astype(np.int64)
    if model == "ridge":
        alpha = float(parameter) if parameter is not None else 0.0
        if alpha <= 0.0 or not math.isfinite(alpha):
            raise ValueError("ridge strength must be finite and positive")
        x_augmented = np.column_stack((np.ones(x_train.shape[0]), x_train))
        targets = np.eye(3, dtype=np.float64)[y_train]
        penalty = np.eye(x_augmented.shape[1], dtype=np.float64) * alpha
        penalty[0, 0] = 0.0
        gram = (x_augmented.T @ x_augmented) / x_augmented.shape[0] + penalty
        coefficients = np.linalg.solve(gram, (x_augmented.T @ targets) / x_augmented.shape[0])
        record_training(np.argmax(x_augmented @ coefficients, axis=1).astype(np.int64))
        test_augmented = np.column_stack((np.ones(x_test.shape[0]), x_test))
        decision_scores = test_augmented @ coefficients
        return decision_scores if return_scores else np.argmax(decision_scores, axis=1).astype(np.int64)
    if model == "logistic":
        alpha = float(parameter) if parameter is not None else 0.0
        fit_record: dict[str, object] = {}
        try:
            prediction = _predict_logistic(
                x_train, y_train, x_test, alpha, diagnostics=fit_record,
                return_training_predictions=training_predictions is not None,
                return_scores=return_scores,
            )
        except Exception:
            if fit_diagnostics is not None:
                fit_diagnostics.append({
                    "context": fit_context,
                    "model": "logistic",
                    "alpha": alpha,
                    "failed": True,
                    **fit_record,
                })
            raise
        if fit_diagnostics is not None:
            fit_diagnostics.append({
                "context": fit_context,
                "model": "logistic",
                "alpha": alpha,
                **fit_record,
            })
        if training_predictions is not None:
            n_test = x_test.shape[0]
            if return_scores:
                record_training(np.argmax(prediction[n_test:], axis=1).astype(np.int64))
            else:
                record_training(prediction[n_test:])
            prediction = prediction[:n_test]
        return prediction
    raise ValueError(f"unknown classifier: {model}")


def validate_outer_loso(
    partition_rows: list[dict[str, str]], participants: set[str]
) -> list[dict[str, str]]:
    """Validate one held-out participant per outer fold and return role maps."""
    rows = [row for row in partition_rows if row.get("level") == "outer"]
    if not participants or not rows:
        raise ValueError("outer LOSO manifest and participant set must be nonempty")
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        fold = str(row.get("outer_fold", ""))
        participant = str(row.get("participant_id", ""))
        role = str(row.get("role", ""))
        if not fold or not participant or role not in {"train", "test"}:
            raise ValueError("invalid outer LOSO row")
        if participant in grouped[fold]:
            raise ValueError("participant appears more than once in an outer fold")
        grouped[fold][participant] = role
    if not grouped:
        raise ValueError("outer LOSO manifest has no folds")
    tested: Counter[str] = Counter()
    folds: list[dict[str, str]] = []
    for fold in sorted(grouped, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        roles = grouped[fold]
        if set(roles) != participants or sum(role == "test" for role in roles.values()) != 1:
            raise ValueError("each outer fold must assign every participant and hold out exactly one")
        held_out = next(person for person, role in roles.items() if role == "test")
        tested[held_out] += 1
        folds.append(roles)
    if set(tested) != participants or any(tested[person] != 1 for person in participants):
        raise ValueError("every participant must appear exactly once as an outer test group")
    return folds


def validate_nested_partitions(
    partition_rows: list[dict[str, str]], participants: set[str]
) -> tuple[list[dict[str, str]], dict[str, list[tuple[set[str], set[str]]]]]:
    outer_folds = validate_outer_loso(partition_rows, participants)
    outer_ids = sorted(
        {str(row["outer_fold"]) for row in partition_rows if row.get("level") == "outer"},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    inner: dict[str, list[tuple[set[str], set[str]]]] = {}
    for fold_id, roles in zip(outer_ids, outer_folds):
        expected_development = {participant for participant, role in roles.items() if role == "train"}
        fold_rows = [row for row in partition_rows if row.get("level") == "inner" and str(row.get("outer_fold")) == fold_id]
        grouped: dict[str, dict[str, str]] = defaultdict(dict)
        for row in fold_rows:
            inner_id = str(row.get("inner_fold", ""))
            participant = str(row.get("participant_id", ""))
            role = str(row.get("role", ""))
            if not inner_id or role not in {"train", "validation"} or participant in grouped[inner_id]:
                raise ValueError("invalid or duplicate nested inner-fold row")
            grouped[inner_id][participant] = role
        if len(grouped) != 4:
            raise ValueError("each outer fold must have four participant-grouped inner folds")
        splits = []
        for inner_id in sorted(grouped, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
            assignment = grouped[inner_id]
            if set(assignment) != expected_development:
                raise ValueError("inner fold includes an outer test subject or omits a development subject")
            train_ids = {person for person, role in assignment.items() if role == "train"}
            validation_ids = {person for person, role in assignment.items() if role == "validation"}
            if train_ids & validation_ids or train_ids | validation_ids != expected_development:
                raise ValueError("inner train and validation groups must be disjoint and exhaustive")
            if len(validation_ids) not in {3, 4}:
                raise ValueError("inner validation fold must contain three or four participants")
            splits.append((train_ids, validation_ids))
        inner[fold_id] = splits
    return outer_folds, inner


def _segment_mask(times: np.ndarray, label_rows: list[dict[str, str]]) -> list[np.ndarray]:
    ordered = sorted(label_rows, key=lambda row: LABEL_TO_INT.get(row["label"], 99))
    if len(ordered) != 3 or [row["label"] for row in ordered] != list(LABEL_TO_INT):
        raise ValueError("each eligible recording must have one ordered row for each temporal third")
    if times.size < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise ValueError("signal timestamps must be finite and strictly increasing")
    start = float(times[0])
    end = float(times[-1])
    if end <= start:
        raise ValueError("recording duration must be positive")
    duration = end - start
    bounds = (start, start + duration / 3.0, start + 2.0 * duration / 3.0, end)
    masks: list[np.ndarray] = []
    for index, label_row in enumerate(ordered):
        if not math.isclose(float(label_row["t_start_s"]), bounds[index], rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("label start differs from the source-precision temporal split")
        if not math.isclose(float(label_row["t_end_s"]), bounds[index + 1], rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("label end differs from the source-precision temporal split")
        if index < 2:
            mask = (times >= bounds[index]) & (times < bounds[index + 1])
        else:
            mask = (times >= bounds[index]) & (times <= bounds[index + 1])
        if int(mask.sum()) != int(label_row["n_samples_in_segment"]):
            raise ValueError("source-precision sample count differs from the frozen label manifest")
        masks.append(mask)
    return masks


def _trim_valid_run_edges(times: np.ndarray, mask: np.ndarray, samples_per_side: int) -> np.ndarray:
    """Trim each contiguous valid run independently, never across acquisition gaps."""
    if not isinstance(samples_per_side, int) or samples_per_side < 0:
        raise ValueError("edge trim must be a nonnegative integer number of samples")
    time_values = np.asarray(times, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if time_values.ndim != 1 or valid.shape != time_values.shape:
        raise ValueError("edge trim timestamps and mask must be compatible one-dimensional arrays")
    if samples_per_side == 0:
        return valid.copy()
    selected = np.flatnonzero(valid)
    output = np.zeros(valid.shape, dtype=bool)
    if selected.size == 0:
        return output
    run_start = 0
    run_positions: list[tuple[int, int]] = []
    for position in range(1, selected.size):
        previous_index = int(selected[position - 1])
        current_index = int(selected[position])
        delta = float(time_values[current_index] - time_values[previous_index])
        if current_index != previous_index + 1 or not (0.0 < delta <= 0.0015):
            run_positions.append((run_start, position))
            run_start = position
    run_positions.append((run_start, selected.size))
    for start, stop in run_positions:
        run = selected[start:stop]
        if run.size > 2 * samples_per_side:
            output[run[samples_per_side:-samples_per_side]] = True
    return output


def extract_filtered_features(
    labels_path: Path, signals_root: Path, additional_edge_trim_samples: int = 0
) -> tuple[list[dict[str, str]], dict[str, int], str, list[dict[str, np.ndarray | float]]]:
    """Create in-memory features for each predeclared filter on a common mask."""
    if not isinstance(additional_edge_trim_samples, int) or additional_edge_trim_samples < 0:
        raise ValueError("additional edge trim must be a nonnegative integer number of samples")
    label_rows = _read_csv(labels_path)
    if len(label_rows) != 414:
        raise ValueError("expected the frozen 414-row temporal-position proxy label table")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        if row.get("recording_id") not in {"R1", "R2", "R3", "R4", "R5"}:
            raise ValueError("only R1-R5 proxy rows are eligible for model training")
        if row.get("label_kind") != "temporal_recording_position_proxy_not_fatigue_ground_truth":
            raise ValueError("label table contains a target other than the frozen temporal proxy")
        key = tuple(row[field] for field in KEY_FIELDS)
        grouped[key].append(row)
    if len(grouped) != 138:
        raise ValueError("expected 138 eligible R1-R5 recordings")

    records: list[dict[str, str]] = []
    signal_segments: list[dict[str, np.ndarray | float]] = []
    signal_hashes: list[str] = []
    sample_counts = Counter()
    for key in sorted(grouped):
        participant, recording, leg, set_index = key
        if recording != f"R{set_index}":
            raise ValueError("recording ID and series index do not agree")
        if leg not in {"left", "right"}:
            raise ValueError("unexpected laterality in the frozen proxy table")
        npz_path = Path(signals_root) / f"{participant}_{recording}_{leg}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError("a private pseudonymized filter trace is missing")
        signal_hashes.append(_sha256(npz_path))
        with np.load(npz_path, allow_pickle=False) as arrays:
            required = {"times_s", "adc_original", "common_comparison_mask", *FILTER_BRANCHES[1:]}
            if not required.issubset(arrays.files):
                raise ValueError("a private filter trace is missing a required branch or timestamp")
            times = np.asarray(arrays["times_s"], dtype=np.float64)
            common = np.asarray(arrays["common_comparison_mask"], dtype=bool).copy()
            original = np.asarray(arrays["adc_original"], dtype=np.float64)
            if common.shape != times.shape or original.shape != times.shape:
                raise ValueError("stored signal, timestamps, and common mask have different shapes")
            masks = _segment_mask(times, grouped[key])
            if additional_edge_trim_samples:
                common = _trim_valid_run_edges(times, common, additional_edge_trim_samples)
            deltas = np.diff(times)
            contiguous_deltas = deltas[deltas <= 0.0015]
            if contiguous_deltas.size == 0:
                raise ValueError("recording has no timestamp intervals available for a sampling-rate estimate")
            estimated_fs_hz = float(1.0 / np.median(contiguous_deltas))
            native_branch_values = {"control": arrays["adc_original"]}
            native_branch_values.update({branch: arrays[branch] for branch in FILTER_BRANCHES[1:]})
            branch_values = {"control": original}
            branch_values.update({branch: np.asarray(arrays[branch], dtype=np.float64) for branch in FILTER_BRANCHES[1:]})
            for branch in FILTER_BRANCHES:
                values = branch_values[branch]
                if values.shape != times.shape or not np.isfinite(values[common]).all():
                    raise ValueError("a branch has mismatched or nonfinite values on the common mask")
                if branch != "control":
                    applied_key = f"applied_{branch}"
                    if applied_key not in arrays.files or np.any(common & ~np.asarray(arrays[applied_key], dtype=bool)):
                        raise ValueError("common mask contains samples outside a filtered branch's valid region")
            for row, temporal_mask in zip(sorted(grouped[key], key=lambda item: LABEL_TO_INT[item["label"]]), masks):
                use_mask = temporal_mask & common
                if int(use_mask.sum()) < 2:
                    raise ValueError("a temporal segment has fewer than two common valid samples")
                original_indices = np.flatnonzero(use_mask).astype(np.int64)
                contiguous_pairs = _contiguous_pair_indices(times, original_indices.tolist())
                values = branch_values
                base = {
                    "participant_id": participant,
                    "recording_id": recording,
                    "leg": leg,
                    "set_index": set_index,
                    "label": row["label"],
                }
                for branch in FILTER_BRANCHES:
                    branch_feature = _features(
                        values[branch][use_mask].tolist(), contiguous_pairs
                    )
                    for feature in PREDICTOR_COLUMNS:
                        base[f"{branch}__{feature}"] = format(branch_feature[feature], ".12g")
                    sample_counts[branch] += int(use_mask.sum())
                segment_traces: dict[str, np.ndarray | float] = {
                    "fs_hz": estimated_fs_hz,
                    "contiguous_pair_indices": np.asarray(contiguous_pairs, dtype=np.int64),
                    "times_s": times[use_mask].astype(np.float64, copy=True),
                }
                for branch in FILTER_BRANCHES:
                    segment_traces[branch] = np.asarray(native_branch_values[branch][use_mask]).copy()
                signal_segments.append(segment_traces)
                records.append(base)
    if len(records) != 414 or len({(r["participant_id"], r["recording_id"], r["leg"], r["set_index"], r["label"]) for r in records}) != 414:
        raise ValueError("filtered feature extraction did not preserve the 414 unique labeled segments")
    digest = hashlib.sha256("".join(sorted(signal_hashes)).encode("ascii")).hexdigest()
    return records, dict(sample_counts), digest, signal_segments


def _indices_for_groups(rows: list[dict[str, str]], groups: set[str]) -> np.ndarray:
    return np.asarray([index for index, row in enumerate(rows) if row["participant_id"] in groups], dtype=np.int64)


def _participant_macro_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    rows: list[dict[str, str]],
    participant_ids: Iterable[str],
) -> list[float]:
    scores = []
    for participant in sorted(participant_ids):
        indices = np.asarray([i for i, row in enumerate(rows) if row["participant_id"] == participant], dtype=np.int64)
        scores.append(macro_f1(truth[indices], prediction[indices]))
    return scores


def _select_parameter(
    model: str,
    candidates: tuple[float | int | None, ...],
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, str]],
    inner_splits: list[tuple[set[str], set[str]]],
    fit_diagnostics: list[dict[str, object]] | None = None,
    *,
    filter_branch: str = "",
    outer_fold: str = "",
    additional_edge_trim_samples: int = 0,
    generalization_gap_diagnostics: list[dict[str, str]] | None = None,
) -> tuple[float | int | None, float]:
    candidate_scores: list[tuple[float, float | int | None]] = []
    for candidate in candidates:
        participant_scores: list[float] = []
        for inner_index, (train_ids, validation_ids) in enumerate(inner_splits, start=1):
            train_indices = _indices_for_groups(rows, train_ids)
            validation_indices = _indices_for_groups(rows, validation_ids)
            train_x, validation_x = standardize_train_test(x[train_indices], x[validation_indices])
            training_predictions: list[np.ndarray] = []
            prediction = predict_model(
                model,
                candidate,
                train_x,
                y[train_indices],
                validation_x,
                fit_diagnostics=fit_diagnostics,
                fit_context="inner_validation",
                training_predictions=(
                    training_predictions if generalization_gap_diagnostics is not None else None
                ),
            )
            validation_scores = _participant_macro_scores(
                y[validation_indices], prediction,
                [rows[index] for index in validation_indices], validation_ids,
            )
            participant_scores.extend(validation_scores)
            if generalization_gap_diagnostics is not None:
                if len(training_predictions) != 1:
                    raise ValueError("training score audit did not receive exactly one in-sample prediction vector")
                train_scores = _participant_macro_scores(
                    y[train_indices], training_predictions[0],
                    [rows[index] for index in train_indices], train_ids,
                )
                train_mean = float(np.mean(train_scores))
                validation_mean = float(np.mean(validation_scores))
                gap = train_mean - validation_mean
                generalization_gap_diagnostics.append({
                    "filter_branch": filter_branch,
                    "model": model,
                    "parameter": "fixed" if candidate is None else str(candidate),
                    "outer_fold": outer_fold,
                    "inner_fold": str(inner_index),
                    "additional_edge_trim_samples_per_run_side": str(additional_edge_trim_samples),
                    "training_participant_count": str(len(train_ids)),
                    "validation_participant_count": str(len(validation_ids)),
                    "training_macro_f1_mean_subject_apparent": f"{train_mean:.6f}",
                    "validation_macro_f1_mean_subject": f"{validation_mean:.6f}",
                    "apparent_generalization_gap_macro_f1": f"{gap:.6f}",
                    "positive_gap_flag": str(gap > 0.0).lower(),
                    "training_score_note": "apparent in-sample score; kNN includes training rows in its neighbor reference",
                })
        if not participant_scores:
            raise ValueError("inner validation produced no participant scores")
        candidate_scores.append((float(np.mean(participant_scores)), candidate))
    best_score = max(score for score, _ in candidate_scores)
    tied = [candidate for score, candidate in candidate_scores if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)]
    # Conservative deterministic tie-break: stronger L2 or more neighbours.
    if model in {"ridge", "logistic", "knn"}:
        selected = max(tied, key=float)
    else:
        selected = tied[0]
    return selected, best_score


def choose_configuration(inner_scores: dict[tuple[str, str], float]) -> tuple[str, str]:
    """Select a filter/model pair from inner scores; never inspect outer scores."""
    if not inner_scores:
        raise ValueError("inner configuration scores must be nonempty")
    for (branch, model), score in inner_scores.items():
        if branch not in FILTER_BRANCHES or model not in MODEL_CANDIDATES:
            raise ValueError("unknown filter/model configuration")
        if not math.isfinite(float(score)):
            raise ValueError("inner configuration scores must be finite")
    best_score = max(float(score) for score in inner_scores.values())
    tied = [
        configuration
        for configuration, score in inner_scores.items()
        if math.isclose(float(score), best_score, rel_tol=0.0, abs_tol=1e-12)
    ]
    return min(
        tied,
        key=lambda configuration: (
            MODEL_COMPLEXITY[configuration[1]],
            FILTER_BRANCHES.index(configuration[0]),
        ),
    )


def outer_fold_seed(base_seed: int, ordered_fold_ids: list[str], fold_id: str) -> int:
    """Derive a stable seed from the manifest order without parsing fold labels."""
    if len(set(ordered_fold_ids)) != len(ordered_fold_ids) or fold_id not in ordered_fold_ids:
        raise ValueError("outer fold must appear exactly once in the ordered manifest IDs")
    return int(base_seed) + ordered_fold_ids.index(fold_id)


def _bootstrap_interval(values: list[float], seed: int, replicates: int) -> tuple[float, float]:
    if len(values) < 2 or replicates < 100:
        raise ValueError("participant bootstrap needs at least two participants and 100 replicates")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(replicates, data.size))
    means = data[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _summary_row(
    branch: str,
    model: str,
    macro_scores: list[float],
    balanced_scores: list[float],
    recalls: dict[str, list[float]],
    tested_segments: int,
    bootstrap_replicates: int,
    additional_edge_trim_samples: int = 0,
) -> dict[str, str]:
    low_f1, high_f1 = _bootstrap_interval(macro_scores, seed=20261014, replicates=bootstrap_replicates)
    low_ba, high_ba = _bootstrap_interval(balanced_scores, seed=20261015, replicates=bootstrap_replicates)
    result = {
        "filter_branch": branch,
        "model": model,
        "target": "within-recording temporal-position proxy; not fatigue ground truth",
        "outer_test_participants": str(len(macro_scores)),
        "outer_test_segments": str(tested_segments),
        "macro_f1_mean_subject": f"{statistics.mean(macro_scores):.6f}",
        "macro_f1_sd_subject": f"{statistics.stdev(macro_scores):.6f}",
        "macro_f1_median_subject": f"{statistics.median(macro_scores):.6f}",
        "macro_f1_min_subject": f"{min(macro_scores):.6f}",
        "macro_f1_max_subject": f"{max(macro_scores):.6f}",
        "macro_f1_ci95_bootstrap_low_subject": f"{low_f1:.6f}",
        "macro_f1_ci95_bootstrap_high_subject": f"{high_f1:.6f}",
        "balanced_accuracy_mean_subject": f"{statistics.mean(balanced_scores):.6f}",
        "balanced_accuracy_sd_subject": f"{statistics.stdev(balanced_scores):.6f}",
        "balanced_accuracy_ci95_bootstrap_low_subject": f"{low_ba:.6f}",
        "balanced_accuracy_ci95_bootstrap_high_subject": f"{high_ba:.6f}",
        "bootstrap_replicates": str(bootstrap_replicates),
        "augmentation": "none; synthetic perturbations not applied in this benchmark",
        "additional_edge_trim_samples_per_run_side": str(additional_edge_trim_samples),
        "interpretation": "exploratory generalization of stored-channel descriptors to temporal thirds only; not physiological filtering quality",
    }
    for label, label_scores in recalls.items():
        result[f"recall_{label}_mean_subject"] = f"{statistics.mean(label_scores):.6f}"
    return result


def _assert_public_schema(rows: list[dict[str, str]]) -> None:
    forbidden = ("participant_id", "recording_id", "filename", "source_id", "participant_name")
    for row in rows:
        lowered = {str(key).casefold() for key in row}
        if any(any(token in field for token in forbidden) for field in lowered):
            raise ValueError("public model aggregate schema contains an identifying field")
        if any(any(token in str(value).casefold() for token in forbidden) for value in row.values()):
            raise ValueError("public model aggregate contains an identifying value")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty model aggregate")
    _assert_public_schema(rows)
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_private_fold_metrics_csv(
    path: Path, rows: list[dict[str, object]], private_root: Path
) -> None:
    """Write linked fold-level results only within restricted local storage."""
    if not rows:
        raise ValueError("refusing to write an empty fold-metrics table")
    root = private_root.resolve()
    destination = path.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("fold-metrics output is outside the private root") from exc
    if destination == root or not destination.name:
        raise ValueError("fold-metrics output must be a file inside the private root")

    fields = list(rows[0])
    if not fields or any(list(row) != fields for row in rows):
        raise ValueError("fold-metrics rows must share a nonempty schema")
    forbidden_fields = (
        "participant_id", "subject_id", "recording_id", "filename",
        "full_name", "participant_name", "source_id", "identity_key",
    )
    if any(any(token in field.casefold() for token in forbidden_fields) for field in fields):
        raise ValueError("fold-metrics output cannot contain an identity field")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def default_output_directory(project_root: Path) -> Path:
    """Keep public aggregates in the project's established derived-data folder."""
    return Path(project_root) / "datasets" / "derivados"


def build_run_summary(
    *,
    summary_path: Path,
    parameter_path: Path,
    selected_pipeline_path: Path,
    selection_frequency_path: Path,
    augmentation_path: Path,
    noise_mix_path: Path,
    generalization_gap_path: Path,
    additional_edge_trim_samples: int,
    receipt_path: Path,
    participants: int,
    recordings: int,
    segments: int,
    filters: int,
    models: int,
    outer_folds: int,
) -> dict[str, object]:
    """Report fold-level fits separately from segment-level held-out predictions."""
    return {
        "summary_csv": str(summary_path),
        "parameter_csv": str(parameter_path),
        "selected_pipeline_csv": str(selected_pipeline_path),
        "selection_frequency_csv": str(selection_frequency_path),
        "augmentation_csv": str(augmentation_path),
        "noise_mix_csv": str(noise_mix_path),
        "generalization_gap_audit_csv": str(generalization_gap_path),
        "additional_edge_trim_samples_per_run_side": additional_edge_trim_samples,
        "receipt": str(receipt_path),
        "participants": participants,
        "recordings": recordings,
        "segments": segments,
        "filters": filters,
        "models": models,
        "aggregate_rows": filters * models,
        "outer_model_fits": filters * models * outer_folds,
        "nested_selected_outer_fits": outer_folds,
        "train_augmentation_outer_fits": outer_folds,
        "outer_test_segments_per_combination": segments,
        "target_is_fatigue": False,
        "synthetic_augmentation_applied_to_train": True,
        "study_signals_modified": False,
    }


def run_benchmark(
    labels_path: Path,
    partitions_path: Path,
    signals_root: Path,
    output_dir: Path,
    bootstrap_replicates: int = 2000,
    additional_edge_trim_samples: int = 0,
) -> dict[str, object]:
    labels = _read_csv(labels_path)
    partitions = _read_csv(partitions_path)
    records, common_sample_counts, signal_digest, signal_segments = extract_filtered_features(
        labels_path, signals_root,
        additional_edge_trim_samples=additional_edge_trim_samples,
    )
    participants = {row["participant_id"] for row in records}
    outer_folds, inner_folds = validate_nested_partitions(partitions, participants)
    outer_rows = [row for row in partitions if row.get("level") == "outer"]
    fold_ids = sorted(
        {str(row["outer_fold"]) for row in outer_rows},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    if len(fold_ids) != 15 or len(participants) != 15:
        raise ValueError("the frozen analysis requires 15 outer folds and 15 participant groups")

    y = np.asarray([LABEL_TO_INT[row["label"]] for row in records], dtype=np.int64)
    if Counter(row["label"] for row in records) != Counter({"temprana": 138, "transicion": 138, "tardia": 138}):
        raise ValueError("frozen proxy class totals changed")
    participant_rows = {participant: [row for row in records if row["participant_id"] == participant] for participant in participants}
    summary_rows: list[dict[str, str]] = []
    parameter_rows: list[dict[str, str]] = []
    branch_matrices: dict[str, np.ndarray] = {}
    tuned_parameters: dict[str, dict[tuple[str, str], float | int | None]] = {
        fold_id: {} for fold_id in fold_ids
    }
    inner_scores_by_fold: dict[str, dict[tuple[str, str], float]] = {
        fold_id: {} for fold_id in fold_ids
    }
    fit_count = 0
    logistic_fit_diagnostics: list[dict[str, object]] = []
    generalization_gap_diagnostics: list[dict[str, str]] = []
    for branch in FILTER_BRANCHES:
        x = np.asarray(
            [[float(row[f"{branch}__{feature}"]) for feature in PREDICTOR_COLUMNS] for row in records],
            dtype=np.float64,
        )
        if not np.isfinite(x).all():
            raise ValueError("filtered feature matrix contains nonfinite values")
        branch_matrices[branch] = x
        for model, candidates in MODEL_CANDIDATES.items():
            macro_scores: list[float] = []
            balanced_scores: list[float] = []
            recalls = {label: [] for label in LABEL_TO_INT}
            parameter_counts: Counter[str] = Counter()
            tested_segments = 0
            for fold_id, outer_roles in zip(fold_ids, outer_folds):
                train_ids = {participant for participant, role in outer_roles.items() if role == "train"}
                test_ids = {participant for participant, role in outer_roles.items() if role == "test"}
                if len(test_ids) != 1:
                    raise ValueError("an outer fold must hold out exactly one participant")
                test_id = next(iter(test_ids))
                parameter, inner_score = _select_parameter(
                    model, candidates, x, y, records, inner_folds[fold_id],
                    fit_diagnostics=logistic_fit_diagnostics,
                    filter_branch=branch,
                    outer_fold=fold_id,
                    additional_edge_trim_samples=additional_edge_trim_samples,
                    generalization_gap_diagnostics=generalization_gap_diagnostics,
                )
                tuned_parameters[fold_id][(branch, model)] = parameter
                inner_scores_by_fold[fold_id][(branch, model)] = inner_score
                parameter_counts["fixed" if parameter is None else str(parameter)] += 1
                train_indices = _indices_for_groups(records, train_ids)
                test_indices = _indices_for_groups(records, test_ids)
                train_x, test_x = standardize_train_test(x[train_indices], x[test_indices])
                prediction = predict_model(
                    model, parameter, train_x, y[train_indices], test_x,
                    fit_diagnostics=logistic_fit_diagnostics,
                    fit_context="outer_grid",
                )
                truth = y[test_indices]
                macro_scores.append(macro_f1(truth, prediction))
                balanced_scores.append(balanced_accuracy(truth, prediction))
                for label, class_index in LABEL_TO_INT.items():
                    present = truth == class_index
                    recalls[label].append(float(np.mean(prediction[present] == class_index)))
                tested_segments += int(test_indices.size)
                fit_count += 1
                if len(participant_rows[test_id]) != int(test_indices.size):
                    raise ValueError("outer test indices do not correspond to exactly one whole participant")
            if len(macro_scores) != 15 or tested_segments != 414:
                raise ValueError("outer evaluation must score each of 15 people and all 414 proxy segments once")
            summary_rows.append(
                _summary_row(branch, model, macro_scores, balanced_scores, recalls,
                             tested_segments, bootstrap_replicates,
                             additional_edge_trim_samples)
            )
            for parameter, count in sorted(parameter_counts.items()):
                parameter_rows.append({
                    "filter_branch": branch,
                    "model": model,
                    "selected_parameter": parameter,
                    "outer_folds": str(count),
                    "additional_edge_trim_samples_per_run_side": str(additional_edge_trim_samples),
                })

    selected_macro_scores: list[float] = []
    selected_balanced_scores: list[float] = []
    selected_recalls = {label: [] for label in LABEL_TO_INT}
    selected_configuration_counts: Counter[tuple[str, str, str]] = Counter()
    augmented_macro_scores: list[float] = []
    augmented_balanced_scores: list[float] = []
    augmented_recalls = {label: [] for label in LABEL_TO_INT}
    synthetic_scenario_counts: Counter[str] = Counter()
    selected_segments = 0
    selected_fit_count = 0
    augmentation_fit_count = 0
    for fold_id, outer_roles in zip(fold_ids, outer_folds):
        if set(inner_scores_by_fold[fold_id]) != {
            (branch, model) for branch in FILTER_BRANCHES for model in MODEL_CANDIDATES
        }:
            raise ValueError("nested selection does not contain every candidate configuration")
        branch, model = choose_configuration(inner_scores_by_fold[fold_id])
        parameter = tuned_parameters[fold_id][(branch, model)]
        train_ids = {participant for participant, role in outer_roles.items() if role == "train"}
        test_ids = {participant for participant, role in outer_roles.items() if role == "test"}
        train_indices = _indices_for_groups(records, train_ids)
        test_indices = _indices_for_groups(records, test_ids)
        train_x, test_x = standardize_train_test(
            branch_matrices[branch][train_indices], branch_matrices[branch][test_indices]
        )
        prediction = predict_model(
            model, parameter, train_x, y[train_indices], test_x,
            fit_diagnostics=logistic_fit_diagnostics,
            fit_context="outer_nested_selected",
        )
        truth = y[test_indices]
        selected_macro_scores.append(macro_f1(truth, prediction))
        selected_balanced_scores.append(balanced_accuracy(truth, prediction))
        for label, class_index in LABEL_TO_INT.items():
            present = truth == class_index
            selected_recalls[label].append(float(np.mean(prediction[present] == class_index)))

        # Synthetic copies are generated only from this outer training index set.
        augmented_features, augmented_labels, fold_scenarios = augment_training_segments(
            signal_segments,
            train_indices,
            branch,
            y,
            seed=outer_fold_seed(20261014, fold_ids, fold_id),
        )
        synthetic_scenario_counts.update(fold_scenarios)
        augmented_train_x = np.vstack((branch_matrices[branch][train_indices], augmented_features))
        augmented_train_y = np.concatenate((y[train_indices], augmented_labels))
        augmented_train_scaled, augmented_test_scaled = standardize_train_test(
            augmented_train_x, branch_matrices[branch][test_indices]
        )
        augmented_prediction = predict_model(
            model, parameter, augmented_train_scaled, augmented_train_y, augmented_test_scaled,
            fit_diagnostics=logistic_fit_diagnostics,
            fit_context="outer_train_augmentation",
        )
        augmented_macro_scores.append(macro_f1(truth, augmented_prediction))
        augmented_balanced_scores.append(balanced_accuracy(truth, augmented_prediction))
        for label, class_index in LABEL_TO_INT.items():
            present = truth == class_index
            augmented_recalls[label].append(float(np.mean(augmented_prediction[present] == class_index)))
        selected_segments += int(test_indices.size)
        selected_fit_count += 1
        augmentation_fit_count += 1
        selected_configuration_counts[(branch, model, "fixed" if parameter is None else str(parameter))] += 1
    if selected_fit_count != 15 or augmentation_fit_count != 15 or selected_segments != 414:
        raise ValueError("nested selected and augmented pipelines must test all 15 participants and 414 segments once")

    summary_path = Path(output_dir) / "RENDIMIENTO_PROXY_TEMPORAL_LOSO_AGREGADO.csv"
    parameter_path = Path(output_dir) / "SELECCION_PARAMETROS_PROXY_TEMPORAL_AGREGADA.csv"
    selected_pipeline_path = Path(output_dir) / "RENDIMIENTO_PIPELINE_SELECCIONADO_NESTED_LOSO.csv"
    selection_frequency_path = Path(output_dir) / "FRECUENCIA_CONFIGURACION_NESTED_PROXY.csv"
    augmentation_path = Path(output_dir) / "RENDIMIENTO_AUGMENTACION_PROXY_TRAIN_ONLY.csv"
    noise_mix_path = Path(output_dir) / "MEZCLA_RUIDO_SINTETICO_TRAIN_ONLY.csv"
    generalization_gap_path = Path(output_dir) / "AUDITORIA_BRECHA_TRAIN_VALIDACION_INTERNA.csv"
    for row in summary_rows:
        row["additional_edge_trim_samples_per_run_side"] = str(additional_edge_trim_samples)
    _write_csv(summary_path, summary_rows)
    _write_csv(parameter_path, parameter_rows)
    _write_csv(
        selected_pipeline_path,
        [_summary_row("inner_cv_selected", "nested_selected_pipeline", selected_macro_scores,
                      selected_balanced_scores, selected_recalls, selected_segments, bootstrap_replicates,
                      additional_edge_trim_samples)],
    )
    _write_csv(selection_frequency_path, [
        {
            "filter_branch": branch,
            "model": model,
            "selected_parameter": parameter,
            "outer_folds_selected": str(count),
        }
        for (branch, model, parameter), count in sorted(selected_configuration_counts.items())
    ])
    paired_macro_deltas = [augmented - base for augmented, base in zip(augmented_macro_scores, selected_macro_scores)]
    paired_balanced_deltas = [augmented - base for augmented, base in zip(augmented_balanced_scores, selected_balanced_scores)]
    macro_delta_low, macro_delta_high = _bootstrap_interval(paired_macro_deltas, seed=20261016, replicates=bootstrap_replicates)
    balanced_delta_low, balanced_delta_high = _bootstrap_interval(paired_balanced_deltas, seed=20261017, replicates=bootstrap_replicates)
    base_augmentation_row = _summary_row(
        "inner_cv_selected_per_fold", "nested_selected_pipeline",
        selected_macro_scores, selected_balanced_scores, selected_recalls,
        selected_segments, bootstrap_replicates, additional_edge_trim_samples,
    )
    base_augmentation_row["condition"] = "selected_pipeline_without_augmentation"
    base_augmentation_row["augmentation"] = "no synthetic copies; validation and outer test retain original traces"
    augmented_row = _summary_row(
        "inner_cv_selected_per_fold", "nested_selected_pipeline",
        augmented_macro_scores, augmented_balanced_scores, augmented_recalls,
        selected_segments, bootstrap_replicates, additional_edge_trim_samples,
    )
    augmented_row["condition"] = "same_selected_pipeline_with_train_only_augmentation"
    augmented_row["augmentation"] = "one simulated copy per outer-training segment; validation and outer test untouched"
    delta_fields = {
        "paired_delta_macro_f1_mean_subject": f"{statistics.mean(paired_macro_deltas):.6f}",
        "paired_delta_macro_f1_ci95_low_subject": f"{macro_delta_low:.6f}",
        "paired_delta_macro_f1_ci95_high_subject": f"{macro_delta_high:.6f}",
        "paired_delta_balanced_accuracy_mean_subject": f"{statistics.mean(paired_balanced_deltas):.6f}",
        "paired_delta_balanced_accuracy_ci95_low_subject": f"{balanced_delta_low:.6f}",
        "paired_delta_balanced_accuracy_ci95_high_subject": f"{balanced_delta_high:.6f}",
    }
    base_augmentation_row.update({key: "" for key in delta_fields})
    augmented_row.update(delta_fields)
    _write_csv(augmentation_path, [base_augmentation_row, augmented_row])
    scenario_metadata = {name: params for name, _kind, params in synthetic_noise_scenarios(float(signal_segments[0]["fs_hz"]))}
    _write_csv(noise_mix_path, [
        {
            "synthetic_scenario": name,
            "parameters": json.dumps(scenario_metadata[name], sort_keys=True),
            "training_copies_across_outer_folds": str(synthetic_scenario_counts[name]),
            "use": "outer-training copies only; validation and test remain original",
        }
        for name in scenario_metadata
    ])
    _write_csv(generalization_gap_path, generalization_gap_diagnostics)
    retry_count = sum(int(row.get("retry_count", 0)) for row in logistic_fit_diagnostics)
    retried_fit_count = sum(int(row.get("retry_count", 0)) > 0 for row in logistic_fit_diagnostics)
    gap_values = [float(row["apparent_generalization_gap_macro_f1"]) for row in generalization_gap_diagnostics]
    receipt = {
        "protocol": "proxy_temporal_registro_v1_nested_loso_numpy_v1",
        "target_interpretation": "within-recording temporal position only; not fatigue, repetitions, active time, technical failure, or physiological ground truth",
        "participants": len(participants),
        "recordings": 138,
        "segments": len(records),
        "class_counts": dict(Counter(row["label"] for row in records)),
        "outer_folds": len(outer_folds),
        "inner_folds": sum(len(value) for value in inner_folds.values()),
        "filters": list(FILTER_BRANCHES),
        "additional_edge_trim_samples_per_contiguous_run_side": additional_edge_trim_samples,
        "models": list(MODEL_CANDIDATES),
        "fit_count_outer": fit_count,
        "nested_selected_outer_fits": selected_fit_count,
        "nested_selected_outer_segments": selected_segments,
        "nested_selection_frequencies": [
            {"filter_branch": branch, "model": model, "parameter": parameter, "outer_folds": count}
            for (branch, model, parameter), count in sorted(selected_configuration_counts.items())
        ],
        "common_valid_samples_by_filter": common_sample_counts,
        "labels_sha256": _sha256(labels_path),
        "partitions_sha256": _sha256(partitions_path),
        "combined_private_filter_trace_digest": signal_digest,
        "synthetic_augmentation": "one simulated copy per outer-training segment; seven predeclared scenarios balanced by fold; validation/test untouched",
        "synthetic_scenario_counts": dict(synthetic_scenario_counts),
        "synthetic_training_copies_total": sum(synthetic_scenario_counts.values()),
        "augmentation_outer_fits": augmentation_fit_count,
        "synthetic_validation_or_test_copies": 0,
        "study_signal_files_modified": False,
        "participant_ids_or_predictions_exported": False,
        "fatigue_prediction_performed": False,
        "generalization_gap_audit": {
            "csv": str(generalization_gap_path),
            "rows": len(generalization_gap_diagnostics),
            "metric": "apparent in-sample participant-mean macro-F1 minus inner-validation participant-mean macro-F1",
            "positive_gap_rows": sum(float(row["apparent_generalization_gap_macro_f1"]) > 0.0 for row in generalization_gap_diagnostics),
            "mean_apparent_gap_macro_f1": statistics.mean(gap_values) if gap_values else None,
            "maximum_apparent_gap_macro_f1": max(gap_values) if gap_values else None,
            "training_metric_limit": "diagnostic only; kNN training predictions include training rows in the neighbor reference",
            "outer_test_used_for_tuning_or_recovery": False,
        },
        "automatic_training_recovery": {
            "logistic_solver": "resume from last finite weight checkpoint; increase iteration budget geometrically; halve step size after a nonfinite update; never return a nonconverged fit",
            "maximum_retry_count_per_fit": LOGISTIC_MAX_RETRIES,
            "retry_iteration_multiplier": LOGISTIC_RETRY_MULTIPLIER,
            "retried_fits": retried_fit_count,
            "total_retries": retry_count,
            "overfitting_response": "nested inner validation selects the model configuration; apparent train-validation gaps are reported without using outer-test results to refit",
        },
        "logistic_fit_diagnostics": {
            "solver": "fista_adaptive_restart",
            "stopping_rule": "gradient infinity norm <= 1e-6",
            "initial_iteration_budget_per_fit": LOGISTIC_INITIAL_ITERATION_BUDGET,
            "fit_count": len(logistic_fit_diagnostics),
            "all_fits_converged": all(bool(row["converged"]) for row in logistic_fit_diagnostics),
            "maximum_iterations_used": max(
                (int(row["iterations"]) for row in logistic_fit_diagnostics), default=0
            ),
            "maximum_gradient_inf_norm": max(
                (float(row["gradient_inf_norm"]) for row in logistic_fit_diagnostics), default=0.0
            ),
            "fits": logistic_fit_diagnostics,
        },
    }
    receipt_name = "benchmark_numpy_v1_receipt.json" if additional_edge_trim_samples == 0 else f"benchmark_numpy_v1_edge_trim_{additional_edge_trim_samples}_receipt.json"
    receipt_path = Path(signals_root).parent / "proxy_temporal_v1" / receipt_name
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if fit_count != len(FILTER_BRANCHES) * len(MODEL_CANDIDATES) * len(outer_folds):
        raise ValueError("outer model-fit count differs from branch/model/fold coverage")
    return build_run_summary(
        summary_path=summary_path,
        parameter_path=parameter_path,
        selected_pipeline_path=selected_pipeline_path,
        selection_frequency_path=selection_frequency_path,
        augmentation_path=augmentation_path,
        noise_mix_path=noise_mix_path,
        generalization_gap_path=generalization_gap_path,
        additional_edge_trim_samples=additional_edge_trim_samples,
        receipt_path=receipt_path,
        participants=len(participants),
        recordings=138,
        segments=len(records),
        filters=len(FILTER_BRANCHES),
        models=len(MODEL_CANDIDATES),
        outer_folds=len(outer_folds),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--signals-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--additional-edge-trim-samples", type=int, default=0)
    args = parser.parse_args(argv)
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    output_dir = args.output_dir
    if args.additional_edge_trim_samples:
        output_dir = output_dir / f"edge_trim_{args.additional_edge_trim_samples}_samples"
    result = run_benchmark(
        args.labels, args.partitions, args.signals_root, output_dir,
        args.bootstrap_replicates, args.additional_edge_trim_samples,
    )
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
