"""Small NumPy-only classifiers for grouped temporal-proxy comparisons.

These implementations operate on fold-scaled feature matrices. They estimate
only the existing time-position proxy; they are not physiological fatigue
models. Third-party ML runtimes are intentionally not loaded here.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import math
from typing import Any

import numpy as np


CLASS_COUNT = 3
SVM_MAX_ITERATIONS = 10_000
SVM_GRADIENT_TOLERANCE = 1e-6
BOOST_FEATURE_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FOREST_THRESHOLD_QUANTILES = BOOST_FEATURE_QUANTILES


def _validated_arrays(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.int64)
    x_test = np.asarray(test_x, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("training and test features must be compatible matrices")
    if x_train.shape[0] == 0 or y_train.shape != (x_train.shape[0],):
        raise ValueError("training labels must match nonempty training rows")
    if x_test.shape[0] == 0:
        raise ValueError("test features must be nonempty")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("model features must be finite")
    if np.any((y_train < 0) | (y_train >= CLASS_COUNT)) or set(np.unique(y_train)) != {0, 1, 2}:
        raise ValueError("all three classes must be represented by labels 0, 1, and 2")
    return x_train, y_train, x_test


def _lda_scores(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, shrinkage: float
) -> np.ndarray:
    if not math.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("LDA shrinkage must be finite and between zero and one")
    means = np.vstack([x_train[y_train == cls].mean(axis=0) for cls in range(CLASS_COUNT)])
    residuals = np.vstack([x_train[y_train == cls] - means[cls] for cls in range(CLASS_COUNT)])
    covariance = (residuals.T @ residuals) / max(1, x_train.shape[0] - CLASS_COUNT)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance.flat[:: covariance.shape[0] + 1] += 1e-8
    discriminants = np.linalg.solve(covariance, means.T).T
    intercept = -0.5 * np.sum(means * discriminants, axis=1)
    priors = np.bincount(y_train, minlength=CLASS_COUNT).astype(np.float64) / y_train.size
    intercept += np.log(np.clip(priors, 1e-15, 1.0))
    return x_test @ discriminants.T + intercept[None, :]


def _svm_binary_scores(
    x_train: np.ndarray, y_binary: np.ndarray, x_test: np.ndarray, c_value: float
) -> np.ndarray:
    """Squared-hinge linear SVM solved by deterministic accelerated gradient descent."""
    if not math.isfinite(c_value) or c_value <= 0.0:
        raise ValueError("linear SVM C must be finite and positive")
    n_rows, n_features = x_train.shape
    design = np.column_stack((x_train, np.ones(n_rows, dtype=np.float64)))
    prediction_design = np.column_stack((x_test, np.ones(x_test.shape[0], dtype=np.float64)))
    penalty = np.ones(n_features + 1, dtype=np.float64)
    penalty[-1] = 0.0
    lipschitz = 2.0 * c_value * float(np.linalg.norm(design, ord=2) ** 2) / n_rows + 1.0
    step = 1.0 / max(lipschitz, 1e-12)
    weights = np.zeros(n_features + 1, dtype=np.float64)
    extrapolated = weights.copy()
    momentum = 1.0
    converged = False
    for _ in range(SVM_MAX_ITERATIONS):
        margins = y_binary * (design @ extrapolated)
        active = np.maximum(0.0, 1.0 - margins)
        gradient = penalty * extrapolated - (2.0 * c_value / n_rows) * (
            design.T @ (y_binary * active)
        )
        updated = extrapolated - step * gradient
        updated_margins = y_binary * (design @ updated)
        updated_active = np.maximum(0.0, 1.0 - updated_margins)
        updated_gradient = penalty * updated - (2.0 * c_value / n_rows) * (
            design.T @ (y_binary * updated_active)
        )
        if not np.isfinite(updated).all() or not np.isfinite(updated_gradient).all():
            raise FloatingPointError("linear SVM produced nonfinite optimization state")
        if float(np.max(np.abs(updated_gradient))) <= SVM_GRADIENT_TOLERANCE:
            weights = updated
            converged = True
            break
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        next_extrapolated = updated + ((momentum - 1.0) / next_momentum) * (updated - weights)
        if float(np.sum((extrapolated - updated) * (updated - weights))) > 0.0:
            next_momentum = 1.0
            next_extrapolated = updated.copy()
        weights = updated
        extrapolated = next_extrapolated
        momentum = next_momentum
    if not converged:
        raise RuntimeError("linear SVM did not meet its fixed gradient stopping rule")
    return prediction_design @ weights


def _svm_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c_value: float) -> np.ndarray:
    columns = []
    for cls in range(CLASS_COUNT):
        binary = np.where(y_train == cls, 1.0, -1.0)
        columns.append(_svm_binary_scores(x_train, binary, x_test, c_value))
    return np.column_stack(columns)


def _gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    proportions = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64) / labels.size
    return float(1.0 - np.sum(proportions * proportions))


def _threshold_candidates(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    thresholds = np.quantile(values, quantiles)
    thresholds = np.unique(thresholds)
    return thresholds[(thresholds > values.min()) & (thresholds < values.max())]


def _build_classification_tree(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    labels = y[rows]
    counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64)
    probabilities = (counts + 1e-6) / (counts.sum() + CLASS_COUNT * 1e-6)
    node: dict[str, Any] = {"probabilities": probabilities}
    if depth >= max_depth or rows.size < 2 * min_samples_leaf or np.count_nonzero(counts) <= 1:
        return node
    features = np.sort(rng.choice(x.shape[1], size=min(max_features, x.shape[1]), replace=False))
    parent_impurity = _gini(labels)
    best_gain = 0.0
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for feature in features:
        values = x[rows, feature]
        for threshold in _threshold_candidates(values, FOREST_THRESHOLD_QUANTILES):
            left_mask = values <= threshold
            left = rows[left_mask]
            right = rows[~left_mask]
            if left.size < min_samples_leaf or right.size < min_samples_leaf:
                continue
            impurity = (left.size * _gini(y[left]) + right.size * _gini(y[right])) / rows.size
            gain = parent_impurity - impurity
            if gain > best_gain + 1e-12:
                best_gain = gain
                best = (int(feature), float(threshold), left, right)
    if best is None:
        return node
    feature, threshold, left, right = best
    node.update({
        "feature": feature,
        "threshold": threshold,
        "left": _build_classification_tree(x, y, left, depth + 1, max_depth, min_samples_leaf, max_features, rng),
        "right": _build_classification_tree(x, y, right, depth + 1, max_depth, min_samples_leaf, max_features, rng),
    })
    return node


def _tree_probability(tree: dict[str, Any], row: np.ndarray) -> np.ndarray:
    node = tree
    while "feature" in node:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    return np.asarray(node["probabilities"], dtype=np.float64)


def _random_forest_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    n_estimators = int(params["n_estimators"])
    max_depth = int(params["max_depth"])
    min_samples_leaf = int(params["min_samples_leaf"])
    if n_estimators < 1 or max_depth < 1 or min_samples_leaf < 1:
        raise ValueError("random-forest parameters must be positive integers")
    rng = np.random.default_rng(seed)
    max_features = max(1, int(math.sqrt(x_train.shape[1])))
    tree_seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_estimators)
    forests = []
    for tree_seed in tree_seeds:
        tree_rng = np.random.default_rng(int(tree_seed))
        sample = tree_rng.integers(0, x_train.shape[0], size=x_train.shape[0], dtype=np.int64)
        forests.append(_build_classification_tree(
            x_train, y_train, sample, 0, max_depth, min_samples_leaf, max_features, tree_rng
        ))
    result = np.zeros((x_test.shape[0], CLASS_COUNT), dtype=np.float64)
    for tree in forests:
        result += np.vstack([_tree_probability(tree, row) for row in x_test])
    return result / n_estimators


def _fit_regression_stump(
    x: np.ndarray, residual: np.ndarray
) -> tuple[int | None, float, float, float]:
    baseline = float(np.mean(residual))
    base_sse = float(np.sum((residual - baseline) ** 2))
    best_sse = base_sse
    best: tuple[int | None, float, float, float] = (None, 0.0, baseline, baseline)
    for feature in range(x.shape[1]):
        values = x[:, feature]
        for threshold in _threshold_candidates(values, BOOST_FEATURE_QUANTILES):
            left_mask = values <= threshold
            left_values = residual[left_mask]
            right_values = residual[~left_mask]
            if left_values.size == 0 or right_values.size == 0:
                continue
            left_mean = float(np.mean(left_values))
            right_mean = float(np.mean(right_values))
            sse = float(np.sum((left_values - left_mean) ** 2) + np.sum((right_values - right_mean) ** 2))
            if sse < best_sse - 1e-12:
                best_sse = sse
                best = (feature, float(threshold), left_mean, right_mean)
    return best


def _stump_predict(stump: tuple[int | None, float, float, float], x: np.ndarray) -> np.ndarray:
    feature, threshold, left_value, right_value = stump
    if feature is None:
        return np.full(x.shape[0], left_value, dtype=np.float64)
    return np.where(x[:, feature] <= threshold, left_value, right_value).astype(np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _gradient_boost_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    learning_rate = float(params["learning_rate"])
    n_estimators = int(params["n_estimators"])
    if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0 or n_estimators < 1:
        raise ValueError("gradient-boosting parameters are outside their valid range")
    one_hot = np.eye(CLASS_COUNT, dtype=np.float64)[y_train]
    priors = np.bincount(y_train, minlength=CLASS_COUNT).astype(np.float64) / y_train.size
    initial = np.log(np.clip(priors, 1e-12, 1.0))
    train_logits = np.repeat(initial[None, :], x_train.shape[0], axis=0)
    test_logits = np.repeat(initial[None, :], x_test.shape[0], axis=0)
    stumps: list[list[tuple[int | None, float, float, float]]] = [[] for _ in range(CLASS_COUNT)]
    for _ in range(n_estimators):
        probabilities = _softmax(train_logits)
        residuals = one_hot - probabilities
        for cls in range(CLASS_COUNT):
            stump = _fit_regression_stump(x_train, residuals[:, cls])
            stumps[cls].append(stump)
            train_logits[:, cls] += learning_rate * _stump_predict(stump, x_train)
            test_logits[:, cls] += learning_rate * _stump_predict(stump, x_test)
    if not np.isfinite(test_logits).all():
        raise FloatingPointError("gradient boosting produced nonfinite class scores")
    return test_logits


def predict_scores(
    model: str,
    parameter: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit one added family and return three continuous class scores.

    LDA, SVM and boosting return decision scores. The random forest returns
    normalized vote probabilities. Scaling, if needed, must be fitted outside
    this function on the training fold only.
    """
    x_train, y_train, x_test = _validated_arrays(train_x, train_y, test_x)
    if not isinstance(parameter, dict):
        raise ValueError("additional-model parameters must be a dictionary")
    if model == "lda":
        return _lda_scores(x_train, y_train, x_test, float(parameter["shrinkage"]))
    if model == "linear_svm":
        return _svm_scores(x_train, y_train, x_test, float(parameter["C"]))
    if model == "random_forest":
        return _random_forest_scores(x_train, y_train, x_test, parameter, int(seed))
    if model == "gradient_boost_stumps":
        return _gradient_boost_scores(x_train, y_train, x_test, parameter)
    raise ValueError(f"unknown additional classifier: {model}")
