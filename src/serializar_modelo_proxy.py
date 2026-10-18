"""Fit and serialize the selected temporal-position proxy model locally.

This final full-cohort refit follows the most frequent pipeline selected by the
already completed nested LOSO evaluation. It is a research artifact for the
time-position proxy, not a fatigue detector or a new performance estimate.
The serialized model and its receipt must remain in restricted local storage.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import entrenar_proxy_temporal as base


SEED = 20261015
PRIVATE_SUBDIR = Path("EMG_Biomecanica") / "datasets" / "proxy_temporal_v1"
EXPECTED_OUTER_FOLDS = 15
EXPECTED_SEGMENTS = 414


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def modal_nested_configuration(path: Path) -> dict[str, Any]:
    """Return the most frequent selected pipeline without returning fold IDs."""
    required = {"outer_fold", "filter_branch", "model", "selected_parameter"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("private nested-selection file is missing required columns")
        rows = list(reader)
    folds = [row["outer_fold"] for row in rows]
    if len(rows) != EXPECTED_OUTER_FOLDS or len(set(folds)) != EXPECTED_OUTER_FOLDS:
        raise ValueError("nested selection must contain exactly one row for each of 15 outer folds")
    counts = Counter((row["filter_branch"], row["model"], row["selected_parameter"]) for row in rows)
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            base.FILTER_BRANCHES.index(item[0][0]) if item[0][0] in base.FILTER_BRANCHES else len(base.FILTER_BRANCHES),
            base.MODEL_COMPLEXITY.get(item[0][1], 999),
            item[0][1],
            item[0][2],
        ),
    )
    (branch, model, parameter), count = ordered[0]
    return {
        "filter_branch": branch,
        "model": model,
        "selected_parameter": parameter,
        "selection_count": int(count),
        "total_selections": len(rows),
        "selection_fraction": float(count / len(rows)),
    }


def fit_logistic_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    branch: str = "hp20",
    alpha: float = 0.01,
    seed: int = SEED,
    participants_n: int | None = None,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Fit the existing multinomial FISTA logistic rule and retain its parameters."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] == 0 or y.shape != (x.shape[0],):
        raise ValueError("features and labels must be compatible nonempty arrays")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("logistic regularization must be positive and finite")
    if not set(np.unique(y)).issuperset({0, 1, 2}) or np.any((y < 0) | (y > 2)):
        raise ValueError("all three classes must be present in labels")
    if branch not in base.FILTER_BRANCHES:
        raise ValueError("unknown predeclared filter branch")
    if participants_n is not None and (not isinstance(participants_n, int) or participants_n < 1 or participants_n > x.shape[0]):
        raise ValueError("known participant count must be a positive integer no greater than segment count")
    names = tuple(feature_names or (f"feature_{index + 1}" for index in range(x.shape[1])))
    if len(names) != x.shape[1] or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature names must match the feature matrix")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    scaled = (x - mean) / scale
    augmented = np.column_stack((np.ones(x.shape[0]), scaled))
    targets = np.eye(3, dtype=np.float64)[y]
    weights = np.zeros((augmented.shape[1], 3), dtype=np.float64)
    penalty = np.full(augmented.shape[1], float(alpha), dtype=np.float64)
    penalty[0] = 1e-8
    spectral = float(np.linalg.norm(augmented, ord=2) ** 2 / x.shape[0])
    step_size = 1.0 / max(0.5 * spectral + float(alpha), 1e-12)
    converged = False
    gradient_inf_norm = float("inf")
    total_iterations = 0
    retry_count = 0
    current_step = step_size
    for attempt in range(base.LOGISTIC_MAX_RETRIES + 1):
        iteration_budget = base.LOGISTIC_INITIAL_ITERATION_BUDGET * (base.LOGISTIC_RETRY_MULTIPLIER ** attempt)
        extrapolated = weights.copy()
        momentum = 1.0
        last_finite = weights.copy()
        for iteration in range(1, iteration_budget + 1):
            probabilities = _softmax(augmented @ extrapolated)
            gradient = augmented.T @ (probabilities - targets) / x.shape[0]
            gradient += penalty[:, None] * extrapolated
            updated = extrapolated - current_step * gradient
            if not np.isfinite(updated).all():
                break
            updated_probabilities = _softmax(augmented @ updated)
            updated_gradient = augmented.T @ (updated_probabilities - targets) / x.shape[0]
            updated_gradient += penalty[:, None] * updated
            gradient_inf_norm = float(np.max(np.abs(updated_gradient)))
            if not math.isfinite(gradient_inf_norm):
                break
            previous = weights
            weights = updated
            last_finite = weights.copy()
            total_iterations += 1
            if gradient_inf_norm <= base.LOGISTIC_GRADIENT_TOLERANCE:
                converged = True
                break
            next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
            next_extrapolated = weights + ((momentum - 1.0) / next_momentum) * (weights - previous)
            if float(np.sum((extrapolated - weights) * (weights - previous))) > 0.0:
                next_momentum = 1.0
                next_extrapolated = weights.copy()
            extrapolated = next_extrapolated
            momentum = next_momentum
        if converged:
            break
        weights = last_finite
        if attempt < base.LOGISTIC_MAX_RETRIES:
            retry_count += 1
    if not converged:
        raise RuntimeError("full-cohort logistic refit did not converge within the configured retry budget")
    return {
        "schema": "emg_temporal_proxy_softmax_logistic_v1",
        "model": "multinomial_logistic_regression",
        "branch": branch,
        "representation": "15 distribution and waveform descriptors from stored ADC_Value",
        "target": "three elapsed-recording-time thirds; not fatigue ground truth",
        "alpha": float(alpha),
        "seed": int(seed),
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "feature_names": names,
        "classes": ("temprana", "transicion", "tardia"),
        "participants_n": int(participants_n) if participants_n is not None else None,
        "segments_n": int(x.shape[0]),
        "fit_diagnostics": {
            "solver": "fista_adaptive_restart",
            "converged": converged,
            "iterations": total_iterations,
            "gradient_inf_norm": gradient_inf_norm,
            "gradient_tolerance": base.LOGISTIC_GRADIENT_TOLERANCE,
            "retry_count": retry_count,
        },
    }


def predict_scores(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != mean.size or scale.shape != mean.shape:
        raise ValueError("features do not match the serialized model schema")
    if weights.shape != (mean.size + 1, 3) or not np.isfinite(x).all():
        raise ValueError("model parameters or features are invalid")
    scaled = (x - mean) / scale
    logits = np.column_stack((np.ones(x.shape[0]), scaled)) @ weights
    return _softmax(logits)


def save_model(path: Path, model: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing model artifact: {destination.name}")
    with destination.open("xb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(model["schema"]),
            model=np.asarray(model["model"]),
            branch=np.asarray(model["branch"]),
            representation=np.asarray(model["representation"]),
            target=np.asarray(model["target"]),
            alpha=np.asarray(model["alpha"], dtype=np.float64),
            seed=np.asarray(model["seed"], dtype=np.int64),
            mean=np.asarray(model["mean"], dtype=np.float64),
            scale=np.asarray(model["scale"], dtype=np.float64),
            weights=np.asarray(model["weights"], dtype=np.float64),
            feature_names=np.asarray(model["feature_names"], dtype=np.str_),
            classes=np.asarray(model["classes"], dtype=np.str_),
            participants_n=np.asarray(-1 if model["participants_n"] is None else model["participants_n"], dtype=np.int64),
            segments_n=np.asarray(model["segments_n"], dtype=np.int64),
            iterations=np.asarray(model["fit_diagnostics"]["iterations"], dtype=np.int64),
            gradient_inf_norm=np.asarray(model["fit_diagnostics"]["gradient_inf_norm"], dtype=np.float64),
            retry_count=np.asarray(model["fit_diagnostics"]["retry_count"], dtype=np.int64),
        )


def load_model(path: Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "schema", "model", "branch", "representation", "target", "alpha", "seed",
            "mean", "scale", "weights", "feature_names", "classes", "participants_n",
            "segments_n", "iterations", "gradient_inf_norm", "retry_count",
        }
        if set(archive.files) != required:
            raise ValueError("serialized model fields do not match the safe model schema")
        return {
            "schema": str(archive["schema"].item()),
            "model": str(archive["model"].item()),
            "branch": str(archive["branch"].item()),
            "representation": str(archive["representation"].item()),
            "target": str(archive["target"].item()),
            "alpha": float(archive["alpha"].item()),
            "seed": int(archive["seed"].item()),
            "mean": archive["mean"].copy(),
            "scale": archive["scale"].copy(),
            "weights": archive["weights"].copy(),
            "feature_names": tuple(str(value) for value in archive["feature_names"]),
            "classes": tuple(str(value) for value in archive["classes"]),
            "participants_n": (lambda value: None if value < 0 else value)(int(archive["participants_n"].item())),
            "segments_n": int(archive["segments_n"].item()),
            "fit_diagnostics": {
                "solver": "fista_adaptive_restart",
                "converged": True,
                "iterations": int(archive["iterations"].item()),
                "gradient_inf_norm": float(archive["gradient_inf_norm"].item()),
                "gradient_tolerance": base.LOGISTIC_GRADIENT_TOLERANCE,
                "retry_count": int(archive["retry_count"].item()),
            },
        }


def run_serialization(
    labels_path: Path,
    signals_root: Path,
    nested_selection_path: Path,
    private_root: Path,
    *,
    model_path: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    labels_file = Path(labels_path).resolve()
    nested_file = Path(nested_selection_path).resolve()
    private = Path(private_root).resolve()
    choice = modal_nested_configuration(nested_file)
    if choice["model"] != "logistic" or choice["filter_branch"] != "hp20":
        raise ValueError("modal nested choice is not the declared HP20 logistic candidate")
    try:
        alpha = float(choice["selected_parameter"])
    except ValueError as exc:
        raise ValueError("modal nested logistic regularization is not numeric") from exc

    rows = base._read_csv(labels_file)
    if len(rows) != EXPECTED_SEGMENTS:
        raise ValueError("full-cohort refit requires the frozen 414-segment proxy table")
    participants = {row["participant_id"] for row in rows}
    if len(participants) != 15 or any(row["recording_id"] not in {"R1", "R2", "R3", "R4", "R5"} for row in rows):
        raise ValueError("model refit scope must be exactly the local 15 participants and R1-R5")
    records, _, signal_digest, _ = base.extract_filtered_features(labels_file, Path(signals_root).resolve())
    if len(records) != EXPECTED_SEGMENTS or {row["participant_id"] for row in records} != participants:
        raise ValueError("feature extraction changed the frozen 15-participant/414-segment scope")
    x = np.asarray(
        [[float(row[f"hp20__{feature}"]) for feature in base.PREDICTOR_COLUMNS] for row in records],
        dtype=np.float64,
    )
    y = np.asarray([base.LABEL_TO_INT[row["label"]] for row in records], dtype=np.int64)
    if Counter(y.tolist()) != Counter({0: 138, 1: 138, 2: 138}):
        raise ValueError("the frozen proxy class counts must remain 138 per class")
    model = fit_logistic_model(
        x, y, branch="hp20", alpha=alpha, seed=SEED, participants_n=len(participants),
        feature_names=tuple(base.PREDICTOR_COLUMNS),
    )
    standardized = (x - model["mean"]) / model["scale"]
    reference_scores = base._predict_logistic(
        standardized, y, standardized, alpha, return_scores=True,
    )
    round_trip_scores = predict_scores(model, x)
    max_score_difference = float(np.max(np.abs(reference_scores - round_trip_scores)))
    if not math.isfinite(max_score_difference) or max_score_difference > 1e-10:
        raise RuntimeError("serialized coefficients do not reproduce the audited logistic scoring rule")

    destination = Path(model_path).resolve() if model_path else private / "models_private" / "proxy_hp20_logistic_full_fit_v2_20261015.npz"
    if not destination.is_relative_to(private):
        raise ValueError("serialized models must remain under the restricted local dataset directory")
    save_model(destination, model)
    loaded = load_model(destination)
    loaded_scores = predict_scores(loaded, x)
    load_difference = float(np.max(np.abs(reference_scores - loaded_scores)))
    if load_difference > 1e-10:
        raise RuntimeError("reloaded model artifact failed score round-trip verification")
    receipt_path = destination.with_suffix(".receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite existing model receipt: {receipt_path.name}")
    receipt = {
        "status": "verified_full_fit_proxy_model",
        "artifact_sha256": _sha256(destination),
        "artifact_path": str(destination),
        "labels_sha256": _sha256(labels_file),
        "nested_selection_receipt_sha256": _sha256(nested_file),
        "signal_trace_digest": signal_digest,
        "model": model["model"],
        "filter_branch": model["branch"],
        "regularization_alpha": model["alpha"],
        "modal_nested_selection": choice,
        "fit_scope": "final full-cohort refit after nested LOSO evaluation; not used to estimate performance",
        "participants_n": model["participants_n"],
        "segments_n": model["segments_n"],
        "class_counts": {"temprana": 138, "transicion": 138, "tardia": 138},
        "feature_names": list(model["feature_names"]),
        "seed": model["seed"],
        "fit_diagnostics": model["fit_diagnostics"],
        "max_score_difference_vs_existing_trainer": max_score_difference,
        "max_score_difference_after_load": load_difference,
        "performance_source": "nested LOSO aggregate CSV only; no in-sample performance reported",
        "target_is_fatigue": False,
        "model_is_fatigue_detector": False,
        "signal_or_participant_rows_serialized": False,
        "external_data_or_compute_used": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return {
        "status": receipt["status"],
        "artifact_path": str(destination),
        "receipt_path": str(receipt_path),
        "artifact_sha256": receipt["artifact_sha256"],
        "model": model["model"],
        "branch": model["branch"],
        "alpha": model["alpha"],
        "modal_nested_selection_count": choice["selection_count"],
        "participants_n": model["participants_n"],
        "segments_n": model["segments_n"],
        "converged": model["fit_diagnostics"]["converged"],
        "iterations": model["fit_diagnostics"]["iterations"],
        "gradient_inf_norm": model["fit_diagnostics"]["gradient_inf_norm"],
        "max_score_difference_vs_existing_trainer": max_score_difference,
        "max_score_difference_after_load": load_difference,
        "performance_note": "full-data fit only; report CV metrics from nested LOSO",
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--signals-root", type=Path, required=True)
    parser.add_argument("--nested-selection", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args(argv)
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    result = run_serialization(args.labels, args.signals_root, args.nested_selection, args.private_root, model_path=args.model_output)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
