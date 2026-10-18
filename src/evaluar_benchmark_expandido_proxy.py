"""Expanded, participant-grouped benchmark for the local temporal-position proxy.

The target remains the frozen three-way within-file time proxy, not fatigue.
Signals, labels, and partition manifests are read-only inputs. Participant and
fold-level results are written only to the explicitly supplied private output directory.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import json
import os
import re
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import analizar_trayectorias_emg as trajectories
import entrenar_proxy_temporal as base
from modelos_proxy_adicionales import predict_scores as predict_additional


SEED = 20261015
BOOTSTRAP_REPLICATES = 2000
WINDOW_SAMPLES = 1024
SPECTRAL_HIGH_HZ = 400.0
MODEL_SCORE_TYPES = {
    "lda": "lda_discriminant_score",
    "linear_svm": "linear_svm_margin",
    "random_forest": "probability",
    "gradient_boost_stumps": "gradient_boosting_logit_score",
    "logistic_rms_mdf": "multinomial_logistic_probability",
    "lda_rms_mdf": "lda_discriminant_score",
}
MODEL_COMPLEXITY = {
    "majority": 0, "nearest_centroid": 1, "lda": 2, "lda_rms_mdf": 2,
    "knn": 2, "ridge": 3, "logistic": 3, "logistic_rms_mdf": 3,
    "linear_svm": 4, "random_forest": 5, "gradient_boost_stumps": 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty result: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share one deterministic field order")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def assert_public_rows(rows: list[dict[str, Any]]) -> None:
    # Fold identifiers are private; aggregate counts such as total_outer_folds are not.
    # Match split-field names and explicit field=value text without blocking descriptive prose.
    forbidden_identity_fields = ("participant_id", "held_out_pseudonym", "recording_id", "filename", "source_id", "participant_name", "set_index")
    split_field = re.compile(r"(?:^|_)(?:outer_fold|inner_fold)(?:_|$)")
    private_assignment = re.compile(
        r"(?<![a-z0-9])(?:participant_id|held_out_pseudonym|recording_id|filename|source_id|"
        r"participant_name|outer_fold|inner_fold|set_index)\s*[:=]"
    )
    for row in rows:
        for key, value in row.items():
            key_text = str(key).casefold()
            value_text = str(value).casefold()
            if (any(token in key_text for token in forbidden_identity_fields)
                    or split_field.search(key_text)
                    or private_assignment.search(value_text)):
                raise ValueError("public aggregate contains an identifying field or value")


def median_frequency_descriptor(
    times_s: np.ndarray,
    signal: np.ndarray,
    expected_fs_hz: float,
    *,
    window_samples: int = WINDOW_SAMPLES,
) -> dict[str, float | int]:
    """Median of per-window MDF values without crossing gaps or invalid edges."""
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or times.size < window_samples:
        raise ValueError("time and signal vectors must match and contain a complete window")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("time and signal inputs must be finite")
    if not np.isfinite(expected_fs_hz) or expected_fs_hz <= 0:
        raise ValueError("timestamp-derived sampling rate must be positive and finite")
    windows = trajectories.iter_contiguous_windows(
        times, values, np.ones(values.size, dtype=bool),
        window_samples=window_samples, gap_screen_s=trajectories.GAP_SCREEN_S,
    )
    mdf = []
    for window in windows:
        features = trajectories.spectral_features(
            window["signal"], float(window["fs_hz"]),
            low_hz=20.0, high_hz=min(SPECTRAL_HIGH_HZ, float(window["fs_hz"]) / 2.0 - 1.0),
        )
        mdf.append(float(features["median_frequency_hz"]))
    if not mdf:
        raise ValueError("segment contains no complete contiguous spectral window")
    return {
        "median_frequency_hz": float(np.median(np.asarray(mdf, dtype=np.float64))),
        "window_count": len(mdf),
        "timestamp_rate_hz": float(expected_fs_hz),
    }


def model_registry() -> dict[str, dict[str, Any]]:
    """Fixed, small candidate grids declared before the outer LOSO scoring."""
    registry: dict[str, dict[str, Any]] = {}
    for name, candidates in base.MODEL_CANDIDATES.items():
        registry[name] = {
            "candidates": candidates, "representation": "15 distribution and waveform descriptors",
            "family": "existing_numpy_baseline", "score_type": base.MODEL_SCORE_TYPES[name],
        }
    registry.update({
        "lda": {"candidates": ({"shrinkage": 0.1}, {"shrinkage": 0.5}, {"shrinkage": 0.9}),
                "representation": "15 distribution and waveform descriptors", "family": "additional_numpy", "score_type": MODEL_SCORE_TYPES["lda"]},
        "linear_svm": {"candidates": ({"C": 0.1}, {"C": 1.0}, {"C": 10.0}),
                       "representation": "15 distribution and waveform descriptors", "family": "additional_numpy", "score_type": MODEL_SCORE_TYPES["linear_svm"]},
        "random_forest": {"candidates": (
                {"n_estimators": 8, "max_depth": 2, "min_samples_leaf": 2},
                {"n_estimators": 12, "max_depth": 3, "min_samples_leaf": 2}),
                "representation": "15 distribution and waveform descriptors", "family": "additional_numpy", "score_type": MODEL_SCORE_TYPES["random_forest"]},
        "gradient_boost_stumps": {"candidates": (
                {"learning_rate": 0.1, "n_estimators": 6},
                {"learning_rate": 0.1, "n_estimators": 12}),
                "representation": "15 distribution and waveform descriptors", "family": "additional_numpy", "score_type": MODEL_SCORE_TYPES["gradient_boost_stumps"]},
        "logistic_rms_mdf": {"candidates": base.MODEL_CANDIDATES["logistic"],
                "representation": "stored-channel RMS plus median frequency (MDF)", "family": "RMS_MDF_baseline", "score_type": MODEL_SCORE_TYPES["logistic_rms_mdf"]},
        "lda_rms_mdf": {"candidates": ({"shrinkage": 0.1}, {"shrinkage": 0.5}, {"shrinkage": 0.9}),
                "representation": "stored-channel RMS plus median frequency (MDF)", "family": "RMS_MDF_baseline", "score_type": MODEL_SCORE_TYPES["lda_rms_mdf"]},
    })
    return registry


def _parameter_text(parameter: Any) -> str:
    return "fixed" if parameter is None else json.dumps(parameter, sort_keys=True, separators=(",", ":"))


def _predict(
    model: str, parameter: Any, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
    *, seed: int, score_type: str,
) -> np.ndarray:
    scaled_train, scaled_test = base.standardize_train_test(train_x, test_x)
    if model == "logistic_rms_mdf":
        return base.predict_model("logistic", parameter, scaled_train, train_y, scaled_test, return_scores=True)
    if model == "lda_rms_mdf":
        return predict_additional("lda", parameter, scaled_train, train_y, scaled_test, seed=seed)
    if model in base.MODEL_CANDIDATES:
        return base.predict_model(model, parameter, scaled_train, train_y, scaled_test, return_scores=True)
    return predict_additional(model, parameter, scaled_train, train_y, scaled_test, seed=seed)


def _features_by_branch(
    rows: list[dict[str, str]], segments: list[dict[str, np.ndarray | float]]
) -> tuple[dict[tuple[str, str], np.ndarray], int]:
    if len(rows) != len(segments):
        raise ValueError("feature records and retained signal segments are not aligned")
    features: dict[tuple[str, str], np.ndarray] = {}
    windows_total = 0
    for branch in base.FILTER_BRANCHES:
        summary = np.asarray([
            [float(row[f"{branch}__{feature}"]) for feature in base.PREDICTOR_COLUMNS]
            for row in rows
        ], dtype=np.float64)
        rms_mdf = np.empty((len(rows), 2), dtype=np.float64)
        for index, (row, segment) in enumerate(zip(rows, segments)):
            descriptor = median_frequency_descriptor(
                np.asarray(segment["times_s"], dtype=np.float64),
                np.asarray(segment[branch], dtype=np.float64), float(segment["fs_hz"]),
            )
            rms_mdf[index] = (float(row[f"{branch}__adc_rms"]), float(descriptor["median_frequency_hz"]))
            windows_total += int(descriptor["window_count"])
        if not np.isfinite(summary).all() or not np.isfinite(rms_mdf).all():
            raise ValueError("nonfinite values found in a fold feature table")
        features[(branch, "summary")] = summary
        features[(branch, "rms_mdf")] = rms_mdf
    return features, windows_total


def _candidate_validation_score(
    model: str, parameter: Any, x: np.ndarray, y: np.ndarray,
    rows: list[dict[str, str]], splits: list[tuple[set[str], set[str]]], *, seed: int,
    score_type: str,
) -> float:
    participant_scores: list[float] = []
    for split_index, (train_ids, validation_ids) in enumerate(splits):
        train_idx = base._indices_for_groups(rows, train_ids)
        valid_idx = base._indices_for_groups(rows, validation_ids)
        scores = _predict(model, parameter, x[train_idx], y[train_idx], x[valid_idx], seed=seed + split_index, score_type=score_type)
        prediction = np.argmax(scores, axis=1).astype(np.int64)
        participant_scores.extend(base._participant_macro_scores(
            y[valid_idx], prediction, [rows[int(i)] for i in valid_idx], validation_ids,
        ))
    if not participant_scores or not np.isfinite(participant_scores).all():
        raise ValueError("inner grouped validation produced invalid participant scores")
    return float(np.mean(participant_scores))


def _select_parameter(
    model: str, candidates: tuple[Any, ...], x: np.ndarray, y: np.ndarray,
    rows: list[dict[str, str]], splits: list[tuple[set[str], set[str]]], *, seed: int,
    score_type: str,
) -> tuple[Any, float]:
    tested = [(_candidate_validation_score(model, candidate, x, y, rows, splits, seed=seed, score_type=score_type), candidate) for candidate in candidates]
    best = max(score for score, _ in tested)
    tied = [candidate for score, candidate in tested if abs(score - best) <= 1e-12]
    if model in {"ridge", "logistic", "logistic_rms_mdf", "knn"}:
        chosen = max(tied, key=lambda item: float(item))
    elif model in {"lda", "lda_rms_mdf"}:
        chosen = max(tied, key=lambda item: float(item["shrinkage"]))
    elif model == "linear_svm":
        chosen = max(tied, key=lambda item: float(item["C"]))
    elif model == "random_forest":
        chosen = min(tied, key=lambda item: (int(item["max_depth"]), int(item["n_estimators"])))
    elif model == "gradient_boost_stumps":
        chosen = min(tied, key=lambda item: int(item["n_estimators"]))
    else:
        chosen = tied[0]
    return chosen, float(best)


def _participant_metrics(
    truth: np.ndarray, scores: np.ndarray, score_type: str
) -> dict[str, Any]:
    predicted = np.argmax(scores, axis=1).astype(np.int64)
    hard = base.classification_metrics(truth, predicted)
    ranking = base.classification_score_metrics(truth, scores, score_type=score_type)
    return {**hard, **ranking}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return float(statistics.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _metric_summary(
    *, model: str, branch: str, registry_item: dict[str, Any], folds: list[dict[str, Any]],
    participant_count: int, bootstrap_seed: int,
) -> dict[str, Any]:
    metric_fields = (
        "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy",
        "macro_roc_auc_ovr", "macro_average_precision_ovr", "matthews_corrcoef", "cohen_kappa",
        "multiclass_brier", "log_loss", "f1_temprana", "f1_transicion", "f1_tardia",
        "recall_temprana", "recall_transicion", "recall_tardia",
        "roc_auc_temprana", "roc_auc_transicion", "roc_auc_tardia",
        "average_precision_temprana", "average_precision_transicion", "average_precision_tardia",
    )
    result: dict[str, Any] = {
        "model": model, "representation": registry_item["representation"],
        "filter_branch": branch, "nested_selection": "outer LOSO test of inner-selected parameters",
        "target": "three elapsed-recording-time thirds; not fatigue ground truth",
        "participants_n": participant_count, "score_type": registry_item["score_type"],
    }
    for field in metric_fields:
        values = [float(fold[field]) for fold in folds if fold.get(field) not in ("", None)]
        if model == "nested_selected_pipeline" and field in {"multiclass_brier", "log_loss"} and len(values) != participant_count:
            values = []
        if not values:
            result[f"{field}_mean_subject"] = ""
            result[f"{field}_sd_subject"] = ""
            continue
        mean, sd = _mean_sd(values)
        result[f"{field}_mean_subject"] = f"{mean:.6f}"
        result[f"{field}_sd_subject"] = f"{sd:.6f}"
        if field in {"macro_f1", "balanced_accuracy", "macro_roc_auc_ovr", "macro_average_precision_ovr"}:
            low, high = base._bootstrap_interval(values, seed=bootstrap_seed, replicates=BOOTSTRAP_REPLICATES)
            result[f"{field}_ci95_subject_low"] = f"{low:.6f}"
            result[f"{field}_ci95_subject_high"] = f"{high:.6f}"
    probability_folds = sum(fold.get("multiclass_brier") not in ("", None) for fold in folds)
    result["probability_metric_subjects_n"] = probability_folds
    if probability_folds == participant_count:
        result["probability_metric_note"] = "native probability-like outputs; calibration not assessed"
    elif probability_folds == 0:
        result["probability_metric_note"] = "not applicable; this model emits decision scores"
    else:
        result["probability_metric_note"] = "not summarized; the selected pipeline mixes probabilities and decision scores"
    for i in range(3):
        for j in range(3):
            result[f"pooled_confusion_{i}{j}"] = str(sum(int(fold[f"confusion_{i}{j}"]) for fold in folds))
    return result


def aggregate_roc_curves(
    prediction_store: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Average one-vs-rest ROC curves across held-out participants on a fixed grid."""
    grid = np.linspace(0.0, 1.0, 101)
    rows: list[dict[str, Any]] = []
    class_names = {index: label for label, index in base.LABEL_TO_INT.items()}
    for (branch, model), fold_predictions in prediction_store.items():
        if not fold_predictions:
            continue
        for class_index in range(3):
            interpolated = []
            for fold in fold_predictions:
                truth = np.asarray(fold["truth"], dtype=np.int64)
                scores = np.asarray(fold["scores"], dtype=np.float64)
                curve = base.binary_ranking_metrics(truth == class_index, scores[:, class_index])
                interpolated.append(np.interp(grid, curve["fpr"], curve["tpr"]))
            values = np.vstack(interpolated)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(grid.size)
            for point, fpr_value in enumerate(grid):
                rows.append({
                    "filter_branch": branch, "model": model,
                    "one_vs_rest_class": class_names[class_index],
                    "false_positive_rate": f"{fpr_value:.2f}",
                    "mean_true_positive_rate_subject": f"{mean[point]:.6f}",
                    "sd_true_positive_rate_subject": f"{sd[point]:.6f}",
                    "held_out_participants": values.shape[0],
                    "target": "elapsed-file-time proxy; not fatigue ground truth",
                })
    assert_public_rows(rows)
    return rows


def _public_model_name(name: str) -> str:
    return name.replace("_", " ")


def run_benchmark(
    labels_path: Path, partitions_path: Path, signals_root: Path,
    private_root: Path, project_root: Path, output_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    private_root = Path(private_root)
    private_root.mkdir(parents=True, exist_ok=True)
    labels = base._read_csv(labels_path)
    partitions = base._read_csv(partitions_path)
    participants = {row["participant_id"] for row in labels}
    if len(participants) != 15:
        raise ValueError("the authorized local benchmark scope is exactly the 15-participant proxy dataset")
    outer_folds, inner_folds = base.validate_nested_partitions(partitions, participants)
    print("validated 15-person nested LOSO manifest; extracting frozen features", flush=True)
    records, sample_counts, signal_digest, signal_segments = base.extract_filtered_features(labels_path, signals_root)
    if {row["participant_id"] for row in records} != participants:
        raise ValueError("feature extraction omitted or introduced a participant")
    features, spectral_windows = _features_by_branch(records, signal_segments)
    print(f"computed 15-descriptor and RMS+MDF features; contiguous MDF windows={spectral_windows}", flush=True)
    y = np.asarray([base.LABEL_TO_INT[row["label"]] for row in records], dtype=np.int64)
    registry = model_registry()
    configurations = [(branch, model) for branch in base.FILTER_BRANCHES for model in registry]
    if len(configurations) != 55:
        raise ValueError("expected 55 predeclared model-by-filter configurations")
    indices_by_group = {group: base._indices_for_groups(records, {group}) for group in participants}
    fold_results: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in configurations}
    fold_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in configurations}
    private_parameters: list[dict[str, Any]] = []
    private_selected: list[dict[str, Any]] = []
    private_fold_rows: list[dict[str, Any]] = []
    private_score_rows: list[dict[str, Any]] = []
    model_order = {name: idx for idx, name in enumerate(registry)}

    for outer_index, roles in enumerate(outer_folds):
        fold_id = sorted({str(row["outer_fold"]) for row in partitions if row.get("level") == "outer"}, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))[outer_index]
        held_out = next(group for group, role in roles.items() if role == "test")
        test_idx = indices_by_group[held_out]
        train_groups = {group for group, role in roles.items() if role == "train"}
        if set(records[int(i)]["participant_id"] for i in test_idx) != {held_out}:
            raise ValueError("outer held-out participant rows are not isolated")
        outer_train_idx = base._indices_for_groups(records, train_groups)
        inner_splits = inner_folds[fold_id]
        inner_scores: dict[tuple[str, str], float] = {}
        fold_cache: dict[tuple[str, str], dict[str, Any]] = {}

        for branch in base.FILTER_BRANCHES:
            for model, item in registry.items():
                representation_key = "rms_mdf" if item["family"] == "RMS_MDF_baseline" else "summary"
                x = features[(branch, representation_key)]
                chosen_parameter, inner_score = _select_parameter(
                    model, tuple(item["candidates"]), x, y, records, inner_splits,
                    seed=SEED + 101 * outer_index + model_order[model], score_type=item["score_type"],
                )
                key = (branch, model)
                inner_scores[key] = inner_score
                parameter_text = _parameter_text(chosen_parameter)
                private_parameters.append({
                    "outer_fold": fold_id, "held_out_pseudonym": held_out,
                    "filter_branch": branch, "model": model,
                    "selected_parameter": parameter_text,
                    "inner_mean_participant_macro_f1": f"{inner_score:.8f}",
                    "inner_splits": len(inner_splits),
                    "hyperparameter_selection_used_outer_test": "false",
                })
                test_scores = _predict(
                    model, chosen_parameter, x[outer_train_idx], y[outer_train_idx], x[test_idx],
                    seed=SEED + 1009 * outer_index + model_order[model], score_type=item["score_type"],
                )
                participant_metric = _participant_metrics(y[test_idx], test_scores, item["score_type"])
                participant_metric.update({
                    "outer_fold": fold_id, "held_out_pseudonym": held_out,
                    "filter_branch": branch, "model": model,
                    "representation": item["representation"],
                    "selected_parameter": parameter_text,
                    "test_segments": int(test_idx.size),
                })
                fold_results[key].append(participant_metric)
                fold_predictions[key].append({"truth": y[test_idx].copy(), "scores": test_scores.copy(), "participant_metric": participant_metric})
                fold_cache[key] = {"scores": test_scores, "metric": participant_metric, "parameter": chosen_parameter}
                for test_row_index, score_row in zip(test_idx, test_scores):
                    source_row = records[int(test_row_index)]
                    score_vector = np.asarray(score_row, dtype=np.float64)
                    private_score_rows.append({
                        "outer_fold": fold_id, "held_out_pseudonym": held_out,
                        "recording_id": source_row["recording_id"], "leg": source_row["leg"],
                        "set_index": source_row["set_index"], "true_proxy_label": source_row["label"],
                        "filter_branch": branch, "model": model,
                        "selected_parameter": parameter_text, "score_type": item["score_type"],
                        "predicted_proxy_label": base.INT_TO_LABEL[int(np.argmax(score_vector))],
                        "score_temprana": f"{score_vector[0]:.12g}",
                        "score_transicion": f"{score_vector[1]:.12g}",
                        "score_tardia": f"{score_vector[2]:.12g}",
                    })

        best_score = max(inner_scores.values())
        tied = [key for key, score in inner_scores.items() if abs(score - best_score) <= 1e-12]
        chosen_key = min(tied, key=lambda key: (
            MODEL_COMPLEXITY[key[1]], base.FILTER_BRANCHES.index(key[0]),
            0 if registry[key[1]]["family"] == "RMS_MDF_baseline" else 1,
            model_order[key[1]],
        ))
        selected = fold_cache[chosen_key]
        selected_metric = dict(selected["metric"])
        selected_metric.update({"model": "nested_selected_pipeline", "filter_branch": chosen_key[0]})
        fold_results.setdefault(("inner_cv_selected", "nested_selected_pipeline"), []).append(selected_metric)
        fold_predictions.setdefault(("inner_cv_selected", "nested_selected_pipeline"), []).append({
            "truth": y[test_idx].copy(), "scores": selected["scores"].copy(),
            "participant_metric": selected_metric,
        })
        for test_row_index, score_row in zip(test_idx, selected["scores"]):
            source_row = records[int(test_row_index)]
            score_vector = np.asarray(score_row, dtype=np.float64)
            private_score_rows.append({
                "outer_fold": fold_id, "held_out_pseudonym": held_out,
                "recording_id": source_row["recording_id"], "leg": source_row["leg"],
                "set_index": source_row["set_index"], "true_proxy_label": source_row["label"],
                "filter_branch": chosen_key[0], "model": "nested_selected_pipeline",
                "selected_parameter": _parameter_text(selected["parameter"]),
                "score_type": registry[chosen_key[1]]["score_type"],
                "predicted_proxy_label": base.INT_TO_LABEL[int(np.argmax(score_vector))],
                "score_temprana": f"{score_vector[0]:.12g}",
                "score_transicion": f"{score_vector[1]:.12g}",
                "score_tardia": f"{score_vector[2]:.12g}",
            })
        private_selected.append({
            "outer_fold": fold_id, "held_out_pseudonym": held_out,
            "filter_branch": chosen_key[0], "model": chosen_key[1],
            "selected_parameter": _parameter_text(selected["parameter"]),
            "inner_mean_participant_macro_f1": f"{best_score:.8f}",
            "selection_used_outer_test": "false",
        })
        print(f"completed_outer_fold={outer_index + 1}/{len(outer_folds)}; fixed_configurations={len(configurations)}", flush=True)

    aggregate_rows: list[dict[str, Any]] = []
    for branch, model in configurations:
        key = (branch, model)
        row = _metric_summary(
            model=model, branch=branch, registry_item=registry[model],
            folds=fold_results[key], participant_count=len(participants),
            bootstrap_seed=SEED + 100 * base.FILTER_BRANCHES.index(branch) + model_order[model],
        )
        aggregate_rows.append(row)
    nested_item = {"representation": "inner-selected branch, model, and hyperparameter from 15-fold outer LOSO", "score_type": "model-specific continuous class scores"}
    aggregate_rows.append(_metric_summary(
        model="nested_selected_pipeline", branch="inner_cv_selected", registry_item=nested_item,
        folds=fold_results[("inner_cv_selected", "nested_selected_pipeline")],
        participant_count=len(participants), bootstrap_seed=SEED + 909,
    ))
    assert_public_rows(aggregate_rows)

    parameter_counts: dict[tuple[str, str], int] = {}
    for row in private_selected:
        key = (str(row["filter_branch"]), str(row["model"]))
        parameter_counts[key] = parameter_counts.get(key, 0) + 1
    selected_frequency = []
    for (branch, model), count in sorted(parameter_counts.items(), key=lambda item: (base.FILTER_BRANCHES.index(item[0][0]), model_order[item[0][1]])):
        selected_frequency.append({"filter_branch": branch, "model": model, "selection_count": count, "total_selections": len(outer_folds), "fraction": f"{count / len(outer_folds):.6f}"})
    assert_public_rows(selected_frequency)

    # Sensitive row-level results are local only; aggregate tables above contain no IDs.
    for branch, model in configurations:
        for metric in fold_results[(branch, model)]:
            private_fold_rows.append(dict(metric))
    for metric in fold_results[("inner_cv_selected", "nested_selected_pipeline")]:
        private_fold_rows.append(dict(metric))
    private_results_path = private_root / "results_expanded_by_model_fold_private_20261015.csv"
    private_parameters_path = private_root / "parameters_expanded_by_fold_private_20261015.csv"
    private_selected_path = private_root / "selected_pipeline_by_fold_private_20261015.csv"
    private_scores_path = private_root / "segment_scores_expanded_private_20261015.csv"
    _write_csv(private_results_path, private_fold_rows)
    _write_csv(private_parameters_path, private_parameters)
    _write_csv(private_selected_path, private_selected)
    _write_csv(private_scores_path, private_score_rows)

    output = output_dir or (project_root / "datasets" / "derivados" / "benchmark_expandido_proxy")
    output.mkdir(parents=True, exist_ok=True)
    aggregate_path = output / "RESUMEN_BENCHMARK_EXPANDIDO_PROXY.csv"
    selection_path = output / "FRECUENCIA_MODELO_SELECCIONADO_PROXY.csv"
    roc_path = output / "CURVAS_ROC_BENCHMARK_EXPANDIDO_PROXY.csv"
    roc_rows = aggregate_roc_curves(fold_predictions)
    if len(aggregate_rows) != 56 or len(roc_rows) != 56 * 3 * 101:
        raise ValueError("expanded aggregate or ROC curve grid failed the declared row-count audit")
    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(selection_path, selected_frequency if selected_frequency else [{"filter_branch": "none", "model": "none", "selection_count": 0, "total_selections": len(outer_folds), "fraction": "0.000000"}])
    _write_csv(roc_path, roc_rows)

    config_sha = _sha256(Path(__file__))
    model_sha = _sha256(Path(__file__).with_name("modelos_proxy_adicionales.py"))
    elapsed = time.perf_counter() - started
    receipt = {
        "status": "completed" if len(aggregate_rows) == 56 and len(private_fold_rows) == 840 and len(private_score_rows) == 56 * len(records) else "failed_validation",
        "target": "within-recording elapsed-file-time thirds; not fatigue ground truth",
        "scope": "local 15-participant dataset only; no external dataset or external compute service",
        "participants": 15, "eligible_recordings": 138, "qc_only_records": 2,
        "segments": len(records), "classes": {label: int(np.sum(y == index)) for label, index in base.LABEL_TO_INT.items()},
        "outer_loso_folds": len(outer_folds), "inner_grouped_folds_per_outer": 4,
        "all_records_subject_grouped": True,
        "filters": list(base.FILTER_BRANCHES), "model_families": list(registry),
        "model_filter_configurations": len(configurations),
        "public_aggregate_rows": len(aggregate_rows), "public_roc_curve_rows": len(roc_rows), "private_fold_rows": len(private_fold_rows),
        "private_segment_score_rows": len(private_score_rows),
        "private_parameter_rows": len(private_parameters), "selected_pipeline_rows": len(private_selected),
        "timestamp_derived_nominal_rate_hz": 1000.0,
        "spectral_windows_mdf_total": spectral_windows,
        "spectral_band_hz": [20.0, SPECTRAL_HIGH_HZ], "spectral_window_samples": WINDOW_SAMPLES,
        "spectral_feature_caution": "stored-channel descriptors; signal mode and hardware chain unknown",
        "input_hashes": {"labels": _sha256(labels_path), "partitions": _sha256(partitions_path), "filter_arrays_digest": signal_digest},
        "code_hashes": {"benchmark": config_sha, "additional_models": model_sha},
        "python": platform.python_version(), "numpy": np.__version__,
        "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "outputs": {"public_summary": str(aggregate_path), "public_selection_frequency": str(selection_path), "public_roc_curves": str(roc_path),
                    "private_results": str(private_results_path), "private_results_sha256": _sha256(private_results_path),
                    "private_parameters": str(private_parameters_path), "private_selected_pipeline": str(private_selected_path),
                    "private_segment_scores": str(private_scores_path), "private_segment_scores_sha256": _sha256(private_scores_path)},
        "data_modified": False, "labels_modified": False, "partition_manifest_modified": False,
        "external_data_used": False, "external_compute_used": False,
        "elapsed_seconds": round(elapsed, 3),
    }
    receipt_path = private_root / "expanded_proxy_benchmark_receipt_20261015.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if receipt["status"] != "completed":
        raise RuntimeError("benchmark row-count audit failed; receipt marked failed_validation")
    return receipt


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--signals-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    print(json.dumps(run_benchmark(args.labels, args.partitions, args.signals_root, args.private_root, args.project_root, args.output_dir), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
