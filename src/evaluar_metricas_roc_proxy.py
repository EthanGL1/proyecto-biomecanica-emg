"""Full participant-held-out score, ROC, and synthetic stress evaluation.

The only target is the frozen within-recording temporal-position proxy. This
module does not infer fatigue, repetition count, technical failure, or clinical
state. Participant identifiers, labels by fold, predictions, and score vectors
remain in memory or in a restricted private metrics receipt.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

import entrenar_proxy_temporal as trainer


LABELS = tuple(trainer.LABEL_TO_INT)
SCENARIO_CLEAN = "clean"
N_BOOTSTRAP_DEFAULT = 2000
ROC_GRID = np.linspace(0.0, 1.0, 101)
AGGREGATE_METRICS = (
    "accuracy", "macro_precision", "macro_recall", "weighted_precision",
    "weighted_recall", "micro_precision", "micro_recall", "macro_f1",
    "weighted_f1", "micro_f1", "balanced_accuracy", "macro_specificity", "macro_npv",
    "matthews_corrcoef", "cohen_kappa", "macro_roc_auc_ovr",
    "micro_roc_auc_ovr", "macro_average_precision_ovr",
    "micro_average_precision_ovr", "roc_auc_temprana", "roc_auc_transicion",
    "roc_auc_tardia", "average_precision_temprana",
    "average_precision_transicion", "average_precision_tardia",
    "multiclass_brier", "log_loss", "delta_macro_f1_vs_clean",
    "delta_balanced_accuracy_vs_clean", "delta_macro_roc_auc_ovr_vs_clean",
    "delta_macro_average_precision_ovr_vs_clean",
) + tuple(
    f"{metric}_{label}"
    for label in LABELS
    for metric in ("precision", "recall", "f1", "specificity", "npv")
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty evaluation table")
    trainer._assert_public_schema(rows)  # protects exported aggregates from identifiers
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("all CSV rows must share one schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def require_new_output_paths(paths: Iterable[Path]) -> None:
    """Refuse to replace prior public/private artifacts on a rerun."""
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("evaluation outputs already exist; choose new versioned paths: " + ", ".join(existing))


def _numeric(value: object) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("participant bootstrap requires at least two finite participant scores")
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError("participant bootstrap requires at least 100 replicates")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(replicates, data.size))
    means = data[indices].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def aggregate_subject_metrics(
    rows: list[dict[str, object]],
    *,
    metrics: Iterable[str] = AGGREGATE_METRICS,
    bootstrap_replicates: int = N_BOOTSTRAP_DEFAULT,
    seed: int = 20261014,
) -> list[dict[str, object]]:
    """Summarize one equally weighted value per held-out participant per group."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        required = ("outer_fold", "filter_branch", "model", "scenario")
        if any(not str(row.get(key, "")) for key in required):
            raise ValueError("each fold metric requires an outer fold and evaluation group")
        grouped[tuple(str(row[key]) for key in required[1:])].append(row)
    results: list[dict[str, object]] = []
    metric_names = tuple(metrics)
    for (branch, model, scenario), group_rows in sorted(grouped.items()):
        fold_ids = [str(row["outer_fold"]) for row in group_rows]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("each group requires unique outer folds")
        if len(fold_ids) != 15:
            raise ValueError("each reported group requires all 15 unique outer folds")
        result: dict[str, object] = {
            "filter_branch": branch,
            "model": model,
            "scenario": scenario,
            "target": "within-recording temporal-position proxy; not fatigue ground truth",
            "outer_folds": 15,
            "outer_test_participants": 15,
            "outer_test_segments": sum(int(row.get("n_test_segments", 0)) for row in group_rows),
            "bootstrap_replicates_subject": bootstrap_replicates,
            "aggregation_unit": "equal-weight held-out participant; 95% percentile bootstrap",
        }
        result["score_output_types"] = "; ".join(sorted({
            str(row["score_type"]) for row in group_rows if row.get("score_type")
        })) or "unknown"
        result["probability_metrics_status"] = "; ".join(sorted({
            str(row["probability_metrics_status"]) for row in group_rows
            if row.get("probability_metrics_status")
        })) or "unknown"
        result["probability_metrics_note"] = (
            "Brier/log loss are proper-score summaries for probability-like outputs; no calibration validation"
            if any(_numeric(row.get("multiclass_brier")) is not None for row in group_rows)
            else "Not calculated because model scores are not probability outputs"
        )
        for metric_index, metric in enumerate(metric_names):
            values = np.asarray(
                [number for row in group_rows if (number := _numeric(row.get(metric))) is not None],
                dtype=np.float64,
            )
            if values.size == 0:
                result[f"{metric}_mean_subject"] = ""
                result[f"{metric}_sd_subject"] = ""
                result[f"{metric}_median_subject"] = ""
                result[f"{metric}_min_subject"] = ""
                result[f"{metric}_max_subject"] = ""
                result[f"{metric}_ci95_low_subject"] = ""
                result[f"{metric}_ci95_high_subject"] = ""
                result[f"{metric}_valid_participants"] = 0
                continue
            if values.size != 15:
                raise ValueError(f"{metric} is present in only {values.size} of 15 participant folds")
            low, high = _bootstrap_ci(
                values, seed=seed + metric_index + sum(map(ord, branch + model + scenario)),
                replicates=bootstrap_replicates,
            )
            result[f"{metric}_mean_subject"] = float(values.mean())
            result[f"{metric}_sd_subject"] = float(values.std(ddof=1))
            result[f"{metric}_median_subject"] = float(np.median(values))
            result[f"{metric}_min_subject"] = float(values.min())
            result[f"{metric}_max_subject"] = float(values.max())
            result[f"{metric}_ci95_low_subject"] = low
            result[f"{metric}_ci95_high_subject"] = high
            result[f"{metric}_valid_participants"] = int(values.size)
        for i in range(3):
            for j in range(3):
                result[f"confusion_{i}{j}_total"] = sum(
                    int(row.get(f"confusion_{i}{j}", 0)) for row in group_rows
                )
        results.append(result)
    return results


def interpolate_roc_curve(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a monotonic ROC envelope, retaining the top of vertical steps."""
    false_positive = np.asarray(fpr, dtype=np.float64)
    true_positive = np.asarray(tpr, dtype=np.float64)
    targets = np.asarray(grid, dtype=np.float64)
    if false_positive.ndim != 1 or true_positive.shape != false_positive.shape or false_positive.size < 2:
        raise ValueError("FPR and TPR must be compatible ROC vectors")
    if targets.ndim != 1 or not np.isfinite(targets).all() or np.any(np.diff(targets) < 0.0):
        raise ValueError("ROC grid must be a finite ascending vector")
    if (not np.isfinite(false_positive).all() or not np.isfinite(true_positive).all()
            or np.any(np.diff(false_positive) < 0.0) or np.any(np.diff(true_positive) < 0.0)):
        raise ValueError("ROC coordinates must be finite and monotonic")
    unique_fpr, starts, counts = np.unique(
        false_positive, return_index=True, return_counts=True
    )
    first_tpr = true_positive[starts]
    last_tpr = true_positive[starts + counts - 1]
    output = np.empty_like(targets)
    for index, target in enumerate(targets):
        exact = np.flatnonzero(np.isclose(unique_fpr, target, rtol=0.0, atol=1e-14))
        if exact.size:
            output[index] = last_tpr[int(exact[0])]
            continue
        right = int(np.searchsorted(unique_fpr, target, side="right"))
        if right == 0:
            output[index] = first_tpr[0]
        elif right >= unique_fpr.size:
            output[index] = last_tpr[-1]
        else:
            left = right - 1
            fraction = (target - unique_fpr[left]) / (unique_fpr[right] - unique_fpr[left])
            output[index] = last_tpr[left] + fraction * (first_tpr[right] - last_tpr[left])
    return output


def perturbed_feature_vector(
    segment: dict[str, np.ndarray | float],
    branch: str,
    scenario: str,
    *,
    seed: int,
) -> np.ndarray:
    """Extract a feature vector from a local, post-filter synthetic test copy."""
    available = {name: (kind, params) for name, kind, params in
                 trainer.synthetic_noise_scenarios(float(segment["fs_hz"]))}
    if scenario not in available:
        raise ValueError(f"unknown predeclared synthetic stress case: {scenario}")
    if branch not in segment:
        raise ValueError("synthetic stress branch is missing from the held-out signal copy")
    original = np.asarray(segment[branch], dtype=np.float64)
    pairs = np.asarray(segment.get("contiguous_pair_indices", []), dtype=np.int64)
    if (original.ndim != 1 or original.size < 2 or not np.isfinite(original).all()
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0):
        raise ValueError("held-out signal segment or contiguous-pair manifest is invalid")
    kind, params = available[scenario]
    perturbed = trainer.augment_signal(original.copy(), kind, params, seed=int(seed))
    feature = trainer._features(perturbed.tolist(), [tuple(map(int, pair)) for pair in pairs])
    vector = np.asarray([feature[column] for column in trainer.PREDICTOR_COLUMNS], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("synthetic copy produced nonfinite features")
    return vector


def _score_row(
    *,
    fold_id: str,
    branch: str,
    model: str,
    parameter: float | int | None,
    inner_score: float,
    scenario: str,
    truth: np.ndarray,
    scores: np.ndarray,
    n_segments: int,
) -> dict[str, object]:
    prediction = np.argmax(scores, axis=1).astype(np.int64)
    hard = trainer.classification_metrics(truth, prediction)
    ranking = trainer.classification_score_metrics(
        truth, scores, score_type=trainer.MODEL_SCORE_TYPES[model]
    )
    row: dict[str, object] = {
        "outer_fold": fold_id,
        "filter_branch": branch,
        "model": model,
        "selected_parameter": "fixed" if parameter is None else str(parameter),
        "inner_validation_macro_f1": inner_score,
        "scenario": scenario,
        "target": "within-recording temporal-position proxy; not fatigue ground truth",
        "n_test_segments": n_segments,
        "delta_macro_f1_vs_clean": "",
        "delta_balanced_accuracy_vs_clean": "",
        "delta_macro_roc_auc_ovr_vs_clean": "",
        "delta_macro_average_precision_ovr_vs_clean": "",
    }
    row.update(hard)
    row.update(ranking)
    return row


def load_audited_inner_selections(
    historical_metrics_path: Path,
    benchmark_receipt_path: Path,
    fold_ids: list[str],
    *,
    labels_path: Path,
    partitions_path: Path,
    combined_signal_digest: str,
) -> tuple[dict[tuple[str, str, str], tuple[float | int | None, float]], dict[str, object]]:
    """Reuse prior nested selections only after matching their exact inputs and receipt."""
    receipt = json.loads(Path(benchmark_receipt_path).read_text(encoding="utf-8"))
    expected_input_hashes = {
        "labels_sha256": _sha256(labels_path),
        "partitions_sha256": _sha256(partitions_path),
        "combined_private_filter_trace_digest": combined_signal_digest,
    }
    for field, expected in expected_input_hashes.items():
        if receipt.get(field) != expected:
            raise ValueError(f"audited benchmark selection cannot be reused: {field} changed")
    if (receipt.get("participants") != 15 or receipt.get("segments") != 414
            or tuple(receipt.get("filters", ())) != trainer.FILTER_BRANCHES
            or tuple(receipt.get("models", ())) != tuple(trainer.MODEL_CANDIDATES)):
        raise ValueError("audited benchmark receipt does not match the frozen participant/model design")
    metric_receipt_path = Path(str(historical_metrics_path) + ".receipt.json")
    metric_receipt = json.loads(metric_receipt_path.read_text(encoding="utf-8"))
    if metric_receipt.get("results_sha256") != _sha256(historical_metrics_path):
        raise ValueError("historical fold-metric file does not match its integrity receipt")
    for receipt_field, expected in (
        ("input_labels_sha256", expected_input_hashes["labels_sha256"]),
        ("input_partitions_sha256", expected_input_hashes["partitions_sha256"]),
        ("filter_trace_digest", combined_signal_digest),
    ):
        if metric_receipt.get(receipt_field) != expected:
            raise ValueError(f"fold-metric receipt does not match current input: {receipt_field}")

    rows = _read_csv(historical_metrics_path)
    expected_keys = {
        (fold, branch, model)
        for fold in fold_ids for branch in trainer.FILTER_BRANCHES
        for model in trainer.MODEL_CANDIDATES
    }
    selections: dict[tuple[str, str, str], tuple[float | int | None, float]] = {}
    for row in rows:
        key = (row["outer_fold"], row["filter_branch"], row["model"])
        if key in selections or key not in expected_keys:
            raise ValueError("historical nested-selection rows contain a duplicate or unknown fold/configuration")
        raw_parameter = row["selected_parameter"]
        if raw_parameter == "fixed":
            parameter = None
        elif key[2] == "knn":
            parameter = int(float(raw_parameter))
        else:
            parameter = float(raw_parameter)
        valid_candidates = trainer.MODEL_CANDIDATES[key[2]]
        if parameter not in valid_candidates:
            raise ValueError("historical selected parameter is outside the frozen candidate grid")
        inner_score = float(row["inner_validation_macro_f1"])
        if not math.isfinite(inner_score) or not 0.0 <= inner_score <= 1.0:
            raise ValueError("historical inner validation score is not finite and in [0,1]")
        selections[key] = (parameter, inner_score)
    if set(selections) != expected_keys:
        raise ValueError("audited nested-selection file does not cover all expected outer/configuration rows")
    audit = {
        "reused_prior_nested_inner_selections": True,
        "selection_rows": len(rows),
        "input_hashes_match_current_inputs": True,
        "benchmark_receipt_path": str(benchmark_receipt_path),
        "fold_metric_receipt_path": str(metric_receipt_path),
        "fold_metrics_sha256_verified": True,
        "outer_test_used_for_selection": False,
    }
    return selections, audit


def _aggregate_roc_rows(
    score_vectors: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[tuple[str, np.ndarray]]] = defaultdict(list)
    for (fold, branch, model, scenario), (truth, score) in score_vectors.items():
        for class_index, label in enumerate(LABELS):
            binary = trainer.binary_ranking_metrics(truth == class_index, score[:, class_index])
            curve = interpolate_roc_curve(binary["fpr"], binary["tpr"], ROC_GRID)
            grouped[(branch, model, scenario, label)].append((fold, curve))
    output: list[dict[str, object]] = []
    for key, fold_curves in sorted(grouped.items()):
        folds = [fold for fold, _curve in fold_curves]
        if len(folds) != 15 or len(set(folds)) != 15:
            raise ValueError("each aggregate ROC curve requires one curve per held-out participant")
        matrix = np.vstack([curve for _fold, curve in fold_curves])
        rng = np.random.default_rng(seed + sum(map(ord, "".join(key))))
        indices = rng.integers(0, matrix.shape[0], size=(bootstrap_replicates, matrix.shape[0]))
        bootstrap_means = matrix[indices].mean(axis=1)
        low, high = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
        branch, model, scenario, label = key
        for index, fpr in enumerate(ROC_GRID):
            output.append({
                "filter_branch": branch,
                "model": model,
                "scenario": scenario,
                "one_vs_rest_class": label,
                "false_positive_rate": float(fpr),
                "mean_sensitivity_subject": float(matrix[:, index].mean()),
                "sensitivity_ci95_low_subject": float(low[index]),
                "sensitivity_ci95_high_subject": float(high[index]),
                "outer_test_participants": 15,
                "target": "within-recording temporal-position proxy; not fatigue ground truth",
                "curve_aggregation": "mean interpolated participant ROC; percentile bootstrap by held-out participant",
            })
    return output


def _svg_text(x: float, y: float, text: str, *, size: int = 11, anchor: str = "middle", color: str = "#222") -> str:
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{safe}</text>'


def write_roc_svg(path: Path, curve_rows: list[dict[str, object]]) -> None:
    """Write a vector panel figure for clean nested-LOSO OvR ROC curves."""
    clean = [row for row in curve_rows if row["scenario"] == SCENARIO_CLEAN]
    branches = (*trainer.FILTER_BRANCHES, "inner_cv_selected_per_fold")
    models = (*trainer.MODEL_CANDIDATES, "nested_selected_pipeline")
    labels = LABELS
    width, height = 1560, 810
    left, top, panel_w, panel_h = 40, 62, 250, 226
    colors = {
        "majority": "#777777", "nearest_centroid": "#2E86AB", "knn": "#F18F01",
        "ridge": "#6A4C93", "logistic": "#2A9D8F", "nested_selected_pipeline": "#C0392B",
    }
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fff"/>',
            _svg_text(width / 2, 24, "ROC one-vs-rest para posición temporal; pliegue de prueba LOSO por participante (sin cohorte externa)", size=16),
            _svg_text(width / 2, 43, "Proxy temporal, no fatiga fisiológica. Líneas: media de 15 participantes; IC95% por bootstrap en CSV.", size=11, color="#555")]
    for row_i, label in enumerate(labels):
        for col_i, branch in enumerate(branches):
            x0 = left + col_i * 255
            y0 = top + row_i * 235
            px, py, pw, ph = x0 + 35, y0 + 20, 195, 150
            body.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#fff" stroke="#888" stroke-width="0.8"/>')
            for tick in (0.0, 0.5, 1.0):
                tx = px + tick * pw
                ty = py + (1.0 - tick) * ph
                body.append(f'<path d="M {tx:.1f} {py:.1f} V {py+ph:.1f} M {px:.1f} {ty:.1f} H {px+pw:.1f}" stroke="#e5e7eb" stroke-width="0.7"/>')
                body.append(_svg_text(tx, py + ph + 13, f"{tick:.1f}", size=8))
                body.append(_svg_text(px - 12, ty + 3, f"{tick:.1f}", size=8, anchor="end"))
            body.append(f'<path d="M {px} {py+ph} L {px+pw} {py}" stroke="#999" stroke-dasharray="4 3" stroke-width="1" fill="none"/>')
            body.append(_svg_text(px + pw / 2, y0 + 12, f"{label} · {branch}", size=10))
            body.append(_svg_text(px + pw / 2, py + ph + 29, "False-positive rate", size=8))
            body.append(_svg_text(x0 + 7, py + ph / 2, "Sensitivity", size=8))
            selected_models = ("nested_selected_pipeline",) if branch == "inner_cv_selected_per_fold" else tuple(trainer.MODEL_CANDIDATES)
            for model_index, model in enumerate(selected_models):
                table = [r for r in clean if r["filter_branch"] == branch and r["model"] == model and r["one_vs_rest_class"] == label]
                if not table:
                    continue
                table.sort(key=lambda r: float(r["false_positive_rate"]))
                points = []
                for r in table:
                    xx = px + float(r["false_positive_rate"]) * pw
                    yy = py + (1.0 - float(r["mean_sensitivity_subject"])) * ph
                    points.append(f"{xx:.1f},{yy:.1f}")
                color = colors[model]
                body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{1.7 if model == "nested_selected_pipeline" else 1.25}"/>')
                lx = px + 8 + (model_index % 3) * 65
                ly = py + ph - 8 - (model_index // 3) * 13
                body.append(f'<path d="M {lx} {ly-3} h 9" stroke="{color}" stroke-width="2"/>')
                body.append(_svg_text(lx + 12, ly, model.replace("_", " "), size=7, anchor="start", color=color))
    body.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _verify_historical_hard_metrics(
    new_rows: list[dict[str, object]],
    historical_path: Path,
    *,
    tolerance: float = 1.1e-6,
) -> dict[str, object]:
    historical = _read_csv(historical_path)
    old_map = {(r["outer_fold"], r["filter_branch"], r["model"]): r for r in historical}
    clean = [r for r in new_rows if r["scenario"] == SCENARIO_CLEAN and r["model"] != "nested_selected_pipeline"]
    if len(old_map) != 375 or len(clean) != 375 or set(old_map) != {
        (str(r["outer_fold"]), str(r["filter_branch"]), str(r["model"])) for r in clean
    }:
        raise ValueError("new clean evaluation does not exactly cover the historical 375 folds")
    numeric_fields = ("accuracy", "macro_f1", "balanced_accuracy") + tuple(
        f"confusion_{i}{j}" for i in range(3) for j in range(3)
    )
    max_differences: dict[str, float] = {}
    mismatches: list[str] = []
    for row in clean:
        key = (str(row["outer_fold"]), str(row["filter_branch"]), str(row["model"]))
        old = old_map[key]
        for metric in numeric_fields:
            difference = abs(float(row[metric]) - float(old[metric]))
            max_differences[metric] = max(max_differences.get(metric, 0.0), difference)
            if difference > tolerance:
                mismatches.append(f"{key}:{metric}:{difference:.12g}")
    if mismatches:
        raise ValueError("clean hard metrics differ from historical predictions: " + "; ".join(mismatches[:5]))
    return {
        "historical_fold_rows": len(historical),
        "new_clean_fold_rows": len(clean),
        "metrics_compared": list(numeric_fields),
        "tolerance": tolerance,
        "max_abs_difference_by_metric": max_differences,
        "all_match": True,
    }


def run_evaluation(
    *,
    project_root: Path,
    private_root: Path,
    labels_path: Path,
    partitions_path: Path,
    signals_root: Path,
    historical_metrics_path: Path,
    benchmark_receipt_path: Path,
    private_output_path: Path,
    aggregate_output_path: Path,
    roc_output_path: Path,
    figure_output_path: Path,
    receipt_path: Path,
    bootstrap_replicates: int = N_BOOTSTRAP_DEFAULT,
) -> dict[str, object]:
    require_new_output_paths((private_output_path, aggregate_output_path, roc_output_path,
                              figure_output_path, receipt_path))
    labels = trainer._read_csv(labels_path)
    partitions = trainer._read_csv(partitions_path)
    records, common_counts, signal_digest, signal_segments = trainer.extract_filtered_features(labels_path, signals_root)
    participants = {str(row["participant_id"]) for row in records}
    outer_folds, inner_folds = trainer.validate_nested_partitions(partitions, participants)
    fold_ids = sorted(
        {str(row["outer_fold"]) for row in partitions if row.get("level") == "outer"},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    if len(labels) != 414 or len(participants) != 15 or len(fold_ids) != 15:
        raise ValueError("evaluation requires the frozen 414 segments and 15 participant-held-out folds")
    audited_selections, selection_reuse_audit = load_audited_inner_selections(
        historical_metrics_path, benchmark_receipt_path, fold_ids,
        labels_path=labels_path, partitions_path=partitions_path,
        combined_signal_digest=signal_digest,
    )
    y = np.asarray([trainer.LABEL_TO_INT[row["label"]] for row in records], dtype=np.int64)
    if len(signal_segments) != len(records):
        raise ValueError("signal copies and feature rows do not have matching frozen order")

    scenario_specs = trainer.synthetic_noise_scenarios(float(signal_segments[0]["fs_hz"]))
    scenarios = (SCENARIO_CLEAN, *(name for name, _kind, _params in scenario_specs))
    fold_rows: list[dict[str, object]] = []
    score_vectors: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    inner_configuration_scores: dict[str, dict[tuple[str, str], float]] = {fold: {} for fold in fold_ids}
    config_fold_data: dict[tuple[str, str, str, str], dict[str, object]] = {}
    outer_fit_count = 0
    logistic_diagnostics: list[dict[str, object]] = []

    for fold_index, (fold_id, outer_roles) in enumerate(zip(fold_ids, outer_folds)):
        train_ids = {participant for participant, role in outer_roles.items() if role == "train"}
        test_ids = {participant for participant, role in outer_roles.items() if role == "test"}
        if len(test_ids) != 1:
            raise ValueError("outer fold does not hold out exactly one complete participant")
        train_indices = trainer._indices_for_groups(records, train_ids)
        test_indices = trainer._indices_for_groups(records, test_ids)
        truth = y[test_indices]
        # Perturbed held-out feature vectors do not depend on model family or
        # hyperparameter. Build once per filter/fold/scenario and reuse across
        # the five model families instead of repeating expensive signal sorts.
        stress_feature_cache: dict[tuple[str, str], np.ndarray] = {}
        for stress_branch in trainer.FILTER_BRANCHES:
            for scenario_index, scenario in enumerate(scenarios[1:], start=1):
                stress_feature_cache[(stress_branch, scenario)] = np.vstack([
                    perturbed_feature_vector(
                        signal_segments[int(segment_index)], stress_branch, scenario,
                        seed=20261014 + fold_index * 100000 + int(segment_index) * 100 + scenario_index,
                    )
                    for segment_index in test_indices
                ])
        per_fold_config: dict[tuple[str, str], dict[str, object]] = {}

        for branch in trainer.FILTER_BRANCHES:
            x = np.asarray(
                [[float(row[f"{branch}__{feature}"]) for feature in trainer.PREDICTOR_COLUMNS] for row in records],
                dtype=np.float64,
            )
            if not np.isfinite(x).all():
                raise ValueError("frozen feature matrix contains nonfinite values")
            for model, candidates in trainer.MODEL_CANDIDATES.items():
                parameter, inner_score = audited_selections[(fold_id, branch, model)]
                inner_configuration_scores[fold_id][(branch, model)] = inner_score
                train_x, clean_test_x = trainer.standardize_train_test(x[train_indices], x[test_indices])
                train_center = x[train_indices].mean(axis=0)
                train_scale = x[train_indices].std(axis=0)
                train_scale[train_scale == 0.0] = 1.0
                test_matrices = {SCENARIO_CLEAN: clean_test_x}
                for scenario in scenarios[1:]:
                    noisy_matrix = stress_feature_cache[(branch, scenario)]
                    test_matrices[scenario] = (noisy_matrix - train_center) / train_scale
                score_lengths = {name: matrix.shape[0] for name, matrix in test_matrices.items()}
                stacked_test_x = np.vstack([test_matrices[name] for name in scenarios])
                stacked_scores = trainer.predict_model(
                    model, parameter, train_x, y[train_indices], stacked_test_x,
                    fit_diagnostics=logistic_diagnostics, fit_context="outer_clean_and_stress_score_metrics",
                    return_scores=True,
                )
                scenario_scores: dict[str, np.ndarray] = {}
                start = 0
                for name in scenarios:
                    stop = start + score_lengths[name]
                    scenario_scores[name] = stacked_scores[start:stop].copy()
                    start = stop
                score = scenario_scores[SCENARIO_CLEAN]
                clean_row = _score_row(
                    fold_id=fold_id, branch=branch, model=model, parameter=parameter,
                    inner_score=inner_score, scenario=SCENARIO_CLEAN, truth=truth,
                    scores=score, n_segments=len(test_indices),
                )
                fold_rows.append(clean_row)
                score_vectors[(fold_id, branch, model, SCENARIO_CLEAN)] = (truth.copy(), score.copy())
                data_for_fold: dict[str, object] = {
                    "clean_row": clean_row,
                    "scores": scenario_scores,
                    "parameter": parameter,
                    "inner_score": inner_score,
                    "x": x,
                    "train_scaled": train_x,
                    "train_y": y[train_indices],
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "train_center": train_center,
                    "train_scale": train_scale,
                }

                for scenario_index, scenario in enumerate(scenarios[1:], start=1):
                    noisy_scores = scenario_scores[scenario]
                    stress_row = _score_row(
                        fold_id=fold_id, branch=branch, model=model, parameter=parameter,
                        inner_score=inner_score, scenario=scenario, truth=truth,
                        scores=noisy_scores, n_segments=len(test_indices),
                    )
                    fold_rows.append(stress_row)
                    data_for_fold["scores"][scenario] = noisy_scores.copy()
                    score_vectors[(fold_id, branch, model, scenario)] = (truth.copy(), noisy_scores.copy())

                clean_metrics = data_for_fold["clean_row"]
                for scenario in scenarios[1:]:
                    row = next(r for r in fold_rows if r["outer_fold"] == fold_id and r["filter_branch"] == branch and r["model"] == model and r["scenario"] == scenario)
                    for metric, delta_name in (
                        ("macro_f1", "delta_macro_f1_vs_clean"),
                        ("balanced_accuracy", "delta_balanced_accuracy_vs_clean"),
                        ("macro_roc_auc_ovr", "delta_macro_roc_auc_ovr_vs_clean"),
                        ("macro_average_precision_ovr", "delta_macro_average_precision_ovr_vs_clean"),
                    ):
                        row[delta_name] = float(row[metric]) - float(clean_metrics[metric])
                per_fold_config[(branch, model)] = data_for_fold
                config_fold_data[(fold_id, branch, model, "__data__")] = data_for_fold
                outer_fit_count += 1

        selected_branch, selected_model = trainer.choose_configuration(inner_configuration_scores[fold_id])
        selected_data = per_fold_config[(selected_branch, selected_model)]
        for scenario in scenarios:
            source = next(
                r for r in fold_rows
                if r["outer_fold"] == fold_id and r["filter_branch"] == selected_branch
                and r["model"] == selected_model and r["scenario"] == scenario
            )
            nested = dict(source)
            nested.update({
                "filter_branch": "inner_cv_selected_per_fold",
                "model": "nested_selected_pipeline",
            })
            fold_rows.append(nested)
            score_vectors[(fold_id, "inner_cv_selected_per_fold", "nested_selected_pipeline", scenario)] = (
                truth.copy(), selected_data["scores"][scenario].copy()
            )

    if outer_fit_count != 375 or len(fold_rows) != (25 * 15 * len(scenarios) + 15 * len(scenarios)):
        raise ValueError("scoring coverage differs from 25 candidates, 15 folds, and all scenarios")
    historical_check = _verify_historical_hard_metrics(fold_rows, historical_metrics_path)

    # Only the private output carries fold identifiers; scores/predictions are never written.
    private_root = Path(private_root)
    trainer.write_private_fold_metrics_csv(private_output_path, fold_rows, private_root)
    aggregate_rows = aggregate_subject_metrics(
        fold_rows, bootstrap_replicates=bootstrap_replicates, seed=20261014
    )
    _write_csv(aggregate_output_path, aggregate_rows)
    roc_rows = _aggregate_roc_rows(
        score_vectors, bootstrap_replicates=bootstrap_replicates, seed=20261015
    )
    _write_csv(roc_output_path, roc_rows)
    write_roc_svg(figure_output_path, roc_rows)

    clean_rows = [r for r in aggregate_rows if r["scenario"] == SCENARIO_CLEAN]
    selected_row = next(r for r in clean_rows if r["model"] == "nested_selected_pipeline")
    baseline_row = next(
        r for r in clean_rows if r["filter_branch"] == "control" and r["model"] == "majority"
    )
    receipt = {
        "protocol": "proxy_temporal_v1_nested_loso_full_score_roc_synthetic_stress",
        "target": "within-recording temporal position only; not fatigue ground truth",
        "participants": 15,
        "segments": 414,
        "class_counts": {label: 138 for label in LABELS},
        "outer_folds": 15,
        "inner_folds_per_outer": 4,
        "filters": list(trainer.FILTER_BRANCHES),
        "models": list(trainer.MODEL_CANDIDATES),
        "candidate_configurations": 25,
        "outer_fits": outer_fit_count,
        "private_rows_including_nested_selection": len(fold_rows),
        "scenarios": list(scenarios),
        "synthetic_test_copies": "post-filter residual perturbations of held-out signal copies; training/selection fixed on clean data; not measured noise and not an optimized adversarial attack",
        "synthetic_test_copy_count": 414 * 25 * (len(scenarios) - 1),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": "held-out participant; resampling within outer LOSO folds",
        "roc_strategy": "per-class one-vs-rest ranking; mean interpolated participant ROC with participant bootstrap interval",
        "probability_metrics": "Brier and log loss only for probability-like outputs; these are not calibration validation",
        "historical_hard_metric_crosscheck": historical_check,
        "audited_selection_reuse": selection_reuse_audit,
        "nested_selected_clean_macro_f1_mean_subject": selected_row["macro_f1_mean_subject"],
        "nested_selected_clean_macro_roc_auc_ovr_mean_subject": selected_row["macro_roc_auc_ovr_mean_subject"],
        "control_majority_clean_macro_roc_auc_ovr_mean_subject": baseline_row["macro_roc_auc_ovr_mean_subject"],
        "inputs_sha256": {
            "labels": _sha256(labels_path), "partitions": _sha256(partitions_path),
            "historical_hard_metrics": _sha256(historical_metrics_path),
            "private_filter_traces_combined": signal_digest,
        },
        "outputs_sha256": {},
        "common_valid_samples_by_filter": common_counts,
        "study_signals_modified": False,
        "participant_ids_predictions_and_scores_exported": False,
        "external_datasets_services_or_models_used": False,
        "fatigue_prediction_performed": False,
        "limitations": [
            "labels identify relative thirds of recording time, not a fatigue ground truth",
            "15 participants limit precision; intervals describe this sample; no external-cohort validation is performed",
            "synthetic corruption tests sensitivity to chosen scenarios only; not empirically calibrated to acquisition noise",
            "calibration, fairness, drift, A/B deployment, LLM safety and general intelligence benchmarks are not evaluated",
        ],
        "logistic_fits_converged": all(bool(row.get("converged")) for row in logistic_diagnostics),
        "logistic_fit_count_outer_scoring": len(logistic_diagnostics),
        "logistic_retry_fit_count": sum(int(row.get("retry_count", 0)) > 0 for row in logistic_diagnostics),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths = [private_output_path, aggregate_output_path, roc_output_path, figure_output_path, receipt_path]
    receipt["outputs_sha256"] = {path.name: _sha256(path) for path in output_paths if path != receipt_path}
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "aggregate_rows": len(aggregate_rows), "roc_curve_points": len(roc_rows),
        "private_fold_rows": len(fold_rows), "outer_fits": outer_fit_count,
        "scenario_count": len(scenarios), "historical_crosscheck": historical_check,
        "nested_selected_clean": selected_row,
        "majority_baseline_clean": baseline_row,
        "aggregate_csv": str(aggregate_output_path), "roc_csv": str(roc_output_path),
        "figure_svg": str(figure_output_path), "private_metrics_csv": str(private_output_path),
        "receipt": str(receipt_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOTSTRAP_DEFAULT)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--signals-root", type=Path, required=True)
    parser.add_argument("--historical-metrics", type=Path, required=True)
    parser.add_argument("--benchmark-receipt", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--roc-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    project_root = Path(__file__).resolve().parents[1]
    result = run_evaluation(
        project_root=project_root, private_root=args.private_root,
        labels_path=args.labels, partitions_path=args.partitions,
        signals_root=args.signals_root, historical_metrics_path=args.historical_metrics,
        benchmark_receipt_path=args.benchmark_receipt,
        private_output_path=args.private_output, aggregate_output_path=args.aggregate_output,
        roc_output_path=args.roc_output, figure_output_path=args.figure_output,
        receipt_path=args.receipt, bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
