"""Post-hoc left/right audit of participant-held-out temporal-proxy predictions.

The script does not fit models or select different models by leg. It reports
all fixed benchmark configurations and the nested selected procedure by side,
with participants as the unit for summaries and paired side contrasts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import entrenar_proxy_temporal as base
import evaluar_benchmark_expandido_proxy as benchmark
from release_paths import validate_data_paths


SCORE_COLUMNS = ("score_temprana", "score_transicion", "score_tardia")
SUMMARY_METRICS = (
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy",
    "matthews_corrcoef", "cohen_kappa", "macro_roc_auc_ovr", "macro_average_precision_ovr",
)
SEED = 20261019
BOOTSTRAP_REPLICATES = 2000


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty aggregate: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("aggregate rows do not share a deterministic field order")
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_interval(values: list[float], *, replicates: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("bootstrap input must contain finite participant-level values")
    if replicates < 200:
        raise ValueError("at least 200 bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _selected_rows(
    prediction_rows: list[dict[str, str]], selection_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected_by_fold: dict[str, str] = {}
    for row in selection_rows:
        fold = str(row.get("outer_fold", "")).strip()
        configuration = str(row.get("selected_configuration_id", "")).strip()
        if not fold or not configuration or fold in selected_by_fold:
            raise ValueError("selection manifest contains a missing or duplicate outer-fold choice")
        selected_by_fold[fold] = configuration
    if not selected_by_fold:
        raise ValueError("no nested selection rows were supplied")
    selected: list[dict[str, str]] = []
    for row in prediction_rows:
        fold = str(row.get("outer_fold", "")).strip()
        if fold in selected_by_fold and row.get("configuration_id") == selected_by_fold[fold]:
            item = dict(row)
            item["analysis_configuration_id"] = "nested_selected_pipeline"
            item["filter_branch"] = "inner_cv_selected"
            item["feature_set"] = "inner_selected"
            selected.append(item)
    expected_folds = set(selected_by_fold)
    seen_folds = {row["outer_fold"] for row in selected}
    if seen_folds != expected_folds:
        raise ValueError("selected out-of-fold rows do not cover every declared outer fold")
    return selected


def _participant_side_metrics(
    grouped_rows: dict[tuple[str, str, str], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    registry = benchmark.model_registry()
    results: list[dict[str, Any]] = []
    for (configuration_id, participant, side), rows in sorted(grouped_rows.items()):
        if side not in {"left", "right"}:
            raise ValueError("predictions contain an unrecognized laterality value")
        models = {row["model"] for row in rows}
        if len(models) != 1 or next(iter(models)) not in registry:
            raise ValueError("one participant-side group must come from one known model family")
        model = next(iter(models))
        truth = np.asarray([base.LABEL_TO_INT[row["true_proxy_label"]] for row in rows], dtype=np.int64)
        predicted = np.asarray([base.LABEL_TO_INT[row["predicted_proxy_label"]] for row in rows], dtype=np.int64)
        scores = np.asarray([[float(row[name]) for name in SCORE_COLUMNS] for row in rows], dtype=np.float64)
        if scores.shape != (len(rows), 3) or not np.isfinite(scores).all():
            raise ValueError("held-out prediction scores must form a finite three-class matrix")
        if not np.array_equal(np.argmax(scores, axis=1), predicted):
            raise ValueError("stored hard predictions disagree with the highest class score")
        if set(np.unique(truth)) != set(base.LABEL_TO_INT.values()):
            raise ValueError("each participant-side group must include all temporal-proxy classes")
        scores_type = str(registry[model]["score_type"])
        metrics = {
            **base.classification_metrics(truth, predicted),
            **base.classification_score_metrics(truth, scores, score_type=scores_type),
        }
        metrics.update({
            "analysis_configuration_id": configuration_id,
            "participant_id": participant,
            "side": side,
            "segments": len(rows),
            "model": model,
            "score_type": scores_type,
        })
        results.append(metrics)
    return results


def _aggregate_side_metrics(
    participant_metrics: list[dict[str, Any]], prediction_rows: list[dict[str, str]],
    *, bootstrap_replicates: int, seed: int,
) -> list[dict[str, Any]]:
    identifiers = sorted({str(row["analysis_configuration_id"]) for row in participant_metrics})
    results: list[dict[str, Any]] = []
    for config_index, config_id in enumerate(identifiers):
        config_metrics = [row for row in participant_metrics if row["analysis_configuration_id"] == config_id]
        persons_by_side = {
            side: {str(row["participant_id"]) for row in config_metrics if row["side"] == side}
            for side in ("left", "right")
        }
        paired = persons_by_side["left"] & persons_by_side["right"]
        for side_index, side in enumerate(("left", "right")):
            rows = [row for row in config_metrics if row["side"] == side]
            if not rows:
                continue
            public: dict[str, Any] = {
                "configuration_id": config_id,
                "side": side,
                "participants": len(rows),
                "paired_participants_with_other_side": len(paired),
                "segments": sum(int(row["segments"]) for row in rows),
                "target": "within-recording temporal thirds; not fatigue ground truth",
                "estimand": "unweighted mean of participant-side metrics",
            }
            for metric_index, metric in enumerate(SUMMARY_METRICS):
                values = [float(row[metric]) for row in rows]
                lower, upper = _bootstrap_interval(values, replicates=bootstrap_replicates,
                    seed=seed + config_index * 31 + side_index * 100000 + metric_index)
                public[f"{metric}_mean_subject"] = float(np.mean(values))
                public[f"{metric}_sd_subject"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                public[f"{metric}_ci95_low_subject_bootstrap"] = lower
                public[f"{metric}_ci95_high_subject_bootstrap"] = upper
            for label in base.LABEL_TO_INT:
                for metric in (f"f1_{label}", f"recall_{label}"):
                    public[f"{metric}_mean_subject"] = float(np.mean([row[metric] for row in rows]))
            results.append(public)
    return results


def _paired_side_contrasts(
    participant_metrics: list[dict[str, Any]], *, bootstrap_replicates: int, seed: int,
) -> list[dict[str, Any]]:
    identifiers = sorted({str(row["analysis_configuration_id"]) for row in participant_metrics})
    rows: list[dict[str, Any]] = []
    for config_index, config_id in enumerate(identifiers):
        left = {str(row["participant_id"]): row for row in participant_metrics
            if row["analysis_configuration_id"] == config_id and row["side"] == "left"}
        right = {str(row["participant_id"]): row for row in participant_metrics
            if row["analysis_configuration_id"] == config_id and row["side"] == "right"}
        paired_ids = sorted(set(left) & set(right))
        if not paired_ids:
            continue
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            differences = np.asarray([float(left[person][metric]) - float(right[person][metric])
                for person in paired_ids], dtype=np.float64)
            lower, upper = _bootstrap_interval(differences.tolist(), replicates=bootstrap_replicates,
                seed=seed + 500000 + config_index * 31 + metric_index)
            rows.append({
                "configuration_id": config_id,
                "metric": metric,
                "paired_participants": len(paired_ids),
                "contrast": "left minus right; paired by held-out student",
                "mean_left_minus_right": float(np.mean(differences)),
                "median_left_minus_right": float(np.median(differences)),
                "q1_left_minus_right": float(np.quantile(differences, 0.25)),
                "q3_left_minus_right": float(np.quantile(differences, 0.75)),
                "ci95_low_paired_subject_bootstrap": lower,
                "ci95_high_paired_subject_bootstrap": upper,
            })
    return rows


def _participant_series_metrics(
    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Compute participant-level metrics separately for each side and R1-R5 series."""
    registry = benchmark.model_registry()
    results: list[dict[str, Any]] = []
    for (configuration_id, participant, side, series), rows in sorted(grouped_rows.items()):
        if side not in {"left", "right"} or series not in {f"R{i}" for i in range(1, 6)}:
            raise ValueError("predictions contain an unrecognized side or R1-R5 series")
        models = {row["model"] for row in rows}
        if len(models) != 1 or next(iter(models)) not in registry:
            raise ValueError("one participant-side-series group must come from one known model family")
        model = next(iter(models))
        truth = np.asarray([base.LABEL_TO_INT[row["true_proxy_label"]] for row in rows], dtype=np.int64)
        predicted = np.asarray([base.LABEL_TO_INT[row["predicted_proxy_label"]] for row in rows], dtype=np.int64)
        scores = np.asarray([[float(row[name]) for name in SCORE_COLUMNS] for row in rows], dtype=np.float64)
        if scores.shape != (len(rows), 3) or not np.isfinite(scores).all():
            raise ValueError("held-out series scores must form a finite three-class matrix")
        if not np.array_equal(np.argmax(scores, axis=1), predicted):
            raise ValueError("stored hard predictions disagree with the highest class score")
        if set(np.unique(truth)) != set(base.LABEL_TO_INT.values()):
            raise ValueError("each recorded R1-R5 series must contain all three frozen time-position labels")
        score_type = str(registry[model]["score_type"])
        metrics = {
            **base.classification_metrics(truth, predicted),
            **base.classification_score_metrics(truth, scores, score_type=score_type),
        }
        metrics.update({
            "analysis_configuration_id": configuration_id,
            "participant_id": participant,
            "side": side,
            "series": series,
            "segments": len(rows),
            "model": model,
        })
        results.append(metrics)
    return results


def _aggregate_series_side_metrics(
    participant_metrics: list[dict[str, Any]], *, bootstrap_replicates: int, seed: int,
) -> list[dict[str, Any]]:
    configurations = sorted({str(row["analysis_configuration_id"]) for row in participant_metrics})
    series_values = sorted({str(row["series"]) for row in participant_metrics})
    results: list[dict[str, Any]] = []
    for config_index, configuration in enumerate(configurations):
        for series_index, series in enumerate(series_values):
            for side_index, side in enumerate(("left", "right")):
                rows = [row for row in participant_metrics if row["analysis_configuration_id"] == configuration
                    and row["series"] == series and row["side"] == side]
                if not rows:
                    continue
                public: dict[str, Any] = {
                    "configuration_id": configuration,
                    "side": side,
                    "series": series,
                    "participants": len(rows),
                    "segments": sum(int(row["segments"]) for row in rows),
                    "target": "within-recording temporal thirds; not fatigue ground truth",
                    "estimand": "unweighted mean of participant-side-series metrics",
                }
                for metric_index, metric in enumerate(SUMMARY_METRICS):
                    values = [float(row[metric]) for row in rows]
                    lower, upper = _bootstrap_interval(values, replicates=bootstrap_replicates,
                        seed=seed + 100000 + config_index * 311 + series_index * 31 + side_index * 1000 + metric_index)
                    public[f"{metric}_mean_subject"] = float(np.mean(values))
                    public[f"{metric}_sd_subject"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    public[f"{metric}_ci95_low_subject_bootstrap"] = lower
                    public[f"{metric}_ci95_high_subject_bootstrap"] = upper
                for label in base.LABEL_TO_INT:
                    for metric in (f"f1_{label}", f"recall_{label}"):
                        public[f"{metric}_mean_subject"] = float(np.mean([row[metric] for row in rows]))
                results.append(public)
    return results


def _paired_series_side_contrasts(
    participant_metrics: list[dict[str, Any]], *, bootstrap_replicates: int, seed: int,
) -> list[dict[str, Any]]:
    configurations = sorted({str(row["analysis_configuration_id"]) for row in participant_metrics})
    series_values = sorted({str(row["series"]) for row in participant_metrics})
    results: list[dict[str, Any]] = []
    for config_index, configuration in enumerate(configurations):
        for series_index, series in enumerate(series_values):
            left = {str(row["participant_id"]): row for row in participant_metrics
                if row["analysis_configuration_id"] == configuration and row["series"] == series and row["side"] == "left"}
            right = {str(row["participant_id"]): row for row in participant_metrics
                if row["analysis_configuration_id"] == configuration and row["series"] == series and row["side"] == "right"}
            paired = sorted(set(left) & set(right))
            if not paired:
                continue
            for metric_index, metric in enumerate(SUMMARY_METRICS):
                differences = [float(left[person][metric]) - float(right[person][metric]) for person in paired]
                lower, upper = _bootstrap_interval(differences, replicates=bootstrap_replicates,
                    seed=seed + 700000 + config_index * 311 + series_index * 31 + metric_index)
                results.append({
                    "configuration_id": configuration,
                    "series": series,
                    "metric": metric,
                    "paired_participants": len(paired),
                    "contrast": "left minus right; paired by held-out student within series",
                    "mean_left_minus_right": float(np.mean(differences)),
                    "median_left_minus_right": float(np.median(differences)),
                    "q1_left_minus_right": float(np.quantile(differences, 0.25)),
                    "q3_left_minus_right": float(np.quantile(differences, 0.75)),
                    "ci95_low_paired_subject_bootstrap": lower,
                    "ci95_high_paired_subject_bootstrap": upper,
                })
    return results


def _confusion_by_side(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    classes = list(base.LABEL_TO_INT)
    identifiers = sorted({row.get("analysis_configuration_id", row["configuration_id"]) for row in prediction_rows})
    output: list[dict[str, Any]] = []
    for config_id in identifiers:
        for side in ("left", "right"):
            counts = np.zeros((3, 3), dtype=np.int64)
            selected = [row for row in prediction_rows
                if row.get("analysis_configuration_id", row.get("configuration_id")) == config_id
                and row.get("leg", "").strip().lower() == side]
            for row in selected:
                counts[base.LABEL_TO_INT[row["true_proxy_label"]],
                    base.LABEL_TO_INT[row["predicted_proxy_label"]]] += 1
            if not selected:
                continue
            for true_index, true_label in enumerate(classes):
                for pred_index, pred_label in enumerate(classes):
                    output.append({
                        "configuration_id": config_id,
                        "side": side,
                        "series": "all_R1_R5",
                        "true_proxy_class": true_label,
                        "predicted_proxy_class": pred_label,
                        "segment_count": int(counts[true_index, pred_index]),
                    })
    return output


def _confusion_by_side_series(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    classes = list(base.LABEL_TO_INT)
    configurations = sorted({row.get("analysis_configuration_id", row["configuration_id"]) for row in prediction_rows})
    output: list[dict[str, Any]] = []
    for configuration in configurations:
        for side in ("left", "right"):
            for series in sorted({row["recording_id"] for row in prediction_rows if row["recording_id"] in {f"R{i}" for i in range(1, 6)}}):
                selected = [row for row in prediction_rows
                    if row.get("analysis_configuration_id", row.get("configuration_id")) == configuration
                    and row.get("leg", "").strip().lower() == side and row.get("recording_id") == series]
                if not selected:
                    continue
                counts = np.zeros((3, 3), dtype=np.int64)
                for row in selected:
                    counts[base.LABEL_TO_INT[row["true_proxy_label"]], base.LABEL_TO_INT[row["predicted_proxy_label"]]] += 1
                for true_index, true_label in enumerate(classes):
                    for pred_index, pred_label in enumerate(classes):
                        output.append({
                            "configuration_id": configuration,
                            "side": side,
                            "series": series,
                            "true_proxy_class": true_label,
                            "predicted_proxy_class": pred_label,
                            "segment_count": int(counts[true_index, pred_index]),
                        })
    return output


def render_nested_laterality_heatmap(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Render aggregate macro-F1 and ROC-AUC by leg and series as editable SVG."""
    selected = [row for row in summary_rows if row["configuration_id"] == "nested_selected_pipeline"]
    index = {(str(row["side"]), str(row["series"])): row for row in selected}
    expected = {(side, f"R{i}") for side in ("left", "right") for i in range(1, 6)}
    if set(index) != expected:
        raise ValueError("nested side-by-series summary must cover left/right and R1-R5")

    width, height = 1280, 560
    panel_x = (76, 672)
    panel_names = ("Macro-F1", "ROC-AUC OvR")
    metrics = ("macro_f1", "macro_roc_auc_ovr")
    cell_w, cell_h = 92, 106
    top = 190
    start_x_offset = 88
    side_names = {"left": "Izquierda", "right": "Derecha"}

    def color(value: float) -> str:
        value = min(1.0, max(0.0, value))
        low = (238, 244, 249)
        high = (16, 77, 112)
        rgb = tuple(round(a + (b - a) * value) for a, b in zip(low, high))
        return "#" + "".join(f"{part:02X}" for part in rgb)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#182230}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px}.panel{font-size:20px;font-weight:700}.head{font-size:14px;font-weight:700}.side{font-size:15px;font-weight:700}.value{font-size:17px;font-weight:700}.n{font-size:13px}.foot{font-size:13px;fill:#475467}</style>',
        f'<text x="{width/2}" y="42" text-anchor="middle" class="title">Rendimiento del clasificador por pierna y serie</text>',
        '<text x="640" y="70" text-anchor="middle" class="subtitle">Selector anidado fuera de pliegue; resultados post hoc sobre R1–R5</text>',
    ]
    for panel_index, (metric, title) in enumerate(zip(metrics, panel_names)):
        x0 = panel_x[panel_index]
        parts.append(f'<text x="{x0+280}" y="120" text-anchor="middle" class="panel">{title}</text>')
        for series_index in range(5):
            x = x0 + start_x_offset + series_index * cell_w
            parts.append(f'<text x="{x+cell_w/2}" y="157" text-anchor="middle" class="head">R{series_index+1}</text>')
        for side_index, side in enumerate(("left", "right")):
            y = top + side_index * cell_h
            parts.append(f'<text x="{x0+78}" y="{y+cell_h/2+5}" text-anchor="end" class="side">{side_names[side]}</text>')
            for series_index in range(5):
                series = f"R{series_index+1}"
                row = index[(side, series)]
                value = float(row[f"{metric}_mean_subject"])
                participants = int(row["participants"])
                x = x0 + start_x_offset + series_index * cell_w
                fill = color(value)
                text_fill = "#FFFFFF" if value >= 0.62 else "#182230"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="5" fill="{fill}" stroke="#FFFFFF" stroke-width="2"/>')
                parts.append(f'<text x="{x+cell_w/2}" y="{y+43}" text-anchor="middle" class="value" fill="{text_fill}">{value:.3f}</text>')
                parts.append(f'<text x="{x+cell_w/2}" y="{y+69}" text-anchor="middle" class="n" fill="{text_fill}">n={participants}</text>')
    parts.extend([
        '<text x="640" y="472" text-anchor="middle" class="foot">n = participantes con esa pierna y serie. Los IC bootstrap y las demás métricas están en las tablas CSV.</text>',
        '<text x="640" y="500" text-anchor="middle" class="foot">La etiqueta evaluada es el tercio temporal del archivo, no fatiga fisiológica. No se seleccionó el modelo por pierna o serie.</text>',
        '<text x="640" y="528" text-anchor="middle" class="foot">ROC-AUC usa puntajes de cada modelo; los valores no son probabilidades calibradas.</text>',
        '</svg>',
    ])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def assert_public_rows(rows: list[dict[str, Any]], participant_ids: set[str] | None = None) -> None:
    forbidden_fields = ("participant_id", "held_out_pseudonym", "recording_id", "filename", "outer_fold", "inner_fold", "set_index")
    forbidden_values = participant_ids or set()
    for row in rows:
        for key, value in row.items():
            if any(token in str(key).casefold() for token in forbidden_fields):
                raise ValueError("aggregate output contains a participant or record identifier field")
            if any(person in str(value) for person in forbidden_values):
                raise ValueError("aggregate output contains a participant identifier value")


def analyze_predictions(
    prediction_rows: list[dict[str, str]], selection_rows: list[dict[str, str]],
    *, bootstrap_replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not prediction_rows:
        raise ValueError("private prediction table is empty")
    required = {"outer_fold", "held_out_pseudonym", "leg", "recording_id", "set_index", "true_proxy_label",
        "predicted_proxy_label", "configuration_id", "model", *SCORE_COLUMNS}
    if not required.issubset(prediction_rows[0]):
        raise ValueError("private prediction table is missing required fields")
    fixed_rows = [dict(row, analysis_configuration_id=row["configuration_id"]) for row in prediction_rows]
    nested_rows = _selected_rows(prediction_rows, selection_rows)
    all_rows = fixed_rows + nested_rows
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        key = (row["analysis_configuration_id"], row["held_out_pseudonym"], row["leg"].strip().lower())
        grouped.setdefault(key, []).append(row)
    participant_metrics = _participant_side_metrics(grouped)
    summary = _aggregate_side_metrics(participant_metrics, all_rows,
        bootstrap_replicates=bootstrap_replicates, seed=seed)
    paired = _paired_side_contrasts(participant_metrics,
        bootstrap_replicates=bootstrap_replicates, seed=seed)
    confusion = _confusion_by_side(all_rows)
    grouped_series: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        key = (row["analysis_configuration_id"], row["held_out_pseudonym"],
            row["leg"].strip().lower(), row["recording_id"].strip().upper())
        grouped_series.setdefault(key, []).append(row)
    participant_series_metrics = _participant_series_metrics(grouped_series)
    by_series = _aggregate_series_side_metrics(participant_series_metrics,
        bootstrap_replicates=bootstrap_replicates, seed=seed)
    paired_by_series = _paired_series_side_contrasts(participant_series_metrics,
        bootstrap_replicates=bootstrap_replicates, seed=seed)
    confusion_by_series = _confusion_by_side_series(all_rows)
    participants_by_side = {
        side: len({row["participant_id"] for row in participant_metrics if row["side"] == side})
        for side in ("left", "right")
    }
    segments_by_side = {
        side: sum(row["leg"].strip().lower() == side for row in all_rows if row.get("analysis_configuration_id") == "nested_selected_pipeline")
        for side in ("left", "right")
    }
    nested_source_rows = [row for row in all_rows if row.get("analysis_configuration_id") == "nested_selected_pipeline"]
    recordings_by_side = {
        side: len({(row["held_out_pseudonym"], row["recording_id"]) for row in nested_source_rows
            if row["leg"].strip().lower() == side})
        for side in ("left", "right")
    }
    classes_by_side = {
        side: {label: sum(row["leg"].strip().lower() == side and row["true_proxy_label"] == label
            for row in nested_source_rows) for label in base.LABEL_TO_INT}
        for side in ("left", "right")
    }
    participants_by_leg_series = {
        side: {
            series: len({row["held_out_pseudonym"] for row in nested_source_rows
                if row["leg"].strip().lower() == side and row["recording_id"].strip().upper() == series})
            for series in (f"R{i}" for i in range(1, 6))
        }
        for side in ("left", "right")
    }
    classes_by_leg_series = {
        side: {
            series: {
                label: sum(row["leg"].strip().lower() == side
                    and row["recording_id"].strip().upper() == series
                    and row["true_proxy_label"] == label for row in nested_source_rows)
                for label in base.LABEL_TO_INT
            }
            for series in (f"R{i}" for i in range(1, 6))
        }
        for side in ("left", "right")
    }
    paired_participants_by_series = {
        series: len(
            {row["held_out_pseudonym"] for row in nested_source_rows
                if row["leg"].strip().lower() == "left" and row["recording_id"].strip().upper() == series}
            & {row["held_out_pseudonym"] for row in nested_source_rows
                if row["leg"].strip().lower() == "right" and row["recording_id"].strip().upper() == series}
        )
        for series in (f"R{i}" for i in range(1, 6))
    }
    config_ids = {row["analysis_configuration_id"] for row in participant_metrics}
    audit = {
        "fixed_configuration_count": len({row["configuration_id"] for row in prediction_rows}),
        "configuration_count_including_nested": len(config_ids),
        "participant_groups": len({row["held_out_pseudonym"] for row in nested_rows}),
        "selected_pipeline_rows": len(nested_rows),
        "participant_side_groups_nested": sum(row["analysis_configuration_id"] == "nested_selected_pipeline" for row in participant_metrics),
        "participants_by_side": participants_by_side,
        "segments_by_side": segments_by_side,
        "recordings_by_side": recordings_by_side,
        "classes_by_side": classes_by_side,
        "participants_by_leg_series": participants_by_leg_series,
        "segments_by_leg_series": {
            side: {
                series: sum(row["leg"].strip().lower() == side and row["recording_id"].strip().upper() == series
                    for row in nested_source_rows)
                for series in (f"R{i}" for i in range(1, 6))
            }
            for side in ("left", "right")
        },
        "classes_by_leg_series": classes_by_leg_series,
        "paired_participants_by_series": paired_participants_by_series,
        "paired_participant_groups": len({row["participant_id"] for row in participant_metrics
            if row["analysis_configuration_id"] == "nested_selected_pipeline" and row["side"] == "left"}
            & {row["participant_id"] for row in participant_metrics
            if row["analysis_configuration_id"] == "nested_selected_pipeline" and row["side"] == "right"}),
        "target": "within-recording temporal position only; no physiological fatigue ground truth",
        "posthoc_stratification_only": True,
        "model_refit_performed": False,
        "side_used_for_model_selection": False,
        "participant_covariates_used": False,
    }
    private_ids = {str(row["held_out_pseudonym"]) for row in prediction_rows}
    assert_public_rows(summary, private_ids)
    assert_public_rows(paired, private_ids)
    assert_public_rows(confusion, private_ids)
    assert_public_rows(by_series, private_ids)
    assert_public_rows(paired_by_series, private_ids)
    assert_public_rows(confusion_by_series, private_ids)
    confusion.extend(confusion_by_series)
    return summary, paired, by_series, paired_by_series, confusion, audit


def run_analysis(predictions_path: Path, selections_path: Path, output_dir: Path) -> dict[str, Any]:
    predictions_path, selections_path, output_dir = map(Path, (predictions_path, selections_path, output_dir))
    validate_data_paths(
        {"predictions": predictions_path, "selections": selections_path, "output_dir": output_dir},
        Path(__file__).resolve().parents[1],
    )
    prediction_rows, selection_rows = _read_csv(predictions_path), _read_csv(selections_path)
    if len(prediction_rows) != 325 * 414 or len(selection_rows) != 15:
        raise ValueError("private benchmark inputs must contain 325 complete configurations and 15 fold choices")
    by_configuration: dict[str, int] = {}
    row_keys = set()
    for row in prediction_rows:
        configuration = row["configuration_id"]
        by_configuration[configuration] = by_configuration.get(configuration, 0) + 1
        row_key = (row["configuration_id"], row["outer_fold"], row["recording_id"], row["leg"], row["set_index"], row["true_proxy_label"])
        row_keys.add(row_key)
    if len(by_configuration) != 325 or any(count != 414 for count in by_configuration.values()) or len(row_keys) != len(prediction_rows):
        raise ValueError("fixed prediction rows are incomplete, duplicated, or unevenly covered")
    summary, paired, by_series, paired_by_series, confusion, audit = analyze_predictions(prediction_rows, selection_rows)
    if audit["fixed_configuration_count"] != 325 or audit["participant_groups"] != 15:
        raise ValueError("joint benchmark input must contain 325 configurations and 15 held-out students")
    if audit["selected_pipeline_rows"] != 414:
        raise ValueError("nested selected pipeline must contain 414 held-out segment predictions")
    if audit["participants_by_side"] != {"left": 14, "right": 15}:
        raise ValueError("side coverage differs from the 15-student local corpus")
    if audit["segments_by_side"] != {"left": 210, "right": 204}:
        raise ValueError("selected segment counts differ from the local R1-R5 inventory")
    if audit["recordings_by_side"] != {"left": 70, "right": 68}:
        raise ValueError("selected recording counts differ from the local R1-R5 inventory")
    expected_series = {f"R{i}" for i in range(1, 6)}
    if any(set(audit["participants_by_leg_series"][side]) != expected_series for side in ("left", "right")):
        raise ValueError("participant coverage must enumerate all five R1-R5 series by leg")
    if sum(audit["segments_by_leg_series"]["left"].values()) != 210 or sum(audit["segments_by_leg_series"]["right"].values()) != 204:
        raise ValueError("participant-side-series segments do not reconcile to the 414 R1-R5 segments")
    for side in ("left", "right"):
        for series in expected_series:
            participants = audit["participants_by_leg_series"][side][series]
            segments = audit["segments_by_leg_series"][side][series]
            class_counts = audit["classes_by_leg_series"][side][series]
            if segments != 3 * participants or set(class_counts.values()) != {participants}:
                raise ValueError(f"frozen three-third labels do not reconcile for {side} {series}")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary": output_dir / "RESUMEN_MODELOS_PROXY_POR_LATERALIDAD.csv",
        "paired": output_dir / "DIFERENCIAS_PAREADAS_MODELOS_PROXY_POR_LATERALIDAD.csv",
        "series": output_dir / "RESUMEN_MODELOS_PROXY_POR_PIERNA_Y_SERIE.csv",
        "paired_series": output_dir / "DIFERENCIAS_PAREADAS_MODELOS_PROXY_POR_PIERNA_Y_SERIE.csv",
        "confusion": output_dir / "MATRICES_CONFUSION_MODELOS_PROXY_POR_LATERALIDAD.csv",
        "heatmap": output_dir / "HEATMAP_METRICAS_NESTED_POR_PIERNA_Y_SERIE.svg",
    }
    _write_csv(paths["summary"], summary)
    _write_csv(paths["paired"], paired)
    _write_csv(paths["series"], by_series)
    _write_csv(paths["paired_series"], paired_by_series)
    _write_csv(paths["confusion"], confusion)
    render_nested_laterality_heatmap(by_series, paths["heatmap"])
    receipt = {
        "status": "completed",
        **audit,
        "metric_unit": "participant-side or participant-side-series; bootstrap intervals use students as independent units",
        "figure_note": "SVG heatmap summarizes participant-level means for the nested procedure; per-series bootstrap intervals are in the CSV and no fatigue ground truth is implied.",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "inputs_sha256": {"predictions": _sha256(predictions_path), "selections": _sha256(selections_path)},
        "outputs": {key: str(path) for key, path in paths.items()},
        "output_sha256": {key: _sha256(path) for key, path in paths.items()},
        "source_sha256": _sha256(Path(__file__)),
    }
    receipt_path = output_dir / "RECIBO_ANALISIS_LATERALIDAD_MODELOS.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(args.predictions, args.selections, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
