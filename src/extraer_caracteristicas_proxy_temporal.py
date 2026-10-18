"""Extract unfiltered stored-channel features for the weak temporal-position proxy.

This script does not filter the signal or interpret the features as physiological
fatigue measures. Its outputs are pseudonymized and must remain in private local
storage while a participant key exists.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


LABEL_ORDER = {"temprana": 0, "transicion": 1, "tardia": 2}
GAP_FLOAT_TOLERANCE_S = 1e-9
KEY_FIELDS = ("participant_id", "recording_id", "leg", "set_index")
PREDICTOR_COLUMNS = (
    "adc_mean",
    "adc_sd_population",
    "adc_rms",
    "adc_mean_absolute",
    "adc_median",
    "adc_median_absolute",
    "adc_p05",
    "adc_p25",
    "adc_p75",
    "adc_p95",
    "adc_peak_to_peak",
    "adc_line_length_mean",
    "adc_median_crossing_rate",
    "adc_skewness_population",
    "adc_excess_kurtosis_population",
)


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        return list(reader), fields


def _key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(str(row[field]).strip() for field in KEY_FIELDS)  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _contiguous_pair_indices(
    times: Sequence[float], sample_indices: Sequence[int], max_gap_s: float = 0.0015
) -> list[tuple[int, int]]:
    """Map compacted selected samples to real adjacent timestamp-valid pairs."""
    if not math.isfinite(max_gap_s) or max_gap_s <= 0.0:
        raise ValueError("maximum contiguous interval must be finite and positive")
    indices = [int(index) for index in sample_indices]
    if any(index < 0 or index >= len(times) for index in indices):
        raise ValueError("selected sample index is outside the timestamp vector")
    pairs: list[tuple[int, int]] = []
    for compact_index, (left, right) in enumerate(zip(indices, indices[1:])):
        delta = float(times[right]) - float(times[left])
        if right == left + 1 and 0.0 < delta <= max_gap_s + GAP_FLOAT_TOLERANCE_S:
            pairs.append((compact_index, compact_index + 1))
    return pairs


def _features(
    values: list[float], contiguous_pair_indices: Sequence[tuple[int, int]]
) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("Each proxy segment must contain at least two signal samples.")
    if not contiguous_pair_indices:
        raise ValueError("Each proxy segment must contain at least one contiguous adjacent sample pair.")
    for left, right in contiguous_pair_indices:
        if left < 0 or right != left + 1 or right >= len(values):
            raise ValueError("contiguous pair indices must identify adjacent compacted samples")
    ordered = sorted(values)
    absolute = sorted(abs(value) for value in values)
    count = len(values)
    mean = sum(values) / count
    centered = [value - mean for value in values]
    variance = sum(value * value for value in centered) / count
    sd = math.sqrt(max(variance, 0.0))
    rms = math.sqrt(sum(value * value for value in values) / count)
    median = statistics.median(ordered)
    median_abs = statistics.median(absolute)
    if sd == 0.0:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        skewness = (sum(value ** 3 for value in centered) / count) / (sd ** 3)
        excess_kurtosis = (sum(value ** 4 for value in centered) / count) / (sd ** 4) - 3.0
    crossing_count = 0
    line_length_sum = 0.0
    for left, right in contiguous_pair_indices:
        previous = values[left] - median
        current = values[right] - median
        line_length_sum += abs(values[right] - values[left])
        if (previous < 0.0 and current > 0.0) or (previous > 0.0 and current < 0.0):
            crossing_count += 1
    return {
        "adc_mean": mean,
        "adc_sd_population": sd,
        "adc_rms": rms,
        "adc_mean_absolute": sum(abs(value) for value in values) / count,
        "adc_median": median,
        "adc_median_absolute": median_abs,
        "adc_p05": _quantile(ordered, 0.05),
        "adc_p25": _quantile(ordered, 0.25),
        "adc_p75": _quantile(ordered, 0.75),
        "adc_p95": _quantile(ordered, 0.95),
        "adc_peak_to_peak": ordered[-1] - ordered[0],
        "adc_line_length_mean": line_length_sum / len(contiguous_pair_indices),
        "adc_median_crossing_rate": crossing_count / len(contiguous_pair_indices),
        "adc_skewness_population": skewness,
        "adc_excess_kurtosis_population": excess_kurtosis,
    }


def _path_under_root(signals_root: Path, relative_path: str) -> Path:
    root = signals_root.resolve()
    relative = Path(*relative_path.replace("\\", "/").split("/"))
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("A manifest path resolves outside the pseudonymized signal root.")
    if not candidate.is_file():
        raise FileNotFoundError("A pseudonymized manifest path does not resolve to a file.")
    return candidate


def extract_proxy_features(
    manifest_path: Path, labels_path: Path, signals_root: Path
) -> tuple[list[dict[str, str]], dict[str, object]]:
    manifest_path = Path(manifest_path).resolve()
    labels_path = Path(labels_path).resolve()
    signals_root = Path(signals_root).resolve()
    manifest, manifest_fields = _read_rows(manifest_path)
    labels, label_fields = _read_rows(labels_path)
    required_manifest = {*KEY_FIELDS, "relative_path"}
    required_labels = {*KEY_FIELDS, "t_start_s", "t_end_s", "n_samples_in_segment", "label", "label_version", "label_confidence"}
    if not required_manifest.issubset(manifest_fields):
        raise ValueError("Pseudonymized signal manifest is missing required columns.")
    if not required_labels.issubset(label_fields):
        raise ValueError("Temporal-proxy labels are missing required columns.")

    eligible_manifest = [row for row in manifest if row["recording_id"] in {"R1", "R2", "R3", "R4", "R5"}]
    manifest_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in eligible_manifest:
        key = _key(row)
        if key in manifest_by_key:
            raise ValueError("Duplicate participant-record-side-set key in the signal manifest.")
        manifest_by_key[key] = row

    labels_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        if row["recording_id"] not in {"R1", "R2", "R3", "R4", "R5"}:
            raise ValueError("The temporal proxy must contain R1-R5 records only.")
        labels_by_key[_key(row)].append(row)
    if not labels_by_key:
        raise ValueError("No R1-R5 temporal-proxy labels were supplied.")
    if set(labels_by_key) != set(manifest_by_key):
        raise ValueError("The label and signal manifests do not cover the same eligible records.")

    output: list[dict[str, str]] = []
    samples_by_label: Counter[str] = Counter()
    signal_hashes: list[str] = []
    participants: set[str] = set()
    for key in sorted(labels_by_key):
        record_labels = sorted(labels_by_key[key], key=lambda row: LABEL_ORDER.get(row["label"], 99))
        if len(record_labels) != 3 or [row["label"] for row in record_labels] != list(LABEL_ORDER):
            raise ValueError("Every eligible record must have exactly one label for each temporal third.")
        if len({row["label_version"] for row in record_labels}) != 1:
            raise ValueError("Temporal-label versions differ within a record.")
        if any(row["label_confidence"] != "low_proxy" for row in record_labels):
            raise ValueError("Unexpected confidence level in temporal-proxy labels.")
        label_bounds = []
        for row in record_labels:
            start = float(row["t_start_s"])
            end = float(row["t_end_s"])
            expected_count = int(row["n_samples_in_segment"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start or expected_count < 2:
                raise ValueError("Invalid temporal boundary or expected sample count in label table.")
            label_bounds.append((start, end, expected_count))
        for index in range(2):
            if not math.isclose(label_bounds[index][1], label_bounds[index + 1][0], rel_tol=0.0, abs_tol=1e-7):
                raise ValueError("Temporal thirds are not contiguous.")

        signal_path = _path_under_root(signals_root, manifest_by_key[key]["relative_path"])
        signal_hashes.append(_sha256(signal_path))
        observed_times: list[float] = []
        observed_values: list[float] = []
        previous_time: float | None = None
        with signal_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required_signal = {"Relative_Time_s", "ADC_Value"}
            if not required_signal.issubset(set(reader.fieldnames or [])):
                raise ValueError("A pseudonymized signal file is missing time or stored-channel columns.")
            for row in reader:
                time_s = float(row["Relative_Time_s"])
                value = float(row["ADC_Value"])
                if not math.isfinite(time_s) or not math.isfinite(value):
                    raise ValueError("Non-finite time or stored-channel value found in an eligible signal file.")
                if previous_time is not None and time_s <= previous_time:
                    raise ValueError("Non-increasing time found in an eligible signal file.")
                previous_time = time_s
                observed_times.append(time_s)
                observed_values.append(value)
        if len(observed_times) < 2 or observed_times[-1] <= observed_times[0]:
            raise ValueError("Insufficient observed time samples in an eligible signal file.")

        # Recreate split cutpoints from the unrounded source timestamps. The CSV
        # stores boundaries to 9 decimals, which can move an exact sample at a
        # cutpoint into an adjacent third if those rounded values are reused.
        record_start = observed_times[0]
        record_end = observed_times[-1]
        duration = record_end - record_start
        exact_bounds = [
            (record_start, record_start + duration / 3.0),
            (record_start + duration / 3.0, record_start + 2.0 * duration / 3.0),
            (record_start + 2.0 * duration / 3.0, record_end),
        ]
        for index, (start, end) in enumerate(exact_bounds):
            if not math.isclose(start, label_bounds[index][0], rel_tol=0.0, abs_tol=1e-7):
                raise ValueError("Label start boundary does not match source-precision temporal thirds.")
            if not math.isclose(end, label_bounds[index][1], rel_tol=0.0, abs_tol=1e-7):
                raise ValueError("Label end boundary does not match source-precision temporal thirds.")

        segment_values: list[list[float]] = [[], [], []]
        segment_sample_indices: list[list[int]] = [[], [], []]
        for sample_index, (time_s, value) in enumerate(zip(observed_times, observed_values)):
            if exact_bounds[0][0] <= time_s < exact_bounds[0][1]:
                segment_values[0].append(value)
                segment_sample_indices[0].append(sample_index)
            elif exact_bounds[1][0] <= time_s < exact_bounds[1][1]:
                segment_values[1].append(value)
                segment_sample_indices[1].append(sample_index)
            elif exact_bounds[2][0] <= time_s <= exact_bounds[2][1]:
                segment_values[2].append(value)
                segment_sample_indices[2].append(sample_index)

        for index, row in enumerate(record_labels):
            values = segment_values[index]
            expected_count = label_bounds[index][2]
            if len(values) != expected_count:
                raise ValueError("Signal sample count does not match the temporal-label manifest.")
            pairs = _contiguous_pair_indices(observed_times, segment_sample_indices[index])
            feature_values = _features(values, pairs)
            output_row = {
                "participant_id": key[0],
                "recording_id": key[1],
                "leg": key[2],
                "set_index": key[3],
                "t_start_s_for_segmentation_only": row["t_start_s"],
                "t_end_s_for_segmentation_only": row["t_end_s"],
                "label": row["label"],
                "label_version": row["label_version"],
                "label_kind": "temporal_position_proxy_not_fatigue_ground_truth",
                "label_confidence": row["label_confidence"],
                "observed_repetition_count": "unknown",
                "active_time_verified": "false",
                "n_samples": str(len(values)),
            }
            output_row.update({name: format(value, ".12g") for name, value in feature_values.items()})
            output.append(output_row)
            samples_by_label[row["label"]] += len(values)
        participants.add(key[0])

    audit: dict[str, object] = {
        "label_version": "proxy_temporal_registro_v1",
        "task_interpretation": "stored-channel features compared for within-recording temporal-position proxy only; not a fatigue, repetition, active-time, or failure-event prediction",
        "feature_source": "ADC_Value only; Voltage_mV is algebraically dependent and excluded",
        "filtering": "none; original stored values preserved; F0 unfiltered baseline",
        "pairwise_feature_adjacency": "line length and median crossings use only original neighboring samples with timestamp interval <= 1.5 ms; acquisition gaps and excluded boundaries are not bridged",
        "signal_mode_and_acquisition_chain": "unknown; features are descriptive digital-channel summaries, not physiological units",
        "segment_rows": len(output),
        "record_count": len(labels_by_key),
        "participant_count": len(participants),
        "sample_count_total": sum(samples_by_label.values()),
        "samples_by_label": {label: samples_by_label[label] for label in LABEL_ORDER},
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "excluded_model_inputs": ["participant_id", "recording_id", "leg", "set_index", "Relative_Time_s", "time thirds/progress", "Voltage_mV", "mechanically verified repetitions", "participant covariates"],
        "participant_covariates_status": "kept in a separate private record-level crosswalk; not model predictors because coverage is incomplete and the values are not tied to the temporal proxy",
        "repetition_counts_verified": 0,
        "R0_R0_1": "QC-only; not assigned proxy labels",
        "manifest_sha256": _sha256(manifest_path),
        "labels_sha256": _sha256(labels_path),
        "signal_files_sha256_combined": hashlib.sha256("\n".join(sorted(signal_hashes)).encode("ascii")).hexdigest(),
        "signal_values_modified": False,
    }
    return output, audit


def write_outputs(rows: list[dict[str, str]], audit: dict[str, object], output_csv: Path, audit_json: Path) -> None:
    output_csv = Path(output_csv).resolve()
    audit_json = Path(audit_json).resolve()
    if output_csv.exists() or audit_json.exists():
        raise FileExistsError("Refusing to overwrite existing private analysis outputs.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audit = dict(audit)
    audit["feature_csv_sha256"] = _sha256(output_csv)
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--signals-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    rows, audit = extract_proxy_features(args.manifest, args.labels, args.signals_root)
    write_outputs(rows, audit, args.output_csv, args.audit_json)
    print(json.dumps({key: audit[key] for key in ("segment_rows", "record_count", "participant_count", "sample_count_total", "samples_by_label", "filtering")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
