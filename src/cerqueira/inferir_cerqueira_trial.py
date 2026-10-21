"""Run unlabeled full-trial or contiguous-fragment inference for Cerqueira sEMG.

Scores predict self-perceived upper-limb fatigue labels 0/1/2. They are not a
functional fatigue or quadriceps-fatigue detector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from output_safety import PublicCheckoutOutputError, ensure_outside_public_checkout  # noqa: E402
import auditar_cerqueira_emg_label_alignment as alignment  # noqa: E402
import entrenar_cerqueira_loso as model_tools  # noqa: E402
import extraer_caracteristicas_cerqueira as features  # noqa: E402


class InferenceError(ValueError):
    """Raised when raw trial input is incompatible with the frozen contract."""


def read_signal_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read one signal CSV with paired, synchronized channel timestamps."""
    source = Path(path)
    try:
        file_size = source.stat().st_size
    except OSError as error:
        raise InferenceError("Cannot inspect signal CSV size") from error
    if file_size > alignment.MAX_ARCHIVE_MEMBER_BYTES:
        raise InferenceError("Signal CSV exceeds configured size limit")
    times: list[float] = []
    values: list[list[float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise InferenceError(f"Empty signal CSV: {source.name}")
        time_indices = [index for index, name in enumerate(header) if alignment._TIME_COLUMN_RE.fullmatch(name.strip())]
        signal_indices = [index for index in range(len(header)) if index not in time_indices]
        if len(time_indices) != 4 or len(signal_indices) != 4:
            raise InferenceError("Expected four timestamp columns and four EMG channel columns")
        previous: float | None = None
        for line_number, row in enumerate(reader, start=2):
            if line_number - 2 >= alignment.MAX_SIGNAL_ROWS:
                raise InferenceError("Signal CSV exceeds configured sample-row limit")
            if len(row) != len(header):
                raise InferenceError(f"Malformed row at line {line_number}")
            try:
                channel_times = np.asarray([float(row[index]) for index in time_indices], dtype=np.float64)
                signal_values = [float(row[index]) for index in signal_indices]
            except (TypeError, ValueError) as error:
                raise InferenceError(f"Non-numeric sample at line {line_number}") from error
            if not np.isfinite(channel_times).all() or not np.isfinite(signal_values).all():
                raise InferenceError(f"Non-finite sample at line {line_number}")
            if float(np.max(channel_times) - np.min(channel_times)) > 1e-9:
                raise InferenceError(f"Channel timestamps diverge at line {line_number}")
            sample_time = float(channel_times[0])
            if previous is not None and sample_time <= previous:
                raise InferenceError(f"Timestamp reset or duplicate at line {line_number}")
            previous = sample_time
            times.append(sample_time)
            values.append(signal_values)
    if len(times) < 10:
        raise InferenceError("Signal CSV has too few samples")
    return np.asarray(times), np.asarray(values), [header[index].strip() for index in signal_indices]


def assemble_fragments(
    fragments: list[tuple[np.ndarray, np.ndarray, list[str]]],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Order and join fragments only when their sample axes are contiguous."""
    if not fragments:
        raise InferenceError("At least one signal fragment is required")
    normalized = []
    expected_names: list[str] | None = None
    for times, signals, names in fragments:
        t = np.asarray(times, dtype=np.float64)
        x = np.asarray(signals, dtype=np.float64)
        current_names = list(names)
        if t.ndim != 1 or x.ndim != 2 or x.shape != (len(t), 4) or len(current_names) != 4:
            raise InferenceError("Fragment timestamp, channel matrix, or header shape is invalid")
        if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(np.diff(t) <= 0):
            raise InferenceError("Fragments must contain finite, increasing timestamps and signal values")
        if expected_names is None:
            expected_names = current_names
        elif current_names != expected_names:
            raise InferenceError("All fragments must use identical channel names and order")
        normalized.append((t, x, current_names))
    normalized.sort(key=lambda item: float(item[0][0]))
    all_diffs = np.concatenate([np.diff(item[0]) for item in normalized])
    nominal_dt = float(np.median(all_diffs))
    if not math.isfinite(nominal_dt) or nominal_dt <= 0:
        raise InferenceError("Could not infer a positive sample interval")
    tolerance = max(1e-9, nominal_dt * 0.01)
    for previous, current in zip(normalized[:-1], normalized[1:]):
        gap = float(current[0][0] - previous[0][-1])
        if abs(gap - nominal_dt) > tolerance:
            raise InferenceError("Fragments overlap or contain a gap; refusing to filter across a discontinuity")
    times = np.concatenate([item[0] for item in normalized])
    signals = np.concatenate([item[1] for item in normalized], axis=0)
    metadata = {"fragment_count": len(normalized), "sample_count": int(len(times)), "nominal_sampling_hz": 1.0 / nominal_dt}
    return times, signals, expected_names or [], metadata


def infer_signal_arrays(
    times: np.ndarray,
    raw_signals: np.ndarray,
    channel_names: list[str],
    sampling_hz: float,
    model_bundle: dict[str, Any],
) -> pd.DataFrame:
    """Generate all complete unlabeled windows and score them with a frozen model."""
    contract = model_bundle.get("feature_extraction", {})
    training_fs = contract.get("sampling_hz_median")
    fs_tolerance = float(contract.get("sampling_rate_relative_tolerance", 0.005))
    if training_fs is not None:
        relative_error = abs(float(sampling_hz) - float(training_fs)) / float(training_fs)
        if relative_error > fs_tolerance:
            raise InferenceError(
                f"Sampling frequency differs from the frozen training contract by {relative_error:.3%}"
            )
    band = contract.get("frequency_band_hz", [20.0, 400.0])
    window_seconds = float(contract.get("window_seconds", 1.0))
    fir_taps = int(contract.get("fir_taps", 251))
    rows = features.compute_unlabeled_trial_features(
        times, raw_signals, channel_names, sampling_hz,
        window_seconds=window_seconds, low_hz=float(band[0]), high_hz=float(band[1]), fir_taps=fir_taps,
    )
    if not rows:
        raise InferenceError("No complete one-second inference windows are available")
    frame = pd.DataFrame(rows)
    columns = list(model_bundle["feature_columns"])
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise InferenceError(f"Raw signal features do not satisfy the frozen model contract: {missing}")
    probabilities = model_tools.predict_model_bundle(model_bundle, frame[columns].to_numpy(dtype=np.float64))
    frame["predicted_label"] = np.argmax(probabilities, axis=1)
    for label in range(3):
        frame[f"p{label}"] = probabilities[:, label]
    return frame


def run_inference(
    input_csvs: list[str | Path],
    model_path: str | Path,
    fold_manifest_path: str | Path | None,
    output_csv: str | Path,
    summary_json: str | Path,
    *,
    final_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if not input_csvs:
        raise InferenceError("At least one input CSV is required")
    if (fold_manifest_path is None) == (final_manifest_path is None):
        raise InferenceError("Provide exactly one fold manifest or final-model manifest")
    out_csv, out_json = Path(output_csv).expanduser(), Path(summary_json).expanduser()
    try:
        ensure_outside_public_checkout((out_csv, out_json))
    except PublicCheckoutOutputError as error:
        raise InferenceError(str(error)) from error
    output_resolved = {out_csv.resolve(), out_json.resolve()}
    if len(output_resolved) != 2:
        raise InferenceError("Prediction CSV and summary JSON outputs must be different files")
    manifest_path = final_manifest_path if final_manifest_path is not None else fold_manifest_path
    assert manifest_path is not None
    protected_resolved = {
        Path(value).expanduser().resolve()
        for value in [*input_csvs, model_path, manifest_path]
    }
    if output_resolved & protected_resolved:
        raise InferenceError("Inference outputs must not overwrite an input, model, or fold manifest")
    if out_csv.exists() or out_json.exists():
        raise InferenceError("Refusing to overwrite existing inference outputs")
    if final_manifest_path is not None:
        bundle = model_tools.load_final_model_bundle(model_path, final_manifest_path)
    else:
        assert fold_manifest_path is not None
        bundle = model_tools.load_fold_model_bundle(model_path, fold_manifest_path)
    fragments = []
    source_records = []
    for path_value in input_csvs:
        path = Path(path_value)
        times, signals, names = read_signal_csv(path)
        fragments.append((times, signals, names))
        source_records.append({"name": path.name, "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    times, signals, names, fragment_metadata = assemble_fragments(fragments)
    dt = np.diff(times)
    median_dt = float(np.median(dt))
    if float(np.max(dt)) > 1.5 * median_dt:
        raise InferenceError("Input contains a sampling gap exceeding 1.5× median interval")
    sampling_hz = 1.0 / median_dt
    result = infer_signal_arrays(times, signals, names, sampling_hz, bundle)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    summary = {
        "task": "self-perceived upper-limb fatigue ordinal inference (0/1/2)",
        "model_file": Path(model_path).name,
        "model_sha256": bundle["manifest_entry"]["model_sha256"],
        "model_family": bundle["family"],
        "model_artifact_type": bundle.get("artifact_type", "loso_fold"),
        "model_feature_columns": bundle["feature_columns"],
        "input_files": source_records,
        "fragment_assembly": fragment_metadata,
        "sampling_hz_median": sampling_hz,
        "samples_processed": int(len(times)),
        "duration_s": float(times[-1] - times[0] + median_dt),
        "window_seconds": float(bundle["feature_extraction"].get("window_seconds", 1.0)),
        "window_overlap_fraction": float(bundle["feature_extraction"].get("window_overlap_fraction", 0.0)),
        "complete_windows_scored": int(len(result)),
        "trailing_samples_not_in_complete_window": int(
            len(times) % round(sampling_hz * float(bundle["feature_extraction"].get("window_seconds", 1.0)))
        ),
        "predicted_label_counts": {str(k): int(v) for k, v in result.predicted_label.value_counts().sort_index().items()},
        "output_csv": out_csv.name,
        "output_csv_sha256": hashlib.sha256(out_csv.read_bytes()).hexdigest(),
        "limitations": [
            "Scores estimate source self-reported upper-limb fatigue labels only.",
            "They are not a validated detector of force loss, functional fatigue, or quadriceps fatigue.",
            "The time-only comparator is diagnostic; this inference uses EMG features only.",
            "Complete contiguous recordings are required; gaps/overlaps are rejected rather than bridged.",
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csvs", nargs="+", type=Path, help="one complete CSV or ordered contiguous fragments")
    parser.add_argument("--model", required=True, type=Path)
    manifest_group = parser.add_mutually_exclusive_group(required=True)
    manifest_group.add_argument("--fold-manifest", type=Path)
    manifest_group.add_argument("--final-manifest", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()
    summary = run_inference(
        args.input_csvs,
        args.model,
        args.fold_manifest,
        args.output_csv,
        args.summary_json,
        final_manifest_path=args.final_manifest,
    )
    print(json.dumps({
        "status": "complete",
        "model_family": summary["model_family"],
        "model_artifact_type": summary["model_artifact_type"],
        "samples_processed": summary["samples_processed"],
        "complete_windows_scored": summary["complete_windows_scored"],
        "predicted_label_counts": summary["predicted_label_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
