"""Create a private, time-supported ordinal sEMG feature table for Cerqueira v2.

The target is self-perceived upper-limb fatigue (0/1/2). This program does not
create functional-fatigue labels and never modifies either source archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from output_safety import PublicCheckoutOutputError, ensure_outside_public_checkout  # noqa: E402
import auditar_cerqueira_emg_label_alignment as alignment  # noqa: E402


FEATURES = ("rms", "mav", "waveform_length", "zero_crossings", "slope_sign_changes", "mdf_hz", "mnf_hz")
CHANNEL_ORDER = ("biceps", "deltoid_anterior", "deltoid_medius", "deltoid_posterior")
_CHANNEL_RE = re.compile(r"^(L|R)\s+(.+?):\s+EMG\s+\d+\s+\[V\]$", re.IGNORECASE)


class FeatureExtractionError(ValueError):
    """Raised when source data or audit provenance fails the extraction contract."""


def design_fir_bandpass(
    sampling_hz: float,
    low_hz: float = 20.0,
    high_hz: float = 400.0,
    taps: int = 251,
) -> np.ndarray:
    """Design a symmetric Hamming-window FIR bandpass and normalize its midband gain."""
    if sampling_hz <= 0 or low_hz <= 0 or high_hz <= low_hz:
        raise FeatureExtractionError("Invalid sampling rate or passband")
    if high_hz >= sampling_hz / 2:
        raise FeatureExtractionError("High cutoff must be below Nyquist")
    if taps < 3 or taps % 2 == 0:
        raise FeatureExtractionError("FIR length must be an odd integer of at least three taps")
    centered = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    h = (2 * high_hz / sampling_hz) * np.sinc(2 * high_hz * centered / sampling_hz)
    h -= (2 * low_hz / sampling_hz) * np.sinc(2 * low_hz * centered / sampling_hz)
    h *= np.hamming(taps)
    midpoint = (low_hz + high_hz) / 2
    response = np.sum(h * np.exp(-2j * np.pi * midpoint * centered / sampling_hz))
    gain = abs(response)
    if not np.isfinite(gain) or gain <= 0:
        raise FeatureExtractionError("FIR normalization failed")
    h /= gain
    return h


def apply_fir_zero_phase(signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply centered symmetric FIR convolution using reflection padding.

    The symmetric impulse response is centered offline, so the linear-phase
    group delay is compensated by cropping. Reflection padding handles edges;
    it does not add samples to the output.
    """
    x = np.asarray(signal, dtype=np.float64)
    h = np.asarray(coefficients, dtype=np.float64)
    if x.ndim != 1 or h.ndim != 1 or len(h) % 2 != 1:
        raise FeatureExtractionError("Signal and odd-length FIR coefficients must be one-dimensional")
    if len(x) <= (len(h) - 1) // 2:
        raise FeatureExtractionError("Signal is too short for the selected FIR edge padding")
    if not np.isfinite(x).all() or not np.isfinite(h).all():
        raise FeatureExtractionError("Non-finite signal or filter coefficient")
    half = (len(h) - 1) // 2
    padded = np.pad(x, (half, half), mode="reflect")
    full_length = len(padded) + len(h) - 1
    nfft = 1 << (full_length - 1).bit_length()
    full = np.fft.irfft(np.fft.rfft(padded, nfft) * np.fft.rfft(h, nfft), nfft)[:full_length]
    same = full[half : half + len(padded)]
    return same[half : half + len(x)]


def assign_window_label(
    label_times: np.ndarray,
    label_values: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    label_rate_hz: float = 50.0,
    min_coverage: float = 0.80,
    min_class_fraction: float = 0.80,
) -> tuple[int | None, float, float]:
    """Return the stable majority ordinal label and observed coverage for one window."""
    times = np.asarray(label_times, dtype=np.float64)
    values = np.asarray(label_values, dtype=np.int8)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        raise FeatureExtractionError("Label time/value arrays are inconsistent")
    if end_s <= start_s or not 0 <= min_coverage <= 1 or not 0 <= min_class_fraction <= 1:
        raise FeatureExtractionError("Invalid window or label thresholds")
    left = int(np.searchsorted(times, start_s, side="left"))
    right = int(np.searchsorted(times, end_s, side="left"))
    local = values[left:right]
    expected = max(1, int(round((end_s - start_s) * label_rate_hz)))
    coverage = min(1.0, len(local) / expected)
    if not len(local) or coverage < min_coverage:
        return None, coverage, 0.0
    counts = np.bincount(local.astype(np.int64), minlength=3)
    fraction = float(counts.max() / len(local))
    if fraction < min_class_fraction:
        return None, coverage, fraction
    return int(np.argmax(counts)), coverage, fraction


def compute_window_features(
    signal: np.ndarray,
    sampling_hz: float,
    *,
    low_hz: float = 20.0,
    high_hz: float = 400.0,
) -> dict[str, float]:
    """Compute time and band-limited spectral features from one processed window."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        raise FeatureExtractionError("A feature window must be finite, one-dimensional, and nontrivial")
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    mav = float(np.mean(np.abs(x)))
    differences = np.diff(x)
    scale = float(np.std(x))
    threshold = 0.01 * scale
    zero_crossings = int(
        np.sum((x[:-1] * x[1:] < 0) & (np.maximum(np.abs(x[:-1]), np.abs(x[1:])) > threshold))
    )
    slope_sign_changes = int(
        np.sum(
            (differences[:-1] * differences[1:] < 0)
            & (np.maximum(np.abs(differences[:-1]), np.abs(differences[1:])) > threshold)
        )
    )
    frequencies = np.fft.rfftfreq(x.size, d=1 / sampling_hz)
    window = np.hanning(x.size)
    spectrum = np.fft.rfft(x * window)
    power = np.abs(spectrum) ** 2
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_frequencies = frequencies[band]
    band_power = power[band]
    total = float(band_power.sum())
    if total <= np.finfo(np.float64).tiny or band_frequencies.size == 0:
        mdf = float("nan")
        mnf = float("nan")
    else:
        cumulative = np.cumsum(band_power)
        median_index = min(int(np.searchsorted(cumulative, total / 2, side="left")), len(band_frequencies) - 1)
        mdf = float(band_frequencies[median_index])
        mnf = float(np.sum(band_frequencies * band_power) / total)
    return {
        "rms": rms,
        "mav": mav,
        "waveform_length": float(np.mean(np.abs(differences))) if differences.size else 0.0,
        "zero_crossings": float(zero_crossings),
        "slope_sign_changes": float(slope_sign_changes),
        "mdf_hz": mdf,
        "mnf_hz": mnf,
    }


def _role_and_side(channel_name: str) -> tuple[str, str]:
    match = _CHANNEL_RE.fullmatch(channel_name.strip())
    if not match:
        raise FeatureExtractionError(f"Unexpected channel name: {channel_name}")
    side, muscle = match.groups()
    muscle = muscle.upper()
    if "BICEPS BRACHII" in muscle:
        role = "biceps"
    elif "DELTOID ANTERIOR" in muscle:
        role = "deltoid_anterior"
    elif "DELTOID MEDIUS" in muscle:
        role = "deltoid_medius"
    elif "DELTOID POSTERIOR" in muscle:
        role = "deltoid_posterior"
    else:
        raise FeatureExtractionError(f"Unrecognized muscle role in channel: {channel_name}")
    return role, side.upper()


def _read_signal_prefix(stream: BinaryIO, expected_rows: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    try:
        reader = csv.reader(text)
        header = next(reader, None)
        if not header:
            raise FeatureExtractionError("Empty EMG trial CSV")
        time_indices = [
            index for index, name in enumerate(header) if alignment._TIME_COLUMN_RE.fullmatch(name.strip())
        ]
        signal_indices = [index for index in range(len(header)) if index not in time_indices]
        if len(time_indices) != len(signal_indices) or len(signal_indices) != 4:
            raise FeatureExtractionError("Expected four paired channel time axes and four EMG channels")
        times = np.empty(expected_rows, dtype=np.float64)
        signals = np.empty((expected_rows, len(signal_indices)), dtype=np.float64)
        previous = None
        for row_index in range(expected_rows):
            row = next(reader, None)
            if row is None or len(row) != len(header):
                raise FeatureExtractionError(f"Unexpected truncated/malformed EMG row at {row_index}")
            try:
                time_values = np.asarray([float(row[index]) for index in time_indices], dtype=np.float64)
                signal_values = np.asarray([float(row[index]) for index in signal_indices], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise FeatureExtractionError(f"Non-numeric EMG row at {row_index}") from error
            if not np.isfinite(time_values).all() or not np.isfinite(signal_values).all():
                raise FeatureExtractionError(f"Non-finite EMG row at {row_index}")
            if np.max(time_values) - np.min(time_values) > 1e-9:
                raise FeatureExtractionError(f"Channel time axes diverge inside valid prefix at row {row_index}")
            if previous is not None and time_values[0] <= previous:
                raise FeatureExtractionError(f"Primary time axis is not increasing at row {row_index}")
            previous = float(time_values[0])
            times[row_index] = time_values[0]
            signals[row_index] = signal_values
        return times, signals, [header[index].strip() for index in signal_indices]
    finally:
        text.close()


def _read_labels(stream: BinaryIO) -> tuple[np.ndarray, np.ndarray]:
    with io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text:
        reader = csv.DictReader(text)
        if reader.fieldnames != ["time", "label"]:
            raise FeatureExtractionError(f"Unexpected label header: {reader.fieldnames}")
        times: list[float] = []
        values: list[int] = []
        for row in reader:
            try:
                time_s = float(row["time"])
                label = int(row["label"])
            except (ValueError, TypeError, KeyError) as error:
                raise FeatureExtractionError("Malformed label row") from error
            if not math.isfinite(time_s) or label not in (0, 1, 2):
                raise FeatureExtractionError("Invalid ordinal label or timestamp")
            times.append(time_s)
            values.append(label)
    time_array = np.asarray(times, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.int8)
    for index in range(1, len(time_array)):
        if time_array[index] <= time_array[index - 1]:
            # Keep only the monotonic prefix; terminal anomalies are never targets.
            time_array = time_array[:index]
            value_array = value_array[:index]
            break
    return time_array, value_array


def _feature_columns() -> list[str]:
    return [f"{role}_{feature}" for role in CHANNEL_ORDER for feature in FEATURES]


def compute_unlabeled_trial_features(
    times: np.ndarray,
    raw_signals: np.ndarray,
    channel_names: list[str],
    sampling_hz: float,
    *,
    window_seconds: float = 1.0,
    low_hz: float = 20.0,
    high_hz: float = 400.0,
    fir_taps: int = 251,
) -> list[dict[str, object]]:
    """Extract the frozen feature contract from a complete unlabeled trial.

    Uses the same FIR and window definitions as training, but does not inspect
    labels or discard windows according to label stability. Only complete,
    non-overlapping windows are emitted; an incomplete trailing window is
    omitted and should be recorded by the calling inference receipt.
    """
    t = np.asarray(times, dtype=np.float64)
    raw = np.asarray(raw_signals, dtype=np.float64)
    if t.ndim != 1 or raw.ndim != 2 or raw.shape[0] != t.size or raw.shape[1] != len(channel_names):
        raise FeatureExtractionError("Times, channel matrix, and channel names have inconsistent shapes")
    if raw.shape[1] != 4 or t.size == 0 or not np.isfinite(t).all() or not np.isfinite(raw).all():
        raise FeatureExtractionError("A complete trial requires finite timestamps and exactly four finite channels")
    if t.size > 1 and np.any(np.diff(t) <= 0):
        raise FeatureExtractionError("Trial timestamps must be strictly increasing")
    if sampling_hz <= 0 or window_seconds <= 0:
        raise FeatureExtractionError("Sampling rate and window duration must be positive")
    role_map: dict[str, int] = {}
    sides: set[str] = set()
    for channel_index, channel_name in enumerate(channel_names):
        role, side = _role_and_side(channel_name)
        if role in role_map:
            raise FeatureExtractionError(f"Duplicate muscle role {role}")
        role_map[role] = channel_index
        sides.add(side)
    if set(role_map) != set(CHANNEL_ORDER) or len(sides) != 1:
        raise FeatureExtractionError("Expected the four recognized muscles from one side")
    coefficients = design_fir_bandpass(sampling_hz, low_hz, high_hz, fir_taps)
    filtered = {
        role: apply_fir_zero_phase(raw[:, role_map[role]], coefficients)
        for role in CHANNEL_ORDER
    }
    window_samples = int(round(sampling_hz * window_seconds))
    if window_samples < 8:
        raise FeatureExtractionError("Window is too short for spectral feature extraction")
    rows: list[dict[str, object]] = []
    for window_index, start in enumerate(range(0, t.size - window_samples + 1, window_samples)):
        stop = start + window_samples
        row: dict[str, object] = {
            "window_index": window_index,
            "window_start_s": float(t[start]),
            "window_end_s": float(t[stop - 1] + 1.0 / sampling_hz),
            "side": next(iter(sides)),
            "sample_count": window_samples,
        }
        flat_roles: list[str] = []
        spectral_valid_channels = 0
        for role in CHANNEL_ORDER:
            raw_window = raw[start:stop, role_map[role]]
            centered_raw = raw_window - raw_window.mean()
            processed_window = filtered[role][start:stop]
            if np.ptp(processed_window) <= np.finfo(float).eps:
                flat_roles.append(role)
            features = compute_window_features(processed_window, sampling_hz, low_hz=low_hz, high_hz=high_hz)
            if math.isfinite(features["mdf_hz"]) and math.isfinite(features["mnf_hz"]):
                spectral_valid_channels += 1
            row[f"{role}_raw_rms"] = float(np.sqrt(np.mean(centered_raw * centered_raw)))
            for feature, value in features.items():
                row[f"{role}_{feature}"] = value
        row["flatline_channel_count"] = len(flat_roles)
        row["flatline_channel_roles"] = ";".join(flat_roles)
        row["spectral_valid_channel_count"] = spectral_valid_channels
        rows.append(row)
    return rows


def extract_feature_table(
    emg_archive_path: str | Path,
    label_archive_path: str | Path,
    audit_json_path: str | Path,
    *,
    trial_coverage_threshold: float = 0.90,
    window_seconds: float = 1.0,
    min_label_coverage: float = 0.80,
    min_class_fraction: float = 0.80,
    low_hz: float = 20.0,
    high_hz: float = 400.0,
    fir_taps: int = 251,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build a participant-grouped private feature table from verified archive intersections."""
    if not 0 < trial_coverage_threshold <= 1:
        raise FeatureExtractionError("Trial coverage threshold must be in (0, 1]")
    emg_path, labels_path, receipt_path = map(Path, (emg_archive_path, label_archive_path, audit_json_path))
    if not all(path.is_file() for path in (emg_path, labels_path, receipt_path)):
        raise FeatureExtractionError("EMG archive, label archive, and alignment receipt are required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = receipt.get("summary", {})
    emg_md5 = alignment._md5(emg_path)
    label_md5 = alignment._md5(labels_path)
    if emg_md5 != summary.get("emg_archive_md5") or label_md5 != summary.get("label_archive_md5"):
        raise FeatureExtractionError("Source archive checksum differs from the alignment receipt")
    if not summary.get("pairing_complete"):
        raise FeatureExtractionError("Alignment receipt reports missing subject/trial pairs")

    trial_metadata = {
        (int(row["subject_index"]), int(row["trial_index"])): row for row in receipt["trials"]
    }
    selected = {
        key: row for key, row in trial_metadata.items()
        if float(row["label_time_support_fraction"]) >= trial_coverage_threshold
    }
    if not selected:
        raise FeatureExtractionError("No trials pass the declared label-support threshold")

    emg_entries = alignment._outer_subject_entries(emg_path)
    label_entries = alignment._outer_subject_entries(labels_path)
    output_rows: list[dict[str, object]] = []
    excluded = Counter()
    accepted_trials: set[tuple[int, int]] = set()
    class_counts = Counter({0: 0, 1: 0, 2: 0})
    channel_flat_windows = Counter()
    signal_min = float("inf")
    signal_max = float("-inf")
    trial_count_by_window_samples: Counter[int] = Counter()
    fs_values: list[float] = []
    fir_coefficients: dict[float, np.ndarray] = {}

    with zipfile.ZipFile(emg_path) as emg_outer, zipfile.ZipFile(labels_path) as label_outer:
        for subject_id in sorted({key[0] for key in selected}):
            if subject_id not in emg_entries or subject_id not in label_entries:
                raise FeatureExtractionError(f"Selected subject {subject_id} is missing from an archive")
            raw_nested = zipfile.ZipFile(io.BytesIO(emg_outer.read(emg_entries[subject_id])))
            label_nested = zipfile.ZipFile(io.BytesIO(label_outer.read(label_entries[subject_id])))
            try:
                raw_trials = alignment._nested_trial_entries(raw_nested, subject_id)
                rating_trials = alignment._nested_trial_entries(label_nested, subject_id)
                for (trial_subject, trial_id), metadata in sorted(selected.items()):
                    if trial_subject != subject_id:
                        continue
                    if trial_id not in raw_trials or trial_id not in rating_trials:
                        raise FeatureExtractionError(f"Selected trial {subject_id}/{trial_id} is missing")
                    valid_samples = int(metadata["signal_monotonic_prefix_sample_count"])
                    mismatch_details = metadata.get("signal_channel_time_mismatch_details", [])
                    if mismatch_details:
                        valid_samples = min(valid_samples, min(int(item["row_index"]) for item in mismatch_details))
                    if valid_samples <= 0:
                        excluded["no_valid_monotonic_samples"] += 1
                        continue
                    with raw_nested.open(raw_trials[trial_id]) as stream:
                        times, raw_signals, channel_names = _read_signal_prefix(stream, valid_samples)
                    with label_nested.open(rating_trials[trial_id]) as stream:
                        label_times, label_values = _read_labels(stream)

                    role_map: dict[str, int] = {}
                    sides: set[str] = set()
                    for channel_index, channel_name in enumerate(channel_names):
                        role, side = _role_and_side(channel_name)
                        if role in role_map:
                            raise FeatureExtractionError(f"Duplicate muscle role {role} in trial {subject_id}/{trial_id}")
                        role_map[role] = channel_index
                        sides.add(side)
                    if set(role_map) != set(CHANNEL_ORDER) or len(sides) != 1:
                        raise FeatureExtractionError(f"Unexpected channel set or side in trial {subject_id}/{trial_id}")

                    fs = float(metadata["signal_sampling_hz_median"])
                    fs_key = round(fs, 6)
                    fs_values.append(fs)
                    window_samples = int(round(fs * window_seconds))
                    trial_count_by_window_samples[window_samples] += 1
                    if fs_key not in fir_coefficients:
                        fir_coefficients[fs_key] = design_fir_bandpass(fs, low_hz, high_hz, fir_taps)
                    coefficients = fir_coefficients[fs_key]
                    for channel in range(raw_signals.shape[1]):
                        signal_min = min(signal_min, float(raw_signals[:, channel].min()))
                        signal_max = max(signal_max, float(raw_signals[:, channel].max()))
                    filtered_by_role: dict[str, np.ndarray] = {}
                    for role in CHANNEL_ORDER:
                        raw_channel = raw_signals[:, role_map[role]]
                        filtered = apply_fir_zero_phase(raw_channel, coefficients)
                        filtered_by_role[role] = filtered

                    hop_samples = window_samples  # non-overlapping 1 s windows
                    n_windows = 0
                    for start in range(0, valid_samples - window_samples + 1, hop_samples):
                        stop = start + window_samples
                        start_s = float(times[start])
                        end_s = float(times[stop - 1] + 1.0 / fs)
                        target, label_coverage, stable_fraction = assign_window_label(
                            label_times,
                            label_values,
                            start_s,
                            end_s,
                            min_coverage=min_label_coverage,
                            min_class_fraction=min_class_fraction,
                        )
                        if target is None:
                            excluded["unsupported_or_transition_window"] += 1
                            continue
                        row: dict[str, object] = {
                            "subject_index": subject_id,
                            "trial_index": trial_id,
                            "side": next(iter(sides)),
                            "window_index": start // hop_samples,
                            "window_start_s": start_s,
                            "window_end_s": end_s,
                            "label": target,
                            "label_coverage": label_coverage,
                            "label_stability": stable_fraction,
                            "sample_count": window_samples,
                        }
                        flat_roles: list[str] = []
                        spectral_valid_channels = 0
                        for role in CHANNEL_ORDER:
                            raw_channel_window = raw_signals[start:stop, role_map[role]]
                            raw_centered = raw_channel_window - raw_channel_window.mean()
                            raw_rms = float(np.sqrt(np.mean(raw_centered**2)))
                            processed_window = filtered_by_role[role][start:stop]
                            if np.ptp(processed_window) <= np.finfo(float).eps:
                                channel_flat_windows[role] += 1
                                flat_roles.append(role)
                            features = compute_window_features(processed_window, fs, low_hz=low_hz, high_hz=high_hz)
                            if math.isfinite(features["mdf_hz"]) and math.isfinite(features["mnf_hz"]):
                                spectral_valid_channels += 1
                            row[f"{role}_raw_rms"] = raw_rms
                            for feature, value in features.items():
                                row[f"{role}_{feature}"] = value
                        row["flatline_channel_count"] = len(flat_roles)
                        row["flatline_channel_roles"] = ";".join(flat_roles)
                        row["spectral_valid_channel_count"] = spectral_valid_channels
                        output_rows.append(row)
                        class_counts[target] += 1
                        n_windows += 1
                    if n_windows:
                        accepted_trials.add((subject_id, trial_id))
                    else:
                        excluded["eligible_trial_without_stable_window"] += 1
            finally:
                raw_nested.close()
                label_nested.close()

    if not output_rows:
        raise FeatureExtractionError("No stable labeled windows were produced")
    output_rows.sort(key=lambda row: (row["subject_index"], row["trial_index"], row["window_index"]))
    summary_out = {
        "schema_version": 1,
        "task": "self-perceived upper-limb fatigue ordinal classification (0/1/2)",
        "source_emg_md5": emg_md5,
        "source_labels_md5": label_md5,
        "alignment_receipt": receipt_path.name,
        "trial_coverage_threshold": trial_coverage_threshold,
        "window_seconds": window_seconds,
        "window_overlap_fraction": 0.0,
        "trial_count_by_window_samples": {str(k): v for k, v in trial_count_by_window_samples.items()},
        "label_rate_hz_assumed_from_verified_median": 50.0,
        "minimum_label_coverage_per_window": min_label_coverage,
        "minimum_majority_class_fraction": min_class_fraction,
        "label_mapping": "majority ordinal label within each window; transition/unsupported windows omitted",
        "feature_columns": _feature_columns(),
        "raw_qc_columns": [f"{role}_raw_rms" for role in CHANNEL_ORDER],
        "time_features": list(FEATURES[:5]),
        "spectral_features": list(FEATURES[5:]),
        "frequency_band_hz": [low_hz, high_hz],
        "fir_design": {
            "type": "symmetric Hamming-window FIR bandpass",
            "taps": fir_taps,
            "sampling_hz_median": float(np.median(fs_values)),
            "zero_phase_offline": True,
            "group_delay_samples_compensated": (fir_taps - 1) // 2,
            "edge_padding": "reflection",
            "coefficient_formula": "ideal sinc bandpass multiplied by Hamming window; normalized to unit gain at passband midpoint",
            "coefficients_by_fs": {str(fs): coeff.tolist() for fs, coeff in fir_coefficients.items()},
            "notch_applied": False,
            "notch_reason": "no demonstrated need or reference-clean signal in this extraction stage",
        },
        "raw_observed_value_range_v": [signal_min, signal_max],
        "saturation_assessment": "not determined; ADC rails and analog gain are not documented",
        "matched_subject_count": len({int(row["subject_index"]) for row in output_rows}),
        "eligible_trial_count_by_coverage": len(selected),
        "accepted_trial_count_with_stable_windows": len(accepted_trials),
        "window_count": len(output_rows),
        "class_counts": {str(k): class_counts[k] for k in (0, 1, 2)},
        "channel_flatline_windows_by_role": dict(channel_flat_windows),
        "spectral_nan_policy": "preserve RMS/time features and retain spectral NaN with a validity count; do not exclude low-amplitude windows automatically",
        "excluded_counts": dict(excluded),
        "limitations": [
            "Timestamp overlap is not independent proof of hardware synchronization.",
            "The labels measure self-perceived upper-limb fatigue, not functional force loss or quadriceps fatigue.",
            "Within-trial motion onset/offset and repetition markers are absent.",
            "The source's analog gain, ADC rails, and physical calibration are not documented.",
        ],
    }
    return output_rows, summary_out


def write_feature_outputs(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    feature_csv: str | Path,
    summary_json: str | Path,
    *,
    protected_paths: Iterable[str | Path] = (),
) -> None:
    csv_path, json_path = Path(feature_csv).expanduser(), Path(summary_json).expanduser()
    if csv_path.resolve() == json_path.resolve():
        raise FeatureExtractionError("Feature CSV and summary JSON outputs must be different files")
    try:
        ensure_outside_public_checkout((csv_path, json_path))
    except PublicCheckoutOutputError as error:
        raise FeatureExtractionError(str(error)) from error
    protected = {Path(path).expanduser().resolve() for path in protected_paths}
    if {csv_path.resolve(), json_path.resolve()} & protected:
        raise FeatureExtractionError("feature outputs must not overwrite an input archive or alignment receipt")
    if csv_path.exists() or json_path.exists():
        raise FeatureExtractionError("Refusing to overwrite existing feature outputs")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_json = dict(summary)
    summary_json["feature_csv"] = csv_path.name
    summary_json["feature_csv_rows"] = len(rows)
    summary_json["feature_csv_bytes"] = csv_path.stat().st_size
    summary_json["feature_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emg_archive", type=Path)
    parser.add_argument("label_archive", type=Path)
    parser.add_argument("alignment_receipt", type=Path)
    parser.add_argument("--trial-coverage-threshold", type=float, default=0.90)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-label-coverage", type=float, default=0.80)
    parser.add_argument("--minimum-class-fraction", type=float, default=0.80)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    protected_paths = [args.emg_archive, args.label_archive, args.alignment_receipt]
    if args.output_csv.resolve() == args.summary_json.resolve():
        parser.error("--output-csv and --summary-json must resolve to different files")
    if {args.output_csv.resolve(), args.summary_json.resolve()} & {
        path.expanduser().resolve() for path in protected_paths
    }:
        parser.error("feature outputs must not overwrite an input archive or alignment receipt")
    if args.output_csv.exists() or args.summary_json.exists():
        parser.error("refusing to overwrite existing feature outputs")
    try:
        ensure_outside_public_checkout((args.output_csv, args.summary_json))
    except PublicCheckoutOutputError as error:
        parser.error(str(error))
    protected_paths = [args.emg_archive, args.label_archive, args.alignment_receipt]
    resolved_protected = {path.expanduser().resolve() for path in protected_paths}
    resolved_outputs = {args.output_csv.expanduser().resolve(), args.summary_json.expanduser().resolve()}
    if len(resolved_outputs) != 2:
        parser.error("--output-csv and --summary-json must resolve to different files")
    if resolved_outputs & resolved_protected:
        parser.error("feature outputs must not overwrite an input archive or alignment receipt")
    if args.output_csv.exists() or args.summary_json.exists():
        parser.error("refusing to overwrite existing feature outputs")
    rows, summary = extract_feature_table(
        args.emg_archive,
        args.label_archive,
        args.alignment_receipt,
        trial_coverage_threshold=args.trial_coverage_threshold,
        window_seconds=args.window_seconds,
        min_label_coverage=args.minimum_label_coverage,
        min_class_fraction=args.minimum_class_fraction,
    )
    write_feature_outputs(rows, summary, args.output_csv, args.summary_json, protected_paths=protected_paths)
    print(json.dumps({k: summary[k] for k in ("matched_subject_count", "eligible_trial_count_by_coverage", "accepted_trial_count_with_stable_windows", "window_count", "class_counts", "excluded_counts")}, indent=2))


if __name__ == "__main__":
    main()
