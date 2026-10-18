"""Read-only 1 Hz / power-line screen and HP10 sensitivity extension.

The stored channel's RAW/RECT/ENV mode is unknown. Results are therefore
descriptive properties of the saved digital channel, not proof of artifact
origin, signal cleanliness, or physiological fatigue.
"""

from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np


GAP_THRESHOLD_S = 0.0015
MIN_LOW_FREQUENCY_SECONDS = 8.0
MAX_INTERVAL_SPREAD_RATIO = 0.05
LOW_BAND = (0.5, 1.5)
MOTION_SCREEN_BAND = (0.5, 10.0)
MID_BAND = (20.0, 450.0)
BRANCHES = ("control", "hp10", "hp20")
PRIVATE_FIELDS = (
    "participant_id", "recording_id", "record_type", "leg", "set_index",
    "filter_branch", "samples_common", "rms_adc", "power_near_1hz_adc2",
    "peak_near_1hz_hz", "peak_near_1hz_amplitude_adc", "power_0p5_10hz_adc2", "power_20_450hz_adc2",
    "mean_frequency_20_450hz", "median_frequency_20_450hz",
    "change_mean_frequency_hz_vs_control", "change_median_frequency_hz_vs_control",
    "change_mean_frequency_pct_vs_control", "change_median_frequency_pct_vs_control",
    "low_to_mid_power_ratio", "prominence_50_db", "prominence_60_db",
    "prominence_100_db", "prominence_120_db", "change_rms_pct_vs_control",
    "change_0p5_10hz_pct_vs_control", "change_20_450hz_pct_vs_control",
    "spectral_runs_analyzed", "spectral_runs_insufficient_or_irregular",
    "best_frequency_resolution_hz", "median_fs_estimated_hz",
    "median_interval_spread_pct",
    "stored_power_near_1hz_adc2", "stored_peak_near_1hz_hz",
    "stored_peak_near_1hz_amplitude_adc", "stored_power_0p5_10hz_adc2",
    "stored_power_20_450hz_adc2", "stored_prominence_50_db",
    "stored_prominence_60_db", "stored_prominence_100_db",
    "stored_prominence_120_db",
)
PUBLIC_FIELDS = (
    "record_type", "leg", "set_index", "filter_branch", "n_participants",
    "n_recordings", "median_rms_adc", "median_power_near_1hz_adc2",
    "median_peak_near_1hz_hz", "median_peak_near_1hz_amplitude_adc", "median_power_0p5_10hz_adc2",
    "median_power_20_450hz_adc2", "median_mean_frequency_20_450hz",
    "median_median_frequency_20_450hz", "median_change_mean_frequency_hz_vs_control",
    "median_change_median_frequency_hz_vs_control", "median_change_mean_frequency_pct_vs_control",
    "median_change_median_frequency_pct_vs_control", "median_low_to_mid_power_ratio",
    "median_prominence_50_db", "median_prominence_60_db",
    "median_prominence_100_db", "median_prominence_120_db",
    "median_change_rms_pct_vs_control",
    "median_change_0p5_10hz_pct_vs_control",
    "median_change_20_450hz_pct_vs_control",
    "median_stored_power_near_1hz_adc2", "median_stored_peak_near_1hz_hz",
    "median_stored_peak_near_1hz_amplitude_adc", "median_stored_power_0p5_10hz_adc2",
    "median_stored_power_20_450hz_adc2", "median_stored_prominence_50_db",
    "median_stored_prominence_60_db", "median_stored_prominence_100_db",
    "median_stored_prominence_120_db",
)
NUMERIC_FIELDS = (
    "rms_adc", "power_near_1hz_adc2", "peak_near_1hz_hz", "peak_near_1hz_amplitude_adc",
    "power_0p5_10hz_adc2", "power_20_450hz_adc2", "mean_frequency_20_450hz",
    "median_frequency_20_450hz", "change_mean_frequency_hz_vs_control",
    "change_median_frequency_hz_vs_control", "change_mean_frequency_pct_vs_control",
    "change_median_frequency_pct_vs_control", "low_to_mid_power_ratio",
    "prominence_50_db", "prominence_60_db", "prominence_100_db",
    "prominence_120_db", "change_rms_pct_vs_control",
    "change_0p5_10hz_pct_vs_control", "change_20_450hz_pct_vs_control",
    "stored_power_near_1hz_adc2", "stored_peak_near_1hz_hz",
    "stored_peak_near_1hz_amplitude_adc", "stored_power_0p5_10hz_adc2",
    "stored_power_20_450hz_adc2", "stored_prominence_50_db",
    "stored_prominence_60_db", "stored_prominence_100_db",
    "stored_prominence_120_db",
)


def _load_filter_engine():
    path = Path(__file__).with_name("comparar_filtros_emg.py")
    spec = importlib.util.spec_from_file_location("comparar_filtros_emg", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("existing filter engine could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER_ENGINE = _load_filter_engine()


def periodogram_psd(signal, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a Hann-windowed, one-sided density-scaled periodogram."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        raise ValueError("signal must be a finite one-dimensional array of at least 8 samples")
    if not math.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    window = np.hanning(x.size)
    window_power = float(np.sum(window**2))
    if window_power <= 0:
        raise ValueError("periodogram window has no positive power")
    spectrum = np.fft.rfft((x - float(np.mean(x))) * window)
    density = (np.abs(spectrum) ** 2) / (fs_hz * window_power)
    if x.size % 2 == 0:
        density[1:-1] *= 2.0
    else:
        density[1:] *= 2.0
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    return frequencies, density


def integrate_bandpower(
    frequencies, psd, low_hz: float, high_hz: float
) -> float:
    """Integrate PSD density over [low_hz, high_hz), without extrapolation."""
    f = np.asarray(frequencies, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    if f.ndim != 1 or p.ndim != 1 or f.size != p.size or f.size < 2:
        raise ValueError("frequencies and psd must be matching one-dimensional arrays")
    if not np.isfinite(f).all() or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("frequencies and psd must be finite and psd nonnegative")
    if not math.isfinite(low_hz) or not math.isfinite(high_hz) or low_hz < 0 or high_hz <= low_hz:
        raise ValueError("band edges must be finite, nonnegative, and increasing")
    steps = np.diff(f)
    df = float(np.median(steps))
    if df <= 0 or not np.allclose(steps, df, rtol=1e-6, atol=1e-12):
        raise ValueError("frequencies must be strictly increasing and evenly spaced")
    selected = (f >= low_hz) & (f < high_hz)
    if np.count_nonzero(selected) < 2:
        raise ValueError("requested band has fewer than two resolvable bins")
    return float(np.sum(p[selected]) * df)


def highpass_contiguous(
    times_s, signal, cutoff_hz: float, gap_s: float = GAP_THRESHOLD_S
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Apply zero-phase Butterworth HP independently within timestamp runs."""
    times = np.asarray(times_s, dtype=np.float64)
    x = np.asarray(signal, dtype=np.float64)
    if times.ndim != 1 or x.ndim != 1 or times.size != x.size:
        raise ValueError("times and signal must be matching one-dimensional arrays")
    if times.size < 2 or not np.isfinite(times).all() or not np.isfinite(x).all():
        raise ValueError("times and signal must contain at least two finite samples")
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be finite and positive")
    out = x.copy()
    valid = np.zeros(x.size, dtype=bool)
    counts = {"runs_seen": 0, "runs_filtered": 0, "runs_skipped": 0, "edge_samples_excluded": 0}
    for start, end in FILTER_ENGINE.split_contiguous_runs(times, gap_s):
        counts["runs_seen"] += 1
        if end - start < 8:
            counts["runs_skipped"] += 1
            continue
        fs_hz, _ = FILTER_ENGINE._estimate_fs(times[start:end])
        if fs_hz is None or fs_hz <= 2.0 * cutoff_hz:
            counts["runs_skipped"] += 1
            continue
        sos = FILTER_ENGINE.design_butterworth_sos(fs_hz, "highpass", (cutoff_hz,), order=4)
        padlen = FILTER_ENGINE.recommended_sos_padlen(sos)
        filtered, edge = FILTER_ENGINE.apply_iir_zero_phase(x[start:end], sos, padlen=padlen)
        if edge == 0 or end - start <= 2 * edge:
            counts["runs_skipped"] += 1
            continue
        out[start:end] = filtered
        valid[start + edge:end - edge] = True
        counts["runs_filtered"] += 1
        counts["edge_samples_excluded"] += 2 * edge
    return out, valid, counts


def _median_or_none(values) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(median(clean)) if clean else None


def spectral_run_eligible(sample_count: int, fs_hz: float, interval_spread_ratio: float) -> bool:
    """Require ≥8 s and ≤5% central interval spread before FFT-based screening."""
    return (
        isinstance(sample_count, (int, np.integer))
        and sample_count >= 2
        and math.isfinite(fs_hz)
        and fs_hz > 0
        and math.isfinite(interval_spread_ratio)
        and interval_spread_ratio <= MAX_INTERVAL_SPREAD_RATIO
        and sample_count / fs_hz >= MIN_LOW_FREQUENCY_SECONDS
    )


def _weighted_mean(pairs) -> float | None:
    usable = [(float(value), float(weight)) for value, weight in pairs
              if value is not None and weight > 0 and math.isfinite(float(value))]
    denominator = sum(weight for _, weight in usable)
    return sum(value * weight for value, weight in usable) / denominator if denominator else None


def _spectrum_metrics(signal, fs_hz: float, engine=FILTER_ENGINE) -> dict[str, float | None]:
    frequencies, psd = periodogram_psd(signal, fs_hz)
    nyquist = fs_hz / 2.0
    high_mid = min(MID_BAND[1], nyquist - fs_hz / len(signal))
    high_low = min(MOTION_SCREEN_BAND[1], nyquist - fs_hz / len(signal))
    metrics: dict[str, float | None] = {
        "power_near_1hz_adc2": None,
        "peak_near_1hz_hz": None,
        "peak_near_1hz_amplitude_adc": None,
        "power_0p5_10hz_adc2": None,
        "power_20_450hz_adc2": None,
        "mean_frequency_20_450hz": None,
        "median_frequency_20_450hz": None,
        "low_to_mid_power_ratio": None,
        "prominence_50_db": None,
        "prominence_60_db": None,
        "prominence_100_db": None,
        "prominence_120_db": None,
    }
    try:
        metrics["power_near_1hz_adc2"] = integrate_bandpower(frequencies, psd, 0.5, 1.5)
        near = (frequencies >= 0.5) & (frequencies < 1.5)
        if near.any():
            near_indices = np.flatnonzero(near)
            peak_index = int(near_indices[int(np.argmax(psd[near]))])
            metrics["peak_near_1hz_hz"] = float(frequencies[peak_index])
            window = np.hanning(len(signal))
            coefficients = np.fft.rfft((np.asarray(signal, dtype=np.float64) - float(np.mean(signal))) * window)
            metrics["peak_near_1hz_amplitude_adc"] = float(2.0 * abs(coefficients[peak_index]) / np.sum(window))
        metrics["power_0p5_10hz_adc2"] = integrate_bandpower(frequencies, psd, 0.5, high_low)
        if high_mid > MID_BAND[0]:
            band = (frequencies >= MID_BAND[0]) & (frequencies < high_mid)
            df = float(frequencies[1] - frequencies[0])
            band_power = float(np.sum(psd[band]) * df)
            metrics["power_20_450hz_adc2"] = band_power
            if band_power > 0.0:
                metrics["mean_frequency_20_450hz"] = float(np.sum(frequencies[band] * psd[band]) * df / band_power)
                cumulative = np.cumsum(psd[band]) * df
                band_frequencies = frequencies[band]
                median_index = min(int(np.searchsorted(cumulative, band_power / 2.0, side="left")), band_frequencies.size - 1)
                metrics["median_frequency_20_450hz"] = float(band_frequencies[median_index])
    except ValueError:
        pass
    low = metrics["power_0p5_10hz_adc2"]
    mid = metrics["power_20_450hz_adc2"]
    if low is not None and mid is not None and mid > 0:
        metrics["low_to_mid_power_ratio"] = low / mid
    for target in (50.0, 60.0, 100.0, 120.0):
        try:
            value = engine._line_prominence_db(np.asarray(signal), fs_hz, target)
        except (ValueError, FloatingPointError):
            value = float("nan")
        metrics[f"prominence_{int(target)}_db"] = float(value) if math.isfinite(float(value)) else None
    return metrics


def _mask_runs(times, valid_mask, gap_s: float = GAP_THRESHOLD_S):
    times = np.asarray(times, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if times.ndim != 1 or mask.ndim != 1 or times.size != mask.size:
        raise ValueError("times and mask must be matching one-dimensional arrays")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero((np.diff(indices) > 1) | (np.diff(times[indices]) > gap_s)) + 1
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in np.split(indices, breaks) if chunk.size]


def _pooled_metrics(times, signal, mask) -> dict[str, float | None]:
    values = np.asarray(signal, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    metric_rows = []
    squares = 0.0
    n_samples = 0
    for start, end in _mask_runs(times, mask):
        x = values[start:end]
        if x.size < 32:
            continue
        fs_hz, _ = FILTER_ENGINE._estimate_fs(times[start:end])
        if fs_hz is None:
            continue
        metrics = _spectrum_metrics(x, fs_hz)
        weight = x.size / fs_hz
        metric_rows.append((metrics, weight))
        centered = x - float(np.mean(x))
        squares += float(np.sum(centered**2))
        n_samples += int(x.size)
    output: dict[str, float | None] = {key: None for key in _spectrum_metrics(np.ones(64), 1000.0)}
    for key in output:
        output[key] = _weighted_mean((row[key], weight) for row, weight in metric_rows)
    output["rms_adc"] = float(math.sqrt(squares / n_samples)) if n_samples else None
    output["samples_common"] = n_samples
    return output


def _analyze_raw_spectrum(times, signal) -> dict[str, float | int | None]:
    spectral_rows = []
    resolutions = []
    fs_values = []
    spreads = []
    skipped = 0
    for start, end in FILTER_ENGINE.split_contiguous_runs(times, GAP_THRESHOLD_S):
        local_times = times[start:end]
        x = signal[start:end]
        if x.size < 2:
            skipped += 1
            continue
        fs_hz, spread = FILTER_ENGINE._estimate_fs(local_times)
        if fs_hz is None or spread is None or not spectral_run_eligible(x.size, fs_hz, spread):
            skipped += 1
            continue
        metrics = _spectrum_metrics(x, fs_hz)
        duration = x.size / fs_hz
        spectral_rows.append((metrics, duration))
        resolutions.append(fs_hz / x.size)
        fs_values.append(fs_hz)
        spreads.append(spread)
    names = tuple(_spectrum_metrics(np.ones(64), 1000.0))
    result: dict[str, float | int | None] = {key: None for key in names}
    for key in names:
        result[key] = _weighted_mean((row[key], duration) for row, duration in spectral_rows)
    result["spectral_runs_analyzed"] = len(spectral_rows)
    result["spectral_runs_insufficient_or_irregular"] = skipped
    result["best_frequency_resolution_hz"] = min(resolutions) if resolutions else None
    result["median_fs_estimated_hz"] = _median_or_none(fs_values)
    spread_ratio = _median_or_none(spreads)
    result["median_interval_spread_pct"] = spread_ratio * 100.0 if spread_ratio is not None else None
    return result


def analyze_record(metadata: dict, path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times_s"], dtype=np.float64)
        original = np.asarray(data["adc_original"], dtype=np.float64)
        control = np.asarray(data["control"], dtype=np.float64)
        common = np.asarray(data["common_comparison_mask"], dtype=bool)
        hp20 = np.asarray(data["hp20"], dtype=np.float64)
        applied20 = np.asarray(data["applied_hp20"], dtype=bool)
    if not (times.shape == original.shape == control.shape == common.shape == hp20.shape == applied20.shape):
        raise ValueError(f"signal arrays have inconsistent dimensions: {path.name}")
    expected_control = original.astype(np.float32).astype(np.float64)
    if not np.array_equal(expected_control, control):
        raise ValueError(f"stored control branch differs from saved ADC: {path.name}")
    hp10, valid10, hp10_counts = highpass_contiguous(times, original, 10.0)
    comparison_mask = common & applied20 & valid10
    branches = {"control": original, "hp10": hp10, "hp20": hp20}
    rows = []
    raw = _analyze_raw_spectrum(times, original)
    for branch in BRANCHES:
        metric = _pooled_metrics(times, branches[branch], comparison_mask)
        branch_metrics = attach_stored_spectrum(metric, raw, include=branch == "control")
        row = {
            "participant_id": metadata["participant_id"],
            "recording_id": metadata["recording_id"],
            "record_type": metadata["record_type"],
            "leg": metadata["leg"],
            "set_index": str(metadata["set_index"]),
            "filter_branch": branch,
            **branch_metrics,
        }
        rows.append(row)
    reference = next(row for row in rows if row["filter_branch"] == "control")
    for row in rows:
        for metric, output in (
            ("rms_adc", "change_rms_pct_vs_control"),
            ("power_0p5_10hz_adc2", "change_0p5_10hz_pct_vs_control"),
            ("power_20_450hz_adc2", "change_20_450hz_pct_vs_control"),
        ):
            base = reference.get(metric)
            current = row.get(metric)
            row[output] = (
                (float(current) - float(base)) / float(base) * 100.0
                if base is not None and current is not None and float(base) != 0.0
                else None
            )
        for metric, delta_field, pct_field in (
            ("mean_frequency_20_450hz", "change_mean_frequency_hz_vs_control", "change_mean_frequency_pct_vs_control"),
            ("median_frequency_20_450hz", "change_median_frequency_hz_vs_control", "change_median_frequency_pct_vs_control"),
        ):
            base = reference.get(metric)
            current = row.get(metric)
            row[delta_field] = float(current) - float(base) if base is not None and current is not None else None
            row[pct_field] = (
                (float(current) - float(base)) / float(base) * 100.0
                if base is not None and current is not None and float(base) != 0.0
                else None
            )
    for row in rows:
        row["hp10_runs_filtered"] = hp10_counts["runs_filtered"]
        row["hp10_runs_skipped"] = hp10_counts["runs_skipped"]
        row["hp10_edge_samples_excluded"] = hp10_counts["edge_samples_excluded"]
    return rows


def attach_stored_spectrum(branch_metrics: dict, stored_metrics: dict, include: bool = True) -> dict:
    """Keep full-record channel spectra separate from filtered common-mask metrics."""
    result = dict(branch_metrics)
    for key, value in stored_metrics.items():
        result[f"stored_{key}"] = value if include else None
    return result


def aggregate_private_rows(rows: list[dict]) -> list[dict]:
    """Aggregate first within participant, then across participants; omit IDs."""
    granular: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    qc: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        branch = row["filter_branch"]
        if row["record_type"] == "series_R1_R5":
            granular[(row["record_type"], row["leg"], str(row["set_index"]), branch)][row["participant_id"]].append(row)
            overall[("R1-R5", "all", "all", branch)][row["participant_id"]].append(row)
        else:
            qc[("preliminary_R0_R0_1", "all", "QC", branch)][row["participant_id"]].append(row)
    output = []
    for source in (granular, overall, qc):
        for (record_type, leg, set_index, branch), by_person in source.items():
            person_rows = []
            for rows_for_person in by_person.values():
                person_rows.append({
                    field: _median_or_none(r.get(field) for r in rows_for_person)
                    for field in NUMERIC_FIELDS
                })
            row = {
                "record_type": record_type,
                "leg": leg,
                "set_index": set_index,
                "filter_branch": branch,
                "n_participants": len(person_rows),
                "n_recordings": sum(len(items) for items in by_person.values()),
            }
            for field in NUMERIC_FIELDS:
                row[f"median_{field}"] = _median_or_none(item[field] for item in person_rows)
            output.append(row)
    output.sort(key=lambda r: (r["record_type"], r["leg"], r["set_index"], BRANCHES.index(r["filter_branch"])))
    return output


def assert_public_safe(fields, rows: list[dict]) -> None:
    """Reject person/record keys and pseudonym values, while allowing counts."""
    forbidden = {
        "participant_id", "recording_id", "input_path", "output_path",
        "source_filename", "filename", "filepath",
    }
    if any(str(field).strip().lower() in forbidden for field in fields):
        raise ValueError("public aggregate contains a person- or record-level identifier field")
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and re.fullmatch(r"P\d{3}", value.strip(), flags=re.IGNORECASE):
                raise ValueError("public aggregate contains a participant pseudonym")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _svg_figure(path: Path, rows: list[dict]) -> None:
    branches = [r for r in rows if r["record_type"] == "R1-R5" and r["leg"] == "all" and r["set_index"] == "all"]
    order = {name: i for i, name in enumerate(BRANCHES)}
    branches.sort(key=lambda r: order[r["filter_branch"]])
    width, height = 1000, 560
    colors = {"control": "#52606d", "hp10": "#0072B2", "hp20": "#D55E00"}
    panels = [
        ("median_change_0p5_10hz_pct_vs_control", "Relative change in 0.5–10 Hz power"),
        ("median_change_20_450hz_pct_vs_control", "Relative change in 20–450 Hz power"),
    ]
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="#ffffff"/>',
               '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#24313a}.title{font-size:23px;font-weight:bold}.axis{font-size:13px}.label{font-size:15px;font-weight:bold}.small{font-size:12px}</style>',
               '<text class="title" x="500" y="35" text-anchor="middle">Stored-channel filter sensitivity (participant-level medians)</text>']
    for panel_index, (field, title) in enumerate(panels):
        top = 78 + panel_index * 230
        left, plot_w, plot_h = 110, 820, 145
        values = [float(r[field]) for r in branches if r.get(field) is not None]
        extent = max([5.0] + [abs(v) for v in values])
        extent = min(max(extent * 1.2, 10.0), 150.0)
        y0 = top + plot_h / 2.0
        content.append(f'<text class="label" x="{left}" y="{top-12}">{title} (%)</text>')
        content.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{left+plot_w}" y2="{y0:.1f}" stroke="#718096" stroke-width="1"/>')
        content.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#a0aec0" stroke-width="1"/>')
        for tick in (-extent, -extent/2, 0.0, extent/2, extent):
            y = y0 - tick / extent * (plot_h/2.0)
            content.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
            content.append(f'<text class="small" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
        for i, row in enumerate(branches):
            value = row.get(field)
            if value is None:
                continue
            x = left + plot_w * (i + 0.5) / len(BRANCHES)
            y = y0 - float(value) / extent * (plot_h/2.0)
            content.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="{colors[row["filter_branch"]]}" stroke-width="7"/>')
            content.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors[row["filter_branch"]]}"/>')
            content.append(f'<text class="axis" x="{x:.1f}" y="{top+plot_h+21}" text-anchor="middle">{row["filter_branch"]}</text>')
            content.append(f'<text class="small" x="{x:.1f}" y="{y-9:.1f}" text-anchor="middle">{float(value):.1f}</text>')
    content.append('<text class="small" x="500" y="550" text-anchor="middle">Changes quantify filter response, not denoising quality; raw/rectified/envelope mode is unconfirmed.</text>')
    content.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def run_analysis(private_root: Path, output_private: Path, output_aggregate: Path, figure_path: Path) -> dict:
    receipt_path = private_root / "receipt_filter_sensitivity_v1.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"prior verified filter receipt not found: {receipt_path}")
    prior = json.loads(receipt_path.read_text(encoding="utf-8"))
    if prior.get("records") != 140 or prior.get("samples") != 4256257:
        raise ValueError("prior filter receipt does not match the verified 140-record local cohort")
    expected = {Path(item["output_path"]).name: item for item in prior["records_private"]}
    files = sorted(private_root.glob("*.npz"))
    if len(files) != 140 or len(expected) != 140:
        raise ValueError("expected exactly 140 private signal arrays and 140 receipt records")
    rows = []
    total_samples = 0
    input_hashes = {}
    for path in files:
        metadata = expected.get(path.name)
        if metadata is None:
            raise ValueError(f"private signal array is absent from the prior receipt: {path.name}")
        file_hash = _sha256(path)
        if file_hash != metadata.get("output_sha256"):
            raise ValueError(f"private signal array hash mismatch: {path.name}")
        input_hashes[path.name] = file_hash
        with np.load(path, allow_pickle=False) as data:
            n = int(data["adc_original"].size)
        total_samples += n
        rows.extend(analyze_record(metadata, path))
    if total_samples != 4256257 or len(rows) != 420:
        raise ValueError("signal sample or branch-row total differs from the verified cohort")
    aggregate = aggregate_private_rows(rows)
    _write_csv(output_private / "diagnostico_baja_frecuencia_por_registro_PRIVADO.csv", rows, PRIVATE_FIELDS)
    _write_csv(output_aggregate, aggregate, PUBLIC_FIELDS)
    _svg_figure(figure_path, aggregate)
    assert_public_safe(PUBLIC_FIELDS, aggregate)
    receipt = {
        "analysis": "local low-frequency / HP10 filter sensitivity extension",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "15 local participants; R1-R5 primary, R0/R0_1 QC only",
        "records": 140,
        "samples": total_samples,
        "rows_private": len(rows),
        "rows_aggregate": len(aggregate),
        "branches": list(BRANCHES),
        "frequency_bands_hz": {"near_1_hz": [0.5, 1.5], "low_frequency_screen": [0.5, 10.0], "mid_band": [20.0, 450.0]},
        "minimum_low_frequency_segment_seconds": MIN_LOW_FREQUENCY_SECONDS,
        "max_interval_spread_pct_for_spectrum": MAX_INTERVAL_SPREAD_RATIO * 100.0,
        "timestamp_gap_split_s": GAP_THRESHOLD_S,
        "power_line_screen_hz": [50, 60, 100, 120],
        "mode_and_sampling_clock": "unknown; fs estimated from timestamps per contiguous run",
        "synthetic_contamination_added": False,
        "external_data_used": False,
        "fatigue_labels_created": False,
        "training_performed": False,
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "filter_engine_script_sha256": _sha256(Path(__file__).with_name("comparar_filtros_emg.py")),
        "test_script_sha256": _sha256(Path(__file__).with_name("tests") / "test_diagnostico_baja_frecuencia_emg.py"),
        "prior_receipt_sha256": _sha256(receipt_path),
        "private_input_sha256": input_hashes,
        "aggregate_sha256": _sha256(output_aggregate),
        "figure_sha256": _sha256(figure_path),
    }
    output_private.mkdir(parents=True, exist_ok=True)
    receipt_file = output_private / "receipt_diagnostico_baja_frecuencia_hp10_v1.json"
    receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-private", type=Path, required=True)
    parser.add_argument("--output-aggregate", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    receipt = run_analysis(args.private_root, args.output_private, args.output_aggregate, args.figure)
    print(json.dumps({key: value for key, value in receipt.items() if key != "private_input_sha256"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
