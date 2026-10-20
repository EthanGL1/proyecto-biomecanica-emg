"""Exploratory, privacy-preserving signal-threshold audit for local EMG files.

Thresholds flag sustained departures from each recording's own early-window
distribution. They are signal-change screens, not calibrated fatigue labels.
Original input CSVs and the historical filter benchmark are read-only.
"""

from __future__ import annotations

from release_paths import resolve_recorded_npz_path, validate_data_paths

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


GAP_THRESHOLD_S = 0.0015
WINDOW_SAMPLES = 1024
BASELINE_FRACTION = 1.0 / 3.0
ROBUST_Z = 1.96
MIN_BASELINE_WINDOWS = 5
MIN_CONSECUTIVE_WINDOWS = 3
LOW_HZ = 20.0
HIGH_HZ = 450.0
BRANCHES = ("control", "hp20", "hp40", "bp20_400", "fir_bp20_400")
BRANCH_LABELS = {
    "control": "RAW/control",
    "hp20": "HP 20 Hz",
    "hp40": "HP 40 Hz",
    "bp20_400": "BP IIR 20–400 Hz",
    "fir_bp20_400": "BP FIR 20–400 Hz",
}
COLORS = {
    "control": "#263238",
    "hp20": "#0072B2",
    "hp40": "#D55E00",
    "bp20_400": "#009E73",
    "fir_bp20_400": "#CC79A7",
}


def spectral_peak(
    signal: Iterable[float], fs_hz: float, low_hz: float = LOW_HZ, high_hz: float = HIGH_HZ
) -> dict[str, float]:
    """Return median frequency and maximum single-bin power within the band."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 8 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional window of at least 8 samples")
    if not math.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    if not (0 <= low_hz < high_hz < fs_hz / 2):
        raise ValueError("analysis band must lie below Nyquist")
    centered = values - np.mean(values)
    taper = np.hanning(values.size)
    power = np.abs(np.fft.rfft(centered * taper)) ** 2 / float(np.sum(taper**2))
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / fs_hz)
    select = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not select.any():
        raise ValueError("analysis band has no FFT bins")
    band_power = power[select]
    band_frequencies = frequencies[select]
    total_power = float(np.sum(band_power))
    if not math.isfinite(total_power) or total_power <= 0:
        raise ValueError("analysis band has no positive spectral power")
    peak_index = int(np.argmax(band_power))
    median_index = min(int(np.searchsorted(np.cumsum(band_power), total_power / 2.0)), band_power.size - 1)
    fft = np.fft.rfft(centered * taper)
    peak_bin = int(np.flatnonzero(select)[peak_index])
    peak_amplitude = 2.0 * abs(fft[peak_bin]) / float(np.sum(taper))
    return {
        "median_frequency_hz": float(band_frequencies[median_index]),
        "peak_frequency_hz": float(band_frequencies[peak_index]),
        "peak_power": float(band_power[peak_index]),
        "peak_amplitude_adc": float(peak_amplitude),
    }


def iter_valid_windows(
    times_s: Iterable[float],
    signal: Iterable[float],
    valid_mask: Iterable[bool],
    window_samples: int = WINDOW_SAMPLES,
    gap_s: float = GAP_THRESHOLD_S,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
) -> list[dict[str, float | int]]:
    """Make non-overlapping windows without joining timestamp gaps or invalid edges."""
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if times.ndim != 1 or values.ndim != 1 or mask.ndim != 1:
        raise ValueError("times, signal, and mask must be one-dimensional")
    if times.size != values.size or times.size != mask.size:
        raise ValueError("times, signal, and mask must have equal lengths")
    if window_samples < 8 or not math.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("window_samples and gap_s are invalid")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("times and signal must be finite")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    time_span = float(times[-1] - times[0]) if times.size > 1 else 0.0
    gaps = np.flatnonzero(np.diff(times) > gap_s) + 1
    bounds = np.r_[0, gaps, times.size]
    rows: list[dict[str, float | int]] = []
    segment_id = 0
    for bound_start, bound_end in zip(bounds[:-1], bounds[1:]):
        local_valid = np.flatnonzero(mask[bound_start:bound_end]) + int(bound_start)
        if local_valid.size == 0:
            continue
        split_at = np.flatnonzero(np.diff(local_valid) > 1) + 1
        for indices in np.split(local_valid, split_at):
            run_start, run_end = int(indices[0]), int(indices[-1]) + 1
            if run_end - run_start < window_samples:
                segment_id += 1
                continue
            for start in range(run_start, run_end - window_samples + 1, window_samples):
                end = start + window_samples
                interval = np.diff(times[start:end])
                if interval.size == 0 or np.any(interval > gap_s):
                    continue
                fs_hz = float(1.0 / np.median(interval))
                band_high = min(float(high_hz), fs_hz / 2.0 - fs_hz / window_samples)
                if band_high <= low_hz:
                    continue
                window = values[start:end]
                centered = window - float(np.mean(window))
                spectrum = spectral_peak(window, fs_hz, low_hz, band_high)
                rows.append(
                    {
                        "start_index": start,
                        "end_index": end,
                        "segment_id": segment_id,
                        "time_s": float((times[start] + times[end - 1]) / 2.0),
                        "time_fraction": float((times[start] + times[end - 1]) / 2.0 - times[0]) / time_span if time_span else 0.0,
                        "fs_hz": fs_hz,
                        "rms_adc": float(np.sqrt(np.mean(centered**2))),
                        "median_frequency_hz": spectrum["median_frequency_hz"],
                        "peak_frequency_hz": spectrum["peak_frequency_hz"],
                        "peak_power_adc2": spectrum["peak_power"],
                        "peak_amplitude_adc": spectrum["peak_amplitude_adc"],
                    }
                )
            segment_id += 1
    return rows


def robust_lower_threshold(
    values: Iterable[float],
    baseline_fraction: float = BASELINE_FRACTION,
    multiplier: float = ROBUST_Z,
    minimum_windows: int = MIN_BASELINE_WINDOWS,
) -> dict[str, float | int | str | None]:
    """Estimate an exploratory lower reference limit from the early window median/MAD."""
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if not (0 < baseline_fraction <= 1) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("baseline_fraction and multiplier must be positive")
    if minimum_windows < 1:
        raise ValueError("minimum_windows must be positive")
    count = int(math.ceil(data.size * baseline_fraction))
    if count < minimum_windows:
        return {"status": "insufficient_baseline", "n_baseline": count, "baseline_median": None, "robust_scale": None, "lower_threshold": None}
    baseline = data[:count]
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= np.finfo(float).eps * max(1.0, abs(center)):
        return {"status": "zero_mad", "n_baseline": count, "baseline_median": center, "robust_scale": scale, "lower_threshold": None}
    return {
        "status": "estimated",
        "n_baseline": count,
        "baseline_median": center,
        "robust_scale": scale,
        "lower_threshold": center - multiplier * scale,
    }


def sustained_crossings(
    below_threshold: Iterable[bool],
    segment_ids: Iterable[int],
    min_consecutive: int = MIN_CONSECUTIVE_WINDOWS,
) -> list[dict[str, int]]:
    """Return below-threshold runs that persist within one contiguous valid segment."""
    flags = list(bool(x) for x in below_threshold)
    segments = list(int(x) for x in segment_ids)
    if len(flags) != len(segments):
        raise ValueError("flags and segment_ids must have equal lengths")
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be positive")
    events: list[dict[str, int]] = []
    start: int | None = None
    active_segment: int | None = None
    for index in range(len(flags) + 1):
        is_below = index < len(flags) and flags[index]
        segment = segments[index] if index < len(flags) else None
        if is_below and (start is None or segment == active_segment):
            if start is None:
                start, active_segment = index, segment
            continue
        if start is not None:
            duration = index - start
            if duration >= min_consecutive:
                events.append({"segment_id": int(active_segment), "start_window_index": start, "duration_windows": duration})
        start = index if is_below else None
        active_segment = segment if is_below else None
    return events


def maximum_absolute_point(
    times_s: Iterable[float], signal: Iterable[float], valid_mask: Iterable[bool]
) -> dict[str, float | int | None]:
    """Find the signed sample and time of the largest absolute amplitude in the mask."""
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if times.ndim != 1 or values.ndim != 1 or mask.ndim != 1 or not (times.size == values.size == mask.size):
        raise ValueError("times, signal, and mask must be equal-length one-dimensional arrays")
    valid = np.flatnonzero(mask & np.isfinite(values) & np.isfinite(times))
    if not valid.size:
        return {"sample_index": None, "time_s": None, "value": None, "absolute_amplitude": None}
    index = int(valid[np.argmax(np.abs(values[valid]))])
    return {"sample_index": index, "time_s": float(times[index]), "value": float(values[index]), "absolute_amplitude": float(abs(values[index]))}


def _safe_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token or "unknown"


def _median(values: Iterable[float]) -> float | None:
    finite = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=float)
    return float(np.median(finite)) if finite.size else None


def _change_pct(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or not math.isfinite(first) or first <= 0:
        return None
    return float((last / first - 1.0) * 100.0)


def _event_crossings(
    windows: list[dict[str, float | int]],
    amplitude_threshold: dict,
    frequency_threshold: dict,
) -> dict[str, list[dict[str, int]]]:
    amp_limit = amplitude_threshold.get("lower_threshold")
    freq_limit = frequency_threshold.get("lower_threshold")
    amp_flags = [amp_limit is not None and float(row["rms_adc"]) < float(amp_limit) for row in windows]
    freq_flags = [freq_limit is not None and float(row["median_frequency_hz"]) < float(freq_limit) for row in windows]
    segment_ids = [int(row["segment_id"]) for row in windows]
    both_flags = [a and f for a, f in zip(amp_flags, freq_flags)]
    return {
        "amplitude": sustained_crossings(amp_flags, segment_ids),
        "frequency": sustained_crossings(freq_flags, segment_ids),
        "both": sustained_crossings(both_flags, segment_ids),
    }


def _csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                clean[field] = value
            writer.writerow(clean)


def _svg_text(x: float, y: float, text: str, size: int = 13, color: str = "#263238", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}">{html.escape(str(text))}</text>'


def _polyline_path(
    times: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    x_map,
    y_map,
    max_points: int = 2600,
) -> list[str]:
    indices = np.flatnonzero(mask & np.isfinite(times) & np.isfinite(values))
    if not indices.size:
        return []
    split_points = np.flatnonzero((np.diff(indices) > 1) | (np.diff(times[indices]) > GAP_THRESHOLD_S)) + 1
    paths = []
    total = indices.size
    stride = max(1, int(math.ceil(total / max_points)))
    for run in np.split(indices, split_points):
        if not run.size:
            continue
        draw = run[::stride]
        if draw[-1] != run[-1]:
            draw = np.r_[draw, run[-1]]
        coords = " ".join(f"{x_map(float(times[i])):.2f},{y_map(float(values[i])):.2f}" for i in draw)
        if coords:
            paths.append(f'<polyline points="{coords}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    return paths


def _window_path(
    windows: list[dict[str, float | int]],
    value_key: str,
    x_map,
    y_map,
    max_points: int = 1600,
) -> list[str]:
    if not windows:
        return []
    paths = []
    stride = max(1, int(math.ceil(len(windows) / max_points)))
    start = 0
    while start < len(windows):
        segment = int(windows[start]["segment_id"])
        end = start + 1
        while end < len(windows) and int(windows[end]["segment_id"]) == segment:
            end += 1
        subset = windows[start:end]
        chosen = subset[::stride]
        if subset and chosen[-1] is not subset[-1]:
            chosen = chosen + [subset[-1]]
        coords = " ".join(
            f"{x_map(float(row['time_s'])):.2f},{y_map(float(row[value_key])):.2f}"
            for row in chosen
            if row.get(value_key) is not None and math.isfinite(float(row[value_key]))
        )
        if coords:
            paths.append(f'<polyline points="{coords}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
        start = end
    return paths


def _record_figure(
    out_path: Path,
    record: dict,
    times: np.ndarray,
    raw: np.ndarray,
    signals: dict[str, np.ndarray],
    common_mask: np.ndarray,
    windows_by_branch: dict[str, list[dict]],
    thresholds_by_branch: dict[str, dict],
    raw_peak: dict,
    max_points: int = 2600,
) -> None:
    width, height = 1240, 1110
    left, right = 106, 1194
    panels = [(115, 405, "Señal muestreada (ADC)"), (480, 665, "RMS por ventanas (ADC)"), (738, 923, "Frecuencia mediana por ventanas (Hz)")]
    time_min = float(times[0]) if times.size else 0.0
    time_max = float(times[-1]) if times.size else 1.0
    if time_max <= time_min:
        time_max = time_min + 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Señal cruda y ramas de filtro; serie {html.escape(str(record["recording_id"]))}, lado {html.escape(str(record["leg"]))}</title>',
        '<desc id="desc">Comparación de la señal guardada con control, pasa-altas de 20 y 40 hertz y dos bandas de 20 a 400 hertz. Incluye ventanas de RMS y frecuencia mediana con umbrales robustos de referencia inicial.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(48, 35, f'EMG exploratorio | {record["participant_id"]} | {record["recording_id"]} | {record["leg"]} | serie {record["set_index"]}', 18, "#17324D"),
        _svg_text(48, 59, f'{record["record_type"]} · {len(raw):,} muestras · modo del canal desconocido · ventana {WINDOW_SAMPLES} muestras', 12, "#455A64"),
    ]
    all_values = [raw]
    all_values.extend(signals[b][common_mask] for b in BRANCHES if np.any(common_mask))
    signal_min = min(float(np.min(v)) for v in all_values if len(v))
    signal_max = max(float(np.max(v)) for v in all_values if len(v))
    signal_pad = max((signal_max - signal_min) * 0.06, 1.0)
    y_ranges = [(signal_min - signal_pad, signal_max + signal_pad), (0.0, 1.0), (LOW_HZ, HIGH_HZ)]
    for panel_index, (y_top, y_bottom, label) in enumerate(panels):
        ymin, ymax = y_ranges[panel_index]
        if panel_index == 1:
            vals = [float(w["rms_adc"]) for ws in windows_by_branch.values() for w in ws]
            lows = [float(t["rms"]["lower_threshold"]) for t in thresholds_by_branch.values() if t["rms"].get("lower_threshold") is not None]
            ymax = max(vals + [0.0]) if vals else 1.0
            ymin = max(0.0, min(lows + [0.0]))
            if ymax <= ymin:
                ymax = ymin + 1.0
        elif panel_index == 2:
            vals = [float(w["median_frequency_hz"]) for ws in windows_by_branch.values() for w in ws]
            lows = [float(t["mdf"]["lower_threshold"]) for t in thresholds_by_branch.values() if t["mdf"].get("lower_threshold") is not None]
            ymin = max(LOW_HZ, min(lows + [LOW_HZ]))
            ymax = min(HIGH_HZ, max(vals + lows)) if vals or lows else HIGH_HZ
            if ymax <= ymin:
                pad = max(5.0, abs(ymin) * 0.05)
                ymin, ymax = max(LOW_HZ, ymin - pad), min(HIGH_HZ, ymax + pad)
            elif ymax - ymin < 10.0:
                pad = (10.0 - (ymax - ymin)) / 2.0
                ymin, ymax = max(LOW_HZ, ymin - pad), min(HIGH_HZ, ymax + pad)
        y_ranges[panel_index] = (ymin, ymax)
        def x_map(t):
            return left + (float(t) - time_min) / (time_max - time_min) * (right - left)
        def y_map(v):
            return y_bottom - (float(v) - ymin) / (ymax - ymin) * (y_bottom - y_top)
        lines.append(_svg_text(48, y_top + 4, label, 14, "#17324D"))
        for tick_index in range(5):
            value = ymin + (ymax - ymin) * tick_index / 4
            y = y_map(value)
            lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#E4E9ED" stroke-width="1"/>')
            lines.append(_svg_text(left - 12, y + 4, f"{value:.1f}", 11, "#52616B", "end"))
        lines.append(f'<line x1="{left}" y1="{y_top}" x2="{left}" y2="{y_bottom}" stroke="#52616B"/>')
        lines.append(f'<line x1="{left}" y1="{y_bottom}" x2="{right}" y2="{y_bottom}" stroke="#52616B"/>')
        if panel_index == 0:
            raw_mask = np.ones(raw.size, dtype=bool)
            for path in _polyline_path(times, raw, raw_mask, x_map, y_map, max_points):
                lines.append(path.replace("stroke-linecap", 'stroke="#65727A" stroke-width="0.85" opacity="0.78" stroke-linecap'))
            for branch in BRANCHES:
                for path in _polyline_path(times, signals[branch], common_mask, x_map, y_map, max_points):
                    lines.append(path.replace("stroke-linecap", f'stroke="{COLORS[branch]}" stroke-width="1.1" opacity="0.76" stroke-linecap'))
            px, py = x_map(float(raw_peak["time_s"])), y_map(float(raw_peak["value"]))
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.8" fill="#B2182B" stroke="#fff" stroke-width="1.5"/>')
            lines.append(_svg_text(right, y_top + 18, f'|ADC| máximo original = {raw_peak["absolute_amplitude"]:.2f} en t={raw_peak["time_s"]:.3f} s', 11, "#B2182B", "end"))
        else:
            key = "rms_adc" if panel_index == 1 else "median_frequency_hz"
            for branch in BRANCHES:
                for path in _window_path(windows_by_branch[branch], key, x_map, y_map):
                    lines.append(path.replace("stroke-linecap", f'stroke="{COLORS[branch]}" stroke-width="1.8" opacity="0.86" stroke-linecap'))
                threshold_key = "rms" if panel_index == 1 else "mdf"
                threshold = thresholds_by_branch[branch][threshold_key].get("lower_threshold")
                if threshold is not None and ymin <= float(threshold) <= ymax:
                    y = y_map(float(threshold))
                    lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{COLORS[branch]}" stroke-width="1" stroke-dasharray="6 4" opacity="0.58"/>')
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            tx = time_min + fraction * (time_max - time_min)
            x = x_map(tx)
            lines.append(f'<line x1="{x:.2f}" y1="{y_bottom}" x2="{x:.2f}" y2="{y_bottom+4}" stroke="#52616B"/>')
            lines.append(_svg_text(x, y_bottom + 18, f"{tx-time_min:.0f}", 10, "#52616B", "middle"))
    lines.append(_svg_text((left + right) / 2, 964, "Tiempo relativo desde el inicio del archivo (s)", 12, "#263238", "middle"))
    legend_x, legend_y = left, 997
    for i, branch in enumerate(BRANCHES):
        x = legend_x + i * 212
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+26}" y2="{legend_y}" stroke="{COLORS[branch]}" stroke-width="3"/>')
        lines.append(_svg_text(x + 34, legend_y + 4, BRANCH_LABELS[branch], 11, "#263238"))
    lines.append(_svg_text(48, 1037, "Referencia: mediana de la primera tercera parte válida − 1,96 × (1,4826 × MAD); cruce requiere 3 ventanas válidas consecutivas.", 11, "#455A64"))
    lines.append(_svg_text(48, 1057, "Amplitud y frecuencia se informan por separado; un cruce no constituye nivel ni diagnóstico de fatiga. Las ventanas sin MAD estimable quedan sin umbral.", 11, "#455A64"))
    peak_text = []
    for branch in BRANCHES:
        best = max(windows_by_branch[branch], key=lambda w: float(w["peak_power_adc2"]), default=None)
        peak_text.append(f'{BRANCH_LABELS[branch]} {float(best["peak_frequency_hz"]):.1f} Hz' if best else f'{BRANCH_LABELS[branch]} N/D')
    lines.append(_svg_text(48, 1081, "Pico espectral de mayor potencia por rama: " + " · ".join(peak_text), 10, "#455A64"))
    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_signal_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {"Relative_Time_s", "ADC_Value"}.issubset(rows.fieldnames or []):
            raise ValueError(f"required signal columns are absent from {path.name}")
        pairs = [(float(row["Relative_Time_s"]), float(row["ADC_Value"])) for row in rows]
    values = np.asarray(pairs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"signal CSV is empty or nonfinite: {path.name}")
    return values[:, 0], values[:, 1]


def _summarize_threshold_rows(record_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    r15 = [row for row in record_rows if row["record_type"] == "series_R1_R5"]
    qc = [row for row in record_rows if row["record_type"] != "series_R1_R5"]
    by_series: dict[tuple, list[dict]] = defaultdict(list)
    for row in r15:
        by_series[(row["recording_id"], row["leg"], row["branch"])].append(row)
    series_rows = []
    for (recording_id, leg, branch), group in sorted(by_series.items()):
        series_rows.append({
            "recording_id": recording_id,
            "leg": leg,
            "filter_branch": branch,
            "n_records": len(group),
            "n_participants": len({row["participant_id"] for row in group}),
            "median_rms_early_adc": _median(row["rms_early_median_adc"] for row in group),
            "median_rms_late_adc": _median(row["rms_late_median_adc"] for row in group),
            "median_rms_change_pct": _median(row["rms_early_to_late_change_pct"] for row in group),
            "median_mdf_early_hz": _median(row["mdf_early_median_hz"] for row in group),
            "median_mdf_late_hz": _median(row["mdf_late_median_hz"] for row in group),
            "median_mdf_change_hz": _median(row["mdf_early_to_late_change_hz"] for row in group),
            "median_peak_abs_amplitude_adc": _median(row["peak_abs_amplitude_adc_common_mask"] for row in group),
            "median_dominant_peak_frequency_hz": _median(row["dominant_window_peak_frequency_hz"] for row in group),
            "median_dominant_peak_power_adc2": _median(row["dominant_window_peak_power_adc2"] for row in group),
            "median_peak_point_spectral_amplitude_adc": _median(row["peak_point_spectral_amplitude_adc"] for row in group),
            "median_peak_point_spectral_frequency_hz": _median(row["peak_point_spectral_frequency_hz"] for row in group),
            "records_with_sustained_rms_drop": sum(bool(row["rms_crossing_count"]) for row in group),
            "records_with_sustained_mdf_drop": sum(bool(row["mdf_crossing_count"]) for row in group),
            "records_with_coincident_drop": sum(bool(row["coincident_crossing_count"]) for row in group),
            "note": "Resumen descriptivo por serie/lado; las ventanas y los umbrales no son etiquetas de fatiga.",
        })

    qc_by_branch: dict[str, list[dict]] = defaultdict(list)
    for row in qc:
        qc_by_branch[row["branch"]].append(row)
    qc_rows = []
    for branch, group in sorted(qc_by_branch.items()):
        qc_rows.append({
            "record_type": "preliminary_R0_R0_1_QC_only",
            "filter_branch": branch,
            "n_records": len(group),
            "n_participants": len({row["participant_id"] for row in group}),
            "median_peak_abs_amplitude_adc": _median(row["peak_abs_amplitude_adc_common_mask"] for row in group),
            "median_peak_window_frequency_hz": _median(row["dominant_window_peak_frequency_hz"] for row in group),
            "median_peak_window_power_adc2": _median(row["dominant_window_peak_power_adc2"] for row in group),
            "median_peak_point_spectral_amplitude_adc": _median(row["peak_point_spectral_amplitude_adc"] for row in group),
            "median_peak_point_spectral_frequency_hz": _median(row["peak_point_spectral_frequency_hz"] for row in group),
            "records_with_sustained_rms_drop": sum(bool(row["rms_crossing_count"]) for row in group),
            "records_with_sustained_mdf_drop": sum(bool(row["mdf_crossing_count"]) for row in group),
            "records_with_coincident_drop": sum(bool(row["coincident_crossing_count"]) for row in group),
            "note": "QC preliminar agregado; no se usa como etiqueta ni entrenamiento.",
        })

    by_subject_branch: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in r15:
        by_subject_branch[(row["participant_id"], row["branch"])].append(row)
    subject_rows = []
    for (participant, branch), group in sorted(by_subject_branch.items()):
        subject_rows.append({
            "participant_id": participant,
            "branch": branch,
            "rms_early_to_late_change_pct": _median(row["rms_early_to_late_change_pct"] for row in group),
            "mdf_early_to_late_change_hz": _median(row["mdf_early_to_late_change_hz"] for row in group),
            "peak_abs_amplitude_adc": _median(row["peak_abs_amplitude_adc_common_mask"] for row in group),
            "dominant_peak_frequency_hz": _median(row["dominant_window_peak_frequency_hz"] for row in group),
            "dominant_peak_power_adc2": _median(row["dominant_window_peak_power_adc2"] for row in group),
            "peak_point_spectral_amplitude_adc": _median(row["peak_point_spectral_amplitude_adc"] for row in group),
            "peak_point_spectral_frequency_hz": _median(row["peak_point_spectral_frequency_hz"] for row in group),
            "has_sustained_rms_drop": any(bool(row["rms_crossing_count"]) for row in group),
            "has_sustained_mdf_drop": any(bool(row["mdf_crossing_count"]) for row in group),
            "has_coincident_drop": any(bool(row["coincident_crossing_count"]) for row in group),
            "n_records": len(group),
        })
    by_branch: dict[str, list[dict]] = defaultdict(list)
    for row in subject_rows:
        by_branch[row["branch"]].append(row)
    overall_rows = []
    rng = np.random.default_rng(20261015)
    for branch in BRANCHES:
        group = by_branch.get(branch, [])
        aggregate = {"filter_branch": branch, "n_participants": len(group)}
        for metric, prefix in (
            ("rms_early_to_late_change_pct", "rms_change_pct"),
            ("mdf_early_to_late_change_hz", "mdf_change_hz"),
            ("peak_abs_amplitude_adc", "peak_abs_amplitude_adc"),
            ("peak_point_spectral_amplitude_adc", "peak_point_spectral_amplitude_adc"),
            ("peak_point_spectral_frequency_hz", "peak_point_spectral_frequency_hz"),
            ("dominant_peak_frequency_hz", "dominant_peak_frequency_hz"),
        ):
            values = np.asarray([row[metric] for row in group if row[metric] is not None], dtype=float)
            if values.size:
                boot_idx = rng.integers(0, values.size, size=(5000, values.size))
                boot_medians = np.median(values[boot_idx], axis=1)
                aggregate[f"median_{prefix}"] = float(np.median(values))
                aggregate[f"ic95_{prefix}_lower"] = float(np.percentile(boot_medians, 2.5))
                aggregate[f"ic95_{prefix}_upper"] = float(np.percentile(boot_medians, 97.5))
            else:
                aggregate[f"median_{prefix}"] = None
                aggregate[f"ic95_{prefix}_lower"] = None
                aggregate[f"ic95_{prefix}_upper"] = None
        for field in ("has_sustained_rms_drop", "has_sustained_mdf_drop", "has_coincident_drop"):
            aggregate[f"participants_{field.removeprefix('has_')}"] = sum(bool(row[field]) for row in group)
        aggregate["note"] = "Un valor mediano por participante sobre R1–R5; cambio intrarchivo exploratorio, no fatiga validada."
        overall_rows.append(aggregate)
    return series_rows, qc_rows, overall_rows, subject_rows


def _public_summary_svg(path: Path, overall_rows: list[dict]) -> None:
    width, height = 1240, 760
    x0, x1 = 190, 1120
    rows = {row["filter_branch"]: row for row in overall_rows}
    branches = list(BRANCHES)
    centers = [195 + i * 205 for i in range(len(branches))]
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Cambios de amplitud y frecuencia por rama de filtro</title>',
        '<desc id="desc">Resumen agregado R1 a R5 a nivel de participante para cambios de RMS, frecuencia mediana y cruces sostenidos de umbrales robustos; no clasifica fatiga.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 38, "Comparación de ramas y umbrales de señal", 21, "#17324D"),
        _svg_text(42, 64, "R1–R5 · una mediana por participante · n independiente máximo = 15 · R0/R0_1 quedan solo en QC", 13, "#455A64"),
    ]
    panels = [
        (110, 285, "Cambio mediano de RMS, temprano→tardío (%)", "median_rms_change_pct", "ic95_rms_change_pct_lower", "ic95_rms_change_pct_upper", "#0072B2"),
        (350, 525, "Cambio mediano de frecuencia mediana (Hz)", "median_mdf_change_hz", "ic95_mdf_change_hz_lower", "ic95_mdf_change_hz_upper", "#D55E00"),
    ]
    for y_top, y_bottom, title, value_key, low_key, high_key, color in panels:
        content.append(_svg_text(42, y_top - 18, title, 15, "#17324D"))
        values = [float(rows[b][value_key]) for b in branches if rows.get(b, {}).get(value_key) is not None]
        if not values:
            continue
        bound = max(1.0, max(abs(min(values)), abs(max(values))) * 1.25)
        ymin, ymax = -bound, bound
        def y_map(v):
            return y_bottom - (float(v) - ymin) / (ymax - ymin) * (y_bottom - y_top)
        for tick in range(5):
            val = ymin + (ymax - ymin) * tick / 4
            y = y_map(val)
            content.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#E4E9ED"/>')
            content.append(_svg_text(x0 - 12, y + 4, f"{val:.1f}", 11, "#52616B", "end"))
        content.append(f'<line x1="{x0}" y1="{y_map(0):.1f}" x2="{x1}" y2="{y_map(0):.1f}" stroke="#53636F" stroke-width="1.2"/>')
        for i, branch in enumerate(branches):
            row = rows[branch]
            value = row.get(value_key)
            if value is None:
                continue
            y = y_map(float(value))
            lower, upper = row.get(low_key), row.get(high_key)
            if lower is not None and upper is not None:
                content.append(f'<line x1="{centers[i]}" y1="{y_map(float(lower)):.1f}" x2="{centers[i]}" y2="{y_map(float(upper)):.1f}" stroke="{color}" stroke-width="3"/>')
                content.append(f'<line x1="{centers[i]-7}" y1="{y_map(float(lower)):.1f}" x2="{centers[i]+7}" y2="{y_map(float(lower)):.1f}" stroke="{color}"/>')
                content.append(f'<line x1="{centers[i]-7}" y1="{y_map(float(upper)):.1f}" x2="{centers[i]+7}" y2="{y_map(float(upper)):.1f}" stroke="{color}"/>')
            content.append(f'<circle cx="{centers[i]}" cy="{y:.1f}" r="6" fill="{color}"/>')
            content.append(_svg_text(centers[i], y_bottom + 20, f"{value:.2f}", 11, "#263238", "middle"))
    content.append(_svg_text(42, 580, "Participantes con cruce sostenido en ≥1 registro R1–R5", 15, "#17324D"))
    bar_top, bar_height = 608, 55
    for i, branch in enumerate(branches):
        row = rows[branch]
        n = int(row.get("n_participants", 0))
        freq = int(row.get("participants_sustained_mdf_drop", 0))
        both = int(row.get("participants_coincident_drop", 0))
        x = centers[i]
        scale = bar_height / max(n, 1)
        content.append(f'<rect x="{x-35}" y="{bar_top}" width="24" height="{freq*scale:.1f}" fill="#E69F00"/>')
        content.append(f'<rect x="{x+3}" y="{bar_top}" width="24" height="{both*scale:.1f}" fill="#009E73"/>')
        content.append(_svg_text(x, 680, f"{freq}/{n}  ·  {both}/{n}", 10, "#263238", "middle"))
        content.append(_svg_text(x, 701, BRANCH_LABELS[branch], 10, "#263238", "middle"))
    content.append('<rect x="900" y="582" width="14" height="10" fill="#E69F00"/>')
    content.append(_svg_text(919, 591, "MDF bajo umbral", 10, "#263238"))
    content.append('<rect x="1054" y="582" width="14" height="10" fill="#009E73"/>')
    content.append(_svg_text(1073, 591, "MDF+RMS simultáneos", 10, "#263238"))
    content.append(_svg_text(42, 735, "Umbral: mediana de la primera tercera parte válida − 1,96×1,4826×MAD; ≥3 ventanas de 1,024 muestras. IC95% bootstrap por participante.", 11, "#455A64"))
    content.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def _public_peak_svg(path: Path, overall_rows: list[dict]) -> None:
    width, height = 1240, 750
    x0, x1 = 190, 1120
    rows = {row["filter_branch"]: row for row in overall_rows}
    branches = list(BRANCHES)
    centers = [195 + i * 205 for i in range(len(branches))]
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Amplitud máxima y frecuencia del pico espectral por filtro</title>',
        '<desc id="desc">Medianas por participante de la mayor amplitud absoluta de muestra y la frecuencia del pico de mayor potencia en ventanas válidas, R1 a R5.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(42, 38, "Picos de amplitud y frecuencia por rama", 21, "#17324D"),
        _svg_text(42, 63, "R1–R5 · medianas calculadas por participante primero · unidades del canal guardado: ADC · pico espectral en banda 20–450 Hz", 13, "#455A64"),
    ]
    panels = [
        (104, 232, "Máximo |ADC| de muestra dentro de máscara común", "median_peak_abs_amplitude_adc", "ic95_peak_abs_amplitude_adc_lower", "ic95_peak_abs_amplitude_adc_upper", "#0072B2"),
        (287, 415, "Amplitud sinusoidal estimada en la ventana del punto máximo (ADC)", "median_peak_point_spectral_amplitude_adc", "ic95_peak_point_spectral_amplitude_adc_lower", "ic95_peak_point_spectral_amplitude_adc_upper", "#009E73"),
        (470, 598, "Frecuencia pico en la ventana del punto máximo (Hz)", "median_peak_point_spectral_frequency_hz", "ic95_peak_point_spectral_frequency_hz_lower", "ic95_peak_point_spectral_frequency_hz_upper", "#D55E00"),
    ]
    for y_top, y_bottom, title, val_key, low_key, high_key, color in panels:
        content.append(_svg_text(42, y_top - 17, title, 15, "#17324D"))
        values = [float(rows[b][val_key]) for b in branches if rows.get(b, {}).get(val_key) is not None]
        if not values:
            continue
        if "amplitude" in val_key:
            ymin = 0.0
            ymax = max(1.0, max(float(rows[b].get(high_key) or rows[b][val_key]) for b in branches if rows.get(b, {}).get(val_key) is not None) * 1.15)
        else:
            ymin = max(LOW_HZ, min(values) - 12.0)
            ymax = min(HIGH_HZ, max(values) + 12.0)
            if ymax <= ymin:
                ymin, ymax = LOW_HZ, HIGH_HZ
        def y_map(value):
            return y_bottom - (float(value) - ymin) / (ymax - ymin) * (y_bottom - y_top)
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            y = y_map(value)
            content.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#E4E9ED"/>')
            content.append(_svg_text(x0 - 12, y + 4, f"{value:.1f}", 11, "#52616B", "end"))
        for i, branch in enumerate(branches):
            row = rows[branch]
            value = row.get(val_key)
            if value is None:
                continue
            lower, upper = row.get(low_key), row.get(high_key)
            if lower is not None and upper is not None:
                content.append(f'<line x1="{centers[i]}" y1="{y_map(float(lower)):.1f}" x2="{centers[i]}" y2="{y_map(float(upper)):.1f}" stroke="{color}" stroke-width="3"/>')
            content.append(f'<circle cx="{centers[i]}" cy="{y_map(float(value)):.1f}" r="6" fill="{color}"/>')
            content.append(_svg_text(centers[i], y_bottom + 19, f"{float(value):.2f}", 11, "#263238", "middle"))
            content.append(_svg_text(centers[i], y_bottom + 36, BRANCH_LABELS[branch], 10, "#263238", "middle"))
    content.append(_svg_text(42, 720, "El máximo de muestra y el pico espectral son medidas distintas; las amplitudes son unidades ADC sin calibración física. No equivalen por sí solas a fatiga.", 11, "#455A64"))
    content.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def analyze_dataset(private_root: Path, output_root: Path, public_root: Path) -> dict:
    started = time.perf_counter()
    local_data = private_root.parent
    manifest_path = local_data / "manifest_pseudonymized.csv"
    signal_root = local_data / "pseudonymized_csv"
    filter_receipt_path = private_root / "receipt_filter_sensitivity_v1.json"
    if not manifest_path.is_file() or not filter_receipt_path.is_file():
        raise FileNotFoundError("private pseudonymized manifest or filter receipt is unavailable")
    filter_receipt = json.loads(filter_receipt_path.read_text(encoding="utf-8"))
    if filter_receipt.get("analysis") != "filter_sensitivity_v1":
        raise ValueError("unexpected filter receipt version")
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != filter_receipt.get("manifest_sha256"):
        raise ValueError("manifest hash differs from the verified filter receipt")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    manifest_keys = [(row["participant_id"], row["recording_id"], row["leg"]) for row in manifest_rows]
    if len(manifest_rows) != 140 or len(set(manifest_keys)) != len(manifest_rows):
        raise ValueError("manifest record count or composite keys do not match the validated inventory")
    receipt_rows = filter_receipt.get("records_private", [])
    receipt_by_key = {(row["participant_id"], row["recording_id"], row["leg"]): row for row in receipt_rows}
    if set(manifest_keys) != set(receipt_by_key):
        raise ValueError("manifest and filter receipt composite keys differ")

    npz_by_key: dict[tuple[str, str, str], Path] = {}
    for filt in receipt_rows:
        key = (filt["participant_id"], filt["recording_id"], filt["leg"])
        npz_path = resolve_recorded_npz_path(
            private_root,
            filt.get("output_path", ""),
            field_name="output_path",
        )
        if _sha256(npz_path) != filt.get("output_sha256"):
            raise ValueError("verified filtered array archive is missing or changed")
        npz_by_key[key] = npz_path

    output_root.mkdir(parents=True, exist_ok=True)
    figures_root = output_root / "figures_private"
    figures_root.mkdir(parents=True, exist_ok=True)
    per_record_rows: list[dict] = []
    per_window_rows: list[dict] = []
    event_rows: list[dict] = []
    gallery_items: list[tuple[str, str, dict]] = []
    sample_total = 0
    total_windows = 0
    branch_count = defaultdict(int)
    for manifest_row in sorted(manifest_rows, key=lambda row: (row["participant_id"], row["recording_id"], row["leg"])):
        key = (manifest_row["participant_id"], manifest_row["recording_id"], manifest_row["leg"])
        source_rel = Path(manifest_row["relative_path"])
        source_path = (signal_root / source_rel).resolve()
        if not source_path.is_relative_to(signal_root.resolve()):
            raise ValueError("manifest source path escapes private pseudonymized data root")
        if _sha256(source_path) != manifest_row["sha256_uncompressed"]:
            raise ValueError("pseudonymized input hash changed since filter analysis")
        filt = receipt_by_key[key]
        npz_path = npz_by_key[key]
        times_csv, raw_csv = _read_signal_csv(source_path)
        with np.load(npz_path, allow_pickle=False) as arrays:
            times = arrays["times_s"].astype(np.float64, copy=False)
            raw = arrays["adc_original"].astype(np.float64, copy=False)
            common_mask = arrays["common_comparison_mask"].astype(bool, copy=False)
            if not np.array_equal(times_csv, times) or not np.array_equal(raw_csv, raw):
                raise ValueError("saved original signal or timestamps differ from the pseudonymized CSV")
            if common_mask.shape != raw.shape or not common_mask.any():
                raise ValueError("common comparison mask is empty or mismatched")
            signals = {branch: (raw.copy() if branch == "control" else arrays[branch].astype(np.float64, copy=True)) for branch in BRANCHES}
            if any(values.shape != raw.shape or not np.isfinite(values).all() for values in signals.values()):
                raise ValueError("signal branch shape mismatch or nonfinite samples")
        sample_total += int(raw.size)
        full_raw_peak = maximum_absolute_point(times, raw, np.ones(raw.size, dtype=bool))
        windows_by_branch: dict[str, list[dict]] = {}
        thresholds_by_branch: dict[str, dict] = {}
        events_by_branch: dict[str, dict] = {}
        for branch in BRANCHES:
            windows = iter_valid_windows(times, signals[branch], common_mask)
            windows_by_branch[branch] = windows
            total_windows += len(windows)
            branch_count[branch] += 1
            rms_threshold = robust_lower_threshold([float(row["rms_adc"]) for row in windows])
            mdf_threshold = robust_lower_threshold([float(row["median_frequency_hz"]) for row in windows])
            thresholds_by_branch[branch] = {"rms": rms_threshold, "mdf": mdf_threshold}
            crossings = _event_crossings(windows, rms_threshold, mdf_threshold)
            events_by_branch[branch] = crossings
            peak_valid = maximum_absolute_point(times, signals[branch], common_mask)
            if not windows:
                status = "no_complete_valid_windows"
            elif rms_threshold["status"] != "estimated" and mdf_threshold["status"] != "estimated":
                status = "threshold_not_estimable"
            else:
                status = "analyzed_exploratory"
            if crossings["both"]:
                category = "sustained_amplitude_and_frequency_drop_candidate"
            elif crossings["frequency"]:
                category = "sustained_median_frequency_drop_candidate"
            elif crossings["amplitude"]:
                category = "sustained_amplitude_drop_candidate"
            elif status != "analyzed_exploratory":
                category = "not_classifiable_threshold"
            else:
                category = "no_sustained_lower_threshold_crossing"
            early = [window for window in windows if float(window["time_fraction"]) < BASELINE_FRACTION]
            late = [window for window in windows if float(window["time_fraction"]) >= 1.0 - BASELINE_FRACTION]
            rms_early = _median(window["rms_adc"] for window in early)
            rms_late = _median(window["rms_adc"] for window in late)
            mdf_early = _median(window["median_frequency_hz"] for window in early)
            mdf_late = _median(window["median_frequency_hz"] for window in late)
            dominant = max(windows, key=lambda row: float(row["peak_power_adc2"]), default=None)
            peak_point_window = next((window for window in windows if peak_valid["sample_index"] is not None and int(window["start_index"]) <= int(peak_valid["sample_index"]) < int(window["end_index"])), None)
            record_summary = {
                "participant_id": manifest_row["participant_id"],
                "session_id": manifest_row["session_id"],
                "trial_id": manifest_row["trial_id"],
                "recording_id": manifest_row["recording_id"],
                "leg": manifest_row["leg"],
                "set_index": manifest_row["set_index"],
                "record_type": filt["record_type"],
                "branch": branch,
                "analysis_status": status,
                "signal_mode": "unknown_stored_channel",
                "n_samples": int(raw.size),
                "common_valid_samples": int(common_mask.sum()),
                "n_valid_windows": len(windows),
                "n_baseline_windows_rms": int(rms_threshold["n_baseline"]),
                "n_baseline_windows_mdf": int(mdf_threshold["n_baseline"]),
                "rms_threshold_status": rms_threshold["status"],
                "rms_early_median_adc": rms_threshold["baseline_median"],
                "rms_robust_scale_adc": rms_threshold["robust_scale"],
                "rms_lower_threshold_adc": rms_threshold["lower_threshold"],
                "mdf_threshold_status": mdf_threshold["status"],
                "mdf_early_median_hz": mdf_threshold["baseline_median"],
                "mdf_robust_scale_hz": mdf_threshold["robust_scale"],
                "mdf_lower_threshold_hz": mdf_threshold["lower_threshold"],
                "rms_late_median_adc": rms_late,
                "rms_early_to_late_change_pct": _change_pct(rms_early, rms_late),
                "mdf_late_median_hz": mdf_late,
                "mdf_early_to_late_change_hz": float(mdf_late - mdf_early) if mdf_late is not None and mdf_early is not None else None,
                "mdf_early_to_late_change_pct": _change_pct(mdf_early, mdf_late),
                "peak_abs_amplitude_adc_full_raw": full_raw_peak["absolute_amplitude"],
                "peak_time_s_full_raw": full_raw_peak["time_s"],
                "peak_signed_value_adc_full_raw": full_raw_peak["value"],
                "peak_abs_amplitude_adc_common_mask": peak_valid["absolute_amplitude"],
                "peak_signed_value_adc_common_mask": peak_valid["value"],
                "peak_time_s_common_mask": peak_valid["time_s"],
                "peak_sample_index_common_mask": peak_valid["sample_index"],
                "peak_point_spectral_frequency_hz": float(peak_point_window["peak_frequency_hz"]) if peak_point_window else None,
                "peak_point_spectral_amplitude_adc": float(peak_point_window["peak_amplitude_adc"]) if peak_point_window else None,
                "peak_point_spectral_power_adc2": float(peak_point_window["peak_power_adc2"]) if peak_point_window else None,
                "peak_point_window_time_s": float(peak_point_window["time_s"]) if peak_point_window else None,
                "dominant_window_peak_frequency_hz": float(dominant["peak_frequency_hz"]) if dominant else None,
                "dominant_window_peak_power_adc2": float(dominant["peak_power_adc2"]) if dominant else None,
                "dominant_window_peak_amplitude_adc": float(dominant["peak_amplitude_adc"]) if dominant else None,
                "dominant_window_peak_time_s": float(dominant["time_s"]) if dominant else None,
                "rms_crossing_count": len(crossings["amplitude"]),
                "mdf_crossing_count": len(crossings["frequency"]),
                "coincident_crossing_count": len(crossings["both"]),
                "first_rms_crossing_time_s": float(windows[crossings["amplitude"][0]["start_window_index"]]["time_s"]) if crossings["amplitude"] else None,
                "first_mdf_crossing_time_s": float(windows[crossings["frequency"][0]["start_window_index"]]["time_s"]) if crossings["frequency"] else None,
                "first_coincident_crossing_time_s": float(windows[crossings["both"][0]["start_window_index"]]["time_s"]) if crossings["both"] else None,
                "signal_change_category": category,
            }
            per_record_rows.append(record_summary)

            for wi, window in enumerate(windows):
                rms_limit = rms_threshold.get("lower_threshold")
                mdf_limit = mdf_threshold.get("lower_threshold")
                rms_below = rms_limit is not None and float(window["rms_adc"]) < float(rms_limit)
                mdf_below = mdf_limit is not None and float(window["median_frequency_hz"]) < float(mdf_limit)
                per_window_rows.append({
                    "participant_id": manifest_row["participant_id"],
                    "recording_id": manifest_row["recording_id"],
                    "leg": manifest_row["leg"],
                    "set_index": manifest_row["set_index"],
                    "record_type": filt["record_type"],
                    "branch": branch,
                    "window_index": wi,
                    **window,
                    "rms_lower_threshold_adc": rms_limit,
                    "mdf_lower_threshold_hz": mdf_limit,
                    "rms_below_threshold": rms_below,
                    "mdf_below_threshold": mdf_below,
                    "both_below_threshold": rms_below and mdf_below,
                    "persistence_rule_windows": MIN_CONSECUTIVE_WINDOWS,
                    "interpretation": "exploratory_signal_change_not_fatigue_label",
                })
            for metric_name, event_group, threshold_obj in (
                ("rms_amplitude", "amplitude", rms_threshold),
                ("median_frequency", "frequency", mdf_threshold),
                ("coincident_rms_and_frequency", "both", mdf_threshold),
            ):
                for event_index, event in enumerate(crossings[event_group], start=1):
                    event_rows.append({
                        "participant_id": manifest_row["participant_id"],
                        "recording_id": manifest_row["recording_id"],
                        "leg": manifest_row["leg"],
                        "set_index": manifest_row["set_index"],
                        "record_type": filt["record_type"],
                        "branch": branch,
                        "metric": metric_name,
                        "event_index": event_index,
                        "segment_id": event["segment_id"],
                        "start_window_index": event["start_window_index"],
                        "event_start_time_s": windows[event["start_window_index"]]["time_s"],
                        "duration_windows": event["duration_windows"],
                        "window_duration_s_approx": float(event["duration_windows"] * WINDOW_SAMPLES / windows[event["start_window_index"]]["fs_hz"]),
                        "lower_threshold": threshold_obj.get("lower_threshold"),
                        "rms_lower_threshold_adc": rms_threshold.get("lower_threshold"),
                        "mdf_lower_threshold_hz": mdf_threshold.get("lower_threshold"),
                        "threshold_unit": "ADC" if metric_name == "rms_amplitude" else "Hz",
                        "interpretation": "candidate_signal_change_not_fatigue_ground_truth",
                    })

        figure_name = "_".join(_safe_token(manifest_row[field]) for field in ("participant_id", "recording_id", "leg")) + ".svg"
        figure_path = figures_root / figure_name
        figure_record = {**manifest_row, "record_type": filt["record_type"]}
        _record_figure(figure_path, figure_record, times, raw, signals, common_mask, windows_by_branch, thresholds_by_branch, full_raw_peak)
        title = f'{manifest_row["participant_id"]} · {manifest_row["recording_id"]} · {manifest_row["leg"]} · serie {manifest_row["set_index"]}'
        gallery_summary = {"signal_change_category": "ver detalle por rama en las tablas privadas"}
        gallery_items.append((title, f"figures_private/{figure_name}", gallery_summary))

    record_fields = list(per_record_rows[0].keys()) if per_record_rows else []
    window_fields = list(per_window_rows[0].keys()) if per_window_rows else []
    event_fields = list(event_rows[0].keys()) if event_rows else ["participant_id", "recording_id", "leg", "set_index", "record_type", "branch", "metric", "event_index", "segment_id", "start_window_index", "event_start_time_s", "duration_windows", "window_duration_s_approx", "lower_threshold", "rms_lower_threshold_adc", "mdf_lower_threshold_hz", "threshold_unit", "interpretation"]
    record_csv = output_root / "umbrales_por_registro_PRIVADO.csv"
    window_csv = output_root / "rasgos_umbral_por_ventana_PRIVADO.csv"
    event_csv = output_root / "cruces_sostenidos_PRIVADO.csv"
    _csv_write(record_csv, per_record_rows, record_fields)
    _csv_write(window_csv, per_window_rows, window_fields)
    _csv_write(event_csv, event_rows, event_fields)

    html_rows = []
    for title, figure_rel, summary in gallery_items:
        html_rows.append(
            "<tr><td>" + html.escape(title) + "</td><td>" + html.escape(str(summary["signal_change_category"])) +
            "</td><td><a href=\"" + html.escape(figure_rel) + "\">Abrir señal y umbrales</a></td></tr>"
        )
    index_html = output_root / "indice_graficas_PRIVADO.html"
    index_html.write_text(
        "<!doctype html><html lang=\"es\"><meta charset=\"utf-8\"><title>Auditoría local de señales EMG</title>"
        "<body><h1>Señal cruda, filtros y umbrales exploratorios</h1><p>Archivo local privado. Las señales por registro no se publican en el proyecto. "
        "Los cruces reflejan cambio de señal, no fatiga validada.</p><table><thead><tr><th>Registro</th><th>Resultado</th><th>Gráfica</th></tr></thead><tbody>"
        + "".join(html_rows) + "</tbody></table></body></html>", encoding="utf-8"
    )

    series_rows, qc_rows, overall_rows, subject_rows = _summarize_threshold_rows(per_record_rows)
    public_root.mkdir(parents=True, exist_ok=True)
    series_csv = public_root / "UMBRALES_SENAL_POR_SERIE_LADO_AGREGADO.csv"
    qc_csv = public_root / "UMBRALES_SENAL_QC_PRELIMINAR_AGREGADO.csv"
    overall_csv = public_root / "UMBRALES_SENAL_AGREGADA.csv"
    series_fields = ["recording_id", "leg", "filter_branch", "n_records", "n_participants", "median_rms_early_adc", "median_rms_late_adc", "median_rms_change_pct", "median_mdf_early_hz", "median_mdf_late_hz", "median_mdf_change_hz", "median_peak_abs_amplitude_adc", "median_peak_point_spectral_amplitude_adc", "median_peak_point_spectral_frequency_hz", "median_dominant_peak_frequency_hz", "median_dominant_peak_power_adc2", "records_with_sustained_rms_drop", "records_with_sustained_mdf_drop", "records_with_coincident_drop", "note"]
    qc_fields = ["record_type", "filter_branch", "n_records", "n_participants", "median_peak_abs_amplitude_adc", "median_peak_window_frequency_hz", "median_peak_window_power_adc2", "median_peak_point_spectral_amplitude_adc", "median_peak_point_spectral_frequency_hz", "records_with_sustained_rms_drop", "records_with_sustained_mdf_drop", "records_with_coincident_drop", "note"]
    overall_fields = ["filter_branch", "n_participants", "median_rms_change_pct", "ic95_rms_change_pct_lower", "ic95_rms_change_pct_upper", "median_mdf_change_hz", "ic95_mdf_change_hz_lower", "ic95_mdf_change_hz_upper", "median_peak_abs_amplitude_adc", "ic95_peak_abs_amplitude_adc_lower", "ic95_peak_abs_amplitude_adc_upper", "median_peak_point_spectral_amplitude_adc", "ic95_peak_point_spectral_amplitude_adc_lower", "ic95_peak_point_spectral_amplitude_adc_upper", "median_peak_point_spectral_frequency_hz", "ic95_peak_point_spectral_frequency_hz_lower", "ic95_peak_point_spectral_frequency_hz_upper", "median_dominant_peak_frequency_hz", "ic95_dominant_peak_frequency_hz_lower", "ic95_dominant_peak_frequency_hz_upper", "participants_sustained_rms_drop", "participants_sustained_mdf_drop", "participants_coincident_drop", "note"]
    _csv_write(series_csv, series_rows, series_fields)
    _csv_write(qc_csv, qc_rows, qc_fields)
    _csv_write(overall_csv, overall_rows, overall_fields)
    figure_public = public_root.parent.parent / "articulo" / "figuras" / "umbrales_senal_comparacion.svg"
    _public_summary_svg(figure_public, overall_rows)
    figure_peaks = public_root.parent.parent / "articulo" / "figuras" / "picos_senal_comparacion.svg"
    _public_peak_svg(figure_peaks, overall_rows)

    outputs = [record_csv, window_csv, event_csv, index_html, series_csv, qc_csv, overall_csv, figure_public, figure_peaks]
    outputs.extend(figures_root.glob("*.svg"))
    output_hashes = {path: _sha256(path) for path in outputs}
    receipt = {
        "analysis": "signal_threshold_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "input_manifest_sha256": manifest_sha,
        "filter_receipt_sha256": _sha256(filter_receipt_path),
        "records": len(per_record_rows) // len(BRANCHES),
        "participants": len({row["participant_id"] for row in per_record_rows}),
        "branch_record_summaries": len(per_record_rows),
        "samples": sample_total,
        "window_features": total_windows,
        "series_records_R1_R5": sum(row["record_type"] == "series_R1_R5" for row in per_record_rows) // len(BRANCHES),
        "preliminary_records_R0_R0_1_QC_only": sum(row["record_type"] != "series_R1_R5" for row in per_record_rows) // len(BRANCHES),
        "filter_branches": list(BRANCHES),
        "window_samples": WINDOW_SAMPLES,
        "frequency_band_hz": [LOW_HZ, HIGH_HZ],
        "gap_threshold_s": GAP_THRESHOLD_S,
        "threshold_rule": "median of first one-third valid windows - 1.96 * 1.4826 * MAD; minimum 5 baseline windows; zero-MAD not estimable",
        "crossing_rule": "strictly below threshold for at least three consecutive non-overlapping valid windows in one contiguous segment",
        "fatigue_ground_truth_available": False,
        "fatigue_score_or_label_generated": False,
        "notch_applied": False,
        "synthetic_noise_applied_to_study_data": False,
        "signal_mode": "unknown_stored_channel_ADC_Value",
        "private_outputs_sha256": {str(path): digest for path, digest in output_hashes.items() if path.resolve().is_relative_to(output_root.resolve())},
        "public_aggregate_outputs_sha256": {str(path): digest for path, digest in output_hashes.items() if not path.resolve().is_relative_to(output_root.resolve())},
        "candidate_event_rows_private": len(event_rows),
        "candidate_event_counts_by_metric_private": {metric: sum(row["metric"] == metric for row in event_rows) for metric in ("rms_amplitude", "median_frequency", "coincident_rms_and_frequency")},
    }
    receipt_path = output_root / "recibo_umbrales_senal_v1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    result = analyze_dataset(args.private_root, args.output_root, args.aggregate_root)
    summary = {key: value for key, value in result.items() if not key.endswith("outputs_sha256")}
    summary["private_outputs_hashed"] = len(result["private_outputs_sha256"])
    summary["public_outputs_hashed"] = len(result["public_aggregate_outputs_sha256"])
    summary["receipt_path"] = str(args.output_root / "recibo_umbrales_senal_v1.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
