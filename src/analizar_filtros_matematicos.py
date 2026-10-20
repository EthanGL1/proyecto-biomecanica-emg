"""Audit digital-filter transfer functions and local EMG filter sensitivity.

This program writes cohort-level summaries and anonymous figures only. The
receipt and per-record NPZ files remain in the user's local application data.
It does not choose a physiological filter or label fatigue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from release_paths import resolve_recorded_npz_path, validate_data_paths
from PIL import Image, ImageDraw, ImageFont

from comparar_filtros_emg import (
    design_butterworth_sos,
    design_fir_bandpass,
    design_notch_sos,
    split_contiguous_runs,
)


APPLIED_BRANCHES = ("hp20", "hp40", "bp20_400", "fir_bp20_400")
SUMMARY_METRICS = (
    "rms_adc",
    "peak_to_peak_adc",
    "power_20_60_adc2",
    "power_60_150_adc2",
    "power_150_400_adc2",
    "prominence_50_db",
    "prominence_60_db",
)
PERCENT_METRICS = frozenset(SUMMARY_METRICS[:5])
BRANCH_COLORS = {
    "control": "#626b73",
    "hp20": "#0072b2",
    "hp40": "#d55e00",
    "bp20_400": "#009e73",
    "fir_bp20_400": "#cc79a7",
    "notch50_candidate": "#7b61a8",
    "notch60_candidate": "#a05195",
}


def design_filter_bank(fs_hz: float) -> dict[str, dict]:
    """Build the exact 1-pass coefficient designs used by the local audit."""
    fs = float(fs_hz)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs_hz must be finite and positive")
    designs = {
        "control": {"kind": "control", "fs_hz": fs, "order": 0},
        "hp20": {
            "kind": "iir_sos", "fs_hz": fs, "sos": design_butterworth_sos(fs, "highpass", (20.0,), order=4),
            "prototype_order": 4, "digital_order": 4, "cutoffs_hz": (20.0,), "implemented_zero_phase": True,
        },
        "hp40": {
            "kind": "iir_sos", "fs_hz": fs, "sos": design_butterworth_sos(fs, "highpass", (40.0,), order=4),
            "prototype_order": 4, "digital_order": 4, "cutoffs_hz": (40.0,), "implemented_zero_phase": True,
        },
        "bp20_400": {
            "kind": "iir_sos", "fs_hz": fs, "sos": design_butterworth_sos(fs, "bandpass", (20.0, 400.0), order=4),
            "prototype_order": 4, "digital_order": 8, "cutoffs_hz": (20.0, 400.0), "implemented_zero_phase": True,
        },
        "fir_bp20_400": {
            "kind": "fir", "fs_hz": fs, "coefficients": design_fir_bandpass(fs, 20.0, 400.0, taps=101),
            "order": 100, "taps": 101, "cutoffs_hz": (20.0, 400.0), "group_delay_samples": 50,
            "implemented_zero_phase": True,
        },
        "notch50_candidate": {
            "kind": "iir_sos", "fs_hz": fs, "sos": design_notch_sos(fs, 50.0, q=30.0),
            "order": 2, "cutoffs_hz": (50.0,), "q": 30.0, "implemented_zero_phase": False,
        },
        "notch60_candidate": {
            "kind": "iir_sos", "fs_hz": fs, "sos": design_notch_sos(fs, 60.0, q=30.0),
            "order": 2, "cutoffs_hz": (60.0,), "q": 30.0, "implemented_zero_phase": False,
        },
    }
    return designs


def frequency_response(design: dict, frequencies_hz: np.ndarray, zero_phase: bool | None = None) -> np.ndarray:
    """Evaluate H(exp(jw)); optionally apply the magnitude/phase of the offline implementation."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    fs = float(design["fs_hz"])
    if frequencies.ndim != 1 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be a finite one-dimensional array")
    if np.any(frequencies < 0) or np.any(frequencies > fs / 2.0):
        raise ValueError("frequencies must lie between DC and Nyquist")
    z1 = np.exp(-2j * np.pi * frequencies / fs)
    if design["kind"] == "iir_sos":
        response = np.ones(frequencies.size, dtype=complex)
        for section in np.asarray(design["sos"], dtype=float):
            numerator = section[0] + section[1] * z1 + section[2] * z1**2
            denominator = section[3] + section[4] * z1 + section[5] * z1**2
            response *= numerator / denominator
        if zero_phase is True:
            return np.abs(response) ** 2 + 0j
        return response
    if design["kind"] == "fir":
        n = np.arange(len(design["coefficients"]), dtype=float)
        response = np.exp(-2j * np.pi * np.outer(frequencies / fs, n)) @ np.asarray(design["coefficients"])
        if zero_phase is True:
            delay = int(design["group_delay_samples"])
            response *= np.exp(2j * np.pi * frequencies * delay / fs)
        return response
    if design["kind"] == "control":
        return np.ones(frequencies.size, dtype=complex)
    raise ValueError(f"unsupported design kind: {design['kind']}")


def summarize_filter_effects(
    records: Iterable[dict],
    branches: tuple[str, ...] = APPLIED_BRANCHES,
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> list[dict]:
    """Summarize joint and side-specific effects, weighting each participant once."""
    laterality_levels = ("all", "left", "right")
    participant_values: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    recording_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for record in records:
        if record.get("record_type") != "series_R1_R5":
            continue
        participant = str(record.get("participant_id", ""))
        if not participant:
            raise ValueError("primary receipt records must carry an internal participant group")
        raw_laterality = str(record.get("leg", "")).strip().lower()
        record_lateralities = ["all"]
        if raw_laterality in ("left", "right"):
            record_lateralities.append(raw_laterality)
        for laterality in record_lateralities:
          for branch in branches:
            key = (branch, laterality, participant)
            values = participant_values.setdefault(key, {metric: [] for metric in metrics})
            filtered = record.get("branches", {}).get(branch, {}).get("metrics_common", {})
            control = record.get("branches", {}).get("control", {}).get("metrics_common", {})
            for metric in metrics:
                value, reference = filtered.get(metric), control.get(metric)
                if value is None or reference is None or not math.isfinite(float(value)) or not math.isfinite(float(reference)):
                    continue
                if metric in PERCENT_METRICS:
                    if float(reference) == 0:
                        continue
                    delta = 100.0 * (float(value) - float(reference)) / float(reference)
                else:
                    delta = float(value) - float(reference)
                values[metric].append(delta)
                counts = recording_counts.setdefault(key, {name: 0 for name in metrics})
                counts[metric] += 1

    rows: list[dict] = []
    for laterality in laterality_levels:
      for branch in branches:
        for metric in metrics:
            subject_level = [float(np.median(values[metric]))
                for (candidate, side, _), values in participant_values.items()
                if candidate == branch and side == laterality and values[metric]]
            if not subject_level:
                continue
            unit = "% change" if metric in PERCENT_METRICS else "dB change"
            total_recordings = sum(counts.get(metric, 0) for (candidate, side, _), counts in recording_counts.items()
                if candidate == branch and side == laterality)
            median_value = float(np.median(subject_level))
            q1_value = float(np.quantile(subject_level, 0.25))
            q3_value = float(np.quantile(subject_level, 0.75))
            rows.append({
                "laterality": laterality,
                "filter_branch": branch,
                "metric": metric,
                "unit": unit,
                "n_participants": len(subject_level),
                "recordings_n": int(total_recordings),
                "median_change": median_value,
                "q1_change": q1_value,
                "q3_change": q3_value,
                # Preserve established aggregate field names while providing unit-neutral fields above.
                "median_change_pct": median_value,
                "q1_change_pct": q1_value,
                "q3_change_pct": q3_value,
                "aggregation": "record median within participant, then participant median and IQR",
            })
    return rows


def summarize_paired_laterality(
    records: Iterable[dict],
    branches: tuple[str, ...] = APPLIED_BRANCHES,
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> list[dict]:
    """Summarize left-minus-right contrasts using only within-student pairs."""
    values: dict[tuple[str, str, str, str], list[float]] = {}
    counts: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        if record.get("record_type") != "series_R1_R5":
            continue
        participant = str(record.get("participant_id", ""))
        side = str(record.get("leg", "")).strip().lower()
        if not participant:
            raise ValueError("primary receipt records must carry an internal participant group")
        if side not in ("left", "right"):
            continue
        for branch in branches:
            filtered = record.get("branches", {}).get(branch, {}).get("metrics_common", {})
            control = record.get("branches", {}).get("control", {}).get("metrics_common", {})
            for metric in metrics:
                value, reference = filtered.get(metric), control.get(metric)
                if value is None or reference is None or not math.isfinite(float(value)) or not math.isfinite(float(reference)):
                    continue
                if metric in PERCENT_METRICS:
                    if float(reference) == 0:
                        continue
                    delta = 100.0 * (float(value) - float(reference)) / float(reference)
                else:
                    delta = float(value) - float(reference)
                values.setdefault((branch, participant, side, metric), []).append(delta)
                count_key = (branch, participant, side, metric)
                counts[count_key] = counts.get(count_key, 0) + 1

    rows: list[dict] = []
    for branch in branches:
        for metric in metrics:
            paired_differences = []
            left_recordings = right_recordings = 0
            for participant in sorted({key[1] for key in values if key[0] == branch}):
                left_values = values.get((branch, participant, "left", metric), [])
                right_values = values.get((branch, participant, "right", metric), [])
                if not left_values or not right_values:
                    continue
                paired_differences.append(float(np.median(left_values) - np.median(right_values)))
                left_recordings += counts.get((branch, participant, "left", metric), 0)
                right_recordings += counts.get((branch, participant, "right", metric), 0)
            if not paired_differences:
                continue
            rows.append({
                "filter_branch": branch,
                "metric": metric,
                "unit": "% change" if metric in PERCENT_METRICS else "dB change",
                "n_paired_participants": len(paired_differences),
                "paired_recordings_left": left_recordings,
                "paired_recordings_right": right_recordings,
                "median_left_minus_right": float(np.median(paired_differences)),
                "q1_left_minus_right": float(np.quantile(paired_differences, 0.25)),
                "q3_left_minus_right": float(np.quantile(paired_differences, 0.75)),
                "aggregation": "within-student median by side; paired left minus right; group median and IQR",
            })
    return rows


def select_representative_record(records: Iterable[dict], case_sha256: str | None = None) -> dict:
    """Select a local R1-R5 recording by median length, without model outcomes."""
    eligible = [record for record in records if record.get("record_type") == "series_R1_R5"]
    if not eligible:
        raise ValueError("no eligible R1-R5 recording is available for the representative plot")
    if case_sha256:
        matches = [record for record in eligible if record.get("output_sha256") == case_sha256]
        if len(matches) != 1:
            raise ValueError("requested archive hash does not identify exactly one eligible recording")
        return matches[0]
    target = float(np.median([int(record["n_samples"]) for record in eligible]))
    return min(
        eligible,
        key=lambda record: (
            abs(int(record["n_samples"]) - target),
            int(record.get("n_runs", 0)),
            str(record.get("recording_id", "")),
            str(record.get("leg", "")),
        ),
    )


def coefficient_rows(designs: dict[str, dict]) -> list[dict]:
    """Return a long-form coefficient table suitable for a public supplement."""
    rows: list[dict] = []
    for branch, design in designs.items():
        if design["kind"] == "iir_sos":
            for section_index, section in enumerate(np.asarray(design["sos"], dtype=float), start=1):
                for index, name in enumerate(("b0", "b1", "b2", "a0", "a1", "a2")):
                    rows.append({
                        "fs_hz": design["fs_hz"], "branch": branch, "section": section_index,
                        "coefficient": name, "index": index, "value": format(float(section[index]), ".17g"),
                    })
        elif design["kind"] == "fir":
            for index, value in enumerate(np.asarray(design["coefficients"], dtype=float)):
                rows.append({
                    "fs_hz": design["fs_hz"], "branch": branch, "section": 1,
                    "coefficient": "b", "index": index, "value": format(float(value), ".17g"),
                })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


class _Canvas:
    """Small deterministic Pillow/SVG canvas; SVG remains editable vector art."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}</style>',
        ]

    def line(self, points: list[tuple[float, float]], color: str = "#202124", width: int = 2, dash: str | None = None):
        if len(points) < 2:
            return
        xy = [(int(round(x)), int(round(y))) for x, y in points]
        self.draw.line(xy, fill=color, width=width, joint="curve")
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>')

    def rect(self, x: float, y: float, width: float, height: float, fill: str | None, stroke: str | None = None):
        pillow_fill = None if fill in (None, "none") else fill
        self.draw.rectangle((int(x), int(y), int(x + width), int(y + height)), fill=pillow_fill, outline=stroke)
        stroke_attr = f' stroke="{stroke}"' if stroke else ""
        svg_fill = "none" if fill in (None, "none") else str(fill)
        self.svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{svg_fill}"{stroke_attr}/>')

    def text(self, x: float, y: float, value: str, size: int = 18, anchor: str = "start", bold: bool = False, color: str = "#202124", rotate: int = 0):
        font = _font(size, bold)
        align = {"start": "la", "middle": "mm", "end": "ra"}[anchor]
        safe = str(value)
        if rotate:
            bounds = self.draw.textbbox((0, 0), safe, font=font, anchor=align)
            tile_width = max(1, bounds[2] - bounds[0] + 8)
            tile_height = max(1, bounds[3] - bounds[1] + 8)
            tile = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((tile_width / 2, tile_height / 2), safe, font=font, fill=color, anchor="mm")
            rotated = tile.rotate(rotate, resample=Image.Resampling.BICUBIC, expand=True)
            self.image.paste(rotated, (round(x - rotated.width / 2), round(y - rotated.height / 2)), rotated)
            self.draw = ImageDraw.Draw(self.image)
        else:
            self.draw.text((int(x), int(y)), safe, font=font, fill=color, anchor=align)
        escaped = html.escape(safe)
        rotate_attr = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
        self.svg.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{"700" if bold else "400"}" text-anchor="{anchor}" fill="{color}"{rotate_attr}>{escaped}</text>')

    def circle(self, x: float, y: float, r: float, fill: str):
        self.draw.ellipse((int(x-r), int(y-r), int(x+r), int(y+r)), fill=fill)
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"/>')

    def save(self, stem: Path):
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(stem.with_suffix(".png"), dpi=(300, 300), optimize=True)
        self.svg.append("</svg>")
        stem.with_suffix(".svg").write_text("\n".join(self.svg), encoding="utf-8")


def _map_x(value: float, low: float, high: float, left: float, right: float, logarithmic: bool = False) -> float:
    if logarithmic:
        value, low, high = math.log10(max(value, 1e-12)), math.log10(low), math.log10(high)
    return left + (value - low) / (high - low) * (right - left)


def _map_y(value: float, low: float, high: float, top: float, bottom: float) -> float:
    return bottom - (value - low) / (high - low) * (bottom - top)


def _draw_axes(
    canvas: _Canvas, box: tuple[int, int, int, int], x_ticks: list[float], y_ticks: list[float],
    x_range: tuple[float, float], y_range: tuple[float, float], x_label: str, y_label: str,
    x_log: bool = False, y_format: str = "g",
):
    left, top, right, bottom = box
    for value in x_ticks:
        x = _map_x(value, *x_range, left, right, logarithmic=x_log)
        canvas.line([(x, top), (x, bottom)], "#e2e5e9", 1)
        canvas.text(x, bottom + 28, f"{value:g}", 15, "middle", color="#495057")
    for value in y_ticks:
        y = _map_y(value, *y_range, top, bottom)
        canvas.line([(left, y), (right, y)], "#e2e5e9", 1)
        canvas.text(left - 12, y, format(value, y_format), 15, "end", color="#495057")
    canvas.line([(left, top), (left, bottom), (right, bottom)], "#454b52", 2)
    canvas.text((left + right) / 2, bottom + 58, x_label, 17, "middle", bold=True)
    canvas.text(left - 75, (top + bottom) / 2, y_label, 17, "middle", bold=True, rotate=-90)


def make_response_figure(designs: dict[str, dict], stem: Path) -> None:
    width, height = 1800, 1460
    canvas = _Canvas(width, height)
    canvas.text(90, 60, "Digital filter response at timestamp-derived fs = 1 kHz", 34, bold=True)
    canvas.text(90, 100, "Designed coefficients; no analog acquisition response is available", 21, color="#586069")
    left, right = 170, 1690
    freq = np.geomspace(1.0, 499.5, 1500)
    top_box, phase_box = (left, 250, right, 650), (left, 850, right, 1170)
    _draw_axes(canvas, top_box, [1, 5, 10, 20, 40, 50, 60, 100, 200, 400], [-120, -100, -80, -60, -40, -20, 0], (1, 500), (-120, 5), "Frequency (Hz, logarithmic)", "Magnitude (dB)", True, ".0f")
    items = [("control", "control", "No digital filter", False), ("hp20", "hp20", "HP20 Butterworth zero-phase", False), ("hp40", "hp40", "HP40 Butterworth zero-phase", False), ("bp20_400", "bp20_400", "BP20-400 Butterworth zero-phase", False), ("fir_bp20_400", "fir_bp20_400", "FIR Hamming centered", False), ("notch50_candidate", "notch50_candidate", "Notch 50 Hz Q=30 (candidate only)", True), ("notch60_candidate", "notch60_candidate", "Notch 60 Hz Q=30 (candidate only)", True)]
    legend_x, legend_y = 1080, 145
    for i, (key, design_key, label, dashed) in enumerate(items):
        design = designs[design_key]
        response = frequency_response(design, freq, zero_phase=(design_key not in ("control", "notch50_candidate", "notch60_candidate")))
        if design_key == "control":
            response = frequency_response(design, freq)
        elif design_key.startswith("notch"):
            response = frequency_response(design, freq, zero_phase=True)
        magnitude = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
        pts = [(_map_x(float(f), 1, 500, left, right, True), _map_y(float(v), -120, 5, 250, 650)) for f, v in zip(freq, magnitude)]
        canvas.line(pts, BRANCH_COLORS[key], 3 if not dashed else 2, "8 6" if dashed else None)
        lx = legend_x + (i % 2) * 355
        ly = legend_y + (i // 2) * 27
        canvas.line([(lx, ly), (lx + 36, ly)], BRANCH_COLORS[key], 4 if not dashed else 2, "8 6" if dashed else None)
        canvas.text(lx + 44, ly, label, 14, "start", color="#343a40")

    # Phase panel: one-pass IIR design phase and the causal FIR linear phase.
    _draw_axes(canvas, phase_box, [1, 5, 10, 20, 40, 50, 60, 100, 200, 400], [-360, -270, -180, -90, 0], (1, 500), (-380, 25), "Frequency (Hz, logarithmic)", "One-pass phase (degrees)", True, ".0f")
    for key in ("hp20", "hp40", "bp20_400"):
        response = frequency_response(designs[key], freq)
        phase = np.unwrap(np.angle(response)) * 180 / np.pi
        pts = [(_map_x(float(f), 1, 500, left, right, True), _map_y(float(v), -380, 25, 850, 1170)) for f, v in zip(freq, phase)]
        canvas.line(pts, BRANCH_COLORS[key], 3)
    fir = designs["fir_bp20_400"]
    fir_phase = np.unwrap(np.angle(frequency_response(fir, freq))) * 180 / np.pi
    # Restrict the display to the first complete phase cycle and indicate the exact linear delay separately.
    fir_visible = np.where((freq >= 1) & (freq <= 5))[0]
    pts = [(_map_x(float(freq[i]), 1, 500, left, right, True), _map_y(float(fir_phase[i]), -380, 25, 850, 1170)) for i in fir_visible]
    canvas.line(pts, BRANCH_COLORS["fir_bp20_400"], 2, "5 5")
    canvas.line([(left, _map_y(0, -380, 25, 850, 1170)), (right, _map_y(0, -380, 25, 850, 1170))], "#111111", 2, "10 7")
    canvas.text(100, 1320, "Forward-backward IIR and centered symmetric FIR have zero phase in the valid interior. One-pass causal FIR delay: 50 samples = 50 ms.", 17, color="#343a40")
    canvas.text(100, 1355, "At each IIR corner, one pass is -3.01 dB; forward-backward application is -6.02 dB because |H_eff| = |H|^2.", 17, color="#343a40")
    canvas.text(100, 1390, "Notch curves illustrate Q=30 designs only; no notch was applied because the recorded 50/60 Hz prominence screen was below 10 dB.", 17, color="#343a40")
    canvas.save(stem)


def make_laterality_figure(effects: list[dict], stem: Path) -> None:
    """Plot participant-level IQRs for each side and each filter branch."""
    metrics = (
        ("rms_adc", "RMS change (%)"),
        ("power_20_60_adc2", "Power 20-60 Hz change (%)"),
        ("power_60_150_adc2", "Power 60-150 Hz change (%)"),
        ("power_150_400_adc2", "Power 150-400 Hz change (%)"),
    )
    branches = ("hp20", "hp40", "bp20_400", "fir_bp20_400")
    labels = {"hp20": "HP20", "hp40": "HP40", "bp20_400": "IIR 20-400", "fir_bp20_400": "FIR 20-400"}
    colors = {"left": "#0072b2", "right": "#d55e00"}
    indexed = {(row.get("laterality"), row.get("filter_branch"), row.get("metric")): row for row in effects}
    sample = indexed.get(("left", "hp20", "rms_adc"), {})
    left_n = int(sample.get("n_participants", 0))
    left_records = int(sample.get("recordings_n", 0))
    sample = indexed.get(("right", "hp20", "rms_adc"), {})
    right_n = int(sample.get("n_participants", 0))
    right_records = int(sample.get("recordings_n", 0))

    width, height = 1840, 1440
    canvas = _Canvas(width, height)
    canvas.text(80, 54, "Filter effects by leg, aggregated within student", 34, bold=True)
    canvas.text(80, 94, "Point: median participant change; whisker: participant Q1-Q3. Each recording is compared with its own stored signal.", 19, color="#586069")
    canvas.line([(1050, 130), (1090, 130)], colors["left"], 4)
    canvas.circle(1070, 130, 6, colors["left"])
    canvas.text(1100, 130, f"Left: {left_n} students, {left_records} recordings", 17)
    canvas.line([(1460, 130), (1500, 130)], colors["right"], 4)
    canvas.circle(1480, 130, 6, colors["right"])
    canvas.text(1510, 130, f"Right: {right_n} students, {right_records} recordings", 17)

    for panel_index, (metric, title) in enumerate(metrics):
        row_index, col_index = divmod(panel_index, 2)
        panel_left = 130 + col_index * 900
        panel_top = 225 + row_index * 560
        plot_left, plot_right = panel_left + 95, panel_left + 830
        plot_top, plot_bottom = panel_top + 35, panel_top + 405
        values = [float(indexed[(side, branch, metric)][field])
            for side in ("left", "right") for branch in branches
            if (side, branch, metric) in indexed
            for field in ("q1_change", "q3_change")]
        low = min([0.0, *values])
        high = max([0.0, *values])
        span = max(high - low, 1.0)
        y_low = math.floor((low - span * 0.08) / 5.0) * 5.0
        y_high = math.ceil((high + span * 0.08) / 5.0) * 5.0
        if y_high <= y_low:
            y_low, y_high = -5.0, 5.0
        canvas.text(panel_left, panel_top, title, 22, bold=True)
        tick_step = max(1.0, math.ceil((y_high - y_low) / 6.0 / 5.0) * 5.0)
        ticks = np.arange(math.ceil(y_low / tick_step) * tick_step, y_high + tick_step, tick_step)
        for tick in ticks:
            y = _map_y(float(tick), y_low, y_high, plot_top, plot_bottom)
            canvas.line([(plot_left, y), (plot_right, y)], "#e2e5e9", 1)
            canvas.text(plot_left - 12, y, f"{tick:g}", 14, "end", color="#495057")
        zero_y = _map_y(0.0, y_low, y_high, plot_top, plot_bottom)
        canvas.line([(plot_left, zero_y), (plot_right, zero_y)], "#71777d", 2, "7 5")
        canvas.line([(plot_left, plot_top), (plot_left, plot_bottom), (plot_right, plot_bottom)], "#454b52", 2)
        for branch_index, branch in enumerate(branches):
            x_center = plot_left + (branch_index + 0.5) * (plot_right - plot_left) / len(branches)
            canvas.text(x_center, plot_bottom + 24, labels[branch], 14, "middle", color="#495057")
            for side, offset in (("left", -13), ("right", 13)):
                result = indexed.get((side, branch, metric))
                if not result:
                    continue
                x = x_center + offset
                q1 = float(result["q1_change"])
                median = float(result["median_change"])
                q3 = float(result["q3_change"])
                y1, ym, y3 = (_map_y(v, y_low, y_high, plot_top, plot_bottom) for v in (q1, median, q3))
                canvas.line([(x, y3), (x, y1)], colors[side], 3)
                canvas.line([(x - 5, y3), (x + 5, y3)], colors[side], 2)
                canvas.line([(x - 5, y1), (x + 5, y1)], colors[side], 2)
                canvas.circle(x, ym, 6, colors[side])
        canvas.text((plot_left + plot_right) / 2, plot_bottom + 58, "Filter branch", 16, "middle", bold=True)
    canvas.text(80, height - 38, "Student counts differ by side because one student has no left-leg R1-R5 recording. No side was imputed.", 17, color="#343a40")
    canvas.save(stem)


def _periodogram(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = x.size
    centered = x - float(np.mean(x))
    window = np.hanning(n)
    transform = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    power = np.abs(transform) ** 2 / (fs_hz * float(np.sum(window**2)))
    amplitude = np.abs(transform) / float(np.sum(window))
    if n % 2 == 0:
        power[1:-1] *= 2.0
        amplitude[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
        amplitude[1:] *= 2.0
    return frequencies, amplitude, power


def _welch(x: np.ndarray, fs_hz: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(nperseg or round(fs_hz))
    step = max(1, nperseg // 2)
    parts = [x[start:start+nperseg] for start in range(0, len(x) - nperseg + 1, step)]
    spectra = [_periodogram(part, fs_hz)[2] for part in parts]
    if not spectra:
        raise ValueError("recording has no complete Welch segment")
    return np.fft.rfftfreq(nperseg, d=1.0 / fs_hz), np.mean(spectra, axis=0)


def _spectrogram(x: np.ndarray, fs_hz: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = int(nperseg or round(fs_hz))
    step = max(1, nperseg // 2)
    starts = range(0, len(x) - nperseg + 1, step)
    blocks = [x[start:start+nperseg] for start in starts]
    if not blocks:
        raise ValueError("recording has no complete spectrogram window")
    psd = np.asarray([_periodogram(block, fs_hz)[2] for block in blocks])
    times = np.asarray([start + nperseg / 2 for start in range(0, len(x) - nperseg + 1, step)]) / fs_hz
    freq = np.fft.rfftfreq(nperseg, d=1.0 / fs_hz)
    return times, freq, psd.T


def count_timestamp_boundaries(records: Iterable[dict]) -> tuple[int, int, int]:
    """Count positive gaps, non-increasing timestamp steps, and split runs from source archives."""
    records = list(records)
    positive_gaps = non_increasing = run_count = 0
    for record in records:
        archive_path = Path(record["output_path"])
        with np.load(archive_path, allow_pickle=False) as archive:
            times = np.asarray(archive["times_s"], dtype=float)
        actual_runs = len(split_contiguous_runs(times))
        if actual_runs != int(record["n_runs"]):
            raise ValueError(
                f"stale filtered archive segmentation: receipt has {record['n_runs']} runs, current rule has {actual_runs}"
            )
        differences = np.diff(times)
        positive = differences[differences > 0]
        median_dt = float(np.median(positive)) if positive.size else float("nan")
        threshold = max(0.0015, 1.5 * median_dt) + 1e-9
        positive_gaps += int(np.sum(differences > threshold))
        non_increasing += int(np.sum(differences <= 0))
        run_count += len(split_contiguous_runs(times))
    if run_count - len(records) != positive_gaps + non_increasing:
        raise ValueError("timestamp boundary counts do not reconcile with contiguous-run splits")
    return positive_gaps, non_increasing, run_count


def _viridis(value: float) -> str:
    stops = [
        (0.00, (68, 1, 84)), (0.25, (59, 82, 139)), (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)), (1.00, (253, 231, 37)),
    ]
    value = min(1.0, max(0.0, value))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if a <= value <= b:
            t = (value - a) / (b - a)
            rgb = tuple(round(x * (1 - t) + y * t) for x, y in zip(ca, cb))
            return "#" + "".join(f"{c:02x}" for c in rgb)
    return "#fde725"


def _draw_trace_panel(canvas: _Canvas, box: tuple[int, int, int, int], times: np.ndarray, traces: dict[str, np.ndarray], span: tuple[float, float]):
    left, top, right, bottom = box
    ymin = min(float(np.quantile(x, 0.01)) for x in traces.values())
    ymax = max(float(np.quantile(x, 0.99)) for x in traces.values())
    pad = max((ymax - ymin) * 0.08, 1.0)
    ymin, ymax = ymin - pad, ymax + pad
    _draw_axes(canvas, box, list(np.linspace(span[0], span[1], 6)), list(np.linspace(math.ceil(ymin/50)*50, math.floor(ymax/50)*50, 5)) if ymax-ymin > 250 else list(np.linspace(ymin,ymax,5)), span, (ymin, ymax), "Time (s)", "Amplitude (ADC counts)", False, ".0f")
    for name, values in traces.items():
        stride = max(1, len(times) // 4500)
        pts = [(_map_x(float(times[i]), *span, left, right), _map_y(float(values[i]), ymin, ymax, top, bottom)) for i in range(0, len(times), stride)]
        canvas.line(pts, BRANCH_COLORS[name], 1 if name == "control" else 2)


def _draw_spectrum_panel(canvas: _Canvas, box: tuple[int, int, int, int], series: dict[str, np.ndarray], fs_hz: float, mode: str):
    left, top, right, bottom = box
    computed = {name: (_periodogram(x, fs_hz) if mode == "fft" else (lambda r: (r[0], None, r[1]))(_welch(x, fs_hz))) for name, x in series.items()}
    if mode == "fft":
        values = [item[1] for item in computed.values()]
        y_label = "Amplitude spectrum (ADC counts)"
        y_max = max(float(np.quantile(v, 0.999)) for v in values)
        y_min = 0.0
        y_ticks = [0, y_max * .25, y_max * .5, y_max * .75, y_max]
    else:
        values = [item[2] for item in computed.values()]
        y_label = "PSD (ADC^2/Hz, dB)"
        all_db = [10*np.log10(np.maximum(v, 1e-16)) for v in values]
        y_min = math.floor(min(float(np.quantile(v, 0.01)) for v in all_db) / 10) * 10
        y_max = math.ceil(max(float(np.quantile(v, 0.99)) for v in all_db) / 10) * 10
        y_ticks = list(np.arange(y_min, y_max + 1, max(10, math.ceil((y_max-y_min)/5/10)*10)))
    _draw_axes(canvas, box, [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], y_ticks, (0, fs_hz/2), (y_min, y_max), "Frequency (Hz)", y_label, False, ".1f" if mode == "fft" else ".0f")
    for name, (frequencies, amplitude, power) in computed.items():
        y_values = amplitude if mode == "fft" else 10*np.log10(np.maximum(power, 1e-16))
        stride = max(1, len(frequencies) // 1600)
        pts = [(_map_x(float(frequencies[i]), 0, fs_hz/2, left, right), _map_y(float(y_values[i]), y_min, y_max, top, bottom)) for i in range(0, len(frequencies), stride)]
        canvas.line(pts, BRANCH_COLORS[name], 2)


def _draw_spectrogram_panel(canvas: _Canvas, box: tuple[int, int, int, int], times: np.ndarray, freq: np.ndarray, psd: np.ndarray, shared_max_db: float, title: str):
    left, top, right, bottom = box
    fmask = (freq >= 0) & (freq <= 500)
    tspan = (float(times[0]), float(times[-1])) if times.size > 1 else (0.0, 1.0)
    if tspan[1] <= tspan[0]:
        tspan = (0.0, 1.0)
    fmax = 500.0
    f_indices = np.flatnonzero(fmask)
    t_indices = np.arange(times.size)
    # Render a bounded cell grid so both the SVG and raster remain tractable.
    f_indices = f_indices[::max(1, len(f_indices)//160)]
    t_indices = t_indices[::max(1, len(t_indices)//220)]
    db = 10 * np.log10(np.maximum(psd, 1e-16))
    dmin = shared_max_db - 60.0
    for ti in t_indices:
        x = _map_x(float(times[ti]), *tspan, left, right)
        x2 = _map_x(float(times[min(ti + max(1, len(times)//220), len(times)-1)]), *tspan, left, right)
        cellw = max(1.0, x2 - x)
        for fi in f_indices:
            y = _map_y(float(freq[fi]), 0, fmax, top, bottom)
            # Approximate cell height from the local frequency-bin spacing.
            cellh = max(1.0, (bottom-top) / max(1, len(f_indices)))
            val = float(db[fi, ti])
            canvas.rect(x, y-cellh, cellw, cellh, _viridis((val-dmin)/60.0))
    canvas.rect(left, top, right-left, bottom-top, "none", "#454b52")
    for tick in [0, 100, 200, 300, 400, 500]:
        y = _map_y(tick, 0, 500, top, bottom)
        canvas.line([(left, y), (right, y)], "#ffffff55", 1)
        canvas.text(left-12, y, str(tick), 15, "end", color="#495057")
    xticks = np.linspace(tspan[0], tspan[1], 6)
    for tick in xticks:
        x = _map_x(float(tick), *tspan, left, right)
        canvas.text(x, bottom+26, f"{tick:.0f}", 15, "middle", color="#495057")
    canvas.line([(left, top), (left, bottom), (right, bottom)], "#454b52", 2)
    canvas.text((left+right)/2, bottom+52, "Time (s)", 16, "middle", bold=True)
    canvas.text(left-65, (top+bottom)/2, "Frequency (Hz)", 16, "middle", bold=True, rotate=-90)
    canvas.text(left, top-24, title, 17, "start", bold=True)


def make_recording_figure(npz_path: Path, fs_hz: float, stem: Path) -> None:
    with np.load(npz_path, allow_pickle=False) as archive:
        times = np.asarray(archive["times_s"], dtype=float)
        mask = np.asarray(archive["common_comparison_mask"], dtype=bool)
        arrays = {name: np.asarray(archive[name], dtype=float) for name in ("control", *APPLIED_BRANCHES)}
        raw = np.asarray(archive["adc_original"], dtype=float)
    runs = split_contiguous_runs(times)
    start, end = max(runs, key=lambda item: item[1]-item[0])
    start += 50
    end -= 50
    available = min(end - start, int(round(30 * fs_hz)))
    if available < round(2 * fs_hz):
        raise ValueError("representative continuous run is shorter than 2 seconds after edge trimming")
    t0 = float(times[start])
    t = times[start:start+available] - t0
    series = {"control": raw[start:start+available]}
    series.update({name: arrays[name][start:start+available] for name in APPLIED_BRANCHES})
    common = np.asarray([mask[start:start+available].all()] * available)
    if not common[0]:
        # The selection is restricted to the longest contiguous run, but still require the shared edge mask.
        valid = np.flatnonzero(mask[start:start+available])
        if valid.size < round(2 * fs_hz):
            raise ValueError("representative run lacks enough samples under the common comparison mask")
        lo, hi = valid[0], valid[-1]+1
        t, series = t[lo:hi], {name: values[lo:hi] for name, values in series.items()}
    n = int(round(fs_hz))
    spec_start = max(0, min(len(t)-n, len(t)//3))
    spectrum_series = {name: values[spec_start:spec_start+n] for name, values in series.items()}
    width, height = 1800, 2120
    canvas = _Canvas(width, height)
    canvas.text(85, 56, "Signal-domain comparison on one anonymized local recording", 34, bold=True)
    canvas.text(85, 96, "R1-R5 recording selected by median sample count; no model score used for selection", 20, color="#586069")
    canvas.text(85, 127, "Amplitude is stored ADC counts; the Voltage_mV field is not a physical calibration", 18, color="#586069")
    canvas.text(85, 151, "A: 5 s excerpt; B: 1 s segment; C-E: longest valid run with 1 s windows and 50% overlap", 15, color="#586069")
    canvas.text(220, 170, "A. Stored signal and four candidate branches", 20, bold=True)
    _draw_trace_panel(canvas, (220, 205, 1710, 440), t, series, (0, min(5.0, float(t[-1]))))
    legend = [("control", "Stored channel"), ("hp20", "HP20"), ("hp40", "HP40"), ("bp20_400", "BP IIR 20-400"), ("fir_bp20_400", "BP FIR Hamming")]
    for i, (name, label) in enumerate(legend):
        x = 260 + i*280
        canvas.line([(x, 525), (x+34, 525)], BRANCH_COLORS[name], 4)
        canvas.text(x+43, 525, label, 15, "start")
    canvas.text(220, 585, "B. One-second Hann-window FFT amplitude spectrum", 20, bold=True)
    _draw_spectrum_panel(canvas, (220, 620, 1710, 810), spectrum_series, fs_hz, "fft")
    canvas.text(220, 910, "C. Welch PSD; 1 s windows, 50% overlap", 20, bold=True)
    _draw_spectrum_panel(canvas, (220, 945, 1710, 1135), series, fs_hz, "psd")
    raw_t, raw_f, raw_s = _spectrogram(series["control"], fs_hz)
    hp_t, hp_f, hp_s = _spectrogram(series["hp20"], fs_hz)
    spectral_mask = (raw_f >= 20) & (raw_f <= 400)
    shared_max = float(np.max(10*np.log10(np.maximum(np.concatenate([raw_s[spectral_mask].ravel(), hp_s[spectral_mask].ravel()]), 1e-16))))
    canvas.text(220, 1215, "D-E. STFT power spectrogram; 1 s Hann windows, 50% overlap, shared dB scale", 20, bold=True)
    _draw_spectrogram_panel(canvas, (220, 1280, 1710, 1500), raw_t, raw_f, raw_s, shared_max, "D. Stored channel")
    _draw_spectrogram_panel(canvas, (220, 1600, 1710, 1820), hp_t, hp_f, hp_s, shared_max, "E. HP20 branch")
    for i, value in enumerate([-60, -40, -20, 0]):
        x = 880 + i*95
        canvas.rect(x, 1900, 74, 22, _viridis((value+60)/60))
        canvas.text(x+37, 1934, f"{value} dB", 14, "middle")
    canvas.text(85, 2000, "Spectrogram levels are relative to the maximum shared 20-400 Hz PSD in these two panels.", 17, color="#343a40")
    canvas.text(85, 2033, "Filtering changes spectral content; without a clean reference, reduced amplitude or power does not establish denoising.", 17, color="#343a40")
    canvas.save(stem)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(
    receipt_path: Path,
    output_dir: Path,
    case_sha256: str | None = None,
    private_root: Path | None = None,
) -> dict:
    receipt_path = Path(receipt_path)
    output_dir = Path(output_dir)
    private_root = Path(private_root) if private_root is not None else receipt_path.parent
    validate_data_paths(
        {"receipt": receipt_path, "output_dir": output_dir, "private_root": private_root},
        Path(__file__).resolve().parents[1],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("analysis") != "filter_sensitivity_v1" or receipt.get("records") != 140 or receipt.get("participants") != 15:
        raise ValueError("filter receipt does not match the audited local corpus")
    if receipt.get("signal_column") != "ADC_Value" or receipt.get("channel_mode") != "unknown_RAW_RECT_ENV":
        raise ValueError("receipt signal metadata differs from the audited channel")
    records = []
    for item in receipt["records_private"]:
        record = dict(item)
        archive_path = resolve_recorded_npz_path(
            private_root,
            record.get("output_path", ""),
            field_name="output_path",
        )
        expected_sha256 = record.get("output_sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("filter receipt output_sha256 is missing or malformed")
        if _sha256_file(archive_path) != expected_sha256.lower():
            raise ValueError("filter receipt output_sha256 does not match its contained NPZ archive")
        record["output_path"] = str(archive_path)
        records.append(record)
    fs_values = [float(record["fs_estimated_median_hz"]) for record in records if record.get("fs_estimated_median_hz")]
    if not fs_values:
        raise ValueError("no timestamp-derived sample-rate values in receipt")
    fs_design = round(float(np.median(fs_values)), 6)
    designs = design_filter_bank(fs_design)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects = summarize_filter_effects(records)
    paired_effects = summarize_paired_laterality(records)
    effect_path = output_dir / "FILTER_EFFECTS_PARTICIPANT_AGGREGATE.csv"
    paired_effect_path = output_dir / "FILTER_EFFECTS_PAIRED_LEFT_MINUS_RIGHT.csv"
    coefficient_path = output_dir / "FILTER_COEFFICIENTS_TIMESTAMP_FS.csv"
    _write_csv(effect_path, effects)
    _write_csv(paired_effect_path, paired_effects)
    _write_csv(coefficient_path, coefficient_rows(designs))

    response_stem = output_dir / "FILTER_BODE_RESPONSE"
    laterality_stem = output_dir / "FILTER_EFFECTS_BY_LEG"
    case_stem = output_dir / "LOCAL_RECORDING_TIME_FFT_PSD_STFT"
    make_response_figure(designs, response_stem)
    make_laterality_figure(effects, laterality_stem)
    selected = select_representative_record(records, case_sha256)
    npz_path = Path(selected["output_path"])
    if not npz_path.is_file():
        raise FileNotFoundError(f"local filtered signal archive is missing: {npz_path}")
    make_recording_figure(npz_path, float(selected["fs_estimated_median_hz"]), case_stem)
    positive_gaps, non_increasing_steps, run_count = count_timestamp_boundaries(records)
    output_names = [effect_path.name, paired_effect_path.name, coefficient_path.name,
        response_stem.with_suffix(".png").name, response_stem.with_suffix(".svg").name,
        laterality_stem.with_suffix(".png").name, laterality_stem.with_suffix(".svg").name,
        case_stem.with_suffix(".png").name, case_stem.with_suffix(".svg").name]
    receipt_out = {
        "analysis": "filter_math_audit_v2",
        "source_filter_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "scope": "local recordings only; R1-R5 for primary filter effects; R0/R0_1 retained in prior QC only",
        "records_audited": receipt["records"],
        "primary_records": receipt["series_records_R1_R5"],
        "preliminary_records_qc_only": receipt["preliminary_records_R0_R0_1"],
        "participants": receipt["participants"],
        "primary_laterality_inventory": {
            side: {
                "participants": len({record["participant_id"] for record in records
                    if record.get("record_type") == "series_R1_R5" and str(record.get("leg", "")).lower() == side}),
                "recordings": sum(1 for record in records
                    if record.get("record_type") == "series_R1_R5" and str(record.get("leg", "")).lower() == side),
            }
            for side in ("left", "right")
        },
        "samples": receipt["samples"],
        "timestamp_fs_hz_min_median_max": [min(fs_values), fs_design, max(fs_values)],
        "timestamp_fs_is_hardware_verified": False,
        "nyquist_hz_at_design_rate": fs_design / 2,
        "total_contiguous_runs": run_count,
        "timestamp_gaps_gt_1_5ms": positive_gaps,
        "non_increasing_timestamp_steps": non_increasing_steps,
        "common_comparison_samples_fraction": sum(int(record["common_comparison_samples"]) for record in records) / sum(int(record["n_samples"]) for record in records),
        "branches": ["control", *APPLIED_BRANCHES],
        "notch_applied": False,
        "notch_design_only": "second order, Q=30, 50/60 Hz candidates",
        "channel_mode": "unknown RAW/rectified/envelope",
        "source_csv_sha256": selected["input_sha256"],
        "private_signal_archive_sha256": selected["output_sha256"],
        "individual_id_in_figures": False,
        "clean_reference_available": False,
        "selection_rule": "eligible R1-R5 recording nearest to median sample count; chosen without model-prediction outcomes",
        "outputs": output_names,
        "output_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in output_names},
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "FILTER_MATH_AUDIT_RECEIPT.json").write_text(json.dumps(receipt_out, indent=2), encoding="utf-8")
    return receipt_out


def main() -> None:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=local / "EMG_Biomecanica/datasets/filter_sensitivity_v1/receipt_filter_sensitivity_v1.json")
    parser.add_argument("--output-dir", type=Path, default=local / "EMG_Biomecanica/reports/filter_math_audit")
    parser.add_argument("--private-root", type=Path, help="Trusted root containing the receipt's filtered NPZ arrays; defaults to the receipt directory")
    parser.add_argument("--case-sha256", help="Optional private archive hash for a previously selected R1-R5 case")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.receipt, args.output_dir, args.case_sha256, args.private_root), indent=2))


if __name__ == "__main__":
    main()
