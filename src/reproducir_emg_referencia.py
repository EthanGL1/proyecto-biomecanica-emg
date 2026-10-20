"""Run public, data-agnostic EMG signal processing on supplied CSV recordings.

The demo mode uses deterministic software fixtures. The command implements
signal processing only; it creates no participant labels, fatigue model,
predictions, or individual figures. Input and output paths must remain outside
the public source checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from comparar_filtros_emg import (
    apply_fir_reflect,
    apply_iir_zero_phase,
    design_butterworth_sos,
    design_fir_bandpass,
    recommended_sos_padlen,
    split_contiguous_runs as _split_contiguous_runs,
)
from release_paths import validate_data_paths


SERIES = ("R1", "R2", "R3", "R4", "R5")
FILTER_BRANCHES = ("control", "hp20", "hp40", "bp20_400", "fir_bp20_400")
GAP_THRESHOLD_S = 0.0015
GAP_FLOAT_TOLERANCE_S = 1e-9
WINDOW_SECONDS = 1.0
WINDOW_OVERLAP = 0.5
MAX_INTERVAL_SPREAD_RATIO = 0.05
LOW_HZ = 20.0
HIGH_HZ = 400.0
FIXTURE_FS_HZ = 1000.0
FIXTURE_RUN_SECONDS = 3.5
FIXTURE_GAP_S = 0.010
FIXTURE_SEED = 20261019

INPUT_COLUMNS = ("series", "time_s", "adc_value")
COHORT_INPUT_COLUMNS = ("subject_id", "side", "series", "time_s", "adc_value")
SIDES = ("left", "right")
COHORT_PREFIX_COLUMNS = ("subject_id", "side")
MASK_COLUMNS = (
    "series", "run_index", "sample_index", "time_s", "control_valid",
    "hp20_valid", "hp40_valid", "bp20_400_valid", "fir_bp20_400_valid",
    "common_valid",
)
SEGMENT_COLUMNS = (
    "series", "run_index", "start_time_s", "end_time_s", "fs_hz",
    "interval_spread_ratio", "sample_count", "common_valid_samples",
    "excluded_edge_samples", "status",
)
FEATURE_COLUMNS = (
    "series", "run_index", "window_index", "time_s", "time_fraction",
    "time_position_proxy", "filter_branch", "sample_count", "rms_adc",
    "mdf_hz", "mnf_hz",
)
FFT_COLUMNS = (
    "series", "run_index", "window_index", "time_s", "filter_branch",
    "frequency_hz", "amplitude_adc",
)
PSD_COLUMNS = (
    "series", "run_index", "filter_branch", "frequency_hz",
    "psd_adc2_per_hz",
)
STFT_COLUMNS = (
    "series", "run_index", "window_index", "time_s", "filter_branch",
    "frequency_hz", "psd_adc2_per_hz",
)
OUTPUT_FILES = ("summary.json", "segments.csv", "mask.csv", "features.csv", "fft.csv", "psd.csv", "stft.csv")


@dataclass
class PipelineResult:
    """In-memory outputs for one or more synthetic/anonymous recordings."""

    summary: dict[str, object]
    mask_rows: list[dict[str, object]] = field(default_factory=list)
    segment_rows: list[dict[str, object]] = field(default_factory=list)
    feature_rows: list[dict[str, object]] = field(default_factory=list)
    fft_rows: list[dict[str, object]] = field(default_factory=list)
    psd_rows: list[dict[str, object]] = field(default_factory=list)
    stft_rows: list[dict[str, object]] = field(default_factory=list)


def split_contiguous_runs(
    times_s: Iterable[float], gap_s: float = GAP_THRESHOLD_S
) -> list[tuple[int, int]]:
    """Split on large timestamp gaps or non-increasing times without filling them."""

    return _split_contiguous_runs(np.asarray(times_s, dtype=float), gap_s=gap_s)


def periodogram(signal: Iterable[float], fs_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-sided Hann FFT amplitude and PSD for a finite sampled window."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 8 or not np.isfinite(values).all():
        raise ValueError("signal must be a finite one-dimensional window of at least 8 samples")
    if not math.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")

    centered = values - float(np.mean(values))
    window = np.hanning(values.size)
    window_energy = float(np.sum(window**2))
    window_sum = float(np.sum(window))
    if window_energy <= 0 or window_sum <= 0:
        raise ValueError("window is too short for a Hann spectrum")
    transform = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / fs_hz)
    psd = np.abs(transform) ** 2 / (fs_hz * window_energy)
    amplitude = np.abs(transform) / window_sum
    if values.size % 2 == 0:
        psd[1:-1] *= 2.0
        amplitude[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
        amplitude[1:] *= 2.0
    return frequencies, amplitude, psd


def _sampling_properties(times: np.ndarray) -> tuple[float | None, float | None]:
    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0:
        return None, None
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return None, None
    spread = float((np.percentile(intervals, 95) - np.percentile(intervals, 5)) / median_interval)
    return 1.0 / median_interval, spread


def _filter_run(
    signal: np.ndarray, fs_hz: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Filter one contiguous run and return branch traces plus common valid mask."""

    if fs_hz <= 2.0 * HIGH_HZ:
        raise ValueError("sampling frequency must exceed 800 Hz for the 20-400 Hz branch")
    designs = {
        "hp20": design_butterworth_sos(fs_hz, "highpass", (20.0,), order=4),
        "hp40": design_butterworth_sos(fs_hz, "highpass", (40.0,), order=4),
        "bp20_400": design_butterworth_sos(fs_hz, "bandpass", (20.0, 400.0), order=4),
    }
    fir = design_fir_bandpass(fs_hz, 20.0, 400.0, taps=101)
    traces: dict[str, np.ndarray] = {"control": np.asarray(signal, dtype=float).copy()}
    branch_masks: dict[str, np.ndarray] = {"control": np.ones(signal.size, dtype=bool)}

    for branch in ("hp20", "hp40", "bp20_400"):
        sos = designs[branch]
        padlen = recommended_sos_padlen(sos)
        mask = np.zeros(signal.size, dtype=bool)
        trace = np.full(signal.size, np.nan, dtype=float)
        if signal.size > 2 * padlen:
            filtered, edge = apply_iir_zero_phase(signal, sos, padlen)
            interior = filtered[edge:signal.size - edge]
            if interior.size:
                trace[edge:signal.size - edge] = interior
                mask[edge:signal.size - edge] = True
        traces[branch] = trace
        branch_masks[branch] = mask

    fir_edge = (fir.size - 1) // 2
    fir_trace = np.full(signal.size, np.nan, dtype=float)
    fir_mask = np.zeros(signal.size, dtype=bool)
    if signal.size > 2 * fir_edge:
        filtered, edge = apply_fir_reflect(signal, fir)
        interior = filtered[edge:signal.size - edge]
        if interior.size:
            fir_trace[edge:signal.size - edge] = interior
            fir_mask[edge:signal.size - edge] = True
    traces["fir_bp20_400"] = fir_trace
    branch_masks["fir_bp20_400"] = fir_mask

    common = np.logical_and.reduce(list(branch_masks.values()))
    return traces, branch_masks, common


def _append_sample_mask(
    result: PipelineResult,
    series: str,
    run_index: int,
    times: np.ndarray,
    branch_masks: Mapping[str, np.ndarray] | None = None,
    common_mask: np.ndarray | None = None,
) -> None:
    all_invalid = np.zeros(times.size, dtype=bool)
    masks = branch_masks or {branch: all_invalid for branch in FILTER_BRANCHES}
    common = common_mask if common_mask is not None else all_invalid
    for sample_index, time_s in enumerate(times):
        row: dict[str, object] = {
            "series": series,
            "run_index": run_index,
            "sample_index": sample_index,
            "time_s": float(time_s),
        }
        for branch in FILTER_BRANCHES:
            row[f"{branch}_valid"] = bool(masks[branch][sample_index])
        row["common_valid"] = bool(common[sample_index])
        result.mask_rows.append(row)


def _spectral_metrics(signal: np.ndarray, fs_hz: float) -> tuple[float, float | None, float | None]:
    centered = signal - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(centered**2)))
    frequencies, _, psd = periodogram(signal, fs_hz)
    band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)
    band_power = psd[band]
    band_frequencies = frequencies[band]
    total = float(np.sum(band_power))
    if band_power.size == 0 or total <= 0 or not math.isfinite(total):
        return rms, None, None
    mnf = float(np.sum(band_frequencies * band_power) / total)
    median_index = min(int(np.searchsorted(np.cumsum(band_power), total / 2.0, side="left")), band_power.size - 1)
    mdf = float(band_frequencies[median_index])
    return rms, mdf, mnf


def _position_proxy(time_fraction: float) -> str:
    """Describe elapsed-time thirds only; this is not a fatigue target."""

    if time_fraction < 1.0 / 3.0:
        return "temprana"
    if time_fraction < 2.0 / 3.0:
        return "transicion"
    return "tardia"


def analyze_arrays(
    recordings: Mapping[str, tuple[Iterable[float], Iterable[float]]],
    *,
    gap_s: float = GAP_THRESHOLD_S,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = WINDOW_OVERLAP,
) -> PipelineResult:
    """Run the reference filter, common-mask, feature, FFT, PSD, and STFT pipeline."""

    if not recordings:
        raise ValueError("at least one R1-R5 recording is required")
    if any(series not in SERIES for series in recordings):
        raise ValueError("series values must be R1-R5")
    if not math.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("gap_s must be finite and positive")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be finite and in [0, 1)")

    result = PipelineResult(summary={})
    processed_runs = 0
    skipped_runs = 0
    total_runs = 0
    all_series = sorted(recordings, key=SERIES.index)

    for series in all_series:
        times = np.asarray(recordings[series][0], dtype=float)
        signal = np.asarray(recordings[series][1], dtype=float)
        if times.ndim != 1 or signal.ndim != 1 or times.size != signal.size:
            raise ValueError("each recording needs matching one-dimensional times and values")
        if times.size < 2 or not np.isfinite(times).all() or not np.isfinite(signal).all():
            raise ValueError("recordings must contain at least two finite timestamps and samples")
        if np.any(np.diff(times) < 0):
            raise ValueError("timestamps must not go backwards; split a recording after a clock reset")
        series_start = float(times[0])
        series_duration = float(times[-1] - times[0])
        if series_duration <= 0:
            raise ValueError("recording timestamps must span a positive duration")
        runs = split_contiguous_runs(times, gap_s=gap_s)
        total_runs += len(runs)

        for run_index, (start, stop) in enumerate(runs, start=1):
            run_times = times[start:stop]
            run_signal = signal[start:stop]
            fs_hz, spread = _sampling_properties(run_times)
            mask_row: dict[str, object] = {
                "series": series,
                "run_index": run_index,
                "start_time_s": float(run_times[0]),
                "end_time_s": float(run_times[-1]),
                "fs_hz": fs_hz,
                "interval_spread_ratio": spread,
                "sample_count": int(run_signal.size),
                "common_valid_samples": 0,
                "excluded_edge_samples": int(run_signal.size),
                "status": "skipped",
            }
            if fs_hz is None or spread is None:
                mask_row["status"] = "insufficient_positive_time_intervals"
                result.segment_rows.append(mask_row)
                _append_sample_mask(result, series, run_index, run_times)
                skipped_runs += 1
                continue
            if spread > MAX_INTERVAL_SPREAD_RATIO:
                mask_row["status"] = "irregular_sampling_interval_spread_over_5_percent"
                result.segment_rows.append(mask_row)
                _append_sample_mask(result, series, run_index, run_times)
                skipped_runs += 1
                continue
            if fs_hz <= 2.0 * HIGH_HZ:
                mask_row["status"] = "sampling_frequency_must_exceed_800_hz"
                result.segment_rows.append(mask_row)
                _append_sample_mask(result, series, run_index, run_times)
                skipped_runs += 1
                continue

            traces, branch_masks, common_mask = _filter_run(run_signal, fs_hz)
            valid_indices = np.flatnonzero(common_mask)
            valid_count = int(valid_indices.size)
            mask_row["common_valid_samples"] = valid_count
            mask_row["excluded_edge_samples"] = int(run_signal.size - valid_count)
            if valid_count == 0:
                mask_row["status"] = "insufficient_samples_after_common_filter_edge_mask"
                result.segment_rows.append(mask_row)
                _append_sample_mask(result, series, run_index, run_times, branch_masks, common_mask)
                skipped_runs += 1
                continue
            mask_row["status"] = "processed"
            result.segment_rows.append(mask_row)
            _append_sample_mask(result, series, run_index, run_times, branch_masks, common_mask)
            processed_runs += 1

            window_samples = int(round(fs_hz * window_seconds))
            step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
            if window_samples < 8:
                raise ValueError("window_seconds produces fewer than 8 samples")
            spectra_by_branch: dict[str, list[np.ndarray]] = {branch: [] for branch in FILTER_BRANCHES}
            window_index = 0
            for window_start in range(0, run_signal.size - window_samples + 1, step_samples):
                window_end = window_start + window_samples
                if not bool(np.all(common_mask[window_start:window_end])):
                    continue
                window_index += 1
                window_time = float((run_times[window_start] + run_times[window_end - 1]) / 2.0)
                time_fraction = min(1.0, max(0.0, (window_time - series_start) / series_duration))
                for branch in FILTER_BRANCHES:
                    window_signal = traces[branch][window_start:window_end]
                    if not np.isfinite(window_signal).all():
                        raise AssertionError("common mask includes an invalid filter edge")
                    rms, mdf, mnf = _spectral_metrics(window_signal, fs_hz)
                    result.feature_rows.append({
                        "series": series,
                        "run_index": run_index,
                        "window_index": window_index,
                        "time_s": window_time,
                        "time_fraction": time_fraction,
                        "time_position_proxy": _position_proxy(time_fraction),
                        "filter_branch": branch,
                        "sample_count": window_samples,
                        "rms_adc": rms,
                        "mdf_hz": mdf,
                        "mnf_hz": mnf,
                    })
                    frequencies, amplitude, psd = periodogram(window_signal, fs_hz)
                    spectra_by_branch[branch].append(psd)
                    for frequency, amplitude_value, psd_value in zip(frequencies, amplitude, psd):
                        result.fft_rows.append({
                            "series": series,
                            "run_index": run_index,
                            "window_index": window_index,
                            "time_s": window_time,
                            "filter_branch": branch,
                            "frequency_hz": float(frequency),
                            "amplitude_adc": float(amplitude_value),
                        })
                        result.stft_rows.append({
                            "series": series,
                            "run_index": run_index,
                            "window_index": window_index,
                            "time_s": window_time,
                            "filter_branch": branch,
                            "frequency_hz": float(frequency),
                            "psd_adc2_per_hz": float(psd_value),
                        })

            for branch, spectra in spectra_by_branch.items():
                if not spectra:
                    continue
                mean_psd = np.mean(np.asarray(spectra), axis=0)
                frequencies = np.fft.rfftfreq(window_samples, d=1.0 / fs_hz)
                for frequency, psd_value in zip(frequencies, mean_psd):
                    result.psd_rows.append({
                        "series": series,
                        "run_index": run_index,
                        "filter_branch": branch,
                        "frequency_hz": float(frequency),
                        "psd_adc2_per_hz": float(psd_value),
                    })

    result.summary = {
        "algorithm": "generic_emg_reference_v1",
        "series": all_series,
        "series_scope": "R1-R5 temporal series labels only",
        "fatigue_target": "none; elapsed-time thirds are a position proxy only",
        "filter_branches": list(FILTER_BRANCHES),
        "filter_definition": {
            "hp20": "4th-order Butterworth high-pass, forward-backward zero phase",
            "hp40": "4th-order Butterworth high-pass, forward-backward zero phase",
            "bp20_400": "4th-order Butterworth 20-400 Hz band-pass, forward-backward zero phase",
            "fir_bp20_400": "101-tap Hamming-window 20-400 Hz band-pass, reflected edges trimmed",
            "control": "unfiltered stored values",
        },
        "gap_rule": "split when delta time exceeds max(1.5 ms, 1.5 times the median positive interval) plus 1 ns; no interpolation",
        "sampling_rule": "median timestamp interval; skip runs with 5th-to-95th percentile spread over 5%; no resampling",
        "window_seconds": float(window_seconds),
        "window_overlap": float(overlap),
        "spectral_band_hz": [LOW_HZ, HIGH_HZ],
        "spectrum": "one-sided Hann-window FFT; PSD in ADC^2/Hz; PSD output is the mean of window periodograms",
        "runs_total": total_runs,
        "runs_processed": processed_runs,
        "runs_skipped": skipped_runs,
        "segment_rows": len(result.segment_rows),
        "sample_mask_rows": len(result.mask_rows),
        "window_rows": len(result.feature_rows),
        "fft_rows": len(result.fft_rows),
        "psd_rows": len(result.psd_rows),
        "stft_rows": len(result.stft_rows),
        "data_scope": "signal processing only; no cohort-level study results are computed",
    }
    return result


def analyze_cohort_arrays(
    recordings: Mapping[tuple[str, str, str], tuple[Iterable[float], Iterable[float]]],
    *,
    gap_s: float = GAP_THRESHOLD_S,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = WINDOW_OVERLAP,
) -> PipelineResult:
    """Process independently keyed recordings without cohort-level comparisons.

    Keys are explicit ``(subject_id, side, series)`` tuples. The function only
    aggregates export rows and structural counts; it does not create labels or
    compare participants, sides, or series.
    """

    if not recordings:
        raise ValueError("at least one subject/side/series recording is required")

    tagged = PipelineResult(summary={})
    group_summaries: list[dict[str, object]] = []
    subjects: set[str] = set()
    series_seen: set[str] = set()
    row_fields = (
        "mask_rows", "segment_rows", "feature_rows", "fft_rows", "psd_rows", "stft_rows"
    )

    for key, (times_value, signal_value) in recordings.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError("each cohort recording key must contain subject_id, side, and series")
        subject_id, side, series = key
        if not isinstance(subject_id, str) or not subject_id or subject_id != subject_id.strip():
            raise ValueError("subject_id must be present and have no surrounding whitespace")
        if side not in SIDES:
            raise ValueError("side values must be exactly left or right")
        if series not in SERIES:
            raise ValueError("series values must be R1-R5")

        times = np.asarray(times_value, dtype=float)
        signal = np.asarray(signal_value, dtype=float)
        if times.ndim != 1 or signal.ndim != 1 or times.size != signal.size:
            raise ValueError("each cohort recording needs matching one-dimensional times and values")
        if times.size < 2 or not np.isfinite(times).all() or not np.isfinite(signal).all():
            raise ValueError("cohort recordings must contain at least two finite timestamps and samples")
        if np.any(np.diff(times) <= 0):
            raise ValueError("cohort timestamps must increase strictly; duplicate samples and clock resets are invalid")

        group_result = analyze_arrays(
            {series: (times, signal)},
            gap_s=gap_s,
            window_seconds=window_seconds,
            overlap=overlap,
        )
        if group_result.summary["runs_processed"] == 0 or not group_result.feature_rows:
            raise ValueError(
                "a subject/side/series group is too short or has no complete valid analysis window"
            )

        subjects.add(subject_id)
        series_seen.add(series)
        group_summaries.append(group_result.summary)
        for field_name in row_fields:
            target_rows = getattr(tagged, field_name)
            for row in getattr(group_result, field_name):
                target_rows.append({"subject_id": subject_id, "side": side, **row})

    first_summary = group_summaries[0]
    tagged.summary = {
        **first_summary,
        "analysis_mode": "grouped_cohort_csv",
        "series": [series for series in SERIES if series in series_seen],
        "subject_count": len(subjects),
        "recording_group_count": len(group_summaries),
        "runs_total": sum(int(summary["runs_total"]) for summary in group_summaries),
        "runs_processed": sum(int(summary["runs_processed"]) for summary in group_summaries),
        "runs_skipped": sum(int(summary["runs_skipped"]) for summary in group_summaries),
        "segment_rows": len(tagged.segment_rows),
        "sample_mask_rows": len(tagged.mask_rows),
        "window_rows": len(tagged.feature_rows),
        "fft_rows": len(tagged.fft_rows),
        "psd_rows": len(tagged.psd_rows),
        "stft_rows": len(tagged.stft_rows),
        "cohort_comparisons_computed": False,
        "data_scope": (
            "caller-supplied grouped CSV; independent signal processing only; "
            "no fatigue labels or cross-subject, cross-side, or cross-series comparisons"
        ),
    }
    return tagged


def read_signal_csv(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read a strict three-column CSV; reject identity or other metadata fields."""

    recordings: dict[str, tuple[list[float], list[float]]] = {}
    seen_series: set[str] = set()
    previous_series: str | None = None
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(fields) != set(INPUT_COLUMNS):
            raise ValueError("input CSV columns must be exactly series,time_s,adc_value")
        for line_number, row in enumerate(reader, start=2):
            series = str(row["series"]).strip()
            if series not in SERIES:
                raise ValueError("series values must be R1-R5")
            if series != previous_series:
                if series in seen_series:
                    raise ValueError("each R1-R5 series must occupy one contiguous CSV block")
                seen_series.add(series)
                previous_series = series
            try:
                time_s = float(row["time_s"])
                adc_value = float(row["adc_value"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value on CSV line {line_number}") from error
            if not math.isfinite(time_s) or not math.isfinite(adc_value):
                raise ValueError(f"CSV values must be finite on line {line_number}")
            if series not in recordings:
                recordings[series] = ([], [])
            recordings[series][0].append(time_s)
            recordings[series][1].append(adc_value)
    if not recordings:
        raise ValueError("input CSV contains no R1-R5 samples")
    return {
        series: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        for series, (times, values) in recordings.items()
    }


def read_cohort_signal_csv(
    path: Path,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Read explicit anonymized subject/side/series blocks from the caller CSV."""

    recordings: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}
    seen_groups: set[tuple[str, str, str]] = set()
    previous_key: tuple[str, str, str] | None = None
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(fields) != set(COHORT_INPUT_COLUMNS):
            raise ValueError(
                "cohort CSV must contain exactly these columns in any order: "
                "subject_id,side,series,time_s,adc_value"
            )
        previous_time_by_key: dict[tuple[str, str, str], float] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"unexpected extra field on CSV line {line_number}")
            raw_subject_id = row["subject_id"]
            if raw_subject_id is None or not raw_subject_id.strip():
                raise ValueError(f"subject_id is required on CSV line {line_number}")
            if raw_subject_id != raw_subject_id.strip():
                raise ValueError(f"subject_id has surrounding whitespace on CSV line {line_number}")
            subject_id = raw_subject_id
            side = row["side"]
            if side not in SIDES:
                raise ValueError(f"side must be exactly left or right on CSV line {line_number}")
            series = row["series"]
            if series not in SERIES:
                raise ValueError(f"series must be R1-R5 on CSV line {line_number}")
            key = (subject_id, side, series)
            if key != previous_key:
                if key in seen_groups:
                    raise ValueError("duplicate subject/side/series block in cohort CSV")
                seen_groups.add(key)
                previous_key = key
            if row["time_s"] is None or row["adc_value"] is None:
                raise ValueError(f"time_s and adc_value are required on CSV line {line_number}")
            try:
                time_s = float(row["time_s"])
                adc_value = float(row["adc_value"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value on CSV line {line_number}") from error
            if not math.isfinite(time_s) or not math.isfinite(adc_value):
                raise ValueError(f"CSV values must be finite on line {line_number}")
            previous_time = previous_time_by_key.get(key)
            if previous_time is not None and time_s <= previous_time:
                if time_s == previous_time:
                    raise ValueError("duplicate sample timestamp within subject/side/series group")
                raise ValueError("clock reset or mixed time order within subject/side/series group")
            previous_time_by_key[key] = time_s
            if key not in recordings:
                recordings[key] = ([], [])
            recordings[key][0].append(time_s)
            recordings[key][1].append(adc_value)
    if not recordings:
        raise ValueError("cohort CSV contains no subject/side/series samples")
    return {
        key: (np.asarray(times, dtype=float), np.asarray(values, dtype=float))
        for key, (times, values) in recordings.items()
    }


def read_cohort_signal_fragments(
    paths: Iterable[Path],
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    """Reassemble CSV fragments by explicit group and original timestamps.

    Each fragment uses the same strict grouped CSV schema as
    :func:`read_cohort_signal_csv`. Fragments may be supplied in any order;
    samples are sorted by timestamp within each pseudonymized group. Overlaps,
    duplicate timestamps, and clock resets are rejected. Real timestamp gaps
    are preserved for the analyzer to split; missing samples are never filled.
    """

    fragment_paths = [Path(path) for path in paths]
    if not fragment_paths:
        raise ValueError("at least one cohort fragment CSV is required")
    pieces: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for path in fragment_paths:
        fragment = read_cohort_signal_csv(path)
        for key, (times, values) in fragment.items():
            pieces.setdefault(key, []).append((times, values))

    recordings: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key in sorted(pieces):
        group_pieces = pieces[key]
        times = np.concatenate([piece[0] for piece in group_pieces])
        values = np.concatenate([piece[1] for piece in group_pieces])
        order = np.argsort(times, kind="stable")
        times = times[order]
        values = values[order]
        if np.any(np.diff(times) <= 0):
            raise ValueError(
                "cohort fragments overlap or duplicate timestamp within subject/side/series group"
            )
        recordings[key] = (times, values)
    return recordings


def write_synthetic_fixture(path: Path, *, overwrite: bool = False) -> Path:
    """Create deterministic multi-series signals with an explicit timestamp gap."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"fixture already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(round(FIXTURE_FS_HZ * FIXTURE_RUN_SECONDS))
    first_times = np.arange(sample_count, dtype=float) / FIXTURE_FS_HZ
    second_times = np.arange(sample_count, 2 * sample_count, dtype=float) / FIXTURE_FS_HZ + FIXTURE_GAP_S
    times = np.concatenate((first_times, second_times))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(INPUT_COLUMNS)
        for index, series in enumerate(SERIES):
            rng = np.random.default_rng(FIXTURE_SEED + index)
            fundamental_hz = 65.0 + 10.0 * index
            amplitude = 1.0 + 0.1 * index
            signal = (
                amplitude * np.sin(2.0 * np.pi * fundamental_hz * times)
                + 0.25 * np.sin(2.0 * np.pi * 180.0 * times)
                + 0.15 * np.sin(2.0 * np.pi * 10.0 * times)
                + 0.01 * rng.standard_normal(times.size)
            )
            for time_s, adc_value in zip(times, signal):
                writer.writerow((series, f"{time_s:.9f}", f"{adc_value:.12f}"))
    return path


def _spreadsheet_safe_identifier(value: object) -> object:
    """Keep untrusted identifiers inert when a CSV is opened in a spreadsheet."""
    if not isinstance(value, str):
        return value
    leading = value.lstrip(" \t\r\n\v\f\ufeff")
    if value.startswith(("\t", "\r", "\n")) or (
        leading and leading[0] in "=+-@"
    ):
        return "'" + value
    return value

def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        if "subject_id" in columns:
            exported_rows = [
                {**row, "subject_id": _spreadsheet_safe_identifier(row.get("subject_id"))}
                for row in rows
            ]
        else:
            exported_rows = rows
        writer.writerows(exported_rows)


def _ensure_output_dir(path: Path, *, overwrite: bool) -> Path:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError("output directory must be new or use --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result(result: PipelineResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, columns, rows in (
        ("segments.csv", SEGMENT_COLUMNS, result.segment_rows),
        ("mask.csv", MASK_COLUMNS, result.mask_rows),
        ("features.csv", FEATURE_COLUMNS, result.feature_rows),
        ("fft.csv", FFT_COLUMNS, result.fft_rows),
        ("psd.csv", PSD_COLUMNS, result.psd_rows),
        ("stft.csv", STFT_COLUMNS, result.stft_rows),
    ):
        if result.summary.get("analysis_mode") == "grouped_cohort_csv":
            columns = COHORT_PREFIX_COLUMNS + columns
        _write_csv(output_dir / name, columns, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV with only series,time_s,adc_value columns")
    source.add_argument(
        "--cohort-input",
        type=Path,
        help="grouped CSV with subject_id,side,series,time_s,adc_value columns",
    )
    source.add_argument(
        "--cohort-fragments",
        type=Path,
        nargs="+",
        help="one or more grouped CSV fragments with original, non-overlapping timestamps",
    )
    source.add_argument("--demo", action="store_true", help="create and process deterministic synthetic R1-R5 data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing outputs in an existing directory")
    args = parser.parse_args(argv)
    checkout_root = Path(__file__).resolve().parents[1]
    path_values: dict[str, object] = {"output_dir": args.output_dir}
    fragment_names: list[str] = []
    if args.input is not None:
        path_values["input"] = args.input
    elif args.cohort_input is not None:
        path_values["cohort_input"] = args.cohort_input
    elif args.cohort_fragments is not None:
        fragment_names = [f"fragment_{index}" for index in range(len(args.cohort_fragments))]
        path_values.update(dict(zip(fragment_names, args.cohort_fragments)))
    paths = validate_data_paths(path_values, checkout_root)
    selected_inputs = (
        [paths["input"]] if args.input is not None else
        [paths["cohort_input"]] if args.cohort_input is not None else
        [paths[name] for name in fragment_names]
    )
    if any(input_path in {paths["output_dir"] / name for name in OUTPUT_FILES} for input_path in selected_inputs):
        raise ValueError("input CSV path conflicts with a generated output filename")
    if args.demo:
        output_dir = _ensure_output_dir(paths["output_dir"], overwrite=args.overwrite)
        fixture = output_dir / "synthetic_emg.csv"
        write_synthetic_fixture(fixture, overwrite=args.overwrite)
        input_path = fixture
        input_source = "synthetic_demo"
        result = analyze_arrays(read_signal_csv(input_path))
        result.summary["input_source"] = input_source
        result.summary["synthetic_fixture"] = (
            "deterministic multisine plus low-amplitude Gaussian noise; "
            "software fixture, not a physiological EMG simulator"
        )
        result.summary["data_scope"] = "synthetic fixtures only; no study results are reproduced and no cohort data are used"
    else:
        if args.cohort_fragments is not None:
            recordings = read_cohort_signal_fragments(selected_inputs)
            result = analyze_cohort_arrays(recordings)
            result.summary["input_source"] = "cohort_fragments"
            result.summary["fragment_file_count"] = len(selected_inputs)
            result.summary["sample_count"] = sum(len(times) for times, _ in recordings.values())
            result.summary["data_scope"] = (
                "caller-supplied grouped CSV fragments reassembled by pseudonymized group and original timestamps; "
                "independent signal processing only; no fatigue labels or cross-subject, cross-side, or cross-series comparisons"
            )
        elif args.cohort_input is not None:
            result = analyze_cohort_arrays(read_cohort_signal_csv(selected_inputs[0]))
            result.summary["input_source"] = "cohort_csv"
        else:
            result = analyze_arrays(read_signal_csv(selected_inputs[0]))
            result.summary["input_source"] = "csv_input"
            result.summary["data_scope"] = (
                "caller-supplied CSV without identity columns; single-recording processing only; "
                "no cohort-level study results are computed"
            )
        output_dir = _ensure_output_dir(paths["output_dir"], overwrite=args.overwrite)
    if not args.demo and (args.cohort_input is not None or args.cohort_fragments is not None):
        if args.cohort_fragments is None:
            result.summary["data_scope"] = (
                "caller-supplied grouped CSV; independent signal processing only; "
                "no fatigue labels or cross-subject, cross-side, or cross-series comparisons"
            )
    write_result(result, output_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

