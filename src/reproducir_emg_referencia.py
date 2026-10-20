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
from emg_dashboard import derive_candidate_activity, render_dashboard_html
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
SOURCE_TIME_COLUMN = "Relative_Time_s"
SOURCE_SIGNAL_COLUMN = "ADC_Value"
SOURCE_OPTIONAL_COLUMNS = ("Voltage_mV",)
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
TRACE_COLUMNS = (
    "series", "run_index", "sample_index", "time_s", "raw_adc", "control_adc",
    "hp20_adc", "hp40_adc", "bp20_400_adc", "fir_bp20_400_adc", "common_valid",
)
ACTIVITY_CANDIDATE_COLUMNS = (
    "series", "run_index", "start_time_s", "end_time_s", "duration_s",
    "threshold_rms_adc", "mean_rms_adc", "peak_rms_adc", "quality", "interpretation",
)
OUTPUT_FILES = (
    "summary.json", "segments.csv", "mask.csv", "features.csv", "fft.csv",
    "psd.csv", "stft.csv", "activity_candidates.csv", "dashboard.html",
    "cohort_by_side_series.csv", "cohort_r1_r5_paired.csv", "cohort_bilateral_paired.csv",
)
COHORT_METRICS = (
    ("rms_adc", "ADC counts"),
    ("mdf_hz", "Hz"),
    ("mnf_hz", "Hz"),
)
COHORT_SUMMARY_COLUMNS = (
    "side", "series", "filter_branch", "metric", "unit", "n_participants",
    "median", "q1", "q3", "aggregation",
)
COHORT_R1_R5_COLUMNS = (
    "side", "filter_branch", "metric", "unit", "n_paired_participants",
    "median_change", "q1_change", "q3_change", "change_definition", "aggregation",
)
COHORT_BILATERAL_COLUMNS = (
    "series", "filter_branch", "metric", "unit", "n_paired_participants",
    "median_left_minus_right", "q1_left_minus_right", "q3_left_minus_right", "aggregation",
)


@dataclass
class PipelineResult:
    """In-memory outputs for one or more synthetic or caller-supplied recordings."""

    summary: dict[str, object]
    mask_rows: list[dict[str, object]] = field(default_factory=list)
    segment_rows: list[dict[str, object]] = field(default_factory=list)
    feature_rows: list[dict[str, object]] = field(default_factory=list)
    fft_rows: list[dict[str, object]] = field(default_factory=list)
    psd_rows: list[dict[str, object]] = field(default_factory=list)
    stft_rows: list[dict[str, object]] = field(default_factory=list)
    trace_rows: list[dict[str, object]] = field(default_factory=list)
    cohort_by_side_series_rows: list[dict[str, object]] = field(default_factory=list)
    cohort_r1_r5_rows: list[dict[str, object]] = field(default_factory=list)
    cohort_bilateral_rows: list[dict[str, object]] = field(default_factory=list)


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


def _append_sample_traces(
    result: PipelineResult,
    series: str,
    run_index: int,
    times: np.ndarray,
    raw: np.ndarray,
    traces: Mapping[str, np.ndarray] | None = None,
    branch_masks: Mapping[str, np.ndarray] | None = None,
    common_mask: np.ndarray | None = None,
) -> None:
    if traces is None or branch_masks is None:
        traces = {}
        branch_masks = {}
    common = common_mask if common_mask is not None else np.zeros(times.size, dtype=bool)
    for index, time_s in enumerate(times):
        row: dict[str, object] = {
            "series": series,
            "run_index": run_index,
            "sample_index": index,
            "time_s": float(time_s),
            "raw_adc": float(raw[index]),
            "common_valid": bool(common[index]),
        }
        for branch in FILTER_BRANCHES:
            valid = branch_masks.get(branch)
            values = traces.get(branch)
            row[f"{branch}_adc"] = (
                float(values[index]) if valid is not None and values is not None and valid[index] else None
            )
        result.trace_rows.append(row)


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
    include_traces: bool = False,
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
                if include_traces:
                    _append_sample_traces(result, series, run_index, run_times, run_signal)
                skipped_runs += 1
                continue
            if spread > MAX_INTERVAL_SPREAD_RATIO:
                mask_row["status"] = "irregular_sampling_interval_spread_over_5_percent"
                result.segment_rows.append(mask_row)
                _append_sample_mask(result, series, run_index, run_times)
                if include_traces:
                    _append_sample_traces(result, series, run_index, run_times, run_signal)
                skipped_runs += 1
                continue
            if fs_hz <= 2.0 * HIGH_HZ:
                mask_row["status"] = "sampling_frequency_must_exceed_800_hz"
                result.segment_rows.append(mask_row)
                _append_sample_mask(result, series, run_index, run_times)
                if include_traces:
                    _append_sample_traces(result, series, run_index, run_times, run_signal)
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
                if include_traces:
                    _append_sample_traces(result, series, run_index, run_times, run_signal, traces, branch_masks, common_mask)
                skipped_runs += 1
                continue
            mask_row["status"] = "processed"
            result.segment_rows.append(mask_row)
            _append_sample_mask(result, series, run_index, run_times, branch_masks, common_mask)
            if include_traces:
                _append_sample_traces(result, series, run_index, run_times, run_signal, traces, branch_masks, common_mask)
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
        "dashboard_trace_rows": len(result.trace_rows),
        "data_scope": "signal processing only; no cohort-level study results are computed",
    }
    return result


def summarize_cohort_features(
    feature_rows: Iterable[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Create descriptive summaries without direct participant identifiers.

    Overlapping windows are first reduced to one median per participant, side,
    series, filter branch, and metric. Cohort summaries then use participants
    as the unit. R1-R5 and bilateral contrasts use complete within-participant
    pairs only. This generic summary has no activity mask and is not a fatigue
    analysis; MDF/MNF estimates are included only when finite.
    """

    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    seen = False
    for row in feature_rows:
        seen = True
        subject_id = row.get("subject_id")
        side = row.get("side")
        series = row.get("series")
        branch = row.get("filter_branch")
        if not isinstance(subject_id, str) or not subject_id or subject_id != subject_id.strip():
            raise ValueError("cohort feature rows require a nonempty subject_id without surrounding whitespace")
        if side not in SIDES:
            raise ValueError("cohort feature rows require side left or right")
        if series not in SERIES:
            raise ValueError("cohort series values must be R1-R5; preliminary R0/R0_1 are not accepted")
        if branch not in FILTER_BRANCHES:
            raise ValueError("cohort feature rows contain an unknown filter_branch")
        for metric, _unit in COHORT_METRICS:
            value = row.get(metric)
            if value is None or value == "":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"cohort feature metric {metric} must be numeric or missing") from error
            if math.isfinite(numeric):
                grouped.setdefault((subject_id, str(side), str(series), str(branch), metric), []).append(numeric)
    if not seen:
        raise ValueError("at least one cohort feature row is required")

    # One feature value per recording prevents overlapping windows from acting
    # as independent observations in the cohort summary.
    recording_values = {
        key: float(np.median(values))
        for key, values in grouped.items()
        if values
    }
    summary_groups: dict[tuple[str, str, str, str], dict[str, float]] = {}
    r1_r5_groups: dict[tuple[str, str, str, str], dict[str, float]] = {}
    bilateral_groups: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for (subject_id, side, series, branch, metric), value in recording_values.items():
        summary_groups.setdefault((side, series, branch, metric), {})[subject_id] = value
        if series == "R1":
            r1_r5_groups.setdefault((side, branch, metric, subject_id), {})[series] = value
        elif series == "R5":
            r1_r5_groups.setdefault((side, branch, metric, subject_id), {})[series] = value
        bilateral_groups.setdefault((series, branch, metric, subject_id), {})[side] = value

    by_side_series: list[dict[str, object]] = []
    unit_lookup = dict(COHORT_METRICS)
    for (side, series, branch, metric), participants in sorted(summary_groups.items()):
        values = np.asarray(list(participants.values()), dtype=np.float64)
        by_side_series.append({
            "side": side,
            "series": series,
            "filter_branch": branch,
            "metric": metric,
            "unit": unit_lookup[metric],
            "n_participants": int(values.size),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "aggregation": "window median per recording; participant median and IQR; descriptive, not a confidence interval",
        })

    r1_r5_paired: list[dict[str, object]] = []
    paired_values: dict[tuple[str, str, str], list[float]] = {}
    for (side, branch, metric, _subject_id), endpoints in r1_r5_groups.items():
        if "R1" not in endpoints or "R5" not in endpoints:
            continue
        initial, final = endpoints["R1"], endpoints["R5"]
        if metric == "rms_adc":
            if initial == 0:
                continue
            difference = 100.0 * (final - initial) / abs(initial)
        else:
            difference = final - initial
        paired_values.setdefault((side, branch, metric), []).append(float(difference))
    for (side, branch, metric), items in sorted(paired_values.items()):
        values = np.asarray(items, dtype=np.float64)
        r1_r5_paired.append({
            "side": side,
            "filter_branch": branch,
            "metric": metric,
            "unit": "%" if metric == "rms_adc" else "Hz",
            "n_paired_participants": int(values.size),
            "median_change": float(np.median(values)),
            "q1_change": float(np.quantile(values, 0.25)),
            "q3_change": float(np.quantile(values, 0.75)),
            "change_definition": "100*(R5-R1)/abs(R1)" if metric == "rms_adc" else "R5-R1",
            "aggregation": "complete participant-side pairs; one recording median per endpoint; participant median and IQR; descriptive",
        })

    bilateral_paired: list[dict[str, object]] = []
    bilateral_values: dict[tuple[str, str, str], list[float]] = {}
    for (series, branch, metric, _subject_id), sides in bilateral_groups.items():
        if "left" not in sides or "right" not in sides:
            continue
        bilateral_values.setdefault((series, branch, metric), []).append(sides["left"] - sides["right"])
    for (series, branch, metric), items in sorted(bilateral_values.items()):
        values = np.asarray(items, dtype=np.float64)
        bilateral_paired.append({
            "series": series,
            "filter_branch": branch,
            "metric": metric,
            "unit": "ADC counts" if metric == "rms_adc" else "Hz",
            "n_paired_participants": int(values.size),
            "median_left_minus_right": float(np.median(values)),
            "q1_left_minus_right": float(np.quantile(values, 0.25)),
            "q3_left_minus_right": float(np.quantile(values, 0.75)),
            "aggregation": "complete participant-side pairs within series; participant median and IQR; descriptive",
        })

    return {
        "by_side_series": by_side_series,
        "r1_r5_paired": r1_r5_paired,
        "bilateral_paired": bilateral_paired,
    }


def analyze_cohort_arrays(
    recordings: Mapping[tuple[str, str, str], tuple[Iterable[float], Iterable[float]]],
    *,
    gap_s: float = GAP_THRESHOLD_S,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = WINDOW_OVERLAP,
    include_traces: bool = False,
    trace_key: tuple[str, str, str] | None = None,
) -> PipelineResult:
    """Process grouped recordings and add participant-level descriptive summaries.

    Keys are explicit ``(subject_id, side, series)`` tuples. The function only
    derives descriptive summaries from feature rows; it does not create labels
    or perform validated fatigue inference. Overlapping windows are reduced to
    recording medians before participant-level summaries and matched contrasts.
    """

    if not recordings:
        raise ValueError("at least one subject/side/series recording is required")
    if include_traces and trace_key is None:
        raise ValueError("cohort dashboard requires an explicit subject/side/series selection")
    if trace_key is not None and trace_key not in recordings:
        raise ValueError("selected dashboard group is not present in the cohort input")

    tagged = PipelineResult(summary={})
    group_summaries: list[dict[str, object]] = []
    subjects: set[str] = set()
    series_seen: set[str] = set()
    row_fields = (
        "mask_rows", "segment_rows", "feature_rows", "fft_rows", "psd_rows", "stft_rows", "trace_rows"
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
            include_traces=include_traces and key == trace_key,
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

    cohort_summaries = summarize_cohort_features(tagged.feature_rows)
    tagged.cohort_by_side_series_rows = cohort_summaries["by_side_series"]
    tagged.cohort_r1_r5_rows = cohort_summaries["r1_r5_paired"]
    tagged.cohort_bilateral_rows = cohort_summaries["bilateral_paired"]

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
        "cohort_comparisons_computed": True,
        "cohort_summary_rows": len(tagged.cohort_by_side_series_rows),
        "cohort_r1_r5_paired_rows": len(tagged.cohort_r1_r5_rows),
        "cohort_bilateral_paired_rows": len(tagged.cohort_bilateral_rows),
        "cohort_summary_method": (
            "overlapping windows reduced to one median per recording; cohort medians and IQRs use participants; "
            "R1-R5 and left-right summaries use complete within-participant pairs"
        ),
        "data_scope": (
            "caller-supplied grouped CSV; descriptive participant-level summaries by side and series plus matched "
            "R1-R5 and bilateral comparisons; no fatigue labels, activity mask, or validated fatigue inference"
        ),
    }
    return tagged


def read_signal_csv(
    path: Path,
    *,
    series_override: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read canonical CSV or an explicit local-source schema without identity fields.

    The source schema uses ``Relative_Time_s`` and ``ADC_Value``. ``Voltage_mV``
    is optional and ignored because its physical calibration is undocumented.
    Since that schema has no series column, callers must supply an explicit
    R1-R5 ``series_override``; series values are never inferred from filenames.
    """

    recordings: dict[str, tuple[list[float], list[float]]] = {}
    seen_series: set[str] = set()
    previous_series: str | None = None
    previous_time_by_series: dict[str, float] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("input CSV contains duplicate column names")
        field_set = set(fields)
        source_required = {SOURCE_TIME_COLUMN, SOURCE_SIGNAL_COLUMN}
        source_allowed = (source_required, source_required | set(SOURCE_OPTIONAL_COLUMNS))
        is_canonical = field_set == set(INPUT_COLUMNS)
        is_source = field_set in source_allowed
        if not is_canonical and not is_source:
            raise ValueError(
                "input CSV columns must be exactly series,time_s,adc_value or "
                "Relative_Time_s,ADC_Value with optional Voltage_mV"
            )
        if is_source and series_override is None:
            raise ValueError("source-schema CSV has no series column; pass an explicit --series R1-R5")
        if is_canonical and series_override is not None:
            raise ValueError("--series is only for source-schema CSV without a series column")
        if series_override is not None and series_override not in SERIES:
            raise ValueError("series must be R1-R5")
        time_column = "time_s" if is_canonical else SOURCE_TIME_COLUMN
        signal_column = "adc_value" if is_canonical else SOURCE_SIGNAL_COLUMN
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"unexpected extra field on CSV line {line_number}")
            required_row_fields = ("series", time_column, signal_column) if is_canonical else (time_column, signal_column)
            if any(row.get(field) is None for field in required_row_fields):
                raise ValueError(f"missing field on CSV line {line_number}")
            if not row[time_column].strip() or not row[signal_column].strip():
                raise ValueError(f"missing required sample value on CSV line {line_number}")
            series = str(row["series"]).strip() if is_canonical else str(series_override)
            if series not in SERIES:
                raise ValueError("series values must be R1-R5")
            if series != previous_series:
                if series in seen_series:
                    raise ValueError("each R1-R5 series must occupy one contiguous CSV block")
                seen_series.add(series)
                previous_series = series
            try:
                time_s = float(row[time_column])
                adc_value = float(row[signal_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value on CSV line {line_number}") from error
            if not math.isfinite(time_s) or not math.isfinite(adc_value):
                raise ValueError(f"CSV values must be finite on line {line_number}")
            previous_time = previous_time_by_series.get(series)
            if previous_time is not None and time_s <= previous_time:
                if time_s == previous_time:
                    raise ValueError(f"duplicate sample timestamp within series {series}")
                raise ValueError(f"timestamps must increase strictly within series {series}")
            previous_time_by_series[series] = time_s
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
    is_cohort = result.summary.get("analysis_mode") == "grouped_cohort_csv"
    if not is_cohort:
        for name in (
            "cohort_by_side_series.csv",
            "cohort_r1_r5_paired.csv",
            "cohort_bilateral_paired.csv",
        ):
            (output_dir / name).unlink(missing_ok=True)
    for name, columns, rows in (
        ("segments.csv", SEGMENT_COLUMNS, result.segment_rows),
        ("mask.csv", MASK_COLUMNS, result.mask_rows),
        ("features.csv", FEATURE_COLUMNS, result.feature_rows),
        ("fft.csv", FFT_COLUMNS, result.fft_rows),
        ("psd.csv", PSD_COLUMNS, result.psd_rows),
        ("stft.csv", STFT_COLUMNS, result.stft_rows),
    ):
        if is_cohort:
            columns = COHORT_PREFIX_COLUMNS + columns
        _write_csv(output_dir / name, columns, rows)
    if is_cohort:
        _write_csv(
            output_dir / "cohort_by_side_series.csv",
            COHORT_SUMMARY_COLUMNS,
            result.cohort_by_side_series_rows,
        )
        _write_csv(
            output_dir / "cohort_r1_r5_paired.csv",
            COHORT_R1_R5_COLUMNS,
            result.cohort_r1_r5_rows,
        )
        _write_csv(
            output_dir / "cohort_bilateral_paired.csv",
            COHORT_BILATERAL_COLUMNS,
            result.cohort_bilateral_rows,
        )
    if result.summary.get("dashboard_exported"):
        identity_fields = {"subject_id", "participant_id"}
        traces = [{key: value for key, value in row.items() if key not in identity_fields} for row in result.trace_rows]
        features = [{key: value for key, value in row.items() if key not in identity_fields} for row in result.feature_rows]
        stft = [{key: value for key, value in row.items() if key not in identity_fields} for row in result.stft_rows]
        segments = [{key: value for key, value in row.items() if key not in identity_fields} for row in result.segment_rows]
        fft = [{key: value for key, value in row.items() if key not in identity_fields} for row in result.fft_rows]
        psd = [{key: value for key, value in row.items() if key not in identity_fields} for row in result.psd_rows]
        activity_frames, activity_events = derive_candidate_activity(traces)
        _write_csv(output_dir / "activity_candidates.csv", ACTIVITY_CANDIDATE_COLUMNS, activity_events)
        result.summary["dashboard_activity_detection"] = (
            "exploratory Otsu threshold on 50 ms RMS windows with 50% overlap of the selected filtered trace; "
            "candidate activity only, not verified contraction/rest/repetition/fatigue"
        )
        result.summary["dashboard_candidate_event_count"] = len(activity_events)
        result.summary["dashboard_candidate_frame_count"] = len(activity_frames)
        html = render_dashboard_html(
            traces, features, stft, segments, fft, psd,
            activity_frames=activity_frames, activity_events=activity_events,
        )
        (output_dir / "dashboard.html").write_text(html, encoding="utf-8")
    else:
        # Keep the known output set deterministic even when the dashboard is disabled.
        _write_csv(output_dir / "activity_candidates.csv", ACTIVITY_CANDIDATE_COLUMNS, [])
        (output_dir / "dashboard.html").unlink(missing_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esquemas para --input:\n"
            "  Canonico: series,time_s,adc_value\n"
            "  Fuente local: Relative_Time_s,ADC_Value[,Voltage_mV]; requiere --series R1-R5.\n"
            "  Voltage_mV se ignora; no se asume calibracion fisica.\n\n"
            "Ejemplo publico sintetico (no usa grabaciones del estudio):\n"
            "  python src\\reproducir_emg_referencia.py --demo --output-dir <PRIVATE_TEMP> --overwrite"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="single-record CSV using the canonical or explicit local-source schema",
    )
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
    parser.add_argument(
        "--series",
        choices=SERIES,
        help="required for source-schema --input CSVs without a series column; never inferred from a filename",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow replacing outputs in an existing directory")
    parser.add_argument("--dashboard", action="store_true", help="write a local interactive HTML viewer; cohort mode requires one selected group")
    parser.add_argument("--dashboard-subject", help="subject pseudonym to show in the dashboard for cohort inputs")
    parser.add_argument("--dashboard-side", choices=SIDES, help="side to show in the dashboard for cohort inputs")
    parser.add_argument("--dashboard-series", choices=SERIES, help="series to show in the dashboard for cohort inputs")
    args = parser.parse_args(argv)
    if args.series is not None and args.input is None:
        parser.error("--series can only be used with --input")
    dashboard_selection = (args.dashboard_subject, args.dashboard_side, args.dashboard_series)
    if any(value is not None for value in dashboard_selection):
        if not args.dashboard or args.cohort_input is None and args.cohort_fragments is None:
            raise ValueError("dashboard group fields require --dashboard with a cohort input")
        if any(value is None for value in dashboard_selection):
            raise ValueError("dashboard group selection requires subject, side, and series")
    if args.dashboard and (args.cohort_input is not None or args.cohort_fragments is not None) and any(value is None for value in dashboard_selection):
        raise ValueError("cohort dashboard requires --dashboard-subject, --dashboard-side, and --dashboard-series")
    trace_key = tuple(dashboard_selection) if all(value is not None for value in dashboard_selection) else None
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
        result = analyze_arrays(read_signal_csv(input_path), include_traces=args.dashboard)
        result.summary["input_source"] = input_source
        result.summary["synthetic_fixture"] = (
            "deterministic multisine plus low-amplitude Gaussian noise; "
            "software fixture, not a physiological EMG simulator"
        )
        result.summary["data_scope"] = "synthetic fixtures only; no study results are reproduced and no cohort data are used"
    else:
        if args.cohort_fragments is not None:
            recordings = read_cohort_signal_fragments(selected_inputs)
            result = analyze_cohort_arrays(recordings, include_traces=args.dashboard, trace_key=trace_key)
            result.summary["input_source"] = "cohort_fragments"
            result.summary["fragment_file_count"] = len(selected_inputs)
            result.summary["sample_count"] = sum(len(times) for times, _ in recordings.values())
            result.summary["data_scope"] = (
                "caller-supplied grouped CSV fragments reassembled by pseudonymized group and original timestamps; "
                "descriptive participant-level summaries by side and series plus matched R1-R5 and bilateral comparisons; "
                "no fatigue labels, activity mask, or validated fatigue inference"
            )
        elif args.cohort_input is not None:
            result = analyze_cohort_arrays(read_cohort_signal_csv(selected_inputs[0]), include_traces=args.dashboard, trace_key=trace_key)
            result.summary["input_source"] = "cohort_csv"
        else:
            result = analyze_arrays(
                read_signal_csv(selected_inputs[0], series_override=args.series),
                include_traces=args.dashboard,
            )
            result.summary["input_source"] = "csv_input"
            result.summary["data_scope"] = (
                "caller-supplied CSV without identity columns; single-recording processing only; "
                "no cohort-level study results are computed"
            )
        output_dir = _ensure_output_dir(paths["output_dir"], overwrite=args.overwrite)
    if not args.demo and (args.cohort_input is not None or args.cohort_fragments is not None):
        if args.cohort_fragments is None:
            result.summary["data_scope"] = (
                "caller-supplied grouped CSV; descriptive participant-level summaries by side and series plus matched "
                "R1-R5 and bilateral comparisons; no fatigue labels, activity mask, or validated fatigue inference"
            )
    if args.dashboard:
        result.summary["dashboard_exported"] = True
        result.summary["dashboard_scope"] = "selected recording/group only; no model scores, fatigue classes, or participant identifier embedded"
    write_result(result, output_dir)
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
