"""Conservative, traceable filter-sensitivity analysis for stored sEMG channels.

This tool does not establish that the stored channel is raw EMG and does not
select a physiologically optimal filter. Per-record traces are written only to
the configured private directory; the project output is aggregated.
"""

from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


GAP_THRESHOLD_S = 0.0015
GAP_FLOAT_TOLERANCE_S = 1e-9
FILTER_BRANCHES = ("control", "hp20", "hp40", "bp20_400", "fir_bp20_400")
FILTER_PARAMS = {
    "hp20": {"type": "Butterworth high-pass", "cutoff_hz": 20.0, "prototype_order": 4},
    "hp40": {"type": "Butterworth high-pass", "cutoff_hz": 40.0, "prototype_order": 4},
    "bp20_400": {"type": "Butterworth band-pass", "cutoffs_hz": [20.0, 400.0], "prototype_order": 4},
    "fir_bp20_400": {"type": "Hamming-window FIR band-pass", "cutoffs_hz": [20.0, 400.0], "taps": 101},
}


def split_contiguous_runs(times: np.ndarray, gap_s: float = GAP_THRESHOLD_S) -> list[tuple[int, int]]:
    """Split runs at gaps greater than the floor/adaptive limit plus 1 ns tolerance."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if not math.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("gap_s must be finite and positive")
    if not np.isfinite(t).all():
        raise ValueError("times must be finite")
    if t.size == 0:
        return []
    dt = np.diff(t)
    positive_dt = dt[dt > 0]
    adaptive_limit = 1.5 * float(np.median(positive_dt)) if positive_dt.size else 0.0
    effective_limit = max(float(gap_s), adaptive_limit) + GAP_FLOAT_TOLERANCE_S
    breaks = np.flatnonzero((dt > effective_limit) | (dt <= 0)) + 1
    edges = np.concatenate(([0], breaks, [t.size]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _butterworth_prototype(order: int) -> np.ndarray:
    return np.asarray(
        [np.exp(1j * np.pi * (2 * k + order + 1) / (2 * order)) for k in range(order)],
        dtype=complex,
    )


def _pair_roots(roots: np.ndarray) -> list[tuple[complex, complex]]:
    pending = [complex(x) for x in roots]
    pairs: list[tuple[complex, complex]] = []
    while pending:
        first = pending.pop(0)
        if not pending:
            raise ValueError("filter roots cannot be paired into real second-order sections")
        target = first.conjugate()
        index = min(range(len(pending)), key=lambda i: abs(pending[i] - target))
        second = pending.pop(index)
        if abs(second - target) > 1e-6 and abs(first.imag) > 1e-8:
            raise ValueError("filter design produced unpaired complex roots")
        pairs.append((first, second))
    return pairs


def _sos_from_zpk(zeros: np.ndarray, poles: np.ndarray, fs_hz: float, gain_frequency_hz: float) -> np.ndarray:
    zero_pairs = _pair_roots(np.asarray(zeros, dtype=complex))
    pole_pairs = _pair_roots(np.asarray(poles, dtype=complex))
    if len(zero_pairs) != len(pole_pairs):
        raise ValueError("zero and pole counts differ")
    sections = np.empty((len(pole_pairs), 6), dtype=float)
    for i, ((z0, z1), (p0, p1)) in enumerate(zip(zero_pairs, pole_pairs)):
        b = np.asarray([1.0, -(z0 + z1).real, (z0 * z1).real])
        a = np.asarray([1.0, -(p0 + p1).real, (p0 * p1).real])
        sections[i] = np.concatenate((b, a))
    omega = 2.0 * np.pi * gain_frequency_hz / fs_hz
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    response = 1.0 + 0.0j
    for sec in sections:
        response *= (sec[0] + sec[1] * z1 + sec[2] * z2) / (sec[3] + sec[4] * z1 + sec[5] * z2)
    magnitude = abs(response)
    if not math.isfinite(magnitude) or magnitude < 1e-15:
        raise ValueError("filter gain normalization is singular")
    sections[0, :3] *= 1.0 / magnitude
    if not np.isfinite(sections).all():
        raise ValueError("filter design produced nonfinite coefficients")
    for sec in sections:
        if np.any(np.abs(np.roots(sec[3:6])) >= 1.0 + 1e-10):
            raise ValueError("filter design is not stable")
    return sections


def design_butterworth_sos(
    fs_hz: float,
    kind: str,
    cutoffs_hz: tuple[float, ...],
    order: int = 4,
) -> np.ndarray:
    """Design Butterworth low-pass, high-pass, or band-pass SOS via bilinear mapping."""
    if not math.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    if not isinstance(order, int) or order < 2 or order % 2:
        raise ValueError("order must be an even integer of at least 2")
    nyquist = fs_hz / 2.0
    cuts = tuple(float(x) for x in cutoffs_hz)
    expected = 2 if kind in ("bandpass", "bandstop") else 1
    if kind not in ("lowpass", "highpass", "bandpass") or len(cuts) != expected:
        raise ValueError("kind/cutoff combination is unsupported")
    if not all(math.isfinite(x) and 0.0 < x < nyquist for x in cuts):
        raise ValueError("all cutoffs must be finite and strictly below Nyquist")
    if kind == "bandpass" and not cuts[0] < cuts[1]:
        raise ValueError("band-pass cutoffs must be increasing")

    warped = [2.0 * fs_hz * math.tan(math.pi * f / fs_hz) for f in cuts]
    prototype = _butterworth_prototype(order)
    if kind == "lowpass":
        analog_poles = prototype * warped[0]
        analog_zeros = np.full(order, np.inf, dtype=complex)
        target_frequency = 0.0
    elif kind == "highpass":
        analog_poles = warped[0] / prototype
        analog_zeros = np.zeros(order, dtype=complex)
        target_frequency = nyquist
    else:
        low, high = warped
        bandwidth = high - low
        center_sq = low * high
        transformed = []
        for pole in prototype:
            root = np.sqrt((bandwidth * pole) ** 2 - 4.0 * center_sq)
            transformed.extend(((bandwidth * pole + root) / 2.0, (bandwidth * pole - root) / 2.0))
        analog_poles = np.asarray(transformed, dtype=complex)
        analog_zeros = np.concatenate((np.zeros(order, dtype=complex), np.full(order, np.inf, dtype=complex)))
        target_frequency = math.sqrt(cuts[0] * cuts[1])

    digital_poles = (2.0 * fs_hz + analog_poles) / (2.0 * fs_hz - analog_poles)
    digital_zeros = np.asarray(
        [1.0 + 0j if z == 0 else -1.0 + 0j for z in analog_zeros], dtype=complex
    )
    if np.any(np.abs(digital_poles) >= 1.0):
        raise ValueError("filter design is not stable")
    return _sos_from_zpk(digital_zeros, digital_poles, fs_hz, target_frequency)


def design_notch_sos(fs_hz: float, f0_hz: float, q: float = 30.0) -> np.ndarray:
    """Design a second-order notch; real-data application remains evidence-gated."""
    if not all(math.isfinite(v) for v in (fs_hz, f0_hz, q)) or fs_hz <= 0 or q <= 0:
        raise ValueError("fs_hz and q must be positive and all parameters finite")
    if not 0.0 < f0_hz < fs_hz / 2.0:
        raise ValueError("notch frequency must be strictly below Nyquist")
    omega = 2.0 * np.pi * f0_hz / fs_hz
    alpha = np.sin(omega) / (2.0 * q)
    scale = 1.0 / (1.0 + alpha)
    return np.asarray(
        [[scale, -2.0 * np.cos(omega) * scale, scale, 1.0, -2.0 * np.cos(omega) * scale, (1.0 - alpha) * scale]],
        dtype=float,
    )


def recommended_sos_padlen(sos: np.ndarray) -> int:
    sections = np.asarray(sos).shape[0]
    return 3 * (2 * sections + 1)


def _sos_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    for sec in np.asarray(sos, dtype=float):
        b0, b1, b2, a0, a1, a2 = sec
        if a0 == 0:
            raise ValueError("SOS denominator leading coefficient cannot be zero")
        b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        if out.size:
            x0 = float(out[0])
            dc_denominator = 1.0 + a1 + a2
            dc_gain = (b0 + b1 + b2) / dc_denominator if abs(dc_denominator) > 1e-15 else 0.0
            y0 = dc_gain * x0
            state1 = y0 - b0 * x0
            state2 = b2 * x0 - a2 * y0
        else:
            state1 = 0.0
            state2 = 0.0
        for i in range(out.size):
            value = out[i]
            result = b0 * value + state1
            state1 = b1 * value - a1 * result + state2
            state2 = b2 * value - a2 * result
            out[i] = result
    return out


def apply_iir_zero_phase(x: np.ndarray, sos: np.ndarray, padlen: int) -> tuple[np.ndarray, int]:
    """Forward-backward SOS filtering; returns per-side edge width or 0 if skipped."""
    data = np.asarray(x, dtype=float)
    sections = np.asarray(sos, dtype=float)
    if data.ndim != 1 or sections.ndim != 2 or sections.shape[1] != 6:
        raise ValueError("x must be 1-D and sos must have shape (n_sections, 6)")
    if not np.isfinite(data).all() or not np.isfinite(sections).all():
        raise ValueError("x and sos must be finite")
    if not isinstance(padlen, int) or padlen < 0:
        raise ValueError("padlen must be a nonnegative integer")
    # At least one sample must remain after excluding the padding on both
    # sides; otherwise this run has no interior sample for comparison.
    if data.size == 0 or data.size <= 2 * padlen:
        return data.copy(), 0
    padded = np.pad(data, (padlen, padlen), mode="reflect") if padlen else data.copy()
    forward = _sos_filter(padded, sections)
    backward = _sos_filter(forward[::-1], sections)[::-1]
    result = backward[padlen:-padlen] if padlen else backward
    return result, padlen


def design_fir_bandpass(fs_hz: float, low_hz: float, high_hz: float, taps: int = 101) -> np.ndarray:
    if not math.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    if not 0.0 < low_hz < high_hz < fs_hz / 2.0:
        raise ValueError("band-pass cutoffs must increase and remain below Nyquist")
    if not isinstance(taps, int) or taps < 3 or taps % 2 == 0:
        raise ValueError("FIR tap count must be an odd integer of at least 3")
    center = (taps - 1) / 2.0
    n = np.arange(taps, dtype=float) - center
    lowpass_high = 2.0 * high_hz / fs_hz * np.sinc(2.0 * high_hz / fs_hz * n)
    lowpass_low = 2.0 * low_hz / fs_hz * np.sinc(2.0 * low_hz / fs_hz * n)
    h = (lowpass_high - lowpass_low) * np.hamming(taps)
    gain = abs(np.sum(h * np.exp(-2j * np.pi * ((low_hz + high_hz) / 2.0) * np.arange(taps) / fs_hz)))
    if not math.isfinite(gain) or gain < 1e-15:
        raise ValueError("FIR normalization failed")
    h /= gain
    return h


def apply_fir_reflect(x: np.ndarray, taps: np.ndarray) -> tuple[np.ndarray, int]:
    data = np.asarray(x, dtype=float)
    coefficients = np.asarray(taps, dtype=float)
    if data.ndim != 1 or coefficients.ndim != 1 or coefficients.size % 2 != 1:
        raise ValueError("x and odd-length taps must be one-dimensional")
    if not np.isfinite(data).all() or not np.isfinite(coefficients).all():
        raise ValueError("x and taps must be finite")
    edge = (coefficients.size - 1) // 2
    # A reflected FIR result with no interior after edge trimming cannot
    # contribute a valid sample and must be recorded as skipped.
    if data.size == 0 or data.size <= 2 * edge:
        return data.copy(), 0
    padded = np.pad(data, (edge, edge), mode="reflect") if edge else data.copy()
    result = np.convolve(padded, coefficients, mode="valid")
    return result, edge


def augment_signal(x: np.ndarray, kind: str, params: dict, seed: int) -> np.ndarray:
    """Return a synthetic copy. This function is not invoked on study files."""
    data = np.asarray(x, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("x must be a finite one-dimensional signal")
    if kind not in {"gaussian", "line_50", "line_60", "motion_low_frequency", "transient"}:
        raise ValueError("unknown synthetic noise kind")
    if data.size == 0:
        return data.copy()
    rng = np.random.default_rng(seed)
    out = data.copy()
    centered_rms = float(np.sqrt(np.mean((data - np.mean(data)) ** 2)))
    signal_scale = centered_rms if centered_rms > 0 else float(np.sqrt(np.mean(data**2)))
    if signal_scale == 0:
        raise ValueError("cannot scale synthetic noise from a zero signal")
    if kind == "gaussian":
        snr_db = float(params["snr_db"])
        if not math.isfinite(snr_db):
            raise ValueError("snr_db must be finite")
        noise_rms = centered_rms / (10.0 ** (snr_db / 20.0))
        out += rng.normal(0.0, noise_rms, size=data.size)
    elif kind in {"line_50", "line_60", "motion_low_frequency"}:
        if "fs_hz" not in params:
            raise ValueError("fs_hz is required for sinusoidal perturbations")
        fs_hz = float(params["fs_hz"])
        if not math.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError("fs_hz must be finite and positive")
        default_frequency = 50.0 if kind == "line_50" else 60.0 if kind == "line_60" else 2.0
        frequency = float(params.get("frequency_hz", default_frequency))
        if not 0.0 < frequency < fs_hz / 2.0:
            raise ValueError("synthetic tone must be strictly below Nyquist")
        amplitude_ratio = float(params.get("amplitude_ratio", 0.1 if kind != "motion_low_frequency" else 0.25))
        if not math.isfinite(amplitude_ratio) or amplitude_ratio < 0:
            raise ValueError("amplitude_ratio must be finite and nonnegative")
        phase = float(params.get("phase_rad", rng.uniform(0.0, 2.0 * np.pi)))
        t = np.arange(data.size, dtype=float) / fs_hz
        out += signal_scale * amplitude_ratio * np.sin(2.0 * np.pi * frequency * t + phase)
    else:
        count = int(params.get("count", 1))
        width = int(params.get("width_samples", 1))
        amplitude_ratio = float(params.get("amplitude_ratio", 3.0))
        if count < 1 or width < 1 or not math.isfinite(amplitude_ratio):
            raise ValueError("transient count/width and amplitude_ratio are invalid")
        positions = rng.choice(data.size, size=min(count, data.size), replace=False)
        pulse = np.hanning(width + 2)[1:-1] if width > 1 else np.ones(1)
        for pos in positions:
            lo = max(0, int(pos) - width // 2)
            hi = min(data.size, lo + width)
            sign = -1.0 if rng.integers(0, 2) else 1.0
            out[lo:hi] += sign * amplitude_ratio * signal_scale * pulse[: hi - lo]
    return out


def _read_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    signal: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Relative_Time_s", "ADC_Value"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"missing required columns in {path.name}")
        for line_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row["Relative_Time_s"]))
                signal.append(float(row["ADC_Value"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value at line {line_number} in {path.name}") from exc
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(signal, dtype=np.float64)
    if t.size != x.size or not np.isfinite(t).all() or not np.isfinite(x).all():
        raise ValueError(f"nonfinite or inconsistent values in {path.name}")
    return t, x


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _estimate_fs(times: np.ndarray) -> tuple[float | None, float | None]:
    dt = np.diff(times)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None, None
    median = float(np.median(dt))
    jitter = float((np.percentile(dt, 95) - np.percentile(dt, 5)) / median) if median > 0 else None
    return (1.0 / median if median > 0 else None), jitter


def _band_power(signal: np.ndarray, fs_hz: float, low_hz: float, high_hz: float) -> float:
    x = np.asarray(signal, dtype=float)
    if x.size < 32 or fs_hz <= 0:
        return float("nan")
    x = x - np.mean(x)
    window = np.hanning(x.size)
    scale = fs_hz * float(np.sum(window**2))
    if scale <= 0:
        return float("nan")
    spectrum = np.fft.rfft(x * window)
    power = (np.abs(spectrum) ** 2) / scale
    if x.size % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.sum(power[mask]) * (fs_hz / x.size))


def _line_prominence_db(signal: np.ndarray, fs_hz: float, target_hz: float) -> float:
    x = np.asarray(signal, dtype=float)
    if x.size < int(fs_hz * 2) or target_hz <= 0 or target_hz >= fs_hz / 2:
        return float("nan")
    nfft = 1 << int(math.ceil(math.log2(x.size)))
    window = np.hanning(x.size)
    spectrum = np.abs(np.fft.rfft((x - np.mean(x)) * window, n=nfft)) ** 2
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs_hz)
    peak = (frequencies >= target_hz - 1.0) & (frequencies <= target_hz + 1.0)
    local = ((frequencies >= target_hz - 10.0) & (frequencies <= target_hz - 2.0)) | (
        (frequencies >= target_hz + 2.0) & (frequencies <= target_hz + 10.0)
    )
    if not np.any(peak) or not np.any(local):
        return float("nan")
    peak_power = float(np.max(spectrum[peak]))
    floor = float(np.median(spectrum[local]))
    return float(10.0 * np.log10(max(peak_power, 1e-30) / max(floor, 1e-30)))


def _metrics(x: np.ndarray, fs_hz: float) -> dict[str, float]:
    if x.size == 0:
        return {"rms_adc": float("nan"), "peak_to_peak_adc": float("nan"), "power_20_60_adc2": float("nan"), "power_60_150_adc2": float("nan"), "power_150_400_adc2": float("nan"), "prominence_50_db": float("nan"), "prominence_60_db": float("nan")}
    centered = x - np.mean(x)
    return {
        "rms_adc": float(np.sqrt(np.mean(centered**2))),
        "peak_to_peak_adc": float(np.ptp(x)),
        "power_20_60_adc2": _band_power(x, fs_hz, 20.0, 60.0),
        "power_60_150_adc2": _band_power(x, fs_hz, 60.0, 150.0),
        "power_150_400_adc2": _band_power(x, fs_hz, 150.0, 400.0),
        "prominence_50_db": _line_prominence_db(x, fs_hz, 50.0),
        "prominence_60_db": _line_prominence_db(x, fs_hz, 60.0),
    }


def _aggregate_private_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        for branch in FILTER_BRANCHES:
            grouped[(record["record_type"], record["leg"], record["set_index"], branch)].append(
                record["branches"][branch]
            )
    rows = []
    for (record_type, leg, set_index, branch), items in sorted(grouped.items()):
        row = {
            "record_type": record_type,
            "leg": leg,
            "set_index": set_index,
            "filter_branch": branch,
            "records_total": len(items),
            "records_filtered": sum(item["status"] == "applied" for item in items),
            "records_skipped_short": sum(item["status"] == "skipped_short" for item in items),
            "records_not_applicable": sum(item["status"] == "not_applicable_sampling_rate" for item in items),
            "records_with_short_segments": sum(item.get("skipped_segments", 0) > 0 for item in items),
            "short_segments_skipped": int(sum(item.get("skipped_segments", 0) for item in items)),
            "records_with_not_applicable_segments": sum(item.get("not_applicable_segments", 0) > 0 for item in items),
            "segments_not_applicable_sampling_rate": int(sum(item.get("not_applicable_segments", 0) for item in items)),
            "input_samples": int(sum(item["input_samples"] for item in items)),
            "samples_filtered": int(sum(item["samples_filtered"] for item in items)),
            "edge_samples_excluded": int(sum(item["edge_samples_excluded"] for item in items)),
            "common_comparison_samples": int(sum(item["common_comparison_samples"] for item in items)),
        }
        metric_names = ("rms_adc", "rms_change_pct_vs_control", "peak_to_peak_adc", "power_20_60_adc2", "power_60_150_adc2", "power_150_400_adc2", "prominence_50_db", "prominence_60_db")
        for metric in metric_names:
            values = np.asarray([item["metrics_common"].get(metric, float("nan")) for item in items], dtype=float)
            values = values[np.isfinite(values)]
            row[f"records_with_{metric}"] = int(values.size)
            row[f"mean_{metric}"] = float(np.mean(values)) if values.size else ""
            row[f"median_{metric}"] = float(np.median(values)) if values.size else ""
        rows.append(row)
    return rows


def run_filter_comparison(
    manifest_path: Path,
    signal_root: Path,
    private_output: Path,
    aggregate_output: Path,
) -> dict:
    manifest_path = Path(manifest_path)
    signal_root = Path(signal_root)
    private_output = Path(private_output)
    aggregate_output = Path(aggregate_output)
    manifest_hash_before = _sha256(manifest_path)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 140:
        raise ValueError(f"expected 140 manifest records; found {len(manifest)}")
    manifest_relpaths = [Path(item["relative_path"]).as_posix() for item in manifest]
    if len(set(manifest_relpaths)) != len(manifest_relpaths):
        raise ValueError("manifest contains duplicate relative paths")
    actual_relpaths = sorted(p.relative_to(signal_root).as_posix() for p in signal_root.rglob("*.csv"))
    if sorted(manifest_relpaths) != actual_relpaths:
        raise ValueError("private CSV file set does not exactly match the manifest")
    keys = [(item["participant_id"], item["recording_id"], item.get("leg", "unknown")) for item in manifest]
    if len(set(keys)) != len(keys):
        raise ValueError("manifest contains duplicate participant-recording-leg keys")
    participants_expected = len({item["participant_id"] for item in manifest})
    prelim_expected = sum(item["recording_id"].upper().startswith("R0") for item in manifest)
    series_expected = sum(item["recording_id"].upper() in {"R1", "R2", "R3", "R4", "R5"} for item in manifest)
    if participants_expected != 15 or prelim_expected != 2 or series_expected != 138:
        raise ValueError("participant/series/preliminary counts differ from the verified inventory")
    private_output.mkdir(parents=True, exist_ok=True)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    private_records: list[dict] = []
    total_samples = 0
    expected_samples = 0
    input_hashes_unchanged = True
    for item in manifest:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest relative_path must stay below the private signal root")
        path = (signal_root / relative).resolve()
        if signal_root.resolve() not in path.parents:
            raise ValueError("manifest path escapes the private signal root")
        expected_hash = item.get("sha256_uncompressed", "")
        if not path.is_file():
            raise FileNotFoundError(f"private pseudonymized signal is missing: {relative}")
        input_hash = _sha256(path)
        if expected_hash and input_hash != expected_hash:
            raise ValueError(f"input hash differs from manifest for {item['recording_id']}")
        t, original = _read_recording(path)
        if str(len(original)) != str(item["n_samples"]):
            raise ValueError(f"sample count differs from manifest for {item['recording_id']}")
        if item.get("sha256_uncompressed") and input_hash != item["sha256_uncompressed"]:
            input_hashes_unchanged = False
        total_samples += int(original.size)
        expected_samples += int(item["n_samples"])
        runs = split_contiguous_runs(t)
        branches = {name: original.copy() for name in FILTER_BRANCHES}
        applied_masks = {name: np.zeros(original.size, dtype=bool) for name in FILTER_BRANCHES if name != "control"}
        common_mask = np.zeros(original.size, dtype=bool)
        segment_log = []
        record_metric_lists = {name: defaultdict(list) for name in FILTER_BRANCHES}
        branch_counters = {
            name: {"status": "control" if name == "control" else "applied", "input_samples": int(original.size), "samples_filtered": 0, "edge_samples_excluded": 0, "common_comparison_samples": 0, "skipped_segments": 0, "not_applicable_segments": 0}
            for name in FILTER_BRANCHES
        }
        fs_estimations = []
        max_common_edge_record = 0
        for start, end in runs:
            segment_t = t[start:end]
            segment = original[start:end]
            fs_hz, jitter = _estimate_fs(segment_t)
            if fs_hz is None or fs_hz < 10.0:
                for name in FILTER_BRANCHES[1:]:
                    branch_counters[name]["not_applicable_segments"] += 1
                    branch_counters[name]["status"] = "not_applicable_sampling_rate"
                segment_log.append({"start": start, "end": end, "fs_estimated_hz": fs_hz, "relative_jitter_p95_p5": jitter, "status": "invalid_timebase"})
                continue
            fs_estimations.append(fs_hz)
            per_segment_edges = {"control": 0}
            per_segment_ok = {"control": True}
            for name in ("hp20", "hp40", "bp20_400"):
                if name == "hp20":
                    kind, cuts = "highpass", (20.0,)
                elif name == "hp40":
                    kind, cuts = "highpass", (40.0,)
                else:
                    kind, cuts = "bandpass", (20.0, 400.0)
                try:
                    sos = design_butterworth_sos(fs_hz, kind, cuts, order=4)
                except ValueError:
                    branch_counters[name]["not_applicable_segments"] += 1
                    branch_counters[name]["status"] = "not_applicable_sampling_rate"
                    per_segment_edges[name] = 0
                    per_segment_ok[name] = False
                    continue
                pad = recommended_sos_padlen(sos)
                filtered, edge = apply_iir_zero_phase(segment, sos, pad)
                branches[name][start:end] = filtered
                per_segment_edges[name] = edge
                per_segment_ok[name] = edge > 0
                if edge > 0:
                    applied_masks[name][start + edge : end - edge] = True
                    branch_counters[name]["samples_filtered"] += int(end - start - 2 * edge)
                    branch_counters[name]["edge_samples_excluded"] += 2 * edge
                else:
                    branch_counters[name]["skipped_segments"] += 1
                    branch_counters[name]["status"] = "skipped_short"
            try:
                taps = design_fir_bandpass(fs_hz, 20.0, 400.0, taps=101)
                filtered, edge = apply_fir_reflect(segment, taps)
                branches["fir_bp20_400"][start:end] = filtered
                per_segment_edges["fir_bp20_400"] = edge
                per_segment_ok["fir_bp20_400"] = edge > 0
                if edge > 0:
                    applied_masks["fir_bp20_400"][start + edge : end - edge] = True
                    branch_counters["fir_bp20_400"]["samples_filtered"] += int(end - start - 2 * edge)
                    branch_counters["fir_bp20_400"]["edge_samples_excluded"] += 2 * edge
                else:
                    branch_counters["fir_bp20_400"]["skipped_segments"] += 1
                    branch_counters["fir_bp20_400"]["status"] = "skipped_short"
            except ValueError:
                branch_counters["fir_bp20_400"]["not_applicable_segments"] += 1
                branch_counters["fir_bp20_400"]["status"] = "not_applicable_sampling_rate"
                per_segment_edges["fir_bp20_400"] = 0
                per_segment_ok["fir_bp20_400"] = False

            all_applied = all(per_segment_ok.get(name, False) for name in FILTER_BRANCHES[1:])
            edge_common = max(per_segment_edges.get(name, 0) for name in FILTER_BRANCHES[1:])
            if all_applied and end - start > 2 * edge_common:
                common_mask[start + edge_common : end - edge_common] = True
                max_common_edge_record = max(max_common_edge_record, edge_common)
                for name in FILTER_BRANCHES:
                    record_metric_lists[name]["_ranges"].append((start + edge_common, end - edge_common, fs_hz))
            segment_log.append({"start": start, "end": end, "fs_estimated_hz": fs_hz, "relative_jitter_p95_p5": jitter, "edge_by_branch": per_segment_edges, "all_branches_applied": all_applied})

        for name in FILTER_BRANCHES:
            branch_counters[name]["common_comparison_samples"] = int(np.sum(common_mask))
            if name != "control":
                if branch_counters[name]["samples_filtered"] > 0:
                    branch_counters[name]["status"] = "applied"
                elif branch_counters[name]["skipped_segments"] > 0:
                    branch_counters[name]["status"] = "skipped_short"
                else:
                    branch_counters[name]["status"] = "not_applicable_sampling_rate"
        metrics_common = {}
        for name in FILTER_BRANCHES:
            metrics_parts = []
            for lo, hi, fs_hz in record_metric_lists[name]["_ranges"]:
                metrics_parts.append(_metrics(branches[name][lo:hi], fs_hz))
            metric_names = ("rms_adc", "peak_to_peak_adc", "power_20_60_adc2", "power_60_150_adc2", "power_150_400_adc2", "prominence_50_db", "prominence_60_db")
            combined = {}
            for metric in metric_names:
                values = [part[metric] for part in metrics_parts if math.isfinite(part[metric])]
                combined[metric] = float(np.mean(values)) if values else float("nan")
            metrics_common[name] = combined
        control_rms = metrics_common["control"]["rms_adc"]
        for name in FILTER_BRANCHES:
            value = metrics_common[name]["rms_adc"]
            metrics_common[name]["rms_change_pct_vs_control"] = (
                100.0 * (value - control_rms) / control_rms if math.isfinite(value) and math.isfinite(control_rms) and control_rms else float("nan")
            )

        record_id = item["recording_id"]
        pid = item["participant_id"]
        leg = item.get("leg", "unknown")
        safe_stem = f"{pid}_{record_id}_{leg}"
        private_record_path = private_output / f"{safe_stem}.npz"
        arrays = {"times_s": t, "adc_original": original, "common_comparison_mask": common_mask}
        arrays.update({name: branches[name].astype(np.float32) for name in FILTER_BRANCHES})
        arrays.update({f"applied_{name}": mask for name, mask in applied_masks.items()})
        np.savez_compressed(private_record_path, **arrays)
        record_type = "preliminary_R0" if record_id.upper().startswith("R0") else "series_R1_R5"
        private_records.append({
            "participant_id": pid,
            "recording_id": record_id,
            "leg": leg,
            "set_index": item.get("set_index", "unknown"),
            "record_type": record_type,
            "input_path": str(relative),
            "input_sha256": input_hash,
            "output_path": str(private_record_path),
            "output_sha256": _sha256(private_record_path),
            "n_samples": int(original.size),
            "n_runs": len(runs),
            "common_comparison_samples": int(np.sum(common_mask)),
            "fs_estimated_median_hz": float(np.median(fs_estimations)) if fs_estimations else float("nan"),
            "branch_counters": branch_counters,
            "branches": {
                name: {
                    **branch_counters[name],
                    "metrics_common": metrics_common[name],
                }
                for name in FILTER_BRANCHES
            },
            "segments": segment_log,
            "notch_status": "not_applied_no_confirmed_local_PLI_evidence",
        })

    if total_samples != expected_samples:
        raise ValueError("sample totals do not reconcile with manifest")
    for record in private_records:
        current_path = signal_root / Path(record["input_path"])
        if _sha256(current_path) != record["input_sha256"]:
            input_hashes_unchanged = False
            break
    if not input_hashes_unchanged:
        raise ValueError("a pseudonymized input changed during processing")
    if manifest_hash_before != _sha256(manifest_path):
        raise ValueError("manifest changed during processing")
    aggregated = _aggregate_private_records(private_records)
    if any("participant_id" in row or "recording_id" in row for row in aggregated):
        raise AssertionError("public aggregation contains a direct record identifier")
    public_fields = list(aggregated[0].keys()) if aggregated else []
    with aggregate_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields)
        writer.writeheader()
        writer.writerows(aggregated)
    receipt = {
        "analysis": "filter_sensitivity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "manifest_sha256": manifest_hash_before,
        "records": len(private_records),
        "participants": len({record["participant_id"] for record in private_records}),
        "series_records_R1_R5": sum(record["record_type"] == "series_R1_R5" for record in private_records),
        "preliminary_records_R0_R0_1": sum(record["record_type"] == "preliminary_R0" for record in private_records),
        "samples": total_samples,
        "branches": FILTER_BRANCHES,
        "parameters": FILTER_PARAMS,
        "gap_threshold_s": GAP_THRESHOLD_S,
        "signal_column": "ADC_Value",
        "channel_mode": "unknown_RAW_RECT_ENV",
        "sampling_clock": "estimated_from_timestamps_not_hardware_verified",
        "notch": "not_applied_no_confirmed_local_PLI_evidence",
        "synthetic_augmentation": "not applied to study files; function only",
        "filter_selection": "none; descriptive sensitivity comparison only",
        "elapsed_seconds": time.time() - started,
        "aggregate_csv_sha256": _sha256(aggregate_output),
        "records_private": private_records,
    }
    receipt_path = private_output / "receipt_filter_sensitivity_v1.json"
    receipt_path.write_text(json.dumps(_json_safe(receipt), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {
        "records": len(private_records),
        "participants": receipt["participants"],
        "samples": total_samples,
        "aggregate_rows": len(aggregated),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "receipt": str(receipt_path),
        "aggregate": str(aggregate_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signal-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    result = run_filter_comparison(args.manifest, args.signal_root, args.private_output, args.aggregate_output)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
