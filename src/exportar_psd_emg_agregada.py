"""Export participant-balanced PSD curves from locally verified EMG arrays.

The analysis uses only the 15-person local cohort, splits at timestamp gaps,
and exports participant-level aggregates without subject or recording keys.
It describes the stored digital channel; it cannot identify artifact source,
confirm device mode, or label physiological fatigue.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEGMENT_SECONDS = 8.0
GRID_FS_HZ = 1000.0
MAX_INTERVAL_SPREAD_RATIO = 0.005
GAP_THRESHOLD_S = 0.0015
MAX_FREQUENCY_HZ = 450.0
PUBLIC_FIELDS = (
    "cohort", "n_participants", "n_recordings", "n_windows", "frequency_hz",
    "median_psd_adc2_per_hz", "q25_psd_adc2_per_hz", "q75_psd_adc2_per_hz",
)


def _load_diagnostic_module():
    module_path = Path(__file__).with_name("diagnostico_baja_frecuencia_emg.py")
    spec = importlib.util.spec_from_file_location("diagnostico_baja_frecuencia_emg", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("the verified timestamp/filter utility could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = _load_diagnostic_module()


def common_frequency_grid(fs_grid: float = GRID_FS_HZ) -> np.ndarray:
    if not math.isfinite(fs_grid) or fs_grid <= 2.0 * MAX_FREQUENCY_HZ:
        raise ValueError("frequency grid sample rate must exceed twice the maximum plotted frequency")
    count = int(math.floor(MAX_FREQUENCY_HZ * SEGMENT_SECONDS)) + 1
    return np.arange(count, dtype=np.float64) / SEGMENT_SECONDS


def segmented_psd(
    times_s,
    signal,
    fs_grid: float = GRID_FS_HZ,
    gap_s: float = GAP_THRESHOLD_S,
    max_interval_spread_ratio: float = MAX_INTERVAL_SPREAD_RATIO,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Average 8-second Hann PSD windows without bridging gaps or irregular timing."""
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size or times.size < 2:
        raise ValueError("time and signal must be matching one-dimensional arrays")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("time and signal must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not math.isfinite(gap_s) or gap_s <= 0:
        raise ValueError("gap threshold must be positive")
    if not math.isfinite(max_interval_spread_ratio) or max_interval_spread_ratio < 0:
        raise ValueError("interval spread limit must be nonnegative")
    frequency_grid = common_frequency_grid(fs_grid)
    spectra = []
    for start, end in DIAGNOSTIC.FILTER_ENGINE.split_contiguous_runs(times, gap_s):
        local_t = times[start:end]
        local_x = values[start:end]
        fs_hz, spread = DIAGNOSTIC.FILTER_ENGINE._estimate_fs(local_t)
        if fs_hz is None or spread is None or spread > max_interval_spread_ratio:
            continue
        if fs_hz <= 2.0 * MAX_FREQUENCY_HZ:
            continue
        segment_n = int(round(SEGMENT_SECONDS * fs_hz))
        if segment_n < 8:
            continue
        n_windows = local_x.size // segment_n
        if n_windows == 0:
            continue
        for window_index in range(n_windows):
            lo = window_index * segment_n
            hi = lo + segment_n
            window = local_x[lo:hi]
            frequency, psd = DIAGNOSTIC.periodogram_psd(window, fs_hz)
            spectra.append(np.interp(frequency_grid, frequency, psd, left=0.0, right=0.0))
    if not spectra:
        return frequency_grid, np.zeros_like(frequency_grid), 0
    return frequency_grid, np.mean(np.vstack(spectra), axis=0), len(spectra)


def aggregate_participant_spectra(
    frequency_hz: np.ndarray,
    participant_record_spectra: dict[str, list[np.ndarray]],
    cohort: str,
) -> list[dict]:
    """Take median across records within participant, then summarize participants."""
    if not cohort or not participant_record_spectra:
        return []
    participant_medians = []
    record_count = 0
    for participant, records in sorted(participant_record_spectra.items()):
        if not participant or not records:
            continue
        arrays = [np.asarray(row, dtype=np.float64) for row in records]
        if any(array.shape != frequency_hz.shape or not np.isfinite(array).all() or np.any(array < 0)
               for array in arrays):
            raise ValueError("participant PSD arrays must be finite, nonnegative, and match the frequency grid")
        participant_medians.append(np.median(np.vstack(arrays), axis=0))
        record_count += len(arrays)
    if not participant_medians:
        return []
    stacked = np.vstack(participant_medians)
    median_psd = np.median(stacked, axis=0)
    q25_psd = np.percentile(stacked, 25, axis=0)
    q75_psd = np.percentile(stacked, 75, axis=0)
    return [
        {
            "cohort": cohort,
            "n_participants": int(stacked.shape[0]),
            "n_recordings": int(record_count),
            "n_windows": None,
            "frequency_hz": float(frequency),
            "median_psd_adc2_per_hz": float(median),
            "q25_psd_adc2_per_hz": float(q25),
            "q75_psd_adc2_per_hz": float(q75),
        }
        for frequency, median, q25, q75 in zip(frequency_hz, median_psd, q25_psd, q75_psd)
    ]


def assert_public_fields_safe(fields) -> None:
    forbidden = {
        "participant_id", "recording_id", "subject_id", "pseudonym", "recording_key",
        "file_path", "input_path", "filename", "file_name", "source_file",
    }
    if any(field.lower() in forbidden for field in fields):
        raise ValueError("public PSD schema contains a participant or record-level key")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    assert_public_fields_safe(PUBLIC_FIELDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _participant_balance(participant_map: dict[str, list[dict]], fs_grid: float):
    frequency_grid = common_frequency_grid(fs_grid)
    spectra_by_participant: dict[str, list[np.ndarray]] = {}
    total_windows = 0
    records_used = 0
    for person, records in participant_map.items():
        record_spectra = []
        for record in records:
            frequency, psd, n_windows = segmented_psd(record["times"], record["signal"], fs_grid)
            if n_windows == 0:
                continue
            if not np.array_equal(frequency, frequency_grid):
                raise RuntimeError("internal frequency-grid mismatch")
            record_spectra.append(psd)
            total_windows += n_windows
            records_used += 1
        if record_spectra:
            spectra_by_participant[person] = record_spectra
    return frequency_grid, spectra_by_participant, total_windows, records_used


def _scale_y(value: float, lo: float, hi: float, top: float, height: float) -> float:
    eps = 1e-30
    log_lo = math.log10(max(lo, eps))
    log_hi = math.log10(max(hi, lo * 1.001))
    log_v = math.log10(max(value, eps))
    return top + height - (log_v - log_lo) / (log_hi - log_lo) * height


def _svg_figure(path: Path, rows: list[dict]) -> None:
    primary = [r for r in rows if r["cohort"] == "R1-R5"]
    if not primary:
        raise ValueError("participant-level R1-R5 PSD aggregate is required for the figure")
    width, height = 1120, 640
    lefts = [95, 595]
    plot_width, plot_height = 430, 415
    panels = [(0.5, 10.0, "Low-frequency detail (0.5–10 Hz)"),
              (20.0, 450.0, "Main EMG band view (20–450 Hz)")]
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#253746}.title{font-size:21px;font-weight:700}.panel{font-size:15px;font-weight:700}.axis{font-size:12px}.note{font-size:12px}</style>',
        '<text class="title" x="560" y="31" text-anchor="middle">Participant-balanced PSD of the stored EMG channel</text>',
    ]
    for panel_index, (low_f, high_f, title) in enumerate(panels):
        left, top = lefts[panel_index], 90
        subset = sorted((r for r in primary if low_f <= float(r["frequency_hz"]) <= high_f),
                        key=lambda r: float(r["frequency_hz"]))
        positive_values = [float(r[key]) for r in subset for key in
                           ("q25_psd_adc2_per_hz", "q75_psd_adc2_per_hz") if float(r[key]) > 0]
        ymin = max(min(positive_values, default=1e-6) * 0.5, 1e-12)
        ymax = max(positive_values, default=1.0) * 2.0
        log_low, log_high = math.log10(low_f), math.log10(high_f)
        x_map = lambda value: left + (math.log10(value) - log_low) / (log_high - log_low) * plot_width
        y_map = lambda value: _scale_y(value, ymin, ymax, top, plot_height)
        content.append(f'<text class="panel" x="{left}" y="65">{title}</text>')
        content.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#718096"/>')
        for exponent in range(math.ceil(math.log10(ymin)), math.floor(math.log10(ymax)) + 1):
            tick = 10.0 ** exponent
            y = y_map(tick)
            content.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}" stroke="#e5eaf0"/>')
            content.append(f'<text class="axis" x="{left-8}" y="{y+4:.2f}" text-anchor="end">10^{exponent}</text>')
        tick_values = [v for v in (0.5, 1, 2, 5, 10, 20, 50, 60, 100, 200, 400) if low_f <= v <= high_f]
        for tick in tick_values:
            x = x_map(tick)
            content.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}" stroke="#f0f2f5"/>')
            content.append(f'<text class="axis" x="{x:.2f}" y="{top+plot_height+20}" text-anchor="middle">{tick:g}</text>')
            if panel_index == 1 and tick in (50, 60):
                content.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}" stroke="#c05621" stroke-dasharray="5 4"/>')
        upper = " ".join(f'{x_map(float(r["frequency_hz"])):.2f},{y_map(float(r["q75_psd_adc2_per_hz"])):.2f}' for r in subset)
        lower = " ".join(f'{x_map(float(r["frequency_hz"])):.2f},{y_map(float(r["q25_psd_adc2_per_hz"])):.2f}' for r in reversed(subset))
        median = " ".join(f'{x_map(float(r["frequency_hz"])):.2f},{y_map(float(r["median_psd_adc2_per_hz"])):.2f}' for r in subset)
        content.append(f'<polygon points="{upper} {lower}" fill="#56B4E9" fill-opacity="0.25" stroke="none"/>')
        content.append(f'<polyline points="{median}" fill="none" stroke="#0072B2" stroke-width="2"/>')
    content.extend([
        '<text class="axis" x="560" y="565" text-anchor="middle">Frequency (Hz; logarithmic axis)</text>',
        '<text class="note" x="560" y="592" text-anchor="middle">Blue: median of 15 participant medians; shaded: participant IQR. 8-s Hann windows; gaps split; 1-kHz timestamp-based grid.</text>',
        '<text class="note" x="560" y="615" text-anchor="middle">PSD is in reported ADC²/Hz. Mode, hardware chain, calibration and artifact source remain unknown; curve is descriptive, not a fatigue or denoising score.</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def run_analysis(private_root: Path, output_csv: Path, figure_path: Path, output_receipt: Path) -> dict:
    prior_path = private_root / "receipt_filter_sensitivity_v1.json"
    if not prior_path.is_file():
        raise FileNotFoundError(f"verified filter-sensitivity receipt not found: {prior_path}")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("records") != 140 or prior.get("samples") != 4256257:
        raise ValueError("verified local-cohort receipt does not match 140 files and 4,256,257 samples")
    array_root = Path(private_root)
    if not array_root.is_dir():
        raise FileNotFoundError(f"verified private filter arrays unavailable: {array_root}")
    expected = {Path(item["output_path"]).name: item for item in prior["records_private"]}
    files = sorted(array_root.glob("*.npz"))
    if len(files) != 140 or len(expected) != 140:
        raise ValueError("expected exactly 140 privately stored arrays")
    primary: dict[str, list[dict]] = {}
    preliminary: dict[str, list[dict]] = {}
    input_hashes = {}
    samples = 0
    participants = set()
    skipped_recordings = []
    windows_primary = windows_preliminary = 0
    used_primary = used_preliminary = 0
    frequency_grid = common_frequency_grid()
    for path in files:
        metadata = expected.get(path.name)
        if metadata is None:
            raise ValueError(f"unreceipted private array encountered: {path.name}")
        digest = _sha256(path)
        if digest != metadata.get("output_sha256"):
            raise ValueError(f"private array hash mismatch: {path.name}")
        input_hashes[path.name] = digest
        participant = metadata["participant_id"]
        participants.add(participant)
        with np.load(path, allow_pickle=False) as data:
            times = np.asarray(data["times_s"], dtype=np.float64)
            signal = np.asarray(data["adc_original"], dtype=np.float64)
        samples += int(signal.size)
        frequency, psd, windows = segmented_psd(times, signal)
        if not np.array_equal(frequency, frequency_grid):
            raise RuntimeError("frequency grid differs within the analysis")
        if windows == 0:
            skipped_recordings.append(path.name)
            continue
        record = {"times": times, "signal": signal, "psd": psd, "windows": windows}
        if metadata["record_type"] == "series_R1_R5":
            primary.setdefault(participant, []).append(record)
            windows_primary += windows
            used_primary += 1
        else:
            preliminary.setdefault(participant, []).append(record)
            windows_preliminary += windows
            used_preliminary += 1
    if samples != 4256257 or len(participants) != 15:
        raise ValueError("cohort sample count or participant count changed from verified source receipt")
    group_rows = []
    for name, cohort_map, n_windows in (
        ("R1-R5", primary, windows_primary),
        ("R0/R0_1 - QC only", preliminary, windows_preliminary),
    ):
        psd_map = {person: [record["psd"] for record in records] for person, records in cohort_map.items()}
        rows = aggregate_participant_spectra(frequency_grid, psd_map, name)
        for row in rows:
            row["n_windows"] = int(n_windows)
        group_rows.extend(rows)
    assert_public_fields_safe(PUBLIC_FIELDS)
    _write_csv(output_csv, group_rows)
    _svg_figure(figure_path, group_rows)
    receipt = {
        "analysis": "participant-balanced, gap-aware PSD visualization of local saved EMG channels",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "15 local participants only; no external dataset",
        "records_source": 140,
        "records_primary_used": used_primary,
        "records_preliminary_used": used_preliminary,
        "samples_verified": samples,
        "participants_verified": len(participants),
        "primary_participants": len(primary),
        "preliminary_participants": len(preliminary),
        "primary_windows": windows_primary,
        "preliminary_windows": windows_preliminary,
        "skipped_recordings_private": len(skipped_recordings),
        "segment_seconds": SEGMENT_SECONDS,
        "target_fs_grid_hz": GRID_FS_HZ,
        "frequency_step_hz": 1.0 / SEGMENT_SECONDS,
        "max_interval_spread_pct": MAX_INTERVAL_SPREAD_RATIO * 100,
        "gap_split_s": GAP_THRESHOLD_S,
        "window": "Hann, 8-second, nonoverlapping; windows never cross timestamp gaps",
        "aggregation": "median records within participant, then median and IQR across participants",
        "units": "reported ADC^2/Hz; not physically calibrated",
        "mode_or_clock": "unknown; timestamps estimate sampling; hardware clock unconfirmed",
        "fatigue_labels_created": False,
        "training_performed": False,
        "external_data_used": False,
        "notch_applied": False,
        "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        "diagnostic_utility_sha256": _sha256(Path(__file__).with_name("diagnostico_baja_frecuencia_emg.py")),
        "filter_engine_script_sha256": _sha256(Path(__file__).with_name("comparar_filtros_emg.py")),
        "test_script_sha256": _sha256(Path(__file__).with_name("tests") / "test_exportar_psd_emg_agregada.py"),
        "source_receipt_sha256": _sha256(prior_path),
        "private_array_sha256": input_hashes,
        "skipped_private_filenames": skipped_recordings,
        "aggregate_csv_sha256": _sha256(output_csv),
        "figure_sha256": _sha256(figure_path),
        "public_fields": list(PUBLIC_FIELDS),
    }
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    receipt = run_analysis(args.private_root, args.output_csv, args.figure, args.receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("private_array_sha256", "skipped_private_filenames")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
