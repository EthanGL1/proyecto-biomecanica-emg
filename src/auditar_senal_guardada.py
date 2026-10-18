"""Aggregate numeric QC of saved EMG-like channels without modifying source files.

The script deliberately does not filter, resample, infer sensor mode, or assign
fatigue labels. Spectral summaries are exploratory and use only contiguous
1024-sample blocks whose timestamp gaps pass the project's existing 1.5 ms
screen. Full per-record outputs contain pseudonymous IDs and must stay private.
"""

from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {"Relative_Time_s", "ADC_Value", "Voltage_mV"}
GAP_SCREEN_S = 0.0015
GAP_FLOAT_TOLERANCE_S = 1e-9
PSD_WINDOW_SAMPLES = 1024


def safe_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if resolved_root not in candidate.parents:
        raise ValueError("Manifest path escapes the pseudonymized signal directory")
    return candidate


def spectral_window(x: np.ndarray, t: np.ndarray) -> tuple[float, float, float, float]:
    dt = np.diff(t)
    fs = float(1.0 / np.median(dt))
    window = np.hanning(x.size)
    centered = x - np.mean(x)
    power = np.abs(np.fft.rfft(centered * window)) ** 2 / np.sum(window**2)
    freq = np.fft.rfftfreq(x.size, d=1.0 / fs)
    valid = (freq >= 20.0) & (freq <= min(450.0, fs / 2.0))
    total = float(np.sum(power[valid]))
    if total <= 0 or not np.any(valid):
        return fs, float("nan"), float("nan"), float("nan")

    peak = float(freq[np.flatnonzero(valid)[np.argmax(power[valid])]])

    def line_ratio(center_hz: float) -> float:
        band = np.abs(freq - center_hz) <= 1.5
        ratio = float(np.sum(power[band]) / total)
        return 10.0 * np.log10(max(ratio, np.finfo(float).tiny))

    return fs, peak, line_ratio(50.0), line_ratio(60.0)


def process_record(row: pd.Series, signals_root: Path) -> dict[str, object]:
    path = safe_path(signals_root, str(row["relative_path"]))
    frame = pd.read_csv(path, usecols=lambda name: name in EXPECTED_COLUMNS)
    if set(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected signal columns in pseudonymized input: {path.name}")

    t = frame["Relative_Time_s"].to_numpy(dtype=np.float64)
    adc = frame["ADC_Value"].to_numpy(dtype=np.float64)
    x = frame["Voltage_mV"].to_numpy(dtype=np.float64)
    if not (np.isfinite(t).all() and np.isfinite(adc).all() and np.isfinite(x).all()):
        raise ValueError(f"Non-finite values in pseudonymized input: {path.name}")
    if t.size < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f"Timestamps are not strictly increasing: {path.name}")

    dt = np.diff(t)
    med_dt = float(np.median(dt))
    gap_idx = np.flatnonzero(dt > GAP_SCREEN_S + GAP_FLOAT_TOLERANCE_S)
    boundaries = np.r_[0, gap_idx + 1, x.size]
    spectral = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        run_x, run_t = x[lo:hi], t[lo:hi]
        nwin = run_x.size // PSD_WINDOW_SAMPLES
        for w in range(nwin):
            a = w * PSD_WINDOW_SAMPLES
            b = a + PSD_WINDOW_SAMPLES
            spectral.append(spectral_window(run_x[a:b], run_t[a:b]))

    sp = np.asarray(spectral, dtype=np.float64) if spectral else np.empty((0, 4))
    formula = np.round(adc * 3300.0 / 4095.0, 2)
    return {
        "participant_id": row["participant_id"],
        "recording_id": row["recording_id"],
        "leg": row["leg"],
        "n_samples": int(x.size),
        "signal_mode": "unknown",
        "stored_voltage_unit": "mV label; calibration unverified",
        "mean_stored_mV": float(np.mean(x)),
        "sd_stored_mV": float(np.std(x, ddof=1)),
        "rms_stored_mV": float(np.sqrt(np.mean(x**2))),
        "median_stored_mV": float(np.median(x)),
        "p01_stored_mV": float(np.percentile(x, 1)),
        "p99_stored_mV": float(np.percentile(x, 99)),
        "minimum_stored_mV": float(np.min(x)),
        "maximum_stored_mV": float(np.max(x)),
        "fs_median_timestamp_hz": float(1.0 / med_dt),
        "dt_median_ms": float(med_dt * 1000),
        "dt_p01_ms": float(np.percentile(dt, 1) * 1000),
        "dt_p99_ms": float(np.percentile(dt, 99) * 1000),
        "gap_screen_ms": GAP_SCREEN_S * 1000,
        "gaps_over_screen": int(gap_idx.size),
        "max_gap_ms": float(np.max(dt) * 1000),
        "p99_abs_dt_deviation_pct": float(
            100 * np.percentile(np.abs(dt - med_dt), 99) / med_dt
        ),
        "adc_voltage_formula_max_abs_error_mV": float(np.max(np.abs(formula - x))),
        "spectrum_windows_1024": int(sp.shape[0]),
        "median_spectrum_peak_20_450_hz": float(np.nanmedian(sp[:, 1])) if sp.size else float("nan"),
        "median_50hz_band_ratio_db": float(np.nanmedian(sp[:, 2])) if sp.size else float("nan"),
        "median_60hz_band_ratio_db": float(np.nanmedian(sp[:, 3])) if sp.size else float("nan"),
        "notch_50_or_60_applied": False,
        "fatigue_label_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-datasets", required=True, type=Path)
    parser.add_argument("--public-summary", required=True, type=Path)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])

    manifest_path = args.private_datasets / "manifest_pseudonymized.csv"
    signals_root = args.private_datasets / "pseudonymized_csv"
    if not manifest_path.is_file() or not signals_root.is_dir():
        raise FileNotFoundError("Private manifest or pseudonymized signal directory is unavailable")
    manifest = pd.read_csv(manifest_path, dtype={"participant_id": str})
    required = {"participant_id", "recording_id", "leg", "relative_path"}
    if not required.issubset(manifest.columns):
        raise ValueError("Private manifest is missing required columns")
    records = pd.DataFrame(process_record(row, signals_root) for _, row in manifest.iterrows())
    private_out = args.private_datasets / "signal_qc_v1"
    private_out.mkdir(parents=True, exist_ok=True)
    private_csv = private_out / "qc_por_registro.csv"
    records.to_csv(private_csv, index=False, encoding="utf-8")

    public = records.assign(
        tipo_registro=np.where(records["recording_id"].isin(["R0", "R0_1"]), "R0_preliminar_o_anomalo", "R1_R5")
    )
    aggregate = (
        public.groupby(["tipo_registro", "leg"], dropna=False)
        .agg(
            registros=("recording_id", "size"),
            muestras=("n_samples", "sum"),
            mediana_fs_hz=("fs_median_timestamp_hz", "median"),
            mediana_rms_mV_reportados=("rms_stored_mV", "median"),
            mediana_p01_mV_reportados=("p01_stored_mV", "median"),
            mediana_p99_mV_reportados=("p99_stored_mV", "median"),
            huecos_mayores_1_5ms=("gaps_over_screen", "sum"),
            mediana_max_gap_ms=("max_gap_ms", "median"),
            median_peak_20_450_hz=("median_spectrum_peak_20_450_hz", "median"),
            median_50hz_band_ratio_db=("median_50hz_band_ratio_db", "median"),
            median_60hz_band_ratio_db=("median_60hz_band_ratio_db", "median"),
            ventanas_espectrales=("spectrum_windows_1024", "sum"),
        )
        .reset_index()
    )
    aggregate = aggregate.rename(
        columns={
            "median_peak_20_450_hz": "mediana_pico_20_450_hz",
            "median_50hz_band_ratio_db": "mediana_proporcion_banda_50hz_db",
            "median_60hz_band_ratio_db": "mediana_proporcion_banda_60hz_db",
        }
    )
    aggregate["nota_de_alcance"] = (
        "Modo RAW/RECT/ENV y calibracion desconocidos; espectro exploratorio en bloques contiguos; "
        "no se aplico filtro ni se selecciono un notch."
    )
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.public_summary, index=False, encoding="utf-8")

    receipt = {
        "recordings_read": int(len(records)),
        "samples_read": int(records["n_samples"].sum()),
        "spectral_windows_1024": int(records["spectrum_windows_1024"].sum()),
        "rows_with_unknown_signal_mode": int(records["signal_mode"].eq("unknown").sum()),
        "private_output": str(private_csv),
        "public_output": str(args.public_summary),
        "source_modified": False,
        "filter_applied": False,
        "fatigue_labels_created": False,
        "reason_filtering_deferred": "Acquisition mode and analog chain are unverified.",
    }
    (private_out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
