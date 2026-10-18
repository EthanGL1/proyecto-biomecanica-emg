"""Exploratory time-trajectory sensitivity for local participant EMG records.

The script reads only the project's private, pseudonymized filter outputs.
It describes how signal features change within a recording and across the
existing candidate filter branches. Time position is not treated as fatigue,
repetition count, active contraction time, or task failure.
"""

from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WINDOW_SAMPLES = 1024
GAP_SCREEN_S = 0.0015
N_TIME_BINS = 5
LOW_HZ = 20.0
HIGH_HZ = 450.0
BRANCHES = ("control", "hp20", "hp40", "bp20_400", "fir_bp20_400")
FEATURES = (
    "rms_centered",
    "median_frequency_hz",
    "mean_frequency_hz",
    "spectral_entropy_norm",
    "zero_crossings_per_s",
)
CHANGE_FIELDS = {
    "rms_centered": ("rms_change_pct", "% desde la primera a la última quinta parte"),
    "median_frequency_hz": ("median_frequency_change_hz", "Hz"),
    "mean_frequency_hz": ("mean_frequency_change_hz", "Hz"),
    "spectral_entropy_norm": ("spectral_entropy_change", "adimensional"),
    "zero_crossings_per_s": ("zero_crossings_change_per_s", "cruces/s"),
}


def iter_contiguous_windows(
    times_s: Iterable[float],
    signal: Iterable[float],
    valid_mask: Iterable[bool],
    window_samples: int = WINDOW_SAMPLES,
    gap_screen_s: float = GAP_SCREEN_S,
) -> list[dict[str, Any]]:
    """Return non-overlapping complete windows without bridging gaps or masks."""
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if times.ndim != 1 or values.ndim != 1 or mask.ndim != 1:
        raise ValueError("times, signal, and valid_mask must be one-dimensional")
    if not (times.size == values.size == mask.size):
        raise ValueError("times, signal, and valid_mask must have equal lengths")
    if window_samples < 2 or gap_screen_s <= 0:
        raise ValueError("window_samples and gap_screen_s must be positive")
    if times.size < window_samples or times[-1] <= times[0]:
        return []
    if not (np.isfinite(times).all() and np.isfinite(values).all()):
        raise ValueError("times and signal must contain only finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    full_duration = float(times[-1] - times[0])
    gap_idx = np.flatnonzero(np.diff(times) > gap_screen_s) + 1
    boundaries = np.r_[0, gap_idx, times.size]
    windows: list[dict[str, Any]] = []

    for run_start, run_end in zip(boundaries[:-1], boundaries[1:]):
        run_mask = mask[run_start:run_end]
        valid_idx = np.flatnonzero(run_mask)
        if not valid_idx.size:
            continue
        valid_breaks = np.flatnonzero(np.diff(valid_idx) > 1) + 1
        valid_runs = np.split(valid_idx, valid_breaks)
        for indices in valid_runs:
            lo = run_start + int(indices[0])
            hi = run_start + int(indices[-1]) + 1
            for start in range(lo, hi - window_samples + 1, window_samples):
                end = start + window_samples
                window_times = times[start:end]
                dt = np.diff(window_times)
                if dt.size == 0 or np.any(dt > gap_screen_s):
                    continue
                fs_hz = float(1.0 / np.median(dt))
                midpoint = float((window_times[0] + window_times[-1]) / 2.0)
                windows.append(
                    {
                        "time_s": midpoint,
                        "time_fraction": float((midpoint - times[0]) / full_duration),
                        "fs_hz": fs_hz,
                        "signal": values[start:end].copy(),
                    }
                )
    return windows


def spectral_features(
    signal: Iterable[float],
    fs_hz: float,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
) -> dict[str, float]:
    """Calculate descriptive amplitude/spectrum features on a demeaned window."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 8:
        raise ValueError("signal must be a one-dimensional window of at least 8 samples")
    if not np.isfinite(values).all() or not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("signal and sampling frequency must be finite; fs_hz must be positive")
    if low_hz < 0 or high_hz <= low_hz or high_hz >= fs_hz / 2:
        raise ValueError("the requested frequency band must lie below Nyquist")

    centered = values - np.mean(values)
    rms = float(np.sqrt(np.mean(centered**2)))
    window = np.hanning(values.size)
    power = np.abs(np.fft.rfft(centered * window)) ** 2 / np.sum(window**2)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / fs_hz)
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_power = power[band]
    band_frequencies = frequencies[band]
    total_power = float(np.sum(band_power))
    if not band_power.size or not np.isfinite(total_power) or total_power <= 0:
        raise ValueError("window has no positive spectral power in the analysis band")

    probabilities = band_power / total_power
    cumulative = np.cumsum(probabilities)
    median_idx = min(int(np.searchsorted(cumulative, 0.5, side="left")), band_frequencies.size - 1)
    nonzero_prob = probabilities[probabilities > 0]
    entropy_denominator = np.log(probabilities.size) if probabilities.size > 1 else 1.0
    entropy = float(-np.sum(nonzero_prob * np.log(nonzero_prob)) / entropy_denominator)
    zero_crossings = float(
        np.count_nonzero(np.signbit(centered[1:]) != np.signbit(centered[:-1]))
        / ((values.size - 1) / fs_hz)
    )
    return {
        "rms_centered": rms,
        "median_frequency_hz": float(band_frequencies[median_idx]),
        "mean_frequency_hz": float(np.sum(band_frequencies * probabilities)),
        "spectral_entropy_norm": entropy,
        "zero_crossings_per_s": zero_crossings,
    }


def bin_feature_medians(
    windows: Iterable[dict[str, Any]], n_bins: int = N_TIME_BINS
) -> list[dict[str, Any]]:
    """Summarize window features by equal-duration recording-time bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    rows = list(windows)
    feature_keys = sorted({key for row in rows for key in row if key != "time_fraction"})
    summaries: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        lower, upper = bin_index / n_bins, (bin_index + 1) / n_bins
        selected = [
            row
            for row in rows
            if lower <= float(row["time_fraction"]) < upper
            or (bin_index == n_bins - 1 and float(row["time_fraction"]) == 1.0)
        ]
        summary: dict[str, Any] = {"time_bin": bin_index + 1, "n_windows": len(selected)}
        for key in feature_keys:
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            summary[key] = float(np.median(values)) if values else None
        summaries.append(summary)
    return summaries


def endpoint_change(time_bins: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute first-to-last-bin changes without assigning biological labels."""
    if len(time_bins) < 2:
        raise ValueError("at least two time bins are required")
    first, last = time_bins[0], time_bins[-1]
    changes: dict[str, float | None] = {}
    for feature, (field, _) in CHANGE_FIELDS.items():
        start, end = first.get(feature), last.get(feature)
        if start is None or end is None or not np.isfinite([start, end]).all():
            changes[field] = None
        elif feature == "rms_centered":
            changes[field] = None if float(start) <= 0 else float(100.0 * (float(end) / float(start) - 1.0))
        else:
            changes[field] = float(end) - float(start)
    return changes


def participant_medians(
    rows: Iterable[dict[str, Any]], value_field: str
) -> dict[str, float]:
    """Collapse repeated files to one median value per participant."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        participant = str(row["participant_id"])
        value = row.get(value_field)
        if value is not None and np.isfinite(float(value)):
            grouped.setdefault(participant, []).append(float(value))
    return {participant: float(np.median(values)) for participant, values in grouped.items()}


def bootstrap_median_ci(
    values: Iterable[float], seed: int = 20261014, n_boot: int = 10000
) -> tuple[float, float]:
    """Participant-level percentile interval for the median; deterministic by seed."""
    sample = np.asarray(list(values), dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if sample.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(n_boot, sample.size))
    medians = np.median(sample[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def stratified_summary(record_rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Summarize set- and side-specific changes after participant aggregation."""
    records = pd.DataFrame(record_rows)
    if records.empty:
        return pd.DataFrame(
            columns=["estrato", "nivel", "rama", "rasgo", "cambio_mediano_por_participante", "n_participantes"]
        )
    records = records[
        records["recording_id"].isin([f"R{i}" for i in range(1, 6)])
        & records["analysis_status"].eq("complete")
    ]
    results: list[dict[str, Any]] = []
    strata = (("serie", "recording_id"), ("lado", "leg"))
    for stratum_name, stratum_column in strata:
        for level, stratum_group in records.groupby(stratum_column, sort=True):
            for branch, branch_group in stratum_group.groupby("branch", sort=True):
                rows = branch_group.to_dict(orient="records")
                for feature, (field, unit) in CHANGE_FIELDS.items():
                    per_subject = participant_medians(rows, field)
                    values = np.asarray(list(per_subject.values()), dtype=np.float64)
                    ci = bootstrap_median_ci(values)
                    results.append(
                        {
                            "estrato": stratum_name,
                            "nivel": str(level),
                            "rama": branch,
                            "rasgo": feature,
                            "cambio_mediano_por_participante": float(np.median(values)) if values.size else np.nan,
                            "ic95_bootstrap_mediana_inf": ci[0],
                            "ic95_bootstrap_mediana_sup": ci[1],
                            "n_participantes": int(values.size),
                            "participantes_cambio_positivo": int(np.sum(values > 0)),
                            "participantes_cambio_negativo": int(np.sum(values < 0)),
                            "unidad": unit,
                            "nota": "Estratificado por serie/lado; un valor por participante; descriptivo y no diagnóstico de fatiga.",
                        }
                    )
    return pd.DataFrame(results)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id_stem(row: pd.Series) -> str:
    return f"{row['participant_id']}_{row['recording_id']}_{row['leg']}"


def _summarize_public(
    record_rows: list[dict[str, Any]],
    private_subject_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = pd.DataFrame(record_rows)
    r15 = records[records["recording_id"].isin([f"R{i}" for i in range(1, 6)])].copy()
    r15 = r15[r15["analysis_status"].eq("complete")]
    subject_rows: list[dict[str, Any]] = []
    for (participant, branch), group in r15.groupby(["participant_id", "branch"], sort=True):
        row: dict[str, Any] = {"participant_id": participant, "branch": branch}
        row["n_records"] = int(group["recording_id"].nunique())
        for _, (field, _) in CHANGE_FIELDS.items():
            vals = group[field].dropna().to_numpy(dtype=float)
            row[field] = float(np.median(vals)) if vals.size else np.nan
        subject_rows.append(row)
    subjects = pd.DataFrame(subject_rows)
    private_subject_path.parent.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(private_subject_path, index=False, encoding="utf-8")

    group_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        branch_rows = subjects[subjects["branch"].eq(branch)]
        for feature, (field, unit) in CHANGE_FIELDS.items():
            vals = branch_rows[field].dropna().to_numpy(dtype=float)
            ci = bootstrap_median_ci(vals)
            group_rows.append(
                {
                    "grupo": "R1-R5",
                    "rama": branch,
                    "rasgo": feature,
                    "cambio_mediano_por_participante": float(np.median(vals)) if vals.size else np.nan,
                    "ic95_bootstrap_mediana_inf": ci[0],
                    "ic95_bootstrap_mediana_sup": ci[1],
                    "n_participantes": int(vals.size),
                    "participantes_cambio_positivo": int(np.sum(vals > 0)),
                    "participantes_cambio_negativo": int(np.sum(vals < 0)),
                    "unidad": unit,
                    "nota": "Trayectoria temporal descriptiva; no es etiqueta de fatiga ni prueba de denoising.",
                }
            )
            if branch == "control":
                continue
            wide = subjects.pivot(index="participant_id", columns="branch", values=field)
            if branch not in wide or "control" not in wide:
                paired = np.asarray([], dtype=float)
                same_sign = 0
            else:
                valid = wide[[branch, "control"]].dropna()
                paired = (valid[branch] - valid["control"]).to_numpy(dtype=float)
                same_sign = int(np.sum(np.sign(valid[branch]) == np.sign(valid["control"])))
            diff_ci = bootstrap_median_ci(paired)
            paired_rows.append(
                {
                    "rama_filtrada": branch,
                    "rasgo": feature,
                    "diferencia_mediana_rama_menos_control": float(np.median(paired)) if paired.size else np.nan,
                    "ic95_bootstrap_diferencia_inf": diff_ci[0],
                    "ic95_bootstrap_diferencia_sup": diff_ci[1],
                    "acuerdo_signo_con_control_n": same_sign,
                    "n_participantes_apareados": int(paired.size),
                    "unidad": unit,
                    "nota": "Compara robustez de trayectorias; no identifica cuál rama se acerca a una señal fisiológica verdadera.",
                }
            )

    r0 = records[records["recording_id"].isin(["R0", "R0_1"])].copy()
    r0_rows: list[dict[str, Any]] = []
    if not r0.empty:
        for branch, group in r0.groupby("branch", sort=True):
            for feature, (field, unit) in CHANGE_FIELDS.items():
                vals = group[field].dropna().to_numpy(dtype=float)
                r0_rows.append(
                    {
                        "grupo": "R0/R0_1_QC_solo",
                        "rama": branch,
                        "rasgo": feature,
                        "cambio_mediano_por_participante": float(np.median(vals)) if vals.size else np.nan,
                        "ic95_bootstrap_mediana_inf": np.nan,
                        "ic95_bootstrap_mediana_sup": np.nan,
                        "n_participantes": 0,
                        "n_registros_qc": int(vals.size),
                        "participantes_cambio_positivo": int(np.sum(vals > 0)),
                        "participantes_cambio_negativo": int(np.sum(vals < 0)),
                        "unidad": unit,
                        "nota": "Dos archivos preliminares/anómalos según declaración; se muestran aparte y no se usan para inferencia por sujeto.",
                    }
                )
    return pd.DataFrame(group_rows + r0_rows), pd.DataFrame(paired_rows), records


def analyze_dataset(private_datasets: Path, public_out: Path) -> dict[str, Any]:
    manifest_path = private_datasets / "manifest_pseudonymized.csv"
    filtered_root = private_datasets / "filter_sensitivity_v1"
    private_out = private_datasets / "temporal_feature_sensitivity_v1"
    if not manifest_path.is_file() or not filtered_root.is_dir():
        raise FileNotFoundError("Missing private manifest or existing filter-sensitivity outputs")
    manifest = pd.read_csv(manifest_path, dtype={"participant_id": str})
    required = {"participant_id", "recording_id", "leg"}
    if not required.issubset(manifest.columns):
        raise ValueError("Private manifest lacks participant, recording, or leg fields")
    if len(manifest) != 140 or manifest["participant_id"].nunique() != 15:
        raise ValueError("Unexpected local study inventory; expected 140 records and 15 participants")

    record_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    total_samples = 0
    total_windows = 0
    missing_inputs = 0
    for _, meta in manifest.iterrows():
        npz_path = filtered_root / f"{_record_id_stem(meta)}.npz"
        if not npz_path.is_file():
            missing_inputs += 1
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            times = data["times_s"].astype(np.float64, copy=False)
            common_mask = data["common_comparison_mask"].astype(bool, copy=False)
            total_samples += int(times.size)
            for branch in BRANCHES:
                values = data[branch]
                raw_windows = iter_contiguous_windows(times, values, common_mask)
                feature_windows: list[dict[str, Any]] = []
                for index, window in enumerate(raw_windows):
                    features = spectral_features(window["signal"], window["fs_hz"])
                    enriched = {
                        "window_index": index,
                        "time_s": window["time_s"],
                        "time_fraction": window["time_fraction"],
                        **features,
                    }
                    feature_windows.append(enriched)
                    window_rows.append(
                        {
                            "participant_id": meta["participant_id"],
                            "recording_id": meta["recording_id"],
                            "leg": meta["leg"],
                            "branch": branch,
                            **enriched,
                        }
                    )
                total_windows += len(feature_windows)
                bins = bin_feature_medians(feature_windows)
                changes = endpoint_change(bins)
                complete_bins = sum(int(row["n_windows"] > 0) for row in bins)
                record_rows.append(
                    {
                        "participant_id": meta["participant_id"],
                        "recording_id": meta["recording_id"],
                        "leg": meta["leg"],
                        "branch": branch,
                        "n_windows": len(feature_windows),
                        "complete_time_bins": complete_bins,
                        "min_windows_per_time_bin": min(row["n_windows"] for row in bins),
                        "analysis_status": "complete" if complete_bins == N_TIME_BINS else "insufficient_time_coverage",
                        **changes,
                    }
                )
    if missing_inputs:
        raise FileNotFoundError(f"Missing {missing_inputs} of 140 private filter-sensitivity records")
    if len(record_rows) != len(manifest) * len(BRANCHES):
        raise RuntimeError("Record-branch coverage check failed")

    private_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(window_rows).to_csv(private_out / "rasgos_por_ventana.csv", index=False, encoding="utf-8")
    group_summary, paired_summary, records = _summarize_public(
        record_rows, private_out / "cambios_por_participante_privado.csv"
    )
    stratified = stratified_summary(record_rows)
    public_out.mkdir(parents=True, exist_ok=True)
    group_path = public_out / "SENSIBILIDAD_TRAYECTORIAS_FILTRADO_AGREGADA.csv"
    paired_path = public_out / "DIFERENCIAS_TRAYECTORIAS_FILTRO_CONTROL_AGREGADAS.csv"
    stratified_path = public_out / "SENSIBILIDAD_TRAYECTORIAS_POR_SERIE_LADO.csv"
    group_summary.to_csv(group_path, index=False, encoding="utf-8")
    paired_summary.to_csv(paired_path, index=False, encoding="utf-8")
    stratified.to_csv(stratified_path, index=False, encoding="utf-8")
    records.to_csv(private_out / "cambios_por_registro_privado.csv", index=False, encoding="utf-8")

    counts = manifest["recording_id"].value_counts()
    main_records = int(manifest["recording_id"].isin([f"R{i}" for i in range(1, 6)]).sum())
    r0_records = int(manifest["recording_id"].isin(["R0", "R0_1"]).sum())
    receipt = {
        "analysis_id": "EMG-TREND-SENS-V1",
        "date": "2026-10-14",
        "input_manifest_sha256": _sha256(manifest_path),
        "input_records": int(len(manifest)),
        "participants": int(manifest["participant_id"].nunique()),
        "R1_R5_records": main_records,
        "R0_R0_1_QC_only_records": r0_records,
        "recording_id_counts": {str(k): int(v) for k, v in counts.sort_index().items()},
        "filter_branches": list(BRANCHES),
        "samples_in_filter_outputs": total_samples,
        "window_samples": WINDOW_SAMPLES,
        "window_duration_median_s_approx": round(WINDOW_SAMPLES / 1000.0, 3),
        "sample_rate_source": "median per-window timestamp differences; device clock unverified",
        "gap_screen_s": GAP_SCREEN_S,
        "time_bins_per_record": N_TIME_BINS,
        "frequency_band_hz": [LOW_HZ, HIGH_HZ],
        "feature_windows_calculated": total_windows,
        "complete_records_per_branch": int(records[records["analysis_status"].eq("complete")].groupby("branch").size().min()),
        "labels_used": False,
        "fatigue_labels_created": False,
        "models_trained": False,
        "external_datasets_used": False,
        "raw_originals_modified": False,
        "interpretation": "Within-record feature trajectory and filter sensitivity only; not fatigue ground truth or evidence that a filter removes noise.",
        "public_outputs": [str(group_path), str(paired_path), str(stratified_path)],
        "private_outputs": [
            str(private_out / "rasgos_por_ventana.csv"),
            str(private_out / "cambios_por_registro_privado.csv"),
            str(private_out / "cambios_por_participante_privado.csv"),
        ],
    }
    if records["participant_id"].nunique() != 15:
        raise RuntimeError("The trajectory outputs do not cover all 15 pseudonymized participants")
    if "participant_id" in group_summary.columns or "participant_id" in paired_summary.columns:
        raise RuntimeError("Public aggregate output contains a participant identifier")
    (private_out / "recibo.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-datasets", required=True, type=Path)
    parser.add_argument("--public-out", required=True, type=Path)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    receipt = analyze_dataset(args.private_datasets, args.public_out)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
