"""Analyze raw EMG CSVs from a ZIP with the repository's descriptive pipeline.

This launcher does not load an ML model, count verified mechanical repetitions,
or predict physiological fatigue. It exports detected activity candidates only.
Input and output data must remain outside the public source checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import webbrowser
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

from analizar_umbral_senal_emg import robust_lower_threshold, sustained_crossings
from release_paths import ReleasePathError, validate_data_paths


MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_CSV_BYTES = 2 * 1024 * 1024 * 1024
RAW_HEADERS = (
    ("Relative_Time_s", "ADC_Value"),
    ("Relative_Time_s", "ADC_Value", "Voltage_mV"),
)
SERIES_RE = re.compile(r"(?<![A-Z0-9])R(?P<number>[0-5])(?P<preliminary>_1)?(?![A-Z0-9])", re.I)
LEFT_RE = re.compile(r"(?<![A-Z0-9])(?:IZQUIERDA|IZQ|IZ|LEFT)(?![A-Z0-9])", re.I)
RIGHT_RE = re.compile(r"(?<![A-Z0-9])(?:DERECHA|DR|RIGHT)(?![A-Z0-9])", re.I)

MANIFEST_FIELDS = (
    "recording_id", "participant_pseudonym", "side", "series", "status",
    "reason", "candidate_event_count", "output_folder",
)
EVENT_FIELDS = (
    "recording_id", "participant_pseudonym", "side", "series", "event_candidate_index",
    "start_time_s", "end_time_s", "duration_s", "threshold_rms_adc",
    "mean_rms_adc", "peak_rms_adc", "quality", "interpretation",
)


class ArchiveSafetyError(ValueError):
    """Raised when a selected ZIP is malformed or unsafe to process."""


@dataclass(frozen=True)
class ZipRecording:
    recording_id: str
    participant_id: str | None
    side: str
    series: str | None
    status: str
    reason: str
    member_name: str = field(repr=False)


def _validate_member_name(info: zipfile.ZipInfo) -> str:
    raw_name = info.filename
    normalized = raw_name.replace("\\", "/")
    trimmed = normalized[:-1] if info.is_dir() and normalized.endswith("/") else normalized
    windows = PureWindowsPath(raw_name)
    if (
        not trimmed
        or normalized.startswith("/")
        or PurePosixPath(normalized).is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in trimmed.split("/"))
    ):
        raise ArchiveSafetyError("ZIP contiene una ruta absoluta o insegura.")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ArchiveSafetyError("ZIP contiene un enlace simbólico; se rechazó por seguridad.")
    if info.flag_bits & 0x1:
        raise ArchiveSafetyError("ZIP contiene archivos cifrados que no pueden verificarse.")
    return normalized


def _read_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, ...] | None:
    try:
        with archive.open(info, "r") as binary:
            first_line = binary.readline(64 * 1024 + 1)
        if len(first_line) > 64 * 1024:
            return None
        text = first_line.decode("utf-8-sig")
        return tuple(cell.strip() for cell in next(csv.reader(io.StringIO(text)), []))
    except (UnicodeDecodeError, csv.Error, OSError, zipfile.BadZipFile, StopIteration):
        return None


def _series_from_stem(stem: str) -> tuple[str | None, str]:
    matches = list(SERIES_RE.finditer(stem))
    if not matches:
        return None, "series_missing"
    if len(matches) != 1:
        return None, "ambiguous_series"
    match = matches[0]
    series = f"R{match.group('number')}"
    if match.group("preliminary"):
        series += "_1"
    return series, "ready"


def _side_from_path(path: str) -> str:
    left = bool(LEFT_RE.search(path))
    right = bool(RIGHT_RE.search(path))
    if left == right:
        return "unknown"
    return "left" if left else "right"


def discover_recordings(zip_path: Path | str) -> list[ZipRecording]:
    """Inspect a ZIP, pseudonymize participant folders, and classify its CSVs.

    Files are not extracted by their archive names. A validated CSV schema and
    an unambiguous R1-R5 label are required for processing. R0/R0_1 are retained
    in the manifest as preliminary QC records and are not batch-processed.
    """
    path = Path(zip_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise ArchiveSafetyError("Seleccione un archivo ZIP válido.")
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ArchiveSafetyError("No se pudo abrir el ZIP seleccionado.") from error

    with archive:
        seen: set[str] = set()
        csv_infos: list[tuple[zipfile.ZipInfo, str, tuple[str, ...] | None]] = []
        total_size = 0
        for info in archive.infolist():
            normalized = _validate_member_name(info)
            key = normalized.casefold()
            if key in seen:
                raise ArchiveSafetyError("ZIP contiene nombres duplicados o ambiguos.")
            seen.add(key)
            if info.is_dir() or not normalized.lower().endswith(".csv"):
                continue
            if info.file_size > MAX_CSV_BYTES:
                raise ArchiveSafetyError("Un CSV del ZIP excede el límite de 512 MiB.")
            total_size += info.file_size
            if total_size > MAX_ARCHIVE_CSV_BYTES:
                raise ArchiveSafetyError("Los CSV del ZIP exceden el límite de 2 GiB.")
            csv_infos.append((info, normalized, _read_header(archive, info)))

        if not csv_infos:
            raise ArchiveSafetyError("El ZIP no contiene archivos CSV para analizar.")

    csv_infos.sort(key=lambda item: item[1].casefold())
    parents: set[str] = set()
    for _, member, _ in csv_infos:
        parent = PurePosixPath(member).parent.as_posix()
        if parent not in {"", "."}:
            parents.add(parent)
    # Only claim participant grouping when the archive structure distinguishes
    # more than one recording directory. A flat ZIP cannot establish subjects.
    pseudonyms = {key: f"P{index:02d}" for index, key in enumerate(sorted(parents), start=1)} if len(parents) > 1 else {}

    records: list[ZipRecording] = []
    for index, (info, member, header) in enumerate(csv_infos, start=1):
        pure = PurePosixPath(member)
        series, state = _series_from_stem(pure.stem)
        if state != "ready":
            reason = "No se procesó: el nombre no identifica una única serie R1–R5."
        elif series in {"R0", "R0_1"}:
            state = "preliminary_qc_only"
            reason = "Registro preliminar; se conserva separado del procesamiento formal R1–R5."
        elif header not in RAW_HEADERS:
            state = "unsupported_schema"
            reason = "CSV no coincide con Relative_Time_s,ADC_Value[,Voltage_mV]."
        else:
            reason = "Se procesará con el pipeline descriptivo F56; sin modelo ni etiquetas de repetición/fatiga."
        parent = pure.parent.as_posix()
        participant_id = pseudonyms.get(parent)
        records.append(
            ZipRecording(
                recording_id=f"REC{index:04d}",
                participant_id=participant_id,
                side=_side_from_path(member),
                series=series,
                status=state,
                reason=reason,
                member_name=info.filename,
            )
        )
    return records


def discover_single_csv(csv_path: Path | str) -> ZipRecording:
    """Validate one caller-supplied EMG CSV without assigning a participant label."""
    path = Path(csv_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ArchiveSafetyError("Seleccione un archivo CSV válido.")
    if zipfile.is_zipfile(path):
        raise ArchiveSafetyError("Seleccione un CSV sin comprimir; no renombre un ZIP como .csv.")
    if zipfile.is_zipfile(path):
        raise ArchiveSafetyError("Seleccione un CSV sin comprimir; no renombre un ZIP como .csv.")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ArchiveSafetyError("El CSV excede el límite de 512 MiB.")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = tuple(cell.strip() for cell in next(csv.reader(handle), []))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ArchiveSafetyError("No se pudo leer el encabezado del CSV.") from error
    series, state = _series_from_stem(path.stem)
    if state != "ready":
        raise ArchiveSafetyError("El nombre del CSV debe identificar una única serie R1–R5.")
    if series in {"R0", "R0_1"}:
        raise ArchiveSafetyError("R0/R0_1 es preliminar; no se procesa como serie formal.")
    if header not in RAW_HEADERS:
        raise ArchiveSafetyError("CSV no coincide con Relative_Time_s,ADC_Value[,Voltage_mV].")
    return ZipRecording(
        recording_id="REC0001",
        participant_id=None,
        side=_side_from_path(path.name),
        series=series,
        status="ready",
        reason="CSV individual; no se infiere identidad ni grupo de participante.",
        member_name=path.name,
    )


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _derive_spectral_threshold_screen(
    run_dir: Path,
    record_rows: list[dict[str, object]],
    events: list[dict[str, str]],
    phases: tuple[tuple[str, str, float, float], ...],
    processed_input_sha256: object = None,
) -> list[dict[str, object]]:
    """Write an explicitly exploratory robust-threshold screen; never a fatigue diagnosis."""
    min_activity_occupancy = 0.50
    phase_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    by_record: dict[str, list[dict[str, str]]] = {}
    for event in events:
        by_record.setdefault(event.get("recording_id", ""), []).append(event)

    def finite_float(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def display_status(status: str) -> str:
        return {
            "fatigue_compatible_change_exploratory": "Fatigue-compatible spectral change (exploratory; not confirmed)",
            "no_sustained_threshold_crossing": "No sustained spectral threshold crossing (not proof of no fatigue)",
            "not_evaluable": "Not evaluable",
        }.get(status, status.replace("_", " "))

    for record in record_rows:
        recording_id = str(record.get("recording_id", ""))
        recording_dir = run_dir / "recordings" / recording_id
        features_path = recording_dir / "features.csv"
        if not features_path.is_file():
            continue

        fs_by_run: dict[str, float] = {}
        segment_path = recording_dir / "segments.csv"
        if segment_path.is_file():
            with segment_path.open("r", encoding="utf-8", newline="") as handle:
                for segment in csv.DictReader(handle):
                    fs = finite_float(segment.get("fs_hz"))
                    if segment.get("status") == "processed" and fs and fs > 0:
                        fs_by_run[str(segment.get("run_index", ""))] = fs

        # Merge candidate intervals to avoid counting overlapping Otsu events twice.
        intervals = sorted(
            (start, end)
            for event in by_record.get(recording_id, [])
            if (start := finite_float(event.get("start_time_s"))) is not None
            and (end := finite_float(event.get("end_time_s"))) is not None
        )
        merged_intervals: list[list[float]] = []
        for start, end in intervals:
            if start is None or end is None or end <= start:
                continue
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append([start, end])
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

        def occupancy_fraction(start: float, end: float) -> float:
            duration = end - start
            if duration <= 0:
                return 0.0
            overlap = sum(max(0.0, min(end, right) - max(start, left)) for left, right in merged_intervals)
            return min(1.0, overlap / duration)

        feature_rows: list[dict[str, object]] = []
        with features_path.open("r", encoding="utf-8", newline="") as handle:
            for feature in csv.DictReader(handle):
                center = finite_float(feature.get("time_s"))
                sample_count = finite_float(feature.get("sample_count"))
                fs = fs_by_run.get(str(feature.get("run_index", "")))
                duration = sample_count / fs if sample_count and fs else None
                fraction = finite_float(feature.get("time_fraction"))
                mdf = finite_float(feature.get("mdf_hz"))
                mnf = finite_float(feature.get("mnf_hz"))
                if center is None or duration is None or duration <= 0 or fraction is None:
                    continue
                window_start, window_end = center - duration / 2.0, center + duration / 2.0
                proxy = str(feature.get("time_position_proxy", "")).strip().lower()
                if proxy not in {phase[1] for phase in phases}:
                    proxy = next((p[1] for p in phases if fraction < p[3]), phases[-1][1])
                phase_label = next(p[0] for p in phases if p[1] == proxy)
                feature_rows.append({
                    **feature,
                    "recording_id": recording_id,
                    "participant_pseudonym": record.get("participant_pseudonym", ""),
                    "side": record.get("side", ""),
                    "series": record.get("series", ""),
                    "time_s_value": center,
                    "time_fraction_value": fraction,
                    "window_start_s_value": window_start,
                    "window_end_s_value": window_end,
                    "mdf_value": mdf,
                    "mnf_value": mnf,
                    "proxy": proxy,
                    "phase_label": phase_label,
                    "activity_overlap_fraction": occupancy_fraction(window_start, window_end),
                    "nonoverlap_window_used": False,
                    "screen_segment_id": "",
                    "screen_status": "overlapping_window_not_used_for_threshold_screen",
                    "screen_reason": "overlapping_window_not_used_for_persistence_rule",
                    "below_mdf_threshold": False,
                    "below_mnf_threshold": False,
                    "sustained_crossing": False,
                })

        # Keep every feature row, but choose non-overlapping windows for baseline and persistence.
        branch_run_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        for feature in feature_rows:
            branch_run_groups.setdefault((str(feature.get("filter_branch", "")), str(feature.get("run_index", ""))), []).append(feature)
        for group in branch_run_groups.values():
            previous_end: float | None = None
            screen_segment_id = 0
            for feature in sorted(group, key=lambda item: (item["time_s_value"], int(str(item.get("window_index", 0))))):
                start = float(feature["window_start_s_value"])
                if previous_end is None or start >= previous_end - 1e-9:
                    if previous_end is not None and start > previous_end + 0.0015:
                        screen_segment_id += 1
                    feature["nonoverlap_window_used"] = True
                    feature["screen_segment_id"] = screen_segment_id
                    previous_end = float(feature["window_end_s_value"])

        by_branch: dict[str, list[dict[str, object]]] = {}
        for feature in feature_rows:
            by_branch.setdefault(str(feature.get("filter_branch", "unknown")), []).append(feature)

        for branch, branch_features in by_branch.items():
            independent = [feature for feature in branch_features if feature["nonoverlap_window_used"]]
            baseline = [
                feature for feature in independent
                if feature["time_fraction_value"] < 1.0 / 3.0
                and feature["activity_overlap_fraction"] >= min_activity_occupancy
                and feature["mdf_value"] is not None and feature["mnf_value"] is not None
            ]
            mdf_threshold = robust_lower_threshold(
                [feature["mdf_value"] for feature in baseline], baseline_fraction=1.0,
                multiplier=1.96, minimum_windows=5,
            )
            mnf_threshold = robust_lower_threshold(
                [feature["mnf_value"] for feature in baseline], baseline_fraction=1.0,
                multiplier=1.96, minimum_windows=5,
            )
            thresholds_ready = mdf_threshold["status"] == "estimated" and mnf_threshold["status"] == "estimated"
            mdf_lower = mdf_threshold["lower_threshold"] if thresholds_ready else None
            mnf_lower = mnf_threshold["lower_threshold"] if thresholds_ready else None

            grouped_by_run: dict[str, list[dict[str, object]]] = {}
            for feature in independent:
                grouped_by_run.setdefault(str(feature.get("run_index", "")), []).append(feature)
            for run_features in grouped_by_run.values():
                run_features.sort(key=lambda item: item["time_s_value"])
                below_both: list[bool] = []
                for feature in run_features:
                    active = feature["activity_overlap_fraction"] >= min_activity_occupancy
                    spectral_finite = feature["mdf_value"] is not None and feature["mnf_value"] is not None
                    if not thresholds_ready:
                        feature["screen_status"] = "not_evaluable"
                        feature["screen_reason"] = f"baseline_not_estimable:mdf_{mdf_threshold['status']};mnf_{mnf_threshold['status']}"
                    elif not spectral_finite:
                        feature["screen_status"] = "not_evaluable"
                        feature["screen_reason"] = "nonfinite_or_missing_spectral_metric"
                    elif not active:
                        feature["screen_status"] = "not_evaluable"
                        feature["screen_reason"] = "candidate_activity_occupancy_below_50_percent"
                    else:
                        mdf_below = float(feature["mdf_value"]) < float(mdf_lower)
                        mnf_below = float(feature["mnf_value"]) < float(mnf_lower)
                        feature["below_mdf_threshold"] = mdf_below
                        feature["below_mnf_threshold"] = mnf_below
                        feature["screen_status"] = "isolated_threshold_crossing" if mdf_below and mnf_below else "no_sustained_threshold_crossing"
                        feature["screen_reason"] = "joint_frequency_crossing_not_persistent" if mdf_below and mnf_below else "both_frequencies_not_below_their_thresholds"
                    below_both.append(bool(feature["below_mdf_threshold"] and feature["below_mnf_threshold"]))
                if thresholds_ready:
                    for crossing in sustained_crossings(
                        below_both,
                        [int(feature["screen_segment_id"]) for feature in run_features],
                        min_consecutive=3,
                    ):
                        start = crossing["start_window_index"]
                        stop = start + crossing["duration_windows"]
                        for feature in run_features[start:stop]:
                            feature["sustained_crossing"] = True
                            feature["screen_status"] = "fatigue_compatible_change_exploratory"
                            feature["screen_reason"] = "both_mdf_and_mnf_below_threshold_for_three_consecutive_nonoverlapping_windows"

            for feature in branch_features:
                if not feature["nonoverlap_window_used"]:
                    continue
                feature.update({
                    "mdf_baseline_median_hz": mdf_threshold["baseline_median"],
                    "mdf_robust_scale_hz": mdf_threshold["robust_scale"],
                    "mdf_lower_threshold_hz": mdf_lower,
                    "mnf_baseline_median_hz": mnf_threshold["baseline_median"],
                    "mnf_robust_scale_hz": mnf_threshold["robust_scale"],
                    "mnf_lower_threshold_hz": mnf_lower,
                    "baseline_window_count": len(baseline),
                    "baseline_status": "estimated" if thresholds_ready else f"mdf_{mdf_threshold['status']};mnf_{mnf_threshold['status']}",
                })
            for feature in branch_features:
                if feature["nonoverlap_window_used"]:
                    continue
                feature.update({
                    "mdf_baseline_median_hz": mdf_threshold["baseline_median"],
                    "mdf_robust_scale_hz": mdf_threshold["robust_scale"],
                    "mdf_lower_threshold_hz": mdf_lower,
                    "mnf_baseline_median_hz": mnf_threshold["baseline_median"],
                    "mnf_robust_scale_hz": mnf_threshold["robust_scale"],
                    "mnf_lower_threshold_hz": mnf_lower,
                    "baseline_window_count": len(baseline),
                    "baseline_status": "estimated" if thresholds_ready else f"mdf_{mdf_threshold['status']};mnf_{mnf_threshold['status']}",
                })

            for label, proxy, _, _ in phases:
                phase_features = [feature for feature in branch_features if feature["proxy"] == proxy]
                phase_independent = [feature for feature in phase_features if feature["nonoverlap_window_used"]]
                active_phase = [feature for feature in phase_independent if feature["activity_overlap_fraction"] >= min_activity_occupancy and feature["mdf_value"] is not None and feature["mnf_value"] is not None]
                sustained_phase = [feature for feature in active_phase if feature["sustained_crossing"]]
                if not thresholds_ready or not active_phase:
                    status = "not_evaluable"
                    reason = "baseline_threshold_not_estimable" if not thresholds_ready else "no_candidate_active_spectral_windows"
                elif sustained_phase:
                    status = "fatigue_compatible_change_exploratory"
                    reason = "persistent_joint_mdf_mnf_threshold_crossing"
                else:
                    status = "no_sustained_threshold_crossing"
                    reason = "no_persistent_joint_mdf_mnf_threshold_crossing"
                phase_rows.append({
                    "recording_id": recording_id,
                    "participant_pseudonym": record.get("participant_pseudonym", ""),
                    "side": record.get("side", ""),
                    "series": record.get("series", ""),
                    "temporal_position": label,
                    "time_position_proxy": proxy,
                    "filter_branch": branch,
                    "n_feature_windows": len(phase_features),
                    "n_nonoverlap_windows": len(phase_independent),
                    "n_activity_candidate_windows": len(active_phase),
                    "n_baseline_windows": len(baseline),
                    "baseline_status": "estimated" if thresholds_ready else f"mdf_{mdf_threshold['status']};mnf_{mnf_threshold['status']}",
                    "mdf_baseline_median_hz": mdf_threshold["baseline_median"],
                    "mdf_robust_scale_hz": mdf_threshold["robust_scale"],
                    "mdf_lower_threshold_hz": mdf_lower,
                    "mnf_baseline_median_hz": mnf_threshold["baseline_median"],
                    "mnf_robust_scale_hz": mnf_threshold["robust_scale"],
                    "mnf_lower_threshold_hz": mnf_lower,
                    "n_joint_below_threshold_windows": sum(bool(feature["below_mdf_threshold"] and feature["below_mnf_threshold"]) for feature in phase_independent),
                    "n_sustained_crossing_windows": len(sustained_phase),
                    "screen_status": status,
                    "screen_label": display_status(status),
                    "screen_reason": reason,
                    "fatigue_ground_truth_available": "false",
                    "interpretation": "exploratory signal-change screen; not a validated fatigue or no-fatigue label",
                })

            for feature in branch_features:
                window_rows.append({
                    "recording_id": recording_id,
                    "participant_pseudonym": record.get("participant_pseudonym", ""),
                    "side": record.get("side", ""),
                    "series": record.get("series", ""),
                    "filter_branch": branch,
                    "run_index": feature.get("run_index", ""),
                    "window_index": feature.get("window_index", ""),
                    "time_s": feature["time_s_value"],
                    "time_fraction": feature["time_fraction_value"],
                    "temporal_position": feature["phase_label"],
                    "time_position_proxy": feature["proxy"],
                    "window_start_s": feature["window_start_s_value"],
                    "window_end_s": feature["window_end_s_value"],
                    "activity_candidate_occupancy_fraction": feature["activity_overlap_fraction"],
                    "nonoverlap_window_used": str(feature["nonoverlap_window_used"]).lower(),
                    "contiguous_window_sequence": feature["screen_segment_id"],
                    "baseline_window_count": feature.get("baseline_window_count", len(baseline)),
                    "baseline_status": feature.get("baseline_status", "not_estimated"),
                    "mdf_hz": feature.get("mdf_hz", ""),
                    "mnf_hz": feature.get("mnf_hz", ""),
                    "mdf_baseline_median_hz": feature.get("mdf_baseline_median_hz"),
                    "mdf_robust_scale_hz": feature.get("mdf_robust_scale_hz"),
                    "mdf_lower_threshold_hz": feature.get("mdf_lower_threshold_hz"),
                    "mnf_baseline_median_hz": feature.get("mnf_baseline_median_hz"),
                    "mnf_robust_scale_hz": feature.get("mnf_robust_scale_hz"),
                    "mnf_lower_threshold_hz": feature.get("mnf_lower_threshold_hz"),
                    "below_mdf_threshold": str(feature["below_mdf_threshold"]).lower(),
                    "below_mnf_threshold": str(feature["below_mnf_threshold"]).lower(),
                    "sustained_crossing": str(feature["sustained_crossing"]).lower(),
                    "screen_status": feature["screen_status"],
                    "screen_label": display_status(str(feature["screen_status"])),
                    "screen_reason": feature["screen_reason"],
                    "interpretation": "Otsu-derived candidate activity and robust spectral thresholds are exploratory; not fatigue ground truth",
                })

    window_fields = (
        "recording_id", "participant_pseudonym", "side", "series", "filter_branch", "run_index", "window_index",
        "time_s", "time_fraction", "temporal_position", "time_position_proxy", "window_start_s", "window_end_s",
        "activity_candidate_occupancy_fraction", "nonoverlap_window_used", "contiguous_window_sequence",
        "baseline_window_count", "baseline_status", "mdf_hz", "mnf_hz", "mdf_baseline_median_hz",
        "mdf_robust_scale_hz", "mdf_lower_threshold_hz", "mnf_baseline_median_hz", "mnf_robust_scale_hz",
        "mnf_lower_threshold_hz", "below_mdf_threshold", "below_mnf_threshold", "sustained_crossing",
        "screen_status", "screen_label", "screen_reason", "interpretation",
    )
    phase_fields = (
        "recording_id", "participant_pseudonym", "side", "series", "temporal_position", "time_position_proxy",
        "filter_branch", "n_feature_windows", "n_nonoverlap_windows", "n_activity_candidate_windows",
        "n_baseline_windows", "baseline_status", "mdf_baseline_median_hz", "mdf_robust_scale_hz",
        "mdf_lower_threshold_hz", "mnf_baseline_median_hz", "mnf_robust_scale_hz", "mnf_lower_threshold_hz",
        "n_joint_below_threshold_windows", "n_sustained_crossing_windows", "screen_status", "screen_label",
        "screen_reason", "fatigue_ground_truth_available", "interpretation",
    )
    windows_path = run_dir / "fatigue_threshold_windows.csv"
    phases_path = run_dir / "fatigue_threshold_phases.csv"
    receipt_path = run_dir / "fatigue_threshold_receipt.json"
    _write_csv(windows_path, window_fields, window_rows)
    _write_csv(phases_path, phase_fields, phase_rows)
    receipt = {
        "analysis": "exploratory_spectral_change_threshold_screen",
        "processed_input_sha256": processed_input_sha256,
        "processed_input_note": "SHA-256 of the normalized ZIP consumed by this processor; when the user supplied a RAR, this is not the original RAR checksum.",
        "fatigue_ground_truth_available": False,
        "validated_fatigue_classifier_used": False,
        "clinical_fatigue_labels_emitted": False,
        "screen_status_categories": [
            "fatigue_compatible_change_exploratory",
            "no_sustained_threshold_crossing",
            "not_evaluable",
        ],
        "screen_status_interpretation": "Signal-change screen only. No-crossing does not establish absence of fatigue; exploratory crossing is not confirmed fatigue.",
        "baseline_definition": "First elapsed-time third; candidate-active windows with finite MDF and MNF; at least five non-overlapping windows per recording and filter branch.",
        "lower_threshold_formula": "median(baseline) - 1.96 * 1.4826 * MAD(baseline)",
        "mad_zero_rule": "threshold not estimable",
        "activity_candidate_mask": "Union of Otsu RMS-envelope candidate intervals; a window requires at least 50% candidate-interval overlap. Candidate intervals are not verified contractions.",
        "source_windowing": "1.0 s windows with 50% overlap from the descriptive processor; every other contiguous window is retained for the threshold screen to avoid overlapping persistence observations.",
        "persistence_rule": "MDF and MNF both strictly below their separate lower thresholds for at least three consecutive non-overlapping windows within one contiguous run.",
        "filter_branches": sorted({str(row.get("filter_branch", "")) for row in window_rows}),
        "window_rows": len(window_rows),
        "phase_rows": len(phase_rows),
        "outputs": {
            "fatigue_threshold_windows.csv": _sha256(windows_path),
            "fatigue_threshold_phases.csv": _sha256(phases_path),
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return phase_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_index(run_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    """Write a single-page overview with a complete, non-overlapping event map."""
    events: list[dict[str, str]] = []
    events_path = run_dir / "activity_event_candidates.csv"
    if events_path.is_file():
        with events_path.open("r", encoding="utf-8", newline="") as handle:
            events = list(csv.DictReader(handle))

    by_record: dict[str, list[dict[str, str]]] = {}
    for event in events:
        by_record.setdefault(event.get("recording_id", ""), []).append(event)

    def number(value: object, fallback: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return fallback
        return result if result == result and abs(result) != float("inf") else fallback

    def finite_values(values: list[object]) -> list[float]:
        parsed: list[float] = []
        for value in values:
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                parsed.append(candidate)
        return parsed

    phases = (
        ("start / inicio", "temprana", 0.0, 1.0 / 3.0),
        ("middle / medio", "transicion", 1.0 / 3.0, 2.0 / 3.0),
        ("final / final", "tardia", 2.0 / 3.0, 1.0),
    )

    def temporal_phase(fraction: float) -> tuple[str, str]:
        bounded = min(1.0, max(0.0, fraction))
        for label, proxy, lower, upper in phases:
            if bounded < upper or upper == 1.0:
                return label, proxy
        return phases[-1][0], phases[-1][1]

    def optional_number(value: object, decimals: int = 3) -> str:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return "not available"
        return f"{parsed:.{decimals}f}" if math.isfinite(parsed) else "not available"

    record_rows: list[dict[str, object]] = []
    max_time = 0.0
    total_continuous_s = 0.0
    for row in rows:
        recording_id = str(row.get("recording_id", ""))
        record_events = sorted(
            by_record.get(recording_id, []),
            key=lambda event: (number(event.get("start_time_s")), number(event.get("end_time_s"))),
        )
        lane_ends: list[float] = []
        for event in record_events:
            start = number(event.get("start_time_s"))
            end = max(start, number(event.get("end_time_s"), start))
            lane = next((i for i, lane_end in enumerate(lane_ends) if start >= lane_end - 1e-9), len(lane_ends))
            if lane == len(lane_ends):
                lane_ends.append(end)
            else:
                lane_ends[lane] = end
            event["_lane"] = str(lane)
            event["_duration_s"] = f"{max(0.0, end - start):.3f}"

        segment_path = run_dir / "recordings" / recording_id / "segments.csv"
        continuous_s = 0.0
        record_end = max((number(event.get("end_time_s")) for event in record_events), default=0.0)
        record_start = min((number(event.get("start_time_s")) for event in record_events), default=0.0)
        segment_starts: list[float] = []
        if segment_path.is_file():
            with segment_path.open("r", encoding="utf-8", newline="") as handle:
                for segment in csv.DictReader(handle):
                    if segment.get("status") == "processed":
                        start = number(segment.get("start_time_s"))
                        end = number(segment.get("end_time_s"), start)
                        segment_starts.append(start)
                        continuous_s += max(0.0, end - start)
                        record_end = max(record_end, end)
        if segment_starts:
            record_start = min(segment_starts)
        time_span = max(0.0, record_end - record_start)
        phase_counts = {proxy: {"count": 0, "duration_s": 0.0} for _, proxy, _, _ in phases}
        for event in record_events:
            start = number(event.get("start_time_s"))
            end = max(start, number(event.get("end_time_s"), start))
            midpoint_fraction = (0.5 * (start + end) - record_start) / time_span if time_span else 0.0
            phase_label, phase_proxy = temporal_phase(midpoint_fraction)
            event["_temporal_phase"] = phase_label
            event["_phase_proxy"] = phase_proxy
            phase_counts[phase_proxy]["count"] += 1
            phase_counts[phase_proxy]["duration_s"] += max(0.0, end - start)
        total_continuous_s += continuous_s
        max_time = max(max_time, record_end)
        record_rows.append({
            **row,
            "events": record_events,
            "lane_count": max(1, len(lane_ends)),
            "time_start_s": record_start,
            "time_end_s": record_end,
            "phase_counts": phase_counts,
            "continuous_duration_s": continuous_s,
        })
    screen_phase_rows = _derive_spectral_threshold_screen(
        run_dir, record_rows, events, phases, summary.get("input_zip_sha256")
    )
    screen_by_phase = {
        (str(row["recording_id"]), str(row["time_position_proxy"]), str(row["filter_branch"])): row
        for row in screen_phase_rows
    }
    max_time = max(max_time, 1.0)

    width, left, right = 1360, 310, 36
    plot_width = width - left - right
    row_heights = [max(34, 16 + int(item["lane_count"]) * 13) for item in record_rows]
    height = 72 + sum(row_heights) + 64
    tick_step = 5 if max_time <= 35 else 10
    tick_max = max(tick_step, math.ceil(max_time / tick_step) * tick_step)
    scale = plot_width / tick_max
    ticks = []
    for tick in range(0, tick_max + 1, tick_step):
        x = left + tick * scale
        ticks.append(
            f'<line x1="{x:.2f}" y1="48" x2="{x:.2f}" y2="{height-45}" stroke="#d9e2ec"/>'
            f'<text x="{x:.2f}" y="31" text-anchor="middle" class="axis">{tick}</text>'
        )
    svg_parts = [
        f'<svg name="overall-activity-timeline" role="img" aria-label="All activity candidate intervals by recording, using each recording local time axis" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<style>.axis{font:13px Arial;fill:#405064}.rowlabel{font:13px Arial;fill:#233447}.muted{font:11px Arial;fill:#607286}</style>',
        f'<text x="{left}" y="17" class="muted">Local time within each recording (s); recordings are separate, not concatenated in real time.</text>',
        *ticks,
    ]
    y = 50
    for item, row_height in zip(record_rows, row_heights):
        side = str(item.get("side", "unknown"))
        color = "#118ab2" if side == "right" else "#e07a29" if side == "left" else "#68778a"
        participant = str(item.get("participant_pseudonym") or "Group unknown")
        label = f'{item["recording_id"]} | {participant} | {side} | {item.get("series", "")}'
        safe_label = html.escape(label)
        svg_parts.append(f'<text x="{left-12}" y="{y+15}" text-anchor="end" class="rowlabel">{safe_label}</text>')
        svg_parts.append(f'<line x1="{left}" y1="{y+row_height-2}" x2="{left+plot_width}" y2="{y+row_height-2}" stroke="#9aa8b6"/>')
        for event in item["events"]:
            start = max(0.0, number(event.get("start_time_s")))
            end = max(start, number(event.get("end_time_s"), start))
            event_y = y + 4 + int(event.get("_lane", "0")) * 13
            x = left + start * scale
            bar_width = max(2.0, (end - start) * scale)
            idx = html.escape(str(event.get("event_candidate_index", "")), quote=True)
            svg_parts.append(
                f'<rect data-event="{idx}" data-lane="{event.get("_lane", "0")}" x="{x:.2f}" y="{event_y}" '
                f'width="{bar_width:.2f}" height="9" rx="2" fill="{color}"><title>{safe_label}; candidate {idx}: '
                f'{start:.3f}-{end:.3f} s; duration {max(0,end-start):.3f} s</title></rect>'
            )
        if not item["events"]:
            svg_parts.append(f'<text x="{left+4}" y="{y+16}" class="muted">No activity candidate exported</text>')
        y += row_height
    svg_parts.append(f'<text x="{left + plot_width/2:.2f}" y="{height-12}" text-anchor="middle" class="axis">Time from start of recording (s)</text></svg>')
    timeline_svg = "".join(svg_parts)

    series_counts = {f"R{i}": {"left": 0, "right": 0, "unknown": 0} for i in range(1, 6)}
    for event in events:
        series = event.get("series", "")
        side = event.get("side", "unknown")
        if series in series_counts:
            series_counts[series][side if side in {"left", "right"} else "unknown"] += 1
    chart_width, chart_height = 860, 290
    chart_left, chart_top, chart_plot_w, chart_plot_h = 72, 28, 750, 205
    count_max = max((sum(counts.values()) for counts in series_counts.values()), default=1) or 1
    count_scale = chart_plot_h / count_max
    count_parts = [f'<svg name="candidate-counts-by-series" role="img" aria-label="Candidate activity intervals per series and side" viewBox="0 0 {chart_width} {chart_height}" xmlns="http://www.w3.org/2000/svg">']
    for grid_index in range(5):
        y_grid = chart_top + chart_plot_h * grid_index / 4
        value = int(round(count_max * (1 - grid_index / 4)))
        count_parts.append(f'<line x1="{chart_left}" y1="{y_grid:.1f}" x2="{chart_left+chart_plot_w}" y2="{y_grid:.1f}" stroke="#d9e2ec"/><text x="{chart_left-10}" y="{y_grid+4:.1f}" text-anchor="end" font-size="12" fill="#405064">{value}</text>')
    bar_w = 42
    for i, (series, sides) in enumerate(series_counts.items()):
        total = sum(sides.values())
        x_center = chart_left + (i + 0.5) * chart_plot_w / 5
        bottom = chart_top + chart_plot_h
        for side, color in (("left", "#e07a29"), ("right", "#118ab2"), ("unknown", "#68778a")):
            value = sides[side]
            bar_h = value * count_scale
            if value:
                count_parts.append(f'<rect x="{x_center-bar_w/2:.1f}" y="{bottom-bar_h:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"><title>{series} {side}: {value}</title></rect>')
                bottom -= bar_h
        count_parts.append(f'<text x="{x_center:.1f}" y="{chart_top+chart_plot_h+20}" text-anchor="middle" font-size="13" fill="#233447">{series}</text>')
        count_parts.append(f'<text x="{x_center:.1f}" y="{max(chart_top+13,bottom-5):.1f}" text-anchor="middle" font-size="12" font-weight="bold" fill="#233447">{total}</text>')
    count_parts.extend([
        f'<rect x="{chart_left}" y="{chart_height-28}" width="12" height="12" fill="#e07a29"/><text x="{chart_left+18}" y="{chart_height-17}" font-size="12" fill="#405064">Left</text>',
        f'<rect x="{chart_left+100}" y="{chart_height-28}" width="12" height="12" fill="#118ab2"/><text x="{chart_left+118}" y="{chart_height-17}" font-size="12" fill="#405064">Right</text>',
        '</svg>',
    ])
    count_svg = "".join(count_parts)

    manifest_rows = []
    for item in record_rows:
        manifest_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(item.get(key, '')))}</td>" for key in (
                "recording_id", "participant_pseudonym", "side", "series", "status", "candidate_event_count"
            )) + f'<td>{number(item["continuous_duration_s"]):.2f}</td></tr>'
        )
    temporal_rows = []
    for item in record_rows:
        for label, proxy, _, _ in phases:
            phase_summary = item["phase_counts"][proxy]
            temporal_rows.append(
                "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
                    item.get("recording_id", ""), item.get("participant_pseudonym", "") or "Group unknown",
                    item.get("side", ""), item.get("series", ""), label, proxy,
                    phase_summary["count"], f'{phase_summary["duration_s"]:.3f}', "Not evaluated",
                )) + "</tr>"
            )

    feature_summary_rows = []
    for item in record_rows:
        recording_id = str(item.get("recording_id", ""))
        feature_path = run_dir / "recordings" / recording_id / "features.csv"
        buckets: dict[tuple[str, str], dict[str, list[object]]] = {}
        if feature_path.is_file():
            with feature_path.open("r", encoding="utf-8", newline="") as handle:
                for feature in csv.DictReader(handle):
                    proxy = str(feature.get("time_position_proxy", "")).strip().lower()
                    matching_phase = next((phase for phase in phases if phase[1] == proxy), None)
                    if matching_phase is None:
                        _, proxy = temporal_phase(number(feature.get("time_fraction")))
                        matching_phase = next(phase for phase in phases if phase[1] == proxy)
                    phase_label = matching_phase[0]
                    branch = str(feature.get("filter_branch", "unknown")) or "unknown"
                    bucket = buckets.setdefault((proxy, branch), {"rms": [], "mdf": [], "mnf": [], "count": []})
                    bucket["count"].append(1)
                    bucket["rms"].append(feature.get("rms_adc"))
                    bucket["mdf"].append(feature.get("mdf_hz"))
                    bucket["mnf"].append(feature.get("mnf_hz"))
        for (proxy, branch), bucket in sorted(buckets.items(), key=lambda pair: (pair[0][0], pair[0][1])):
            phase_label = next(phase[0] for phase in phases if phase[1] == proxy)
            screen = screen_by_phase.get((recording_id, proxy, branch), {})
            rms_values = finite_values(bucket["rms"])
            mdf_values = finite_values(bucket["mdf"])
            mnf_values = finite_values(bucket["mnf"])
            feature_summary_rows.append(
                "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
                    recording_id, item.get("participant_pseudonym", "") or "Group unknown",
                    item.get("side", ""), item.get("series", ""), phase_label, proxy, branch,
                    len(bucket["count"]),
                    f"{statistics.median(rms_values):.3f}" if rms_values else "not available",
                    f"{statistics.median(mdf_values):.3f}" if mdf_values else "not available",
                    f"{statistics.median(mnf_values):.3f}" if mnf_values else "not available",
                    screen.get("n_activity_candidate_windows", 0),
                    screen.get("n_baseline_windows", 0),
                    screen.get("baseline_status", "not available"),
                    optional_number(screen.get("mdf_lower_threshold_hz")),
                    optional_number(screen.get("mdf_robust_scale_hz")),
                    optional_number(screen.get("mnf_lower_threshold_hz")),
                    optional_number(screen.get("mnf_robust_scale_hz")),
                    screen.get("n_sustained_crossing_windows", 0),
                    screen.get("screen_label", "Not evaluable"),
                    screen.get("screen_reason", "not available"),
                )) + "</tr>"
            )

    event_table_rows = []
    for event in sorted(events, key=lambda item: (
        item.get("participant_pseudonym", ""), item.get("side", ""), item.get("series", ""),
        number(item.get("start_time_s")), item.get("recording_id", ""),
    )):
        start = number(event.get("start_time_s"))
        end = number(event.get("end_time_s"), start)
        fields = (
            event.get("recording_id", ""), event.get("participant_pseudonym", "") or "Group unknown",
            event.get("side", ""), event.get("series", ""), event.get("event_candidate_index", ""),
            f"{start:.3f}-{end:.3f}", f"{max(0.0, end-start):.3f}",
            f"{number(event.get('threshold_rms_adc')):.2f}", f"{number(event.get('mean_rms_adc')):.2f}",
            f"{number(event.get('peak_rms_adc')):.2f}", event.get("quality", ""),
            event.get("_temporal_phase", "not available"), "Not evaluated",
        )
        event_table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in fields) + "</tr>")

    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMG analysis - full run overview</title><style>
*{{box-sizing:border-box}} body{{font:15px/1.5 Segoe UI,Arial,sans-serif;margin:0 auto;max-width:1500px;padding:28px 3vw 56px;color:#17202a;background:#f6f8fa}}
h1,h2{{line-height:1.2}} h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:21px;margin:0 0 12px}}
.sub{{color:#586879;margin:0 0 20px}} .warning{{border-left:6px solid #d99b19;background:#fff8e6;padding:14px 18px;margin:18px 0}}
.status{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}}
.status div,.section{{background:white;border:1px solid #d5dde5;border-radius:8px;padding:14px}}
.status strong{{display:block;font-size:23px;color:#134e69}} .status span{{color:#586879;font-size:13px}}
.section{{margin:16px 0 22px;overflow-x:auto}} figure{{margin:0;min-width:760px}} svg{{width:100%;height:auto;display:block}}
.note{{color:#586879;font-size:13px;margin:10px 0 0}} table{{border-collapse:collapse;width:100%;margin-top:10px;background:#fff}}
th,td{{border:1px solid #d5dde5;padding:6px 8px;text-align:left;font-variant-numeric:tabular-nums}} th{{background:#edf2f6;position:sticky;top:0}}
.table-wrap{{max-height:560px;overflow:auto}} input{{padding:9px 10px;width:min(420px,100%);border:1px solid #aab7c4;border-radius:5px}}
.fatigue{{background:#f3f4f6;border-left:6px solid #6b7280;padding:14px 18px}}
</style></head><body>
<h1>EMG activity candidates — complete run overview</h1>
<p class="sub">One page for all CSVs in this run. Recording-specific time axes remain separate; no artificial continuous timeline is created.</p>
<div class="warning"><strong>Model status:</strong> no predictive model was loaded. The intervals below are exploratory activity candidates from an RMS-envelope threshold, not verified mechanical repetitions.</div>
<div class="fatigue"><strong>Validated fatigue prediction: unavailable.</strong> This run has no independent fatigue labels or validated fatigue classifier. The separate robust-threshold screen below flags exploratory spectral changes only; it does not confirm fatigue, and no threshold crossing does not establish absence of fatigue.</div>
<div class="status"><div><strong>{summary.get('csv_count', 0)}</strong><span>CSV files recognized</span></div><div><strong>{summary.get('processed_count', 0)}</strong><span>recordings processed</span></div><div><strong>{summary.get('candidate_event_count', 0)}</strong><span>activity candidates</span></div><div><strong>{total_continuous_s/60:.1f} min</strong><span>summed continuous recording duration; files remain separate</span></div><div><strong>No</strong><span>model loaded or fatigue scores emitted</span></div></div>
<section class="section"><h2>All event times across all recordings</h2><figure>{timeline_svg}</figure><p class="note">Each row uses the recording's own time from its start. If candidate intervals overlap within a recording, the chart stacks them in separate lanes without merging or dropping events. Bars mark threshold-derived intervals only.</p></section>
<section class="section"><h2>Candidate intervals by series and side</h2><figure>{count_svg}</figure><p class="note">Counts include all exported activity candidates; they are not repetition counts.</p></section>
<section class="section"><h2>Recording inventory</h2><div class="table-wrap"><table><thead><tr><th>Recording</th><th>Group</th><th>Side</th><th>Series</th><th>Status</th><th>Activity candidates</th><th>Summed continuous time (s)</th></tr></thead><tbody>{''.join(manifest_rows)}</tbody></table></div></section>
<section class="section"><h2>Temporal thirds (time-position proxy, not fatigue labels)</h2><p class="note">Start, middle and final refer only to thirds of each recording's elapsed-time span. Event candidates are assigned by their midpoint. These positions are not fatigue/no-fatigue classes.</p><div class="table-wrap"><table><thead><tr><th>Recording</th><th>Group</th><th>Side</th><th>Series</th><th>Temporal position</th><th>Stored proxy</th><th>Activity candidates</th><th>Candidate duration (s)</th><th>Fatigue label</th></tr></thead><tbody>{''.join(temporal_rows)}</tbody></table></div></section>
<section class="section"><h2>Windowed feature summaries and exploratory spectral threshold screen</h2><p class="note">Start, middle and final are elapsed-time thirds. Median RMS is in ADC counts; MDF and MNF are in Hz. For each recording and filter branch, the lower reference threshold is median of candidate-active windows in the first third minus 1.96 × 1.4826 × MAD; at least five baseline windows are required and zero MAD is not estimable. A screen crossing requires both MDF and MNF below their thresholds for three consecutive non-overlapping one-second windows in the same continuous run. The activity mask is the Otsu envelope candidate interval set, with at least 50% of a window overlapping it. These are exploratory signal-change categories, not fatigue/no-fatigue labels.</p><p class="note">Full window-level values and decisions are in fatigue_threshold_windows.csv; phase summaries and thresholds are in fatigue_threshold_phases.csv; calculation settings and hashes are in fatigue_threshold_receipt.json. All detailed EMG outputs remain in this local results folder.</p><div class="table-wrap"><table><thead><tr><th>Recording</th><th>Group</th><th>Side</th><th>Series</th><th>Temporal position</th><th>Stored proxy</th><th>Filter branch</th><th>Feature windows</th><th>Median RMS (ADC)</th><th>Median MDF (Hz)</th><th>Median MNF (Hz)</th><th>Candidate-active independent windows</th><th>Baseline windows</th><th>Baseline status</th><th>MDF lower threshold (Hz)</th><th>MDF robust scale (Hz)</th><th>MNF lower threshold (Hz)</th><th>MNF robust scale (Hz)</th><th>Sustained crossing windows</th><th>Exploratory screen result</th><th>Screen reason</th></tr></thead><tbody>{''.join(feature_summary_rows)}</tbody></table></div></section>
<section class="section"><h2>Complete candidate event table</h2><label for="event-search">Filter rows by recording, group, side, series, time, temporal position, fatigue status, or quality</label><br><input id="event-search" type="search" placeholder="Search this table"><div class="table-wrap"><table id="event-table"><thead><tr><th>Recording</th><th>Group</th><th>Side</th><th>Series</th><th>Candidate</th><th>Start-end (s)</th><th>Duration (s)</th><th>Envelope threshold (ADC)</th><th>Mean RMS (ADC)</th><th>Peak RMS (ADC)</th><th>Quality status</th><th>Temporal position</th><th>Fatigue status</th></tr></thead><tbody>{''.join(event_table_rows)}</tbody></table></div></section>
<p class="note">The stored Voltage_mV column is not validated as a physical calibration. ADC amplitude is reported in counts. Keep this report and source-derived results in restricted storage.</p>
<script>document.getElementById('event-search').addEventListener('input',function(){{const q=this.value.toLowerCase();for(const row of document.querySelectorAll('#event-table tbody tr'))row.hidden=!row.textContent.toLowerCase().includes(q)}});</script>
</body></html>"""
    (run_dir / "index.html").write_text(html_text, encoding="utf-8")


def _new_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"analisis_emg_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"analisis_emg_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def process_archive(zip_path: Path | str, output_base: Path | str, *, open_browser: bool = True) -> Path:
    """Run the F56 descriptive processor on compatible R1-R5 CSV members."""
    zip_file = Path(zip_path).expanduser().resolve(strict=True)
    base = Path(output_base).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    try:
        validate_data_paths({"input_zip": zip_file, "output_dir": base}, repository)
    except (ReleasePathError, OSError, ValueError) as error:
        raise ArchiveSafetyError("El ZIP y las salidas deben permanecer fuera del checkout público.") from error

    records = discover_recordings(zip_file)
    run_dir = _new_run_dir(base)
    processor = repository / "src" / "reproducir_emg_referencia.py"
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    candidate_total = 0

    try:
        with zipfile.ZipFile(zip_file, "r") as archive, tempfile.TemporaryDirectory(prefix="emg_zip_input_") as temp_name:
            temp_root = Path(temp_name)
            for record in records:
                row: dict[str, object] = {
                    "recording_id": record.recording_id,
                    "participant_pseudonym": record.participant_id or "",
                    "side": record.side,
                    "series": record.series or "",
                    "status": record.status,
                    "reason": record.reason,
                    "candidate_event_count": 0,
                    "output_folder": "",
                }
                if record.status != "ready":
                    rows.append(row)
                    continue

                member_info = archive.getinfo(record.member_name)
                source = temp_root / f"{record.recording_id}.csv"
                with archive.open(member_info, "r") as src, source.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

                output_name = record.recording_id
                record_output = run_dir / "recordings" / output_name
                command = [
                    sys.executable,
                    str(processor),
                    "--input",
                    str(source),
                    "--series",
                    str(record.series),
                    "--dashboard",
                    "--output-dir",
                    str(record_output),
                ]
                try:
                    completed = subprocess.run(
                        command,
                        cwd=repository,
                        capture_output=True,
                        text=True,
                        timeout=1800,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    row["status"] = "processing_timeout"
                    row["reason"] = "El procesamiento superó el límite de 30 minutos para esta grabación."
                    rows.append(row)
                    continue
                if completed.returncode != 0:
                    row["status"] = "processing_error"
                    row["reason"] = f"El procesador devolvió código {completed.returncode}; revise la entrada y el registro local."
                    rows.append(row)
                    continue

                candidates_path = record_output / "activity_candidates.csv"
                candidates: list[dict[str, str]] = []
                if candidates_path.is_file():
                    with candidates_path.open("r", encoding="utf-8", newline="") as handle:
                        candidates = list(csv.DictReader(handle))
                row["status"] = "processed"
                row["reason"] = "Procesado; eventos son candidatos de actividad, no repeticiones verificadas."
                row["candidate_event_count"] = len(candidates)
                row["output_folder"] = (Path("recordings") / output_name).as_posix()
                rows.append(row)
                candidate_total += len(candidates)
                for index, event in enumerate(candidates, start=1):
                    event_rows.append(
                        {
                            "recording_id": record.recording_id,
                            "participant_pseudonym": record.participant_id or "",
                            "side": record.side,
                            "series": record.series,
                            "event_candidate_index": index,
                            **event,
                        }
                    )
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as error:
        raise ArchiveSafetyError("No se pudo leer o procesar el ZIP; se conservaron los originales.") from error

    _write_csv(run_dir / "run_manifest.csv", MANIFEST_FIELDS, rows)
    _write_csv(run_dir / "activity_event_candidates.csv", EVENT_FIELDS, event_rows)
    processed = sum(row["status"] == "processed" for row in rows)
    skipped = len(rows) - processed
    summary: dict[str, object] = {
        "analysis": "descriptive_raw_emg_pipeline",
        "pipeline_version": "public_F59",
        "input_zip_sha256": _sha256(zip_file),
        "csv_count": len(records),
        "processed_count": processed,
        "skipped_count": skipped,
        "candidate_event_count": candidate_total,
        "model_loaded": False,
        "mechanical_repetition_prediction": None,
        "physiological_fatigue_prediction": None,
        "interpretation": (
            "Features and activity candidates only. The package has no validated repetition labels, "
            "no independent quadriceps fatigue target, and no compatible final model."
        ),
        "private_output_warning": "Signals are embedded in per-recording dashboards; keep this folder restricted.",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_index(run_dir, summary, rows)
    if open_browser and processed:
        webbrowser.open((run_dir / "index.html").resolve().as_uri())
    return run_dir


def process_single_csv(csv_path: Path | str, output_base: Path | str, *, open_browser: bool = True) -> Path:
    """Run the descriptive processor on one R1-R5 CSV, without cohort inference."""
    csv_file = Path(csv_path).expanduser().resolve(strict=True)
    base = Path(output_base).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    try:
        validate_data_paths({"input_csv": csv_file, "output_dir": base}, repository)
    except (ReleasePathError, OSError, ValueError) as error:
        raise ArchiveSafetyError("El CSV y las salidas deben permanecer fuera del checkout público.") from error
    record = discover_single_csv(csv_file)
    run_dir = _new_run_dir(base)
    record_output = run_dir / "recordings" / record.recording_id
    processor = repository / "src" / "reproducir_emg_referencia.py"
    row: dict[str, object] = {
        "recording_id": record.recording_id,
        "participant_pseudonym": "",
        "side": record.side,
        "series": record.series or "",
        "status": "processing_error",
        "reason": "El procesador no completó el CSV.",
        "candidate_event_count": 0,
        "output_folder": "",
    }
    candidates: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="emg_csv_input_") as temp_name:
        safe_source = Path(temp_name) / "REC0001.csv"
        shutil.copyfile(csv_file, safe_source)
        command = [
            sys.executable,
            str(processor),
            "--input",
            str(safe_source),
            "--series",
            str(record.series),
            "--dashboard",
            "--output-dir",
            str(record_output),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repository,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
        except subprocess.TimeoutExpired:
            row["reason"] = "El procesamiento superó el límite de 30 minutos."
        else:
            if completed.returncode == 0:
                candidates_path = record_output / "activity_candidates.csv"
                if candidates_path.is_file():
                    with candidates_path.open("r", encoding="utf-8", newline="") as handle:
                        candidates = list(csv.DictReader(handle))
                row.update(
                    status="processed",
                    reason="Procesado; eventos son candidatos de actividad, no repeticiones verificadas.",
                    candidate_event_count=len(candidates),
                    output_folder=(Path("recordings") / record.recording_id).as_posix(),
                )
            else:
                row["reason"] = f"El procesador devolvió código {completed.returncode}; revise la calidad y el esquema del CSV."

    _write_csv(run_dir / "run_manifest.csv", MANIFEST_FIELDS, [row])
    event_rows = [
        {
            "recording_id": record.recording_id,
            "participant_pseudonym": "",
            "side": record.side,
            "series": record.series,
            "event_candidate_index": index,
            **event,
        }
        for index, event in enumerate(candidates, start=1)
    ]
    _write_csv(run_dir / "activity_event_candidates.csv", EVENT_FIELDS, event_rows)
    processed = int(row["status"] == "processed")
    summary: dict[str, object] = {
        "analysis": "descriptive_raw_emg_pipeline",
        "pipeline_version": "public_F59",
        "input_csv_sha256": _sha256(csv_file),
        "csv_count": 1,
        "processed_count": processed,
        "skipped_count": 1 - processed,
        "candidate_event_count": len(candidates),
        "model_loaded": False,
        "mechanical_repetition_prediction": None,
        "physiological_fatigue_prediction": None,
        "input_scope": "one caller-supplied CSV; no cohort inference or participant identity",
        "interpretation": (
            "Features and activity candidates only. No validated repetition labels or independent "
            "quadriceps fatigue target are available for this input."
        ),
        "private_output_warning": "Signals are embedded in per-recording dashboards; keep this folder restricted.",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_index(run_dir, summary, [row])
    if open_browser and processed:
        webbrowser.open((run_dir / "index.html").resolve().as_uri())
    return run_dir


def default_output_dir() -> Path:
    """Return a per-user local default, avoiding the commonly synced Documents folder."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata).expanduser() / "EMG_Biomecanica" / "Resultados"
    return Path.home() / ".local" / "share" / "EMG_Biomecanica" / "Resultados"


def _pick_paths() -> tuple[Path, Path] | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as error:  # pragma: no cover - platform UI dependency
        raise RuntimeError("No se pudo iniciar el selector de archivos de Windows.") from error
    root = tk.Tk()
    root.withdraw()
    try:
        zip_name = filedialog.askopenfilename(
            title="Seleccione un CSV EMG o un ZIP de grabaciones",
            filetypes=[("CSV EMG o ZIP", "*.csv *.zip"), ("CSV EMG", "*.csv"), ("Archivos ZIP", "*.zip")],
        )
        if not zip_name:
            return None
        default_dir = default_output_dir()
        default_dir.mkdir(parents=True, exist_ok=True)
        output_name = filedialog.askdirectory(
            title="Seleccione una carpeta local restringida; evite carpetas sincronizadas o compartidas",
            initialdir=str(default_dir),
            mustexist=False,
        )
        if not output_name:
            return None
        return Path(zip_name), Path(output_name)
    finally:
        root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Procesamiento descriptivo de un CSV o un ZIP EMG; no ejecuta un modelo de fatiga ni valida repeticiones."
    )
    parser.add_argument("--pick", action="store_true", help="abre selectores nativos para CSV/ZIP y carpeta de resultados")
    parser.add_argument("--csv", type=Path, help="CSV de una grabación formal R1-R5")
    parser.add_argument("--zip", type=Path, help="ZIP de CSV EMG")
    parser.add_argument("--output-dir", type=Path, help="carpeta padre para resultados privados")
    parser.add_argument("--no-open", action="store_true", help="no abrir la página de resultados al terminar")
    args = parser.parse_args(argv)
    if args.pick:
        if args.csv is not None or args.zip is not None or args.output_dir is not None:
            parser.error("--pick no se combina con --csv/--zip/--output-dir")
        try:
            selected = _pick_paths()
        except RuntimeError as error:
            parser.error(str(error))
        if selected is None:
            print("Selección cancelada; no se modificaron datos.")
            return 0
        input_path, output_dir = selected
    else:
        if (args.csv is None) == (args.zip is None):
            parser.error("indique exactamente una de --csv RUTA, --zip RUTA o use --pick")
        input_path = args.csv if args.csv is not None else args.zip
        output_dir = args.output_dir or default_output_dir()

    try:
        if input_path.suffix.lower() == ".csv":
            run_dir = process_single_csv(input_path, output_dir, open_browser=not args.no_open)
        elif input_path.suffix.lower() == ".zip":
            run_dir = process_archive(input_path, output_dir, open_browser=not args.no_open)
        else:
            raise ArchiveSafetyError("Elija un archivo CSV o ZIP.")
    except (ArchiveSafetyError, OSError, ValueError) as error:
        print(f"No se pudo procesar el ZIP: {error}", file=sys.stderr)
        return 2
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Carpeta de resultados: {run_dir}")
    print(
        "Procesadas: {processed_count}; omitidas/no evaluables: {skipped_count}; "
        "eventos candidatos: {candidate_event_count}.".format(**summary)
    )
    print("No se calcularon conteos de repeticiones ni predicciones de fatiga.")
    if not summary["processed_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
