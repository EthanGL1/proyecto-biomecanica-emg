"""Audit timestamp and trial pairing between Cerqueira sEMG and self-report data.

The audit is descriptive: matching timestamps do not prove hardware trigger
synchronization or make self-reported fatigue a functional force-loss label.
It reads the official nested archives in place and never modifies them.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import copy
import hashlib
import io
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterable

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from output_safety import PublicCheckoutOutputError, ensure_outside_public_checkout


EXPECTED_EMG_MD5 = "ea7338b95543ada98b1ddc65d4b6aabc"
EXPECTED_LABEL_MD5 = "3352c990b2baf2246cfa7b331e52607a"
_SUBJECT_RE = re.compile(r"(?:^|/)subject_(\d+)\.zip$", re.IGNORECASE)
_TRIAL_RE = re.compile(r"(?:^|/)subject_(\d+)/trial_(\d+)\.csv$", re.IGNORECASE)
_TIME_COLUMN_RE = re.compile(r"^x\s*\[s\](?:\.\d+)?$", re.IGNORECASE)

# Generous limits sized above the checked official corpus. They bound archive
# expansion and row accumulation for other inputs accepted by this public CLI.
MAX_ARCHIVE_FILE_BYTES = 8 * 1024**3
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024**2
MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_ARCHIVE_COMPRESSION_RATIO = 10_000.0
MAX_SIGNAL_ROWS = 1_000_000
MAX_LABEL_ROWS = 1_000_000


def _validate_archive_path(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise AlignmentAuditError("Cannot inspect archive file size") from error
    if size > MAX_ARCHIVE_FILE_BYTES:
        raise AlignmentAuditError("Archive exceeds configured resource limit")


def _validate_zip_members(infos: list[zipfile.ZipInfo]) -> None:
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise AlignmentAuditError("Archive exceeds configured resource limit: member count")
    total_uncompressed = 0
    for info in infos:
        if info.is_dir():
            continue
        if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise AlignmentAuditError("Archive exceeds configured resource limit: member size")
        total_uncompressed += info.file_size
        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > MAX_ARCHIVE_COMPRESSION_RATIO:
            raise AlignmentAuditError("Archive exceeds configured resource limit: compression ratio")
    if total_uncompressed > MAX_ARCHIVE_TOTAL_UNCOMPRESSED_BYTES:
        raise AlignmentAuditError("Archive exceeds configured resource limit: expanded size")


def _spreadsheet_safe_cell(value: object) -> object:
    """Keep untrusted strings inert when a CSV is opened in a spreadsheet."""
    if not isinstance(value, str):
        return value
    significant = value.lstrip(" \t\r\n\v\f\ufeff")
    if value.startswith(("\t", "\r", "\n")) or (
        significant and significant[0] in "=+-@"
    ):
        return "'" + value
    return value


class AlignmentAuditError(ValueError):
    """Raised when an archive cannot satisfy the expected data contract."""


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _outer_subject_entries(path: Path) -> dict[int, zipfile.ZipInfo]:
    entries: dict[int, zipfile.ZipInfo] = {}
    _validate_archive_path(path)
    with zipfile.ZipFile(path) as outer:
        infos = outer.infolist()
        _validate_zip_members(infos)
        for info in infos:
            if info.is_dir():
                continue
            match = _SUBJECT_RE.search(info.filename)
            if not match:
                continue
            subject_id = int(match.group(1))
            if subject_id in entries:
                raise AlignmentAuditError(
                    f"Duplicate nested subject archive for index {subject_id}: {path.name}"
                )
            entries[subject_id] = info
    return entries


def _nested_trial_entries(
    nested: zipfile.ZipFile, subject_id: int
) -> dict[int, zipfile.ZipInfo]:
    entries: dict[int, zipfile.ZipInfo] = {}
    infos = nested.infolist()
    _validate_zip_members(infos)
    for info in infos:
        if info.is_dir():
            continue
        match = _TRIAL_RE.search(info.filename)
        if not match:
            continue
        entry_subject, trial_id = map(int, match.groups())
        if entry_subject != subject_id:
            raise AlignmentAuditError(
                f"Nested path subject mismatch: expected {subject_id}, got {info.filename}"
            )
        if trial_id in entries:
            raise AlignmentAuditError(
                f"Duplicate Trial_{trial_id} for subject index {subject_id}"
            )
        entries[trial_id] = info
    return entries


def _read_signal_csv(stream: BinaryIO) -> dict[str, object]:
    text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    try:
        reader = csv.reader(text)
        header = next(reader, None)
        if not header:
            raise AlignmentAuditError("Empty sEMG CSV")
        time_indices = [
            index for index, name in enumerate(header) if _TIME_COLUMN_RE.fullmatch(name.strip())
        ]
        signal_indices = [index for index in range(len(header)) if index not in time_indices]
        if not time_indices or len(time_indices) != len(signal_indices):
            raise AlignmentAuditError(
                f"Expected paired time/sEMG columns; header has {len(time_indices)} time and "
                f"{len(signal_indices)} signal columns: {header}"
            )

        times: list[float] = []
        nonfinite_signal_values = 0
        malformed_rows = 0
        channel_time_mismatch_rows = 0
        first_channel_time_mismatch_row_index: int | None = None
        first_channel_time_mismatch_times: list[float] | None = None
        channel_time_mismatch_max_abs_s = 0.0
        channel_time_mismatch_details: list[dict[str, object]] = []
        previous_time_values: list[float] | None = None
        for row_index, row in enumerate(reader):
            if row_index >= MAX_SIGNAL_ROWS:
                raise AlignmentAuditError("sEMG CSV exceeds configured row limit")
            if len(row) != len(header):
                malformed_rows += 1
                continue
            try:
                time_values = [float(row[index]) for index in time_indices]
                signal_values = [float(row[index]) for index in signal_indices]
            except (TypeError, ValueError):
                malformed_rows += 1
                continue
            if not all(math.isfinite(value) for value in time_values):
                malformed_rows += 1
                continue
            channel_time_delta = max(time_values) - min(time_values)
            channel_time_mismatch_max_abs_s = max(
                channel_time_mismatch_max_abs_s, channel_time_delta
            )
            if channel_time_delta > 1e-9:
                channel_time_mismatch_rows += 1
                channel_time_mismatch_details.append(
                    {
                        "row_index": row_index,
                        "time_values_s": time_values,
                        "deviation_s": channel_time_delta,
                        "previous_time_values_s": previous_time_values,
                        "next_time_values_s": None,
                    }
                )
                if first_channel_time_mismatch_row_index is None:
                    first_channel_time_mismatch_row_index = row_index
                    first_channel_time_mismatch_times = time_values
            for detail in channel_time_mismatch_details:
                if (
                    detail["next_time_values_s"] is None
                    and row_index == int(detail["row_index"]) + 1
                ):
                    detail["next_time_values_s"] = time_values
            nonfinite_signal_values += sum(not math.isfinite(value) for value in signal_values)
            times.append(time_values[0])
            previous_time_values = time_values

        deltas = [right - left for left, right in zip(times, times[1:])]
        positive_deltas = [delta for delta in deltas if delta > 0 and math.isfinite(delta)]
        median_dt = statistics.median(positive_deltas) if positive_deltas else None
        nonincreasing = sum(delta <= 0 for delta in deltas)
        first_nonincreasing_index = next(
            (index for index, delta in enumerate(deltas, start=1) if delta <= 0),
            None,
        )
        monotonic_prefix_count = (
            first_nonincreasing_index if first_nonincreasing_index is not None else len(times)
        )
        monotonic_times = times[:monotonic_prefix_count]
        gap_15 = sum(delta > 1.5 * median_dt for delta in positive_deltas) if median_dt else 0
        gap_2x = sum(delta > 2.0 * median_dt for delta in positive_deltas) if median_dt else 0
        return {
            "signal_sample_count": len(times),
            "signal_monotonic_prefix_sample_count": monotonic_prefix_count,
            "signal_channel_count": len(signal_indices),
            "signal_channels": [header[index].strip() for index in signal_indices],
            "signal_time_channel_columns": [header[index].strip() for index in time_indices],
            "signal_time_start_s": monotonic_times[0] if monotonic_times else None,
            "signal_time_end_s": monotonic_times[-1] if monotonic_times else None,
            "signal_observed_time_start_s": times[0] if times else None,
            "signal_observed_time_end_s": times[-1] if times else None,
            "signal_duration_s": (
                monotonic_times[-1] - monotonic_times[0] if len(monotonic_times) > 1 else 0.0
            ),
            "signal_sampling_hz_median": 1.0 / median_dt if median_dt else None,
            "signal_dt_median_s": median_dt,
            "signal_dt_min_s": min(positive_deltas) if positive_deltas else None,
            "signal_dt_p95_s": _quantile(positive_deltas, 0.95),
            "signal_dt_max_s": max(positive_deltas) if positive_deltas else None,
            "signal_nonincreasing_intervals": nonincreasing,
            "signal_intervals_gt_1_5x_median": gap_15,
            "signal_intervals_gt_2x_median": gap_2x,
            "signal_first_nonincreasing_interval_index": first_nonincreasing_index,
            "signal_first_nonincreasing_previous_time_s": (
                times[first_nonincreasing_index - 1]
                if first_nonincreasing_index is not None
                else None
            ),
            "signal_first_nonincreasing_current_time_s": (
                times[first_nonincreasing_index]
                if first_nonincreasing_index is not None
                else None
            ),
            "signal_channel_time_mismatch_rows": channel_time_mismatch_rows,
            "signal_first_channel_time_mismatch_row_index": first_channel_time_mismatch_row_index,
            "signal_first_channel_time_mismatch_times_s": first_channel_time_mismatch_times,
            "signal_channel_time_mismatch_max_abs_s": channel_time_mismatch_max_abs_s,
            "signal_channel_time_mismatch_details": channel_time_mismatch_details,
            "signal_nonfinite_values": nonfinite_signal_values,
            "signal_malformed_rows": malformed_rows,
            "_signal_times": monotonic_times,
        }
    finally:
        text.close()


def _read_label_csv(stream: BinaryIO) -> dict[str, object]:
    text = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(text)
        if reader.fieldnames != ["time", "label"]:
            raise AlignmentAuditError(f"Unexpected fatigue-label CSV header: {reader.fieldnames}")
        times: list[float] = []
        labels: list[int] = []
        malformed_rows = 0
        for row_index, row in enumerate(reader):
            if row_index >= MAX_LABEL_ROWS:
                raise AlignmentAuditError("Fatigue-label CSV exceeds configured row limit")
            try:
                time_s = float(row["time"])
                label = int(row["label"])
            except (TypeError, ValueError, KeyError):
                malformed_rows += 1
                continue
            if not math.isfinite(time_s) or label not in (0, 1, 2):
                malformed_rows += 1
                continue
            times.append(time_s)
            labels.append(label)

        class_counts = Counter(labels)
        usable_count = len(times)
        terminal_anomalies = 0
        internal_anomalies = 0
        for index in range(1, len(times)):
            if times[index] <= times[index - 1]:
                if index == len(times) - 1:
                    terminal_anomalies += 1
                    usable_count = index
                else:
                    internal_anomalies += 1
                    usable_count = index
                break
        usable_times = times[:usable_count]
        usable_deltas = [
            right - left
            for left, right in zip(usable_times, usable_times[1:])
            if right > left
        ]
        median_dt = statistics.median(usable_deltas) if usable_deltas else None
        transitions = sum(left != right for left, right in zip(labels, labels[1:]))
        return {
            "label_rows": len(labels),
            "usable_label_rows": usable_count,
            "label_class_counts": {str(key): class_counts[key] for key in (0, 1, 2)},
            "label_transitions": transitions,
            "label_time_start_s": usable_times[0] if usable_times else None,
            "label_time_end_s": usable_times[-1] if usable_times else None,
            "label_duration_s": usable_times[-1] - usable_times[0] if len(usable_times) > 1 else 0.0,
            "label_sampling_hz_median": 1.0 / median_dt if median_dt else None,
            "label_dt_median_s": median_dt,
            "label_dt_min_s": min(usable_deltas) if usable_deltas else None,
            "label_dt_max_s": max(usable_deltas) if usable_deltas else None,
            "terminal_label_timestamp_anomalies": terminal_anomalies,
            "internal_label_timestamp_anomalies": internal_anomalies,
            "label_malformed_rows": malformed_rows,
            "_label_times": usable_times,
            "_label_values": labels[:usable_count],
        }
    finally:
        text.close()


def _pair_trial(
    subject_id: int,
    trial_id: int,
    signal_info: zipfile.ZipInfo,
    signal_archive: zipfile.ZipFile,
    label_info: zipfile.ZipInfo,
    label_archive: zipfile.ZipFile,
) -> dict[str, object]:
    with signal_archive.open(signal_info) as stream:
        signal = _read_signal_csv(stream)
    with label_archive.open(label_info) as stream:
        labels = _read_label_csv(stream)

    signal_times = signal.pop("_signal_times")
    label_times = labels.pop("_label_times")
    label_values = labels.pop("_label_values")
    signal_start = signal["signal_time_start_s"]
    signal_end = signal["signal_time_end_s"]
    label_start = labels["label_time_start_s"]
    label_end = labels["label_time_end_s"]
    if signal_times and signal_times == sorted(signal_times):
        support_indices = [
            index
            for index, time_s in enumerate(label_times)
            if signal_start is not None and signal_end is not None and signal_start <= time_s <= signal_end
        ]
        support_times = [label_times[index] for index in support_indices]
        nearest_errors: list[float] = []
        for time_s in support_times:
            index = bisect.bisect_left(signal_times, time_s)
            candidates = signal_times[max(0, index - 1) : min(len(signal_times), index + 1)]
            if candidates:
                nearest_errors.append(min(abs(candidate - time_s) for candidate in candidates))
    else:
        support_indices = []
        support_times = []
        nearest_errors = []

    within_count = len(support_times)
    usable_label_count = int(labels["usable_label_rows"])
    coverage = within_count / usable_label_count if usable_label_count else 0.0
    support_class_counts = Counter(label_values[index] for index in support_indices)
    time_axes_match = (
        signal["signal_channel_time_mismatch_rows"] == 0
        and signal["signal_nonincreasing_intervals"] == 0
    )
    if coverage == 1.0 and usable_label_count and time_axes_match:
        alignment = "complete_timestamp_overlap"
    elif within_count:
        alignment = "partial_timestamp_overlap_or_signal_time_issue"
    else:
        alignment = "no_timestamp_overlap"

    result: dict[str, object] = {
        "subject_index": subject_id,
        "trial_index": trial_id,
        "signal_path_in_archive": signal_info.filename,
        "label_path_in_archive": label_info.filename,
        **signal,
        **labels,
        "labels_within_signal_time_support": within_count,
        "label_class_counts_in_emg_time_support": {
            str(key): support_class_counts[key] for key in (0, 1, 2)
        },
        "label_time_support_fraction": coverage,
        "label_start_minus_signal_start_s": (
            label_start - signal_start
            if label_start is not None and signal_start is not None
            else None
        ),
        "label_end_minus_signal_end_s": (
            label_end - signal_end if label_end is not None and signal_end is not None else None
        ),
        "nearest_emg_sample_error_median_s": statistics.median(nearest_errors)
        if nearest_errors
        else None,
        "nearest_emg_sample_error_p95_s": _quantile(nearest_errors, 0.95),
        "nearest_emg_sample_error_max_s": max(nearest_errors) if nearest_errors else None,
        "timestamp_pairing_status": alignment,
        "task_boundary_note": (
            "Trial index is matched from paired filenames/protocol table; within-trial movement onset/offset "
            "is not separately marked by these two CSVs."
        ),
    }
    return result


def audit_alignment(
    emg_archive_path: str | Path,
    label_archive_path: str | Path,
    *,
    expected_emg_md5: str | None = None,
    expected_label_md5: str | None = None,
) -> dict[str, object]:
    """Pair EMG and label trials by numeric subject/trial keys and compare time support."""
    emg_path = Path(emg_archive_path)
    label_path = Path(label_archive_path)
    for path in (emg_path, label_path):
        if not path.is_file():
            raise AlignmentAuditError(f"Archive does not exist: {path}")
    emg_md5 = _md5(emg_path)
    label_md5 = _md5(label_path)
    if expected_emg_md5 and emg_md5.lower() != expected_emg_md5.lower():
        raise AlignmentAuditError(
            f"sEMG archive MD5 mismatch: expected {expected_emg_md5.lower()}, got {emg_md5.lower()}"
        )
    if expected_label_md5 and label_md5.lower() != expected_label_md5.lower():
        raise AlignmentAuditError(
            f"label archive MD5 mismatch: expected {expected_label_md5.lower()}, got {label_md5.lower()}"
        )

    emg_outer_entries = _outer_subject_entries(emg_path)
    label_outer_entries = _outer_subject_entries(label_path)
    emg_subjects = set(emg_outer_entries)
    label_subjects = set(label_outer_entries)
    trial_rows: list[dict[str, object]] = []
    unmatched_emg: list[tuple[int, int]] = []
    unmatched_labels: list[tuple[int, int]] = []
    class_counts: Counter[str] = Counter({"0": 0, "1": 0, "2": 0})
    supported_class_counts: Counter[str] = Counter({"0": 0, "1": 0, "2": 0})
    all_emg_trials = all_label_trials = 0

    with zipfile.ZipFile(emg_path) as emg_outer, zipfile.ZipFile(label_path) as label_outer:
        for subject_id in sorted(emg_subjects | label_subjects):
            raw_info = emg_outer_entries.get(subject_id)
            label_info = label_outer_entries.get(subject_id)
            raw_bytes = emg_outer.read(raw_info) if raw_info else None
            label_bytes = label_outer.read(label_info) if label_info else None

            raw_entries: dict[int, zipfile.ZipInfo] = {}
            label_entries: dict[int, zipfile.ZipInfo] = {}
            raw_nested = zipfile.ZipFile(io.BytesIO(raw_bytes)) if raw_bytes is not None else None
            label_nested = zipfile.ZipFile(io.BytesIO(label_bytes)) if label_bytes is not None else None
            try:
                if raw_nested:
                    raw_entries = _nested_trial_entries(raw_nested, subject_id)
                    all_emg_trials += len(raw_entries)
                if label_nested:
                    label_entries = _nested_trial_entries(label_nested, subject_id)
                    all_label_trials += len(label_entries)
                for trial_id in sorted(set(raw_entries) | set(label_entries)):
                    if trial_id not in raw_entries:
                        unmatched_labels.append((subject_id, trial_id))
                        continue
                    if trial_id not in label_entries:
                        unmatched_emg.append((subject_id, trial_id))
                        continue
                    if raw_nested is None or label_nested is None:
                        continue
                    trial = _pair_trial(
                        subject_id,
                        trial_id,
                        raw_entries[trial_id],
                        raw_nested,
                        label_entries[trial_id],
                        label_nested,
                    )
                    class_counts.update(trial["label_class_counts"])
                    supported_class_counts.update(trial["label_class_counts_in_emg_time_support"])
                    trial_rows.append(trial)
            finally:
                if raw_nested:
                    raw_nested.close()
                if label_nested:
                    label_nested.close()

    trial_rows.sort(key=lambda row: (int(row["subject_index"]), int(row["trial_index"])))
    matched_subjects = emg_subjects & label_subjects
    overlap_trials = sum(
        int(row["labels_within_signal_time_support"]) for row in trial_rows
    )
    usable_labels = sum(int(row["usable_label_rows"]) for row in trial_rows)
    support_complete_count = sum(
        row["timestamp_pairing_status"] == "complete_timestamp_overlap" for row in trial_rows
    )
    summary = {
        "emg_archive": emg_path.name,
        "emg_archive_bytes": emg_path.stat().st_size,
        "emg_archive_md5": emg_md5,
        "expected_emg_md5": expected_emg_md5,
        "emg_checksum_verified": expected_emg_md5 is None or emg_md5.lower() == expected_emg_md5.lower(),
        "label_archive": label_path.name,
        "label_archive_bytes": label_path.stat().st_size,
        "label_archive_md5": label_md5,
        "expected_label_md5": expected_label_md5,
        "label_checksum_verified": expected_label_md5 is None or label_md5.lower() == expected_label_md5.lower(),
        "emg_subject_count": len(emg_subjects),
        "label_subject_count": len(label_subjects),
        "matched_subject_count": len(matched_subjects),
        "unmatched_emg_subjects": sorted(emg_subjects - label_subjects),
        "unmatched_label_subjects": sorted(label_subjects - emg_subjects),
        "emg_trial_count": all_emg_trials,
        "label_trial_count": all_label_trials,
        "matched_trial_count": len(trial_rows),
        "unmatched_emg_trial_count": len(unmatched_emg),
        "unmatched_label_trial_count": len(unmatched_labels),
        "unmatched_emg_trials": unmatched_emg,
        "unmatched_label_trials": unmatched_labels,
        "timestamp_overlap_complete_trials": support_complete_count,
        "matched_trial_count_with_any_timestamp_overlap": sum(
            int(row["labels_within_signal_time_support"]) > 0 for row in trial_rows
        ),
        "labels_in_emg_time_support": overlap_trials,
        "usable_label_rows_in_matched_trials": usable_labels,
        "signal_trials_with_nonincreasing_timestamps": sum(
            int(row["signal_nonincreasing_intervals"]) > 0 for row in trial_rows
        ),
        "signal_trials_with_channel_time_mismatch": sum(
            int(row["signal_channel_time_mismatch_rows"]) > 0 for row in trial_rows
        ),
        "signal_channel_time_mismatch_rows_total": sum(
            int(row["signal_channel_time_mismatch_rows"]) for row in trial_rows
        ),
        "label_class_counts_in_matched_trials": {
            key: class_counts[key] for key in ("0", "1", "2")
        },
        "label_class_counts_in_emg_time_support": {
            key: supported_class_counts[key] for key in ("0", "1", "2")
        },
        "pairing_complete": (
            not unmatched_emg
            and not unmatched_labels
            and emg_subjects == label_subjects
            and all_emg_trials == all_label_trials
        ),
        "synchronization_claim": (
            "Trial filenames and timestamp support are audited. Shared timestamp basis is consistent with "
            "the official notebook's nearest-time mapping, but independent hardware-trigger synchronization "
            "is not documented by the archive."
        ),
        "fatigue_label_semantics": "self-reported upper-limb fatigue 0/1/2; not quadriceps functional fatigue",
    }
    return {"schema_version": 1, "summary": summary, "trials": trial_rows}


def repair_nonmonotonic_trials(
    emg_archive_path: str | Path,
    label_archive_path: str | Path,
    existing_report: dict[str, object],
    *,
    expected_emg_md5: str | None = None,
    expected_label_md5: str | None = None,
) -> dict[str, object]:
    """Reprocess only trials flagged for signal-time integrity anomalies.

    This supports a focused correction of an earlier exhaustive report without
    rescanning the unaffected trials. The input report remains unchanged.
    """
    emg_path = Path(emg_archive_path)
    label_path = Path(label_archive_path)
    summary = existing_report.get("summary")
    if not isinstance(summary, dict) or not isinstance(existing_report.get("trials"), list):
        raise AlignmentAuditError("Existing report must contain summary and trials fields")
    if not emg_path.is_file() or not label_path.is_file():
        raise AlignmentAuditError("Both source archives must exist")

    emg_md5 = _md5(emg_path)
    label_md5 = _md5(label_path)
    if emg_md5 != summary.get("emg_archive_md5"):
        raise AlignmentAuditError("EMG archive checksum differs from the existing report")
    if label_md5 != summary.get("label_archive_md5"):
        raise AlignmentAuditError("Label archive checksum differs from the existing report")
    if expected_emg_md5 and emg_md5.lower() != expected_emg_md5.lower():
        raise AlignmentAuditError("EMG archive does not match the expected checksum")
    if expected_label_md5 and label_md5.lower() != expected_label_md5.lower():
        raise AlignmentAuditError("Label archive does not match the expected checksum")

    report = copy.deepcopy(existing_report)
    trial_rows = report["trials"]
    replacements = {
        (int(row["subject_index"]), int(row["trial_index"])): index
        for index, row in enumerate(trial_rows)
        if int(row.get("signal_nonincreasing_intervals", 0)) > 0
        or int(row.get("signal_channel_time_mismatch_rows", 0)) > 0
    }
    for row in trial_rows:
        if (int(row["subject_index"]), int(row["trial_index"])) in replacements:
            continue
        row.setdefault("signal_monotonic_prefix_sample_count", row["signal_sample_count"])
        row.setdefault("signal_observed_time_start_s", row.get("signal_time_start_s"))
        row.setdefault("signal_observed_time_end_s", row.get("signal_time_end_s"))
        row.setdefault("signal_first_nonincreasing_interval_index", None)
        row.setdefault("signal_first_nonincreasing_previous_time_s", None)
        row.setdefault("signal_first_nonincreasing_current_time_s", None)
        row.setdefault("signal_time_channel_columns", [])
        row.setdefault("signal_first_channel_time_mismatch_row_index", None)
        row.setdefault("signal_first_channel_time_mismatch_times_s", None)
        row.setdefault("signal_channel_time_mismatch_max_abs_s", 0.0)
        row.setdefault("signal_channel_time_mismatch_details", [])

    emg_outer_entries = _outer_subject_entries(emg_path)
    label_outer_entries = _outer_subject_entries(label_path)
    targets_by_subject: dict[int, set[int]] = {}
    for subject_id, trial_id in replacements:
        targets_by_subject.setdefault(subject_id, set()).add(trial_id)

    with zipfile.ZipFile(emg_path) as emg_outer, zipfile.ZipFile(label_path) as label_outer:
        for subject_id, trial_ids in sorted(targets_by_subject.items()):
            if subject_id not in emg_outer_entries or subject_id not in label_outer_entries:
                raise AlignmentAuditError(f"Subject {subject_id} is absent from one source archive")
            raw_nested = zipfile.ZipFile(io.BytesIO(emg_outer.read(emg_outer_entries[subject_id])))
            label_nested = zipfile.ZipFile(io.BytesIO(label_outer.read(label_outer_entries[subject_id])))
            try:
                raw_entries = _nested_trial_entries(raw_nested, subject_id)
                label_entries = _nested_trial_entries(label_nested, subject_id)
                for trial_id in sorted(trial_ids):
                    if trial_id not in raw_entries or trial_id not in label_entries:
                        raise AlignmentAuditError(
                            f"Flagged subject/trial pair {subject_id}/{trial_id} is missing"
                        )
                    updated = _pair_trial(
                        subject_id,
                        trial_id,
                        raw_entries[trial_id],
                        raw_nested,
                        label_entries[trial_id],
                        label_nested,
                    )
                    original = trial_rows[replacements[(subject_id, trial_id)]]
                    if updated["label_rows"] != original["label_rows"]:
                        raise AlignmentAuditError(
                            f"Label row count changed for subject/trial {subject_id}/{trial_id}"
                        )
                    trial_rows[replacements[(subject_id, trial_id)]] = updated
            finally:
                raw_nested.close()
                label_nested.close()

    # Label rows are small relative to the signal archive. Recompute supported
    # class totals for every trial using its audited valid signal time interval.
    row_by_key = {
        (int(row["subject_index"]), int(row["trial_index"])): row for row in trial_rows
    }
    with zipfile.ZipFile(label_path) as label_outer:
        for subject_id, label_info in sorted(label_outer_entries.items()):
            label_nested = zipfile.ZipFile(io.BytesIO(label_outer.read(label_info)))
            try:
                label_entries = _nested_trial_entries(label_nested, subject_id)
                for trial_id, label_entry in label_entries.items():
                    row = row_by_key.get((subject_id, trial_id))
                    if row is None:
                        continue
                    with label_nested.open(label_entry) as stream:
                        label_data = _read_label_csv(stream)
                    label_times = label_data.pop("_label_times")
                    label_values = label_data.pop("_label_values")
                    signal_start = row.get("signal_time_start_s")
                    signal_end = row.get("signal_time_end_s")
                    supported_labels = [
                        label_values[index]
                        for index, time_s in enumerate(label_times)
                        if signal_start is not None
                        and signal_end is not None
                        and signal_start <= time_s <= signal_end
                    ]
                    support_counts = Counter(supported_labels)
                    row["labels_within_signal_time_support"] = len(supported_labels)
                    row["label_class_counts_in_emg_time_support"] = {
                        str(key): support_counts[key] for key in (0, 1, 2)
                    }
                    usable = int(row["usable_label_rows"])
                    row["label_time_support_fraction"] = (
                        len(supported_labels) / usable if usable else 0.0
                    )
                    time_axes_match = (
                        row.get("signal_channel_time_mismatch_rows", 0) == 0
                        and row.get("signal_nonincreasing_intervals", 0) == 0
                    )
                    if row["label_time_support_fraction"] == 1.0 and usable and time_axes_match:
                        row["timestamp_pairing_status"] = "complete_timestamp_overlap"
                    elif supported_labels:
                        row["timestamp_pairing_status"] = (
                            "partial_timestamp_overlap_or_signal_time_issue"
                        )
                    else:
                        row["timestamp_pairing_status"] = "no_timestamp_overlap"
            finally:
                label_nested.close()

    summary = report["summary"]
    summary["timestamp_overlap_complete_trials"] = sum(
        row["timestamp_pairing_status"] == "complete_timestamp_overlap" for row in trial_rows
    )
    summary["matched_trial_count_with_any_timestamp_overlap"] = sum(
        int(row["labels_within_signal_time_support"]) > 0 for row in trial_rows
    )
    summary["labels_in_emg_time_support"] = sum(
        int(row["labels_within_signal_time_support"]) for row in trial_rows
    )
    summary["usable_label_rows_in_matched_trials"] = sum(
        int(row["usable_label_rows"]) for row in trial_rows
    )
    summary["label_class_counts_in_emg_time_support"] = {
        str(key): sum(row["label_class_counts_in_emg_time_support"][str(key)] for row in trial_rows)
        for key in (0, 1, 2)
    }
    summary["signal_trials_with_nonincreasing_timestamps"] = sum(
        int(row.get("signal_nonincreasing_intervals", 0)) > 0 for row in trial_rows
    )
    summary["signal_trials_with_channel_time_mismatch"] = sum(
        int(row.get("signal_channel_time_mismatch_rows", 0)) > 0 for row in trial_rows
    )
    summary["signal_channel_time_mismatch_rows_total"] = sum(
        int(row.get("signal_channel_time_mismatch_rows", 0)) for row in trial_rows
    )
    summary.pop("timestamp_repaired_trial_count", None)
    summary["timestamp_repair_method"] = (
        "Focused re-read of all trials flagged with nonincreasing signal timestamps or cross-channel time "
        "mismatches; alignment uses only the monotonic prefix before the first nonincreasing interval. "
        "No time offset is inferred."
    )
    summary["signal_time_anomaly_trials_reprocessed"] = len(replacements)
    report["schema_version"] = max(int(report.get("schema_version", 1)), 2)
    return report


def _validate_report_outputs(
    json_path: str | Path,
    csv_path: str | Path,
    protected_paths: Iterable[str | Path] = (),
) -> tuple[Path, Path]:
    json_target = Path(json_path).expanduser()
    csv_target = Path(csv_path).expanduser()
    try:
        ensure_outside_public_checkout((json_target, csv_target))
    except PublicCheckoutOutputError as error:
        raise AlignmentAuditError(str(error)) from error
    json_resolved, csv_resolved = json_target.resolve(), csv_target.resolve()
    if json_resolved == csv_resolved:
        raise AlignmentAuditError("JSON and CSV report outputs must be different files")
    protected = {Path(path).expanduser().resolve() for path in protected_paths}
    if {json_resolved, csv_resolved} & protected:
        raise AlignmentAuditError("report outputs must not overwrite an input archive or repair receipt")
    if json_target.exists() or csv_target.exists():
        raise AlignmentAuditError("Refusing to overwrite existing audit outputs")
    return json_target, csv_target


def write_report(
    report: dict[str, object],
    json_path: str | Path,
    csv_path: str | Path,
    *,
    protected_paths: Iterable[str | Path] = (),
) -> None:
    """Write a JSON receipt and a per-trial CSV without changing source archives."""
    json_target, csv_target = _validate_report_outputs(json_path, csv_path, protected_paths)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trials = report["trials"]
    fields: list[str] = []
    for row in trials:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in trials:
            serialized = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(
                {key: _spreadsheet_safe_cell(value) for key, value in serialized.items()}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emg_archive", type=Path)
    parser.add_argument("label_archive", type=Path)
    parser.add_argument("--expected-emg-md5", default=EXPECTED_EMG_MD5)
    parser.add_argument("--expected-label-md5", default=EXPECTED_LABEL_MD5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--repair-time-anomalies-from",
        "--repair-nonincreasing-from",
        dest="repair_time_anomalies_from",
        type=Path,
        help=(
            "Reprocess only signal trials with nonincreasing or cross-channel time anomalies from an "
            "earlier exhaustive JSON report"
        ),
    )
    args = parser.parse_args()
    protected_paths = [args.emg_archive, args.label_archive]
    if args.repair_time_anomalies_from:
        protected_paths.append(args.repair_time_anomalies_from)
    try:
        _validate_report_outputs(args.output_json, args.output_csv, protected_paths)
    except AlignmentAuditError as error:
        parser.error(str(error))
    if args.repair_time_anomalies_from:
        existing_report = json.loads(args.repair_time_anomalies_from.read_text(encoding="utf-8"))
        report = repair_nonmonotonic_trials(
            args.emg_archive,
            args.label_archive,
            existing_report,
            expected_emg_md5=args.expected_emg_md5,
            expected_label_md5=args.expected_label_md5,
        )
    else:
        report = audit_alignment(
            args.emg_archive,
            args.label_archive,
            expected_emg_md5=args.expected_emg_md5,
            expected_label_md5=args.expected_label_md5,
        )
    write_report(report, args.output_json, args.output_csv, protected_paths=protected_paths)
    summary = report["summary"]
    safe_summary = {
        "status": "complete",
        "emg_subject_count": summary["emg_subject_count"],
        "label_subject_count": summary["label_subject_count"],
        "matched_subject_count": summary["matched_subject_count"],
        "matched_trial_count": summary["matched_trial_count"],
        "pairing_complete": summary["pairing_complete"],
        "timestamp_overlap_complete_trials": summary["timestamp_overlap_complete_trials"],
    }
    print(json.dumps(safe_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
