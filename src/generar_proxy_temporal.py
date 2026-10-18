"""Create weak within-recording temporal labels and subject-wise LOSO folds.

The labels are temporal-position proxies for R1-R5 recordings declared by the
user as series taken to technical failure. They are not repetition counts,
active-time thirds, observed failure events, or fatigue ground truth.
Record-level outputs contain pseudonymous participant IDs and must remain in
restricted local storage outside synchronized project folders.
"""
from __future__ import annotations

from release_paths import validate_data_paths

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


LABELS = ("temprana", "transicion", "tardia")
STANDARD_RECORDS = {"R1", "R2", "R3", "R4", "R5"}
LABEL_VERSION = "proxy_temporal_registro_v1"
SEED_DEFAULT = 20261010


def split_recording_thirds(start: float, end: float) -> list[tuple[str, float, float]]:
    """Return three contiguous thirds of the observed recording-time span."""
    start = float(start)
    end = float(end)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Recording start/end must be finite and end must exceed start.")
    duration = end - start
    first = start + duration / 3.0
    second = start + 2.0 * duration / 3.0
    return [
        (LABELS[0], start, first),
        (LABELS[1], first, second),
        (LABELS[2], second, end),
    ]


def phase_for_time(time_s: float, start: float, end: float) -> str | None:
    """Assign half-open thirds; include the final observed time in the last third."""
    time_s = float(time_s)
    if not math.isfinite(time_s):
        raise ValueError("Time must be finite.")
    thirds = split_recording_thirds(start, end)
    if time_s < thirds[0][1] or time_s > thirds[-1][2]:
        return None
    for label, _, boundary in thirds[:2]:
        if time_s < boundary:
            return label
    return LABELS[2]


def build_nested_loso_rows(
    participant_ids: list[str], seed: int = SEED_DEFAULT, n_inner: int = 4
) -> list[dict[str, str]]:
    """Build participant-level outer LOSO and grouped inner validation rows."""
    ids = [str(value).strip() for value in participant_ids]
    if any(not value for value in ids):
        raise ValueError("Participant IDs must be non-empty.")
    if len(ids) != len(set(ids)):
        raise ValueError("Participant IDs must be unique.")
    if len(ids) < 3:
        raise ValueError("At least three participants are required for nested LOSO.")
    if not isinstance(n_inner, int) or n_inner < 2 or n_inner > len(ids) - 1:
        raise ValueError("Inner folds must be between 2 and the outer development count.")

    order = sorted(ids)
    random.Random(seed).shuffle(order)
    rows: list[dict[str, str]] = []
    for outer_index, test_id in enumerate(order, start=1):
        outer_fold = f"outer_{outer_index:02d}"
        development = [pid for pid in sorted(ids) if pid != test_id]
        for pid in sorted(ids):
            rows.append(
                {
                    "level": "outer",
                    "outer_fold": outer_fold,
                    "inner_fold": "",
                    "participant_id": pid,
                    "role": "test" if pid == test_id else "train",
                    "seed": str(seed),
                }
            )

        inner_order = list(development)
        random.Random(seed + outer_index).shuffle(inner_order)
        validation_groups = [inner_order[i::n_inner] for i in range(n_inner)]
        for inner_index, validation in enumerate(validation_groups, start=1):
            inner_fold = f"inner_{inner_index:02d}"
            validation_set = set(validation)
            for pid in sorted(development):
                rows.append(
                    {
                        "level": "inner",
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "participant_id": pid,
                        "role": "validation" if pid in validation_set else "train",
                        "seed": str(seed),
                    }
                )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_time_bounds(signal_path: Path) -> tuple[float, float, int]:
    start = None
    previous = None
    count = 0
    with signal_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Relative_Time_s", "ADC_Value", "Voltage_mV"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Unexpected CSV columns in pseudonymized input: {signal_path.name}")
        for row in reader:
            time_s = float(row["Relative_Time_s"])
            if not math.isfinite(time_s):
                raise ValueError(f"Non-finite time in pseudonymized input: {signal_path.name}")
            if previous is not None and time_s <= previous:
                raise ValueError(f"Non-increasing time in pseudonymized input: {signal_path.name}")
            if start is None:
                start = time_s
            previous = time_s
            count += 1
    if start is None or previous is None or count < 2 or previous <= start:
        raise ValueError(f"Insufficient time samples in pseudonymized input: {signal_path.name}")
    return start, previous, count


def _count_samples_by_phase(signal_path: Path, start: float, end: float) -> Counter:
    counts: Counter = Counter()
    with signal_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = phase_for_time(float(row["Relative_Time_s"]), start, end)
            if label is not None:
                counts[label] += 1
    return counts


def generate(manifest_path: Path, signals_root: Path, output_dir: Path, seed: int, n_inner: int) -> dict:
    manifest_path = manifest_path.resolve()
    signals_root = signals_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; refusing to overwrite it.")
    if not manifest_path.is_file() or not signals_root.is_dir():
        raise FileNotFoundError("Manifest or pseudonymized signal root is unavailable.")

    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest_reader = csv.DictReader(stream)
        manifest_rows = list(manifest_reader)
        manifest_fields = set(manifest_reader.fieldnames or [])
    required_fields = {"participant_id", "recording_id", "leg", "set_index", "relative_path"}
    if not required_fields.issubset(manifest_fields):
        raise ValueError("Pseudonymized manifest is missing required fields.")

    eligible = [row for row in manifest_rows if row["recording_id"] in STANDARD_RECORDS]
    if not eligible:
        raise ValueError("No R1-R5 recordings were found in the pseudonymized manifest.")
    if any(not row["participant_id"] for row in eligible):
        raise ValueError("An eligible record has no participant group ID.")
    identities = [
        (row["participant_id"], row["recording_id"], row["leg"])
        for row in eligible
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate participant/recording/leg rows in the manifest.")

    labels_rows: list[dict[str, str]] = []
    sample_totals = Counter()
    records_seen = set()
    participant_ids = sorted({row["participant_id"] for row in eligible})
    for record in eligible:
        relative = Path(*record["relative_path"].replace("\\", "/").split("/"))
        signal_path = signals_root / relative
        if not signal_path.is_file():
            raise FileNotFoundError("A pseudonymized manifest path does not resolve.")
        start, end, sample_count = _read_time_bounds(signal_path)
        per_phase = _count_samples_by_phase(signal_path, start, end)
        thirds = split_recording_thirds(start, end)
        if any(per_phase[label] == 0 for label in LABELS):
            raise ValueError("A recording has an empty temporal third; no label was written.")
        sample_totals.update(per_phase)
        records_seen.add((record["participant_id"], record["recording_id"], record["leg"]))
        for label, t_start, t_end in thirds:
            labels_rows.append(
                {
                    "participant_id": record["participant_id"],
                    "recording_id": record["recording_id"],
                    "leg": record["leg"],
                    "set_index": record["set_index"],
                    "t_start_s": f"{t_start:.9f}",
                    "t_end_s": f"{t_end:.9f}",
                    "duration_s": f"{(t_end - t_start):.9f}",
                    "n_samples_in_segment": str(per_phase[label]),
                    "label": label,
                    "label_version": LABEL_VERSION,
                    "label_kind": "temporal_recording_position_proxy_not_fatigue_ground_truth",
                    "label_source": "user_declared_R1_R5_series_to_technical_failure_plus_Relative_Time_s",
                    "label_confidence": "low_proxy",
                    "observed_repetition_count": "unknown",
                    "active_time_verified": "false",
                    "interval_rule": "[start,end); final segment includes final observed sample",
                }
            )

    partition_rows = build_nested_loso_rows(participant_ids, seed=seed, n_inner=n_inner)
    output_dir.mkdir(parents=True, exist_ok=False)
    label_path = output_dir / "labels_proxy_temporal.csv"
    partition_path = output_dir / "partitions_LOSO_nested.csv"
    with label_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels_rows[0]))
        writer.writeheader()
        writer.writerows(labels_rows)
    with partition_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(partition_rows[0]))
        writer.writeheader()
        writer.writerows(partition_rows)

    metadata = {
        "label_version": LABEL_VERSION,
        "label_scope": "R1-R5 only; R0/R0_1 remain QC-only because their role is preliminary/noisy and not resolved per file",
        "label_interpretation": "within-recording temporal position proxy; not repetition thirds, active-time thirds, observed failure, or fatigue ground truth",
        "pause_handling": "all observed recording time is divided into thirds; within-series pauses are unknown and remain included",
        "excluded_predictors": ["participant_id", "recording_id", "leg", "set_index", "absolute_or_relative_elapsed_time", "repetition_count"],
        "outer_evaluation": "leave-one-participant-out (LOSO)",
        "inner_evaluation": f"{n_inner}-fold grouped validation over outer development participants",
        "seed": seed,
        "participant_count": len(participant_ids),
        "recording_count": len(records_seen),
        "label_row_count": len(labels_rows),
        "partition_row_count": len(partition_rows),
        "samples_by_label": {label: sample_totals[label] for label in LABELS},
        "manifest_sha256": sha256(manifest_path),
        "signal_values_read": False,
        "source_paths_and_identity_key_exported": False,
    }
    (output_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--signals-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--inner-folds", type=int, default=4)
    args = parser.parse_args()
    validate_data_paths(vars(args), Path(__file__).resolve().parents[1])
    print(
        json.dumps(
            generate(args.manifest, args.signals_root, args.output_dir, args.seed, args.inner_folds),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
