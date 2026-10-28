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
import subprocess
import sys
import tempfile
import time
import webbrowser
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

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
        if segment_path.is_file():
            with segment_path.open("r", encoding="utf-8", newline="") as handle:
                for segment in csv.DictReader(handle):
                    if segment.get("status") == "processed":
                        start = number(segment.get("start_time_s"))
                        end = number(segment.get("end_time_s"), start)
                        continuous_s += max(0.0, end - start)
                        record_end = max(record_end, end)
        total_continuous_s += continuous_s
        max_time = max(max_time, record_end)
        record_rows.append({
            **row,
            "events": record_events,
            "lane_count": max(1, len(lane_ends)),
            "time_end_s": record_end,
            "continuous_duration_s": continuous_s,
        })
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
<div class="fatigue"><strong>Fatigue prediction: unavailable.</strong> This run does not contain a validated quadriceps-fatigue model or independent fatigue labels. The system does not mark fatigue times or levels.</div>
<div class="status"><div><strong>{summary.get('csv_count', 0)}</strong><span>CSV files recognized</span></div><div><strong>{summary.get('processed_count', 0)}</strong><span>recordings processed</span></div><div><strong>{summary.get('candidate_event_count', 0)}</strong><span>activity candidates</span></div><div><strong>{total_continuous_s/60:.1f} min</strong><span>summed continuous recording duration; files remain separate</span></div><div><strong>No</strong><span>model loaded or fatigue scores emitted</span></div></div>
<section class="section"><h2>All event times across all recordings</h2><figure>{timeline_svg}</figure><p class="note">Each row uses the recording's own time from its start. If candidate intervals overlap within a recording, the chart stacks them in separate lanes without merging or dropping events. Bars mark threshold-derived intervals only.</p></section>
<section class="section"><h2>Candidate intervals by series and side</h2><figure>{count_svg}</figure><p class="note">Counts include all exported activity candidates; they are not repetition counts.</p></section>
<section class="section"><h2>Recording inventory</h2><div class="table-wrap"><table><thead><tr><th>Recording</th><th>Group</th><th>Side</th><th>Series</th><th>Status</th><th>Activity candidates</th><th>Summed continuous time (s)</th></tr></thead><tbody>{''.join(manifest_rows)}</tbody></table></div></section>
<section class="section"><h2>Complete candidate event table</h2><label for="event-search">Filter rows by recording, group, side, series, time, or quality</label><br><input id="event-search" type="search" placeholder="Search this table"><div class="table-wrap"><table id="event-table"><thead><tr><th>Recording</th><th>Group</th><th>Side</th><th>Series</th><th>Candidate</th><th>Start-end (s)</th><th>Duration (s)</th><th>Envelope threshold (ADC)</th><th>Mean RMS (ADC)</th><th>Peak RMS (ADC)</th><th>Quality status</th></tr></thead><tbody>{''.join(event_table_rows)}</tbody></table></div></section>
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
