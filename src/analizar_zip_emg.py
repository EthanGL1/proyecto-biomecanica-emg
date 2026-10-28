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
    body_rows = []
    for row in rows:
        dashboard = row.get("output_folder", "")
        link = ""
        if row.get("status") == "processed" and dashboard:
            link = f'<a href="{html.escape(str(dashboard), quote=True)}/dashboard.html">Abrir señal y análisis</a>'
        body_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(key, '')))}</td>"
                for key in ("recording_id", "participant_pseudonym", "side", "series", "status", "candidate_event_count")
            )
            + f"<td>{link}</td></tr>"
        )
    html_text = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análisis EMG por grabación</title><style>
body{{font:16px/1.5 Segoe UI,Arial,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#17202a}}
.warning{{border-left:5px solid #d99b19;background:#fff8e6;padding:1rem}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}th,td{{border:1px solid #ccd2d8;padding:.55rem;text-align:left}}th{{background:#eef3f6}}
</style></head><body><h1>Análisis descriptivo de señales EMG</h1>
<div class="warning"><strong>Alcance:</strong> se muestran características de señal y eventos de activación candidatos.
No se cargó un modelo; no se verificaron repeticiones mecánicas ni se predijo un nivel de fatiga.</div>
<p>CSV reconocidos: {summary['csv_count']}; procesados: {summary['processed_count']}; preliminares/no evaluables: {summary['skipped_count']}; eventos candidatos: {summary['candidate_event_count']}.</p>
<table><thead><tr><th>Registro</th><th>Participante</th><th>Lado</th><th>Serie</th><th>Estado</th><th>Eventos candidatos</th><th>Vista</th></tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
<p>La salida usa pseudónimos cuando el ZIP distingue carpetas de registro. Mantenga este informe y los tableros en almacenamiento autorizado: contienen señales procesadas.</p>
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
        "pipeline_version": "public_F56",
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
            title="Seleccione el ZIP de grabaciones EMG",
            filetypes=[("Archivos ZIP", "*.zip")],
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
        description="Procesamiento descriptivo de CSV EMG dentro de ZIP; no ejecuta un modelo de fatiga ni valida repeticiones."
    )
    parser.add_argument("--pick", action="store_true", help="abre selectores nativos para ZIP y carpeta de resultados")
    parser.add_argument("--zip", type=Path, help="ZIP de CSV EMG")
    parser.add_argument("--output-dir", type=Path, help="carpeta padre para resultados privados")
    parser.add_argument("--no-open", action="store_true", help="no abrir la página de resultados al terminar")
    args = parser.parse_args(argv)
    if args.pick:
        if args.zip is not None or args.output_dir is not None:
            parser.error("--pick no se combina con --zip/--output-dir")
        try:
            selected = _pick_paths()
        except RuntimeError as error:
            parser.error(str(error))
        if selected is None:
            print("Selección cancelada; no se modificaron datos.")
            return 0
        zip_path, output_dir = selected
    else:
        if args.zip is None:
            parser.error("indique --zip RUTA o use --pick")
        zip_path = args.zip
        if args.output_dir is None:
            selected = _pick_paths()
            if selected is None:
                print("Selección cancelada; no se modificaron datos.")
                return 0
            _, output_dir = selected
        else:
            output_dir = args.output_dir

    try:
        run_dir = process_archive(zip_path, output_dir, open_browser=not args.no_open)
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
