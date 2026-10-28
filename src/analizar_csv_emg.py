"""Analyze one raw EMG CSV with the repository's descriptive pipeline.

This launcher does not load an ML model, count verified mechanical repetitions,
or predict physiological fatigue. It exports detected activity candidates only.
ZIP archives are not supported. Input and output data must remain outside the
public source checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from release_paths import ReleasePathError, validate_data_paths


MAX_CSV_BYTES = 512 * 1024 * 1024
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


class InputValidationError(ValueError):
    """Raised when a selected input is unsupported or fails validation."""


@dataclass(frozen=True)
class CsvRecording:
    recording_id: str
    participant_id: str | None
    side: str
    series: str | None
    status: str
    reason: str
    source_name: str = field(repr=False)


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


def discover_single_csv(csv_path: Path | str) -> CsvRecording:
    """Validate one caller-supplied EMG CSV without assigning a participant label."""
    path = Path(csv_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".csv" or zipfile.is_zipfile(path):
        raise InputValidationError("Seleccione un CSV sin comprimir; los archivos ZIP no se admiten.")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise InputValidationError("El CSV excede el límite de 512 MiB.")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = tuple(cell.strip() for cell in next(csv.reader(handle), []))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise InputValidationError("No se pudo leer el encabezado del CSV.") from error
    series, state = _series_from_stem(path.stem)
    if state != "ready":
        raise InputValidationError("El nombre del CSV debe identificar una única serie R1–R5.")
    if series in {"R0", "R0_1"}:
        raise InputValidationError("R0/R0_1 es preliminar; no se procesa como serie formal.")
    if header not in RAW_HEADERS:
        raise InputValidationError("CSV no coincide con Relative_Time_s,ADC_Value[,Voltage_mV].")
    return CsvRecording(
        recording_id="REC0001",
        participant_id=None,
        side=_side_from_path(path.name),
        series=series,
        status="ready",
        reason="CSV individual; no se infiere identidad ni grupo de participante.",
        source_name=path.name,
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
<p>La entrada se procesa como una grabación CSV individual; no se infiere identidad ni agrupamiento de participante. Mantenga este informe y el tablero en almacenamiento autorizado: contienen señales procesadas.</p>
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


def process_single_csv(csv_path: Path | str, output_base: Path | str, *, open_browser: bool = True) -> Path:
    """Run the descriptive processor on one R1-R5 CSV, without cohort inference."""
    csv_file = Path(csv_path).expanduser().resolve(strict=True)
    base = Path(output_base).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    try:
        validate_data_paths({"input_csv": csv_file, "output_dir": base}, repository)
    except (ReleasePathError, OSError, ValueError) as error:
        raise InputValidationError("El CSV y las salidas deben permanecer fuera del checkout público.") from error
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
        "pipeline_version": "public_F56",
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
        csv_name = filedialog.askopenfilename(
            title="Seleccione un CSV EMG individual R1–R5",
            filetypes=[("CSV EMG sin comprimir", "*.csv")],
        )
        if not csv_name:
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
        return Path(csv_name), Path(output_name)
    finally:
        root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Procesamiento descriptivo de un CSV EMG individual; no acepta ZIP ni predice fatiga o repeticiones."
    )
    parser.add_argument("--pick", action="store_true", help="abre selectores para un CSV individual y la carpeta de resultados")
    parser.add_argument("--csv", type=Path, help="CSV de una grabación formal R1-R5")
    parser.add_argument("--output-dir", type=Path, help="carpeta padre para resultados privados")
    parser.add_argument("--no-open", action="store_true", help="no abrir la página de resultados al terminar")
    args = parser.parse_args(argv)
    if args.pick:
        if args.csv is not None or args.output_dir is not None:
            parser.error("--pick no se combina con --csv/--output-dir")
        try:
            selected = _pick_paths()
        except RuntimeError as error:
            parser.error(str(error))
        if selected is None:
            print("Selección cancelada; no se modificaron datos.")
            return 0
        input_path, output_dir = selected
    else:
        if args.csv is None:
            parser.error("indique --csv RUTA o use --pick; ZIP no es compatible")
        input_path = args.csv
        output_dir = args.output_dir or default_output_dir()

    try:
        run_dir = process_single_csv(input_path, output_dir, open_browser=not args.no_open)
    except (InputValidationError, OSError, ValueError) as error:
        print(f"No se pudo procesar el CSV: {error}", file=sys.stderr)
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
