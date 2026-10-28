"""Normalize the supplied Pxx_setNN_side RAR naming and run the local demo.

The original RAR is read-only. Windows tar streams each validated CSV into a
temporary ZIP, mapping set01..set05 to R1..R5 and source participant codes to
P01..Pn. The existing descriptive pipeline then creates a local HTML report.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from analizar_zip_emg import RAW_HEADERS, default_output_dir, main as run_emg_demo


MEMBER_RE = re.compile(
    r"(?P<participant>P\d+)_set(?P<set>0[1-5])_(?P<side>dr|iz)\.csv\Z", re.IGNORECASE
)
MAX_CSV_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class RawArchiveError(ValueError):
    """The RAR structure or CSV input does not match the supported contract."""


def map_rar_members(member_names: list[str]) -> dict[str, str]:
    """Map a flat Pxx_set01-05_dr/iz archive to anonymized ZIP member names."""
    if not member_names:
        raise RawArchiveError("El archivo RAR no contiene miembros.")

    seen: set[str] = set()
    parsed: list[tuple[str, int, str, str]] = []
    for member in member_names:
        normalized = member.replace("\\", "/")
        windows_path = PureWindowsPath(member)
        if (
            not member
            or "/" in normalized
            or normalized.startswith("/")
            or PurePosixPath(normalized).is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or normalized in {".", ".."}
        ):
            raise RawArchiveError("El RAR incluye una ruta anidada o insegura; no se procesó.")

        key = normalized.casefold()
        if key in seen:
            raise RawArchiveError("El RAR incluye un miembro duplicado.")
        seen.add(key)

        match = MEMBER_RE.fullmatch(normalized)
        if not match:
            raise RawArchiveError("Nombre no compatible; se esperaba Pxx_set01-05_dr/iz.csv.")
        participant = match.group("participant").upper()
        series_number = int(match.group("set")[-2:])
        side = match.group("side").lower()
        parsed.append((normalized, participant, series_number, side))

    participants = sorted({participant for _, participant, _, _ in parsed})
    expected = {(series, side) for series in range(1, 6) for side in ("dr", "iz")}
    mapping: dict[str, str] = {}
    for person_index, participant in enumerate(participants, start=1):
        observed = {(series, side) for _, person, series, side in parsed if person == participant}
        if observed != expected:
            raise RawArchiveError("La matriz de cinco series y dos lados está incompleta.")
        for member, person, series, side in parsed:
            if person == participant:
                mapping[member] = f"P{person_index:02d}/{side}_R{series}.csv"
    return mapping


def _validate_csv_header(data: bytes, member_name: str) -> None:
    try:
        header_text = data.splitlines()[0].decode("utf-8-sig")
        header = tuple(cell.strip() for cell in next(csv.reader(io.StringIO(header_text))))
    except (IndexError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise RawArchiveError("Un miembro CSV está vacío o tiene una cabecera ilegible.") from error
    if header not in RAW_HEADERS:
        raise RawArchiveError("Un miembro CSV no usa el esquema EMG compatible.")


def create_normalized_zip(
    rar_path: Path | str,
    zip_path: Path | str,
    *,
    tar_executable: str = "tar",
) -> tuple[int, int]:
    """Read a RAR via tar without extracting member-controlled paths."""
    archive = Path(rar_path).expanduser().resolve(strict=True)
    if not archive.is_file() or archive.suffix.lower() != ".rar":
        raise RawArchiveError("Seleccione un archivo .rar existente.")
    tar = shutil.which(tar_executable)
    if not tar:
        raise RawArchiveError("No se encontró tar.exe; se requiere Windows con soporte RAR.")

    try:
        listing = subprocess.run(
            [tar, "-tf", str(archive)], capture_output=True, check=True, timeout=60
        ).stdout.decode("utf-8-sig").splitlines()
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        raise RawArchiveError("Windows no pudo listar este RAR; el original no se modificó.") from error

    mapping = map_rar_members(listing)
    output_zip = Path(zip_path).expanduser().resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    try:
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
            for source_member, normalized_member in mapping.items():
                try:
                    extracted = subprocess.run(
                        [tar, "-xOf", str(archive), source_member],
                        capture_output=True,
                        check=True,
                        timeout=120,
                    ).stdout
                except (OSError, subprocess.SubprocessError) as error:
                    raise RawArchiveError("No se pudo leer un miembro CSV; el RAR original sigue intacto.") from error
                if len(extracted) > MAX_CSV_BYTES:
                    raise RawArchiveError("Un CSV supera el límite de 512 MiB.")
                total_bytes += len(extracted)
                if total_bytes > MAX_TOTAL_BYTES:
                    raise RawArchiveError("Los CSV superan el límite total de 2 GiB.")
                _validate_csv_header(extracted, source_member)
                target.writestr(normalized_member, extracted)
    except Exception:
        output_zip.unlink(missing_ok=True)
        raise
    return len(mapping), total_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepara el RAR Pxx_set01-05_dr/iz y abre el informe EMG descriptivo local."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / "Downloads" / "raw.rar",
        help="RAR de entrada (por defecto: Downloads/raw.rar)",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--no-open", action="store_true", help="no abrir index.html automáticamente")
    args = parser.parse_args(argv)

    try:
        with tempfile.TemporaryDirectory(prefix="emg_raw_rar_demo_") as temporary:
            normalized_zip = Path(temporary) / "normalized_emg.zip"
            file_count, total_bytes = create_normalized_zip(args.archive, normalized_zip)
            print(
                f"Preparación segura completada: {file_count} CSV compatibles "
                f"({total_bytes / (1024 * 1024):.1f} MiB); el RAR original no se modificó."
            )
            command = ["--zip", str(normalized_zip), "--output-dir", str(args.output_dir)]
            if args.no_open:
                command.append("--no-open")
            return run_emg_demo(command)
    except (OSError, RawArchiveError, zipfile.BadZipFile) as error:
        print(f"No se inició el demo: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
