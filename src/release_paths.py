"""Validate that CLI data and output paths stay outside the public checkout."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Mapping


class ReleasePathError(ValueError):
    """Raised when a data or output path points into the source checkout."""



_WINDOWS_RESERVED_COMPONENTS = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def validate_path_component(value: object, *, field_name: str = "path component") -> str:
    """Validate an untrusted identifier before using it in a filename.

    Safe pseudonyms such as P01 and recording keys such as R0_1 are preserved
    exactly. Separators, drive syntax, control characters, Windows device
    names, and ambiguous leading/trailing whitespace or dots are rejected.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        raise ReleasePathError(f"{field_name} must be a non-empty filename component")
    forbidden = set('/\\\\<>:"|?*')
    if (
        value in {".", ".."}
        or value.endswith(".")
        or any(ord(character) < 32 or character in forbidden for character in value)
    ):
        raise ReleasePathError(f"{field_name} contains unsafe filename syntax")
    device_name = value.split(".", 1)[0].upper()
    if device_name in _WINDOWS_RESERVED_COMPONENTS:
        raise ReleasePathError(f"{field_name} is a reserved filename component")
    return value


def resolve_under_root(
    root: Path,
    relative_path: str | Path,
    *,
    field_name: str = "path",
    must_exist: bool = True,
) -> Path:
    """Resolve an untrusted relative file path and require a strict root child.

    Both POSIX and Windows absolute/drive forms are rejected on every platform.
    The final containment check uses resolved paths, so existing symlinks that
    point outside the configured root are rejected as well.
    """
    try:
        raw_path = Path(relative_path)
        raw_text = str(relative_path)
    except (TypeError, ValueError, OSError) as error:
        raise ReleasePathError(f"{field_name} is not a valid relative path") from error
    windows_path = PureWindowsPath(raw_text)
    if (
        not raw_text
        or "\x00" in raw_text
        or raw_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ReleasePathError(f"{field_name} must be a relative path beneath its root")
    parts = raw_text.replace("\\", "/").split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ReleasePathError(f"{field_name} contains an unsafe relative path component")

    try:
        resolved_root = Path(root).resolve(strict=True)
        if not resolved_root.is_dir():
            raise ReleasePathError(f"{field_name} root must be an existing directory")
        candidate = resolved_root.joinpath(*parts).resolve(strict=False)
    except ReleasePathError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise ReleasePathError(f"{field_name} cannot be safely resolved beneath its root") from error

    try:
        relative = candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ReleasePathError(f"{field_name} escapes its configured root") from error
    if not relative.parts:
        raise ReleasePathError(f"{field_name} must name a file below its configured root")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"{field_name} does not exist beneath its configured root")
    return candidate


def resolve_recorded_path(
    root: Path,
    recorded_path: str | Path,
    *,
    field_name: str = "recorded path",
    must_exist: bool = True,
) -> Path:
    """Resolve a path recorded in a receipt while confining it to ``root``.

    Existing receipts store absolute output paths, while portable examples may
    use relative paths. Both forms are accepted only when their resolved file
    is a strict descendant of the explicitly configured root. Symlinks are
    resolved before the containment check.
    """
    try:
        raw_path = Path(recorded_path)
        raw_text = str(recorded_path)
        windows_path = PureWindowsPath(raw_text)
        resolved_root = Path(root).resolve(strict=True)
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        raise ReleasePathError(f"{field_name} cannot be safely resolved beneath its root") from error
    if not raw_text or "\x00" in raw_text:
        raise ReleasePathError(f"{field_name} must name a file beneath its root")
    if not resolved_root.is_dir():
        raise ReleasePathError(f"{field_name} root must be an existing directory")

    if raw_path.is_absolute() or windows_path.is_absolute():
        try:
            candidate = raw_path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise ReleasePathError(f"{field_name} cannot be safely resolved beneath its root") from error
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ReleasePathError(f"{field_name} escapes its configured root") from error
        if not relative.parts:
            raise ReleasePathError(f"{field_name} must name a file below its configured root")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"{field_name} does not exist beneath its configured root")
        return candidate

    if windows_path.drive:
        raise ReleasePathError(f"{field_name} must be a relative path beneath its root")
    return resolve_under_root(
        resolved_root,
        raw_text,
        field_name=field_name,
        must_exist=must_exist,
    )


def resolve_recorded_npz_path(
    root: Path,
    recorded_path: str | Path,
    *,
    field_name: str = "recorded output_path",
) -> Path:
    """Resolve a receipt archive as an existing direct-child NPZ under root."""
    resolved_root = Path(root).resolve(strict=True)
    candidate = resolve_recorded_path(
        resolved_root,
        recorded_path,
        field_name=field_name,
        must_exist=True,
    )
    if candidate.parent != resolved_root or candidate.suffix.lower() != ".npz" or not candidate.is_file():
        raise ReleasePathError(f"{field_name} must name a direct-child NPZ archive beneath its root")
    return candidate


def validate_data_paths(
    values: Mapping[str, object], checkout_root: Path, cwd: Path | None = None
) -> dict[str, Path]:
    """Resolve path-like CLI values and reject paths inside ``checkout_root``.

    ``project_root`` is excluded because some analysis utilities use it only
    to locate code or verify project-relative reports. Input and output paths
    must be explicitly supplied outside the repository so private data and
    generated results cannot be committed by accident.
    """
    checkout = Path(checkout_root).resolve()
    base = Path(cwd or Path.cwd()).resolve()
    resolved: dict[str, Path] = {}
    for name, value in values.items():
        if name == "project_root" or value is None:
            continue
        try:
            path = Path(value)  # type: ignore[arg-type]
        except TypeError:
            continue
        absolute = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if absolute == checkout or checkout in absolute.parents:
            raise ReleasePathError(
                f"{name} must resolve outside the public source checkout"
            )
        resolved[name] = absolute
    return resolved
