"""Validate that CLI data and output paths stay outside the public checkout."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


class ReleasePathError(ValueError):
    """Raised when a data or output path points into the source checkout."""


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
