"""Keep participant-level Cerqueira products out of the public source checkout."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class PublicCheckoutOutputError(ValueError):
    """Raised when a private analysis product targets the public repository."""


def ensure_outside_public_checkout(
    paths: Iterable[str | Path],
    *,
    checkout_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Resolve paths and reject destinations beneath the repository root.

    The default root is the Git checkout containing ``src/cerqueira``. Tests
    and embedders may provide a root explicitly. Resolution follows existing
    symlink parents, so a path that escapes or enters through a symlink is
    compared by its resolved destination.
    """
    root = Path(checkout_root).expanduser().resolve() if checkout_root else Path(__file__).resolve().parents[2]
    resolved_paths = tuple(Path(path).expanduser().resolve() for path in paths)
    for path in resolved_paths:
        if path == root or root in path.parents:
            raise PublicCheckoutOutputError(
                "participant-level outputs must be stored outside the public checkout"
            )
    return resolved_paths
