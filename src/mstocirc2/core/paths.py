"""Path helpers used across pipeline stages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def coerce_path(value: str | Path | None) -> Path | None:
    """Convert CLI text input to a Path while preserving optional values."""
    if value in (None, "", "none"):
        return None
    return Path(value).expanduser()


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_empty_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    import shutil

    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    return directory


def make_timestamped_output_dir(prefix: str, parent: str | Path | None = None) -> Path:
    """Create a date-stamped output directory with collision-safe suffixes."""
    root = Path(parent).expanduser() if parent else Path.cwd()
    stamp = datetime.now().strftime("%y-%m-%d")
    base = root / f"{prefix}_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{prefix}_{stamp}.{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def make_default_module_output_dir(
    module_name: str,
    parent: str | Path | None = None,
) -> Path:
    """Create the default per-run output directory for a CLI module."""
    root = Path(parent).expanduser() if parent else Path.cwd()
    stamp = datetime.now().strftime("%y-%m-%d")
    counter = 1
    while True:
        candidate = root / f"MStoCIRC2_{module_name}_{stamp}.{counter}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            counter += 1
