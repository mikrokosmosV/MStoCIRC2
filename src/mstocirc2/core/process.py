"""Wrappers for external command discovery and execution."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from .exceptions import DependencyError, ExternalCommandError


def require_command(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved is None:
        raise DependencyError(f"Required executable not found on PATH: '{executable}'.")
    return resolved


def run_external_command(
    command: Sequence[str],
    cwd: str | Path | None = None,
    timeout: int | None = None,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        timeout=timeout,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "External command failed."
        raise ExternalCommandError(message)
    return result
