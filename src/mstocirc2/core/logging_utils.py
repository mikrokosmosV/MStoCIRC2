"""Central logging configuration."""

from __future__ import annotations

import logging
import shutil
import sys


class TerminalProgressBar:
    """Minimal terminal progress bar that does not flood the log history."""

    def __init__(self, total: int, prefix: str, width: int = 28) -> None:
        self.total = max(int(total), 0)
        self.prefix = prefix
        self.width = max(int(width), 10)
        self._enabled = self.total > 0 and sys.stderr.isatty()
        self._closed = False

    def update(self, current: int) -> None:
        if not self._enabled or self._closed:
            return
        current = max(0, min(int(current), self.total))
        columns = shutil.get_terminal_size((100, 20)).columns
        bar_width = min(self.width, max(10, columns - len(self.prefix) - 28))
        filled = int(bar_width * current / self.total) if self.total else bar_width
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = 100.0 * current / self.total if self.total else 100.0
        sys.stderr.write(f"\r{self.prefix} [{bar}] {current}/{self.total} ({percent:5.1f}%)")
        sys.stderr.flush()

    def close(self) -> None:
        if not self._enabled or self._closed:
            return
        self._closed = True
        sys.stderr.write("\n")
        sys.stderr.flush()


class CompactCliFormatter(logging.Formatter):
    """Compact terminal formatter for end users."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"[ERROR] {message}"
        if record.levelno >= logging.WARNING:
            return f"[WARN] {message}"
        if record.levelno == logging.DEBUG:
            return f"[DEBUG] {record.name}: {message}"
        return message


def configure_logging(verbose: int = 0, quiet: bool = False) -> None:
    """Configure a single global logging policy for the CLI."""
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.INFO

    handler = logging.StreamHandler()
    handler.setFormatter(CompactCliFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
