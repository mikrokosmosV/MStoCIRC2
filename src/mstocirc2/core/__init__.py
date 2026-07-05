"""Shared infrastructure for the MStoCIRC2 CLI."""

from .exceptions import CLIUsageError, DependencyError, ExternalCommandError, MStoCIRC2Error
from .logging_utils import TerminalProgressBar, configure_logging
from .paths import coerce_path, ensure_directory, ensure_empty_directory, make_timestamped_output_dir
from .process import require_command, run_external_command

__all__ = [
    "CLIUsageError",
    "DependencyError",
    "ExternalCommandError",
    "MStoCIRC2Error",
    "TerminalProgressBar",
    "configure_logging",
    "coerce_path",
    "ensure_directory",
    "ensure_empty_directory",
    "make_timestamped_output_dir",
    "require_command",
    "run_external_command",
]
