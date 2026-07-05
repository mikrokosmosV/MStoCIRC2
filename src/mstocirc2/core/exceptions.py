"""Project-level exception types."""


class MStoCIRC2Error(Exception):
    """Base exception for all user-facing pipeline failures."""


class CLIUsageError(MStoCIRC2Error):
    """Raised when CLI inputs are incomplete or inconsistent."""


class DependencyError(MStoCIRC2Error):
    """Raised when an external dependency is missing."""


class ExternalCommandError(MStoCIRC2Error):
    """Raised when a delegated external command fails."""
