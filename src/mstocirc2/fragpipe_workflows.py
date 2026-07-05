"""Default MStoCIRC2 FragPipe workflow templates."""

from __future__ import annotations

from pathlib import Path

from .core import CLIUsageError

DEFAULT_WORKFLOW_DIR = Path(__file__).resolve().parent / "spectral_searching" / "workflows"
DEFAULT_NON_QUANT_WORKFLOW = DEFAULT_WORKFLOW_DIR / "MStoCIRC2_non_quant.workflow"
DEFAULT_QUANT_WORKFLOW = DEFAULT_WORKFLOW_DIR / "MStoCIRC2_quant.workflow"
DEFAULT_DIA_WORKFLOW = DEFAULT_WORKFLOW_DIR / "MStoCIRC2_DIA.workflow"
DEFAULT_WORKFLOWS = (
    DEFAULT_NON_QUANT_WORKFLOW,
    DEFAULT_QUANT_WORKFLOW,
    DEFAULT_DIA_WORKFLOW,
)


def ensure_default_workflows_available() -> None:
    missing = [str(path) for path in DEFAULT_WORKFLOWS if not path.is_file()]
    if missing:
        raise CLIUsageError(
            "MStoCIRC2 default FragPipe workflow templates are missing. "
            "Expected files: "
            + ", ".join(missing)
        )


def default_workflow_for_command(data_type: str, command_name: str) -> Path:
    ensure_default_workflows_available()
    if data_type == "DIA":
        return DEFAULT_DIA_WORKFLOW
    normalized = command_name.strip().lower()
    if normalized == "quant":
        return DEFAULT_QUANT_WORKFLOW
    return DEFAULT_NON_QUANT_WORKFLOW
