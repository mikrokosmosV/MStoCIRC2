"""Workflow injection – replace the database.db-path inside a FragPipe .workflow file."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


def _replace_db_line(line: str, new_path: str) -> str:
    """If *line* starts with 'database.db-path=', replace its value with *new_path*."""
    if line.strip().startswith("database.db-path="):
        return f"database.db-path={new_path}\n"
    return line


def inject_database_path(
    workflow_template: str,
    database_fasta: str,
    output_dir: str,
) -> Path:
    """Parse *workflow_template* (.workflow file), replace the `database.db-path`
    line with the absolute path to *database_fasta*, and write the result to
    ``<output_dir>/run.workflow``.

    On Windows the path is escaped with double backslashes to conform to the
    FragPipe workflow format.

    Returns the absolute path of the generated `run.workflow` file.
    """
    template_path = Path(workflow_template)
    db_path = Path(database_fasta).resolve()

    if sys.platform == "win32":
        # FragPipe .workflow files require double backslashes on Windows
        db_path_str = str(db_path).replace("\\", "\\\\")
    else:
        db_path_str = db_path.as_posix()

    lines: List[str] = []
    with open(template_path, "r", encoding="utf-8") as fh:
        found = False
        for line in fh:
            if _replace_db_line(line, db_path_str) != line:
                found = True
                lines.append(_replace_db_line(line, db_path_str))
            else:
                lines.append(line)

    if not found:
        log.warning("No 'database.db-path=' line found in workflow template; appending it.")
        lines.append(f"database.db-path={db_path_str}\n")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_workflow = out_dir / "run.workflow"
    with open(out_workflow, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    log.info(f"Workflow injected → {out_workflow}")
    return out_workflow.resolve()