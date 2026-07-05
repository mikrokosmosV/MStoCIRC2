"""Engine executor for headless FragPipe runs."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def run_fragpipe(
    fragpipe_bin: str,
    run_workflow: str,
    manifest: str,
    workdir: str,
    tools_folder: str,
    diann_bin: str,
    python_bin: str,
) -> int:
    """Execute FragPipe in headless mode and return its exit code."""
    cmd: list[str]
    fragpipe_path = Path(fragpipe_bin).resolve()
    workflow_path = Path(run_workflow).resolve()
    manifest_path = Path(manifest).resolve()
    workdir_path = Path(workdir).resolve()
    tools_path = Path(tools_folder).resolve()
    diann_path = Path(diann_bin).resolve()
    python_path = Path(python_bin).resolve()
    log_path = workdir_path / "fragpipe.run.log"

    if sys.platform == "win32" and str(fragpipe_path).lower().endswith((".bat", ".cmd")):
        cmd = ["cmd.exe", "/c", str(fragpipe_path)]
    else:
        cmd = [str(fragpipe_path)]

    cmd += [
        "--headless",
        "--workflow", str(workflow_path),
        "--manifest", str(manifest_path),
        "--workdir", str(workdir_path),
        "--config-tools-folder", str(tools_path),
        "--config-diann", str(diann_path),
        "--config-python", str(python_path),
    ]

    log.info("Launching FragPipe from %s", workdir_path)
    log.info("FragPipe live log: %s", log_path)

    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(f"Command: {' '.join(cmd)}\n")
        handle.write(f"Working directory: {workdir_path}\n\n")
        handle.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            text = line.rstrip()
            if text:
                log.info("[FragPipe] %s", text)

        return_code = proc.wait()

    if return_code != 0:
        log.error("FragPipe exited with code %s. See %s for details.", return_code, log_path)
    else:
        log.info("FragPipe finished successfully. See %s for the full run log.", log_path)
    return return_code
