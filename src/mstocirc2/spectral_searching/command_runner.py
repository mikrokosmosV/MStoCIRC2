"""Command runner for the `search` subcommand."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..cli_ui import CLIHelpFormatter, branded_description, help_block, join_blocks
from ..core import CLIUsageError, make_default_module_output_dir, require_command
from ..fragpipe_workflows import default_workflow_for_command
from .fasta_assembler import assemble_database
from .engine_executor import run_fragpipe
from .workflow_injector import inject_database_path

log = logging.getLogger("spectral_searching")

_DIA_REQUIRED_PYTHON_PACKAGES = ("fragpipe-speclib", "easypqp", "lxml")
_FRAGPIPE_PYTHON_ENV = "MSTOCIRC2_FRAGPIPE_PYTHON"


@dataclass(frozen=True)
class SearchRuntime:
    data_type: str
    python_bin: str
    tools_folder: Path
    diann_bin: Path
    fragpipe_bin: Path


def _walk_for_diann_executable(root: Path) -> Path | None:
    names = ("diann.exe", "DIA-NN.exe") if platform.system() == "Windows" else ("diann", "diann-*", "DIA-NN*")
    for pattern in names:
        for candidate in root.rglob(pattern):
            if candidate.is_file():
                return candidate
    return None


def _detect_manifest_data_type(manifest_path: str) -> str:
    values: set[str] = set()
    with Path(manifest_path).open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            cells = [cell.strip() for cell in row]
            if len(cells) >= 4 and cells[3]:
                values.add(cells[3].upper())
    if not values:
        raise CLIUsageError(
            "Unable to infer data type from manifest. Ensure column 4 contains DDA or DIA."
        )
    if len(values) > 1:
        raise CLIUsageError(
            f"Manifest mixes data types {sorted(values)}. Use a single acquisition mode per run."
        )
    data_type = values.pop()
    if data_type not in {"DDA", "DIA"}:
        raise CLIUsageError(
            f"Unsupported manifest data type '{data_type}'. Expected 'DDA' or 'DIA'."
        )
    return data_type


def _normalize_manifest_for_fragpipe(manifest_path: str, output_dir: str | Path) -> Path:
    """Write a FragPipe-friendly manifest without altering the user-provided file.

    FragPipe headless mode expects four tab-separated columns:
    LC-MS file, experiment, bioreplicate, and data type.

    The integrated pipeline uses column 3 as a user-facing grouping/condition field.
    For DDA runs this is often blank, but for DIA runs users may provide labels such as
    ``P2``/``P21`` instead of numeric bioreplicates. FragPipe can silently skip all
    inputs in that case. To keep the original manifest semantics for downstream DEA
    while still feeding FragPipe a valid file, normalize the manifest into the current
    search output directory.
    """

    source = Path(manifest_path)
    normalized_path = Path(output_dir) / "fragpipe_input.fp-manifest"

    rows: list[list[str]] = []
    with source.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            rows.append([cell.strip() for cell in row])

    if not rows:
        raise CLIUsageError(f"Manifest file is empty: '{manifest_path}'.")

    replicate_counters: defaultdict[str, int] = defaultdict(int)
    normalized_rows: list[list[str]] = []

    for row in rows:
        if len(row) < 2 or not row[0] or not row[1]:
            raise CLIUsageError(
                "Each manifest row must contain at least the LC-MS path and experiment/sample label. "
                f"Offending row: {row!r}"
            )

        if len(row) >= 4 and row[3]:
            data_type = row[3].upper()
            third_col = row[2] if len(row) >= 3 else ""

            if not third_col:
                experiment = row[1]
                bioreplicate = ""
            elif third_col.isdigit():
                experiment = row[1]
                bioreplicate = third_col
            else:
                experiment = third_col
                replicate_counters[experiment] += 1
                bioreplicate = str(replicate_counters[experiment])
        elif len(row) >= 3 and row[2].upper() in {"DDA", "DIA"}:
            # Legacy three-column layout: file, experiment, data_type.
            experiment = row[1]
            bioreplicate = ""
            data_type = row[2].upper()
        else:
            raise CLIUsageError(
                "Unable to normalize manifest row for FragPipe. Expected either "
                "`file<TAB>experiment<TAB>bioreplicate<TAB>data_type` or the "
                "`file<TAB>experiment<TAB>data_type` compatibility layout. "
                f"Offending row: {row!r}"
            )

        normalized_rows.append([row[0], experiment, bioreplicate, data_type])

    with normalized_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows(normalized_rows)

    log.info("Normalized FragPipe manifest written to %s", normalized_path.resolve())
    return normalized_path


def _resolve_search_workflow(
    workflow: str | None,
    manifest: str,
    quant_mode: bool | None = None,
) -> str:
    if workflow:
        return workflow
    data_type = _detect_manifest_data_type(manifest)
    return str(default_workflow_for_command(data_type, "search"))


def _detect_diann_bin(tools_folder: str | Path) -> Path | None:
    diann_dir = Path(tools_folder) / "diann"
    if diann_dir.is_dir():
        found = _walk_for_diann_executable(diann_dir)
        if found:
            return found
    return None


def _detect_diann_bin_on_path() -> Path | None:
    names = ("diann.exe", "DIA-NN.exe", "diann") if platform.system() == "Windows" else ("diann",)
    for name in names:
        try:
            return Path(require_command(name))
        except Exception:
            continue
    return None


def _infer_tools_folder(explicit_tools_folder: str | None, explicit_fragpipe_bin: str | None) -> Path:
    if explicit_tools_folder:
        tools_folder = Path(explicit_tools_folder).expanduser().resolve()
        if not tools_folder.is_dir():
            raise CLIUsageError(f"FragPipe tools directory not found: '{tools_folder}'.")
        return tools_folder

    if explicit_fragpipe_bin:
        fragpipe_path = Path(explicit_fragpipe_bin).expanduser().resolve()
        inferred = fragpipe_path.resolve().parent.parent / "tools"
        if inferred.is_dir():
            return inferred

    raise CLIUsageError(
        "Provide '--tools-dir' pointing to your local FragPipe tools directory, or use "
        "'--fragpipe-bin' from a standard FragPipe installation so MStoCIRC2 can infer "
        "the sibling 'tools' directory automatically."
    )


def _validate_fragpipe_python(python_bin: str, data_type: str) -> None:
    python_path = Path(python_bin)
    if not python_path.is_file():
        raise CLIUsageError(f"Python interpreter not found: '{python_bin}'.")

    if data_type != "DIA":
        return

    missing: list[str] = []
    for package in _DIA_REQUIRED_PYTHON_PACKAGES:
        proc = subprocess.run(
            [str(python_path), "-m", "pip", "show", "--files", package],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            missing.append(package)

    if missing:
        missing_str = ", ".join(missing)
        raise CLIUsageError(
            "DIA workflows require '--python-bin' to point to a Python environment that already "
            f"contains the FragPipe DIA spectral-library dependencies: {missing_str}. "
            f"The provided interpreter '{python_bin}' is missing them. "
            "DIA runs cannot start or complete successfully until that dedicated DIA Python "
            "environment has been prepared. Please install the missing packages into your DIA-specific "
            "Python and rerun, for example with '/path/to/your/dia-python'."
        )


def _resolve_fragpipe_python(cli_value: str | None) -> str:
    python_bin = (cli_value or "").strip() or os.environ.get(_FRAGPIPE_PYTHON_ENV, "").strip() or sys.executable
    candidate = Path(python_bin).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    try:
        return require_command(python_bin)
    except Exception as exc:
        raise CLIUsageError(
            "Python interpreter not found. Provide '--python-bin', set "
            f"'{_FRAGPIPE_PYTHON_ENV}', or ensure the interpreter is on PATH: '{python_bin}'."
        ) from exc


def resolve_search_runtime(
    *,
    manifest: str,
    tools_folder: str | None,
    fragpipe_bin: str | None,
    diann_bin: str | None,
    python_bin: str | None,
) -> SearchRuntime:
    data_type = _detect_manifest_data_type(manifest)
    if not fragpipe_bin:
        raise CLIUsageError(
            "Missing required '--fragpipe-bin'. Provide the executable from your local FragPipe "
            "installation, for example '/path/to/fragpipe/bin/fragpipe'."
        )
    resolved_python = _resolve_fragpipe_python(python_bin)
    resolved_tools = _infer_tools_folder(tools_folder, fragpipe_bin)
    resolved_diann = (
        Path(diann_bin).expanduser().resolve()
        if diann_bin
        else _detect_diann_bin(resolved_tools) or _detect_diann_bin_on_path()
    )
    if resolved_diann is None:
        raise CLIUsageError(
            "Unable to locate a DIA-NN executable. Use '--diann-bin' or provide a FragPipe "
            "tools directory that contains DIA-NN."
        )
    if not resolved_diann.is_file():
        raise CLIUsageError(f"DIA-NN executable not found: '{resolved_diann}'.")
    resolved_fragpipe = Path(fragpipe_bin).expanduser().resolve()
    if not resolved_fragpipe.is_file():
        raise CLIUsageError(f"FragPipe executable not found: '{resolved_fragpipe}'.")
    _validate_fragpipe_python(resolved_python, data_type)
    return SearchRuntime(
        data_type=data_type,
        python_bin=resolved_python,
        tools_folder=resolved_tools,
        diann_bin=resolved_diann,
        fragpipe_bin=resolved_fragpipe,
    )


def add_search_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    epilog = join_blocks(
        help_block(
            "Examples",
            [
                "  mstocirc2 search -co out_orf/circRNA_bsj_corf.fasta -cp canonical.fasta -mf manifest.tsv -wf custom.workflow -fb /opt/fragpipe/bin/fragpipe -td /opt/fragpipe/tools -db /opt/fragpipe/tools/diann/diann -py /opt/dia/bin/python -o out_search",
            ],
        ),
    )
    p = subparsers.add_parser(
        "search",
        help="CircRNA-specific MS database searching via FragPipe/DIA-NN",
        formatter_class=CLIHelpFormatter,
        description=branded_description(
            "Build a circRNA-aware FASTA database, normalize the FragPipe manifest, inject the database path "
            "into a workflow template, and launch FragPipe in native '--headless' mode.",
            "mstocirc2 search -cp <FILE> -mf <FILE> -fb <PATH> [options]",
        ),
        epilog=epilog,
    )
    p.set_defaults(runner=run_search)
    p.add_argument("-co", "--circ-orf", dest="circ_orf", metavar="<FILE>", default=None, help="Path to the predicted circRNA ORF FASTA exported by the ORF stage.")
    p.add_argument(
        "-cp",
        "--canonical-protein",
        dest="linear_protein",
        metavar="<FILE>",
        required=True,
        help="Path to the canonical linear proteome FASTA used to assemble the combined search database.",
    )
    p.add_argument(
        "-wf",
        "--workflow",
        metavar="<FILE>",
        default=None,
        help="Path to a custom FragPipe workflow template. If omitted, MStoCIRC2 selects a built-in workflow template from the manifest data type and pipeline mode.",
    )
    p.add_argument("-mf", "--manifest", dest="manifest", metavar="<FILE>", required=True, help="Path to the FragPipe manifest file.")
    p.add_argument(
        "-fb",
        "--fragpipe-bin",
        dest="fragpipe_bin",
        metavar="<PATH>",
        required=True,
        help="Local FragPipe executable for native '--headless' execution.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        dest="outdir",
        metavar="<DIR>",
        default=None,
        help=(
            "Directory for search-stage outputs, generated workflow files, and FragPipe logs. "
            "If omitted, MStoCIRC2 creates 'MStoCIRC2_search_YY-MM-DD.N' under the "
            "current working directory."
        ),
    )
    p.add_argument(
        "-td",
        "--tools-dir",
        dest="tools_folder",
        metavar="<DIR>",
        default=None,
        help=(
            "Path to the FragPipe tools directory. If omitted, MStoCIRC2 infers '<fragpipe_root>/tools' "
            "from '--fragpipe-bin' when the installation uses the standard layout."
        ),
    )
    p.add_argument("-db", "--diann-bin", dest="diann_bin", metavar="<PATH>", default=None, help="Path to the DIA-NN executable. Recommended for DIA workflows; optional for DDA when discoverable from the tools directory or PATH.")
    p.add_argument(
        "-py",
        "--python-bin",
        dest="python_bin",
        metavar="<PATH>",
        default="",
        help=(
            f"FragPipe runtime Python. Default: the current MStoCIRC2 Python or '{_FRAGPIPE_PYTHON_ENV}'. "
            "For DIA workflows, it must include fragpipe-speclib, easypqp, and lxml."
        ),
    )
    return p


def run_search(args: argparse.Namespace) -> int:
    if args.outdir and str(args.outdir).strip():
        out_dir = Path(args.outdir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = make_default_module_output_dir("search")
    args.outdir = str(out_dir)
    runtime = resolve_search_runtime(
        manifest=args.manifest,
        tools_folder=args.tools_folder,
        fragpipe_bin=args.fragpipe_bin,
        diann_bin=args.diann_bin,
        python_bin=args.python_bin,
    )

    db_fasta = assemble_database(
        circ_orf_path=args.circ_orf or "",
        linear_protein_path=args.linear_protein,
        output_dir=str(out_dir),
    )

    run_wf = inject_database_path(
        workflow_template=_resolve_search_workflow(args.workflow, args.manifest),
        database_fasta=str(db_fasta),
        output_dir=str(out_dir),
    )

    fragpipe_manifest = _normalize_manifest_for_fragpipe(args.manifest, out_dir)

    log.info("Tools directory: %s", runtime.tools_folder.resolve())
    log.info("FragPipe executable: %s", runtime.fragpipe_bin.resolve())
    log.info("DIA-NN executable: %s", runtime.diann_bin.resolve())
    log.info("FragPipe Python: %s", runtime.python_bin)
    log.info("FragPipe manifest: %s", fragpipe_manifest.resolve())

    return run_fragpipe(
        fragpipe_bin=str(runtime.fragpipe_bin),
        run_workflow=str(run_wf),
        manifest=str(fragpipe_manifest),
        workdir=str(out_dir),
        tools_folder=str(runtime.tools_folder),
        diann_bin=str(runtime.diann_bin),
        python_bin=runtime.python_bin,
    )
