"""MStoCIRC2 CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .cli_ui import CLIHelpFormatter, branded_description, emit_completion_banner, emit_startup_banner
from .core import CLIUsageError, MStoCIRC2Error, configure_logging
from .differential_expression.command_runner import add_dea_subparser
from .integrated_pipeline import add_integrated_subparsers
from .orf_prediction.command_runner import add_orf_subparser
from .spectral_searching.command_runner import add_search_subparser
from .translation_validation.command_runner import add_eval_subparser

Runner = Callable[[argparse.Namespace], int]
log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=CLIHelpFormatter,
        description=branded_description(
            "Unified command-line interface for circRNA ORF discovery, FragPipe-assisted peptide search, "
            "translation evidence evaluation, and downstream DEA.",
            "mstocirc2 <command> [options]",
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="Available Commands",
        description="Choose one pipeline module below, then run 'mstocirc2 COMMAND -h' for full details.",
    )
    subparsers.required = True

    add_integrated_subparsers(subparsers)
    add_orf_subparser(subparsers)
    add_search_subparser(subparsers)
    add_eval_subparser(subparsers)
    add_dea_subparser(subparsers)
    return parser


def _resolve_primary_output_path(args: argparse.Namespace) -> Path | None:
    for name in ("output_dir", "outdir", "file_out", "out_circ_file"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
    return None


def _resolve_key_result_path(args: argparse.Namespace, output_path: Path | None) -> Path | None:
    if output_path is None:
        return None
    command = getattr(args, "command", "")
    candidates: dict[str, list[Path]] = {
        "orf": [output_path / "circRNA_bsj_corf.fasta"],
        "search": [output_path / "fragpipe_input.fp-manifest", output_path / "fragpipe.workflow"],
        "eval": [output_path / "circ_predict.txt"],
        "dea": [output_path / "run_metadata.json"],
        "nonquant": [output_path / "eval" / "circ_predict.txt"],
        "quant": [output_path / "eval" / "circ_predict.txt", output_path / "dea" / "run_metadata.json"],
    }
    for candidate in candidates.get(command, []):
        if candidate.exists():
            return candidate.resolve()
    return None


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    emit_startup_banner()

    runner: Runner | None = getattr(args, "runner", None)
    if runner is None:
        parser.print_help()
        return 2

    start_time = time.perf_counter()
    try:
        exit_code = int(runner(args) or 0)
        if exit_code == 0:
            output_path = _resolve_primary_output_path(args)
            key_result = _resolve_key_result_path(args, output_path)
            emit_completion_banner(
                command_name=getattr(args, "command", "run"),
                elapsed_seconds=time.perf_counter() - start_time,
                output_path=output_path,
                key_result=key_result,
            )
        return exit_code
    except CLIUsageError as exc:
        log.error(str(exc))
        return 2
    except MStoCIRC2Error as exc:
        log.error(str(exc))
        return 1
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1
    except KeyboardInterrupt:
        log.error("Execution interrupted by user.")
        return 130
    except Exception:
        log.exception("Unhandled pipeline error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
