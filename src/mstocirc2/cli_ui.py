"""Shared CLI presentation helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CLI_LOGO = r"""
 __  __  ____   _          ____ ___ ____   ____     ____
|  \/  |/ ___| | |_ ___   / ___|_ _|  _ \ / ___/   |___ \
| |\/| |\___ \ | __/ _ \ | |    | || |_) | |        __) |
| |  | | ___) || || (_) || |___ | ||  _ <| |___    / __/
|_|  |_||____/  \__\___/  \____|___|_| \_\\____|  |_____|

"""

CLI_TAGLINE = "Mass-spectrometry-driven circRNA translation discovery toolkit"
HELP_WIDTH = 220
HELP_MAX_POSITION = 32
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[92m"
ANSI_CYAN = "\033[96m"


class CLIHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
):
    """Help formatter with stable width and readable raw-text sections."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=HELP_MAX_POSITION, width=HELP_WIDTH)

    @staticmethod
    def _format_default_value(value: object) -> str:
        if value is None:
            return "none"
        if value == "":
            return '""'
        if isinstance(value, (list, tuple)):
            return ",".join(str(item) for item in value) or "none"
        return str(value)

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if help_text is argparse.SUPPRESS or not action.option_strings:
            return help_text
        prefix = "[Required] " if action.required else "[Optional] "
        if help_text.startswith("[Required]") or help_text.startswith("[Optional]"):
            text = help_text
        else:
            text = prefix + help_text

        if (
            not action.required
            and action.default is not argparse.SUPPRESS
            and action.default not in (None, "none")
            and "Default:" not in text
            and "Defaults to" not in text
        ):
            text = f"{text} Default: {self._format_default_value(action.default)}."
        return text


def _wrap_cli_text(line: str) -> str:
    return line.rstrip()


def help_block(title: str, lines: list[str]) -> str:
    body = "\n".join(_wrap_cli_text(line) for line in lines)
    return f"{title}\n{body}".rstrip()


def join_blocks(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block.strip())


def branded_description(summary: str, syntax: str | None = None) -> str:
    lines = [CLI_LOGO.rstrip(), CLI_TAGLINE, "", _wrap_cli_text(summary)]
    if syntax:
        lines.extend(["", "Syntax:", _wrap_cli_text(f"  {syntax}")])
    return "\n".join(lines).rstrip()


def emit_startup_banner(stream=None) -> None:
    target = stream or sys.stderr
    target.write(CLI_LOGO.rstrip() + "\n")
    target.write(CLI_TAGLINE + "\n\n")
    target.flush()


def format_duration(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    if total_seconds < 60:
        rounded = round(total_seconds, 1)
        if float(rounded).is_integer():
            return f"{int(rounded)}s"
        return f"{rounded:.1f}s"
    minutes, secs = divmod(int(round(total_seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"


def emit_completion_banner(
    *,
    command_name: str,
    elapsed_seconds: float,
    output_path: str | Path | None,
    key_result: str | Path | None = None,
    stream=None,
) -> None:
    target = stream or sys.stderr
    output_text = str(output_path) if output_path else "<not available>"
    duration_text = format_duration(elapsed_seconds)
    if getattr(target, "isatty", lambda: False)():
        title = f"{ANSI_BOLD}{ANSI_GREEN}COMPLETED | mstocirc2 {command_name}{ANSI_RESET}"
        duration = f"{ANSI_BOLD}Duration  |{ANSI_RESET} {ANSI_CYAN}{duration_text}{ANSI_RESET}"
        output = f"{ANSI_BOLD}Output    |{ANSI_RESET} {ANSI_CYAN}{output_text}{ANSI_RESET}"
        key_line = (
            f"{ANSI_BOLD}Key Result|{ANSI_RESET} {ANSI_CYAN}{key_result}{ANSI_RESET}"
            if key_result
            else None
        )
    else:
        title = f"COMPLETED | mstocirc2 {command_name}"
        duration = f"Duration  | {duration_text}"
        output = f"Output    | {output_text}"
        key_line = f"Key Result| {key_result}" if key_result else None

    lines = [
        "=" * 72,
        title,
        duration,
        output,
    ]
    if key_line:
        lines.append(key_line)
    lines.append("=" * 72)
    target.write("\n" + "\n".join(lines) + "\n")
    target.flush()
