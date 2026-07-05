from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Any

from ..cli_ui import CLIHelpFormatter, branded_description, help_block, join_blocks
from ..core import CLIUsageError, require_command
from .bsj_truncator import filter_within_circ, postprocess_orf_records
from .circular_translator import translate_circ_orfs
from .genome_extractor import extract_sequences
from .linear_homology import filter_by_references
from .sequence_exporter import (
    cleanup_intermediate_files,
    restore_duplicate_orfs,
    write_final_outputs,
)

log = logging.getLogger(__name__)
DEFAULT_START_CODONS = ["ATG", "TTG", "CTG", "GTG"]


def add_orf_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    epilog = join_blocks(
        help_block(
            "Input Modes",
            [
                "  Sequence mode: provide '--circ-seq' and optionally '--circ-info'.",
                "  Genome mode: provide '--dna-seq', '--dna-gff', and at least one circRNA source "
                "from '--circ-info', '--find-circ', '--circexplorer', or '--ciri'.",
            ],
        ),
        help_block(
            "Examples",
            [
                "  mstocirc2 orf -cs circ.fasta -cp canonical.fasta -o out_orf",
                "  mstocirc2 orf -ds genome.fa -dg genome.gtf -fc find_circ.txt -cp canonical.fasta -o out_orf",
            ],
        ),
    )
    parser = subparsers.add_parser(
        "orf",
        help="circRNA ORF prediction and BSJ-aware sequence export",
        formatter_class=CLIHelpFormatter,
        description=branded_description(
            "Predict circRNA ORFs, apply BSJ-aware truncation and deduplication, and export the FASTA and "
            "mapping files required by downstream search and evaluation modules.",
            "mstocirc2 orf -cp <FILE> [sequence-mode options | genome-mode options] [options]",
        ),
        epilog=epilog,
    )
    parser.set_defaults(runner=run_orf_prediction)
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="out_circ_file",
        metavar="<DIR>",
        default=".",
        help="Directory for ORF-stage outputs. Default: current working directory.",
    )

    req = parser.add_argument_group("Core inputs")
    req.add_argument(
        "-cp",
        "--canonical-protein",
        dest="canonical_protein",
        metavar="<FILE>",
        nargs="+",
        required=True,
        help="One or more canonical linear proteome FASTA files used for homology filtering.",
    )
    seq = parser.add_argument_group("Sequence-input mode")
    seq.add_argument(
        "-cs",
        "--circ-seq",
        dest="circ_seq_file",
        metavar="<FILE>",
        default="none",
        help="Path to the full-length mature circRNA FASTA file.",
    )
    seq.add_argument(
        "-ci",
        "--circ-info",
        dest="circ_info",
        metavar="<FILE>",
        default="none",
        help="Path to the circRNA annotation table used for gene and strand annotation.",
    )

    genome = parser.add_argument_group("Genome-based mode")
    genome.add_argument(
        "-ds",
        "--dna-seq",
        dest="dna_seq",
        metavar="<FILE>",
        default=None,
        help="Path to the reference genome FASTA file.",
    )
    genome.add_argument(
        "-dg",
        "--dna-gff",
        dest="dna_gff",
        metavar="<FILE>",
        default=None,
        help="Path to the reference genome GFF or GTF annotation file.",
    )
    genome.add_argument(
        "-fc",
        "--find-circ",
        dest="find_circ",
        metavar="<FILE>",
        default="none",
        help="Path to the find_circ result file.",
    )
    genome.add_argument(
        "-ce",
        "--circexplorer",
        dest="circexplorer",
        metavar="<FILE>",
        default="none",
        help="Path to the CIRCexplorer result file.",
    )
    genome.add_argument(
        "-ciri",
        "--ciri",
        dest="CIRI",
        metavar="<FILE>",
        default="none",
        help="Path to the CIRI result file.",
    )

    opt = parser.add_argument_group("ORF settings")
    opt.add_argument(
        "-ml",
        "--min-orf-len",
        dest="min_orf_len",
        metavar="<INT>",
        type=int,
        default=6,
        help="Minimum amino-acid length retained after ORF translation. Valid range: value >= 1. Default: 6.",
    )
    opt.add_argument(
        "-fl",
        "--flank-aa-len",
        dest="flank_aa_len",
        metavar="<INT>",
        type=int,
        default=24,
        help="Flanking amino-acid window applied during BSJ-aware truncation. Valid range: value >= 0. Default: 24.",
    )
    opt.add_argument(
        "-sc",
        "--start-codon",
        dest="start_codon",
        metavar="<CODON_LIST>",
        default=None,
        help=(
            "Comma-separated DNA start codons used for ORF prediction, for example "
            f"'ATG,TTG,CTG,GTG'. Default: {','.join(DEFAULT_START_CODONS)}."
        ),
    )
    return parser


def _resolve_output_dir(value: str) -> Path:
    output_dir = Path(value or ".").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _validate_orf_inputs(args: argparse.Namespace) -> bool:
    use_sequence_mode = args.circ_seq_file != "none"
    if use_sequence_mode:
        return True

    if not args.dna_seq or not args.dna_gff:
        raise CLIUsageError(
            "Genome-based ORF prediction requires '--dna-seq' and '--dna-gff' when '--circ-seq' is not provided."
        )
    if all(getattr(args, name) == "none" for name in ("circ_info", "find_circ", "circexplorer", "CIRI")):
        raise CLIUsageError(
            "Provide at least one circRNA source via '--circ-info', '--find-circ', '--circexplorer', or '--ciri'."
        )
    return False


def _parse_start_codon_list(raw_values: str | None) -> list[str]:
    if raw_values is None:
        return list(DEFAULT_START_CODONS)
    if re.search(r"\s", raw_values):
        raise CLIUsageError(
            "Invalid '--start-codon' value. Use comma-separated DNA codons, for example: ATG,TTG,CTG,GTG."
        )

    start_codons: list[str] = []
    for codon in raw_values.split(","):
        normalized = codon.strip().upper()
        if not normalized:
            continue
        if not re.fullmatch(r"[ACGT]{3}", normalized):
            raise CLIUsageError(
                f"Invalid start codon '{normalized}'. Start codons must be 3-base DNA codons using A/C/G/T."
            )
        if normalized not in start_codons:
            start_codons.append(normalized)

    if not start_codons:
        raise CLIUsageError("At least one valid start codon must be provided via '--start-codon'.")
    return start_codons


def run_orf_prediction(args: Any) -> int:
    """Run the ORF prediction stage."""
    use_sequence_mode = _validate_orf_inputs(args)
    if not use_sequence_mode:
        require_command("bedtools")
    start_codon_list = _parse_start_codon_list(args.start_codon)
    output_dir = _resolve_output_dir(args.out_circ_file)
    output_prefix = f"{output_dir.as_posix()}/"
    protein_ref_list = (
        args.canonical_protein
        if isinstance(args.canonical_protein, list)
        else [args.canonical_protein]
    )

    mode_label = "sequence-input mode" if use_sequence_mode else "genome-based mode"
    log.info("ORF stage started in %s", mode_label)
    log.info("Output directory: %s", output_dir.resolve())
    log.info("Start codons: %s", ",".join(start_codon_list))

    extract_sequences(args, output_prefix)

    raw_records = translate_circ_orfs(
        fasta_path=f"{output_prefix}circRNA_full_length.fasta",
        min_orf_len=args.min_orf_len,
        flank_aa_len=args.flank_aa_len,
        start_codons=start_codon_list,
    )
    raw_orf_path = output_dir / "circRNA_all_ORF.fasta"
    with raw_orf_path.open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(f">{record['header']}\n{record['sequence']}\n")
    log.info("Raw ORFs exported: %d", len(raw_records))

    log.info("Applying BSJ-aware truncation.")
    stage1 = postprocess_orf_records(
        raw_records,
        args.flank_aa_len,
        start_codon_list=start_codon_list,
    )
    log.info("BSJ-filtered ORFs: %d", len(stage1))

    log.info("Removing intra-circRNA duplicate ORFs.")
    stage2 = filter_within_circ(stage1)
    log.info("Deduplicated ORFs: %d", len(stage2))

    log.info("Filtering ORFs against linear reference proteomes.")
    stage3 = filter_by_references(stage2, protein_ref_list)
    log.info("Homology-filtered ORFs: %d", len(stage3))

    final_records = restore_duplicate_orfs(stage3)
    write_final_outputs(output_prefix, final_records)
    cleanup_intermediate_files(output_prefix, use_sequence_mode)

    log.info("ORF stage completed: %s", output_dir.resolve())
    return 0
