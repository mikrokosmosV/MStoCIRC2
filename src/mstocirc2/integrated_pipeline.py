"""High-level orchestration for end-to-end MStoCIRC2 workflows."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, Sequence

from .cli_ui import CLIHelpFormatter, branded_description, help_block, join_blocks
from .core import CLIUsageError, MStoCIRC2Error, ensure_directory, make_default_module_output_dir
from .differential_expression.command_runner import run_differential_analysis
from .differential_expression.workflow_strategy import STRATEGY_REGISTRY
from .fragpipe_workflows import default_workflow_for_command
from .orf_prediction.command_runner import run_orf_prediction
from .spectral_searching.command_runner import resolve_search_runtime, run_search
from .translation_validation.command_runner import run_eval

log = logging.getLogger(__name__)
_PIPELINE_RULE = "=" * 86
_STAGE_RULE = "-" * 86
_DEFAULT_ORGANISM = "hsa"
_KEGG_REFERENCE_URL = "https://www.kegg.jp/kegg/tables/br08606.html"


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict[str, object]) -> None:
    log.debug("%s %s %s %s", hypothesis_id, location, msg, data)


def _debug_search_tree(root: Path) -> dict[str, object]:
    if not root.exists():
        return {"exists": False}
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    return {"exists": True, "files": files[:200], "file_count": len(files)}


def _require_eval_result(eval_dir: Path) -> Path:
    circ_predict = eval_dir / "circ_predict.txt"
    if not circ_predict.exists():
        raise MStoCIRC2Error(
            f"Translation evaluation finished without generating '{circ_predict.name}' in '{eval_dir}'."
        )

    with circ_predict.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
        first_data_row = handle.readline()

    if not header.strip() or not first_data_row.strip():
        raise MStoCIRC2Error(
            f"Translation evaluation produced an empty '{circ_predict.name}' in '{eval_dir}'."
        )

    return circ_predict


def _namespace_to_argv(namespace: argparse.Namespace, keys: Iterable[str]) -> dict[str, object]:
    return {key: getattr(namespace, key) for key in keys}


def _manifest_rows(manifest_path: str | Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with Path(manifest_path).open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            rows.append([cell.strip() for cell in row])
    if not rows:
        raise CLIUsageError(f"Manifest file is empty: '{manifest_path}'.")
    return rows


def detect_manifest_data_type(manifest_path: str | Path) -> str:
    values = []
    for row in _manifest_rows(manifest_path):
        if len(row) >= 4 and row[3]:
            values.append(row[3].upper())
    if not values:
        raise CLIUsageError(
            "Unable to infer data type from manifest. Ensure column 4 contains DDA or DIA."
        )
    normalized = set(values)
    if len(normalized) > 1:
        raise CLIUsageError(
            f"Manifest mixes data types {sorted(normalized)}. Use a single acquisition mode per run."
        )
    data_type = normalized.pop()
    if data_type not in {"DDA", "DIA"}:
        raise CLIUsageError(
            f"Unsupported manifest data type '{data_type}'. Expected 'DDA' or 'DIA'."
        )
    return data_type


def build_design_from_manifest(manifest_path: str | Path, output_path: str | Path) -> Path:
    rows = _manifest_rows(manifest_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "condition"])
        for row in rows:
            if len(row) < 2 or not row[1]:
                raise CLIUsageError(
                    f"Manifest row is missing the sample/group column required for DEA: {row!r}"
                )
            ms_path = Path(row[0])
            sample_name = row[1] or ms_path.stem
            condition = row[2] if len(row) >= 3 and row[2] else row[1]
            writer.writerow([sample_name, condition])
    return out_path


def build_design_from_experiment_annotation(
    annotation_path: str | Path,
    output_path: str | Path,
) -> Path:
    source = Path(annotation_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise CLIUsageError(f"Experiment annotation file is empty: '{annotation_path}'.")
        field_map = {field.strip().lower(): field for field in reader.fieldnames if field and field.strip()}
        sample_field = field_map.get("sample") or field_map.get("file")
        condition_field = field_map.get("condition")
        if sample_field is None or condition_field is None:
            raise CLIUsageError(
                "Experiment annotation must contain 'sample' (or 'file') and 'condition' columns "
                f"for DEA design reconstruction: '{annotation_path}'."
            )

        rows: list[tuple[str, str]] = []
        for row in reader:
            sample = str(row.get(sample_field, "")).strip()
            condition = str(row.get(condition_field, "")).strip()
            if not sample and not condition:
                continue
            if not sample or not condition:
                raise CLIUsageError(
                    "Experiment annotation row is missing the sample/file or condition value "
                    f"required for DEA: {row!r}"
                )
            rows.append((sample, condition))

    if not rows:
        raise CLIUsageError(f"Experiment annotation file has no usable rows: '{annotation_path}'.")

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "condition"])
        writer.writerows(rows)
    return out_path


def _manifest_conditions(manifest_path: str | Path) -> list[str]:
    conditions: list[str] = []
    for row in _manifest_rows(manifest_path):
        condition = row[2] if len(row) >= 3 and row[2] else (row[1] if len(row) >= 2 else "")
        if condition:
            conditions.append(condition)
    return sorted(set(conditions))


def resolve_workflow_path(workflow: str | None, manifest: str, command_name: str) -> str:
    if workflow:
        return workflow
    data_type = detect_manifest_data_type(manifest)
    return str(default_workflow_for_command(data_type, command_name))


def _find_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_fragpipe_peptide_table(search_dir: str | Path, prefer_quant: bool = False) -> Path:
    root = Path(search_dir)
    _debug_report(
        "A",
        "integrated_pipeline.discover_fragpipe_peptide_table:entry",
        "[DEBUG] scanning FragPipe outputs for peptide table",
        {
            "search_dir": str(root),
            "prefer_quant": prefer_quant,
            "tree": _debug_search_tree(root),
        },
    )
    if prefer_quant:
        dia_path = root / "dia-quant-output" / "report.pr_matrix.tsv"
        if dia_path.exists():
            _debug_report(
                "E",
                "integrated_pipeline.discover_fragpipe_peptide_table:dia",
                "[DEBUG] selected DIA quantitative matrix",
                {"selected": str(dia_path)},
            )
            return dia_path

    candidates: list[Path] = []
    combined = root / "combined_peptide.tsv"
    if combined.exists():
        candidates.append(combined)
    direct = root / "peptide.tsv"
    if direct.exists():
        candidates.append(direct)

    grouped_candidates: list[Path] = []
    for child in sorted(root.iterdir()) if root.exists() else []:
        if not child.is_dir():
            continue
        grouped = child / "peptide.tsv"
        if grouped.exists():
            grouped_candidates.append(grouped)

    candidates.extend(grouped_candidates)

    if combined.exists() or direct.exists():
        found = _find_existing(candidates)
        if found is not None:
            _debug_report(
                "A",
                "integrated_pipeline.discover_fragpipe_peptide_table:direct",
                "[DEBUG] selected root-level peptide table",
                {"selected": str(found), "candidates": [str(path) for path in candidates]},
            )
            return found

    if len(grouped_candidates) == 1:
        _debug_report(
            "A",
            "integrated_pipeline.discover_fragpipe_peptide_table:grouped",
            "[DEBUG] selected grouped peptide table",
            {
                "selected": str(grouped_candidates[0]),
                "grouped_candidates": [str(path) for path in grouped_candidates],
            },
        )
        return grouped_candidates[0]
    if len(grouped_candidates) > 1:
        _debug_report(
            "A",
            "integrated_pipeline.discover_fragpipe_peptide_table:ambiguous",
            "[DEBUG] multiple grouped peptide tables found without combined table",
            {"grouped_candidates": [str(path) for path in grouped_candidates]},
        )
        raise CLIUsageError(
            f"Found multiple grouped FragPipe peptide tables under '{root}' but no "
            "`combined_peptide.tsv`. Please provide a workflow/output layout that produces "
            "a combined table or point `eval/dea` to a specific peptide file."
        )

    found = _find_existing(candidates)
    if found is None:
        _debug_report(
            "A",
            "integrated_pipeline.discover_fragpipe_peptide_table:missing",
            "[DEBUG] no peptide table discovered",
            {
                "search_dir": str(root),
                "candidates": [str(path) for path in candidates],
                "grouped_candidates": [str(path) for path in grouped_candidates],
            },
        )
        raise CLIUsageError(
            f"Could not locate a FragPipe peptide table under '{root}'. "
            "Expected `combined_peptide.tsv`, `peptide.tsv`, or `dia-quant-output/report.pr_matrix.tsv`."
        )
    _debug_report(
        "A",
        "integrated_pipeline.discover_fragpipe_peptide_table:fallback",
        "[DEBUG] selected fallback peptide table",
        {"selected": str(found)},
    )
    return found


def _resolve_stage_output(parent: Path, explicit: str | None, default_name: str) -> str:
    if explicit:
        return explicit
    return str(parent / default_name)


def _as_absolute_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _prepare_runtime_input_path(stage_outdir: str | Path, filename: str) -> Path:
    stage_dir = Path(stage_outdir)
    runtime_dir = stage_dir.parent / ".mstocirc2_runtime" / stage_dir.name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / filename


def _log_stage_banner(stage_number: int, stage_total: int, stage_name: str, output_dir: str | Path) -> None:
    stage_output = _as_absolute_path(output_dir)
    log.info(_STAGE_RULE)
    log.info("Stage %d/%d START | %s", stage_number, stage_total, stage_name)
    log.info("Output            | %s", stage_output)
    log.info(_STAGE_RULE)


def _log_stage_complete(
    stage_number: int,
    stage_total: int,
    stage_name: str,
    output_dir: str | Path,
    key_result: str | Path | None = None,
) -> None:
    stage_output = _as_absolute_path(output_dir)
    log.info(_STAGE_RULE)
    log.info("Stage %d/%d END   | %s", stage_number, stage_total, stage_name)
    log.info("Output            | %s", stage_output)
    if key_result is not None:
        log.info("Key Result        | %s", _as_absolute_path(key_result))
    log.info(_STAGE_RULE)


def _log_pipeline_start(pipeline_name: str, root_out: Path, workflow: str, manifest: str) -> None:
    log.info(_PIPELINE_RULE)
    log.info("Pipeline | %s", pipeline_name)
    log.info("Output   | %s", _as_absolute_path(root_out))
    log.info("Manifest | %s", _as_absolute_path(manifest))
    log.info("Workflow | %s", _as_absolute_path(workflow))
    log.info(_PIPELINE_RULE)


def _canonical_protein_refs(args: argparse.Namespace) -> list[str]:
    value = args.canonical_protein
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _canonical_protein_fasta(args: argparse.Namespace) -> str:
    refs = _canonical_protein_refs(args)
    if len(refs) != 1:
        raise CLIUsageError(
            "Integrated workflows currently require exactly one '--canonical-protein' FASTA "
            "because the same reference is reused by FragPipe and translation evaluation."
        )
    return refs[0]


def _preflight_integrated_search_runtime(args: argparse.Namespace) -> None:
    runtime = resolve_search_runtime(
        manifest=args.manifest,
        tools_folder=args.tools_folder,
        fragpipe_bin=args.fragpipe_bin,
        diann_bin=args.diann_bin,
        python_bin=args.python_bin,
    )
    if not getattr(args, "python_bin", "").strip():
        args.python_bin = runtime.python_bin


def _run_orf_stage(args: argparse.Namespace, output_dir: str) -> Path:
    orf_args = argparse.Namespace(
        **_namespace_to_argv(
            args,
            [
                "circ_seq_file",
                "circ_info",
                "dna_seq",
                "dna_gff",
                "find_circ",
                "circexplorer",
                "CIRI",
                "min_orf_len",
                "flank_aa_len",
                "start_codon",
            ],
        ),
        canonical_protein=_canonical_protein_refs(args),
        out_circ_file=output_dir,
    )
    exit_code = run_orf_prediction(orf_args)
    if exit_code:
        raise MStoCIRC2Error(f"ORF stage failed with exit code {exit_code}.")
    return Path(output_dir)


def _run_search_stage(
    args: argparse.Namespace,
    circ_orf_path: str,
    search_outdir: str,
    workflow: str,
) -> Path:
    _debug_report(
        "B",
        "integrated_pipeline._run_search_stage:before",
        "[DEBUG] entering search stage",
        {
            "circ_orf_path": circ_orf_path,
            "search_outdir": search_outdir,
            "workflow": workflow,
            "manifest": args.manifest,
        },
    )
    search_args = argparse.Namespace(
        circ_orf=circ_orf_path,
        linear_protein=_canonical_protein_fasta(args),
        workflow=workflow,
        manifest=args.manifest,
        outdir=search_outdir,
        tools_folder=args.tools_folder,
        diann_bin=args.diann_bin,
        python_bin=args.python_bin,
        fragpipe_bin=args.fragpipe_bin,
    )
    exit_code = run_search(search_args)
    _debug_report(
        "B",
        "integrated_pipeline._run_search_stage:after",
        "[DEBUG] search stage returned",
        {
            "exit_code": exit_code,
            "search_outdir": search_outdir,
            "tree": _debug_search_tree(Path(search_outdir)),
        },
    )
    if exit_code:
        raise MStoCIRC2Error(f"FragPipe stage failed with exit code {exit_code}.")
    return Path(search_outdir)


def _run_eval_stage(
    args: argparse.Namespace,
    orf_dir: Path,
    search_dir: Path,
    eval_outdir: str,
) -> Path:
    try:
        peptide_table = discover_fragpipe_peptide_table(search_dir)
    except CLIUsageError as exc:
        raise CLIUsageError(
            f"FragPipe stage finished but no usable peptide table was found under '{search_dir}'. "
            "Integrated translation evaluation requires `combined_peptide.tsv` or `peptide.tsv`. "
            f"Details: {exc}"
        ) from exc
    _debug_report(
        "A",
        "integrated_pipeline._run_eval_stage:selected_peptide",
        "[DEBUG] eval stage selected peptide table",
        {
            "search_dir": str(search_dir),
            "peptide_table": str(peptide_table),
            "eval_outdir": eval_outdir,
        },
    )
    circ_info_path = orf_dir / "circ_info_omit.gff"
    eval_args = argparse.Namespace(
        circ_file_input=None,
        circ_seq=str(orf_dir / "circRNA_full_length.fasta"),
        circ_info=str(circ_info_path) if circ_info_path.exists() else None,
        circ_orf=str(orf_dir / "circRNA_bsj_corf.fasta"),
        path_ms_input=str(peptide_table),
        circ_mapping=str(orf_dir / "circRNA_ORF_mapping.tsv"),
        canonical_protein=_canonical_protein_fasta(args),
        file_out=eval_outdir,
        deepcip_path=args.deepcip_path,
        deepcip_python=args.deepcip_python,
    )
    exit_code = run_eval(eval_args)
    if exit_code:
        raise MStoCIRC2Error(f"Translation evaluation stage failed with exit code {exit_code}.")
    eval_dir = Path(eval_outdir)
    _require_eval_result(eval_dir)
    return eval_dir


def _run_dea_stage(
    args: argparse.Namespace,
    search_dir: Path,
    eval_dir: Path,
    dea_outdir: str,
) -> Path:
    dea_dir = Path(dea_outdir)
    data_type = detect_manifest_data_type(args.manifest)
    circ_predict = _require_eval_result(eval_dir)
    peptide_matrix = discover_fragpipe_peptide_table(
        search_dir,
        prefer_quant=data_type == "DIA",
    )
    annotation_path = search_dir / "experiment_annotation.tsv"
    if data_type == "DIA" and annotation_path.exists():
        design_path = build_design_from_experiment_annotation(
            annotation_path,
            _prepare_runtime_input_path(dea_dir, "design.from_annotation.tsv"),
        )
        _debug_report(
            "C",
            "integrated_pipeline._run_dea_stage:design_annotation",
            "[DEBUG] DEA stage selected DIA experiment annotation for design reconstruction",
            {"annotation_path": str(annotation_path), "design_path": str(design_path)},
        )
    else:
        design_path = build_design_from_manifest(
            args.manifest,
            _prepare_runtime_input_path(dea_dir, "design.from_manifest.tsv"),
        )
        _debug_report(
            "C",
            "integrated_pipeline._run_dea_stage:design_manifest",
            "[DEBUG] DEA stage selected manifest for design reconstruction",
            {"manifest": args.manifest, "design_path": str(design_path)},
        )
    dea_args = argparse.Namespace(
        peptide_matrix=str(peptide_matrix),
        circrna_reference=str(circ_predict),
        design=str(design_path),
        output_dir=dea_outdir,
        strategy=args.strategy,
        use_adj_pvalue=args.use_adj_pvalue,
        organism=args.organism,
    )
    exit_code = run_differential_analysis(dea_args)
    if exit_code:
        raise MStoCIRC2Error(f"Differential expression stage failed with exit code {exit_code}.")
    return Path(dea_outdir)


def run_non_quant_pipeline(args: argparse.Namespace) -> int:
    if args.output_dir and str(args.output_dir).strip():
        root_out = ensure_directory(args.output_dir)
    else:
        root_out = make_default_module_output_dir("nonquant")
    args.output_dir = str(root_out)
    workflow = resolve_workflow_path(args.workflow, args.manifest, "nonquant")
    _preflight_integrated_search_runtime(args)
    log.info("Mode     | nonquant")
    _log_pipeline_start("nonquant", root_out, workflow, args.manifest)
    orf_outdir = _resolve_stage_output(root_out, args.orf_output_dir, "orf")
    _log_stage_banner(1, 3, "ORF Prediction", orf_outdir)
    orf_dir = _run_orf_stage(args, orf_outdir)
    _log_stage_complete(1, 3, "ORF Prediction", orf_dir, orf_dir / "circRNA_bsj_corf.fasta")
    search_outdir = _resolve_stage_output(root_out, args.search_output_dir, "search")
    _log_stage_banner(2, 3, "Spectral Search", search_outdir)
    search_dir = _run_search_stage(
        args,
        circ_orf_path=str(orf_dir / "circRNA_bsj_corf.fasta"),
        search_outdir=search_outdir,
        workflow=workflow,
    )
    _log_stage_complete(2, 3, "Spectral Search", search_dir, search_dir / "fragpipe.run.log")
    eval_outdir = _resolve_stage_output(root_out, args.eval_output_dir, "eval")
    _log_stage_banner(3, 3, "Translation Evaluation", eval_outdir)
    eval_dir = _run_eval_stage(
        args,
        orf_dir=orf_dir,
        search_dir=search_dir,
        eval_outdir=eval_outdir,
    )
    _log_stage_complete(3, 3, "Translation Evaluation", eval_dir, eval_dir / "circ_predict.txt")
    return 0


def run_quant_pipeline(args: argparse.Namespace) -> int:
    conditions = _manifest_conditions(args.manifest)
    if args.output_dir and str(args.output_dir).strip():
        root_out = ensure_directory(args.output_dir)
    else:
        root_out = make_default_module_output_dir("quant")
    args.output_dir = str(root_out)
    workflow = resolve_workflow_path(args.workflow, args.manifest, "quant")
    _preflight_integrated_search_runtime(args)
    log.info("Mode     | quant")
    _log_pipeline_start("quant", root_out, workflow, args.manifest)
    log.info("Groups   | %s", ", ".join(conditions) if conditions else "<none>")
    orf_outdir = _resolve_stage_output(root_out, args.orf_output_dir, "orf")
    _log_stage_banner(1, 4, "ORF Prediction", orf_outdir)
    orf_dir = _run_orf_stage(args, orf_outdir)
    _log_stage_complete(1, 4, "ORF Prediction", orf_dir, orf_dir / "circRNA_bsj_corf.fasta")
    search_outdir = _resolve_stage_output(root_out, args.search_output_dir, "search")
    _log_stage_banner(2, 4, "Quantitative Spectral Search", search_outdir)
    search_dir = _run_search_stage(
        args,
        circ_orf_path=str(orf_dir / "circRNA_bsj_corf.fasta"),
        search_outdir=search_outdir,
        workflow=workflow,
    )
    _log_stage_complete(2, 4, "Quantitative Spectral Search", search_dir, search_dir / "fragpipe.run.log")
    eval_outdir = _resolve_stage_output(root_out, args.eval_output_dir, "eval")
    _log_stage_banner(3, 4, "Translation Evaluation", eval_outdir)
    eval_dir = _run_eval_stage(
        args,
        orf_dir=orf_dir,
        search_dir=search_dir,
        eval_outdir=eval_outdir,
    )
    _log_stage_complete(3, 4, "Translation Evaluation", eval_dir, eval_dir / "circ_predict.txt")
    dea_outdir = _resolve_stage_output(root_out, args.dea_output_dir, "dea")
    _log_stage_banner(4, 4, "Differential Expression Analysis", dea_outdir)
    if len(conditions) < 2:
        skip_dir = Path(dea_outdir)
        skip_dir.mkdir(parents=True, exist_ok=True)
        message = (
            "DEA skipped: fewer than two groups were detected in the manifest. "
            f"Detected groups: {conditions or ['<none>']}."
        )
        (skip_dir / "dea_skipped.txt").write_text(message + "\n", encoding="utf-8")
        log.warning(message)
        _log_stage_complete(4, 4, "Differential Expression Analysis", skip_dir, skip_dir / "dea_skipped.txt")
        _debug_report(
            "C",
            "integrated_pipeline.run_quant_pipeline:skip_dea",
            "[DEBUG] skipping DEA because fewer than two groups were detected",
            {"conditions": conditions, "dea_dir": str(skip_dir)},
        )
        return 0
    dea_dir = _run_dea_stage(
        args,
        search_dir=search_dir,
        eval_dir=eval_dir,
        dea_outdir=dea_outdir,
    )
    _log_stage_complete(4, 4, "Differential Expression Analysis", dea_dir, dea_dir / "run_metadata.json")
    return 0


def _add_orf_inputs(parser: argparse.ArgumentParser) -> None:
    req = parser.add_argument_group("ORF inputs")
    req.add_argument(
        "-cp",
        "--canonical-protein",
        dest="canonical_protein",
        metavar="<FILE>",
        nargs="+",
        required=True,
        help="One or more canonical linear proteome FASTA files used for ORF homology filtering and search database assembly.",
    )
    seq = parser.add_argument_group("Sequence-input mode")
    seq.add_argument("-cs", "--circ-seq", dest="circ_seq_file", metavar="<FILE>", default="none", help="Path to the full-length mature circRNA FASTA file.")
    seq.add_argument("-ci", "--circ-info", dest="circ_info", metavar="<FILE>", default="none", help="Path to the circRNA annotation table used for gene and strand annotation.")
    genome = parser.add_argument_group("Genome-based mode")
    genome.add_argument("-ds", "--dna-seq", dest="dna_seq", metavar="<FILE>", default=None, help="Path to the reference genome FASTA file.")
    genome.add_argument("-dg", "--dna-gff", dest="dna_gff", metavar="<FILE>", default=None, help="Path to the reference genome GFF or GTF annotation file.")
    genome.add_argument("-fc", "--find-circ", dest="find_circ", metavar="<FILE>", default="none", help="Path to the find_circ result file.")
    genome.add_argument("-ce", "--circexplorer", dest="circexplorer", metavar="<FILE>", default="none", help="Path to the CIRCexplorer result file.")
    genome.add_argument("-ciri", "--ciri", dest="CIRI", metavar="<FILE>", default="none", help="Path to the CIRI result file.")
    opt = parser.add_argument_group("ORF settings")
    opt.add_argument("-ml", "--min-orf-len", dest="min_orf_len", metavar="<INT>", type=int, default=6, help="Minimum amino-acid length retained after ORF translation. Valid range: value >= 1. Default: 6.")
    opt.add_argument("-fl", "--flank-aa-len", dest="flank_aa_len", metavar="<INT>", type=int, default=24, help="Flanking amino-acid window applied during BSJ-aware truncation. Valid range: value >= 0. Default: 24.")
    opt.add_argument("-sc", "--start-codon", dest="start_codon", metavar="<CODON_LIST>", default=None, help="Comma-separated DNA start codons used for ORF prediction, for example 'ATG,TTG,CTG,GTG'. Default: ATG,TTG,CTG,GTG.")


def _add_fragpipe_inputs(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("FragPipe inputs")
    grp.add_argument("-wf", "--workflow", metavar="<FILE>", default=None, help="Path to a custom FragPipe workflow template. If omitted, MStoCIRC2 selects a built-in workflow template from the manifest data type and pipeline mode.")
    grp.add_argument("-mf", "--manifest", metavar="<FILE>", required=True, help="Path to the FragPipe manifest file.")
    grp.add_argument(
        "-fb",
        "--fragpipe-bin",
        dest="fragpipe_bin",
        metavar="<PATH>",
        required=True,
        help="Local FragPipe executable for native '--headless' execution.",
    )
    grp.add_argument(
        "-td",
        "--tools-dir",
        dest="tools_folder",
        metavar="<DIR>",
        default=None,
        help="Path to the FragPipe tools directory. If omitted, MStoCIRC2 infers '<fragpipe_root>/tools' from '--fragpipe-bin' when the installation uses the standard layout.",
    )
    grp.add_argument("-db", "--diann-bin", dest="diann_bin", metavar="<PATH>", default=None, help="Path to the DIA-NN executable. Recommended for DIA workflows; optional for DDA when discoverable from the tools directory or PATH.")
    grp.add_argument(
        "-py",
        "--python-bin",
        dest="python_bin",
        metavar="<PATH>",
        default="",
        help=(
            "FragPipe runtime Python. Default: the current MStoCIRC2 Python or 'MSTOCIRC2_FRAGPIPE_PYTHON'. "
            "For DIA workflows, it must include fragpipe-speclib, easypqp, and lxml."
        ),
    )


def _add_eval_predictor_inputs(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("Translation evaluation settings")
    grp.add_argument(
        "-dc", "--deepcip-path",
        dest="deepcip_path",
        metavar="<PATH>",
        default="",
        help="DeepCIP directory. Default: current environment lookup only. You can also set 'MSTOCIRC2_DEEPCIP_PATH'.",
    )
    grp.add_argument(
        "-dp", "--deepcip-python",
        dest="deepcip_python",
        metavar="<PATH>",
        default="",
        help="DeepCIP Python. Default: the current MStoCIRC2 Python or 'MSTOCIRC2_DEEPCIP_PYTHON'.",
    )


def _add_dea_inputs(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("DEA settings")
    grp.add_argument("-st", "--strategy", metavar="<NAME>", default="generic", help=f"DEA strategy preset controlling matrix interpretation, imputation, and statistical backend. Supported values: {', '.join(sorted(STRATEGY_REGISTRY.keys()))}. Default: generic.")
    grp.add_argument("-ap", "--use-adj-pvalue", action="store_true", help="Use the adjusted P-value column (BH FDR) instead of the raw P-value column when reporting significance. Default: False.")
    grp.add_argument("-org", "--organism", metavar="<ABBR>", default=_DEFAULT_ORGANISM, help=f"KEGG organism abbreviation used for enrichment, for example 'hsa', 'mmu', or 'ath'. Default: {_DEFAULT_ORGANISM} (Homo sapiens). Values must come from the bundled KEGG BR08606 registry. See {_KEGG_REFERENCE_URL}.")


def add_integrated_subparsers(subparsers: argparse._SubParsersAction) -> None:
    nonquant_epilog = help_block(
        "Example",
        [
            "  mstocirc2 nonquant -cs circ.fasta -cp canonical.fasta -mf manifest.tsv "
            "-fb /opt/fragpipe/bin/fragpipe -o run_nonquant",
        ],
    )
    non_quant = subparsers.add_parser(
        "nonquant",
        help="Run ORF prediction + FragPipe search + translation evaluation.",
        formatter_class=CLIHelpFormatter,
        description=branded_description(
            "Execute the non-quantitative integrated workflow: circRNA ORF prediction, FragPipe search, and "
            "translation evidence evaluation.",
            "mstocirc2 nonquant -cp <FILE> -mf <FILE> -fb <PATH> [options]",
        ),
        epilog=nonquant_epilog,
    )
    non_quant.set_defaults(runner=run_non_quant_pipeline)
    non_quant.add_argument(
        "-o",
        "--output-dir",
        metavar="<DIR>",
        default=None,
        help=(
            "Root output directory for the integrated workflow. If omitted, "
            "MStoCIRC2 creates 'MStoCIRC2_nonquant_YY-MM-DD.N' under the "
            "current working directory, then writes stage outputs under "
            "'orf', 'search', and 'eval' subdirectories."
        ),
    )
    non_quant.add_argument("-oo", "--orf-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    non_quant.add_argument("-so", "--search-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    non_quant.add_argument("-eo", "--eval-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    _add_orf_inputs(non_quant)
    _add_fragpipe_inputs(non_quant)
    _add_eval_predictor_inputs(non_quant)

    quant_epilog = help_block(
        "Example",
        [
            "  mstocirc2 quant -cs circ.fasta -cp canonical.fasta -mf manifest.tsv "
            "-fb /opt/fragpipe/bin/fragpipe -o run_quant",
        ],
    )
    quant = subparsers.add_parser(
        "quant",
        help="Run ORF prediction + quantitative FragPipe search + translation evaluation + DEA.",
        formatter_class=CLIHelpFormatter,
        description=branded_description(
            "Execute the quantitative integrated workflow: circRNA ORF prediction, quantitative FragPipe search, "
            "translation evidence evaluation, and downstream DEA.",
            "mstocirc2 quant -cp <FILE> -mf <FILE> -fb <PATH> [options]",
        ),
        epilog=quant_epilog,
    )
    quant.set_defaults(runner=run_quant_pipeline)
    quant.add_argument(
        "-o",
        "--output-dir",
        metavar="<DIR>",
        default=None,
        help=(
            "Root output directory for the integrated workflow. If omitted, "
            "MStoCIRC2 creates 'MStoCIRC2_quant_YY-MM-DD.N' under the current "
            "working directory, then writes stage outputs under 'orf', "
            "'search', 'eval', and 'dea' subdirectories."
        ),
    )
    quant.add_argument("-oo", "--orf-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    quant.add_argument("-so", "--search-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    quant.add_argument("-eo", "--eval-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    quant.add_argument("-do", "--dea-output-dir", metavar="<DIR>", default=None, help=argparse.SUPPRESS)
    _add_orf_inputs(quant)
    _add_fragpipe_inputs(quant)
    _add_eval_predictor_inputs(quant)
    _add_dea_inputs(quant)
