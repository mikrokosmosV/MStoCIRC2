"""Command runner for the dea subcommand – orchestrates downstream analysis."""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ..cli_ui import CLIHelpFormatter, branded_description, help_block, join_blocks
from ..core import CLIUsageError, DependencyError, ensure_directory, make_default_module_output_dir
from .r_runtime_bridge import RSCRIPT
from .workflow_strategy import resolve_strategy, STRATEGY_REGISTRY
from .expression_io import (
    parse_circrna_reference,
    parse_design,
    match_samples_to_columns,
    load_peptide_matrix,
)
from .homology_refining import filter_peptides, STRIPPED_SEQ_CANDIDATES, PROTEIN_ASSIGN_CANDIDATES, _first_present
from .intensity_aggregator import rollup_peptides_to_proteins
from .missing_value_imputer import impute
from .statistical_dea import run_dea
from .mrna_coexpression import correlate_circrna_vs_mrna
from .functional_enrichment import run_enrichment
from .expression_visualizer import volcano_plot
from .kegg_registry import describe_kegg_organism, normalize_kegg_organism

log = logging.getLogger("circrna_dea")

# ---------------------------------------------------------------------------
# Internal defaults kept stable across DEA strategy presets
# ---------------------------------------------------------------------------
_CORRELATION_METHOD = "pearson"
_TOP_PROXY_MRNA = 100
_TOP_LABEL = 10
_ROTS_B = 200
_ROTS_K = 200
_SEED = 42
_DEFAULT_FDR_THRESHOLD = 0.05
_DEFAULT_LOG2FC_THRESHOLD = 1.0
_DEFAULT_ORGANISM = "hsa"
_KEGG_REFERENCE_URL = "https://www.kegg.jp/kegg/tables/br08606.html"


class DifferentialExpressionError(Exception):
    """Raised when the DEA pipeline cannot proceed."""


def add_dea_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the 'dea' subparser with essential arguments only."""
    epilog = join_blocks(
        help_block(
            "Input Rules",
            [
                "  - '--peptide-matrix' must contain quantification columns matching the design sample names.",
                "  - '--circrna-reference' must come from the same ORF/search universe as the peptide matrix.",
                "  - Current DEA implementation requires exactly two experimental conditions.",
            ],
        ),
        help_block(
            "Examples",
            [
                "  mstocirc2 dea -pm peptide_matrix.tsv -cr circ_predict.txt -de design.txt -st generic -o out_dea",
            ],
        ),
    )
    p = subparsers.add_parser(
        "dea",
        help="Downstream differential expression, correlation, and enrichment analysis",
        formatter_class=CLIHelpFormatter,
        description=branded_description(
            "Run circRNA-aware quantitative roll-up, imputation, differential expression analysis, volcano plotting, "
            "optional circRNA-mRNA correlation, and enrichment analysis.",
            "mstocirc2 dea -pm <FILE> -cr <FILE> -de <FILE> [options]",
        ),
        epilog=epilog,
    )
    p.set_defaults(runner=run_differential_analysis)
    p.add_argument("-pm", "--peptide-matrix", metavar="<FILE>", required=True, help="Path to the peptide quantification matrix.")
    p.add_argument("-cr", "--circrna-reference", metavar="<FILE>", required=True, help="Path to the circRNA reference table, for example 'circ_predict.txt'.")
    p.add_argument("-de", "--design", metavar="<FILE>", required=True, help="Path to the sample design or metadata table (.txt or .tsv).")
    p.add_argument(
        "-o",
        "--output-dir",
        metavar="<DIR>",
        default=None,
        help=(
            "Directory for DEA outputs, plots, and metadata. If omitted, "
            "MStoCIRC2 creates 'MStoCIRC2_dea_YY-MM-DD.N' under the current "
            "working directory."
        ),
    )
    p.add_argument(
        "-st", "--strategy",
        metavar="<NAME>",
        default="generic",
        choices=sorted(STRATEGY_REGISTRY.keys()),
        help=(
            "DEA strategy preset controlling matrix interpretation, imputation, and statistical backend. "
            f"Supported values: {', '.join(sorted(STRATEGY_REGISTRY.keys()))}. Default: generic."
        ),
    )
    p.add_argument(
        "-ap", "--use-adj-pvalue",
        action="store_true",
        help="Use the adjusted P-value column (BH FDR) instead of the raw P-value column when reporting significance. Default: False.",
    )
    p.add_argument(
        "-org",
        "--organism",
        metavar="<ABBR>",
        default=_DEFAULT_ORGANISM,
        help=(
            "KEGG organism abbreviation used for enrichment, for example 'hsa', 'mmu', or 'ath'. "
            f"Default: {_DEFAULT_ORGANISM} (Homo sapiens). Values must come from the bundled KEGG BR08606 registry. "
            f"See {_KEGG_REFERENCE_URL}."
        ),
    )
    return p


def _prepare_outdir(out: Path) -> None:
    ensure_directory(out)


def run_differential_analysis(args: argparse.Namespace) -> int:
    """Execute the full DEA and enrichment pipeline."""
    args.fdr_threshold = _DEFAULT_FDR_THRESHOLD
    args.log2fc_threshold = _DEFAULT_LOG2FC_THRESHOLD
    try:
        args.organism = normalize_kegg_organism(args.organism)
    except ValueError as exc:
        raise CLIUsageError(str(exc)) from exc
    if RSCRIPT is None:
        raise DependencyError(
            "Rscript not found. `mstocirc2 dea` requires an R installation plus the `Rscript` "
            "launcher. Configure `Rscript` on PATH or set `MSTOCIRC2_RSCRIPT`, `RSCRIPT`, or "
            "`R_HOME` before running DEA."
        )

    if args.output_dir and str(args.output_dir).strip():
        out = Path(args.output_dir).expanduser()
        _prepare_outdir(out)
    else:
        out = make_default_module_output_dir("dea")
    args.output_dir = str(out)
    start_ts = time.time()

    strategy = resolve_strategy(args.strategy)
    log.info("Strategy '%s' -> %s", args.strategy, strategy.as_dict())
    log.info("Enrichment organism: %s", describe_kegg_organism(args.organism))

    design = parse_design(args.design)
    conds = design["condition"].unique().tolist()
    log.info("  design: %d samples, conditions: %s", len(design), conds)

    peptide_df, matrix_fmt = load_peptide_matrix(args.peptide_matrix, design["sample"].tolist())
    log.info("  peptide matrix: %s (format: %s)", peptide_df.shape, matrix_fmt)

    _, circ_ref_peptides, circ_orf_ids = parse_circrna_reference(args.circrna_reference)
    log.info("  circRef_Set: %d peptides, %d ORF IDs", len(circ_ref_peptides), len(circ_orf_ids))

    sample_cols = match_samples_to_columns(design["sample"].tolist(), list(peptide_df.columns))
    log.info("  %d samples matched", len(sample_cols))

    filtered, filter_report, filter_summary = filter_peptides(
        peptide_df, circ_ref_peptides, circ_orf_ids
    )
    filter_report.to_csv(out / "peptide_filter_report.tsv", sep="\t", index=False)
    log.info("Filter: %s", filter_summary)
    if filter_summary.get("circrna_peptides_retained", 0) <= 0:
        raise DifferentialExpressionError(
            "No circRNA peptides remained after circRNA reference matching and homology filtering. "
            "Check that the peptide matrix and circRNA reference come from the same search/ORF universe."
        )

    # Pre-filter counts
    seq_col_pre = _first_present(peptide_df, STRIPPED_SEQ_CANDIDATES)
    prot_col_pre = _first_present(peptide_df, PROTEIN_ASSIGN_CANDIDATES)
    pre_filter_counts = (
        peptide_df[[prot_col_pre, seq_col_pre]]
        .dropna()
        .astype(str)
        .groupby(prot_col_pre)[seq_col_pre]
        .nunique()
        .rename("input_peptide_count")
    )
    pre_filter_counts.index.name = "protein_id"

    # prot2gene
    prot2gene = {}
    gene_col = None
    for cand in ("Genes", "Gene.Names", "Gene Name", "Gene", "gene"):
        if cand in filtered.columns:
            gene_col = cand
            break
    if gene_col:
        prot_col = _first_present(filtered, PROTEIN_ASSIGN_CANDIDATES)
        for pid, gene in zip(
            filtered[prot_col].astype(str), filtered[gene_col].astype(str)
        ):
            pid = pid.strip()
            gene = gene.strip()
            if pid and gene and pid not in prot2gene and gene.lower() != "nan":
                prot2gene[pid] = gene.split(";")[0]

    # Roll-up
    log.info(
        "Roll-up: filtered rows=%d, sample columns=%d, matrix_type=%s",
        len(filtered),
        len(sample_cols),
        strategy.expression_matrix_type,
    )
    protein_matrix, rollup_summary = rollup_peptides_to_proteins(
        filtered,
        sample_cols,
        strategy.expression_matrix_type,
        circ_orf_ids,
        pre_filter_counts,
    )
    log.info(
        "Roll-up complete: protein rows=%d, summary rows=%d",
        len(protein_matrix),
        len(rollup_summary),
    )
    protein_matrix.to_csv(out / "protein_quant.tsv", sep="\t")
    rollup_summary.to_csv(out / "rollup_summary.tsv", sep="\t", index=False)
    log.info("Protein matrix: %s", protein_matrix.shape)

    # Imputation
    if protein_matrix.empty:
        raise DifferentialExpressionError("Protein matrix is empty – cannot proceed.")
    protein_imputed = impute(protein_matrix, strategy.imputation)
    protein_imputed.to_csv(out / "protein_quant_imputed.tsv", sep="\t")

    # DEA
    groups_sorted = sorted(conds)
    if len(groups_sorted) != 2:
        raise DifferentialExpressionError("DEA currently requires exactly two conditions.")
    group_a, group_b = groups_sorted
    contrast = f"{group_a}_vs_{group_b}"
    log.info("Contrast: %s", contrast)
    if design["condition"].value_counts().min() < 2:
        raise DifferentialExpressionError("Each group needs at least 2 samples.")

    peptide_counts = rollup_summary.set_index("protein_id")["retained_peptide_count"]
    is_circrna = rollup_summary.set_index("protein_id")["is_circrna"]
    dea_df = run_dea(
        method=strategy.dea_method,
        protein_matrix_raw=protein_matrix,
        protein_matrix_imputed=protein_imputed,
        design=design,
        contrast=contrast,
        peptide_counts=peptide_counts,
        is_circrna=is_circrna,
        rots_b=_ROTS_B,
        rots_k=_ROTS_K,
        seed=_SEED,
    )
    dea_df.to_csv(out / "dea_results.tsv", sep="\t", index=False)

    p_col = "adj.P.Val" if args.use_adj_pvalue else "P.Value"
    sig_count = (dea_df[p_col] < args.fdr_threshold).sum()
    log.info("DEA: %d rows, %d below %s threshold", len(dea_df), sig_count, p_col)

    # Volcano
    vstats = volcano_plot(
        dea_df,
        str(out / "volcano_plot.pdf"),
        str(out / "volcano_plot.png"),
        fdr_threshold=args.fdr_threshold,
        log2fc_threshold=args.log2fc_threshold,
        top_label=_TOP_LABEL,
        group_a=group_a,
        group_b=group_b,
        p_col=p_col,
    )
    log.info("Volcano: %s", vstats)

    circ_ids_in_matrix = [p for p in protein_imputed.index if is_circrna.get(p, False)]
    mrna_ids_in_matrix = [p for p in protein_imputed.index if not is_circrna.get(p, False)]
    log.info("circRNA: %d, mRNA: %d", len(circ_ids_in_matrix), len(mrna_ids_in_matrix))

    sig = dea_df[
        (dea_df[p_col] < args.fdr_threshold)
        & (dea_df["log2FC"].abs() > args.log2fc_threshold)
        & dea_df["is_circrna"]
    ]
    de_circ_ids = sig["protein_id"].tolist()
    log.info("DE circRNA: %d", len(de_circ_ids))

    per_group_samples = {
        group_a: [s for s in design.loc[design["condition"] == group_a, "sample"] if s in protein_imputed.columns],
        group_b: [s for s in design.loc[design["condition"] == group_b, "sample"] if s in protein_imputed.columns],
    }

    enrichment_summary = {}
    for grp_name, sign_fn in ((group_a, lambda v: v > 0), (group_b, lambda v: v < 0)):
        subset = sig[sig["log2FC"].apply(sign_fn)]
        if subset.empty:
            log.info("[%s] no upreg circRNA; skip", grp_name)
            continue
        grp_samples = per_group_samples[grp_name]
        n_grp = len(grp_samples)
        log.info("[%s] upreg circRNA: %d, samples: %d", grp_name, len(subset), n_grp)

        grp_corr = correlate_circrna_vs_mrna(
            protein_imputed,
            list(subset["protein_id"]),
            mrna_ids_in_matrix,
            method=_CORRELATION_METHOD,
            sample_subset=grp_samples,
            top_k_per_circrna=_TOP_PROXY_MRNA if n_grp < 3 else None,
            p_threshold=args.fdr_threshold,
            use_adj_p=args.use_adj_pvalue,
        )
        grp_corr.to_csv(out / f"correlation_{grp_name}.tsv", sep="\t", index=False)
        if grp_corr.empty:
            log.warning("[%s] no correlated mRNAs", grp_name)
            continue

        if n_grp >= 3:
            per_circ_top = (
                grp_corr
                .assign(abs_corr=grp_corr["correlation"].abs())
                .sort_values(["circrna_id", "abs_corr"], ascending=[True, False])
                .groupby("circrna_id").head(_TOP_PROXY_MRNA)
            )
            proxy_mrnas = per_circ_top["mrna_id"].drop_duplicates().tolist()
        else:
            per_circ_top = (
                grp_corr
                .assign(abs_score=grp_corr["rank_score"].abs())
                .sort_values(["circrna_id", "abs_score"], ascending=[True, False])
                .groupby("circrna_id").head(_TOP_PROXY_MRNA)
            )
            proxy_mrnas = per_circ_top["mrna_id"].drop_duplicates().tolist()

        enrichment_summary[grp_name] = {"upreg_circ": len(subset), "proxy_count": len(proxy_mrnas)}
        gene_list = list(
            dict.fromkeys(
                prot2gene.get(m.strip(), m)
                for m in proxy_mrnas
                if prot2gene.get(m.strip(), m).lower() not in ("", "nan")
            )
        )
        pd.DataFrame(
            {
                "mrna_id": proxy_mrnas[: len(gene_list)],
                "gene_symbol": gene_list[: len(proxy_mrnas)],
            }
        ).to_csv(out / f"proxy_genes_{grp_name}.tsv", sep="\t", index=False)

        if len(gene_list) >= 5:
            log.info("[%s] enrichment on %d genes", grp_name, len(gene_list))
            try:
                run_enrichment(
                    gene_list,
                    args.organism,
                    str(out / f"enrichment_{grp_name}"),
                    use_adj_p=args.use_adj_pvalue,
                )
            except Exception as e:
                log.warning("[%s] enrichment failed: %s", grp_name, e)

    metadata = {
        "tool": "MStoCIRC2",
        "module": "differential_expression",
        "strategy": args.strategy,
        "organism": args.organism,
        "organism_label": describe_kegg_organism(args.organism),
        "strategy_detail": strategy.as_dict(),
        "contrast": contrast,
        "filter_summary": filter_summary,
        "volcano_stats": vstats,
        "enrichment_summary": enrichment_summary,
        "elapsed_sec": round(time.time() - start_ts, 2),
    }
    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    log.info("Done. Output: %s", out.resolve())
    return 0
