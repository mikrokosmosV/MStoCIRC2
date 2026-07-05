"""GO/KEGG enrichment. use_adj_p controls significance filtering and display."""

from __future__ import annotations

from typing import Dict, List
import logging
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .kegg_registry import (
    describe_kegg_organism,
    load_kegg_organism_map,
    normalize_kegg_organism,
)

log = logging.getLogger(__name__)
GO_CMAPS = {"go_BP": "Oranges", "go_CC": "Greens", "go_MF": "Blues"}
GO_LABEL = {"go_BP": "BP", "go_CC": "CC", "go_MF": "MF"}
KEGG_CMAP = "Purples"
P_CUTOFF = 0.05
_GSEAPY_ORGANISM_ALIASES = {
    "hsa": ["human", "homo sapiens", "hs"],
    "mmu": ["mouse", "mus musculus", "mm"],
    "rno": ["rat", "rattus norvegicus"],
    "dme": ["fly", "drosophila melanogaster", "drosophila"],
    "dre": ["zebrafish", "danio rerio", "fish"],
    "sce": ["yeast", "saccharomyces cerevisiae", "saccharomyces"],
    "cel": ["worm", "caenorhabditis elegans", "celegans", "nematode"],
}


def _organism_name_candidates(normalized: str) -> list[str]:
    candidates: list[str] = []
    for candidate in _GSEAPY_ORGANISM_ALIASES.get(normalized, []):
        if candidate not in candidates:
            candidates.append(candidate)

    record = load_kegg_organism_map().get(normalized)
    if record is None:
        return candidates

    raw_name = str(record["name"]).strip()
    scientific_name = raw_name.split("(", 1)[0].strip()
    common_name = raw_name.split("(", 1)[1].rstrip(")").strip() if "(" in raw_name else ""

    for candidate in (scientific_name, scientific_name.lower(), common_name.lower()):
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _resolve_gseapy_organism(gp, organism: str) -> tuple[str, list[str]]:
    normalized = normalize_kegg_organism(organism)
    attempted: list[str] = []
    last_error: Exception | None = None

    for candidate in _organism_name_candidates(normalized):
        if not candidate or candidate in attempted:
            continue
        attempted.append(candidate)
        try:
            libraries = gp.get_library_name(organism=candidate)
            log.info(
                "Resolved KEGG organism '%s' to gseapy organism '%s' for enrichment.",
                normalized,
                candidate,
            )
            return candidate, libraries
        except Exception as exc:
            last_error = exc
            continue

    attempted_text = ", ".join(attempted) if attempted else "<none>"
    raise RuntimeError(
        f"Unable to map validated KEGG organism '{normalized}' to a gseapy/enrichr-supported "
        f"organism string. Tried: {attempted_text}. Last error: {last_error}"
    ) from last_error


def _resolve_kegg_library(libraries: list[str], organism: str) -> str:
    normalized = normalize_kegg_organism(organism)
    kegg_libraries = [name for name in libraries if name.upper().startswith("KEGG_")]
    if kegg_libraries:
        return sorted(kegg_libraries, reverse=True)[0]

    raise RuntimeError(
        "No KEGG enrichment library is available through gseapy for the validated "
        f"organism '{normalized}'."
    )


def _parse_gene_ratio(value: str) -> float:
    try:
        numerator, denominator = str(value).split("/")
        return float(numerator) / float(denominator) if float(denominator) > 0 else np.nan
    except Exception:
        return np.nan


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Term": "term_description",
            "Overlap": "gene_ratio_str",
            "P-value": "p_value",
            "Adjusted P-value": "adj_p_value",
            "Genes": "gene_symbols",
        }
    )
    if "term_id" not in df.columns:
        df["term_id"] = df["term_description"]
    df["gene_count"] = df["gene_ratio_str"].astype(str).str.split("/").str[0].astype(int)
    df["bg_ratio"] = df["gene_ratio_str"].astype(str).str.split("/").str[-1]
    df["gene_ratio"] = df["gene_ratio_str"].apply(_parse_gene_ratio)
    return df[
        [
            "term_id",
            "term_description",
            "gene_count",
            "gene_ratio_str",
            "gene_ratio",
            "bg_ratio",
            "p_value",
            "adj_p_value",
            "gene_symbols",
        ]
    ]


def _p_col(use_adj_p: bool) -> str:
    return "adj_p_value" if use_adj_p else "p_value"


def _p_label(use_adj_p: bool) -> str:
    return "adj. P-value" if use_adj_p else "P-value"


def _prep_go_table(
    go_results: Dict[str, pd.DataFrame],
    top_per_ontology: int = 10,
    use_adj_p: bool = False,
) -> tuple[pd.DataFrame, list[int]]:
    p_col = _p_col(use_adj_p)
    pieces = []
    counts = []
    for kind in ("go_BP", "go_CC", "go_MF"):
        df = go_results.get(kind)
        if df is None or df.empty:
            counts.append(0)
            continue
        sig = df[df[p_col] < P_CUTOFF].sort_values(p_col).head(top_per_ontology).copy()
        if sig.empty:
            counts.append(0)
            continue
        sig["ontology"] = kind
        pieces.append(sig)
        counts.append(len(sig))
    if not pieces:
        return pd.DataFrame(), counts

    blocks = []
    for kind in ("go_MF", "go_CC", "go_BP"):
        sub = pd.concat([p[p["ontology"] == kind] for p in pieces], ignore_index=True)
        if sub.empty:
            continue
        blocks.append(sub.sort_values("gene_ratio", ascending=True))
    return pd.concat(blocks, ignore_index=True), counts


def _clean_label(row: pd.Series) -> str:
    text = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(row["term_description"]))
    return f"[{GO_LABEL[row['ontology']]}] {text[:65]}"


def _color_per_row(combined: pd.DataFrame, use_adj_p: bool = False) -> np.ndarray:
    p_col = _p_col(use_adj_p)
    colors = np.zeros((len(combined), 4))
    for ontology in ("go_BP", "go_CC", "go_MF"):
        mask = (combined["ontology"] == ontology).to_numpy()
        if not mask.any():
            continue
        vals = combined.loc[mask, p_col].to_numpy()
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            vmin, vmax = max(0, vmin - 0.01), vmax + 0.01
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(GO_CMAPS[ontology])
        mapped = 0.2 + 0.75 * (1.0 - norm(vals))
        colors[mask] = cmap(mapped)
    return colors


def _draw_separators(ax: plt.Axes, counts: list[int]) -> None:
    bp, cc, mf = counts
    running = 0
    for count in (mf, cc):
        if count == 0:
            continue
        running += count
        ax.axhline(y=running - 0.5, color="#888", lw=0.7, ls="--")


def _attach_go_colorbars(
    fig: plt.Figure,
    ax: plt.Axes,
    combined: pd.DataFrame,
    counts: list[int],
    use_adj_p: bool = False,
) -> None:
    p_col = _p_col(use_adj_p)
    p_label = _p_label(use_adj_p)
    bp, cc, mf = counts
    total = len(combined)
    if total == 0:
        return

    segments = []
    running = 0
    for ontology, count in [("go_MF", mf), ("go_CC", cc), ("go_BP", bp)]:
        if count > 0:
            segments.append((ontology, running, running + count))
            running += count

    fig.canvas.draw()
    bbox = ax.get_position()
    x_start = bbox.x1 + 0.008
    cb_width = 0.013

    for ontology, y_start, y_end in segments:
        fb = bbox.y0 + (y_start / total) * bbox.height
        ft = bbox.y0 + (y_end / total) * bbox.height
        cb_h = max(0.025, (ft - fb) * 0.8)
        cb_y = fb + (ft - fb - cb_h) / 2
        vals = combined.loc[combined["ontology"] == ontology, p_col].to_numpy()
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            vmin, vmax = max(0, vmin - 0.01), vmax + 0.01
        cax = fig.add_axes([x_start, cb_y, cb_width, cb_h])
        cmap = plt.get_cmap(GO_CMAPS[ontology])
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        tick_positions = np.linspace(0, 1, 4)
        tick_pvals = vmax - tick_positions * (vmax - vmin)
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([f"{v:.1e}" if v < 0.01 else f"{v:.3f}" for v in tick_pvals])
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(f"{GO_LABEL[ontology]}: {p_label}", fontsize=14, labelpad=6)


def _plot_go(
    combined: pd.DataFrame,
    counts: list[int],
    title_tag: str,
    out_prefix: str,
    plot_type: str,
    use_adj_p: bool = False,
) -> None:
    if combined.empty:
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.0})
    n = len(combined)
    y = np.arange(n)
    gene_ratio = combined["gene_ratio"].to_numpy()
    gene_count = combined["gene_count"].to_numpy()
    colors = _color_per_row(combined, use_adj_p)
    p_label = _p_label(use_adj_p)

    fig_h = max(5.0, 0.65 * n + 2.0)
    fig, ax = plt.subplots(figsize=(14.0, fig_h), dpi=150)

    if plot_type == "barplot":
        ax.barh(y, gene_ratio, color=colors, edgecolor="#333", lw=0.4)
        for yi, count in zip(y, gene_count):
            ax.text(gene_ratio[yi] + gene_ratio.max() * 0.012, yi, f"n={int(count)}", fontsize=15.0, va="center", color="#333")
    else:
        log_gc = np.log2(gene_count.clip(min=1) + 1)
        sizes = 40 + 120 * (log_gc - log_gc.min()) / (log_gc.max() - log_gc.min() + 1e-6)
        ax.scatter(gene_ratio, y, s=sizes, c=colors, edgecolors="#333", linewidths=0.5)
        size_values = np.unique(np.round(np.linspace(gene_count.min(), gene_count.max(), 3)).astype(int))
        handles = [
            plt.scatter(
                [],
                [],
                s=40 + 120 * (np.log2(v + 1) - log_gc.min()) / (log_gc.max() - log_gc.min() + 1e-6),
                c="#888",
                edgecolors="#333",
                lw=0.4,
                label=f"n={int(v)}",
            )
            for v in size_values
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=16, title="gene count", title_fontsize=16)

    labels = combined.apply(_clean_label, axis=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=17.0)
    ax.set_xlabel("Gene Ratio", fontsize=20)
    ax.set_title(f"{title_tag} — GO {plot_type} ({p_label}<0.05)", fontsize=24, pad=16)
    ax.grid(axis="x", ls=":", color="#CCC", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, gene_ratio.max() * 1.2)
    _draw_separators(ax, counts)

    plt.subplots_adjust(right=0.74)
    _attach_go_colorbars(fig, ax, combined, counts, use_adj_p)
    fig.savefig(f"{out_prefix}_GO_{plot_type}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}_GO_{plot_type}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_kegg(
    df: pd.DataFrame,
    title_tag: str,
    out_prefix: str,
    plot_type: str,
    top_n: int = 10,
    use_adj_p: bool = False,
) -> None:
    p_col = _p_col(use_adj_p)
    p_label = _p_label(use_adj_p)
    sig = df[df[p_col] < P_CUTOFF].sort_values(p_col).head(top_n)
    if sig.empty:
        return
    sig = sig.sort_values("gene_ratio", ascending=True)
    n = len(sig)
    y = np.arange(n)
    gene_ratio = sig["gene_ratio"].to_numpy()
    gene_count = sig["gene_count"].to_numpy()
    vals = sig[p_col].to_numpy()
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax == vmin:
        vmin, vmax = max(0, vmin - 0.01), vmax + 0.01
    cmap = plt.get_cmap(KEGG_CMAP)
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(0.2 + 0.75 * (1.0 - norm(vals)))

    fig_h = max(4.0, 0.52 * n + 1.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h), dpi=150)

    if plot_type == "barplot":
        ax.barh(y, gene_ratio, color=colors, edgecolor="#333", lw=0.4)
        for yi, count in zip(y, gene_count):
            ax.text(gene_ratio[yi] + gene_ratio.max() * 0.012, yi, f"n={int(count)}", fontsize=11.25, va="center", color="#333")
    else:
        log_gc = np.log2(gene_count.clip(min=1) + 1)
        sizes = 40 + 120 * (log_gc - log_gc.min()) / (log_gc.max() - log_gc.min() + 1e-6)
        ax.scatter(gene_ratio, y, s=sizes, c=colors, edgecolors="#333", linewidths=0.5)

    labels = sig["term_description"].astype(str).str.slice(0, 65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12.75)
    ax.set_xlabel("Gene Ratio", fontsize=15.0)
    ax.set_title(f"{title_tag} — KEGG {plot_type} ({p_label}<0.05)", fontsize=18.0, pad=12)
    ax.grid(axis="x", ls=":", color="#CCC", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, gene_ratio.max() * 1.2)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.75)
    tick_positions = np.linspace(0, 1, 4)
    tick_pvals = vmax - tick_positions * (vmax - vmin)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f"{v:.1e}" if v < 0.01 else f"{v:.3f}" for v in tick_pvals])
    cbar.ax.tick_params(labelsize=10.5)
    cbar.set_label(p_label, fontsize=13.5)

    plt.tight_layout()
    fig.savefig(f"{out_prefix}_KEGG_{plot_type}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}_KEGG_{plot_type}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_enrichment(
    genes: List[str],
    organism: str,
    out_prefix: str,
    use_adj_p: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run GO/KEGG enrichment using the original publication-grade plotting chain."""
    try:
        import gseapy as gp
    except Exception:
        log.warning("gseapy not available")
        return {}

    organism = normalize_kegg_organism(organism)
    organism_label = describe_kegg_organism(organism)
    log.info("Enrichment organism: %s", organism_label)
    gseapy_organism, libraries = _resolve_gseapy_organism(gp, organism)
    kegg_lib = _resolve_kegg_library(libraries, organism)
    libs = {
        "go_BP": "GO_Biological_Process_2023",
        "go_CC": "GO_Cellular_Component_2023",
        "go_MF": "GO_Molecular_Function_2023",
        "kegg": kegg_lib,
    }
    title_tag = os.path.basename(out_prefix)
    results: Dict[str, pd.DataFrame] = {}
    for kind, library in libs.items():
        tsv_path = f"{out_prefix}_{kind}.tsv"
        try:
            enr = gp.enrichr(
                gene_list=list(genes),
                gene_sets=library,
                organism=gseapy_organism,
                outdir=None,
                cutoff=0.5,
                no_plot=True,
            )
            df = enr.results if hasattr(enr, "results") else None
            if df is None or df.empty:
                pd.DataFrame().to_csv(tsv_path, sep="\t", index=False)
                continue
            df = _standardise(df)
            df.to_csv(tsv_path, sep="\t", index=False)
            results[kind] = df
        except Exception as exc:
            log.warning("Enrichment %s failed: %s", kind, exc)
            pd.DataFrame().to_csv(tsv_path, sep="\t", index=False)

    go_sub = {k: v for k, v in results.items() if k.startswith("go_")}
    combined, counts = _prep_go_table(go_sub, use_adj_p=use_adj_p)
    if not combined.empty:
        _plot_go(combined, counts, title_tag, out_prefix, "barplot", use_adj_p=use_adj_p)
        _plot_go(combined, counts, title_tag, out_prefix, "dotplot", use_adj_p=use_adj_p)
    else:
        log.info("[%s] no significant GO terms; skip GO plots", title_tag)

    if "kegg" in results and not results["kegg"].empty:
        _plot_kegg(results["kegg"], title_tag, out_prefix, "barplot", use_adj_p=use_adj_p)
        _plot_kegg(results["kegg"], title_tag, out_prefix, "dotplot", use_adj_p=use_adj_p)
    return results
