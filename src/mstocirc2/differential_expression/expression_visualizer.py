"""Volcano plotting utilities with circRNA-specific styling."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text

    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False

GRAY, UP_RED, DN_BLUE, CIRC_EDGE = "#B5B5B5", "#D7263D", "#1B6CA8", "#222222"


def _short_label(pid: str) -> str:
    return pid.split("(", 1)[0] if "(" in pid else pid


def volcano_plot(
    dea_df: pd.DataFrame,
    out_pdf: str,
    out_png: str,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    top_label: int = 10,
    group_a: str | None = None,
    group_b: str | None = None,
    p_col: str = "P.Value",
) -> dict[str, int]:
    """Render a volcano plot and return summary counts."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.1})
    df = dea_df.dropna(subset=["log2FC", p_col]).copy()
    if df.empty:
        raise ValueError(
            "Volcano plotting requires at least one row with both 'log2FC' and "
            f"'{p_col}' populated."
        )

    df["-log10p"] = -np.log10(df[p_col].clip(1e-300).astype(float))
    df["is_sig"] = (df[p_col] < fdr_threshold) & (df["log2FC"].abs() > log2fc_threshold)
    total_dep = int(df["is_sig"].sum())
    circ_dep = int((df["is_sig"] & df["is_circrna"]).sum())

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=150)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#CCC", zorder=0)

    # Non-sig mRNA
    ns_m = df[~df["is_sig"] & ~df["is_circrna"]]
    ax.scatter(
        ns_m["log2FC"],
        ns_m["-log10p"],
        c=GRAY,
        alpha=0.45,
        s=26,
        marker="o",
        linewidths=0,
        zorder=1,
    )

    # Sig mRNA
    sig_m = df[df["is_sig"] & ~df["is_circrna"]]
    for sub, color, label in [
        (sig_m[sig_m.log2FC > 0], UP_RED, f"Up in {group_a}"),
        (sig_m[sig_m.log2FC < 0], DN_BLUE, f"Up in {group_b}"),
    ]:
        if len(sub):
            ax.scatter(
                sub["log2FC"],
                sub["-log10p"],
                c=color,
                alpha=0.82,
                s=40,
                marker="o",
                linewidths=0,
                zorder=2,
                label=label,
            )

    # Non-sig circRNA
    ns_c = df[~df["is_sig"] & df["is_circrna"]]
    if len(ns_c):
        ax.scatter(
            ns_c["log2FC"],
            ns_c["-log10p"],
            c=GRAY,
            alpha=0.75,
            s=70,
            marker="^",
            edgecolors=CIRC_EDGE,
            linewidths=0.6,
            zorder=3,
        )

    # Sig circRNA
    sig_c = df[df["is_sig"] & df["is_circrna"]]
    for sub, color, label in [
        (sig_c[sig_c.log2FC > 0], UP_RED, f"circRNA up {group_a}"),
        (sig_c[sig_c.log2FC < 0], DN_BLUE, f"circRNA up {group_b}"),
    ]:
        if len(sub):
            ax.scatter(
                sub["log2FC"],
                sub["-log10p"],
                c=color,
                alpha=0.95,
                s=150,
                marker="^",
                edgecolors=CIRC_EDGE,
                linewidths=1.0,
                zorder=4,
                label=label,
            )

    # Threshold lines
    ax.axvline(log2fc_threshold, color="#666", ls="--", lw=0.9, zorder=0.5)
    ax.axvline(-log2fc_threshold, color="#666", ls="--", lw=0.9, zorder=0.5)
    ax.axhline(-np.log10(fdr_threshold), color="#666", ls="--", lw=0.9, zorder=0.5)

    # Labels for top circRNAs (improved: use adjustText with force parameters)
    circ_sig = df[df["is_sig"] & df["is_circrna"]].sort_values(p_col)
    if len(circ_sig):
        to_label = circ_sig.head(top_label)
        texts = []
        for _, r in to_label.iterrows():
            texts.append(ax.text(
                r["log2FC"], r["-log10p"],
                _short_label(str(r["protein_id"])),
                fontsize=8, fontweight="bold", ha="left", va="bottom", color="#111",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            ))
        if _HAS_ADJUST and texts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    adjust_text(
                        texts,
                        ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#555", lw=0.5),
                    )
                except Exception:
                    pass  # If adjustText fails for any reason, labels stay in place

    # Axis limits
    xpad = max(0.5, 0.08 * (df["log2FC"].max() - df["log2FC"].min()))
    ax.set_xlim(df["log2FC"].min() - xpad, df["log2FC"].max() + xpad)
    ymax = max(df["-log10p"].max() * 1.08, -np.log10(fdr_threshold) * 1.5)
    ax.set_ylim(-0.02 * ymax, ymax)

    # Labels
    p_display = "adj. P-value" if p_col == "adj.P.Val" else "P-value"
    ax.set_xlabel(r"$\log_2$ Fold Change", fontsize=12)
    ax.set_ylabel(f"$-\\log_{{10}}$ ({p_display})", fontsize=12)
    ax.set_title(
        f"Volcano: {group_a} vs {group_b}" if group_a else "Volcano",
        fontsize=13,
        pad=10,
    )

    # Corner annotation
    ax.text(
        0.985,
        0.985,
        f"Total DEPs: {total_dep}\nCircRNA DEPs: {circ_dep}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#888", lw=0.7),
    )

    if any(df["is_sig"]) or any(df["is_circrna"]):
        ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9, edgecolor="#AAA")

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"total_deps": total_dep, "circrna_deps": circ_dep}
