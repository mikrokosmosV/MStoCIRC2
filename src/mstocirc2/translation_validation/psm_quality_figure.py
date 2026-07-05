"""Plot publication-style spectral-quality distributions for PSM groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_SPECS = [
    ("Decoy PSM", "#D9D9D9"),
    ("mRNA Target PSM", "#9A9A9A"),
    ("Shared circRNA PSM", "#111111"),
    ("MStoCIRC-only PSM", "#019092"),
    ("MStoCIRC2-only PSM", "#0095FF"),
]

METRICS = [
    ("xcorr_like", "XCorr-like Score"),
    ("spectral_angle", "Spectral Angle"),
]


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def _standardize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _pick_peptide_key(df: pd.DataFrame) -> str:
    for candidate in ("peptide_plain", "plain_peptide", "peptide", "Peptide"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No peptide identifier column found.")


def _collapse_shared_circrna(
    mstocirc_df: pd.DataFrame,
    mstocirc2_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mstocirc_df = mstocirc_df.copy()
    mstocirc2_df = mstocirc2_df.copy()

    peptide_key = _pick_peptide_key(mstocirc_df)
    if _pick_peptide_key(mstocirc2_df) != peptide_key:
        raise KeyError("circRNA input files do not expose a consistent peptide key.")

    shared_peptides = set(mstocirc_df[peptide_key].dropna()) & set(mstocirc2_df[peptide_key].dropna())

    shared_candidates = pd.concat(
        [
            mstocirc_df[mstocirc_df[peptide_key].isin(shared_peptides)],
            mstocirc2_df[mstocirc2_df[peptide_key].isin(shared_peptides)],
        ],
        ignore_index=True,
    )
    shared_candidates = _standardize_numeric(shared_candidates, ["native_hyperscore"])
    shared = (
        shared_candidates.sort_values(
            by=[peptide_key, "native_hyperscore", "software"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=[peptide_key], keep="first")
        .copy()
    )

    mstocirc_only = mstocirc_df[~mstocirc_df[peptide_key].isin(shared_peptides)].copy()
    mstocirc2_only = mstocirc2_df[~mstocirc2_df[peptide_key].isin(shared_peptides)].copy()
    return shared, mstocirc_only, mstocirc2_only


def build_plot_dataframe(mrna_path: Path, mstocirc_path: Path, mstocirc2_path: Path) -> pd.DataFrame:
    mrna = _standardize_numeric(_read_table(mrna_path), [metric for metric, _ in METRICS])
    mstocirc = _standardize_numeric(
        _read_table(mstocirc_path), [metric for metric, _ in METRICS] + ["native_hyperscore"]
    )
    mstocirc2 = _standardize_numeric(
        _read_table(mstocirc2_path), [metric for metric, _ in METRICS] + ["native_hyperscore"]
    )

    shared, mstocirc_only, mstocirc2_only = _collapse_shared_circrna(mstocirc, mstocirc2)

    groups = {
        "Decoy PSM": mrna[mrna["source_type"] == "decoy"].copy(),
        "mRNA Target PSM": mrna[mrna["source_type"] == "mRNA"].copy(),
        "Shared circRNA PSM": shared.copy(),
        "MStoCIRC-only PSM": mstocirc_only.copy(),
        "MStoCIRC2-only PSM": mstocirc2_only.copy(),
    }

    records: list[pd.DataFrame] = []
    for label, _ in GROUP_SPECS:
        frame = groups[label]
        for metric, metric_label in METRICS:
            subset = frame[[metric]].copy()
            subset.columns = ["value"]
            subset["group"] = label
            subset["metric"] = metric_label
            records.append(subset.dropna(subset=["value"]))

    return pd.concat(records, ignore_index=True)


def summarize_plot_dataframe(plot_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        plot_df.groupby(["metric", "group"])["value"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    return summary


def _draw_metric_panel(ax: plt.Axes, panel_df: pd.DataFrame, metric_label: str, rng: np.random.Generator) -> None:
    positions = np.arange(1, len(GROUP_SPECS) + 1)
    datasets = []
    xticklabels = []

    for label, _ in GROUP_SPECS:
        values = panel_df.loc[panel_df["group"] == label, "value"].to_numpy()
        datasets.append(values)
        xticklabels.append(f"{label}\n(n={len(values)})")

    violin = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.86,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, (_, color) in zip(violin["bodies"], GROUP_SPECS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22 if color != "#111111" else 0.18)

    box = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "edgecolor": "#444444"},
    )

    for patch, (_, color) in zip(box["boxes"], GROUP_SPECS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7 if color != "#111111" else 0.85)

    for pos, (label, color), values in zip(positions, GROUP_SPECS, datasets):
        if len(values) == 0:
            continue
        sample_size = min(450, len(values))
        sampled = values if len(values) <= sample_size else rng.choice(values, size=sample_size, replace=False)
        jitter = rng.uniform(-0.14, 0.14, size=len(sampled))
        ax.scatter(
            np.full(len(sampled), pos) + jitter,
            sampled,
            s=10,
            alpha=0.20 if label != "Shared circRNA PSM" else 0.24,
            color=color,
            edgecolors="none",
            zorder=3,
        )

    ax.set_title(metric_label, fontsize=13, pad=10, weight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#D0D0D0")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", labelsize=10)


def plot_psm_quality(
    mrna_path: Path,
    mstocirc_path: Path,
    mstocirc2_path: Path,
    output_prefix: Path,
) -> dict[str, Path]:
    plot_df = build_plot_dataframe(mrna_path, mstocirc_path, mstocirc2_path)
    summary = summarize_plot_dataframe(plot_df)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9), dpi=300, constrained_layout=True)
    rng = np.random.default_rng(42)

    for panel_idx, ((_, metric_label), ax) in enumerate(zip(METRICS, axes)):
        panel_df = plot_df[plot_df["metric"] == metric_label]
        _draw_metric_panel(ax, panel_df, metric_label, rng)
        ax.text(
            0.01,
            0.98,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            weight="bold",
        )

    axes[0].set_ylabel("Score", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    fig.suptitle(
        "Spectral quality distributions across background and circRNA-derived PSM groups",
        fontsize=15,
        weight="bold",
        y=1.02,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    summary_path = output_prefix.with_name(f"{output_prefix.stem}_summary.tsv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.6f")
    return {"png": png_path, "svg": svg_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot spectral-quality distributions for decoy, mRNA, and circRNA PSM groups."
    )
    parser.add_argument("--mrna", type=Path, required=True, help="Path to mRNA best-PSM table")
    parser.add_argument("--mstocirc", type=Path, required=True, help="Path to circRNA_MStoCIRC.txt")
    parser.add_argument("--mstocirc2", type=Path, required=True, help="Path to circRNA_MStoCIRC2.txt")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix without extension, for example ./results/psm_quality_distribution",
    )
    args = parser.parse_args()

    outputs = plot_psm_quality(args.mrna, args.mstocirc, args.mstocirc2, args.output_prefix)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
