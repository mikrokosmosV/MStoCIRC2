"""Generate multiple candidate publication-style figures for PSM quality comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

# Allow direct script execution from the repository root without installing the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstocirc2.translation_validation.psm_quality_figure import (
    _collapse_shared_circrna,
    _read_table,
    _standardize_numeric,
)


METRIC_X = "xcorr_like"
METRIC_Y = "spectral_angle"


@dataclass(frozen=True)
class GroupStyle:
    label: str
    color: str
    short: str


GROUPS = [
    GroupStyle("Decoy PSM", "#D9D9D9", "Decoy"),
    GroupStyle("mRNA Target PSM", "#8E8E8E", "mRNA"),
    GroupStyle("Shared circRNA PSM", "#111111", "Shared"),
    GroupStyle("MStoCIRC-only PSM", "#019092", "MStoCIRC"),
    GroupStyle("MStoCIRC2-only PSM", "#0095FF", "MStoCIRC2"),
]

RGBA_SCATTER = {style.label: mcolors.to_rgba(style.color, alpha=0.95) for style in GROUPS}
RGBA_CLOUD = {
    style.label: mcolors.to_rgba(style.color, alpha=0.42 if style.label != "Shared circRNA PSM" else 0.55)
    for style in GROUPS
}


def load_group_tables(mrna_path: Path, mstocirc_path: Path, mstocirc2_path: Path) -> dict[str, pd.DataFrame]:
    mrna = _standardize_numeric(_read_table(mrna_path), [METRIC_X, METRIC_Y, "native_hyperscore"])
    mstocirc = _standardize_numeric(_read_table(mstocirc_path), [METRIC_X, METRIC_Y, "native_hyperscore"])
    mstocirc2 = _standardize_numeric(_read_table(mstocirc2_path), [METRIC_X, METRIC_Y, "native_hyperscore"])

    shared, mstocirc_only, mstocirc2_only = _collapse_shared_circrna(mstocirc, mstocirc2)

    groups = {
        "Decoy PSM": mrna[mrna["source_type"] == "decoy"].copy(),
        "mRNA Target PSM": mrna[mrna["source_type"] == "mRNA"].copy(),
        "Shared circRNA PSM": shared.copy(),
        "MStoCIRC-only PSM": mstocirc_only.copy(),
        "MStoCIRC2-only PSM": mstocirc2_only.copy(),
    }

    for key in groups:
        groups[key] = groups[key].dropna(subset=[METRIC_X, METRIC_Y]).copy()

    return groups


def to_long_df(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for style in GROUPS:
        df = groups[style.label].copy()
        df["group"] = style.label
        records.append(df[["group", METRIC_X, METRIC_Y]])
    long_df = pd.concat(records, ignore_index=True)
    return long_df


def compute_quality_index(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    x_med = df[METRIC_X].median()
    y_med = df[METRIC_Y].median()
    x_iqr = df[METRIC_X].quantile(0.75) - df[METRIC_X].quantile(0.25)
    y_iqr = df[METRIC_Y].quantile(0.75) - df[METRIC_Y].quantile(0.25)
    x_iqr = x_iqr if x_iqr > 0 else 1.0
    y_iqr = y_iqr if y_iqr > 0 else 1.0
    df["quality_index_raw"] = 0.5 * ((df[METRIC_X] - x_med) / x_iqr + (df[METRIC_Y] - y_med) / y_iqr)
    ranks = df["quality_index_raw"].rank(method="average", pct=True)
    df["quality_index_pct"] = 100.0 * ranks
    return df


def _set_base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _axis_limits(long_df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x_lo = max(-0.02, long_df[METRIC_X].quantile(0.001) - 0.03)
    x_hi = long_df[METRIC_X].quantile(0.995) + 0.05
    y_lo = max(-0.02, long_df[METRIC_Y].quantile(0.001) - 0.03)
    y_hi = long_df[METRIC_Y].quantile(0.995) + 0.05
    return (x_lo, x_hi), (y_lo, y_hi)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def _format_pvalue(p_value: float) -> str:
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}"


def _ks_max_point(reference: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.sort(np.asarray(values, dtype=float))
    support = np.unique(np.concatenate([reference, values]))
    ref_cdf = np.searchsorted(reference, support, side="right") / len(reference)
    val_cdf = np.searchsorted(values, support, side="right") / len(values)
    diff = np.abs(ref_cdf - val_cdf)
    idx = int(np.argmax(diff))
    ks_stat, ks_p = ks_2samp(reference, values)
    return float(support[idx]), float(ks_stat), float(ks_p)


def candidate_facet_density(groups: dict[str, pd.DataFrame], out_prefix: Path) -> Path:
    _set_base_style()
    long_df = to_long_df(groups)
    (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(long_df)
    x_grid = np.linspace(x_lo, x_hi, 180)
    y_grid = np.linspace(y_lo, y_hi, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.2), dpi=300, constrained_layout=True)
    axes = axes.flatten()

    for idx, style in enumerate(GROUPS):
        ax = axes[idx]
        df = groups[style.label]
        x = df[METRIC_X].to_numpy()
        y = df[METRIC_Y].to_numpy()
        sample_size = min(350, len(df))
        rng = np.random.default_rng(100 + idx)
        take = np.arange(len(df)) if len(df) <= sample_size else rng.choice(len(df), sample_size, replace=False)
        ax.scatter(x[take], y[take], s=8, color=style.color, alpha=0.18, edgecolors="none", rasterized=True)

        if len(df) >= 10:
            kde = gaussian_kde(np.vstack([x, y]), bw_method=0.25)
            zz = np.reshape(kde(positions).T, xx.shape)
            levels = np.quantile(zz[zz > 0], [0.70, 0.85, 0.94])
            ax.contourf(xx, yy, zz, levels=np.r_[levels[0], levels[1], levels[2], zz.max()], colors=[style.color], alpha=0.10)
            ax.contour(xx, yy, zz, levels=levels, colors=style.color, linewidths=[0.8, 1.1, 1.4])

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, linestyle=":", linewidth=0.45, color="#D7D7D7")
        ax.set_title(f"{style.short}  (n={len(df)})", fontsize=11, weight="bold", color=style.color)
        if idx >= 3:
            ax.set_xlabel("XCorr-like Score", fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel("Spectral Angle", fontsize=10)

    axes[-1].axis("off")
    fig.suptitle("Candidate A: Faceted 2D density maps keep the joint structure while avoiding overlap", fontsize=14, weight="bold")

    out_path = out_prefix.with_name(f"{out_prefix.stem}_candidate_A_facet_density.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def candidate_overlay_inset(groups: dict[str, pd.DataFrame], out_prefix: Path) -> Path:
    _set_base_style()
    long_df = to_long_df(groups)
    (x_lo, x_hi), (y_lo, y_hi) = _axis_limits(long_df)
    non_decoy = long_df[long_df["group"] != "Decoy PSM"]
    zoom_x = (
        max(0.0, non_decoy[METRIC_X].quantile(0.02) - 0.01),
        non_decoy[METRIC_X].quantile(0.98) + 0.02,
    )
    zoom_y = (
        max(0.0, non_decoy[METRIC_Y].quantile(0.02) - 0.01),
        non_decoy[METRIC_Y].quantile(0.98) + 0.02,
    )

    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=300)
    rng = np.random.default_rng(42)

    for idx, style in enumerate(GROUPS):
        df = groups[style.label]
        x = df[METRIC_X].to_numpy()
        y = df[METRIC_Y].to_numpy()
        sample_size = min(450, len(df))
        take = np.arange(len(df)) if len(df) <= sample_size else rng.choice(len(df), sample_size, replace=False)
        ax.scatter(
            x[take],
            y[take],
            s=9 if style.label != "Decoy PSM" else 7,
            color=style.color,
            alpha=0.10 if style.label == "Decoy PSM" else 0.18,
            edgecolors="none",
            rasterized=True,
            label=f"{style.short} (n={len(df)})",
        )
        if len(df) >= 12:
            kde = gaussian_kde(np.vstack([x, y]), bw_method=0.22)
            xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, 140), np.linspace(y_lo, y_hi, 140))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            level = np.quantile(zz[zz > 0], 0.90)
            ax.contour(xx, yy, zz, levels=[level], colors=style.color, linewidths=1.6 if style.label == "Shared circRNA PSM" else 1.1)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("XCorr-like Score", fontsize=11)
    ax.set_ylabel("Spectral Angle", fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.45, color="#D8D8D8")
    ax.set_title("Candidate B: Global overlay with a zoomed high-quality inset", fontsize=13, weight="bold")

    inset = ax.inset_axes([0.53, 0.10, 0.42, 0.42])
    for style in GROUPS:
        df = groups[style.label]
        inset.scatter(df[METRIC_X], df[METRIC_Y], s=7, color=style.color, alpha=0.10 if style.label == "Decoy PSM" else 0.14, edgecolors="none", rasterized=True)
        if len(df) >= 12:
            kde = gaussian_kde(np.vstack([df[METRIC_X], df[METRIC_Y]]), bw_method=0.22)
            xx, yy = np.meshgrid(np.linspace(*zoom_x, 120), np.linspace(*zoom_y, 120))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            level = np.quantile(zz[zz > 0], 0.90)
            inset.contour(xx, yy, zz, levels=[level], colors=style.color, linewidths=1.0)
    inset.set_xlim(*zoom_x)
    inset.set_ylim(*zoom_y)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Zoom", fontsize=9)
    ax.indicate_inset_zoom(inset, edgecolor="#666666", linewidth=0.8)

    legend = ax.legend(loc="upper left", fontsize=8, frameon=True, edgecolor="#D0D0D0")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.95)

    out_path = out_prefix.with_name(f"{out_prefix.stem}_candidate_B_overlay_inset.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def candidate_ecdf(long_df: pd.DataFrame, out_prefix: Path) -> Path:
    _set_base_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=300, constrained_layout=True)
    metric_specs = [(METRIC_X, "XCorr-like Score"), (METRIC_Y, "Spectral Angle")]

    for ax, (metric, title) in zip(axes, metric_specs):
        decoy_values = long_df.loc[long_df["group"] == "Decoy PSM", metric].to_numpy()
        line_annotations = []
        for style in GROUPS:
            values = np.sort(long_df.loc[long_df["group"] == style.label, metric].to_numpy())
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, color=style.color, linewidth=2.2, label=style.short, zorder=3)
            if style.label != "Decoy PSM":
                ks_x, ks_stat, ks_p = _ks_max_point(decoy_values, values)
                ax.axvline(
                    ks_x,
                    color=style.color,
                    linewidth=1.7,
                    alpha=0.85,
                    linestyle=(0, (3, 3)),
                    zorder=2,
                )
                line_annotations.append((ks_x, style, ks_stat, ks_p))
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Cumulative fraction of PSMs", fontsize=11)
        ax.set_title(f"Candidate C: ECDF of {title}", fontsize=12, weight="bold")
        ax.grid(True, linestyle=":", linewidth=0.45, color="#D8D8D8")
        ax.set_ylim(-0.02, 1.13)
        for idx, (ks_x, style, ks_stat, _ks_p) in enumerate(sorted(line_annotations, key=lambda item: item[0])):
            y_text = 1.035 + 0.035 * (idx % 2)
            ax.text(
                ks_x,
                y_text,
                f"{style.short}\n{ks_x:.2f}",
                color=style.color,
                fontsize=7.8,
                ha="center",
                va="bottom",
            )
        stat_text = "\n".join(
            f"{style.short}: D={ks_stat:.2f}, p={_format_pvalue(ks_p)}"
            for _, style, ks_stat, ks_p in line_annotations
        )
        ax.text(
            0.02,
            0.98,
            "K-S vs Decoy\n" + stat_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            bbox=dict(facecolor="white", edgecolor="#D0D0D0", boxstyle="round,pad=0.30", alpha=0.92),
        )

    legend = axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 0.03), fontsize=8, frameon=True, edgecolor="#D0D0D0", ncol=3)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.96)

    out_path = out_prefix.with_name(f"{out_prefix.stem}_candidate_C_ecdf.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def candidate_quality_index(long_df: pd.DataFrame, out_prefix: Path) -> Path:
    _set_base_style()
    df = compute_quality_index(long_df)
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.4),
        dpi=300,
        sharex=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    panel_specs = [
        ("Background PSMs", ["Decoy PSM", "mRNA Target PSM"]),
        ("circRNA-derived PSMs", ["Shared circRNA PSM", "MStoCIRC-only PSM", "MStoCIRC2-only PSM"]),
    ]

    reference = df.loc[df["group"] == "Decoy PSM", "quality_index_pct"].to_numpy()

    for ax, (panel_title, labels) in zip(axes, panel_specs):
        styles = [style for style in GROUPS if style.label in labels]
        ordered = labels[::-1]
        positions = np.arange(len(ordered), 0, -1)
        datasets = [df.loc[df["group"] == label, "quality_index_pct"].to_numpy() for label in ordered]
        violin = ax.violinplot(
            datasets,
            positions=positions,
            vert=False,
            widths=0.92,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body, label in zip(violin["bodies"], ordered):
            vertices = body.get_paths()[0].vertices
            center_y = np.mean(vertices[:, 1])
            vertices[:, 1] = np.clip(vertices[:, 1], center_y, np.inf)
            body.set_facecolor(RGBA_CLOUD[label])
            body.set_edgecolor("none")

        stat_lines = []
        for pos, label, values in zip(positions, ordered, datasets):
            style = next(item for item in styles if item.label == label)
            q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            sample_size = min(240, len(values))
            sampled = values if len(values) <= sample_size else rng.choice(values, size=sample_size, replace=False)
            jitter = rng.uniform(-0.022, 0.022, size=len(sampled))
            scatter_y = pos - 0.24
            ax.scatter(
                sampled,
                np.full(len(sampled), scatter_y) + jitter,
                s=11,
                color=RGBA_SCATTER[label],
                edgecolors="none",
                zorder=2,
            )
            ax.boxplot(
                values,
                positions=[scatter_y],
                vert=False,
                widths=0.065,
                showcaps=True,
                showfliers=False,
                whis=1.5,
                patch_artist=True,
                boxprops=dict(facecolor="none", edgecolor="#000000", linewidth=1.2, zorder=5),
                whiskerprops=dict(color="#000000", linewidth=1.1, zorder=5),
                capprops=dict(color="#000000", linewidth=1.1, zorder=5),
                medianprops=dict(color="#000000", linewidth=1.6, zorder=6),
            )
            ax.text(
                101.0,
                pos - 0.08,
                f"Med: {med:.1f}\nIQR: {iqr:.1f}\nn={len(values)}",
                fontsize=8.4,
                ha="right",
                va="center",
                bbox=dict(facecolor="#FAFAFA", edgecolor="#D3D3D3", boxstyle="round,pad=0.30", alpha=0.92),
            )
            if label != "Decoy PSM":
                _, p_value = mannwhitneyu(values, reference, alternative="greater")
                stat_lines.append(f"{style.short} > Decoy: p={_format_pvalue(p_value)}")

        ax.set_yticks(positions)
        ax.set_yticklabels(
            [next(style.short for style in styles if style.label == label) for label in ordered],
            fontsize=11,
            fontweight="bold",
        )
        ax.set_title(panel_title, fontsize=12, weight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5, color="#CFCFCF")
        ax.set_xlim(-2, 104)
        ax.set_ylim(0.45, len(ordered) + 0.7)
        if stat_lines:
            ax.text(
                0.98,
                0.96,
                "\n".join(stat_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.3,
                bbox=dict(facecolor="white", edgecolor="#D0D0D0", boxstyle="round,pad=0.35", alpha=0.94),
            )

    axes[0].set_ylabel("PSM Group", fontsize=12, fontweight="bold", labelpad=8)
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("Composite spectral quality index (percentile rank)", fontsize=12, fontweight="bold", labelpad=8)
    fig.suptitle("Candidate D: half-raincloud view of composite spectral quality", fontsize=14, weight="bold")

    out_path = out_prefix.with_name(f"{out_prefix.stem}_candidate_D_quality_index.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary(long_df: pd.DataFrame, out_prefix: Path) -> Path:
    summary = (
        compute_quality_index(long_df)
        .groupby("group")
        .agg(
            n=(METRIC_X, "size"),
            xcorr_median=(METRIC_X, "median"),
            spectral_angle_median=(METRIC_Y, "median"),
            quality_index_median=("quality_index_pct", "median"),
        )
        .reset_index()
    )
    out_path = out_prefix.with_name(f"{out_prefix.stem}_candidate_summary.tsv")
    summary.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple candidate PSM-quality figures.")
    parser.add_argument("--mrna", type=Path, required=True)
    parser.add_argument("--mstocirc", type=Path, required=True)
    parser.add_argument("--mstocirc2", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    groups = load_group_tables(args.mrna, args.mstocirc, args.mstocirc2)
    long_df = to_long_df(groups)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    outputs = [candidate_ecdf(long_df, args.output_prefix), candidate_quality_index(long_df, args.output_prefix)]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
