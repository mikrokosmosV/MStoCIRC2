"""Differential expression analysis – delegates to R native packages (limma, DEqMS, ROTS, proDA)."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ..core import DependencyError

log = logging.getLogger(__name__)


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    if mask.sum() == 0:
        return adj
    pm = p[mask]; n = pm.size
    order = np.argsort(pm); ranked = pm[order]
    adj_ranked = ranked * n / (np.arange(n) + 1)
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj_ranked = np.clip(adj_ranked, 0, 1)
    out = np.empty(n); out[order] = adj_ranked
    adj[mask] = out
    return adj


def run_dea(method, protein_matrix_raw, protein_matrix_imputed, design, contrast,
            peptide_counts, is_circrna, rots_b=200, rots_k=200, seed=42):
    from .r_runtime_bridge import r_dea, RSCRIPT
    group_a, group_b = contrast.split("_vs_")
    conds = set(design["condition"].unique())
    missing = {group_a, group_b} - conds
    if missing:
        raise ValueError(f"Contrast references missing conditions: {missing}")
    if RSCRIPT is None:
        raise DependencyError(
            "Rscript not found. R is required for DEA. Configure `Rscript` on PATH or set "
            "`MSTOCIRC2_RSCRIPT`, `RSCRIPT`, or `R_HOME`."
        )
    log.info(f"DEA: {method} via R native package")
    base = r_dea(imputed_matrix=protein_matrix_imputed, design=design,
                 contrast=contrast, method=method, peptide_counts=peptide_counts,
                 rots_b=rots_b, rots_k=rots_k, seed=seed)
    if base is None:
        raise DependencyError(
            f"R DEA '{method}' failed. Check the required R packages and your R runtime."
        )
    base["method"] = method
    base["peptide_count"] = peptide_counts.reindex(base["protein_id"]).fillna(0).astype(int).to_numpy()
    base["is_circrna"] = is_circrna.reindex(base["protein_id"]).fillna(False).astype(bool).to_numpy()
    return base[["protein_id","is_circrna","log2FC","P.Value","adj.P.Val","method","peptide_count"]]
