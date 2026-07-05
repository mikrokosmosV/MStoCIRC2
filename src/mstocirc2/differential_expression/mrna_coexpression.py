"""circRNA ↔ mRNA co‑expression analysis (vectorised). Only DE circRNAs. NaN → 0."""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from scipy import stats


def _bh(pvals):
    from .statistical_dea import _bh_adjust
    return _bh_adjust(pvals)


def _rankdata_rows(X):
    ranks = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        ranks[i] = stats.rankdata(X[i], method="average")
    return ranks


def _vectorised_corr(A, B):
    A_mean = A.mean(axis=1, keepdims=True)
    B_mean = B.mean(axis=1, keepdims=True)
    A_std = A.std(axis=1, keepdims=True, ddof=0)
    B_std = B.std(axis=1, keepdims=True, ddof=0)
    A_std = np.where(A_std == 0, 1e-12, A_std)
    B_std = np.where(B_std == 0, 1e-12, B_std)
    return ((A - A_mean) / A_std @ ((B - B_mean) / B_std).T) / A.shape[1]


def correlate_circrna_vs_mrna(
    protein_matrix: pd.DataFrame,
    circrna_ids: List[str],
    mrna_ids: List[str],
    method: str = "pearson",
    sample_subset: List[str] | None = None,
    top_k_per_circrna: int | None = None,
    correlation_threshold: float = 0.7,
    p_threshold: float = 0.05,
    use_adj_p: bool = False,
) -> pd.DataFrame:
    if sample_subset:
        cols = [c for c in sample_subset if c in protein_matrix.columns]
        pm = protein_matrix[cols]
    else:
        pm = protein_matrix
    circ_ids = [c for c in circrna_ids if c in pm.index]
    mrna_ids_eff = [m for m in mrna_ids if m in pm.index]
    n = pm.shape[1]
    if not circ_ids or not mrna_ids_eff or n < 2:
        return pd.DataFrame(columns=["circrna_id","mrna_id","correlation",
                                      "p_value","adj_p_value","n_samples","rank_score"])
    circ_mat = np.log2(pm.loc[circ_ids].to_numpy(dtype=float).clip(min=1e-10))
    mrna_mat = np.log2(pm.loc[mrna_ids_eff].to_numpy(dtype=float).clip(min=1e-10))
    circ_mat = np.where(np.isfinite(circ_mat), circ_mat, 0.0)
    mrna_mat = np.where(np.isfinite(mrna_mat), mrna_mat, 0.0)
    if method == "spearman":
        circ_in, mrna_in = _rankdata_rows(circ_mat), _rankdata_rows(mrna_mat)
    else:
        circ_in, mrna_in = circ_mat, mrna_mat
    R = np.clip(_vectorised_corr(circ_in, mrna_in), -1+1e-12, 1-1e-12)
    if n >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = R * np.sqrt((n-2)/(1-R*R))
        P = 2 * stats.t.sf(np.abs(t), df=n-2)
    else:
        P = np.full_like(R, np.nan)
    circ_range = np.where(np.ptp(circ_mat, axis=1) > 0, np.ptp(circ_mat, axis=1), 1e-6)
    mrna_range = np.where(np.ptp(mrna_mat, axis=1) > 0, np.ptp(mrna_mat, axis=1), 1e-6)
    rank_mat = R * np.sqrt(np.outer(circ_range, mrna_range)) if n < 3 else R
    rows = []
    for i, cid in enumerate(circ_ids):
        r_row, rank_row, p_row = R[i], rank_mat[i], P[i]
        if n >= 3:
            # First pass: filter by |corr| threshold and raw p-value
            mask = (np.abs(r_row) >= correlation_threshold) & (p_row < p_threshold)
            idx = np.where(mask)[0]
        else:
            k = min(top_k_per_circrna or 100, len(mrna_ids_eff))
            if k <= 0:
                idx = np.array([], dtype=int)
            elif k >= len(rank_row):
                idx = np.argsort(-np.abs(rank_row))[:k]
            else:
                idx = np.argpartition(-np.abs(rank_row), k)[:k]
        for j in idx:
            p_val = float(p_row[j]) if np.isfinite(p_row[j]) else np.nan
            rows.append((cid, mrna_ids_eff[j], float(r_row[j]), p_val, int(n), float(rank_row[j])))
    if not rows:
        return pd.DataFrame(columns=["circrna_id","mrna_id","correlation",
                                      "p_value","adj_p_value","n_samples","rank_score"])
    df = pd.DataFrame(rows, columns=["circrna_id","mrna_id","correlation",
                                      "p_value","n_samples","rank_score"])
    df["adj_p_value"] = _bh(df["p_value"].to_numpy()) if df["p_value"].notna().any() else np.nan
    out = df[["circrna_id","mrna_id","correlation","p_value","adj_p_value","n_samples","rank_score"]]
    # When n>=3, the filtering used p_value; if use_adj_p, re-filter by adj_p_value
    if n >= 3 and use_adj_p:
        out = out[out["adj_p_value"] < p_threshold].copy()
    return out
