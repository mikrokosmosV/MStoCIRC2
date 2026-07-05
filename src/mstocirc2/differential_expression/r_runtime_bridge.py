"""Bridge to R via subprocess for DEA and imputation."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _discover_rscript() -> str | None:
    candidates = []
    for env_name in ("MSTOCIRC2_RSCRIPT", "RSCRIPT", "R_HOME"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        if env_name == "R_HOME":
            candidates.append(str(Path(env_value) / "bin" / "Rscript"))
            candidates.append(str(Path(env_value) / "bin" / "x64" / "Rscript.exe"))
        else:
            candidates.append(env_value)

    resolved = shutil.which("Rscript")
    if resolved:
        candidates.append(resolved)

    for candidate in candidates:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=10, check=False)
            return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    return None


RSCRIPT: str | None = _discover_rscript()


def _to_r_path(p: str) -> str:
    return p.replace("\\", "/")


def _run_r(script: str, timeout: int = 600) -> str:
    if RSCRIPT is None:
        raise RuntimeError("Rscript not found.")
    fd, rfile = tempfile.mkstemp(suffix=".R", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script)
        result = subprocess.run(
            [RSCRIPT, rfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R failed (exit {result.returncode}):\n{result.stderr[-3000:]}")
        return result.stdout
    finally:
        try:
            os.unlink(rfile)
        except OSError:
            pass


def r_impute(protein_matrix: pd.DataFrame, method: str) -> pd.DataFrame | None:
    scripts: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="mstocirc2_r_impute_") as tmpdir:
        td = Path(tmpdir)
        in_path = td / "input.tsv"
        out_path = td / "output.tsv"
        protein_matrix.to_csv(in_path, sep="\t")
        in_r, out_r = _to_r_path(str(in_path)), _to_r_path(str(out_path))
        scripts = {
            "MinDet": f'''
mat <- as.matrix(read.delim("{in_r}", row.names=1, check.names=FALSE))
mat[mat <= 0] <- NA; mat_log <- log2(mat)
for(j in 1:ncol(mat_log)) {{ col <- mat_log[,j]; if(all(is.na(col))) next
  col[is.na(col)] <- quantile(col, 0.01, na.rm=TRUE); mat_log[,j] <- col }}
write.table(2^mat_log, file="{out_r}", sep="\t", quote=FALSE)
''',
            "MinProb": f'''
set.seed(42)
mat <- as.matrix(read.delim("{in_r}", row.names=1, check.names=FALSE))
mat[mat <= 0] <- NA; mat_log <- log2(mat)
for(j in 1:ncol(mat_log)) {{ col <- mat_log[,j]; if(all(is.na(col))) next
  center <- quantile(col, 0.01, na.rm=TRUE); sigma <- sd(col, na.rm=TRUE)*0.3
  if(is.na(sigma)||sigma==0) sigma <- 0.001; na_idx <- which(is.na(col))
  col[na_idx] <- rnorm(length(na_idx), center, sigma); mat_log[,j] <- col }}
write.table(2^mat_log, file="{out_r}", sep="\t", quote=FALSE)
''',
            "bpca": f'''
library(pcaMethods)
mat <- as.matrix(read.delim("{in_r}", row.names=1, check.names=FALSE))
mat[mat <= 0] <- NA; mat_log <- log2(mat)
pc <- pca(mat_log, method="bpca", nPcs=2)
write.table(2^(completeObs(pc)), file="{out_r}", sep="\t", quote=FALSE)
''',
            "Impseq": f'''
library(rrcovNA)
mat <- as.matrix(read.delim("{in_r}", row.names=1, check.names=FALSE))
mat[mat <= 0] <- NA; mat_log <- log2(mat)
write.table(2^impSeq(mat_log), file="{out_r}", sep="\t", quote=FALSE)
''',
            "SeqKNN": f'''
mat <- as.matrix(read.delim("{in_r}", row.names=1, check.names=FALSE))
mat[mat <= 0] <- NA; mat_log <- log2(mat)
k <- min(10, nrow(mat_log)-1)
for(j in 1:ncol(mat_log)) {{
  na_rows <- which(is.na(mat_log[,j])); if(length(na_rows)==0) next
  non_na <- which(!is.na(mat_log[,j]))
  if(length(non_na)==0) {{ mat_log[na_rows,j] <- min(mat_log,na.rm=TRUE); next }}
  for(i in na_rows) {{
    other <- setdiff(1:ncol(mat_log), j)
    if(length(other)==0) {{ mat_log[i,j] <- mean(mat_log[non_na,j],na.rm=TRUE); next }}
    row_i <- mat_log[i, other]
    dists <- apply(mat_log[non_na, other, drop=FALSE], 1, function(r) {{
      shared <- !is.na(r) & !is.na(row_i)
      if(sum(shared)==0) return(Inf); sqrt(sum((r[shared]-row_i[shared])^2)) }})
    k_eff <- min(k, length(non_na))
    neighbors <- non_na[order(dists)[1:k_eff]]
    mat_log[i,j] <- mean(mat_log[neighbors,j], na.rm=TRUE) }} }}
mat_log[is.na(mat_log)] <- min(mat_log, na.rm=TRUE)
write.table(2^mat_log, file="{out_r}", sep="\t", quote=FALSE)
''',
        }
        if method not in scripts:
            return None
        try:
            _run_r(scripts[method], timeout=1200)
            if not out_path.exists():
                raise RuntimeError(
                    f"R imputation '{method}' finished without producing '{out_path.name}'."
                )
            result = pd.read_csv(out_path, sep="\t", index_col=0)
            result.index.name = protein_matrix.index.name
            return result
        except Exception as e:
            log.warning(f"R imputation ({method}) failed: {e}")
            return None


def r_dea(imputed_matrix, design, contrast, method, peptide_counts=None,
           rots_b=200, rots_k=200, seed=42) -> pd.DataFrame | None:
    group_a, group_b = contrast.split("_vs_")
    with tempfile.TemporaryDirectory(prefix="mstocirc2_r_dea_") as tmpdir:
        td = Path(tmpdir)
        mat_path = td / "matrix.tsv"
        design_path = td / "design.tsv"
        pc_path = td / "peptide_counts.tsv"
        out_path = td / "dea_result.tsv"
        log2_mat = np.log2(imputed_matrix.clip(lower=1e-10))
        log2_mat.to_csv(mat_path, sep="\t")
        design.to_csv(design_path, sep="\t", index=False)
        if peptide_counts is not None:
            peptide_counts.to_frame("peptide_count").to_csv(pc_path, sep="\t")
        mat_r, des_r, pc_r, out_r = [_to_r_path(str(p)) for p in [mat_path, design_path, pc_path, out_path]]
        preamble = f'''
mat <- as.matrix(read.delim("{mat_r}", row.names=1, check.names=FALSE))
design_df <- read.delim("{des_r}", stringsAsFactors=FALSE, check.names=FALSE)
design_df$sample <- trimws(design_df$sample)
idx <- match(design_df$sample, colnames(mat))
if(any(is.na(idx))) {{
  bn_d <- tools::file_path_sans_ext(basename(design_df$sample))
  bn_m <- tools::file_path_sans_ext(basename(colnames(mat)))
  idx <- match(bn_d, bn_m) }}
if(any(is.na(idx))) {{
  idx <- sapply(design_df$sample, function(s) {{
    hits <- grep(tools::file_path_sans_ext(basename(s)), colnames(mat), fixed=TRUE)
    if(length(hits)>=1) hits[1] else NA_integer_ }}) }}
if(any(is.na(idx))) stop("Cannot match design samples to matrix columns")
mat <- mat[, idx, drop=FALSE]
group <- factor(design_df$condition, levels=c("{group_b}", "{group_a}"))
'''
        scripts = {
            "limma": preamble + f'''
library(limma)
fit <- lmFit(mat, model.matrix(~group)); fit <- eBayes(fit)
res <- topTable(fit, coef=2, number=Inf, sort.by="none"); res$protein_id <- rownames(mat)
write.table(res[,c("protein_id","logFC","P.Value","adj.P.Val")], file="{out_r}", sep="\t", quote=FALSE, row.names=FALSE)
''',
            "DEqMS": preamble + f'''
library(limma); library(DEqMS)
fit <- lmFit(mat, model.matrix(~group)); fit <- eBayes(fit)
pc_df <- read.delim("{pc_r}", row.names=1, check.names=FALSE)
fit$count <- pc_df[rownames(mat),"peptide_count"]; fit$count[is.na(fit$count)] <- 1
fit <- spectraCounteBayes(fit); res <- outputResult(fit, coef_col=2)
res$protein_id <- rownames(res)
write.table(res[,c("protein_id","logFC","sca.P.Value","sca.adj.pval")], file="{out_r}", sep="\t", quote=FALSE, row.names=FALSE)
''',
            "ROTS": preamble + f'''
library(ROTS); set.seed({seed})
groups <- ifelse(design_df$condition=="{group_a}", 1, 0)
rots_res <- ROTS(data=mat, groups=groups, B={rots_b}, K={rots_k}, seed={seed})
res <- data.frame(protein_id=rownames(mat),
  logFC=rowMeans(mat[,groups==1,drop=FALSE])-rowMeans(mat[,groups==0,drop=FALSE]),
  P.Value=rots_res$pvalue, adj.P.Val=rots_res$FDR, stringsAsFactors=FALSE)
write.table(res, file="{out_r}", sep="\t", quote=FALSE, row.names=FALSE)
''',
            "proDA": preamble + f'''
library(proDA); mat[!is.finite(mat)] <- NA
cond <- design_df$condition
fit <- proDA(mat, design=~cond, col_data=data.frame(cond=cond))
res <- test_diff(fit, contrast="cond{group_a}"); res$protein_id <- rownames(mat)
write.table(res[,c("protein_id","diff","pval","adj_pval")], file="{out_r}", sep="\t", quote=FALSE, row.names=FALSE)
''',
        }
        if method not in scripts:
            return None
        try:
            _run_r(scripts[method], timeout=1200)
            if not out_path.exists():
                raise RuntimeError(
                    f"R DEA '{method}' finished without producing '{out_path.name}'."
                )
            result = pd.read_csv(out_path, sep="\t")
            col_map = {"logFC":"log2FC","diff":"log2FC","P.Value":"P.Value",
                       "sca.P.Value":"P.Value","pval":"P.Value",
                       "adj.P.Val":"adj.P.Val","sca.adj.pval":"adj.P.Val","adj_pval":"adj.P.Val"}
            result = result.rename(columns=col_map)
            return result[["protein_id","log2FC","P.Value","adj.P.Val"]]
        except Exception as e:
            log.warning(f"R DEA ({method}) failed: {e}")
            return None
