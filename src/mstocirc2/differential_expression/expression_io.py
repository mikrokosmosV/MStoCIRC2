"""Expression matrix I/O – parsing peptide matrices (DIA‑NN, FragPipe), circRNA reference, design file, and sample‑name matching."""
from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set
import pandas as pd

log = logging.getLogger(__name__)
_TABLE_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    last_error: Exception | None = None
    for encoding in _TABLE_ENCODINGS:
        try:
            df = pd.read_csv(p, sep="\t", low_memory=False, encoding=encoding)
            if df.shape[1] == 1:
                df = pd.read_csv(p, sep=",", low_memory=False, encoding=encoding)
            return df
        except UnicodeError as exc:
            last_error = exc
            continue
        except Exception:
            raise

    raise UnicodeDecodeError(
        getattr(last_error, "encoding", "utf-8"),
        getattr(last_error, "object", b""),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 1),
        (
            f"Unable to decode table '{path}'. Supported text encodings include UTF-8, "
            "UTF-8 with BOM, and UTF-16."
        ),
    ) from last_error


def detect_peptide_matrix_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "Protein.Group" in cols and "Stripped.Sequence" in cols:
        return "diann"
    if ("Peptide Sequence" in cols and "Protein" in cols
            and any(c.endswith(" Intensity") and not c.endswith(" MaxLFQ Intensity")
                    for c in df.columns)):
        return "fragpipe"
    return "unknown"


def normalize_fragpipe_matrix(
    df: pd.DataFrame, design_samples: List[str] | None = None,
) -> pd.DataFrame:
    intensity_pattern = re.compile(r"^(.+) Intensity$")
    maxlfq_suffix = " MaxLFQ Intensity"
    sample_cols: Dict[str, str] = {}
    for c in df.columns:
        if c.endswith(maxlfq_suffix):
            continue
        m = intensity_pattern.match(c)
        if m:
            sample_name = m.group(1).strip()
            if sample_name.endswith(" MaxLFQ"):
                continue
            sample_cols[sample_name] = c
    if not sample_cols:
        raise ValueError("FragPipe matrix has no '<sample> Intensity' columns.")
    if design_samples:
        keep = {s for s in design_samples if s in sample_cols}
        if not keep:
            raise ValueError("None of the design samples match FragPipe columns.")
        sample_cols = {k: v for k, v in sample_cols.items() if k in keep}
    out_cols = {
        "Stripped.Sequence": df["Peptide Sequence"],
        "Protein.Group": df["Protein"].astype(str).str.strip(),
    }
    if "Gene" in df.columns:
        out_cols["Genes"] = df["Gene"]
    elif "Genes" in df.columns:
        out_cols["Genes"] = df["Genes"]
    for sample_name, col_name in sample_cols.items():
        out_cols[sample_name] = pd.to_numeric(df[col_name], errors="coerce")
    out = pd.DataFrame(out_cols)
    log.info(f"FragPipe matrix normalized: {len(out)} rows, {len(sample_cols)} samples kept")
    return out


def load_peptide_matrix(
    path: str, design_samples: List[str] | None = None,
) -> Tuple[pd.DataFrame, str]:
    df = _read_table(path)
    fmt = detect_peptide_matrix_format(df)
    if fmt == "diann":
        log.info("Peptide matrix format: DIA-NN pr_matrix")
        return df, fmt
    if fmt == "fragpipe":
        log.info("Peptide matrix format: FragPipe combined_peptide")
        return normalize_fragpipe_matrix(df, design_samples), fmt
    raise ValueError(
        f"Unknown peptide matrix format. Got columns: {list(df.columns)[:10]}..."
    )


_COUNT_SUFFIX_RE = re.compile(r"\(\d+(?:\|\d+)?\)\s*$")


def parse_circrna_reference(path: str) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    df = _read_table(path)
    missing = [c for c in ("circ_ORF_ID", "peptide_seq") if c not in df.columns]
    if missing:
        raise ValueError(f"CircRNA_Reference missing column(s): {missing}")
    circ_orf_ids: Set[str] = set(df["circ_ORF_ID"].dropna().astype(str).str.strip())
    circ_orf_ids.discard("")
    circ_ref_peptides: Set[str] = set()
    raw_total = 0
    for entry in df["peptide_seq"].dropna().astype(str):
        for piece in entry.split(";"):
            piece = piece.strip()
            if not piece:
                continue
            raw_total += 1
            piece = _COUNT_SUFFIX_RE.sub("", piece).strip()
            if piece:
                circ_ref_peptides.add(piece.upper())
    if not circ_ref_peptides:
        raise ValueError("CircRNA_Reference has no valid peptide_seq entries")
    return df, circ_ref_peptides, circ_orf_ids


def parse_design(path: str) -> pd.DataFrame:
    df = _read_table(path)
    missing = [c for c in ("sample", "condition") if c not in df.columns]
    if missing:
        raise ValueError(f"Design_File missing column(s): {missing}")
    df = df[["sample", "condition"]].copy()
    df["sample"] = df["sample"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    conds = df["condition"].unique().tolist()
    if len(conds) != 2:
        raise ValueError(
            f"Design must have exactly 2 conditions, got {len(conds)}: {conds}"
        )
    return df


def _norm_sample(name: str) -> str:
    base = Path(str(name)).name
    while True:
        stem, ext = Path(base).stem, Path(base).suffix
        if ext.lower() in (".d", ".raw", ".mzml", ".mzxml", ".wiff", ".tsv", ".csv"):
            base = stem
            continue
        break
    return base


def match_samples_to_columns(samples: List[str], columns: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    unmatched: List[str] = []
    norm_cols = {c: _norm_sample(c) for c in columns}
    for s in samples:
        if s in columns:
            mapping[s] = s
            continue
        ns = _norm_sample(s)
        hit = None
        for col, ncol in norm_cols.items():
            if ncol == ns:
                hit = col
                break
        if hit is None:
            for col, ncol in norm_cols.items():
                if ns and (ns in ncol or ncol in ns):
                    hit = col
                    break
        if hit is None:
            unmatched.append(s)
        else:
            mapping[s] = hit
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} design sample(s) unmatched: {unmatched[:5]}"
            + ("..." if len(unmatched) > 5 else "")
        )
    return mapping
