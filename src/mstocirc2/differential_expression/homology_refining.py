"""circRNA‑mRNA homologous peptide filter – removes peptides that appear in both circRNA and linear proteomes."""
from __future__ import annotations
import re
from typing import List, Tuple, Set, Dict
import pandas as pd

STRIPPED_SEQ_CANDIDATES = ["Stripped.Sequence", "Sequence", "peptide_seq",
                            "PeptideSeq", "Peptide"]
PROTEIN_ASSIGN_CANDIDATES = ["Protein.Group", "Protein.Ids",
                              "Leading razor protein", "Proteins", "Protein IDs"]

_ORF_PREFIX_RE = re.compile(
    r"^[a-zA-Z]{1,5}_[\w.\-]+_[0-9]+-ORF[\w]+\(.*\):.*", re.IGNORECASE
)


def _first_present(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Missing columns: {candidates}. Got: {list(df.columns)[:15]}")


def _circ_orf_base_ids(circ_orf_ids: Set[str]) -> Set[str]:
    return {orf.split("(", 1)[0] for orf in circ_orf_ids if orf}


def looks_like_circrna_id(
    pid: str,
    circ_orf_ids: Set[str],
    circ_orf_base_ids: Set[str] | None = None,
) -> bool:
    if not pid:
        return False
    pid = pid.strip()
    if pid in circ_orf_ids:
        return True
    head = pid.split("(", 1)[0]
    if head in (circ_orf_base_ids if circ_orf_base_ids is not None else _circ_orf_base_ids(circ_orf_ids)):
        return True
    return bool(_ORF_PREFIX_RE.match(pid))


def filter_peptides(
    peptide_df: pd.DataFrame,
    circ_ref_peptides: Set[str],
    circ_orf_ids: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    seq_col = _first_present(peptide_df, STRIPPED_SEQ_CANDIDATES)
    prot_col = _first_present(peptide_df, PROTEIN_ASSIGN_CANDIDATES)
    input_n = len(peptide_df)
    report_rows: List[Dict[str, str]] = []
    keep = []
    circ_kept = 0
    circ_orf_base_ids = _circ_orf_base_ids(circ_orf_ids)
    for seq, prot in zip(peptide_df[seq_col].astype(str),
                         peptide_df[prot_col].astype(str)):
        seq_upper = seq.strip().upper()
        pid = prot.strip()
        is_circ = looks_like_circrna_id(pid, circ_orf_ids, circ_orf_base_ids)
        if is_circ:
            if seq_upper in circ_ref_peptides:
                keep.append(True)
                circ_kept += 1
            else:
                keep.append(False)
                report_rows.append({
                    "peptide_seq": seq_upper,
                    "protein_group": pid,
                    "removed_reason": "mrna_homology",
                })
        else:
            keep.append(True)
    out = peptide_df.loc[keep].copy().reset_index(drop=True)
    summary = {
        "input_peptide_count": int(input_n),
        "peptides_removed_mrna_homology": int(sum(1 for k in keep if not k)),
        "peptides_retained": int(len(out)),
        "circrna_peptides_retained": int(circ_kept),
    }
    report = pd.DataFrame(report_rows, columns=[
        "peptide_seq", "protein_group", "removed_reason"])
    return out, report, summary
