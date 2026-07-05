"""FASTA assembly – reading, decoy generation (Met‑retention), header tagging, and merging."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .header_tagger import tag_header

log = logging.getLogger(__name__)


def _generate_decoy(record: SeqRecord) -> SeqRecord:
    """Generate a decoy protein with FragPipe‑style Met‑retention reversal."""
    seq_str = str(record.seq)
    if seq_str.startswith("M"):
        decoy_seq = "M" + seq_str[1:][::-1]
    else:
        decoy_seq = seq_str[::-1]

    decoy_id = f"rev_{record.id}"
    # description: copy original description, will be tagged later
    return SeqRecord(Seq(decoy_seq), id=decoy_id, description=record.description)


def _read_records(fasta_path: str) -> List[SeqRecord]:
    """Read all records from a FASTA file, returning empty list if file does not exist."""
    if not fasta_path:
        return []
    path = Path(fasta_path)
    if not path.exists():
        log.warning(f"FASTA file not found: {fasta_path}")
        return []
    if path.is_dir():
        log.warning(f"FASTA path is a directory, skipping: {fasta_path}")
        return []
    return list(SeqIO.parse(str(path), "fasta"))


def assemble_database(
    circ_orf_path: str,
    linear_protein_path: str,
    output_dir: str,
) -> Path:
    """Build the unified decoy-contaminant database.

    Returns the absolute path of the generated 'decoys-contam-database.fasta'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decoys-contam-database.fasta"

    merged: List[SeqRecord] = []

    # ---- circRNA targets ----
    circ_records = _read_records(circ_orf_path)
    for rec in circ_records:
        rec.description = tag_header(rec.description, "PE=2")
        merged.append(rec)
        decoy = _generate_decoy(rec)
        decoy.description = tag_header(decoy.description, "PE=2")
        merged.append(decoy)
    log.info(f"circRNA: {len(circ_records)} targets + {len(circ_records)} decoys")

    # ---- linear protein targets ----
    linear_records = _read_records(linear_protein_path)
    for rec in linear_records:
        rec.description = tag_header(rec.description, "PE=1")
        merged.append(rec)
        decoy = _generate_decoy(rec)
        decoy.description = tag_header(decoy.description, "PE=1")
        merged.append(decoy)
    log.info(f"linear proteins: {len(linear_records)} targets + {len(linear_records)} decoys")

    if not merged:
        raise RuntimeError("No sequences to write. At least one input FASTA must be provided.")

    SeqIO.write(merged, str(out_path), "fasta")
    log.info(f"Unified database written → {out_path}  ({len(merged)} total sequences)")
    return out_path.resolve()
