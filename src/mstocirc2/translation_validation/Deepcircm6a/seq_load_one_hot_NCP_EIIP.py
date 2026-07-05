from __future__ import annotations

from Bio import SeqIO


def convert_seq_to_bicoding(seq: str) -> list[float]:
    seq = seq.replace("U", "T")
    feat_bicoding: list[float] = []
    bicoding_dict = {
        "A": [1, 0, 0, 0, 1, 1, 1, 0.1260],
        "C": [0, 1, 0, 0, 0, 1, 0, 0.1340],
        "G": [0, 0, 1, 0, 1, 0, 0, 0.0806],
        "T": [0, 0, 0, 1, 0, 0, 1, 0.1335],
        "N": [0, 0, 0, 0, 0, 0, 0, 0],
        "a": [1, 0, 0, 0, 1, 1, 1, 0.1260],
        "c": [0, 1, 0, 0, 0, 1, 0, 0.1340],
        "g": [0, 0, 1, 0, 1, 0, 0, 0.0806],
        "t": [0, 0, 0, 1, 0, 0, 1, 0.0],
        "n": [0, 0, 0, 0, 0, 0, 0, 0.0],
    }
    if len(seq) < 51:
        seq = seq + ("N" * (51 - len(seq)))
    for each_nt in seq:
        feat_bicoding += bicoding_dict.get(each_nt, bicoding_dict["N"])
    return feat_bicoding


def load_data_bicoding_with_header(in_fa: str) -> tuple[list[list[float]], list[str]]:
    data: list[list[float]] = []
    fa_header: list[str] = []
    for record in SeqIO.parse(in_fa, "fasta"):
        seq = str(record.seq)
        bicoding = convert_seq_to_bicoding(seq)
        data.append(bicoding)
        fa_header.append(str(record.description))
    return data, fa_header
