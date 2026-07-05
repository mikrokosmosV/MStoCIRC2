"""Publication-aligned translation validation pipeline rebuilt from the original BM3 script."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from ..core import MStoCIRC2Error
from .evidence_scorer import build_source_bsj_location, is_peptide_crossing_bsj, parse_bsj_location
from .legacy_predictor import batch_ires_predict, batch_m6a_predict
from .linear_filter import batch_diamond_alignment_filter, tsv_remove_line_protaid
from .mass_spec_parser import process_ms_directory

log = logging.getLogger(__name__)

_ORF_PREFIXES = [
    "F_exon_",
    "F_base_",
    "F_self_",
    "F_noexonic_",
    "F_unknown_",
    "R_exon_",
    "R_base_",
    "R_self_",
    "R_noexonic_",
    "R_unknown_",
    "exon_",
]
_CIRC_ORF_HEADER_RE = re.compile(r"([A-Za-z0-9]+_circ_.+)")


def _resolve_plot_font_family() -> str:
    available_families = {font.name for font in fm.fontManager.ttflist}
    for family in ("Arial", "DejaVu Sans"):
        if family in available_families:
            return family
    return "sans-serif"


_PLOT_FONT_FAMILY = _resolve_plot_font_family()


def extract_circ_info_id(orf_base: str) -> str:
    for prefix in _ORF_PREFIXES:
        if orf_base.startswith(prefix):
            return orf_base[len(prefix) :]
    return orf_base


def normalize_orf_header(header: str) -> str:
    """Normalize search/mapping ORF identifiers to the shared circ header form."""
    match = _CIRC_ORF_HEADER_RE.search(header)
    return match.group(1) if match else header


def load_mapping_context(circ_mapping: str | None) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[int, int]]]:
    dic_output_to_source: dict[str, str] = {}
    dic_source_aa: dict[str, str] = {}
    dic_source_orf_coords: dict[str, tuple[int, int]] = {}
    if circ_mapping and Path(circ_mapping).exists():
        with Path(circ_mapping).open("r", encoding="utf-8", errors="ignore") as handle:
            handle.readline()
            for line in handle:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                dic_output_to_source[cols[0]] = cols[1]
                if len(cols) >= 3:
                    dic_source_aa[cols[1]] = cols[2]
                match = re.search(r"\((-?\d+),(-?\d+)\)", cols[1])
                if match:
                    coords = (int(match.group(1)), int(match.group(2)))
                    dic_source_orf_coords[cols[0]] = coords
                    dic_source_orf_coords[cols[1]] = coords
    return dic_output_to_source, dic_source_aa, dic_source_orf_coords


def read_circ_sequences(circ_seq: str) -> dict[str, str]:
    dic_circ_id_seq: dict[str, list[str]] = {}
    current_name: str | None = None
    with Path(circ_seq).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_name = line[1:].split(":")[0]
                dic_circ_id_seq[current_name] = []
            elif current_name is not None:
                dic_circ_id_seq[current_name].append(line)
    return {key: "".join(value) for key, value in dic_circ_id_seq.items()}


def circ_forward_reverse(circ_info_omit_file: str | None) -> dict[str, str]:
    dic_circ_strand: dict[str, str] = {}
    if not circ_info_omit_file or not Path(circ_info_omit_file).exists():
        return dic_circ_strand
    with Path(circ_info_omit_file).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                dic_circ_strand[parts[0]] = parts[5]
    return dic_circ_strand


def map_to_source_orfs(
    dic_id_peptide: dict[str, str],
    dic_output_to_source: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    dic_source_peptide_counts: dict[str, dict[tuple[str, str], int]] = {}
    dic_source_search_orfs: dict[str, list[str]] = {}
    for output_orf_id, pep_string in dic_id_peptide.items():
        normalized_output_orf_id = normalize_orf_header(output_orf_id)
        source_header = dic_output_to_source.get(normalized_output_orf_id, normalized_output_orf_id)
        dic_source_search_orfs.setdefault(source_header, []).append(output_orf_id)
        source_counts = dic_source_peptide_counts.setdefault(source_header, {})
        for raw_entry in pep_string.strip(";").split(";"):
            if not raw_entry.strip():
                continue
            parts = raw_entry.strip().split("###")
            if len(parts) < 2:
                continue
            ms_id = parts[0].strip()
            peptide = parts[1].strip().upper()
            try:
                count = max(1, int(float(parts[2]))) if len(parts) >= 3 else 1
            except (TypeError, ValueError):
                count = 1
            if not ms_id or not peptide:
                continue
            key = (ms_id, peptide)
            source_counts[key] = source_counts.get(key, 0) + count
    dic_source_peptides = {
        key: ";".join(
            f"{ms_id}###{peptide}###{count}"
            for (ms_id, peptide), count in sorted(value.items(), key=lambda item: (item[0][1], item[0][0]))
        )
        for key, value in dic_source_peptide_counts.items()
    }
    return dic_source_peptides, dic_source_search_orfs


def load_orf_fallback_sequences(circ_orf: str) -> dict[str, str]:
    dic_orf_seqs_fallback: dict[str, str] = {}
    if not Path(circ_orf).exists():
        return dic_orf_seqs_fallback
    cur_id = ""
    cur_seq = ""
    cur_base = ""
    with Path(circ_orf).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">"):
                if cur_id and cur_seq:
                    dic_orf_seqs_fallback[cur_id] = cur_seq
                    dic_orf_seqs_fallback[cur_base] = cur_seq
                cur_id = line[1:].strip()
                cur_base = cur_id.split("(")[0]
                cur_seq = ""
            else:
                cur_seq += line
        if cur_id and cur_seq:
            dic_orf_seqs_fallback[cur_id] = cur_seq
            dic_orf_seqs_fallback[cur_base] = cur_seq
    return dic_orf_seqs_fallback


def _matches_known_orf(source_header: str, known_orf_ids: set[str]) -> bool:
    normalized_header = normalize_orf_header(source_header)
    if normalized_header in known_orf_ids:
        return True
    base_header = normalized_header.split("(")[0] if "(" in normalized_header else normalized_header
    return base_header in known_orf_ids


def _raise_for_unmatched_search_ids(
    source_ids: set[str],
    known_orf_ids: set[str],
) -> None:
    if not source_ids:
        return
    if any(_matches_known_orf(source_id, known_orf_ids) for source_id in source_ids):
        return

    sample_ids = ", ".join(sorted(source_ids)[:5])
    raise MStoCIRC2Error(
        "Translation evaluation could not match any protein identifiers from the supplied search "
        "result table to the provided circRNA ORF/mapping files. This usually means the `-ms/--ms` "
        "input is a generic linear-proteome peptide table instead of a circRNA-aware FragPipe/MStoCIRC2 "
        f"search result. Example unmatched identifiers: {sample_ids}"
    )


def circ_orf_tsv_peptide_map(
    circ_orf_id: str,
    corf: str,
    dic_id_peptide: dict[str, str],
    dic_circ_orf_bsj_location: dict[str, str],
) -> tuple[Any, ...]:
    if circ_orf_id not in dic_id_peptide:
        return "none", "none", "none", "none", "none", "none", "none", {}, 0, ""

    mgf_pep_score_assemble = dic_id_peptide[circ_orf_id]
    dic_pro_sore: dict[str, dict[str, int]] = {}
    dic_pro_ms_counts: dict[str, set[str]] = {}
    dic_mgf_score: dict[str, int] = {}
    list_mgf_id: list[str] = []
    stats_cross = {"pep_count": 0, "unique_pep": set(), "ms_ids": set()}
    stats_non = {"pep_count": 0, "unique_pep": set(), "ms_ids": set()}
    corf_clean = corf.strip("*").upper()
    corf_clean_il = corf_clean.replace("I", "L")
    corf_s = len(corf_clean)
    coverage_mask = [False] * corf_s
    coverage_map = ["x"] * corf_s
    bsj_location_str = dic_circ_orf_bsj_location.get(circ_orf_id.split("(")[0], "0")
    bsj_list = parse_bsj_location(bsj_location_str)
    entries = [x.strip().split("###") for x in mgf_pep_score_assemble.strip(";").split(";") if x.strip()]

    for parts in entries:
        if len(parts) < 2:
            continue
        mgf_id = parts[0]
        peptide = parts[1].upper()
        peptide_il = peptide.replace("I", "L")
        try:
            score = max(1, int(float(parts[2]))) if len(parts) >= 3 else 1
        except (TypeError, ValueError):
            score = 1
        list_mgf_id.append(mgf_id)
        dic_mgf_score[mgf_id] = dic_mgf_score.get(mgf_id, 0) + score
        dic_pro_sore.setdefault(peptide, {})
        dic_pro_sore[peptide][mgf_id] = dic_pro_sore[peptide].get(mgf_id, 0) + score
        dic_pro_ms_counts.setdefault(peptide, set()).add(mgf_id)

        start = 0
        is_crossing = False
        while True:
            idx = corf_clean_il.find(peptide_il, start)
            if idx == -1:
                break
            end = idx + len(peptide)
            for k in range(idx, min(end, corf_s)):
                coverage_mask[k] = True
            current_instance_crossing = is_peptide_crossing_bsj(idx, end, bsj_list)
            if current_instance_crossing:
                is_crossing = True
            for i in range(idx, min(end, corf_s)):
                char = corf[i]
                if current_instance_crossing:
                    coverage_map[i] = char.upper()
                elif coverage_map[i] == "x" or coverage_map[i].islower():
                    coverage_map[i] = char.lower()
            start = idx + 1

        if is_crossing:
            stats_cross["pep_count"] += score
            stats_cross["unique_pep"].add(peptide)
            stats_cross["ms_ids"].add(mgf_id)
        else:
            stats_non["pep_count"] += score
            stats_non["unique_pep"].add(peptide)
            stats_non["ms_ids"].add(mgf_id)

    pro_number_score_parts = []
    number_all = stats_cross["pep_count"] + stats_non["pep_count"]
    unique_peptide_seqs_list = []
    all_unique_peps = set(dic_pro_sore.keys())
    unique_peptide_count = len(all_unique_peps)
    for peptide in all_unique_peps:
        total_count = sum(dic_pro_sore[peptide].values())
        ms_hit_count = len(dic_pro_ms_counts.get(peptide, set()))
        avg_score = (total_count / ms_hit_count) if ms_hit_count > 0 else 0.0
        pro_number_score_parts.append(f"{peptide}|{total_count}|{avg_score:.3f}")
        seq_str = peptide.upper() if peptide in stats_cross["unique_pep"] else peptide.lower()
        unique_peptide_seqs_list.append(f"{seq_str}({ms_hit_count}|{total_count})")

    pro_number_score = ";".join(pro_number_score_parts) + ";"
    peptide_seq_str = ";".join(unique_peptide_seqs_list)
    mgf_score_all = "".join(f"{k}:{v};" for k, v in dic_mgf_score.items())
    raw_sum = len(set(list_mgf_id))
    covered_len = sum(coverage_mask)
    corf_peptide_ratio = covered_len / corf_s if corf_s > 0 else 0.0
    merged_peptide_multiple = "".join(coverage_map)
    if "*" in corf:
        merged_peptide_multiple += "*"

    merged_peptide_bsj_multiple = merged_peptide_multiple
    if bsj_list:
        seen_floats = set()
        unique_bsj = []
        for b_value in bsj_list:
            if b_value not in seen_floats:
                seen_floats.add(b_value)
                unique_bsj.append(b_value)
        pos_groups: dict[int, list[float]] = {}
        for b_value in unique_bsj:
            pos_groups.setdefault(int(b_value), []).append(b_value)
        temp_list = list(merged_peptide_multiple)
        for pos in sorted(pos_groups.keys(), reverse=True):
            if 0 < pos < len(temp_list):
                for _ in range(len(pos_groups[pos])):
                    temp_list.insert(pos, "|")
        merged_peptide_bsj_multiple = "".join(temp_list)

    detailed_stats = {
        "cross": {
            "pep_count": stats_cross["pep_count"],
            "unique_pep": len(stats_cross["unique_pep"]),
            "ms_count": len(stats_cross["ms_ids"]),
        },
        "non": {
            "pep_count": stats_non["pep_count"],
            "unique_pep": len(stats_non["unique_pep"]),
            "ms_count": len(stats_non["ms_ids"]),
        },
    }
    return (
        pro_number_score,
        merged_peptide_bsj_multiple,
        corf_peptide_ratio,
        bsj_location_str,
        raw_sum,
        number_all,
        mgf_score_all,
        detailed_stats,
        unique_peptide_count,
        peptide_seq_str,
    )


def calculate_and_write_scores(results_buffer: list[dict[str, Any]], file_path_out: str, has_info: bool) -> None:
    def get_detail_bucket(item: dict[str, Any], key: str) -> dict[str, int]:
        detailed_stats = item["stats"].get("detailed_stats") or {}
        bucket = detailed_stats.get(key) or {}
        return {
            "pep_count": int(bucket.get("pep_count", 0)),
            "unique_pep": int(bucket.get("unique_pep", 0)),
            "ms_count": int(bucket.get("ms_count", 0)),
        }

    def get_stat_list(key1: str, key2: str) -> list[int]:
        return [get_detail_bucket(d, key1)[key2] for d in results_buffer]

    cross_pep_vals = get_stat_list("cross", "pep_count")
    cross_uni_vals = get_stat_list("cross", "unique_pep")
    cross_ms_vals = get_stat_list("cross", "ms_count")
    non_pep_vals = get_stat_list("non", "pep_count")
    non_uni_vals = get_stat_list("non", "unique_pep")
    non_ms_vals = get_stat_list("non", "ms_count")
    c_p_max = max(cross_pep_vals) if cross_pep_vals else 0
    c_u_max = max(cross_uni_vals) if cross_uni_vals else 0
    c_m_max = max(cross_ms_vals) if cross_ms_vals else 0
    n_p_max = max(non_pep_vals) if non_pep_vals else 0
    n_u_max = max(non_uni_vals) if non_uni_vals else 0
    n_m_max = max(non_ms_vals) if non_ms_vals else 0
    all_orf_lengths = [d["stats"]["L"] for d in results_buffer]
    median_orf_len = float(np.median(all_orf_lengths)) if all_orf_lengths else 1.0

    def get_nonzero_percentile(values: list[int], q: int = 10, default: float = 1.0) -> float:
        nonzero = [v for v in values if v > 0]
        if not nonzero:
            return default
        return float(np.percentile(nonzero, q))

    k_c_p = get_nonzero_percentile(cross_pep_vals, q=10)
    k_c_u = get_nonzero_percentile(cross_uni_vals, q=10)
    k_c_m = get_nonzero_percentile(cross_ms_vals, q=10)
    k_n_p = get_nonzero_percentile(non_pep_vals, q=10)
    k_n_u = get_nonzero_percentile(non_uni_vals, q=10)
    k_n_m = get_nonzero_percentile(non_ms_vals, q=10)

    def mixed_score(val: int, k: float, max_val: int) -> float:
        if val <= 0 or max_val <= 0:
            return 0.0
        hill = val / (val + k) if k > 0 else 0.0
        log_ratio = math.log(1 + val) / math.log(1 + max_val)
        return 0.7 * hill + 0.3 * log_ratio

    header_cols = ["circ_ORF_ID", "search_ORF_IDs", "circRNA_ID", "gene"]
    if has_info:
        header_cols.append("strand")
    header_cols.extend(
        [
            "IRES_Up50_Center",
            "IRES_Up150_Center",
            "IRES_Down_Stop",
            "m6A_Count",
            "cORF_seq",
            "cORF_len",
            "cORF_map",
            "peptide_coverage",
            "cORF_bsj_position",
            "MS_count",
            "total_peptide_count",
            "unique_peptide_count",
            "peptide_seq",
            "ms_sample_evidence",
            "coding_function_score",
        ]
    )
    scored_rows = []
    for item in results_buffer:
        stats = item["stats"]
        ds_cross = get_detail_bucket(item, "cross")
        ds_non = get_detail_bucket(item, "non")
        score_ires = 0.15 * float(stats["IRES_SCORE"])
        score_m6a = 0.15 * (1 - math.exp(-0.15 * stats["m"]))
        len_adjustment = (stats["L"] / median_orf_len) ** 0.1 if median_orf_len > 0 else 1.0
        score_cov = 0.10 * stats["cov"] * len_adjustment
        s_c_p = mixed_score(ds_cross["pep_count"], k_c_p, c_p_max)
        s_c_u = mixed_score(ds_cross["unique_pep"], k_c_u, c_u_max)
        s_c_m = mixed_score(ds_cross["ms_count"], k_c_m, c_m_max)
        score_cross = 0.2 * s_c_p + 0.4 * s_c_u + 0.4 * s_c_m
        s_n_p = mixed_score(ds_non["pep_count"], k_n_p, n_p_max)
        s_n_u = mixed_score(ds_non["unique_pep"], k_n_u, n_u_max)
        s_n_m = mixed_score(ds_non["ms_count"], k_n_m, n_m_max)
        score_non = 0.2 * s_n_p + 0.4 * s_n_u + 0.4 * s_n_m
        total_score = score_ires + score_m6a + score_cov + 0.4 * score_cross + 0.2 * score_non

        parts = item["data_row"].strip().split("\t")
        if not has_info:
            parts.pop(4)
        base_parts = parts[: len(header_cols) - 1]
        base_line = "\t".join(base_parts)
        scored_rows.append((total_score, f"{base_line}\t{total_score:.4f}\n"))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    with Path(file_path_out).open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header_cols) + "\n")
        for _, row_text in scored_rows:
            handle.write(row_text)


def process_orf(
    circ_orf_id: str,
    circ_orf_seq: str,
    dic_ires_results: dict[str, dict[str, float]],
    precomputed: dict[str, dict[str, Any]],
    dic_m6a_results: dict[str, int],
    run_ires: bool,
) -> dict[str, Any] | None:
    data = precomputed.get(circ_orf_id)
    if not data:
        return None
    ires_data = {"p1": 0.0, "p2": 0.0, "p3": 0.0} if not run_ires else dic_ires_results.get(circ_orf_id.split()[0], {"p1": 0.0, "p2": 0.0, "p3": 0.0})
    p1 = ires_data.get("p1", 0.0)
    p2 = ires_data.get("p2", 0.0)
    p3 = ires_data.get("p3", 0.0)
    val_p1 = p1 if p1 >= 0 else 0.0
    val_p2 = p2 if p2 >= 0 else 0.0
    val_p3 = p3 if p3 >= 0 else 0.0
    weighted_ires_score = 0.5 * val_p1 + 0.3 * val_p2 + 0.2 * val_p3
    circ_m6a_count = str(dic_m6a_results.get(circ_orf_id, "0"))
    orf_len = len(circ_orf_seq.strip("*"))
    ires_str = "\t".join(
        f"{value:.4f}" if value >= 0 else "na"
        for value in (p1, p2, p3)
    )
    row_circ_predict = (
        f"{circ_orf_id}\t{data['search_orf_ids']}\t{data['circ_info_id']}\t{data['gene']}\t{data['circ_strand']}\t{ires_str}\t"
        f"{circ_m6a_count}\t{circ_orf_seq}\t{orf_len}\t{data['marge_peptide_bsj_multiple']}\t{data['corf_peptide_ratio']:.4f}\t"
        f"{data['bsj_location']}\t{data['raw_sum']}\t{data['number_all']}\t{data['unique_peptide_count']}\t{data['peptide_seq_str']}\t"
        f"{data['mgf_score_all']}"
    )
    try:
        m_val = float(circ_m6a_count)
    except Exception:
        m_val = 0.0
    stats = {
        "IRES_SCORE": weighted_ires_score,
        "m": m_val,
        "cov": float(data["corf_peptide_ratio"]),
        "L": len(circ_orf_seq.strip("*")),
        "MS": data["raw_sum"],
        "PEP": data["number_all"],
        "detailed_stats": data["detailed_stats"],
    }
    return {"data_row": row_circ_predict, "stats": stats}


def round_up_to_nice_number(n: int) -> int:
    if n <= 0:
        return 1
    s = str(int(n))
    if len(s) == 1:
        return 10
    first_digit = int(s[0])
    rest = int(s[1:])
    if rest == 0:
        return int(n)
    return (first_digit + 1) * (10 ** (len(s) - 1))


def get_top_entries(file_input: str, count: int) -> tuple[list[tuple[float, str]], list[str] | None]:
    entries: list[tuple[float, str]] = []
    try:
        with Path(file_input).open("r", encoding="utf-8", errors="ignore") as handle:
            header = handle.readline()
            header_parts = header.strip().split("\t")
            try:
                idx_score = header_parts.index("coding_function_score")
            except ValueError:
                idx_score = len(header_parts) - 1
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) <= idx_score:
                    continue
                try:
                    score = float(parts[idx_score])
                except Exception:
                    score = 0.0
                entries.append((score, line))
    except Exception as exc:
        log.warning("Failed to read result file: %s", exc)
        return [], None
    entries.sort(key=lambda x: x[0], reverse=True)
    return entries[:count], header_parts


def calculate_global_max_stats(
    top_entries: list[tuple[float, str]],
    dic_id_peptide_ref: dict[str, str],
    dic_stripped_to_full: dict[str, str],
    idx_corf_id: int,
    idx_corf_seq: int,
) -> tuple[int, int]:
    max_pep_count = 0
    max_ms_count = 0
    for _, line_input in top_entries:
        parts = line_input.strip().split("\t")
        if len(parts) <= max(idx_corf_id, idx_corf_seq):
            continue
        corf_id = parts[idx_corf_id]
        key = None
        if corf_id in dic_id_peptide_ref:
            key = corf_id
        elif corf_id in dic_stripped_to_full:
            key = dic_stripped_to_full[corf_id]
        else:
            base = corf_id.split("(")[0]
            if base in dic_id_peptide_ref:
                key = base
            elif base in dic_stripped_to_full:
                key = dic_stripped_to_full[base]
        if not key:
            continue
        pep_info = dic_id_peptide_ref[key]
        corf_seq = parts[idx_corf_seq].strip("*")
        seq_len = len(corf_seq)
        counts = [0] * seq_len
        ms_sets = [set() for _ in range(seq_len)]
        for entry in pep_info.strip(";").split(";"):
            if not entry:
                continue
            subparts = entry.split("###")
            if len(subparts) < 2:
                continue
            mgf_id = subparts[0]
            pep_seq = subparts[1]
            start = 0
            while True:
                idx = corf_seq.find(pep_seq, start)
                if idx == -1:
                    break
                end = idx + len(pep_seq)
                for i in range(idx, end):
                    if i < seq_len:
                        counts[i] += 1
                        ms_sets[i].add(mgf_id)
                start = idx + 1
        if counts:
            max_pep_count = max(max_pep_count, max(counts))
        current_max_ms = max(len(s) for s in ms_sets) if ms_sets else 0
        max_ms_count = max(max_ms_count, current_max_ms)
    return max_pep_count, max_ms_count


def draw_msgf_second(file_input: str, excellent_count: int, file_out: str, dic_id_peptide: dict[str, str]) -> None:
    if not Path(file_input).exists():
        return
    peptide_map_dir = Path(file_out) / "Peptide_coverage_map"
    peptide_map_dir.mkdir(parents=True, exist_ok=True)
    top_entries, header_parts = get_top_entries(file_input, excellent_count)
    if not top_entries or header_parts is None:
        return

    try:
        idx_corf_id = header_parts.index("circ_ORF_ID")
        idx_circ_gene = header_parts.index("gene")
        idx_corf_seq = header_parts.index("cORF_seq")
        idx_seq_cov = header_parts.index("peptide_coverage")
        idx_bsj_pos = header_parts.index("cORF_bsj_position")
    except ValueError:
        idx_corf_id, idx_circ_gene, idx_corf_seq, idx_seq_cov, idx_bsj_pos = 0, 3, 8, 10, 11

    dic_stripped_to_full = {key.split("(")[0]: key for key in dic_id_peptide if key.split("(")[0] != key}
    global_max_pep_count, global_max_ms_count = calculate_global_max_stats(
        top_entries,
        dic_id_peptide,
        dic_stripped_to_full,
        idx_corf_id,
        idx_corf_seq,
    )
    scale_max_pep = round_up_to_nice_number(global_max_pep_count) if global_max_pep_count > 0 else 1
    scale_max_ms = round_up_to_nice_number(global_max_ms_count) if global_max_ms_count > 0 else 1

    for _, line_input in top_entries:
        parts = line_input.strip().split("\t")
        if len(parts) <= max(idx_corf_id, idx_circ_gene, idx_corf_seq, idx_seq_cov, idx_bsj_pos):
            continue
        corf_id = parts[idx_corf_id]
        circ_gene = parts[idx_circ_gene]
        corf_seq = parts[idx_corf_seq]
        try:
            float_circ_seq_coverage = float(parts[idx_seq_cov]) * 100
        except Exception:
            float_circ_seq_coverage = 0.0

        circ_orf_id_key = None
        if corf_id in dic_id_peptide:
            circ_orf_id_key = corf_id
        elif corf_id in dic_stripped_to_full:
            circ_orf_id_key = dic_stripped_to_full[corf_id]
        else:
            base_id = corf_id.split("(")[0]
            if base_id in dic_id_peptide:
                circ_orf_id_key = base_id
            elif base_id in dic_stripped_to_full:
                circ_orf_id_key = dic_stripped_to_full[base_id]
        if not circ_orf_id_key:
            continue

        pep_info_str = dic_id_peptide[circ_orf_id_key]
        corf_seq_clean = corf_seq.strip("*")
        corf_seq_len = len(corf_seq_clean)
        site_pep_counts = [0] * corf_seq_len
        site_ms_sets = [set() for _ in range(corf_seq_len)]
        for entry in pep_info_str.strip(";").split(";"):
            if not entry:
                continue
            subparts = entry.split("###")
            if len(subparts) < 2:
                continue
            mgf_id = subparts[0]
            pep_seq = subparts[1]
            start = 0
            while True:
                idx = corf_seq_clean.find(pep_seq, start)
                if idx == -1:
                    break
                end = idx + len(pep_seq)
                for i in range(idx, end):
                    if i < corf_seq_len:
                        site_pep_counts[i] += 1
                        site_ms_sets[i].add(mgf_id)
                start = idx + 1

        if corf_seq_len <= 20:
            column_num = max(corf_seq_len, 10)
        else:
            column_num = max(20, min(60, corf_seq_len // 10))
        corf_list_int = corf_seq_len // column_num
        corf_list_remainder = corf_seq_len - corf_list_int * column_num
        x = list(range(column_num)) * corf_list_int + list(range(corf_list_remainder))
        yyy_list = list(range(1, corf_list_int + 1))
        yyy_list.sort(reverse=True)
        y_list = []
        for yy in yyy_list:
            y_list.extend([yy] * column_num)
        y = y_list + [0] * corf_list_remainder

        norm = colors.Normalize(vmin=0, vmax=scale_max_pep)
        cmap = LinearSegmentedColormap.from_list("custom_teal", ["#13bbe9", "#035166"])
        fig_width = max(12, column_num * 0.4)
        fig = plt.figure(figsize=(fig_width, 10))
        plt.subplots_adjust(right=0.8)
        ax = fig.add_subplot(111)
        font_size = max(4, min(10, 300 / column_num))
        for aa in range(len(corf_seq_clean)):
            if aa < len(x) and aa < len(y):
                ax.text(x[aa], y[aa], corf_seq_clean[aa], fontsize=font_size, color="red", ha="center", va="center", zorder=10)
        try:
            for pos_str in parts[idx_bsj_pos].split("|"):
                try:
                    pos = int(float(pos_str))
                    if pos < len(x) and pos < len(y):
                        ax.text(x[pos] - 0.5, y[pos], "|", fontsize=600 / column_num, color="black", ha="center", va="center", zorder=11)
                except Exception:
                    continue
        except Exception:
            pass

        base_s = 75
        max_s = 750
        valid_indices = [i for i, c in enumerate(site_pep_counts) if c > 0]
        if valid_indices:
            plot_x = [x[i] for i in valid_indices]
            plot_y = [y[i] for i in valid_indices]
            plot_c = [site_pep_counts[i] for i in valid_indices]
            plot_ms = [len(site_ms_sets[i]) for i in valid_indices]
            plot_s = []
            plot_c_clipped = []
            for ms_c in plot_ms:
                effective_ms = min(ms_c, scale_max_ms)
                ratio = effective_ms / scale_max_ms if scale_max_ms > 0 else 0
                plot_s.append(base_s + ratio * (max_s - base_s))
            for pep_c in plot_c:
                plot_c_clipped.append(min(pep_c, scale_max_pep))

            segments = []
            line_widths = []
            line_colors = []
            for k in range(len(valid_indices) - 1):
                idx_curr = valid_indices[k]
                idx_next = valid_indices[k + 1]
                if idx_next == idx_curr + 1 and plot_y[k] == plot_y[k + 1]:
                    segments.append([(plot_x[k], plot_y[k]), (plot_x[k + 1], plot_y[k + 1])])
                    avg_s = (plot_s[k] + plot_s[k + 1]) / 2
                    line_widths.append(2 * np.sqrt(avg_s / np.pi))
                    line_colors.append((plot_c_clipped[k] + plot_c_clipped[k + 1]) / 2)
            if segments:
                lc = LineCollection(segments, linewidths=line_widths, cmap=cmap, norm=norm, alpha=0.9, zorder=4)
                lc.set_array(np.array(line_colors))
                ax.add_collection(lc)
            sc = ax.scatter(plot_x, plot_y, s=plot_s, marker="o", c=plot_c_clipped, norm=norm, cmap=cmap, alpha=0.9, zorder=5)
            cax = fig.add_axes([0.82, 0.45, 0.02, 0.3])
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label("Coverage Peptide Count", fontsize=10)
            ticks = [scale_max_pep * 0.3, scale_max_pep * 0.6, scale_max_pep * 0.9]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(t)}" for t in ticks])
            legend_elements = []
            for t in [0.2, 0.4, 0.6, 0.8]:
                val = int(scale_max_ms * t)
                if val == 0:
                    continue
                size = base_s + t * (max_s - base_s)
                legend_elements.append(
                    plt.scatter([], [], s=size, facecolors="none", edgecolors="#035166", alpha=0.9, marker="o", label=f"{val}")
                )
            if legend_elements:
                ax.legend(handles=legend_elements, title="Coverage Peptide-MS Count", bbox_to_anchor=(0.95, 0.08), loc="lower left", borderaxespad=0.0, labelspacing=1.5, frameon=False)

        ax.set_xlim(xmin=-1, xmax=column_num + 1)
        y_max = 2 if corf_list_int <= 20 else corf_list_int * 0.1
        ax.set_ylim(ymax=corf_list_int + y_max, ymin=-y_max * 2)
        ax.text(1, -y_max / 2, f"cORF:{corf_id}", fontsize=12)
        ax.text(1, -y_max, f"Gene_name:{circ_gene}", fontsize=12)
        ax.text(1, -y_max / 0.66, f"Sequence cover:{float_circ_seq_coverage:.2f}%", fontsize=12)
        ax.axis("off")
        match = re.search(r"(.*-ORF\d+)_", corf_id)
        save_name = match.group(1) + "_coverage" if match else corf_id.replace("|", "_").replace(":", "_")
        plt.savefig(peptide_map_dir / f"{save_name}.pdf", bbox_inches="tight")
        plt.close()


def draw_circ_corf(
    file_out: str,
    circ_predict_path: str,
    dic_output_to_source: dict[str, str],
    dic_circ_id_seq: dict[str, str],
) -> None:
    if not Path(circ_predict_path).exists():
        return
    top_entries, header_parts = get_top_entries(circ_predict_path, 10)
    if not top_entries or header_parts is None:
        return

    try:
        idx_circ_gene = header_parts.index("gene")
        idx_full_orf = header_parts.index("circ_ORF_ID")
        idx_corf_seq = header_parts.index("cORF_seq")
        idx_corf_map = header_parts.index("cORF_map")
    except ValueError:
        idx_circ_gene, idx_full_orf, idx_corf_seq, idx_corf_map = 3, 0, 8, 10

    temp_request_map: dict[str, list[str]] = {}
    for _, row_circ_predict in top_entries:
        parts = row_circ_predict.strip().split("\t")
        if len(parts) <= max(idx_circ_gene, idx_full_orf, idx_corf_seq, idx_corf_map):
            continue
        circ_gene = parts[idx_circ_gene]
        full_orf_id = parts[idx_full_orf]
        circ_corf_seq = parts[idx_corf_seq]
        circ_corf_seq_x = parts[idx_corf_map]
        value = f"{circ_gene}###{circ_corf_seq}###{circ_corf_seq_x}###{full_orf_id}"
        base_key = full_orf_id.split("-")[0]
        keys_to_try = [base_key]
        for prefix in ["P_base_", "P_self_", "F_base_", "F_self_", "F_exon_", "R_base_", "R_self_", "R_exon_", "F_noexonic_", "R_noexonic_", "F_unknown_", "R_unknown_"]:
            if base_key.startswith(prefix):
                keys_to_try.append(base_key[len(prefix) :])
                break
        for key in keys_to_try:
            temp_request_map.setdefault(key, [])
            if value not in temp_request_map[key]:
                temp_request_map[key].append(value)

    dic_circ_corf_txt = {
        key: ";".join(value)
        for key, value in temp_request_map.items()
        if key in dic_circ_id_seq
    }
    circ_map_dir = Path(file_out) / "circRNA_ORF_map"
    circ_map_dir.mkdir(parents=True, exist_ok=True)

    for circ_rna_key, orf_blob in dic_circ_corf_txt.items():
        circ_sequence = dic_circ_id_seq[circ_rna_key]
        circ_seq_len = len(circ_sequence)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        limit = 6.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axis("off")
        radius = 5.0
        theta = np.linspace(0, 2 * np.pi, 1000)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="black", linewidth=4.5, zorder=1)
        ax.plot(0, radius, marker=">", markersize=18, color="black", clip_on=False, zorder=10)
        ax.text(0, radius + 0.6, "BSJ", ha="center", va="center", fontsize=12, fontweight="bold", color="black")

        for percent in [0.2, 0.4, 0.6, 0.8]:
            bp_val = int(round(circ_seq_len * percent / 10.0)) * 10
            angle = np.pi / 2 - (percent * 2 * np.pi)
            x_out = radius * np.cos(angle)
            y_out = radius * np.sin(angle)
            x_in = (radius - 0.2) * np.cos(angle)
            y_in = (radius - 0.2) * np.sin(angle)
            ax.plot([x_out, x_in], [y_out, y_in], color="black", linewidth=1.5)
            text_r = radius - 0.6
            ax.text(text_r * np.cos(angle), text_r * np.sin(angle), f"{bp_val}bp", ha="center", va="center", fontsize=6, fontweight="bold", family="sans-serif", rotation=0)

        gene_name_display = "NA"
        has_gene = False
        for orf_info in orf_blob.split(";"):
            parts_orf = orf_info.split("###")
            if len(parts_orf) < 4:
                continue
            c_gene, c_seq, c_map, c_full_id = parts_orf[0], parts_orf[1], parts_orf[2], parts_orf[3]
            if c_gene not in ["NA", "nan", "None", "", "unknown"]:
                gene_name_display = c_gene
                has_gene = True
            source_orf_id = dic_output_to_source.get(c_full_id, c_full_id)
            match = re.search(r"\((-?\d+),(-?\d+)\)", source_orf_id)
            if not match or circ_seq_len == 0:
                continue
            start_pos = int(match.group(1))
            orf_start_abs = start_pos % circ_seq_len
            base_r_orf = 4.2
            r_drop_per_turn = 0.36
            points = []
            coverage_status = []
            clean_map = c_map.replace("|", "")
            for k in range(len(c_seq)):
                dist_from_start_nt = k * 3
                current_abs_nt = (orf_start_abs + dist_from_start_nt) % circ_seq_len
                angle = np.pi / 2 - (current_abs_nt / circ_seq_len) * 2 * np.pi
                radius_now = base_r_orf - (dist_from_start_nt / circ_seq_len) * r_drop_per_turn
                points.append((radius_now * np.cos(angle), radius_now * np.sin(angle)))
                if k < len(clean_map) and clean_map[k].isalpha() and clean_map[k].lower() != "x":
                    coverage_status.append(1)
                else:
                    coverage_status.append(0)
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color="#E0E0E0", linewidth=6, zorder=2, solid_capstyle="round")
            current_segment_indices = []
            for i, covered in enumerate(coverage_status):
                if covered == 1:
                    current_segment_indices.append(i)
                else:
                    if current_segment_indices:
                        seg_xs = [points[ix][0] for ix in current_segment_indices]
                        seg_ys = [points[ix][1] for ix in current_segment_indices]
                        if len(current_segment_indices) == 1:
                            ax.scatter(seg_xs[0], seg_ys[0], color="#13bbe9", s=40, zorder=3)
                        else:
                            ax.plot(seg_xs, seg_ys, color="#13bbe9", linewidth=6, zorder=3, solid_capstyle="round")
                    current_segment_indices = []
            if current_segment_indices:
                seg_xs = [points[ix][0] for ix in current_segment_indices]
                seg_ys = [points[ix][1] for ix in current_segment_indices]
                if len(current_segment_indices) == 1:
                    ax.scatter(seg_xs[0], seg_ys[0], color="#13bbe9", s=40, zorder=3)
                else:
                    ax.plot(seg_xs, seg_ys, color="#13bbe9", linewidth=6, zorder=3, solid_capstyle="round")

        if not has_gene:
            ax.text(0, 0, circ_rna_key, ha="center", va="center", fontsize=14, fontweight="bold", color="black")
        else:
            ax.text(0, 0.5, circ_rna_key, ha="center", va="center", fontsize=14, fontweight="bold", color="black")
            ax.text(0, -0.5, gene_name_display, ha="center", va="center", fontsize=14, style="italic", color="black")

        legend_elements = [
            Line2D([0], [0], color="black", lw=4.5, label="circRNA"),
            Line2D([0], [0], color="#E0E0E0", lw=6, label="ORF(Predicted)"),
            Line2D([0], [0], color="#13bbe9", lw=6, label="ORF(peptide)"),
        ]
        font_prop = fm.FontProperties(family=_PLOT_FONT_FAMILY, size=10)
        ax.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, prop=font_prop)
        save_name = circ_rna_key.replace("|", "_").replace(":", "_")
        plt.savefig(circ_map_dir / f"{save_name}_circ.pdf", bbox_inches="tight")
        plt.close()


def resolve_predictor_path(explicit_path: str, marker_file: str, *candidates: Path) -> str:
    if explicit_path:
        path = Path(explicit_path)
        return str(path) if (path / marker_file).exists() else ""
    for candidate in candidates:
        if (candidate / marker_file).exists():
            return str(candidate)
    return ""


def run_translation_validation(
    circ_seq: str,
    circ_info_omit_file: str | None,
    circ_orf: str,
    circ_mapping: str | None,
    path_ms_input: str,
    file_input_protein: str,
    file_out: str,
    deepcip_path: str,
    deepcip_python: str,
    deepcircm6a_path: str,
) -> int:
    output_dir = Path(file_out)
    output_dir.mkdir(parents=True, exist_ok=True)

    dic_output_to_source, dic_source_aa, dic_source_orf_coords = load_mapping_context(circ_mapping)
    dic_circ_id_seq = read_circ_sequences(circ_seq)

    dic_id_peptide = process_ms_directory(path_ms_input, str(output_dir), file_input_protein)
    dic_removed_line_peptides = {}
    if file_input_protein != "none":
        dic_id_peptide, dic_removed_line_peptides = tsv_remove_line_protaid(file_input_protein, str(output_dir))
        if dic_id_peptide:
            all_peptides_set = set()
            for info_str in dic_id_peptide.values():
                for entry in info_str.strip(";").split(";"):
                    if not entry:
                        continue
                    parts = entry.split("###")
                    if len(parts) >= 2:
                        all_peptides_set.add(parts[1])
            if all_peptides_set:
                blacklisted_peptides = batch_diamond_alignment_filter(list(all_peptides_set), file_input_protein, str(output_dir))
                if blacklisted_peptides:
                    cleaned_dic_id_peptide = {}
                    for circ_id, info_str in dic_id_peptide.items():
                        new_entries = []
                        for entry in info_str.strip(";").split(";"):
                            if not entry:
                                continue
                            parts = entry.split("###")
                            if len(parts) >= 2 and parts[1] in blacklisted_peptides:
                                continue
                            new_entries.append(entry)
                        if new_entries:
                            cleaned_dic_id_peptide[circ_id] = ";".join(new_entries)
                    dic_id_peptide = cleaned_dic_id_peptide

    dic_id_peptide, dic_source_search_orfs = map_to_source_orfs(dic_id_peptide, dic_output_to_source)
    dic_circ_orf_bsj_location = build_source_bsj_location(circ_orf, circ_mapping or "")
    dic_circ_strand = circ_forward_reverse(circ_info_omit_file)
    dic_orf_seqs_fallback = load_orf_fallback_sequences(circ_orf)
    known_orf_ids = set(dic_source_aa.keys()) | set(dic_orf_seqs_fallback.keys())

    valid_source_ids = set(dic_id_peptide.keys())
    _raise_for_unmatched_search_ids(valid_source_ids, known_orf_ids)
    precomputed: dict[str, dict[str, Any]] = {}
    candidate_orfs: list[tuple[str, str]] = []
    for source_header in valid_source_ids:
        source_seq = dic_source_aa.get(source_header, "")
        if not source_seq:
            source_base = source_header.split("(")[0]
            source_seq = dic_orf_seqs_fallback.get(source_header, "") or dic_orf_seqs_fallback.get(source_base, "")
        if not source_seq:
            for out_id in dic_source_search_orfs.get(source_header, []):
                source_seq = dic_orf_seqs_fallback.get(out_id, "") or dic_orf_seqs_fallback.get(out_id.split("(")[0], "")
                if source_seq:
                    break
        if not source_seq:
            continue

        (
            pro_number_score,
            marge_peptide_bsj_multiple,
            corf_peptide_ratio,
            _,
            raw_sum,
            number_all,
            mgf_score_all,
            detailed_stats,
            unique_peptide_count,
            peptide_seq_str,
        ) = circ_orf_tsv_peptide_map(source_header, source_seq, dic_id_peptide, dic_circ_orf_bsj_location)
        if marge_peptide_bsj_multiple == "none" and corf_peptide_ratio == "none":
            continue

        colon_parts = source_header.split(":")
        gene = colon_parts[1] if len(colon_parts) >= 3 and colon_parts[1] else "unknown"
        base_id = source_header.split("(")[0] if "(" in source_header else source_header
        orf_base = base_id.split("-")[0] if "-" in base_id else base_id
        circ_info_id = extract_circ_info_id(orf_base)
        circ_strand = dic_circ_strand.get(circ_info_id, "NA")
        bsj_location = dic_circ_orf_bsj_location.get(source_header.split("(")[0], "0")
        search_orf_str = ",".join(dic_source_search_orfs.get(source_header, [source_header]))
        precomputed[source_header] = {
            "pro_number_score": pro_number_score,
            "marge_peptide_bsj_multiple": marge_peptide_bsj_multiple,
            "corf_peptide_ratio": corf_peptide_ratio,
            "bsj_location": bsj_location,
            "raw_sum": raw_sum,
            "number_all": number_all,
            "mgf_score_all": mgf_score_all,
            "detailed_stats": detailed_stats,
            "circ_info_id": circ_info_id,
            "circ_strand": circ_strand,
            "unique_peptide_count": unique_peptide_count,
            "peptide_seq_str": peptide_seq_str,
            "gene": gene,
            "search_orf_ids": search_orf_str,
        }
        candidate_orfs.append((source_header, source_seq))

    if not candidate_orfs:
        raise MStoCIRC2Error(
            "Translation evaluation found no circRNA ORFs with retained peptide evidence after "
            "mass-spec parsing and canonical-protein filtering, so `circ_predict.txt` was not generated."
        )

    ms_orf_report_file = output_dir / "ms_sample_orf_identification.txt"
    all_ms_files = set()
    ms_to_orfs: dict[str, set[str]] = {}
    for orf_id, _ in candidate_orfs:
        if orf_id not in dic_id_peptide:
            continue
        for entry in dic_id_peptide[orf_id].strip(";").split(";"):
            if not entry:
                continue
            parts = entry.split("###")
            if len(parts) < 2:
                continue
            ms_id = parts[0]
            all_ms_files.add(ms_id)
            ms_to_orfs.setdefault(ms_id, set()).add(orf_id)
    with ms_orf_report_file.open("w", encoding="utf-8") as handle:
        handle.write("MS_Sample_ID\tIdentified_ORF_Count\tORF_IDs(semicolon_separated)\n")
        for ms_id in sorted(all_ms_files):
            orfs = ms_to_orfs.get(ms_id, set())
            handle.write(f"{ms_id}\t{len(orfs)}\t{';'.join(sorted(orfs)) if orfs else ''}\n")

    run_ires = bool(deepcip_path)
    dic_ires_results = batch_ires_predict(
        [oid for oid, _ in candidate_orfs],
        f"{output_dir.as_posix()}/",
        deepcip_path,
        deepcip_python,
        dic_source_orf_coords,
        dic_circ_id_seq,
    ) if run_ires else {}
    dic_m6a_results = batch_m6a_predict(
        candidate_orfs,
        f"{output_dir.as_posix()}/",
        deepcircm6a_path,
        dic_source_orf_coords,
        dic_circ_id_seq,
    ) if deepcircm6a_path else {}

    results_buffer = []
    for orf_id, orf_seq in candidate_orfs:
        result = process_orf(orf_id, orf_seq, dic_ires_results, precomputed, dic_m6a_results, run_ires)
        if result:
            results_buffer.append(result)

    circ_predict_path = output_dir / "circ_predict.txt"
    has_info = bool(circ_info_omit_file and Path(circ_info_omit_file).exists())
    calculate_and_write_scores(results_buffer, str(circ_predict_path), has_info)
    if not circ_predict_path.exists():
        raise MStoCIRC2Error(
            f"Translation evaluation did not generate the expected result file: '{circ_predict_path}'."
        )
    draw_msgf_second(str(circ_predict_path), 10, str(output_dir), dic_id_peptide)
    draw_circ_corf(str(output_dir), str(circ_predict_path), dic_output_to_source, dic_circ_id_seq)
    log.info("Evaluation stage completed: %s", circ_predict_path)
    return 0
