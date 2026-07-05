from pathlib import Path
import subprocess
from typing import Dict, Set, Tuple
import logging
log = logging.getLogger(__name__)


def normalize_prot_id(pid: str) -> str:
    p = pid.strip()
    if '|' in p:
        p = p.split('|')[-1]
    p = p.split()[0]
    if '.' in p:
        p = p.split('.')[0]
    return p.upper()

def normalize_circ_key(cid: str) -> str:
    return cid.strip()

KMER_SIZE_LINEAR = 6

def tsv_remove_line_protaid(file_input_protein: str, file_out: str) -> Tuple[Dict[str, str], Dict[str, list]]:
    dic_removed_line_peptides = {}
    file_input_protein_path = Path(file_input_protein)
    output_dir = Path(file_out)
    current_id = None
    current_seq = ''
    list_line_entries = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with file_input_protein_path.open("r", encoding="utf-8", errors="ignore") as file_line_protide:
        for line_protide in file_line_protide:
            if line_protide[0] == '>':
                if current_id is not None:
                    list_line_entries.append((current_id, current_seq, current_seq.upper().replace('I', 'L')))
                current_id = line_protide[1:].strip().split()[0]
                current_seq = ''
            else:
                current_seq += line_protide.strip()
        if current_id is not None:
            list_line_entries.append((current_id, current_seq, current_seq.upper().replace('I', 'L')))
    
    kmer_index = {}
    for i, (prot_id, prot_seq, prot_seq_IL) in enumerate(list_line_entries):
        if len(prot_seq_IL) < KMER_SIZE_LINEAR:
            continue
        seen = set()
        for j in range(len(prot_seq_IL) - KMER_SIZE_LINEAR + 1):
            kmer = prot_seq_IL[j:j + KMER_SIZE_LINEAR]
            if kmer not in seen:
                seen.add(kmer)
                kmer_index.setdefault(kmer, set()).add(i)
                
    try:
        with (output_dir / "tsv_result.txt").open('r', encoding='utf-8', errors='ignore') as file_tsv_result, \
             (output_dir / 'tsv_result_remove_line.txt').open('w+', encoding='utf-8') as file_remove, \
             (output_dir / 'peptide_line.txt').open('w+', encoding='utf-8') as file_peptide_line:
            file_tsv_result_header = file_tsv_result.readline()
            file_remove.write(file_tsv_result_header)
            file_peptide_line.write('peptide\tlinear_protein\tMS_ID\n')

            set_line_p = set()
            set_circ_p = set()
            dic_id_peptide = {}

            from .mass_spec_parser import dic_circ_mgf_pep_add

            for line_tsv_result in file_tsv_result:
                parts = line_tsv_result.strip().split('\t')
                if len(parts) < 3:
                    continue
                mgf_id = parts[0]
                peptide = parts[1]
                circ_id = parts[2]
                peptide_IL = peptide.upper().replace('I', 'L')
                if peptide_IL in set_line_p:
                    continue
                if peptide_IL in set_circ_p:
                    file_remove.write(line_tsv_result)
                    dic_id_peptide = dic_circ_mgf_pep_add(circ_id, mgf_id, peptide, dic_id_peptide)
                    continue

                matched_prot_id = None
                matched_prot_norm = None
                if len(peptide_IL) >= KMER_SIZE_LINEAR:
                    candidate_indices = None
                    for j in range(len(peptide_IL) - KMER_SIZE_LINEAR + 1):
                        kmer = peptide_IL[j:j + KMER_SIZE_LINEAR]
                        indices = kmer_index.get(kmer)
                        if indices is None:
                            candidate_indices = set()
                            break
                        if candidate_indices is None or len(indices) < len(candidate_indices):
                            candidate_indices = indices
                        if len(candidate_indices) == 1:
                            break
                    if candidate_indices:
                        for idx in candidate_indices:
                            prot_id, prot_seq, prot_seq_IL = list_line_entries[idx]
                            if peptide_IL in prot_seq_IL:
                                matched_prot_id = prot_id
                                matched_prot_norm = normalize_prot_id(prot_id)
                                break
                    else:
                        for prot_id, prot_seq, prot_seq_IL in list_line_entries:
                            if peptide_IL in prot_seq_IL:
                                matched_prot_id = prot_id
                                matched_prot_norm = normalize_prot_id(prot_id)
                                break
                else:
                    for prot_id, prot_seq, prot_seq_IL in list_line_entries:
                        if peptide_IL in prot_seq_IL:
                            matched_prot_id = prot_id
                            matched_prot_norm = normalize_prot_id(prot_id)
                            break

                if matched_prot_id:
                    set_line_p.add(peptide_IL)
                    file_peptide_line.write(f"{peptide}\t{matched_prot_norm}\t{mgf_id}\n")
                    circ_key_full = normalize_circ_key(circ_id)
                    lst = dic_removed_line_peptides.get(circ_key_full, [])
                    lst.append((peptide, mgf_id, matched_prot_norm))
                    dic_removed_line_peptides[circ_key_full] = lst
                else:
                    set_circ_p.add(peptide_IL)
                    file_remove.write(line_tsv_result)
                    dic_id_peptide = dic_circ_mgf_pep_add(circ_id, mgf_id, peptide, dic_id_peptide)

            TOTAL_LINEAR_PEPS = len(set_line_p)
            TOTAL_CIRC_PEPS = len(set_circ_p)
            log.info(f"Total linear RNA peptides: {TOTAL_LINEAR_PEPS}")
            log.info(f"Total circular peptides: {TOTAL_CIRC_PEPS}")
            return dic_id_peptide, dic_removed_line_peptides
    except FileNotFoundError:
        log.info("[WARNING] %s does not exist", output_dir / "tsv_result.txt")
        return {}, {}

def batch_diamond_alignment_filter(peptide_list: list, file_input_protein: str, file_out: str) -> set:
    output_prefix = Path(file_out)
    output_prefix.mkdir(parents=True, exist_ok=True)
    file_input_protein_path = Path(file_input_protein)
    if not file_input_protein_path.exists():
        return set()
    if not peptide_list:
        return set()
        
    db_name = output_prefix / "eval_diamond_protein_db"
    if not db_name.with_suffix(".dmnd").exists():
        log.info("Building Diamond database...")
        db_proc = subprocess.run(
            ['diamond', 'makedb', '--in', str(file_input_protein_path), '-d', str(db_name), '--quiet'],
            check=False,
        )
        if db_proc.returncode != 0:
            log.warning(
                "Diamond database creation failed (exit %d): %s",
                db_proc.returncode,
                (getattr(db_proc, "stderr", None) or getattr(db_proc, "stdout", None) or "no output").strip(),
            )
            return set()
        
    temp_query = output_prefix / "diamond_peptides_query.fasta"
    with temp_query.open('w+', encoding='utf-8') as f:
        for i, pep in enumerate(peptide_list):
            f.write(f'>pep_{i}\n{pep}\n')
            
    temp_result = output_prefix / "diamond_filter_result.txt"
    cmd_args = ['diamond', 'blastp', '-q', str(temp_query), '-d', str(db_name), '-o', str(temp_result),
                '-f', '6', 'qseqid', 'sseqid', 'pident', 'length', 'qlen',
                '-p', '8', '-e', '0.001', '--id', '95', '--quiet']
    log.info(f"Running peptide homology scan (Diamond, {len(peptide_list)} peptides)...")
    try:
        scan_proc = subprocess.run(cmd_args, check=False)
    except OSError as e:
        log.info(f"[WARNING] Diamond execution error: {e}")
        return set()
    if scan_proc.returncode != 0:
        log.warning(
            "Diamond peptide scan failed (exit %d): %s",
            scan_proc.returncode,
            (getattr(scan_proc, "stderr", None) or getattr(scan_proc, "stdout", None) or "no output").strip(),
        )
        return set()
        
    blacklisted_indices = set()
    if temp_result.exists():
        with temp_result.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                qseqid = parts[0]
                try:
                    pident = float(parts[2])
                    aln_len = int(parts[3])
                    q_len = int(parts[4])
                except (ValueError, IndexError):
                    continue
                if q_len == 0:
                    continue
                effective_coverage = (aln_len * (pident / 100.0)) / q_len
                if effective_coverage < 0.95:
                    continue
                try:
                    idx = int(qseqid.split('_')[1])
                    blacklisted_indices.add(idx)
                except (IndexError, ValueError):
                    pass
                    
    blacklisted_peptides = set()
    for idx in blacklisted_indices:
        if idx < len(peptide_list):
            blacklisted_peptides.add(peptide_list[idx])
    log.info(f"Homology scan complete, marked {len(blacklisted_peptides)} peptides highly similar (>=95%) to linear proteins.")
    return blacklisted_peptides
