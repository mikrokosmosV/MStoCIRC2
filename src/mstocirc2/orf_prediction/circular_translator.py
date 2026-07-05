from pathlib import Path
from typing import Dict, List, Any, Sequence

from ..core import TerminalProgressBar
from .genetic_codes import PROTEIN_TABLE_STANDARD, PROTEIN_TABLE_MT
import logging

log = logging.getLogger(__name__)


def is_mt_chromosome(identifier: str) -> bool:
    return any(m in identifier.upper() for m in ['CHRMT', 'CHRMIT', 'MT_', '_MT', 'MITOCHONDRION'])

def RNA_protein(
    RNA_string: str,
    min_len: int,
    flank_len: int,
    is_mt_chr: bool,
    start_codons: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Biological model of Rolling Circle Translation.
    Mirrors continuous loop transcription to bridge the BSJ.
    """
    seq_len = len(RNA_string)
    RNA_string_extended = RNA_string * 2 + RNA_string[:flank_len * 3]
    protein_table = PROTEIN_TABLE_MT if is_mt_chr else PROTEIN_TABLE_STANDARD
    candidates = []
    start_positions = []; search_seq = RNA_string + RNA_string[:2]
    
    for sc in start_codons:
        pos = search_seq.find(sc)
        while pos != -1:
            if pos < seq_len: start_positions.append((pos, sc))
            pos = search_seq.find(sc, pos + 1)
    start_positions.sort()
    
    for start, sc_used in start_positions:
        aa_seq = []; is_rct = False; k = 0
        max_nt_len = seq_len + flank_len * 3; current_nt_len = 0
        while True:
            codon = RNA_string_extended[start + k: start + k + 3]
            if len(codon) < 3: break
            aa = 'X' if ('N' in codon or 'n' in codon) else protein_table.get(codon, 'X')
            if aa == '*': aa_seq.append(aa); current_nt_len += 3; break
            aa_seq.append(aa); current_nt_len += 3; k += 3
            if current_nt_len >= max_nt_len: is_rct = True; break
            
        effective_len = len(aa_seq)
        if aa_seq and aa_seq[-1] == '*': effective_len -= 1
        if effective_len < min_len: continue
        
        effective_nt_len = current_nt_len
        if aa_seq and aa_seq[-1] == '*': effective_nt_len -= 3
        if start + effective_nt_len <= seq_len: continue
        
        breakpoints = []; j1_dist = seq_len - start; current_j_dist = j1_dist
        while current_j_dist < effective_nt_len:
            aa_idx = (current_j_dist // 3) + 1; rem = current_j_dist % 3
            if aa_idx <= 0: aa_idx = 1
            bp_str = f"{aa_idx-1}.3" if rem == 0 else f"{aa_idx}.{rem}"
            breakpoints.append(bp_str); current_j_dist += seq_len
            
        bp_string = "|".join(breakpoints)
        name_suffix = "_" + sc_used
        if is_rct: name_suffix += "_RCT"
        candidates.append({
            'start': start, 'end': start + current_nt_len, 
            'aa_seq': "".join(aa_seq), 'nt_seq': RNA_string_extended[start:start+current_nt_len], 
            'breakpoints': bp_string, 'suffix': name_suffix
        })
    return candidates

def translate_circ_orfs(
    fasta_path: str,
    min_orf_len: int,
    flank_aa_len: int,
    start_codons: Sequence[str],
) -> List[Dict[str, Any]]:
    log.info('=' * 50)
    log.info('[STEP] ORF Prediction')
    log.info('=' * 50)

    fasta_file = Path(fasta_path)
    if not fasta_file.exists():
        raise FileNotFoundError(f"CircRNA FASTA file not found: '{fasta_file}'.")

    with fasta_file.open("r", encoding="utf-8", errors="ignore") as counter_handle:
        total_sequences = sum(1 for line in counter_handle if line.startswith('>'))
    progress = TerminalProgressBar(total_sequences, "[ORF] Sequence progress")
    raw_records = []
    total_seq_count = 0

    def _append_records(header: str, sequence: str) -> None:
        nonlocal total_seq_count
        is_mt = is_mt_chromosome(header)
        for i, c in enumerate(RNA_protein(sequence, min_orf_len, flank_aa_len, is_mt, start_codons)):
            aa, nt = c['aa_seq'], c['nt_seq']
            if aa.endswith('*') and len(nt) >= 3:
                aa, nt = aa[:-1], nt[:-3]
            parts = header[1:].rsplit(':', 1)
            base = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            gene = rest[:-2] if rest.endswith(('_F', '_R')) else rest
            h = base + '-ORF' + str(i + 1) + c['suffix'] + '(' + str(c['start']) + ',' + str(c['end']) + ')'
            h += ':' + (gene if gene else '') + ':' + c['breakpoints']
            raw_records.append({
                'header': h, 'sequence': aa, 'nt_sequence': nt, 'order': len(raw_records),
                '_base': base, '_gene': gene if gene else '', '_start': c['start'],
                '_end': c['end'], '_bps': c['breakpoints'], '_suffix': c['suffix']
            })
        total_seq_count += 1
        progress.update(total_seq_count)

    with fasta_file.open("r", encoding="utf-8", errors="ignore") as fi:
        cur_h, cur_s = '', ''
        for line in fi:
            line = line.strip()
            if not line: continue
            if line[0] == '>':
                if cur_h:
                    _append_records(cur_h, cur_s)
                cur_h, cur_s = line, ''
            else: cur_s += line.upper()
        if cur_h:
            _append_records(cur_h, cur_s)
    progress.close()

    log.info("[ORF] Processed %d sequences. Total raw ORFs found: %d", total_seq_count, len(raw_records))
    return raw_records
