from pathlib import Path
from typing import Dict, List, Any
import logging
log = logging.getLogger(__name__)


def restore_duplicate_orfs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Restores multi-mapped ORFs mirroring biological duplicates or shared exons."""
    seq_to_records = {}
    for r in records:
        seq_to_records.setdefault(r['sequence'], []).append(r)
    dup_seqs = {seq for seq, recs in seq_to_records.items() if len(recs) > 1}
    restored = []
    for r in records:
        if r['sequence'] in dup_seqs:
            source_h = r.get('source_header', r['header'])
            source_seq = r.get('source_sequence', r['sequence'])
            source_nt = r.get('source_nt_sequence', r.get('nt_sequence', ''))
            restored.append({
                'header': source_h,
                'sequence': source_seq,
                'nt_sequence': source_nt,
                'order': r['order'],
                'source_header': source_h,
                'source_sequence': source_seq,
                'source_nt_sequence': source_nt
            })
        else:
            restored.append(r)
    return sorted(restored, key=lambda r: r['order'])

def write_final_outputs(out_circ_file: str, final_records: List[Dict[str, Any]]) -> None:
    """Write the canonical ORF outputs used by downstream modules."""
    out_dir = Path(out_circ_file)

    with (out_dir / 'circRNA_bsj_corf.fasta').open('w', encoding='utf-8') as f:
        for r in final_records:
            f.write('>' + r['header'] + '\n' + r['sequence'] + '\n')

    with (out_dir / 'circRNA_bsj_corf_nt.fasta').open('w', encoding='utf-8') as f:
        for r in final_records:
            f.write('>' + r['header'] + '\n' + r.get('nt_sequence', '') + '\n')

    with (out_dir / 'circRNA_ORF_mapping.tsv').open('w', encoding='utf-8') as f:
        f.write("Output_ORF\tSource_ORF\tSource_AA_Seq\tSource_NT_Seq\n")
        for r in final_records:
            f.write(
                f"{r['header']}\t{r.get('source_header', r['header'])}\t"
                f"{r.get('source_sequence', r['sequence'])}\t"
                f"{r.get('source_nt_sequence', r.get('nt_sequence', ''))}\n"
            )

def cleanup_intermediate_files(out_circ_file: str, use_sequence_mode: bool) -> None:
    out_dir = Path(out_circ_file)
    kept = {'circRNA_full_length.fasta', 'circRNA_all_ORF.fasta', 'circRNA_bsj_corf.fasta',
            'circRNA_bsj_corf_nt.fasta', 'circRNA_ORF_mapping.tsv'}
    if not use_sequence_mode:
        kept.add('circ_info_omit.gff')

    for path in out_dir.iterdir():
        if not path.is_file() or path.name in kept:
            continue
        path.unlink()
