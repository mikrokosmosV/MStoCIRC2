import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Seq import reverse_complement
import logging
from ..core import DependencyError
log = logging.getLogger(__name__)


def check_command_exists(cmd: str) -> None:
    import shutil
    if shutil.which(cmd) is None:
        raise DependencyError(f"Required command '{cmd}' was not found. Please install it.")


def _require_existing_file(path_value: str | None, label: str) -> Path:
    if not path_value or path_value == "none":
        raise FileNotFoundError(f"Required {label} path was not provided.")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: '{path}'.")
    return path


def _run_bedtools(command: list[str], output_name: str | None = None, stdout_handle: Any = None) -> None:
    try:
        subprocess.run(command, stdout=stdout_handle, check=True)
    except subprocess.CalledProcessError as exc:
        target = f" while writing '{output_name}'" if output_name else ""
        raise RuntimeError(
            f"bedtools command failed{target} with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc

def extract_sequences(args: Any, out_circ_file: str) -> None:
    """
    Biological alignment logic integrating genomic annotations with bedtools.
    Parses CI/Prediction outputs and generates full-length predicted circRNAs.
    """
    out_dir = Path(out_circ_file)

    def _out_path(name: str) -> Path:
        return out_dir / name

    def _out_str(name: str) -> str:
        return str(_out_path(name))

    circ_info = args.circ_info
    circ_seq_file = args.circ_seq_file
    dna_seq = args.dna_seq
    dna_gff = args.dna_gff
    find_circ = args.find_circ
    circexplorer = args.circexplorer
    CIRI = args.CIRI

    use_sequence_mode = (circ_seq_file != 'none')
    use_ci_mode = (circ_info != 'none')
    has_prediction_inputs = (find_circ != 'none' or circexplorer != 'none' or CIRI != 'none')

    if use_sequence_mode:
        _require_existing_file(circ_seq_file, "circRNA sequence")
        if use_ci_mode:
            _require_existing_file(circ_info, "circRNA annotation")
    else:
        _require_existing_file(dna_seq, "reference genome FASTA")
        _require_existing_file(dna_gff, "reference genome annotation")
        if not use_ci_mode and not has_prediction_inputs:
            raise FileNotFoundError(
                "Genome-based extraction requires '--circ-info' or at least one prediction file "
                "from find_circ, CIRCexplorer, or CIRI."
            )
        if find_circ != "none":
            _require_existing_file(find_circ, "find_circ result")
        if circexplorer != "none":
            _require_existing_file(circexplorer, "CIRCexplorer result")
        if CIRI != "none":
            _require_existing_file(CIRI, "CIRI result")
        check_command_exists("bedtools")

    file_dna_chr = ''
    if not use_sequence_mode and dna_seq is not None:
        with open(dna_seq, 'r', encoding='utf-8', errors='ignore') as f_tmp:
            for line in f_tmp:
                if line.startswith('>'): 
                    file_dna_chr = line.strip().strip('>').split()[0]; break
    has_chr_prefix = 'Chr' in file_dna_chr

    def normalize_chr(chr_name: str) -> str:
        if re.match(r'^(Chr|chr)?(\d+|[XYM]|MT|Z|W)$', chr_name, re.IGNORECASE):
            num = re.sub(r'^(Chr|chr)', '', chr_name, flags=re.IGNORECASE)
            if num.upper() not in ['X','Y','M','MT','Z','W'] and num != '0': num = str(int(num))
            if has_chr_prefix and not chr_name.lower().startswith('chr'): return 'Chr' + num
            elif not has_chr_prefix and chr_name.lower().startswith('chr'): return num
            else: return chr_name
        elif re.search(r'(Mt|MT|Pt)', chr_name, re.IGNORECASE):
            clean = re.sub(r'^Chr', '', chr_name, flags=re.IGNORECASE)
            if has_chr_prefix: return 'Chr' + clean
            return clean
        return chr_name

    def read_attr_value(attr_text: str, key_list: List[str]) -> str:
        for attr in attr_text.strip().strip(';').split(';'):
            attr = attr.strip()
            if attr == '': continue
            if '=' in attr: key, value = attr.split('=', 1)
            elif ' ' in attr: key, value = attr.split(' ', 1)
            else: continue
            if key.strip() in key_list: return value.strip().strip('"')
        return 'NA'

    dic_gff_info_gene: Dict[str, str] = {}
    dic_gff_info_s: Dict[str, str] = {}
    dic_circ_exon_info: Dict[str, List[Tuple[int, int]]] = {}
    gene_exon_db: Dict[str, List[Tuple[str, int, int, str]]] = {}

    def dic_gff_info_gene_s(circ_gff: str, circ_gene: str, circ_s: str) -> None:
        if circ_gff not in dic_gff_info_gene: dic_gff_info_gene[circ_gff] = circ_gene
        elif dic_gff_info_gene[circ_gff] == 'NA': dic_gff_info_gene[circ_gff] = circ_gene
        if circ_gff not in dic_gff_info_s: dic_gff_info_s[circ_gff] = circ_s
        elif dic_gff_info_s[circ_gff] == 'NA': dic_gff_info_s[circ_gff] = circ_s
        elif dic_gff_info_s[circ_gff] != 'NA' and circ_s != 'NA' and dic_gff_info_s[circ_gff] != circ_s: 
            dic_gff_info_s[circ_gff] = '.'

    def parse_circexplorer_exons(cols: List[str]) -> Optional[List[Tuple[int, int]]]:
        if len(cols) < 12: return None
        try:
            chrom_start = int(cols[1]); block_count = int(cols[9])
            block_sizes = [int(x) for x in cols[10].rstrip(',').split(',')]
            block_starts = [int(x) for x in cols[11].rstrip(',').split(',')]
            if len(block_sizes) != block_count or len(block_starts) != block_count: return None
            return [(chrom_start + block_starts[i], chrom_start + block_starts[i] + block_sizes[i]) for i in range(block_count)]
        except (TypeError, ValueError, IndexError):
            return None

    # Implementations directly matching biology from original monolithic module
    def read_circexplorer_results():
        with open(circexplorer, 'r', encoding='utf-8', errors='ignore') as f:
            for row in f:
                cols = row.strip().split('\t')
                if len(cols) < 6: continue
                circ_chr = normalize_chr(cols[0]); circ_start = str(int(cols[1]) + 1); circ_end = cols[2]
                circ_s = cols[5] if cols[5] in ['-', '+'] else 'NA'
                circ_gene = 'NA'
                if len(cols) > 14: g = cols[14]; circ_gene = g.split(':')[1] if ':' in g else g
                elif len(cols) > 13: g = cols[13]; circ_gene = g.split(':')[1] if ':' in g else g
                circ_gff = circ_chr + '\t' + circ_start + '\t' + circ_end + '\t'
                dic_gff_info_gene_s(circ_gff, circ_gene, circ_s)
                exons = parse_circexplorer_exons(cols)
                if exons: 
                    circ_key = circ_chr + '_' + circ_start + '_' + circ_end + '_' + circ_s
                    dic_circ_exon_info[circ_key] = exons

    def read_find_circ_results():
        with open(find_circ, 'r', encoding='utf-8', errors='ignore') as f:
            for row in f:
                cols = row.strip().split('\t'); circ_chr = normalize_chr(cols[0]); circ_start = str(int(cols[1]) + 1); circ_end = cols[2]
                circ_s = cols[5] if cols[5] in ['-', '+'] else 'NA'
                dic_gff_info_gene_s(circ_chr + '\t' + circ_start + '\t' + circ_end + '\t', 'NA', circ_s)

    def read_ciri_results():
        with open(CIRI, 'r', encoding='utf-8', errors='ignore') as f:
            for row in f:
                cols = row.strip().split('\t')
                if len(cols) < 11 or 'chr' == cols[1]: continue
                circ_chr = normalize_chr(cols[1]); circ_start = cols[2]; circ_end = cols[3]
                circ_gene = cols[9].strip(',')
                if circ_gene.lower() in ['nan', 'na', '']: circ_gene = 'NA'
                elif ',' in circ_gene: circ_gene = circ_gene.split(',')[0]
                circ_s = cols[10] if cols[10] in ['-', '+'] else 'NA'
                dic_gff_info_gene_s(circ_chr + '\t' + circ_start + '\t' + circ_end + '\t', circ_gene, circ_s)

    def circ_info_omit():
        with _out_path('circ_info_omit.gff').open('w', encoding='utf-8') as f:
            num = 1
            for key in dic_gff_info_s.keys():
                circ_num = str(num).rjust(6, '0')
                if dic_gff_info_s[key] == '.':
                    f.write('circ_' + circ_num + '\t' + key + dic_gff_info_gene[key] + '\t+\n'); num += 1
                    circ_num = str(num).rjust(6, '0')
                    f.write('circ_' + circ_num + '\t' + key + dic_gff_info_gene[key] + '\t-\n'); num += 1
                else:
                    f.write('circ_' + circ_num + '\t' + key + dic_gff_info_gene[key] + '\t' + dic_gff_info_s[key] + '\n'); num += 1

    if not use_sequence_mode:
        if find_circ != 'none': read_find_circ_results()
        if circexplorer != 'none': read_circexplorer_results()
        if CIRI != 'none': read_ciri_results()
        if dic_gff_info_gene and dic_gff_info_s: circ_info_omit()
        
        with open(dna_gff, 'r', encoding='utf-8', errors='ignore') as file_dna_gff, \
             _out_path('dna_exon.gff').open('w', encoding='utf-8') as file_dna_exon_gff, \
             _out_path('dna_exon_s.gff').open('w', encoding='utf-8') as file_dna_exon_s_gff, \
             _out_path('dna_gene_id.gff').open('w', encoding='utf-8') as file_dna_omit_gene_gff:
            set_dna_gff_omit = set(); has_exon_line = False
            for line_dna_gff in file_dna_gff:
                if line_dna_gff and line_dna_gff[0] != '#':
                    line_split = line_dna_gff.strip().split('\t')
                    if len(line_split) < 8: continue
                    seq_type = line_split[2]; dna_chr = normalize_chr(line_split[0])
                    if seq_type.lower() == 'exon':
                        line_dna_gff_omit = dna_chr + '\t' + line_split[3] + '\t' + line_split[4]
                        exon_direction = line_split[6]
                        if line_dna_gff_omit not in set_dna_gff_omit:
                            set_dna_gff_omit.add(line_dna_gff_omit); has_exon_line = True
                            file_dna_exon_gff.write(line_dna_gff_omit + '\n')
                            file_dna_exon_s_gff.write(line_dna_gff_omit + '\t' + exon_direction + '\n')
                        if len(line_split) > 8:
                            gene_id = read_attr_value(line_split[8], ['gene_id', 'Parent'])
                            if gene_id != 'NA':
                                if gene_id.startswith('gene:'): gene_id = gene_id.split(':', 1)[1]
                                base_id = gene_id.split('.')[0]
                                for gid in [gene_id, base_id]: 
                                    gene_exon_db.setdefault(gid, []).append((dna_chr, int(line_split[3]), int(line_split[4]), exon_direction))
                    elif seq_type == 'gene':
                        line_attr = line_split[8] if len(line_split) > 8 else ''
                        line_gene = read_attr_value(line_attr, ['gene_id', 'ID', 'Name', 'gene', 'gene_name'])
                        if line_gene.startswith('gene:'): line_gene = line_gene.split(':', 1)[1]
                        file_dna_omit_gene_gff.write(dna_chr + '\t' + line_split[3] + '\t' + line_split[4] + '\t' + line_gene + '\n')
            if not has_exon_line: log.info('dna_gff file is error')

    def parse_ci_for_sequence_mode(ci_path: str) -> Tuple[Dict[str,str], Dict[str,str]]:
        dic_ci_gene = {}; dic_ci_strand = {}
        with open(ci_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line == '' or line[0] == '#': continue
                cols = line.split('\t')
                if len(cols) < 3: continue
                circ_id, circ_gene, circ_strand = '', 'NA', 'NA'
                if len(cols) >= 12 and re.match(r'^\d+$', cols[3]) and re.match(r'^\d+$', cols[4]):
                    circ_id = cols[0]; circ_gene = cols[7] if cols[7] != '' else 'NA'; circ_strand = cols[8] if cols[8] in ['+', '-'] else 'NA'
                elif len(cols) >= 4 and re.match(r'^\d+$', cols[2]) and re.match(r'^\d+$', cols[3]):
                    circ_id = cols[0]
                    if len(cols) >= 6:
                        if cols[4] in ['+', '-']: circ_strand = cols[4]; circ_gene = cols[5] if cols[5] != '' else 'NA'
                        elif cols[5] in ['+', '-']: circ_strand = cols[5]; circ_gene = cols[4] if cols[4] != '' else 'NA'
                        else: circ_gene = cols[4] if cols[4] != '' else 'NA'
                elif len(cols) >= 6 and re.match(r'^\d+$', cols[1]) and re.match(r'^\d+$', cols[2]):
                    if cols[3] not in ['', '.']: circ_id = cols[3]
                    if cols[5] in ['+', '-']: circ_strand = cols[5]
                if circ_id == '': continue
                dic_ci_gene[circ_id] = circ_gene; dic_ci_strand[circ_id] = circ_strand
        return dic_ci_gene, dic_ci_strand

    dic_id_gene = {}; dic_id_strand = {}; list_circ_unknown = []

    def write_cs_sequence(header_raw: str, seq: str, file_out: Any) -> None:
        circ_id = header_raw.split(':')[0]
        circ_gene = dic_id_gene.get(circ_id, 'none')
        if circ_gene == 'NA': circ_gene = 'none'
        if 'N' in seq: return
        file_out.write('>' + circ_id + ':' + circ_gene + '\n' + seq + '\n')


    if use_sequence_mode:
        if use_ci_mode:
            dic_ci_gene, dic_ci_strand = parse_ci_for_sequence_mode(circ_info)
            dic_id_gene.update(dic_ci_gene); dic_id_strand.update(dic_ci_strand)
            log.info('[STRD] loaded from -ci:', len(dic_ci_strand))
        with open(circ_seq_file, 'r') as file_seq, _out_path('circ_pr.txt').open('w', encoding='utf-8') as file_id_gene_seq_pr:
            circ_header_raw, circ_seq = '', ''
            for line in file_seq:
                line = line.strip()
                if line.startswith('>'):
                    if circ_header_raw != '': write_cs_sequence(circ_header_raw, circ_seq, file_id_gene_seq_pr)
                    circ_header_raw = line[1:]; circ_seq = ''
                else: circ_seq += line.upper()
            if circ_header_raw != '': write_cs_sequence(circ_header_raw, circ_seq, file_id_gene_seq_pr)
        cs_circ_count = 0
        with _out_path('circ_pr.txt').open('r', encoding='utf-8') as fin, _out_path('circ_exon_connect.fasta').open('w', encoding='utf-8') as fout:
            for line in fin:
                if line.startswith('>'):
                    cs_circ_count += 1
                    fout.write('>' + line[1:])
                else:
                    fout.write(line)
        log.info(f'[INFO] Loaded {cs_circ_count} circRNA sequences from -cs file.')

    else:
        dic_circ_bed_circ_id = {}; dic_circ_id_start_seq = {}; dic_circ_id_end_seq = {}
        line_circ_info_gff_all = ''; line_circ_info_uniq_bed_all = ''; auto_id_num = 1
        if use_ci_mode:
            with open(circ_info, 'r', encoding='utf-8', errors='ignore') as file_circ_info:
                for line_circ_info in file_circ_info:
                    line_circ_info = line_circ_info.strip()
                    if line_circ_info == '' or line_circ_info[0] == '#': continue
                    line_cols = line_circ_info.split('\t')
                    if len(line_cols) < 3: continue
                    circ_id, circ_chr_info, circ_start_info, circ_end_info = '', '', '', ''
                    circ_gene, circ_direction = 'NA', 'NA'; circ_start_seq, circ_end_seq = '', ''
                    if len(line_cols) >= 12 and re.match(r'^\d+$', line_cols[3]) and re.match(r'^\d+$', line_cols[4]):
                        circ_id = line_cols[0]; circ_chr_info = line_cols[2]; circ_start_info = line_cols[3]; circ_end_info = line_cols[4]
                        circ_gene = line_cols[7] if line_cols[7] != '' else 'NA'; circ_direction = line_cols[8] if line_cols[8] in ['+', '-'] else 'NA'
                        circ_start_seq = line_cols[11].upper(); circ_end_seq = line_cols[10].upper()
                    elif len(line_cols) >= 4 and re.match(r'^\d+$', line_cols[2]) and re.match(r'^\d+$', line_cols[3]):
                        circ_id = line_cols[0]; circ_chr_info = line_cols[1]; circ_start_info = line_cols[2]; circ_end_info = line_cols[3]
                        if len(line_cols) >= 6:
                            if line_cols[4] in ['+', '-']: circ_direction = line_cols[4]; circ_gene = line_cols[5] if line_cols[5] != '' else 'NA'
                            elif line_cols[5] in ['+', '-']: circ_gene = line_cols[4] if line_cols[4] != '' else 'NA'; circ_direction = line_cols[5]
                            else: circ_gene = line_cols[4] if line_cols[4] != '' else 'NA'
                    elif len(line_cols) >= 3 and re.match(r'^\d+$', line_cols[1]) and re.match(r'^\d+$', line_cols[2]):
                        circ_chr_info = line_cols[0]; circ_start_info = str(int(line_cols[1]) + 1); circ_end_info = line_cols[2]
                        if len(line_cols) >= 4 and line_cols[3] not in ['', '.']: circ_id = line_cols[3]
                        if len(line_cols) >= 6 and line_cols[5] in ['+', '-']: circ_direction = line_cols[5]
                    if circ_chr_info == '' or circ_start_info == '' or circ_end_info == '': continue
                    if circ_id == '' or circ_id == '.': circ_id = 'circ_' + str(auto_id_num).rjust(6, '0'); auto_id_num += 1
                    circ_chr_info = normalize_chr(circ_chr_info)
                    dic_id_gene[circ_id] = circ_gene; dic_id_strand[circ_id] = circ_direction
                    dic_circ_id_start_seq[circ_id] = circ_start_seq; dic_circ_id_end_seq[circ_id] = circ_end_seq
                    line_circ_info_gff_all += circ_id + '\t' + circ_chr_info + '\t' + circ_start_info + '\t' + circ_end_info + '\t' + circ_gene + '\t' + circ_direction + '\n'
                    line_circ_info_uniq_bed_all += circ_chr_info + '\t' + str(int(circ_start_info)-1) + '\t' + circ_end_info + '\t' + circ_id + '\t.\t' + circ_direction + '\n'
                    circ_bed_id = circ_chr_info + ':' + str(int(circ_start_info)-1) + '-' + circ_end_info
                    dic_circ_bed_circ_id[circ_bed_id] = circ_id
        elif _out_path('circ_info_omit.gff').exists():
            with _out_path('circ_info_omit.gff').open('r', encoding='utf-8') as f:
                for line in f:
                    cols = line.strip().split('\t')
                    if len(cols) < 6: continue
                    circ_id, circ_chr, circ_start, circ_end, circ_gene, circ_strand = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]
                    dic_id_gene[circ_id] = circ_gene; dic_id_strand[circ_id] = circ_strand
                    circ_bed_id = circ_chr + ':' + str(int(circ_start)-1) + '-' + circ_end
                    dic_circ_bed_circ_id[circ_bed_id] = circ_id
                    line_circ_info_gff_all += line
                    line_circ_info_uniq_bed_all += circ_chr + '\t' + str(int(circ_start)-1) + '\t' + circ_end + '\t' + circ_id + '\t.\t' + circ_strand + '\n'
        
        with _out_path('circ_info_omit.gff').open('w', encoding='utf-8') as f: f.write(line_circ_info_gff_all)
        with _out_path('circ_uniq.bed').open('w', encoding='utf-8') as f: f.write(line_circ_info_uniq_bed_all)
        
        log.info(f'[INFO] Parsed {len(dic_circ_bed_circ_id)} circRNA annotations.')
        log.info('[STEP] Extracting sequences from genome...')

        dic_chr_len = {rec.id: len(rec.seq) for rec in SeqIO.parse(dna_seq, 'fasta')}
        with _out_path('circ_uniq.bed').open('r', encoding='utf-8') as fin, _out_path('circ_uniq.filtered.bed').open('w', encoding='utf-8') as fout:
            for line in fin:
                cols = line.strip().split('\t')
                if len(cols) < 3: continue
                if cols[0] not in dic_chr_len or int(cols[1]) < 0 or int(cols[2]) > dic_chr_len[cols[0]]: continue
                fout.write(line)
        _run_bedtools(
            ['bedtools', 'getfasta', '-fi', dna_seq, '-bed', _out_str('circ_uniq.filtered.bed'), '-fo', _out_str('circ_seq.fasta'), '-s'],
            output_name='circ_seq.fasta',
        )
        line_id_seq_all = ''
        with _out_path('circ_seq.fasta').open('r', encoding='utf-8') as file_circ_info:
            circ_bed_new_id = ''
            for line_circ_info in file_circ_info:
                if line_circ_info[0] == '>': 
                    circ_bed_new_id = line_circ_info.strip().strip('>').split('(')[0]
                    circ_new_id = dic_circ_bed_circ_id.get(circ_bed_new_id, '')
                else:
                    circ_new_seq = line_circ_info.strip().upper()
                    if 'N' in circ_new_seq or not circ_new_id: continue
                    if not dic_circ_id_end_seq: 
                        line_id_seq_all += '>' + circ_new_id + '\n' + circ_new_seq + '\n'; continue
                    if circ_new_id in dic_circ_id_start_seq:
                        if circ_new_seq.find(dic_circ_id_start_seq[circ_new_id]) == 0 or circ_new_seq.find(dic_circ_id_end_seq[circ_new_id]) == 0:
                            line_id_seq_all += '>' + circ_new_id + '\n' + circ_new_seq + '\n'
        
        circ_seq_file = _out_str('circ_slef_seq.fasta')
        with open(circ_seq_file, 'w', encoding='utf-8') as f:
            f.write(line_id_seq_all)
        with open(circ_seq_file, 'r', encoding='utf-8') as extracted_handle:
            extracted_count = sum(1 for l in extracted_handle if l.startswith('>'))
        log.info(f'[INFO] Successfully extracted {extracted_count} circRNA sequences.')

        with open(circ_seq_file, 'r', encoding='utf-8') as file_circ_id_seq, _out_path('circ_pr.txt').open('w', encoding='utf-8') as file_id_gene_seq_pr:
            for line_circ_id_seq in file_circ_id_seq:
                if '>' in line_circ_id_seq: 
                    circ_header_raw = line_circ_id_seq.strip().strip('>')
                    circ_id = circ_header_raw.split(':')[0]
                else:
                    circ_seq_val = line_circ_id_seq.strip().upper()
                    if 'N' in circ_seq_val: continue
                    circ_strand = dic_id_strand.get(circ_id, 'NA')
                    circ_gene = dic_id_gene.get(circ_id, 'none')
                    circ_seq_r = str(reverse_complement(circ_seq_val))
                    if circ_strand == '+': file_id_gene_seq_pr.write('>' + circ_id + ':' + circ_gene + '_F\n' + circ_seq_val + '\n')
                    elif circ_strand == '-': file_id_gene_seq_pr.write('>' + circ_id + ':' + circ_gene + '_R\n' + circ_seq_r + '\n')
                    else:
                        file_id_gene_seq_pr.write('>' + circ_id + ':' + circ_gene + '_F\n' + circ_seq_val + '\n')
                        file_id_gene_seq_pr.write('>' + circ_id + ':' + circ_gene + '_R\n' + circ_seq_r + '\n')
                        list_circ_unknown.append(circ_id)
        
        _out_path('circ_exon_connect.fasta').open('w', encoding='utf-8').close()
        if dic_circ_exon_info:
            exon_bed_lines = []; exon_circ_map = {}
            for circ_key, exons in dic_circ_exon_info.items():
                parts = circ_key.rsplit('_', 1); chr_start_end = parts[0]; strand = parts[1] if len(parts) > 1 else '+'
                chr_name = '_'.join(chr_start_end.split('_')[:-2])
                for i, (e_start, e_end) in enumerate(exons):
                    exon_key = f"{chr_name}_{e_start}_{e_end}_{i}"
                    exon_bed_lines.append(f"{chr_name}\t{e_start}\t{e_end}\t{exon_key}\t.\t{strand}\n")
                    exon_circ_map[exon_key] = circ_key
            with _out_path('ce_exon.bed').open('w', encoding='utf-8') as f: f.writelines(exon_bed_lines)
            _run_bedtools(
                ['bedtools', 'getfasta', '-fi', dna_seq, '-bed', _out_str('ce_exon.bed'), '-fo', _out_str('ce_exon.fasta'), '-s'],
                output_name='ce_exon.fasta',
            )
            circ_exon_seqs = {}
            for rec in SeqIO.parse(_out_str('ce_exon.fasta'), 'fasta'):
                exon_key = rec.id.split('(')[0]; circ_key = exon_circ_map.get(exon_key)
                if circ_key: circ_exon_seqs.setdefault(circ_key, []).append(str(rec.seq).upper())
            with _out_path('circ_exon_connect.fasta').open('w', encoding='utf-8') as fout:
                for circ_key, seqs in circ_exon_seqs.items():
                    parts = circ_key.rsplit('_', 1); chr_start_end = parts[0]; strand = parts[1]
                    chr_name = '_'.join(chr_start_end.split('_')[:-2]); start_end = chr_start_end.split('_')[-2:]
                    spliced = ''.join(seqs)
                    if 'N' in spliced: continue
                    gene = dic_gff_info_gene.get(chr_name + '\t' + start_end[0] + '\t' + start_end[1] + '\t', 'none')
                    circ_id_prefix = chr_name + '_' + start_end[0] + '_' + start_end[1]
                    if strand == '-': fout.write('>exon_' + circ_id_prefix + ':' + gene + '_R\n' + spliced + '\n')
                    else: fout.write('>exon_' + circ_id_prefix + ':' + gene + '_F\n' + spliced + '\n')
        
        with _out_path('circ_exon.gff').open('w', encoding='utf-8') as circ_exon_handle:
            _run_bedtools(
                ['bedtools', 'intersect', '-a', _out_str('circ_uniq.bed'), '-b', _out_str('dna_exon.gff'), '-wo'],
                output_name='circ_exon.gff',
                stdout_handle=circ_exon_handle,
            )
        circ_exon_data = {}
        with _out_path('circ_exon.gff').open('r', encoding='utf-8') as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) < 10: continue
                circ_id = cols[3]; circ_strand = cols[5]; exon_start = cols[7]; exon_end = cols[8]; exon_strand = cols[12]
                circ_exon_data.setdefault(circ_id, {'strand': circ_strand, 'exons': []})
                circ_exon_data[circ_id]['exons'].append((int(exon_start), int(exon_end), exon_strand))
        
        exon_bed_lines = []
        for circ_id, data in circ_exon_data.items():
            circ_strand = data['strand']
            fwd_exons = [e for e in data['exons'] if e[2] in ['+', '.']]; rev_exons = [e for e in data['exons'] if e[2] in ['-', '.']]
            strands_to_process = []
            if circ_strand == '+': strands_to_process.append(('+', fwd_exons))
            elif circ_strand == '-': strands_to_process.append(('-', rev_exons))
            else:
                if fwd_exons: strands_to_process.append(('+', fwd_exons))
                if rev_exons: strands_to_process.append(('-', rev_exons))
            for strand_type, exons in strands_to_process:
                if not exons: continue
                sorted_exons = sorted(list(set(exons)), key=lambda x: x[0])
                for i, (e_start, e_end, _) in enumerate(sorted_exons):
                    exon_id = f"{circ_id}_{strand_type}_{i}"
                    exon_bed_lines.append(f"{cols[0]}\t{e_start-1}\t{e_end}\t{exon_id}\t.\t{strand_type}\n")
        
        with _out_path('fallback_exon.bed').open('w', encoding='utf-8') as f: f.writelines(exon_bed_lines)
        _run_bedtools(
            ['bedtools', 'getfasta', '-fi', dna_seq, '-bed', _out_str('fallback_exon.bed'), '-fo', _out_str('fallback_exon.fasta'), '-s'],
            output_name='fallback_exon.fasta',
        )
        fallback_seqs = {}
        for rec in SeqIO.parse(_out_str('fallback_exon.fasta'), 'fasta'):
            exon_id = rec.id.split('(')[0]; parts = exon_id.rsplit('_', 2)
            circ_id = parts[0]; strand_type = parts[1]; exon_idx = parts[2]
            fallback_seqs.setdefault((circ_id, strand_type), []).append(str(rec.seq).upper())
        with _out_path('circ_exon_connect.fasta').open('a', encoding='utf-8') as fout:
            for (circ_id, strand_type), seqs in fallback_seqs.items():
                spliced = ''.join(seqs)
                if 'N' in spliced: continue
                circ_gene = dic_id_gene.get(circ_id, 'none')
                if strand_type == '+': fout.write('>exon_' + circ_id + ':' + circ_gene + '_F\n' + spliced + '\n')
                else: fout.write('>exon_' + circ_id + ':' + circ_gene + '_R\n' + spliced + '\n')

    # Shared alignment phase wrapping genomes and transcript sequence
    with _out_path('circ_pr.txt').open('r', encoding='utf-8') as file_circbase, _out_path('circ_exon_connect.fasta').open('r', encoding='utf-8') as file_circ_exon:
        list_base_ID, dic_base = [], {}
        for line_base in file_circbase:
            if line_base[0] == '>':
                base_ID = line_base.strip('>').strip()
                base_ID_l = base_ID.rstrip('_F').rstrip('_R') if not use_sequence_mode else base_ID
                list_base_ID.append(base_ID_l)
            else: dic_base[base_ID] = line_base.strip()
        list_exon_ID, dic_exon = [], {}
        for line_exon in file_circ_exon:
            if line_exon[0] == '>':
                exon_ID = line_exon.strip().replace('>exon_', '') if not use_sequence_mode else line_exon.strip().strip('>')
                exon_ID_l = exon_ID.rstrip('_F').rstrip('_R') if not use_sequence_mode else exon_ID
                list_exon_ID.append(exon_ID_l)
            else: dic_exon[exon_ID] = line_exon.strip()
            
    set_base_exon = set(list_base_ID) & set(list_exon_ID)
    set_only_base = set(list_base_ID) - set(list_exon_ID)
    
    with _out_path('circRNA_full_length.fasta').open('w', encoding='utf-8') as now_circ_marge:
        for circ_id in set_base_exon:
            circ_id_F = circ_id + '_F' if not use_sequence_mode else circ_id
            circ_id_R = circ_id + '_R' if not use_sequence_mode else None
            if circ_id_F in dic_base and circ_id_F in dic_exon:
                now_circ_marge.write(('>F_exon_' if not use_sequence_mode else '>') + circ_id + '\n' + dic_exon[circ_id_F] + '\n')
            elif circ_id_F in dic_base:
                now_circ_marge.write(('>F_base_' if not use_sequence_mode else '>') + circ_id + '\n' + dic_base[circ_id_F] + '\n')
            elif circ_id_F in dic_exon:
                now_circ_marge.write(('>F_self_' if not use_sequence_mode else '>') + circ_id + '\n' + dic_exon[circ_id_F] + '\n')
            if circ_id_R:
                if circ_id_R in dic_base and circ_id_R in dic_exon:
                    now_circ_marge.write('>R_exon_' + circ_id + '\n' + dic_exon[circ_id_R] + '\n')
                elif circ_id_R in dic_base:
                    now_circ_marge.write('>R_base_' + circ_id + '\n' + dic_base[circ_id_R] + '\n')
                elif circ_id_R in dic_exon:
                    now_circ_marge.write('>R_self_' + circ_id + '\n' + dic_exon[circ_id_R] + '\n')
        for circ_id in set_only_base:
            circ_id_F = circ_id + '_F' if not use_sequence_mode else circ_id
            circ_id_R = circ_id + '_R' if not use_sequence_mode else None
            list_circ_unknown_ref = list_circ_unknown if not use_sequence_mode else []
            type_prefix = 'noexonic_' if circ_id not in list_circ_unknown_ref else 'unknown_'
            if circ_id_F in dic_base:
                now_circ_marge.write(('>F_' + type_prefix if not use_sequence_mode else '>') + circ_id + '\n' + dic_base[circ_id_F] + '\n')
            if circ_id_R and circ_id_R in dic_base:
                now_circ_marge.write('>R_' + type_prefix + circ_id + '\n' + dic_base[circ_id_R] + '\n')
