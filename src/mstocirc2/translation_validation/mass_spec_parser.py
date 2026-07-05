import re
from pathlib import Path
from typing import Dict, Any, Set, TextIO
import logging

log = logging.getLogger(__name__)


def clean_peptide_sequence(pep: str) -> str:
    """Unified peptide cleaning engine."""
    pep = pep.strip().upper()
    if '.' in pep:
        match = re.match(r'^[A-Z\-_]\.(.+)\.[A-Z\-_]$', pep)
        if match:
            pep = match.group(1)
    pep = re.sub(r'[^A-Z]', '', pep)
    return pep

def extract_circ_info_id(orf_base: str) -> str:
    """Extract circRNA ID from ORF base ID, stripping known prefixes."""
    PREFIXES = [
        'F_exon_', 'F_base_', 'F_self_', 'F_noexonic_', 'F_unknown_',
        'R_exon_', 'R_base_', 'R_self_', 'R_noexonic_', 'R_unknown_',
        'exon_',
    ]
    for pfx in PREFIXES:
        if orf_base.startswith(pfx):
            return orf_base[len(pfx):]
    return orf_base

def open_safe_read(file_path: str) -> TextIO:
    try:
        return Path(file_path).open("r", encoding="utf-8", errors="ignore")
    except (FileNotFoundError, OSError, UnicodeError):
        return Path(file_path).open('r', encoding='latin-1', errors='ignore')

def dic_circ_mgf_pep_add(circ_id: str, mgf_id: str, peptide: str, dic_id_peptide: Dict[str, str], count: int = 1) -> Dict[str, str]:
    try:
        count_val = max(1, int(float(count)))
    except (TypeError, ValueError):
        count_val = 1
    val = f"{mgf_id}###{peptide}###{count_val}"
    if circ_id in dic_id_peptide:
        dic_id_peptide[circ_id] += ';' + val
    else:
        dic_id_peptide[circ_id] = val
    return dic_id_peptide

def detect_file_type(file_path: str) -> str:
    try:
        with open_safe_read(file_path) as f:
            header = f.readline()
            if not header:
                return 'unknown'
            if 'MS_ID' in header and 'peptide' in header and 'cORF_ID' in header:
                return 'summary'
            parts = [h.strip() for h in header.split('\t')]
            if 'Stripped.Sequence' in parts and ('Protein.Group' in parts or 'Protein.Ids' in parts):
                return 'fragpipe_combined'
            
            if ('Peptide Sequence' in parts or 'Peptide' in parts or 'Sequence' in parts) and \
               ('Protein' in parts or 'Protein ID' in parts):
                if any('Spectral Count' in h for h in parts):
                    return 'fragpipe'
                if 'SpecEValue' in parts or 'EValue' in parts:
                    return 'msgf'
                return 'fragpipe'
            
            if '#SpecFile' in header or 'SpecEValue' in parts or 'EValue' in parts:
                return 'msgf'
            return 'unknown'
    except (FileNotFoundError, OSError, UnicodeError):
        return 'unknown'

def parse_msgf_file(file_path: str, file_line_circ: TextIO, dic_id_peptide: Dict[str, str], global_stats: Dict[str, Any], file_input_protein: str) -> Dict[str, str]:
    with open_safe_read(file_path) as content:
        header_line = content.readline()
        if not header_line:
            return dic_id_peptide
        headers = header_line.strip().split('\t')
        try:
            idx_evalue = headers.index('EValue')
        except ValueError:
            try:
                idx_evalue = headers.index('SpecEValue')
            except ValueError:
                idx_evalue = 13
        try:
            idx_file = headers.index('#SpecFile')
        except ValueError:
            try:
                idx_file = headers.index('SpecFile')
            except ValueError:
                idx_file = 0
        try:
            idx_peptide = headers.index('Peptide')
        except ValueError:
            idx_peptide = 9
        try:
            idx_protein = headers.index('Protein')
        except ValueError:
            idx_protein = 10
        for line in content:
            columns = line.strip().split('\t')
            if len(columns) <= max(idx_evalue, idx_file, idx_peptide, idx_protein):
                continue
            try:
                EValue = float(columns[idx_evalue].strip())
            except ValueError:
                continue
            if EValue <= 0.01:
                file_id = columns[idx_file].replace('.mzML', '.mgf')
                global_stats['psms'] += 1
                global_stats['files'].add(file_id)
                raw_pep = columns[idx_peptide].strip()
                Peptide = clean_peptide_sequence(raw_pep)
                if len(Peptide) < 6:
                    continue
                protein_col = columns[idx_protein].strip()
                circ_id_multiple = protein_col.split(', ')
                for circ_id in circ_id_multiple:
                    clean_circ_id = circ_id
                    line_out = f"{file_id}\t{Peptide}\t{clean_circ_id}\n"
                    file_line_circ.write(line_out)
                    if file_input_protein == 'none':
                        dic_id_peptide = dic_circ_mgf_pep_add(clean_circ_id, file_id, Peptide, dic_id_peptide)
    return dic_id_peptide

def parse_fragpipe_file(file_path: str, file_line_circ: TextIO, dic_id_peptide: Dict[str, str], global_stats: Dict[str, Any], file_input_protein: str) -> Dict[str, str]:
    with open_safe_read(file_path) as content:
        header_line = content.readline()
        headers = header_line.strip().split('\t')
        
        try:
            if 'Peptide Sequence' in headers:
                idx_peptide = headers.index('Peptide Sequence')
            elif 'Peptide' in headers:
                idx_peptide = headers.index('Peptide')
            elif 'Sequence' in headers:
                idx_peptide = headers.index('Sequence')
            else:
                return dic_id_peptide
            
            if 'Protein' in headers:
                idx_protein = headers.index('Protein')
            elif 'Protein ID' in headers:
                idx_protein = headers.index('Protein ID')
            else:
                return dic_id_peptide
            
            sample_cols = []
            for i, h in enumerate(headers):
                if h.endswith(' Spectral Count'):
                    sample_name = h.replace(' Spectral Count', '').strip()
                    sample_cols.append({'idx': i, 'name': sample_name, 'type': 'count'})
            
            if not sample_cols and 'Spectral Count' in headers:
                idx_spec = headers.index('Spectral Count')
                base_name = Path(file_path).stem
                sample_name = base_name.split('peptide_', 1)[1] if base_name.startswith('peptide_') else base_name
                sample_cols = [{'idx': idx_spec, 'name': sample_name, 'type': 'count'}]
            
            if not sample_cols:
                for i, h in enumerate(headers):
                    if h.endswith(' MaxLFQ Intensity') or h.endswith(' Intensity'):
                        sample_name = h.replace(' MaxLFQ Intensity', '').replace(' Intensity', '').strip()
                        if not any(s['name'] == sample_name for s in sample_cols):
                            sample_cols.append({'idx': i, 'name': sample_name, 'type': 'intensity'})
            
            if not sample_cols:
                return dic_id_peptide
        except ValueError:
            return dic_id_peptide

        for line in content:
            line = line.strip()
            if not line:
                continue
            columns = line.split('\t')
            max_idx = max([s['idx'] for s in sample_cols])
            if len(columns) <= max_idx:
                continue
            
            raw_pep = columns[idx_peptide].strip().strip('"')
            peptide_seq = clean_peptide_sequence(raw_pep)
            if len(peptide_seq) < 6:
                continue
            
            protein_val = columns[idx_protein].strip().strip('"')
            clean_circ_id = protein_val
            
            for sample in sample_cols:
                try:
                    val_str = columns[sample['idx']].strip()
                    if sample['type'] == 'intensity':
                        count_val = 1 if float(val_str) > 0 else 0
                    else:
                        count_val = int(float(val_str)) if val_str else 0
                except ValueError:
                    count_val = 0
                
                if count_val > 0:
                    file_id = sample['name']
                    global_stats['files'].add(file_id)
                    for _ in range(count_val):
                        global_stats['psms'] += 1
                        line_out = f"{file_id}\t{peptide_seq}\t{clean_circ_id}\n"
                        file_line_circ.write(line_out)
                    if file_input_protein == 'none':
                        dic_id_peptide = dic_circ_mgf_pep_add(clean_circ_id, file_id, peptide_seq, dic_id_peptide, count_val)
    return dic_id_peptide

def parse_fragpipe_combined_file(file_path: str, file_line_circ: TextIO, dic_id_peptide: Dict[str, str], global_stats: Dict[str, Any], file_input_protein: str) -> Dict[str, str]:
    with open_safe_read(file_path) as content:
        header_line = content.readline()
        headers = header_line.strip().split('\t')
        try:
            idx_peptide = headers.index('Stripped.Sequence')
        except ValueError:
            return dic_id_peptide
        idx_protein = -1
        for col_name in ['Protein.Group', 'Protein.Ids', 'Protein', 'Protein ID']:
            if col_name in headers:
                idx_protein = headers.index(col_name)
                break
        if idx_protein == -1:
            return dic_id_peptide
        sample_cols = []
        for i, h in enumerate(headers):
            if h.endswith(' Spectral Count'):
                sample_name = h.replace(' Spectral Count', '').strip()
                sample_cols.append({'idx': i, 'name': sample_name, 'type': 'count'})
        if not sample_cols:
            for i, h in enumerate(headers):
                if h.endswith(' MaxLFQ Intensity') or h.endswith(' Intensity'):
                    sample_name = h.replace(' MaxLFQ Intensity', '').replace(' Intensity', '').strip()
                    if not any(s['name'] == sample_name for s in sample_cols):
                        sample_cols.append({'idx': i, 'name': sample_name, 'type': 'intensity'})
        if not sample_cols:
            return dic_id_peptide
        for line in content:
            line = line.strip()
            if not line:
                continue
            columns = line.split('\t')
            max_idx = max([s['idx'] for s in sample_cols])
            if len(columns) <= max(idx_peptide, idx_protein, max_idx):
                continue
            peptide_seq = columns[idx_peptide].strip().strip('"')
            peptide_seq = clean_peptide_sequence(peptide_seq)
            if len(peptide_seq) < 6:
                continue
            protein_val = columns[idx_protein].strip().strip('"')
            clean_circ_id = protein_val
            for sample in sample_cols:
                if sample['idx'] >= len(columns):
                    continue
                try:
                    val_str = columns[sample['idx']].strip()
                    if sample['type'] == 'intensity':
                        count_val = 1 if float(val_str) > 0 else 0
                    else:
                        count_val = int(float(val_str)) if val_str else 0
                except (ValueError, IndexError):
                    continue
                if count_val > 0:
                    file_id = sample['name']
                    if '/' in file_id:
                        file_id = Path(file_id).stem
                    global_stats['files'].add(file_id)
                    for _ in range(count_val):
                        global_stats['psms'] += 1
                        line_out = f"{file_id}\t{peptide_seq}\t{clean_circ_id}\n"
                        file_line_circ.write(line_out)
                    if file_input_protein == 'none':
                        dic_id_peptide = dic_circ_mgf_pep_add(clean_circ_id, file_id, peptide_seq, dic_id_peptide, count_val)
    return dic_id_peptide

def parse_summary_file(file_path: str, file_line_circ: TextIO, dic_id_peptide: Dict[str, str], global_stats: Dict[str, Any], file_input_protein: str) -> Dict[str, str]:
    with open_safe_read(file_path) as content:
        header = content.readline()
        for line in content:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            file_id = parts[0]
            peptide = clean_peptide_sequence(parts[1])
            if len(peptide) < 6:
                continue
            circ_id = parts[2]
            global_stats['psms'] += 1
            global_stats['files'].add(file_id)
            line_out = f"{file_id}\t{peptide}\t{circ_id}\n"
            file_line_circ.write(line_out)
            if file_input_protein == 'none':
                dic_id_peptide = dic_circ_mgf_pep_add(circ_id, file_id, peptide, dic_id_peptide)
    return dic_id_peptide

def process_ms_directory(path_ms_input: str, file_out: str, file_input_protein: str) -> Dict[str, str]:
    dic_id_peptide = {}
    output_dir = Path(file_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile_path = output_dir / "tsv_result.txt"
    input_path = Path(path_ms_input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Mass-spectrometry input path does not exist: '{input_path}'."
        )

    global_stats = {'psms': 0, 'files': set()}
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = sorted(
            candidate
            for candidate in input_path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in (".tsv", ".txt")
        )

    log.info(f"Scanning {len(files_to_process)} files in {path_ms_input}...")
    if not files_to_process:
        raise FileNotFoundError(
            f"No supported peptide summary files were found under '{input_path}'."
        )

    success_count = 0
    with outfile_path.open("w+", encoding="utf-8") as file_line_circ:
        file_line_circ.write("MS_ID\tpeptide\tcORF_ID\n")
        for file_path in files_to_process:
            ftype = detect_file_type(str(file_path))
            if ftype == 'msgf':
                dic_id_peptide = parse_msgf_file(str(file_path), file_line_circ, dic_id_peptide, global_stats, file_input_protein)
                success_count += 1
            elif ftype == 'fragpipe':
                dic_id_peptide = parse_fragpipe_file(str(file_path), file_line_circ, dic_id_peptide, global_stats, file_input_protein)
                success_count += 1
            elif ftype == 'fragpipe_combined':
                dic_id_peptide = parse_fragpipe_combined_file(str(file_path), file_line_circ, dic_id_peptide, global_stats, file_input_protein)
                success_count += 1
            elif ftype == 'summary':
                dic_id_peptide = parse_summary_file(str(file_path), file_line_circ, dic_id_peptide, global_stats, file_input_protein)
                success_count += 1

    TOTAL_INPUT_PSMS = global_stats['psms']
    TOTAL_MS_FILES = len(global_stats['files'])
    log.info(f"Successfully processed {success_count} files.")
    log.info(f"Total input PSMs: {TOTAL_INPUT_PSMS}")
    log.info(f"Total MS files/samples: {TOTAL_MS_FILES}")
    return dic_id_peptide
