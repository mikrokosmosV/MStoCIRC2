import os
import sys
import subprocess
import time
import math
import re
import logging
from pathlib import Path
log = logging.getLogger(__name__)


def build_ires_input_seq(circ_orf_id, dic_source_orf_coords, dic_circ_id_seq):
    coords = dic_source_orf_coords.get(circ_orf_id)
    if coords:
        start_0based, end_0based = coords
    else:
        m = re.search(r'\((-?\d+),(-?\d+)\)', circ_orf_id)
        if not m:
            return []
        start_0based = int(m.group(1))
        end_0based = int(m.group(2))
    circ_id = circ_orf_id.strip().split('-', 1)[0]
    full_seq = dic_circ_id_seq.get(circ_id, '')
    if not full_seq:
        return []
    s = full_seq.upper().replace('U', 'T')
    circ_len = len(s)
    if circ_len < 10:
        return []
    orf_length = end_0based - start_0based
    if orf_length < 0:
        orf_length = 0
    required_len = orf_length + 1000
    multiplier = max(10, int(math.ceil(required_len / circ_len)) + 4)
    s_multi = s * multiplier
    mid_copy_idx = multiplier // 2
    normalized_start = start_0based % circ_len
    anchor_start = normalized_start + mid_copy_idx * circ_len
    anchor_end = anchor_start + orf_length
    seqs = []
    s1_start = anchor_start - 137
    s1_end = anchor_start + 37
    if s1_start < 0 or s1_end > len(s_multi):
        seqs.append('N' * 174)
    else:
        seqs.append(s_multi[s1_start: s1_end])
    s2_start = anchor_start - 237
    s2_end = anchor_start - 63
    if s2_start < 0 or s2_end > len(s_multi):
        seqs.append('N' * 174)
    else:
        seqs.append(s_multi[s2_start: s2_end])
    s3_start = anchor_end
    s3_end = anchor_end + 174
    if s3_start < 0 or s3_end > len(s_multi):
        seqs.append('N' * 174)
    else:
        seqs.append(s_multi[s3_start: s3_end])
    return seqs

def batch_ires_predict(orf_ids, file_out, deepcip_path, deepcip_python, dic_source_orf_coords, dic_circ_id_seq):
    dic_orf_seqs = {}
    log.info(f" -> Preparing sequences for IRES prediction...")
    skipped_count = 0
    for orf_id in orf_ids:
        seqs = build_ires_input_seq(orf_id, dic_source_orf_coords, dic_circ_id_seq)
        if not seqs:
            skipped_count += 1
            continue
        oid = orf_id.split()[0]
        dic_orf_seqs[oid] = seqs
    if skipped_count > 0:
        log.info(f" -> Note: Skipped {skipped_count} ORFs during IRES sequence extraction.")
    if not dic_orf_seqs:
        return {}
    output_dir = Path(file_out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_fasta = output_dir / 'all_circRNA_ires.fasta'
    with temp_fasta.open('w+', encoding='utf-8') as fp:
        for oid, seqs in dic_orf_seqs.items():
            if seqs[0] and len(seqs[0]) == 174:
                fp.write(f'>{oid}_seq1\n{seqs[0]}\n')
            if seqs[1] and len(seqs[1]) == 174:
                fp.write(f'>{oid}_seq2\n{seqs[1]}\n')
            if len(seqs) > 2 and seqs[2] and len(seqs[2]) == 174:
                fp.write(f'>{oid}_seq3\n{seqs[2]}\n')
    if not deepcip_path or not os.path.exists(os.path.join(deepcip_path, 'DeepCIP.py')):
        log.info(
            "[WARNING] DeepCIP path is unavailable; IRES-related translation-potential scoring "
            "will be incomplete for this run."
        )
        return {}
        
    import shutil
    deepcip_data_dir = os.path.join(deepcip_path, 'data')
    if not os.path.exists(deepcip_data_dir):
        os.makedirs(deepcip_data_dir, exist_ok=True)
    deepcip_results_dir = os.path.join(deepcip_path, 'results')
    if not os.path.exists(deepcip_results_dir):
        os.makedirs(deepcip_results_dir, exist_ok=True)
    deepcip_input_name = f'temp_ires_{int(time.time())}.fasta'
    deepcip_input_path = os.path.join(deepcip_data_dir, deepcip_input_name)
    shutil.copy(str(temp_fasta), deepcip_input_path)
    deepcip_python = (deepcip_python or "").strip() or sys.executable
    predict_script_abs = os.path.join(deepcip_path, 'DeepCIP.py')
    dataset_name = f'run_{int(time.time())}'
    cmd_args = [deepcip_python, predict_script_abs, '-n', dataset_name, '-i', deepcip_input_name, '-m', '0']
    log.info(f" -> Running DeepCIP prediction using: {deepcip_python}")
    try:
        subprocess.run(cmd_args, cwd=deepcip_path, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log.info(f" -> Error running DeepCIP (Exit code {e.returncode}): {e}")
    except Exception as e:
        log.info(f" -> Unexpected error running DeepCIP: {e}")
        
    input_base_name = os.path.splitext(deepcip_input_name)[0]
    deepcip_res_csv = os.path.join(deepcip_path, 'results', f'{input_base_name}_mode_0.csv')
    dic_ires_result = {}
    if os.path.exists(deepcip_res_csv):
        try:
            import pandas as pd
            df = pd.read_csv(deepcip_res_csv)
            temp_probs = {}
            rows_processed = 0
            for _, row in df.iterrows():
                seq_id = str(row['Sequence_name']).strip()
                try:
                    prob = float(row['Predict_probs'])
                except (ValueError, KeyError):
                    continue
                if '_seq' in seq_id:
                    parts_split = seq_id.rsplit('_seq', 1)
                    if len(parts_split) == 2:
                        oid = parts_split[0]
                        seq_idx = parts_split[1]
                        if oid not in temp_probs:
                            temp_probs[oid] = {}
                        temp_probs[oid][seq_idx] = prob
                        rows_processed += 1
            log.info(f" -> Parsed {rows_processed} rows from DeepCIP results")
            for oid, probs in temp_probs.items():
                p1 = probs.get('1', -1.0)
                p2 = probs.get('2', -1.0)
                p3 = probs.get('3', -1.0)
                dic_ires_result[oid] = {'p1': p1, 'p2': p2, 'p3': p3}
            log.info(f" -> Successfully mapped results for {len(dic_ires_result)} ORFs")
        except (KeyError, ValueError, TypeError) as e:
            log.info(f" -> Error parsing DeepCIP results: {e}")
            log.debug("DeepCIP parsing traceback", exc_info=True)
    try:
        os.remove(deepcip_input_path)
        if os.path.exists(deepcip_res_csv):
            os.remove(deepcip_res_csv)
    except OSError:
        pass
    return dic_ires_result

def batch_m6a_predict(orf_entries, file_out, deepcircm6a_path, dic_source_orf_coords, dic_circ_id_seq):
    log.info(f" -> Preparing batch m6A prediction input sequences...")
    output_dir = Path(file_out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_fasta = output_dir / 'all_circRNA_m6a_windows.fasta'
    valid_windows_count = 0
    dic_m6a_counts = {}
    with temp_fasta.open('w+', encoding='utf-8') as fp:
        for orf_id, orf_seq in orf_entries:
            coords = dic_source_orf_coords.get(orf_id)
            if coords:
                start_0based = coords[0]
            else:
                m = re.search(r'\((-?\d+),-?\d+\)', orf_id)
                if not m:
                    continue
                start_0based = int(m.group(1))
            circ_id = orf_id.strip().split('-')[0]
            seq = dic_circ_id_seq.get(circ_id, '')
            if not seq:
                continue
            s = seq.upper().replace('U', 'T')
            circ_len = len(s)
            if circ_len < 10:
                continue
            s5 = s * 5
            anchor_pos = start_0based + 2 * circ_len
            region_start = anchor_pos - 126
            region_end = anchor_pos + 25
            if region_start < 0 or region_end > len(s5):
                continue
            region = s5[region_start: region_end]
            n = len(region)
            for idx in range(25, n - 25):
                if region[idx] == 'A':
                    window = region[idx - 25: idx + 26]
                    if len(window) != 51:
                        continue
                    rel_pos = idx - 126
                    header = f"{orf_id}###{rel_pos}"
                    fp.write(f">{header}\n{window}\n")
                    valid_windows_count += 1
    if valid_windows_count == 0:
        return {}
    log.info(f" -> Generated {valid_windows_count} prediction windows, calling Deepcircm6a...")
    if not deepcircm6a_path or not os.path.exists(os.path.join(deepcircm6a_path, 'predict.py')):
        log.info(
            "[WARNING] DeepCircM6A path is unavailable; m6A-related translation-potential scoring "
            "will be incomplete for this run."
        )
        return {}
    predict_dir = Path(deepcircm6a_path).resolve()
    predict_py = predict_dir / 'predict.py'
    out_path = output_dir / 'all_circRNA_m6a_pred.txt'
    failure_log_path = output_dir / 'deepcircm6a.stderr.log'
    run_log_path = output_dir / 'deepcircm6a.run.log'
    cmd_args = [sys.executable, str(predict_py), '-predict_fa', str(temp_fasta), '-model_path', str(predict_dir), '-outfile', str(out_path)]
    log.info(f" -> Running Deepcircm6a prediction...")
    log.info(f" -> DeepCircM6A run log: {run_log_path}")
    try:
        completed = subprocess.run(
            cmd_args,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(predict_dir),
        )
        stdout_text = (completed.stdout or '').strip()
        stderr_text = (completed.stderr or '').strip()
        run_log_path.write_text(
            "\n".join(
                [
                    f"Command: {' '.join(cmd_args)}",
                    f"Return code: {completed.returncode}",
                    "",
                    "[STDOUT]",
                    stdout_text or "<empty>",
                    "",
                    "[STDERR]",
                    stderr_text or "<empty>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        if failure_log_path.exists():
            failure_log_path.unlink()
    except subprocess.CalledProcessError as e:
        stderr_text = (e.stderr or '').strip()
        stdout_text = (e.stdout or '').strip()
        details = stderr_text or stdout_text
        run_log_path.write_text(
            "\n".join(
                [
                    f"Command: {' '.join(cmd_args)}",
                    f"Return code: {e.returncode}",
                    "",
                    "[STDOUT]",
                    stdout_text or "<empty>",
                    "",
                    "[STDERR]",
                    stderr_text or "<empty>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        failure_log_path.write_text(
            "\n".join(
                [
                    f"Command: {' '.join(cmd_args)}",
                    "",
                    "[STDOUT]",
                    stdout_text or "<empty>",
                    "",
                    "[STDERR]",
                    stderr_text or "<empty>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        detail_preview = details[:1200]
        if len(details) > 1200:
            detail_preview += "...(truncated)"
        if detail_preview:
            log.error(
                " -> Error running DeepCircM6A (Exit code %d). Details saved to %s and %s. %s",
                e.returncode,
                run_log_path,
                failure_log_path,
                detail_preview,
            )
        else:
            log.error(
                " -> Error running DeepCircM6A (Exit code %d). Details saved to %s and %s.",
                e.returncode,
                run_log_path,
                failure_log_path,
            )
    except OSError as e:
        log.error(f" -> Unexpected error running DeepCircM6A: {e}")
    if not out_path.exists():
        log.warning(
            " -> DeepCircM6A finished without producing '%s'. See %s for captured output.",
            out_path.name,
            run_log_path,
        )
        return {}
    if out_path.exists():
        with out_path.open('r', encoding='utf-8', errors='ignore') as fr:
            for line in fr:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                try:
                    header = parts[0]
                    prob = float(parts[-1])
                    if prob >= 0.5:
                        real_orf_id = header.split('###')[0]
                        dic_m6a_counts[real_orf_id] = dic_m6a_counts.get(real_orf_id, 0) + 1
                except (ValueError, IndexError):
                    continue
    log.info(
        " -> DeepCircM6A completed: %d ORFs with predicted m6A-positive windows.",
        len(dic_m6a_counts),
    )
    return dic_m6a_counts
