from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection

def draw_needle_map(orf_id: str, orf_aa_len: int, mapped_peptides: list, out_dir: str):
    if orf_aa_len <= 0 or not mapped_peptides:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, orf_aa_len)
    ax.set_ylim(-1, 5)
    
    segments = []
    colors_list = []
    
    for (start, end, is_bsj) in mapped_peptides:
        y_val = 1
        segments.append([(start, y_val), (end, y_val)])
        if is_bsj:
            colors_list.append('red')
        else:
            colors_list.append('blue')
            
    lc = LineCollection(segments, colors=colors_list, linewidths=2)
    ax.add_collection(lc)
    ax.plot([0, orf_aa_len], [0, 0], color='black', linewidth=1)
    
    ax.set_title(f"Peptide Coverage: {orf_id}")
    ax.set_xlabel("Amino Acid Position")
    ax.set_yticks([])
    
    plot_path = Path(out_dir) / f"{orf_id.replace(':', '_').replace('|', '_')}_coverage.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_all_coverage(results: dict, dic_source_aa: dict, dic_bsj: dict, out_dir: str):
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    from .evidence_scorer import parse_bsj_location, is_peptide_crossing_bsj, get_base_bsj_location
    
    for orf_id, data in results.items():
        orf_seq = dic_source_aa.get(orf_id, '')
        if not orf_seq:
            continue
            
        bsj_str = get_base_bsj_location(orf_id, dic_bsj)
        bsj_list = []
        if bsj_str and bsj_str != '0':
            bsj_list = parse_bsj_location(bsj_str)
            
        mapped = []
        for p in data['peptides']:
            start = orf_seq.find(p)
            if start != -1:
                end = start + len(p)
                is_bsj = False
                if bsj_list:
                    is_bsj = is_peptide_crossing_bsj(start, end, bsj_list)
                mapped.append((start, end, is_bsj))
                
        draw_needle_map(orf_id, len(orf_seq), mapped, str(plot_dir))
