from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
import logging
log = logging.getLogger(__name__)


HEADER_RE = re.compile(r"^(?P<base>.+)-ORF(?P<orfnum>\d+)(?:_(?P<suffix>[A-Za-z0-9_]+))?\((?P<start>-?\d+),(?P<end>-?\d+)\)(?::(?P<gene>[^:]*))?(?::(?P<bps>.*))?$")

def parse_breakpoints_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    aa_no_stop_len = len(entry['aa_no_stop'])
    for token in (entry['breakpoints_raw'] or '').split('|'):
        token = token.strip()
        if not token: continue
        parts = token.split('.')
        if len(parts) == 2:
            try:
                aa_idx = int(parts[0]); bp_type = int(parts[1])
                if 1 <= aa_idx <= aa_no_stop_len:
                    events.append({'entry': entry, 'aa_idx': aa_idx, 'bp_type': bp_type})
            except ValueError:
                continue
    return events

def choose_best_event(
    events: List[Dict[str, Any]],
    up_max: int,
    down_max: int,
    start_codon_priority: Dict[str, int],
) -> Any:
    best = None; best_score = None
    for ev in events:
        e = ev['entry']
        if 'X' in e['aa_no_stop'][max(0,ev['aa_idx']-up_max-1):ev['aa_idx']+down_max]: continue
        cover = min(up_max, ev['aa_idx']-1) + min(down_max, len(e['aa_no_stop'])-ev['aa_idx'])
        score = (
            cover,
            -len(e['aa_no_stop']),
            start_codon_priority.get(e['codon'], 0),
            -e['start'],
            -e['order'],
        )
        if best_score is None or score > best_score: best_score = score; best = ev
    return best

def make_window(ev: Dict[str, Any], up_max: int, down_max: int) -> List[Any]:
    e = ev['entry']; l = max(1, ev['aa_idx']-up_max); r = min(len(e['aa_no_stop']), ev['aa_idx']+down_max)
    return [e['aa_no_stop'][l-1:r], e['nt_no_stop'][(l-1)*3:r*3], e['start']+(l-1)*3, e['start']+r*3, str(ev['aa_idx']-l+1)+'.'+str(ev['bp_type'])]

def build_header(base: str, tag: str, s: int, e: int, gene: str, bp: str) -> str:
    h = base + '-' + tag + '(' + str(s) + ',' + str(e) + ')'
    h += ':' + (gene if gene else '')
    if bp:
        h += ':' + bp
    return h

def postprocess_orf_records(
    records: List[Dict[str, Any]],
    flank_len: int,
    start_codon_list: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Isolates translation events crossing the back-spliced junction (BSJ) and truncates them.
    Proteogenomic direct evidence relies on detecting peptides across the BSJ.
    """
    grouped = {}; order_bases = []; order_bases_set = set(); unmatched = []
    start_codon_list = list(start_codon_list or ["ATG", "TTG", "CTG", "GTG"])
    start_codon_priority = {
        codon: len(start_codon_list) - idx
        for idx, codon in enumerate(start_codon_list)
    }
    total_recs = len(records)
    for idx_r, rec in enumerate(records):
        if (idx_r + 1) % 500000 == 0 or idx_r + 1 == total_recs:
            log.debug("Grouping ORFs: %d/%d", idx_r + 1, total_recs)
        if '_base' in rec:
            base = rec['_base']; gene = rec['_gene']; start = rec['_start']; end = rec['_end']; bps = rec['_bps']; suffix = rec['_suffix']
        else:
            m = HEADER_RE.match(rec['header'])
            if not m: unmatched.append(rec); continue
            base = m.group('base'); gene = m.group('gene') or ''; start = int(m.group('start')); end = int(m.group('end')); bps = m.group('bps') or ''; suffix = m.group('suffix') or ''
        
        aa = rec['sequence']; nt = rec.get('nt_sequence','')
        aa_ns = aa[:-1] if aa.endswith('*') else aa; nt_ns = nt[:-3] if (aa.endswith('*') and len(nt)>=3) else nt
        codon = ''
        for p in (suffix or '').split('_'):
            if p in {'ATG','CTG','TTG','GTG'}: codon = p; break
        entry = {'raw_header':rec['header'],'base':base,'start':start,'end':end,'gene':gene,'breakpoints_raw':bps,'aa_seq':aa,'aa_no_stop':aa_ns,'nt_seq':nt,'nt_no_stop':nt_ns,'codon':codon,'order':rec['order']}
        grouped.setdefault(base, []).append(entry)
        if base not in order_bases_set: order_bases_set.add(base); order_bases.append(base)
    log.info('')
    out = []
    total_bases = len(order_bases)
    for idx_base, base in enumerate(order_bases):
        if (idx_base + 1) % 5000 == 0 or idx_base + 1 == total_bases:
            log.debug("BSJ selection: %d/%d", idx_base + 1, total_bases)
        entries = grouped[base]; events = []
        for e in entries: events.extend(parse_breakpoints_entry(e))
        if not events:
            for e in entries: out.append({'header':e['raw_header'],'sequence':e['aa_seq'],'nt_sequence':e['nt_seq'],'order':e['order'],'source_header':e['raw_header'],'source_sequence':e['aa_seq'],'source_nt_sequence':e['nt_seq']})
            continue
        by_type = {1:[],2:[],3:[]}
        for ev in events: by_type[ev['bp_type']].append(ev)
        selected = {
            t: choose_best_event(
                by_type[t],
                flank_len,
                flank_len,
                start_codon_priority,
            )
            for t in [1, 2, 3]
            if by_type[t]
        }
        if not selected:
            for e in entries: out.append({'header':e['raw_header'],'sequence':e['aa_seq'],'nt_sequence':e['nt_seq'],'order':e['order'],'source_header':e['raw_header'],'source_sequence':e['aa_seq'],'source_nt_sequence':e['nt_seq']})
            continue
        gene = next((e['gene'] for e in entries if e['gene']), '')
        for t in [1,2,3]:
            if t not in selected: continue
            seq, nt_seq, s, e, bp = make_window(selected[t], flank_len, flank_len)
            out.append({'header':build_header(base,'ORF'+str(t),s,e,gene,bp),'sequence':seq,'nt_sequence':nt_seq,'order':selected[t]['entry']['order'],'source_header':selected[t]['entry']['raw_header'],'source_sequence':selected[t]['entry']['aa_seq'],'source_nt_sequence':selected[t]['entry']['nt_seq']})
    for rec in unmatched:
        rec.setdefault('source_header', rec['header']); rec.setdefault('source_sequence', rec['sequence']); rec.setdefault('source_nt_sequence', rec.get('nt_sequence',''))
        out.append(rec)
    log.info('')
    return sorted(out, key=lambda r: r['order'])

def filter_within_circ(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes internal duplicates inside the same circRNA based on subsequence subsumption."""
    grouped = {}
    for rec in records:
        h = rec['header']
        key = h.split('-ORF')[0] if '-ORF' in h else h.split()[0]
        grouped.setdefault(key, []).append(rec)
    kept = []
    total_grps = len(grouped)
    for idx_grp, (grp_key, grp) in enumerate(grouped.items()):
        if (idx_grp + 1) % 5000 == 0 or idx_grp + 1 == total_grps:
            log.debug("Deduplication: %d/%d", idx_grp + 1, total_grps)
        by_seq = {}
        for r in grp:
            by_seq.setdefault(r['sequence'], []).append(r)
        dups = [sorted(v, key=lambda r:r['order'])[0] for v in by_seq.values()]
        if len(dups) <= 1:
            kept.extend(dups)
            continue
        surv = sorted(dups, key=lambda r:(-len(r['sequence']),r['order']))
        rm = set()
        for i in range(len(surv)):
            if id(surv[i]) in rm: continue
            long_seq = surv[i]['sequence']
            long_len = len(long_seq)
            K = 3
            if long_len >= K:
                long_kmers = set(long_seq[j:j+K] for j in range(long_len - K + 1))
            else:
                long_kmers = None
            for j in range(i+1, len(surv)):
                if id(surv[j]) in rm: continue
                short_seq = surv[j]['sequence']
                short_len = len(short_seq)
                if short_len >= long_len: continue
                if long_kmers is not None and short_len >= K:
                    skip = False
                    for j2 in range(short_len - K + 1):
                        if short_seq[j2:j2+K] not in long_kmers:
                            skip = True; break
                    if skip: continue
                if short_seq in long_seq:
                    rm.add(id(surv[j]))
        kid = set(id(r) for r in dups)
        kept.extend([r for r in grp if id(r) not in rm and id(r) in kid])
    log.info('')
    return sorted(kept, key=lambda r: r['order'])
