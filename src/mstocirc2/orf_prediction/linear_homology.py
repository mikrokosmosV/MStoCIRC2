from pathlib import Path
from typing import Dict, List, Any
import logging
log = logging.getLogger(__name__)


def filter_by_references(records: List[Dict[str, Any]], ref_fastas: List[str], kmer_size: int = 6) -> List[Dict[str, Any]]:
    """
    Biological null hypothesis test.
    K-mer based filtering against standard linear proteomes to eliminate artifacts.
    """
    def read_recs(p):
        recs=[];h='';s=[];o=0
        with open(p) as f:
            for l in f:
                l=l.strip()
                if not l: continue
                if l[0]=='>':
                    if h: recs.append({'header':h,'sequence':''.join(s).upper(),'order':o}); o+=1
                    h=l[1:];s=[]
                else: s.append(l)
            if h: recs.append({'header':h,'sequence':''.join(s).upper(),'order':o})
        return recs
        
    def build_idx(recs, k):
        exact=set(r['sequence'] for r in recs if r['sequence']); seqs=list(exact); idx={}
        for i,s in enumerate(seqs):
            if len(s)<k: continue
            seen=set()
            for j in range(len(s)-k+1):
                km=s[j:j+k]
                if km not in seen: seen.add(km); idx.setdefault(km,[]).append(i)
        return exact,seqs,idx
        
    def contained(q, exact, seqs, idx, k):
        if q in exact: return True
        anc=None
        for i in range(len(q)-k+1):
            p=idx.get(q[i:i+k])
            if p is None: return False
            if anc is None or len(p)<len(anc): anc=p
            if len(anc)==1: break
        if anc is None: return any(len(s)>=len(q) and q in s for s in seqs)
        return any(len(seqs[i])>=len(q) and q in seqs[i] for i in anc)
        
    cur = records
    for rf in ref_fastas:
        if rf == 'none' or not Path(rf).exists():
            continue
        exact,seqs,idx = build_idx(read_recs(rf), kmer_size)
        total_cur = len(cur)
        cur_filtered = []
        for idx_f, r in enumerate(cur):
            if (idx_f + 1) % 50000 == 0 or idx_f + 1 == total_cur:
                log.info("[ORF] Homology filter (%s): %d/%d", Path(rf).name, idx_f + 1, total_cur)
            if not contained(r['sequence'], exact, seqs, idx, kmer_size):
                cur_filtered.append(r)
        cur = sorted(cur_filtered, key=lambda r:r['order'])
    return cur
