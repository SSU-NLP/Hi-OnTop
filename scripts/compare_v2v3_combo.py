#!/usr/bin/env python3
"""v2(ewma 적응 임계치) × v3(geometry-merge) 조합 정량 비교 — AMI + DTS 3벤치.
config: {raw, merge} × {percentile p80, ewma adaptive, oracle 상한}.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import (load_dialogs, delta_eff_seq, official_pk_wd,
                                    boundaries, best_score_dstar)
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

EWMA_C = 1.0
from sentence_transformers import SentenceTransformer
_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
    model_kwargs={"provider": "CPUExecutionProvider", "file_name": "onnx/model_quint8_avx2.onnx"})

def batch_embed(list_of_texts):
    flat, offs = [], [0]
    for ts in list_of_texts:
        flat.extend(ts); offs.append(len(flat))
    if not flat:
        return []
    a = np.asarray(_enc.encode(flat, normalize_embeddings=True, batch_size=128,
                               show_progress_bar=False), dtype=np.float64)
    return [a[offs[i]:offs[i+1]] for i in range(len(list_of_texts))]

def geom_flag(E, m=0.0):
    e = [v/(np.linalg.norm(v)+1e-12) for v in E]; fl=[False]*len(e)
    for i in range(1, len(e)-1):
        sp, sn, sb = float(e[i-1]@e[i]), float(e[i]@e[i+1]), float(e[i-1]@e[i+1])
        if sb > sp+m and sb > sn+m: fl[i]=True
    return fl

def merge(utts, yt, fl):
    groups, cur = [], None
    for i in range(len(utts)):
        if fl[i] and cur is not None: cur.append(i)
        else: cur=[i]; groups.append(cur)
    mt = [" ".join(utts[k] for k in g) for g in groups]
    my = [1 if any(yt[k]==1 for k in g) else 0 for g in groups]
    if my: my[-1]=0
    return mt, my

def score(yts, yps):
    pks, wds, g, p = [], [], [], []
    for yt, yp in zip(yts, yps):
        pk, wd = official_pk_wd(yt, yp); pks.append(pk); wds.append(wd); g+=yt; p+=yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk, wd = float(np.mean(pks)), float(np.mean(wds))
    return dict(score=0.5*f1+0.25*(1-pk)+0.25*(1-wd), f1=f1, pk=pk, wd=wd)

def eval_block(name, yts, deffs):
    pool = np.array([v for s in deffs for v in s[1:]])
    d80 = float(np.percentile(pool, 80))
    rows = {
        "percentile p80": [boundaries(s, d80) for s in deffs],
        f"ewma c={EWMA_C}":  [adaptive_boundaries(s, c=EWMA_C, mode="ewma") for s in deffs],
        "otsu (auto)":     [adaptive_boundaries(s, mode="otsu") for s in deffs],
    }
    d_or = best_score_dstar([(None, y) for y in yts], deffs)
    rows["oracle (상한)"] = [boundaries(s, d_or) for s in deffs]
    for rule, yps in rows.items():
        r = score(yts, yps)
        print(f"  {name:6} {rule:16} Score={r['score']:.4f} F1={r['f1']:.4f} "
              f"Pk={r['pk']:.3f} WD={r['wd']:.3f}", flush=True)

def run_dts(ds):
    dialogs = load_dialogs(ds, "test")
    embs = batch_embed([list(u) for u, _ in dialogs])
    yts_raw = [yt for _, yt in dialogs]
    deffs_raw = [delta_eff_seq(e) for e in embs]
    # merge
    yts_m, m_texts, reuse = [], [], []
    for (utts, yt), e in zip(dialogs, embs):
        fl = geom_flag(e); mt, my = merge(utts, yt, fl)
        yts_m.append(my); m_texts.append(mt); reuse.append((sum(fl)==0, e))
    m_embs_new = batch_embed([mt for mt, (no, _) in zip(m_texts, reuse) if not no])
    it = iter(m_embs_new); deffs_m = []
    for (no, e), mt in zip(reuse, m_texts):
        deffs_m.append(delta_eff_seq(e if no else next(it)))
    print(f"[{ds}] n={len(dialogs)}", flush=True)
    eval_block("raw", yts_raw, deffs_raw)
    eval_block("merge", yts_m, deffs_m)
    print(flush=True)

def run_ami():
    TOPIC = REPO/"data"/"ami"/"topic"; CACHE = REPO/"outputs"/"runs"/"_misc"/"ami_emb"
    mids = sorted(m["meeting"] for m in json.load(open(TOPIC/"manifest.json")))
    yts_raw, deffs_raw, yts_m, deffs_m = [], [], [], []
    m_texts, reuse, raw_turns = [], [], []
    for mid in mids:
        cp = CACHE/f"{mid}.pkl"
        if not cp.exists(): continue
        emb = np.asarray(pickle.load(open(cp, "rb")), dtype=np.float64)
        d = json.load(open(TOPIC/f"{mid}.json")); turns=d["turns"]; bt=d["bnd_top"]
        yts_raw.append(bt); deffs_raw.append(delta_eff_seq(emb))
        fl = geom_flag(emb); utts=[t["text"] for t in turns]; mt, my = merge(utts, bt, fl)
        yts_m.append(my); m_texts.append(mt); reuse.append((sum(fl)==0, emb))
    m_new = batch_embed([mt for mt, (no, _) in zip(m_texts, reuse) if not no])
    it = iter(m_new)
    for (no, e), mt in zip(reuse, m_texts):
        deffs_m.append(delta_eff_seq(e if no else next(it)))
    print(f"[AMI] n={len(yts_raw)}", flush=True)
    eval_block("raw", yts_raw, deffs_raw)
    eval_block("merge", yts_m, deffs_m)
    print(flush=True)

if __name__ == "__main__":
    print(f"v2(ewma c={EWMA_C}) × v3(geometry-merge) 조합 — Score/F1/Pk/WD\n", flush=True)
    run_ami()
    for ds in ("tiage", "dialseg711", "superseg"):
        run_dts(ds)
