#!/usr/bin/env python3
"""δ* 규칙 비교 (DTS 3벤치): 고정 percentile vs 적응 μ+cσ(online/offline) vs oracle.
모두 base Hi-OnTop δ_eff 위에서 boundary 규칙만 교체. label-free 비교(oracle 제외).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import (load_dialogs, delta_eff_seq, official_pk_wd,
                                    boundaries, best_score_dstar, score_set)
from hi_ontop.hi_ontop_v2 import adaptive_boundaries

ENC = dict(model="sentence-transformers/all-MiniLM-L6-v2",
           file_name="onnx/model_quint8_avx2.onnx")


def score_yps(dialogs, yps):
    pks, wds, g, p = [], [], [], []
    for (utts, yt), yp in zip(dialogs, yps):
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk, wd = float(np.mean(pks)), float(np.mean(wds))
    return dict(pk=pk, wd=wd, f1=f1, score=0.5*f1 + 0.25*(1-pk) + 0.25*(1-wd))


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENC["model"], backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider", "file_name": ENC["file_name"]})

    def batch_embed(list_of_texts):
        flat, offs = [], [0]
        for ts in list_of_texts:
            flat.extend(ts); offs.append(len(flat))
        allemb = np.asarray(enc.encode(flat, normalize_embeddings=True, batch_size=128,
                                       show_progress_bar=False), dtype=np.float64)
        return [allemb[offs[i]:offs[i+1]] for i in range(len(list_of_texts))]

    print("δ* 규칙 비교 — DTS test, MiniLM-int8, base Hi-OnTop δ_eff (Score / F1)\n", flush=True)
    print(f"{'벤치':11} {'rule':22} {'Score':>7} {'F1':>7} {'Pk':>6} {'WD':>6}", flush=True)
    print("-"*62, flush=True)
    for ds in ("tiage", "dialseg711", "superseg"):
        dialogs = load_dialogs(ds, "test")
        embs = batch_embed([list(u) for u, _ in dialogs])
        deffs = [delta_eff_seq(e) for e in embs]
        pool = np.array([v for seq in deffs for v in seq[1:]])
        rules = {}
        # 고정 percentile p80 (현 deployable)
        d80 = float(np.percentile(pool, 80))
        rules["percentile p80"] = [boundaries(s, d80) for s in deffs]
        # 적응 μ+cσ — online / online_nf(개선) / ewma(개선) / offline, c sweep
        for c in (0.5, 1.0):
            for mode in ("online", "online_nf", "ewma", "offline"):
                rules[f"{mode} c={c}"] = [adaptive_boundaries(s, c=c, mode=mode) for s in deffs]
        # oracle (참조 상한)
        d_or = best_score_dstar(dialogs, deffs)
        rules["oracle best-Score"] = [boundaries(s, d_or) for s in deffs]

        first = True
        for name, yps in rules.items():
            r = score_yps(dialogs, yps)
            tag = ds if first else ""
            first = False
            print(f"{tag:11} {name:22} {r['score']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['pk']:>6.3f} {r['wd']:>6.3f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
