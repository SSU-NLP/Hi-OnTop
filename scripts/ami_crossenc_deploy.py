#!/usr/bin/env python3
"""고정 cross-encoder coherence 를 경계 신호로 쓴 deploy — cosine 대비 ±2F1 실측.

가설(사용자): cosine 거리 대신 '다음 발화가 현재 화제에 자연스럽게 이어지나'를 고정 cross-encoder
(STS/coherence)로 판정하면 deploy 가 나아지나? prototype = 현재 segment 최근 K발화 텍스트,
신호 = 1 − CE_sim(발화_t, 최근맥락). 적응 임계치 μ+cσ(online) 로 경계. cosine de-neut deploy / δ_eff
와 같은 미팅·같은 임계 방식으로 비교. (cross-encoder forward/발화라 비싸 subset.)
"""
from __future__ import annotations
import sys, json, glob, pickle, argparse
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from hi_ontop.hi_ontop_cr import segment
from run_encoder_comparison import delta_eff_seq, official_pk_wd

TOPIC = "data/ami/topic"; AC = "outputs/runs/_misc/ami_emb"


def tol_f1(gold, pred, tol=2):
    pred = [p for p in pred if 0 < p]
    if not pred or not gold:
        return 0.0
    p = sum(1 for i in pred if any(abs(i - j) <= tol for j in gold)) / len(pred)
    r = sum(1 for j in gold if any(abs(i - j) <= tol for i in pred)) / len(gold)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def best_c_metrics(sig_or_segfn, golds, ns, cs=(2.0, 1.5, 1.2, 1.0, 0.8), is_sig=True):
    """c sweep 후 **Score 기준** best (deploy 적응임계). 반환: (c, Score, ±2F1, pred합)."""
    best = (-1, -1.0, 0.0, 0)
    for c in cs:
        F, SC = [], []; tot = 0
        for x, gold, n in zip(sig_or_segfn, golds, ns):
            if is_sig:
                pred = [i for i, b in enumerate(adaptive_boundaries(x, c=c, mode="ewma")) if b]
            else:
                pred = x(c)
            pred = [p for p in pred if 0 < p < n - 1]; tot += len(pred)
            f2 = tol_f1(gold, pred)
            yt = [1 if i in set(gold) else 0 for i in range(n)]
            yp = [1 if i in set(pred) else 0 for i in range(n)]
            pk, wd = official_pk_wd(yt, yp)
            F.append(f2); SC.append(0.5 * f2 + 0.25 * (1 - pk) + 0.25 * (1 - wd))
        sc = float(np.mean(SC))
        if sc > best[1]:
            best = (c, sc, float(np.mean(F)), tot)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--model", default="cross-encoder/stsb-distilroberta-base")
    args = ap.parse_args()

    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))[:args.limit]
    metas = []
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
        gold = [i for i, b in enumerate(bt) if b == 1]
        txt = [t["text"] for t in d["turns"]]
        e = np.asarray(pickle.load(open(f"{AC}/{mid}.pkl", "rb")), dtype=np.float64)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        metas.append((mid, n, gold, txt, e))

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(args.model, max_length=256)
    print(f"AMI {len(metas)}미팅, cross-encoder={args.model}, K={args.K}", flush=True)

    ce_sigs, golds, ns, segfns_cos, segfns_deff = [], [], [], [], []
    for mid, n, gold, txt, e in metas:
        pairs = [(txt[t], " ".join(txt[max(0, t - args.K):t])) for t in range(1, n)]
        sc = ce.predict(pairs, batch_size=64, show_progress_bar=False) if pairs else []
        sig = [0.0] + [-float(s) for s in sc]      # 낮은 coherence = 높은 신호(=경계)
        ce_sigs.append(sig); golds.append(gold); ns.append(n)
        segfns_cos.append(lambda c, e=e: segment(e, reset="threshold", c=c))
        segfns_deff.append(lambda c, e=e: [i for i, b in enumerate(
            adaptive_boundaries(list(delta_eff_seq(e)), c=c, mode="ewma")) if b])
        print(f"  {mid} done (n={n})", flush=True)

    # commit-refine(de-neut, drift deploy)도 포함
    segfns_cr = [lambda c, e=e: segment(e, reset="commit_refine", c=c) for *_, e in metas]
    rows = [
        ("δ_eff cosine", best_c_metrics(segfns_deff, golds, ns, is_sig=False)),
        ("de-neut cosine (threshold)", best_c_metrics(segfns_cos, golds, ns, is_sig=False)),
        ("de-neut commit-refine", best_c_metrics(segfns_cr, golds, ns, is_sig=False)),
        ("cross-encoder coherence", best_c_metrics(ce_sigs, golds, ns, is_sig=True)),
    ]
    print("\n=== deploy (best-c by Score, 같은 미팅·적응임계) ===")
    print(f"  {'method':<28}{'Score':>7}{'±2F1':>7}{'pred':>7}{'  c':>5}")
    for name, (c, sc, f2, tot) in rows:
        print(f"  {name:<28}{sc:>7.3f}{f2:>7.3f}{tot:>7}{c:>5}")


if __name__ == "__main__":
    main()
