#!/usr/bin/env python3
"""Oracle 실험 — prototype = '평균' vs '진짜 요약(gold topic description)'.

질문: prototype 을 running-mean 대신 AMI gold topic 요약(other_description=topic_levels[].label)
의 임베딩으로 쓰면 경계 신호가 좋아지나? (= '요약다운 요약'이 평균 대비 headroom 이 있나)
둘 다 oracle(gold 경계/topic 사용) — 신호 천장만 비교. de-neut V 신호 + per-meeting 임계 sweep, ±2F1.
"""
from __future__ import annotations
import sys, json, glob, pickle, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import _encoder

TOPIC = "data/ami/topic"; AC = "outputs/runs/_misc/ami_emb"


def nr(v):
    return v / (np.linalg.norm(v) + 1e-9)


def deneut(x, g, beta):
    xc = x - beta * float(x @ g) * g
    return xc / (np.linalg.norm(xc) + 1e-9)


def beta_rl(k, A=2.0, B=1.0, L0=8):
    return min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)


def sig_mean(e, gold, lam=0.6, g_rho=0.15, rho_min=0.05):
    """gold-reset 평균 prototype (= V_rel oracle 재현)."""
    n = len(e); gs = set(gold); s = np.zeros(n); m = e[0].copy(); k = 1; g = e[0].copy()
    for t in range(1, n):
        x = e[t]; b = beta_rl(k)
        s[t] = (1 - float(deneut(x, g, b) @ deneut(m, g, b))) - lam * (1 - float(x @ g))
        if t in gs:
            m = x.copy(); k = 1
        else:
            rho = max(rho_min, 1.0 / (k + 1)); m = nr((1 - rho) * m + rho * x); k += 1
        gr = max(g_rho, 1.0 / (t + 1)); g = nr((1 - gr) * g + gr * x)
    return s


def sig_label(e, gold, topic_of, label_embs, lam=0.6, g_rho=0.15):
    """prototype = 현재(직전 turn) topic 의 gold 요약 임베딩."""
    n = len(e); s = np.zeros(n); g = e[0].copy(); seg_start = 0
    gs = set(gold)
    for t in range(1, n):
        x = e[t]
        tp = topic_of[t - 1]; m = label_embs[tp]; k = (t - 1) - seg_start + 1
        b = beta_rl(k)
        s[t] = (1 - float(deneut(x, g, b) @ deneut(m, g, b))) - lam * (1 - float(x @ g))
        if t in gs:
            seg_start = t
        gr = max(g_rho, 1.0 / (t + 1)); g = nr((1 - gr) * g + gr * x)
    return s


def oracle_f1(s, gold, n, tol=2, ncand=80):
    cand = sorted(set(float(s[i]) for i in range(1, n - 1)))
    if len(cand) > ncand:
        cand = list(np.quantile(cand, np.linspace(0, 1, ncand)))
    best = 0.0
    for thr in cand:
        pred = [i for i in range(1, n - 1) if s[i] > thr]
        if not pred:
            continue
        p = sum(1 for i in pred if any(abs(i - j) <= tol for j in gold)) / len(pred)
        r = sum(1 for j in gold if any(abs(i - j) <= tol for i in pred)) / len(gold)
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        best = max(best, f)
    return best


def main():
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    # 모든 topic label 수집 → 한 번에 인코딩
    meta = []
    all_labels = []
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json"))
        bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
        gold = [i for i, b in enumerate(bt) if b == 1]
        tl1 = sorted([x for x in d["topic_levels"] if x["depth"] == 1], key=lambda z: z["start_turn"])
        labels = [x["label"] for x in tl1]
        # segment(=topic) index per turn from gold boundaries
        seg_starts = [0] + gold
        topic_of = np.zeros(n, dtype=int)
        for si, st in enumerate(seg_starts):
            topic_of[st:] = si
        nseg = len(seg_starts)
        # 라벨 수와 segment 수 정렬 (불일치 시 짧은 쪽에 맞춤)
        L = min(nseg, len(labels))
        if nseg != len(labels):
            pass  # 경미한 불일치 허용
        base = len(all_labels)
        all_labels.extend(labels[:L] if L else ["topic"])
        e = nr(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl", "rb")), dtype=np.float64))
        meta.append((mid, e, gold, n, topic_of, base, L))
    enc = _encoder("minilm-int8")
    lab_emb_all = nr(np.asarray(enc.encode(all_labels, normalize_embeddings=True), dtype=np.float64))

    Fm, Fl, skipped = [], [], 0
    for mid, e, gold, n, topic_of, base, L in meta:
        if n <= 3 or not gold or L == 0:
            skipped += 1; continue
        # 이 미팅의 label 임베딩 (topic idx → emb); 범위 넘으면 마지막으로 clamp
        labs = lab_emb_all[base:base + L]
        topic_clamped = np.minimum(topic_of, L - 1)
        sm = sig_mean(e, gold)
        sl = sig_label(e, gold, topic_clamped, labs)
        Fm.append(oracle_f1(sm, gold, n)); Fl.append(oracle_f1(sl, gold, n))
    print(f"AMI {len(Fm)}미팅 (skip {skipped}). prototype oracle ±2F1 비교:")
    print(f"  평균(gold-reset mean) prototype : {np.mean(Fm):.3f}")
    print(f"  요약(gold topic label) prototype: {np.mean(Fl):.3f}")
    print(f"  차이(label−mean)               : {np.mean(Fl)-np.mean(Fm):+.3f}")
    win = sum(1 for a, b in zip(Fl, Fm) if a > b)
    print(f"  미팅별 label>mean: {win}/{len(Fm)}")


if __name__ == "__main__":
    main()
