#!/usr/bin/env python3
"""단일-config 후보생성기 비교 — recall 천장 (LLM 비용 0, codex 1순위, 2026-06-20).

[[feedback_one_tool_no_per_dataset_tuning]] 준수: 모든 후보생성기는 전역 상수 1개(budget q 또는 c)만,
데이터셋별 튜닝 없음. 4개 벤치(AMI ±2tol shift0 / DTS exact shift-1)에 동일 적용.

codex 진단: μ+cσ(가변 발화율) 단일 c 는 dense(superseg 23%)에서 6%만 발화 → 천장 붕괴.
대안: **고정-quantile(상수 발화율 = budget)** + local abruptness/window novelty/union(multi-scale).
고정-q 는 dense 든 sparse 든 budget 만큼 발화 → dense recall 확보, sparse 는 judge 가 거름.

지표(천장만): cand_rate=후보율, recall=gold 중 후보가 ±tol 내 덮은 비율, pjF1=perfect-judge F1(precision→1 가정).
사용: python scripts/measure_candidate_generators.py
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))
from run_encoder_comparison import load_dialogs, delta_eff_seq, CACHE  # noqa: E402
from hi_ontop import ami_scoring as AS  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

WIN = 8  # window novelty 윈도우 (전역 상수)


def _nr_rows(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


# ---- per-turn novelty 신호들 (전부 online·전역상수, reset 없음) ----
def sig_abrupt(e):
    """local abruptness a_t = 1 - cos(x_t, x_{t-1})."""
    n = len(e); s = np.zeros(n)
    for t in range(1, n):
        s[t] = 1 - float(e[t] @ e[t - 1])
    return s


def sig_window(e, W=WIN):
    """window novelty = 1 - cos(x_t, mean(last W))."""
    n = len(e); s = np.zeros(n)
    for t in range(1, n):
        lo = max(0, t - W); m = e[lo:t].mean(0); m = m / (np.linalg.norm(m) + 1e-9)
        s[t] = 1 - float(e[t] @ m)
    return s


def sig_deltaeff(e):
    return np.asarray(list(delta_eff_seq(e)))


def topq_positions(sig, n, q):
    """interior turn(1..n-2)에서 sig 상위 q 비율 = spike turn 집합."""
    cand_t = list(range(1, n - 1))
    if not cand_t:
        return set()
    k = max(1, int(round(q * n)))
    order = sorted(cand_t, key=lambda i: -sig[i])
    return set(order[:k])


# ---- generator: spike turn 집합 반환 (전역 config) ----
def gen_quant(e, sig_fn, q):
    n = len(e); return topq_positions(sig_fn(e), n, q)


def gen_union(e, q):
    """multi-scale union: abrupt 상위 q/2 ∪ window 상위 q/2."""
    n = len(e)
    return topq_positions(sig_abrupt(e), n, q / 2) | topq_positions(sig_window(e), n, q / 2)


# ---- 적응형 generator (단일 config, 발화율이 stream 에 따라 변함) ----
def gen_peak(e, sig_fn, W=20, k=1.0):
    """density-적응 peak: s_t 가 직전보다 상승(s_t>s_{t-1}) AND 과거 W 의 robust baseline(med+k·MAD) 초과.
    실제 prominent bump 수에 비례해 발화 → dense stream 자연히 후보↑, stable 자연히 ↓. 0-lag."""
    s = sig_fn(e); n = len(e); out = set()
    for t in range(2, n - 1):
        lo = max(1, t - W); past = s[lo:t]
        if len(past) < 3:
            continue
        med = float(np.median(past)); mad = float(np.median(np.abs(past - med))) + 1e-9
        if s[t] > s[t - 1] and (s[t] - med) > k * 1.4826 * mad:
            out.add(t)
    return out


def gen_rollz(e, sig_fn, c=1.0, W=40, warmup=8):
    """rolling μ+cσ (전역 평균 아닌 최근 W). 0-lag, refractory 없음."""
    s = sig_fn(e); n = len(e); out = set()
    for t in range(1, n - 1):
        lo = max(0, t - W); past = s[lo + 1:t] if lo == 0 else s[lo:t]
        if len(past) < warmup:
            continue
        if s[t] > float(np.mean(past)) + c * float(np.std(past)):
            out.add(t)
    return out


# ---- 평가 (천장) ----
def eval_ami(data, genfn):
    rates, recs, pjf = [], [], []
    for e, bt, gold in data:
        n = len(e); spikes = {t for t in genfn(e) if 0 < t < n - 1}
        rates.append(len(spikes) / n)
        if gold:
            cov = sum(1 for g in gold if any(abs(g - t) <= 2 for t in spikes))
            recs.append(cov / len(gold))
            true_t = [t for t in spikes if any(abs(t - g) <= 2 for g in gold)]
            pjf.append(AS.tol_f1(gold, true_t))
    return float(np.mean(rates)), float(np.mean(recs)), float(np.mean(pjf))


def eval_dts(data, genfn):
    rates, recs, pjf = [], [], []
    for e, yt in data:
        n = len(yt); gold = set(i for i, b in enumerate(yt) if b == 1)
        spikes = {t for t in genfn(e) if 0 < t < n - 1}
        bset = {t - 1 for t in spikes}  # shift -1
        rates.append(len(spikes) / n)
        if gold:
            recs.append(len(gold & bset) / len(gold))
            ytl = list(yt); ytl[-1] = 0
            yp = [1 if (i in bset and i in gold) else 0 for i in range(n)]
            pjf.append(f1_score(ytl, yp, zero_division=0))
    return float(np.mean(rates)), float(np.mean(recs)), float(np.mean(pjf))


def load_ami():
    TOPIC = os.path.join(REPO, "data/ami/topic"); EMB = os.path.join(REPO, "outputs/runs/_misc/ami_emb")
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
    out = []
    for mid in mids:
        bt = list(json.load(open(f"{TOPIC}/{mid}.json"))["bnd_top"]); bt[-1] = 0
        e = _nr_rows(np.asarray(pickle.load(open(f"{EMB}/{mid}.pkl", "rb")), dtype=np.float64))
        out.append((e, bt, [i for i, b in enumerate(bt) if b == 1]))
    return out


def load_dts_ds(ds):
    dl = load_dialogs(ds, "test")
    emb = [_nr_rows(np.asarray(x, dtype=np.float64))
           for x in pickle.load(open(CACHE / f"enccmp_{ds}_test_minilm-int8.pkl", "rb"))]
    return [(e, yt) for (u, yt), e in zip(dl, emb)]


if __name__ == "__main__":
    AMI = load_ami()
    DTS = {ds: load_dts_ds(ds) for ds in ("tiage", "dialseg711", "superseg")}
    gens = []
    # 적응형 (발화율이 stream 따라 변함 — 밀도적응)
    for sg, sf in (("deff", sig_deltaeff), ("abrupt", sig_abrupt)):
        for k in (0.5, 1.0, 1.5):
            gens.append((f"peak[{sg}] k={k}", lambda e, sf=sf, k=k: gen_peak(e, sf, W=20, k=k)))
        for c in (0.5, 1.0):
            gens.append((f"rollz[{sg}] c={c}", lambda e, sf=sf, c=c: gen_rollz(e, sf, c=c, W=40)))
    # 참조용 고정-quantile
    for q in (0.20, 0.25):
        gens.append((f"FIX deff q={q}", lambda e, q=q: gen_quant(e, sig_deltaeff, q)))

    print("후보생성기 천장 비교 (단일 config, 모든 벤치 동일). 형식: rate/recall/pjF1")
    hdr = f"{'generator':<16}" + "".join(f"{d:>22}" for d in ("AMI(±2)", "tiage", "dialseg711", "superseg"))
    print(hdr)
    for name, gf in gens:
        ar, arc, apf = eval_ami(AMI, gf)
        cells = [f"{ar:.2f}/{arc:.2f}/{apf:.2f}"]
        for ds in ("tiage", "dialseg711", "superseg"):
            r, rc, pf = eval_dts(DTS[ds], gf)
            cells.append(f"{r:.2f}/{rc:.2f}/{pf:.2f}")
        print(f"{name:<16}" + "".join(f"{c:>22}" for c in cells), flush=True)
    print("\n(참조 현 de-neut c=0.5 천장 pjF1: AMI 0.81 / tiage 0.10 / dialseg 0.28 / superseg 0.11)")
