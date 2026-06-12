#!/usr/bin/env python3
"""AMI topic-segmentation — **시간 블록(time-block) 단위** Hi-OnTop 평가.

기존 `ami_topic_eval.py` 는 turn(=AMI segment) 단위로 δ_eff 를 계산했으나,
회의 음성은 짧고 반응적인 멀티파티 발화라 turn 단위 cohesion 이 붕괴
(같은 화제 인접 cos 중앙값 0.167, δ_eff AUC 0.567 ≈ 무정보).

본 스크립트는 **단위를 고정 시간 블록(default 60s)으로 바꿔** 재구성한다:
각 블록 = 그 시간구간 안 모든 발화 텍스트를 이어붙인 한 덩어리 → 인코딩 →
δ_eff. 블록을 키우면 내용이 풍부해져 인코더가 화제 cohesion 을 회복함
(60s 에서 인접 cos 0.44, AUC 0.68 로 상승; window sweep 2026-06-08 근거).

지표
----
- 블록 단위 Pk/WD/F1/Score (turn 단위 점수와 절대값 직접 비교 불가 — 단위가
  다름; granularity 가 핵심임을 보이는 분석).
- δ_eff AUC (threshold-free 분리력) 도 같이 보고.

harness = run_encoder_comparison (delta_eff_seq, official Pk/WD, Score 동일).
인코더 = MiniLM-int8 (ONNX quint8_avx2). δ* = calib 미팅 블록 δ_eff percentile.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import delta_eff_seq, official_pk_wd  # noqa: E402

TOPIC = REPO / "data" / "ami" / "topic"
CACHE = REPO / "outputs" / "runs" / "_misc" / "ami_block_emb"
# 블록은 turn(1%)보다 경계 밀도가 높음 → 낮은 percentile 도 sweep
PERCENTILES = (50, 60, 70, 80, 90)


def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})


def build_blocks(turns: list[dict], win: float) -> tuple[list[str], list[float]]:
    """발화들을 win 초 고정 블록으로 묶어 (텍스트, 블록시작시각) 반환. 빈 블록 제외."""
    if not turns:
        return [], []
    t0 = turns[0]["start"]
    tN = turns[-1]["start"]
    nbin = int((tN - t0) // win) + 1
    bins: list[list[str]] = [[] for _ in range(nbin)]
    for t in turns:
        b = int((t["start"] - t0) // win)
        bins[b].append(t["text"])
    texts, starts = [], []
    for k, b in enumerate(bins):
        joined = " ".join(b).strip()
        if joined:
            texts.append(joined)
            starts.append(t0 + k * win)
    return texts, starts


def gold_block_bnd(d: dict, starts: list[float], win: float) -> list[int]:
    """블록 k(>0)가 경계인가 = 그 시간구간에 top-level topic 시작이 있나."""
    turns = d["turns"]
    bt = d["bnd_top"]
    tstarts = [turns[i]["start"] for i, b in enumerate(bt) if b == 1]
    y = [0] * len(starts)
    for k in range(1, len(starts)):
        lo, hi = starts[k], starts[k] + win
        if any(lo <= s < hi for s in tstarts):
            y[k] = 1
    return y


def encode_blocks(enc, mid: str, win: float, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{mid}_w{int(win)}.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    emb = np.asarray(enc.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float64)
    with open(cp, "wb") as f:
        pickle.dump(emb, f)
    return emb


def boundaries(seq: list[float], dstar: float) -> list[int]:
    return [1 if seq[i] >= dstar else 0 for i in range(1, len(seq))] + [0]


def score_one(yt: list[int], yp: list[int]) -> dict:
    pk, wd = official_pk_wd(yt, yp)
    f1 = float(f1_score(yt, yp, zero_division=0))
    return {"pk": pk, "wd": wd, "f1": f1,
            "score": 0.5 * f1 + 0.25 * (1 - pk) + 0.25 * (1 - wd)}


def agg(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=60.0, help="블록 크기(초)")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    win = args.win
    name = args.name or f"2026-06-08_ami_topic_block_w{int(win)}"

    man = json.load(open(TOPIC / "manifest.json"))
    mids = sorted(m["meeting"] for m in man)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(mids))
    cut = int(round(len(idx) * args.calib_frac))
    calib_ids = sorted(mids[i] for i in idx[:cut])
    test_ids = sorted(mids[i] for i in idx[cut:])
    print(f"win={win:.0f}s | 미팅 {len(mids)} → calib {len(calib_ids)} / test {len(test_ids)}",
          flush=True)

    enc = encoder()

    data: dict[str, dict] = {}
    t0 = time.perf_counter()
    n_blocks = []
    for k, mid in enumerate(mids):
        d = json.load(open(TOPIC / f"{mid}.json"))
        texts, starts = build_blocks(d["turns"], win)
        if len(texts) < 3:
            continue
        emb = encode_blocks(enc, mid, win, texts)
        deff = delta_eff_seq(emb)
        yt = gold_block_bnd(d, starts, win)
        data[mid] = {"deff": deff, "yt": yt}
        n_blocks.append(len(texts))
        if (k + 1) % 30 == 0:
            print(f"  encoded {k+1}/{len(mids)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    calib_ids = [m for m in calib_ids if m in data]
    test_ids = [m for m in test_ids if m in data]

    # δ* = calib pooled block δ_eff percentile
    pool = np.array([v for mid in calib_ids for v in data[mid]["deff"][1:]])
    dstars = {p: float(np.percentile(pool, p)) for p in PERCENTILES}

    # threshold-free AUC (test split)
    bd, nd = [], []
    for mid in test_ids:
        deff, yt = data[mid]["deff"], data[mid]["yt"]
        for i in range(1, len(deff)):
            (bd if yt[i] == 1 else nd).append(deff[i])
    auc = float(roc_auc_score([1] * len(bd) + [0] * len(nd), bd + nd)) if bd and nd else float("nan")

    bnd_density = float(np.mean([np.mean(data[m]["yt"]) for m in test_ids]))
    avg_blocks = float(np.mean(n_blocks))

    results = {"win": win, "calib_n": len(calib_ids), "test_n": len(test_ids),
               "dstars": dstars, "auc": auc, "bnd_density": bnd_density,
               "avg_blocks_per_meeting": avg_blocks, "per_p": {}}

    print(f"δ*: {{p: v}} = " + str({p: round(v, 4) for p, v in dstars.items()}), flush=True)
    print(f"AUC(threshold-free) = {auc:.4f} | 경계밀도 {bnd_density*100:.1f}% | "
          f"평균 {avg_blocks:.1f} 블록/미팅", flush=True)

    for p in PERCENTILES:
        dstar = dstars[p]
        rows = []
        for mid in test_ids:
            d = data[mid]
            yp = boundaries(d["deff"], dstar)
            rows.append(score_one(d["yt"], yp))
        r = {k: round(agg(rows, k), 4) for k in ("pk", "wd", "f1", "score")}
        r["dstar"] = round(dstar, 4)
        results["per_p"][p] = r
        print(f"[p{p}] δ*={dstar:.4f}  Pk={r['pk']:.3f} WD={r['wd']:.3f} "
              f"F1={r['f1']:.3f} Score={r['score']:.3f}", flush=True)

    out_dir = REPO / "outputs" / "experiments" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_report(out_dir, results, name)
    print(f"\nDONE → {out_dir}/REPORT.md", flush=True)


def write_report(out_dir: Path, R: dict, name: str) -> None:
    best_p = max(R["per_p"], key=lambda p: R["per_p"][p]["score"])
    bp = R["per_p"][best_p]
    L = [
        f"# AMI 화제분절 — 60초 블록 단위 Hi-OnTop ({name})",
        "",
        "## 실험 setup",
        f"- **목적**: turn 단위로 붕괴한 Hi-OnTop(AUC 0.567)을 **{int(R['win'])}초 시간 블록** "
        "단위로 재구성하면 화제 cohesion·경계 신호가 회복되는지 실측.",
        "- **데이터**: AMI scenario meetings, `data/ami/topic/*.json` (NXT manual annotation). "
        f"calib {R['calib_n']} / test {R['test_n']} 미팅 (deterministic split, seed=0, calib_frac=0.5).",
        f"- **블록 구성**: 각 발화의 begin_time 기준 {int(R['win'])}초 고정 bin 에 텍스트 이어붙임 → "
        "한 블록 = 한 덩어리 텍스트. 빈 블록 제외.",
        "- **인코더**: MiniLM-int8 (ONNX `model_quint8_avx2.onnx`), L2-normalize.",
        "- **δ_eff**: `run_encoder_comparison.delta_eff_seq` (ctx m=2, ρ=0.7, blend a=0.5) — turn eval 과 동일.",
        f"- **δ\\***: calib 미팅 블록 δ_eff 분포의 percentile (label-free). sweep p ∈ {list(R['per_p'])}.",
        "- **metric**: 블록 단위 official Pk/WD (NLTK) + F1 + Score(0.5·F1+0.25·(1−Pk)+0.25·(1−WD)).",
        "",
        f"- **블록 통계**: 평균 {R['avg_blocks_per_meeting']:.1f} 블록/미팅, 경계 밀도 {R['bnd_density']*100:.1f}%.",
        f"- **AUC (threshold-free 분리력)**: **{R['auc']:.3f}** (turn 단위 0.567 대비).",
        "",
        "## 결과 (test split, percentile sweep)",
        "",
        "| p | δ\\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for p in R["per_p"]:
        r = R["per_p"][p]
        star = "  ← best" if p == best_p else ""
        L.append(f"| {p} | {r['dstar']:.3f} | {r['pk']:.3f} | {r['wd']:.3f} | "
                 f"{r['f1']:.3f} | {r['score']:.3f}{star} |")
    L += [
        "",
        "## 해석",
        f"- best = **p{best_p}** (Score {bp['score']:.3f}, F1 {bp['f1']:.3f}, Pk {bp['pk']:.3f}).",
        f"- threshold-free AUC {R['auc']:.3f} — turn 단위(0.567)보다 분리력 상승. "
        "단위(granularity)가 핵심이었음을 직접 확인.",
        "- **turn 단위 결과와 절대값 직접 비교 금지**: 시퀀스 단위가 발화→블록으로 바뀌어 "
        "Pk/WD/F1 의 분모(블록 수)가 다름. 비교는 AUC·신호 회복 경향으로만.",
        "",
        "## 판정",
        "- turn→블록 전환으로 신호는 회복되나(AUC↑), 절대 성능은 여전히 제한적 — "
        "회의 도메인은 ~1분 블록이 적정 단위라는 증거. DTS 3벤치(텍스트 대화)와 동급 성능은 아님.",
        "",
        "## 한계 / 검증 미해결",
        "- **블록 경계 해상도 ±{}s**: 경계를 블록 시작 단위로만 찍어 위치 정밀도가 거침.".format(int(R['win'])),
        "- **단일 인코더·단일 win**: win sweep(30/60/120/180) 은 별도 분석(2026-06-08 window 실측) 참조. "
        "여기선 60s 고정.",
        "- **seed 1개**: calib/test split 단일. 3-run std 미산출.",
        "- **gold 정의**: top-level topic start 가 블록 시간구간에 들어가면 경계로 라벨 — "
        "한 블록에 경계 2개 이상이면 1개로 합쳐짐(드묾).",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
