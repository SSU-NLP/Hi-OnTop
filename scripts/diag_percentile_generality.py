#!/usr/bin/env python3
"""Unsupervised percentile heuristic 일반화 — 분석 2.

질문: p80 만 oracle 근사하나? 아니면 p60/p70/p90 도 비슷하게 되나?
→ 인사이트가 p80 특정값에 묶이지 않고 percentile-heuristic 일반에 적용된다면
"라벨 없는 데이터에 portable" claim 이 더 강해짐.

방법:
- 각 (인코더, 벤치) 셀에서 **full calib** 사용:
    TIAGE: train 300 (전체)
    Dialseg711: test 70:30 calib 498 (전체)
    SuperDialseg: train N=2000 (이미 encoded; enccmp_superseg_train_{enc}_n2000.pkl)
- δ*_{p60, p70, p80, p90} 계산 → test 에서 Score 측정
- test-side oracle Score 와 비교 → gap 계산
- bootstrap 없음 (full calib, single point per cell)

표:
| 벤치 | 인코더 | p60 | p70 | p80 | p90 | oracle |  (Score)
| 벤치 | 인코더 | p60 | p70 | p80 | p90 | oracle |  (δ*)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hi_ontop.hi_ontop import HiOnTop  # noqa: E402
from run_encoder_comparison import (  # noqa: E402
    DSTAR_GRID, M, RHO, A, ENCODERS, MPNET_REUSE,
    load_dialogs, score_set,
)

CACHE = REPO / "outputs" / "runs" / "_misc"
ENCS = ("mpnet", "minilm", "minilm-int8")
BENCHES = ("tiage", "dialseg711", "superseg")
PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def delta_eff_seq(emb):
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def load_test_emb(enc: str, ds: str, dialogs):
    if enc == "mpnet" and (ds, "test") in MPNET_REUSE:
        p = CACHE / MPNET_REUSE[(ds, "test")]
        if p.exists():
            with open(p, "rb") as fh:
                cached = pickle.load(fh)
            if len(cached) == len(dialogs):
                return [np.asarray(e) for e in cached]
    p = CACHE / f"enccmp_{ds}_test_{enc}.pkl"
    with open(p, "rb") as fh:
        return [np.asarray(e) for e in pickle.load(fh)]


def load_calib_full(enc: str, ds: str):
    """(calib_dia, calib_emb, test_dia, test_emb) — full calib for each cell."""
    if ds == "dialseg711":
        all_dia = load_dialogs(ds, "test")
        all_emb = load_test_emb(enc, ds, all_dia)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(all_dia))
        cut = int(round(len(idx) * 0.70))
        ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
        return ([all_dia[i] for i in ci], [all_emb[i] for i in ci],
                [all_dia[i] for i in ti], [all_emb[i] for i in ti])
    elif ds == "tiage":
        tr_dia = load_dialogs(ds, "train")
        if enc == "mpnet":
            p = CACHE / MPNET_REUSE[(ds, "train")]
        else:
            p = CACHE / f"enccmp_{ds}_train_{enc}.pkl"
        with open(p, "rb") as fh:
            tr_emb = [np.asarray(e) for e in pickle.load(fh)]
        assert len(tr_dia) == len(tr_emb)
        te_dia = load_dialogs(ds, "test")
        te_emb = load_test_emb(enc, ds, te_dia)
        return tr_dia, tr_emb, te_dia, te_emb
    elif ds == "superseg":
        # full N=2000 from enccmp_superseg_train_{enc}_n2000.pkl (dict idx→emb)
        tr_dia_all = load_dialogs(ds, "train")
        rng = np.random.default_rng(0)
        for e in ENCS:
            _ = rng.permutation(711)
            perm = rng.permutation(6948)
            if e == enc:
                break
        idx_2000 = sorted(map(int, perm[:2000]))
        ext_p = CACHE / f"enccmp_{ds}_train_{enc}_n2000.pkl"
        with open(ext_p, "rb") as fh:
            idx_to_emb = pickle.load(fh)
        tr_dia = [tr_dia_all[i] for i in idx_2000]
        tr_emb = [np.asarray(idx_to_emb[i]) for i in idx_2000]
        te_dia = load_dialogs(ds, "test")
        te_emb = load_test_emb(enc, ds, te_dia)
        return tr_dia, tr_emb, te_dia, te_emb
    raise ValueError(ds)


def test_oracle(te_dia, te_deff):
    best = (-1.0, 0.0)
    for d in DSTAR_GRID:
        sc = score_set(te_dia, te_deff, d)["score"]
        if sc > best[0]:
            best = (sc, float(d))
    return best


def main() -> None:
    grid = {}  # (enc, ds) -> {p60/p70/p80/p90: (dstar, score), oracle: (dstar, score), cap}
    for enc in ENCS:
        for ds in BENCHES:
            print(f"\n========== {enc} / {ds} ==========", flush=True)
            cal_dia, cal_emb, te_dia, te_emb = load_calib_full(enc, ds)
            cap = len(cal_dia)
            print(f"  full calib cap = {cap} dialog", flush=True)
            cal_deff = [delta_eff_seq(e) for e in cal_emb]
            te_deff = [delta_eff_seq(e) for e in te_emb]
            allv = np.array([d for s in cal_deff for d in s[1:]])
            n_delta = len(allv)
            print(f"  calib δ_eff samples = {n_delta}", flush=True)
            cell = dict(cap=cap, n_delta=n_delta)
            for p in PERCENTILES:
                dstar = float(np.percentile(allv, p))
                score = score_set(te_dia, te_deff, dstar)["score"]
                cell[f"p{p}"] = (dstar, score)
                print(f"  p{p}  δ*={dstar:.4f}  Score={score:.4f}", flush=True)
            oracle_score, oracle_dstar = test_oracle(te_dia, te_deff)
            cell["oracle"] = (oracle_dstar, oracle_score)
            print(f"  oracle  δ*={oracle_dstar:.4f}  Score={oracle_score:.4f}",
                  flush=True)
            grid[(enc, ds)] = cell

    # ---- REPORT ----
    out = REPO / "outputs" / "experiments" / "2026-05-23_percentile_generality"
    out.mkdir(parents=True, exist_ok=True)
    pct_cols_md = " | ".join(f"p{p}" for p in PERCENTILES)
    pct_align_md = "---:|" * len(PERCENTILES)
    L = [
        "# Percentile heuristic 일반화 — 분석 2 (확장 p50–p95, 5-unit)",
        "",
        "**질문**: 단일 percentile (p80) 이 universal default 인가? 아니면 "
        "boundary 밀도 따라 다르나? → 일반화되면 \"라벨 없는 portable\" claim "
        "강화, 안되면 \"percentile family + 도메인 적응\" 형식의 claim 으로 약화.",
        "",
        "**방법**: 각 (인코더, 벤치) full calib (TIAGE 300 · Dialseg711 "
        "498 · SuperDialseg 2000) → δ\\*_{p50,…,p95} (5 단위) 계산 → test "
        "Score. test-side oracle 과 비교.",
        "",
        "**HP**: Hi-OnTop m=2, ρ=0.7, a=0.5. metric = 공식 SuperDialseg "
        "(0.5F1+0.25(1−Pk)+0.25(1−WD)).",
        "",
        "## Score (각 percentile 의 test Score; **bold** = 셀 best)",
        "",
        f"| 벤치 | 인코더 | calib N | {pct_cols_md} | oracle | best p | gap |",
        f"|---|---|---:|{pct_align_md}---:|---|---:|",
    ]
    for ds in BENCHES:
        for enc in ENCS:
            c = grid[(enc, ds)]
            scores = {p: c[f"p{p}"][1] for p in PERCENTILES}
            best_p = max(scores, key=lambda p: scores[p])
            gap = c["oracle"][1] - scores[best_p]
            cells = []
            for p in PERCENTILES:
                s = scores[p]
                mark = "**" if p == best_p else ""
                cells.append(f"{mark}{s:.4f}{mark}")
            L.append(f"| {ds} | {enc} | {c['cap']} | " + " | ".join(cells)
                     + f" | {c['oracle'][1]:.4f} | p{best_p} | "
                     f"{gap:+.4f} |")

    L += ["", "## δ\\* (각 percentile 의 값)", "",
          f"| 벤치 | 인코더 | {pct_cols_md} | oracle |",
          f"|---|---|{pct_align_md}---:|"]
    for ds in BENCHES:
        for enc in ENCS:
            c = grid[(enc, ds)]
            cells = [f"{c[f'p{p}'][0]:.4f}" for p in PERCENTILES]
            L.append(f"| {ds} | {enc} | " + " | ".join(cells)
                     + f" | {c['oracle'][0]:.4f} |")

    L += ["", "## Oracle 대비 gap (Score; **bold** = min gap)", "",
          f"| 벤치 | 인코더 | {pct_cols_md} | min p | min gap |",
          f"|---|---|{pct_align_md}---|---:|"]
    for ds in BENCHES:
        for enc in ENCS:
            c = grid[(enc, ds)]
            gaps = {p: c["oracle"][1] - c[f"p{p}"][1] for p in PERCENTILES}
            min_p = min(gaps, key=lambda p: gaps[p])
            cells = []
            for p in PERCENTILES:
                mark = "**" if p == min_p else ""
                cells.append(f"{mark}{gaps[p]:+.4f}{mark}")
            L.append(f"| {ds} | {enc} | " + " | ".join(cells)
                     + f" | p{min_p} | {gaps[min_p]:+.4f} |")

    # ---- 인사이트 자동 계산 ----
    avg_gap = {p: [] for p in PERCENTILES}
    for ds in BENCHES:
        for enc in ENCS:
            c = grid[(enc, ds)]
            for p in PERCENTILES:
                avg_gap[p].append(c["oracle"][1] - c[f"p{p}"][1])
    L += ["", "## 평균 gap (9 셀 평균, oracle − p_x)", "",
          "| percentile | 평균 gap |",
          "|---:|---:|"]
    for p in PERCENTILES:
        gs = avg_gap[p]
        L.append(f"| p{p} | {np.mean(gs):+.4f} (std {np.std(gs):.4f}) |")

    # 가장 좋은 percentile 비율
    best_count = {p: 0 for p in PERCENTILES}
    for ds in BENCHES:
        for enc in ENCS:
            c = grid[(enc, ds)]
            gaps = {p: c["oracle"][1] - c[f"p{p}"][1] for p in PERCENTILES}
            best_count[min(gaps, key=lambda p: gaps[p])] += 1
    L += ["", "## \"가장 oracle 에 가까운 percentile\" 분포 (9 셀)", "",
          "| percentile | 1st-place 수 / 9 |",
          "|---:|---:|"]
    for p in PERCENTILES:
        L.append(f"| p{p} | {best_count[p]} |")

    L += ["",
          "## 해석 가이드",
          "",
          "- **평균 gap 표** 가 작은 percentile (특히 p70~p80) 이 일반화 좋은 "
          "후보. 셋 다 비슷하면 \"percentile heuristic 자체\" 가 robust.",
          "- **1st-place 분포** 가 한 percentile 에 집중되지 않으면 → 데이터별 "
          "최적 percentile 다름 → 인코더 선택만큼 percentile 선택도 영향 있음.",
          "- **gap 이 ±0.01 안** 인 셀은 oracle 근사 잘 됨. 0.02 이상이면 그 "
          "percentile 이 그 (인코더, 데이터) 에서 구조적 미스매치.",
          ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nreport → {out/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
