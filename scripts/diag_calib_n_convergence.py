#!/usr/bin/env python3
"""Calib N convergence 분석 — Hi-OnTop portability.

목적: Hi-OnTop 를 토픽 분절 외 데이터에도 제안하려면 "calib 비용" 이 작아야
한다. (인코더, 데이터) 셀 마다 **N\*** = test-side oracle 천장의 −0.005 이내로
들어오는 최소 calib N (3-seed σ < 0.005) 를 측정.

각 셀에서:
1. N ∈ {25, 50, 100, 200, 400, max} 3-seed bootstrap → mean ± σ Score (p80 / best)
2. test-side oracle Score (calib 무관 절대 천장) 계산
3. N\* (p80) = mean(score) >= oracle - 0.005 이며 σ < 0.005 인 최소 N

데이터:
- TIAGE: train 300 (max N=300)
- Dialseg711: test 70:30 (calib 498 max)
- SuperDialseg: train 6948 (현재 400 encoded; N=2000 encoding 진행 중 → 후처리 가능)
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
    load_dialogs, score_set, best_score_dstar,
)

CACHE = REPO / "outputs" / "runs" / "_misc"
ENCS = ("mpnet", "minilm", "minilm-int8")
SEEDS = [0, 1, 2]
TOL_SCORE = 0.005   # Score 가 oracle 의 −TOL 이내면 도달로 간주
TOL_STD = 0.005     # σ 안정성 threshold


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
    if p.exists():
        with open(p, "rb") as fh:
            return [np.asarray(e) for e in pickle.load(fh)]
    raise FileNotFoundError(f"no test cache: {enc}/{ds}")


def load_calib(enc: str, ds: str):
    """(calib_dia, calib_emb, test_dia, test_emb) 반환."""
    if ds == "dialseg711":
        # test 70:30
        all_dia = load_dialogs(ds, "test")
        all_emb = load_test_emb(enc, ds, all_dia)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(all_dia))
        cut = int(round(len(idx) * 0.70))
        ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
        return ([all_dia[i] for i in ci], [all_emb[i] for i in ci],
                [all_dia[i] for i in ti], [all_emb[i] for i in ti])
    elif ds == "tiage":
        # train 300, test 100, full encoded
        tr_dia = load_dialogs(ds, "train")
        # train cache: mpnet 은 별도 캐시 있음, 그 외엔 enccmp_tiage_train_{enc}
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
        # 400 calib (현재 encoded 한도)
        tr_dia_all = load_dialogs(ds, "train")
        # RNG 상태 재현 (인코더 패스 순서 기반)
        rng = np.random.default_rng(0)
        for e in ENCS:
            _ = rng.permutation(711)
            perm = rng.permutation(6948)
            if e == enc:
                break
        idx_400 = sorted(map(int, perm[:400]))
        with open(CACHE / f"enccmp_{ds}_train_{enc}.pkl", "rb") as fh:
            tr_emb = [np.asarray(e) for e in pickle.load(fh)]
        assert len(tr_emb) == 400
        tr_dia = [tr_dia_all[i] for i in idx_400]
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


def n_grid_for(cap: int) -> list[int]:
    base = [25, 50, 100, 200, 400, 1000, 2000]
    out = [n for n in base if n <= cap]
    if out[-1] != cap:
        out.append(cap)
    return out


def main() -> None:
    summary = {}  # (enc, ds) -> dict
    for enc in ENCS:
        for ds in ("tiage", "dialseg711", "superseg"):
            print(f"\n========== {enc} / {ds} ==========", flush=True)
            cal_dia, cal_emb, te_dia, te_emb = load_calib(enc, ds)
            cap = len(cal_dia)
            ns = n_grid_for(cap)
            print(f"  calib cap={cap}, N grid={ns}", flush=True)
            cal_deff = [delta_eff_seq(e) for e in cal_emb]
            te_deff = [delta_eff_seq(e) for e in te_emb]
            oracle_score, oracle_dstar = test_oracle(te_dia, te_deff)
            print(f"  test-side oracle: Score={oracle_score:.4f} "
                  f"δ*={oracle_dstar:.4f}", flush=True)
            rows = []
            for N in ns:
                sp80, sbs = [], []
                if N == cap:
                    # full → no bootstrap, single point
                    seed_iter = [None]
                else:
                    seed_iter = SEEDS
                for seed in seed_iter:
                    if seed is None:
                        idx = list(range(cap))
                    else:
                        rng = np.random.default_rng(seed)
                        idx = sorted(rng.permutation(cap)[:N])
                    cd = [cal_dia[i] for i in idx]
                    cde = [cal_deff[i] for i in idx]
                    allv = np.array([d for s in cde for d in s[1:]])
                    dstar_p80 = float(np.percentile(allv, 80))
                    dstar_bs = best_score_dstar(cd, cde)
                    sp80.append(score_set(te_dia, te_deff, dstar_p80)["score"])
                    sbs.append(score_set(te_dia, te_deff, dstar_bs)["score"])
                rows.append((N, np.mean(sp80), np.std(sp80),
                             np.mean(sbs), np.std(sbs)))
                print(f"    N={N:5d}  p80={np.mean(sp80):.4f}±{np.std(sp80):.4f}  "
                      f"best={np.mean(sbs):.4f}±{np.std(sbs):.4f}", flush=True)
            # N* (p80): mean >= oracle - TOL_SCORE AND std < TOL_STD
            n_star_p80 = None
            for (N, mp, sp, mb, sb) in rows:
                if mp >= oracle_score - TOL_SCORE and sp < TOL_STD:
                    n_star_p80 = N; break
            n_star_bs = None
            for (N, mp, sp, mb, sb) in rows:
                if mb >= oracle_score - TOL_SCORE and sb < TOL_STD:
                    n_star_bs = N; break
            summary[(enc, ds)] = dict(
                rows=rows, oracle=oracle_score, cap=cap,
                n_star_p80=n_star_p80, n_star_bs=n_star_bs)
            print(f"  → N*(p80)={n_star_p80} N*(best)={n_star_bs}", flush=True)

    # ---- REPORT ----
    out = REPO / "outputs" / "experiments" / "2026-05-23_calib_n_convergence"
    out.mkdir(parents=True, exist_ok=True)

    L = ["# Calib N convergence — Hi-OnTop portability 분석",
         "",
         "**목적**: Hi-OnTop 를 토픽 분절 외 데이터에도 제안하려면 calib 비용 "
         "(N\\*) 을 알아야 함. (인코더, 벤치) 셀 마다 천장 도달 최소 N 측정.",
         "",
         "**판정**: N\\* = mean Score ≥ test-side oracle − 0.005 AND 3-seed σ "
         "< 0.005 인 최소 N.",
         "",
         "**HP**: Hi-OnTop m=2, ρ=0.7, a=0.5. metric = 공식 SuperDialseg.",
         ""]

    # summary table
    L += ["## 요약: N* (천장 도달 최소 calib dialog 수)",
          "",
          "| 벤치 | calib cap | MPNet N\\* (p80 / best) | MiniLM N\\* (p80 / best) "
          "| MiniLM-int8 N\\* (p80 / best) |",
          "|---|---:|---:|---:|---:|"]
    for ds in ("tiage", "dialseg711", "superseg"):
        cells = []
        cap_str = str(summary[("mpnet", ds)]["cap"])
        for enc in ENCS:
            s = summary[(enc, ds)]
            p80 = s["n_star_p80"] or f">{s['cap']}"
            bs = s["n_star_bs"] or f">{s['cap']}"
            cells.append(f"{p80} / {bs}")
        L.append(f"| {ds} | {cap_str} | " + " | ".join(cells) + " |")
    L += ["",
          "> N\\* 가 `> cap` 이면 현재 calib 범위 내에서 천장 미도달 (더 큰 "
          "calib 또는 더 큰 알고리즘 capacity 필요).",
          ""]

    # per-cell detail
    L += ["## 셀별 N-Score 곡선 (3-seed bootstrap, mean ± σ)", ""]
    for enc in ENCS:
        for ds in ("tiage", "dialseg711", "superseg"):
            s = summary[(enc, ds)]
            L += [f"### {enc} / {ds}", "",
                  f"**oracle (calib 무관 천장)**: Score = "
                  f"{s['oracle']:.4f}", "",
                  "| N | Score (p80) | Score (best) | gap to oracle "
                  "(best) |",
                  "|---:|---:|---:|---:|"]
            for (N, mp, sp, mb, sb) in s["rows"]:
                gap = s["oracle"] - mb
                L.append(f"| {N} | {mp:.4f} ± {sp:.4f} | "
                         f"{mb:.4f} ± {sb:.4f} | {gap:+.4f} |")
            L.append("")

    L += ["## 인사이트 가이드",
          "",
          "1. **N\\* 작은 셀** → portability 좋음. 이 (인코더, 데이터) 조합은 "
          "적은 calib 으로 즉시 배포 가능.",
          "2. **N\\* = `> cap`** → 현재 calib 풀 안에서 천장 미도달. 더 많은 "
          "calib 가 도움 될 수 있음 (특히 superseg N=2000 결과 대기 중).",
          "3. **p80 N\\* vs best N\\* 차이**:",
          "   - p80 N\\* 만 클 경우 → unsupervised (label-free) 가 supervised "
          "보다 calib 더 필요. raw cost 는 낮지만 양 필요.",
          "   - best N\\* 가 클 경우 → labeled calib 도 많이 필요. 데이터 "
          "본질적으로 어려움.",
          "4. **천장-크기 무관성**: 별도 분석 (`outputs/experiments/2026-05-23_"
          "data_size_vs_ceiling/`) 에서 천장이 데이터 크기 와 무관함을 확인. "
          "여기 N\\* 는 *천장 도달 비용* 의 지표.",
          ""]

    if any(summary[(enc, "superseg")]["cap"] == 400 for enc in ENCS):
        L += ["## TODO",
              "",
              "- superseg cap=400 → N=2000 encoding 진행 중. 완료 시 "
              "diag_superseg_calib_2000*.py 결과로 superseg N grid 확장 후 "
              "재계산.",
              ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nreport → {out/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
