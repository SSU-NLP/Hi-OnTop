#!/usr/bin/env python3
"""데이터 크기 vs 천장 — 3 벤치 × 3 인코더 격자.

질문: 천장 (test-side oracle) 이 데이터 크기 (총 대화 수, 총 턴 수) 와
상관 있나? 아니면 데이터 자체 난이도 (인코더와 무관) 가 지배적인가?

방법:
1. 벤치별 size 표 (calib + test, dialog 수 / turn 수).
2. (인코더, 벤치) 격자 9 셀 모두에서 test-side oracle Score (δ* sweep on test,
   data-snooping 허용) → 천장 측정.
3. 한 표 안에 size 와 천장 같이.
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


def delta_eff_seq(emb):
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def load_test_emb(enc: str, ds: str, dialogs):
    """test 캐시 우선순위: MPNET_REUSE → enccmp_{ds}_test_{enc} → enccmp_{ds}_{split}_{enc}."""
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
    raise FileNotFoundError(f"no test cache for {enc}/{ds}")


def main() -> None:
    # ---- size 표 ----
    sizes = {}  # ds -> {calib_dia, calib_turn, test_dia, test_turn, has_train}
    for ds in BENCHES:
        if ds == "dialseg711":
            test = load_dialogs(ds, "test")
            # 70:30 split (run_encoder_comparison 과 동일 seed=0)
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(test))
            cut = int(round(len(idx) * 0.70))
            calib_di = [test[i] for i in idx[:cut]]
            test_di = [test[i] for i in idx[cut:]]
            calib_dia = len(calib_di); calib_turn = sum(len(d[0]) for d in calib_di)
            test_dia = len(test_di); test_turn = sum(len(d[0]) for d in test_di)
            sizes[ds] = dict(calib_dia=calib_dia, calib_turn=calib_turn,
                             test_dia=test_dia, test_turn=test_turn,
                             note="test 70:30")
        else:
            tr = load_dialogs(ds, "train")
            te = load_dialogs(ds, "test")
            sizes[ds] = dict(calib_dia=len(tr),
                             calib_turn=sum(len(d[0]) for d in tr),
                             test_dia=len(te),
                             test_turn=sum(len(d[0]) for d in te),
                             note="train split")

    # ---- 천장 격자 ----
    ceilings = {}  # (enc, ds) -> {oracle_score, oracle_dstar}
    for enc in ENCS:
        for ds in BENCHES:
            te_dia = load_dialogs(ds, "test")
            try:
                te_emb = load_test_emb(enc, ds, te_dia)
            except FileNotFoundError as e:
                print(f"[skip] {enc}/{ds}: {e}", flush=True)
                ceilings[(enc, ds)] = None
                continue
            # dialseg711 은 70:30 test 사이드만 (label leakage 차단 위해)
            if ds == "dialseg711":
                rng = np.random.default_rng(0)
                idx = rng.permutation(len(te_dia))
                cut = int(round(len(idx) * 0.70))
                ti = sorted(idx[cut:])
                te_dia = [te_dia[i] for i in ti]
                te_emb = [te_emb[i] for i in ti]
            te_deff = [delta_eff_seq(e) for e in te_emb]
            best = (-1.0, 0.0)
            for d in DSTAR_GRID:
                sc = score_set(te_dia, te_deff, d)["score"]
                if sc > best[0]:
                    best = (sc, float(d))
            ceilings[(enc, ds)] = dict(score=best[0], dstar=best[1],
                                       n_dia=len(te_dia),
                                       n_turn=sum(len(d[0]) for d in te_dia))
            print(f"[{enc}/{ds}] test n_dia={ceilings[(enc,ds)]['n_dia']} "
                  f"n_turn={ceilings[(enc,ds)]['n_turn']}  "
                  f"oracle Score={best[0]:.4f} (δ*={best[1]:.4f})",
                  flush=True)

    # ---- 표 출력 ----
    print("\n=== 데이터 크기 (calib + test) ===\n")
    print(f"{'벤치':14} | {'calib dia':>10} | {'calib turn':>11} | "
          f"{'test dia':>9} | {'test turn':>10} | note")
    print("-" * 80)
    for ds in BENCHES:
        s = sizes[ds]
        print(f"{ds:14} | {s['calib_dia']:>10} | {s['calib_turn']:>11} | "
              f"{s['test_dia']:>9} | {s['test_turn']:>10} | {s['note']}")

    print("\n=== test-side oracle (절대 천장, calib 무관) ===\n")
    print(f"{'벤치':14} | "
          + " | ".join(f"{e:>20}" for e in ENCS))
    print("-" * 80)
    for ds in BENCHES:
        cells = []
        for enc in ENCS:
            c = ceilings.get((enc, ds))
            if c is None:
                cells.append(f"{'-':>20}")
            else:
                cells.append(f"  Score={c['score']:.4f} δ*={c['dstar']:.3f}")
        print(f"{ds:14} | " + " | ".join(cells))

    # ---- REPORT ----
    out = REPO / "outputs" / "experiments" / "2026-05-23_data_size_vs_ceiling"
    out.mkdir(parents=True, exist_ok=True)
    L = [
        "# 데이터 크기 vs 천장 — 3 벤치 × 3 인코더 격자",
        "",
        "**질문**: 천장 (test-side oracle Score) 이 데이터 크기 (대화 수 / 턴 수) "
        "와 상관 있나? 아니면 인코더 + 벤치 난이도 가 지배적인가?",
        "",
        "**방법**: 각 (인코더, 벤치) 셀에서 test 자체에서 δ* sweep [0.35, 0.95] "
        "(data-snooping 허용) → calib 와 무관한 알고리즘+데이터 천장.",
        "",
        "**HP**: Hi-OnTop m=2, ρ=0.7, a=0.5. metric = 공식 SuperDialseg "
        "(0.5F1+0.25(1−Pk)+0.25(1−WD)).",
        "",
        "## 데이터 크기",
        "",
        "| 벤치 | calib dialog | calib turn | test dialog | test turn | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for ds in BENCHES:
        s = sizes[ds]
        L.append(f"| {ds} | {s['calib_dia']} | {s['calib_turn']} | "
                 f"{s['test_dia']} | {s['test_turn']} | {s['note']} |")
    L += ["", "## test-side oracle Score (천장) — 인코더 × 벤치",
          "",
          "| 벤치 | MPNet | MiniLM | MiniLM-int8 |",
          "|---|---:|---:|---:|"]
    for ds in BENCHES:
        cells = []
        for enc in ENCS:
            c = ceilings.get((enc, ds))
            cells.append(f"{c['score']:.4f}" if c else "—")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    L += ["", "## test-side oracle δ* — 인코더 × 벤치",
          "",
          "| 벤치 | MPNet | MiniLM | MiniLM-int8 |",
          "|---|---:|---:|---:|"]
    for ds in BENCHES:
        cells = []
        for enc in ENCS:
            c = ceilings.get((enc, ds))
            cells.append(f"{c['dstar']:.3f}" if c else "—")
        L.append(f"| {ds} | " + " | ".join(cells) + " |")
    L += ["",
          "## 해석 가이드",
          "",
          "- 같은 벤치 안에서 인코더별 천장 차 작으면 → **벤치 난이도** 지배적.",
          "- 같은 인코더 안에서 벤치별 천장 차 크면 → **데이터 특성** 지배적.",
          "- 데이터 크기 (turn 수) 큰 벤치에서 천장이 *낮으면* → 크기 ≠ 점수. "
          "어려운 데이터가 큰 경우.",
          "- δ* 가 인코더 마다 크게 다르면 → 인코더 별 cosine 분포 자체가 달라 "
          "calib 불가피.",
    ]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nreport → {out/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
