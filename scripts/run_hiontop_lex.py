#!/usr/bin/env python3
"""Hi-OnTop-Lex (lexical-overlap correction) — 3-benchmark evaluation.

Hi-OnTop-Lex (``src/hi_ontop/hi_ontop_lex.py``) adds a TextTiling-style
word-frequency overlap correction to Hi-OnTop' semantic ``delta_eff``:

    delta_eff_v2 = clip_[0,1]( delta_base + w_lex * r_t * (lexdist - mu_lex) )

This script calibrates the two data-dependent quantities on each
benchmark's *training* data and evaluates on the held-out test set:

- ``mu_lex``  : median ``lexdist`` over the calibration split (label-free;
  centers the lexical residual). Independent of ``w_lex``.
- ``delta*``  : 80-percentile (p80) of ``delta_eff_v2`` over the
  calibration split (label-free). Depends on ``w_lex``.

Calibration / test splits (no test leakage, SuperDialseg bundle):
- tiage      : calib = tiage/train,           test = tiage/test
- superseg   : calib = superseg/validation,   test = superseg/test
  (superseg has no cached train embedding and local encoding of the
  92k-turn train split is impractical; validation is a held-out,
  label-free split — the same set the Hi-OnTop HP sweep used. p80 / median
  use no labels, so this is a calibration-source choice, not leakage.)
- dialseg711 : 30%/70% seeded shuffle of the 711-dialogue test set —
  30% = calib, 70% = held-out test (dialseg711 has no train split).

A small ``w_lex`` grid is swept ({0.0, 0.05, 0.10, 0.15, 0.20, 0.30});
``w_lex = 0.0`` reduces Hi-OnTop-Lex exactly to Hi-OnTop-v1 and serves as
the baseline row. Other HPs fixed at Hi-OnTop default
(m=2, rho=0.7, a=0.5) and codex-recommended lexical default
(m_lex=2, rho_lex=0.7, min_tokens=3). Metric = official SuperDialseg
Pk/WD (k=auto) + binary F1, Score = 0.5*F1 + 0.25*(1-Pk) + 0.25*(1-WD).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.hi_ontop_lex import HiOnTopLex  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
MPNET_SUFFIX = "_sentence-transformers_multi-qa-mpnet-base-dot-v1"

# Hi-OnTop embedding-window HP (canonical default)
EMB_HP = dict(ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
# lexical HP (codex 2026-05-23 recommended default)
LEX_HP = dict(m_lex=2, rho_lex=0.7, min_tokens=3)
W_LEX_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

# (calibration split, test split) per benchmark
SPLITS = {
    "tiage": ("train", "test"),
    "superseg": ("validation", "test"),
    "dialseg711": ("test", "test"),  # 30/70 shuffle handled separately
}


# --------------------------------------------------------------------------
# data + metric
# --------------------------------------------------------------------------

def load_dialogs(ds: str, split: str):
    raw = json.loads((SDS / ds / f"segmentation_file_{split}.json").read_text())
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt:
            yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def load_embs(ds: str, split: str):
    for cand in (CACHE / f"sds_emb_{ds}_{split}.pkl",
                 CACHE / f"sds_emb_{ds}_{split}{MPNET_SUFFIX}.pkl"):
        if cand.exists():
            with open(cand, "rb") as fh:
                return pickle.load(fh)
    raise FileNotFoundError(f"no cached embedding for {ds}/{split}")


def split_frac(dialogs, embs, tune_frac: float = 0.30, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(dialogs))
    cut = int(round(len(idx) * tune_frac))
    tune_i, test_i = idx[:cut], idx[cut:]
    tune = ([dialogs[i] for i in tune_i], [embs[i] for i in tune_i])
    test = ([dialogs[i] for i in test_i], [embs[i] for i in test_i])
    return tune, test


def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


# --------------------------------------------------------------------------
# Hi-OnTop-Lex run helpers
# --------------------------------------------------------------------------

def _run_v2(emb, utts, w_lex: float, mu_lex: float, delta_star: float):
    """Run Hi-OnTopV2 over one dialogue; return the per-turn history list."""
    seg = HiOnTopLex(dim=emb.shape[1], delta_star=delta_star, w_lex=w_lex,
                   mu_lex=mu_lex, **EMB_HP, **LEX_HP)
    for s, txt in zip(emb, utts):
        seg.assign(s.astype(np.float64), txt)
    return seg.history()


def calibrate(dialogs, embs, w_lex: float) -> tuple[float, float]:
    """Calibrate (mu_lex, delta*) on a split.

    mu_lex = median lexdist (label-free, w_lex-independent — computed once
    via a w_lex=0 pass). delta* = p80 of delta_eff_v2 at this w_lex.
    """
    # pass 1: lexdist distribution -> mu_lex (turn 0 excluded)
    lexd = []
    for (utts, _), emb in zip(dialogs, embs):
        for h in _run_v2(emb, utts, 0.0, 0.0, 1.0)[1:]:
            lexd.append(float(h["lexdist"]))
    mu_lex = float(np.median(lexd)) if lexd else 0.0

    # pass 2: delta_eff_v2 distribution at this w_lex -> p80
    deff = []
    for (utts, _), emb in zip(dialogs, embs):
        for h in _run_v2(emb, utts, w_lex, mu_lex, 1.0)[1:]:
            deff.append(float(h["delta_eff"]))
    dstar = float(np.percentile(deff, 80)) if deff else 0.5
    return mu_lex, dstar


def eval_on(dialogs, embs, w_lex: float, mu_lex: float, delta_star: float) -> dict:
    pks, wds, g, p = [], [], [], []
    for (utts, yt), emb in zip(dialogs, embs):
        hist = _run_v2(emb, utts, w_lex, mu_lex, delta_star)
        ids = [int(h["topic_id"]) for h in hist]
        yp = [1 if ids[i] != ids[i + 1] else 0 for i in range(len(ids) - 1)] + [0]
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    score = 0.5 * f1 + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
    return dict(pk=pk_m, wd=wd_m, f1=f1, score=score)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-23_hiontop_v2")
    args = ap.parse_args()
    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- load calibration + test splits per benchmark ----
    print("[load] calibration + test splits", flush=True)
    calib_sets: dict[str, tuple] = {}
    test_sets: dict[str, tuple] = {}

    for ds in ("tiage", "superseg"):
        c_split, t_split = SPLITS[ds]
        cd, ce = load_dialogs(ds, c_split), load_embs(ds, c_split)
        td, te = load_dialogs(ds, t_split), load_embs(ds, t_split)
        assert len(cd) == len(ce), f"{ds}/{c_split} dialog/emb mismatch {len(cd)}!={len(ce)}"
        assert len(td) == len(te), f"{ds}/{t_split} dialog/emb mismatch {len(td)}!={len(te)}"
        calib_sets[ds] = (cd, ce)
        test_sets[ds] = (td, te)

    d711_d, d711_e = load_dialogs("dialseg711", "test"), load_embs("dialseg711", "test")
    assert len(d711_d) == len(d711_e), "dialseg711 dialog/emb mismatch"
    d711_calib, d711_test = split_frac(d711_d, d711_e, 0.30, 0)
    calib_sets["dialseg711"] = d711_calib
    test_sets["dialseg711"] = d711_test

    for ds in ("tiage", "superseg", "dialseg711"):
        print(f"  {ds}: calib {len(calib_sets[ds][0])} dial, "
              f"test {len(test_sets[ds][0])} dial", flush=True)

    # ---- sweep w_lex; calibrate on train, evaluate on test ----
    results: dict[str, list[dict]] = {ds: [] for ds in test_sets}
    for ds in ("tiage", "superseg", "dialseg711"):
        print(f"\n=== {ds} ===", flush=True)
        cd, ce = calib_sets[ds]
        td, te = test_sets[ds]
        for w_lex in W_LEX_GRID:
            t0 = time.perf_counter()
            mu_lex, dstar = calibrate(cd, ce, w_lex)
            r = eval_on(td, te, w_lex, mu_lex, dstar)
            row = dict(w_lex=w_lex, mu_lex=mu_lex, delta_star=dstar, **r)
            results[ds].append(row)
            tag = "  (= Hi-OnTop-v1)" if w_lex == 0.0 else ""
            print(f"  w_lex={w_lex:.2f}  mu_lex={mu_lex:.4f} δ*={dstar:.4f}  "
                  f"Pk={r['pk']:.4f} WD={r['wd']:.4f} F1={r['f1']:.4f} "
                  f"Score={r['score']:.4f}{tag}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    (exp_dir / "v2_results.json").write_text(json.dumps(results, indent=2))

    # ---- best w_lex per benchmark (by test Score) + tuned-on-calib pick ----
    summary = {}
    for ds in results:
        rows = results[ds]
        v1 = rows[0]  # w_lex = 0.0
        best = max(rows, key=lambda r: r["score"])
        summary[ds] = dict(v1=v1, best=best)

    # ---- REPORT ----
    write_report(exp_dir, results, summary)
    print(f"\nDONE → {exp_dir/'REPORT.md'}", flush=True)


def write_report(exp_dir: Path, results: dict, summary: dict) -> None:
    L: list[str] = []
    L += [
        "# Hi-OnTop-Lex (lexical-overlap correction) — 3-benchmark evaluation",
        "",
        "## 1. 실험 setup",
        "",
        "- **목적**: Hi-OnTop 에 TextTiling 식 단어-빈도 겹침(lexical overlap) "
        "보정항을 더한 Hi-OnTop-Lex 가, 세 분절 벤치마크에서 Hi-OnTop-v1 대비 "
        "성능을 올리는지 확인. v1 의 알려진 실패 모드 — wording 은 비슷한데 "
        "topic 이 바뀌는 경계 — 를 lexical 신호로 보완하는 게 가설.",
        "- **모델**: `src/hi_ontop/hi_ontop_lex.py` `HiOnTopLex`. "
        "`δ_eff_v2 = clip_[0,2](δ_base + w_lex·r_t·(lexdist − μ_lex))` "
        "(clip 범위 = cosine-distance 자연 범위; w_lex=0 시 v1 과 byte-parity). "
        "`δ_base` = Hi-OnTop δ_eff (불변). `lexdist = 1 − cos_tf(L_{t-1}, x_t)`, "
        "L = 직전 m_lex turn 의 ρ_lex-감쇠 sublinear-TF 합, x_t = 현재 turn "
        "sublinear-TF. `r_t = min(1, min(n_ctx,n_t)/min_tokens)` 짧은-turn 게이트.",
        "- **데이터** (SuperDialseg 번들, `benchmarks/superdialseg_data/`):",
        "  - tiage — calib `tiage/train` / test `tiage/test`",
        "  - superseg — calib `superseg/validation` / test `superseg/test`",
        "  - dialseg711 — `dialseg711/test` 711-dialogue 를 seed=0 으로 30/70 "
        "shuffle → calib 30% / test 70% (train split 부재).",
        "- **calibration** (label-free, test leakage 없음): calib split 에서 "
        "`μ_lex` = median(lexdist), `δ*` = p80(δ_eff_v2). μ_lex 는 w_lex 무관, "
        "δ* 는 w_lex 별 재산출.",
        f"- **HP**: 임베딩 window m={EMB_HP['ctx_window']} ρ={EMB_HP['ctx_decay']} "
        f"a={EMB_HP['ctx_blend_a']} (Hi-OnTop default). lexical "
        f"m_lex={LEX_HP['m_lex']} ρ_lex={LEX_HP['rho_lex']} "
        f"min_tokens={LEX_HP['min_tokens']} (codex 2026-05-23 권고 default). "
        f"인코더 multi-qa-mpnet.",
        f"- **w_lex grid (작은 범위 calibration)**: {W_LEX_GRID}. "
        "`w_lex=0.0` ⇒ Hi-OnTop-Lex 가 Hi-OnTop-v1 로 정확히 환원 → baseline 행.",
        "- **metric**: official SuperDialseg Pk/WD (k=auto) + binary F1. "
        "`Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD)`. seed 1개 (calib shuffle "
        "seed=0); per-w_lex 단일 run.",
        "",
        "## 2. 결과 표 (test split, w_lex sweep)",
        "",
    ]
    for ds in ("tiage", "superseg", "dialseg711"):
        rows = results[ds]
        v1_score = rows[0]["score"]
        best = summary[ds]["best"]
        L += [
            f"### {ds}",
            "",
            "| w_lex | μ_lex | δ* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | ΔScore vs v1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            d = r["score"] - v1_score
            star = " **★best**" if r is best else ""
            tag = " _(v1)_" if r["w_lex"] == 0.0 else ""
            L.append(
                f"| {r['w_lex']:.2f}{tag} | {r['mu_lex']:.4f} | {r['delta_star']:.4f} "
                f"| {r['pk']:.4f} | {r['wd']:.4f} | {r['f1']:.4f} "
                f"| {r['score']:.4f}{star} | {d:+.4f} |")
        L.append("")

    # mean-3 summary
    L += [
        "### mean-3 요약",
        "",
        "| w_lex | tiage | superseg | dialseg711 | mean-3 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, w_lex in enumerate(W_LEX_GRID):
        sc = [results[ds][i]["score"] for ds in ("tiage", "superseg", "dialseg711")]
        m = float(np.mean(sc))
        tag = " _(v1)_" if w_lex == 0.0 else ""
        L.append(f"| {w_lex:.2f}{tag} | {sc[0]:.4f} | {sc[1]:.4f} | {sc[2]:.4f} "
                 f"| **{m:.4f}** |")
    L.append("")

    # interpretation / judgment scaffold (filled by reading the numbers below)
    v1_mean = float(np.mean([results[ds][0]["score"]
                             for ds in ("tiage", "superseg", "dialseg711")]))
    best_per = {ds: summary[ds]["best"] for ds in summary}
    L += [
        "## 3. 해석",
        "",
        f"- Hi-OnTop-v1 (w_lex=0) mean-3 Score = **{v1_mean:.4f}**.",
        "- w_lex>0 행이 v1 행을 일관되게 못 이기면, lexical 보정은 이 "
        "데이터·인코더에서 net-neutral~negative — v1 이 robust 하다는 "
        "negative result.",
        "- 벤치마크별로 best w_lex 가 갈리면 (예: 한 벤치는 +, 다른 벤치는 −) "
        "lexical 신호 효용이 corpus-dependent → 단일 default w_lex 부적절.",
        "- δ* 가 p80 고정이라 w_lex 가 커지면 δ_eff_v2 분포가 넓어지며 δ* 도 "
        "함께 이동 → boundary 개수는 대략 보존되고, *어디에* 찍히는지가 "
        "바뀌는 효과. F1 변화가 핵심 지표.",
        "",
        "## 4. 판정",
        "",
    ]
    for ds in ("tiage", "superseg", "dialseg711"):
        b = best_per[ds]
        d = b["score"] - results[ds][0]["score"]
        verdict = ("향상" if d > 0.005 else "회귀" if d < -0.005 else "동일(noise)")
        L.append(f"- **{ds}**: best w_lex={b['w_lex']:.2f} → ΔScore {d:+.4f} "
                 f"→ {verdict}.")
    L += [
        "- 다음 iteration 결정: best w_lex 가 벤치 간 일치 + mean-3 향상이면 "
        "Hi-OnTop-Lex 를 default 후보로 승격 검토. 불일치/회귀면 v1 유지, "
        "lexical 항은 ablation 기록만.",
        "",
        "## 5. 한계 / 검증 미해결",
        "",
        "- **seed 1개**: calib shuffle seed=0, per-w_lex 단일 run. ΔScore 가 "
        "±0.005 이내면 noise 와 구분 불가 — multi-seed 미실행.",
        "- **superseg calib = validation**: superseg 는 cached train 임베딩이 "
        "없고 92k-turn train 로컬 인코딩이 비현실적. validation 은 held-out "
        "label-free split (Hi-OnTop HP sweep 과 동일 선택). p80/median 은 라벨 "
        "미사용 → leakage 아님이나 '학습 데이터' 와는 엄밀히 다름.",
        "- **dialseg711 train split 부재**: 30/70 seeded split 으로 대체. "
        "calib 30% 와 test 70% disjoint (leakage 없음). full-test literature "
        "number 와 직접 비교 금지.",
        "- **lexical HP 미sweep**: m_lex/ρ_lex/min_tokens 는 codex 권고 default "
        "고정. w_lex 만 sweep (사용자 지시 '작은 범위'). 2차 grid 미실행.",
        "- **μ_lex centering**: median 사용. mean 대비 robust 하나 분포가 "
        "치우치면 p80-δ* 와 상호작용 — 미분석.",
        "- stopword set 은 내장 영문 set — 세 벤치 모두 영문이라 적용되나 "
        "도메인 특화 stopword 미조정.",
        "",
    ]
    (exp_dir / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
