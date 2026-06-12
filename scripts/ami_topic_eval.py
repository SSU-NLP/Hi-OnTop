#!/usr/bin/env python3
"""AMI topic-segmentation: Hi-OnTop robustness + 분절 입도(granularity) 분석.

DTS 3벤치 밖 도메인(비즈니스 회의)에서 Hi-OnTop 성능 검증 + AMI 계층 라벨
(top-level vs 하위)을 이용한 분절 입도 프로파일.

지표
----
- Pk/WD/F1/Score vs **top-level** 경계 (굵은 화제) — 표준 robustness
- Pk/WD/F1/Score vs **all-level** 경계 (하위 포함) — 촘촘한 분절
- **입도 프로파일**: 예측 경계 각각이 top-hit / sub-only-hit / false-alarm 중
  무엇인지 (±tol turn 허용). percentile(p60/p70/p80) 별로 입도가 어떻게
  이동하는지 → "percentile 로 분절 입도 조절 가능" 실증.

harness = run_encoder_comparison (delta_eff, official Pk/WD, Score 동일).
인코더 = MiniLM-int8 (ONNX quint8_avx2). delta* = calib 미팅 delta_eff percentile.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import delta_eff_seq, official_pk_wd  # noqa: E402

TOPIC = REPO / "data" / "ami" / "topic"
CACHE = REPO / "outputs" / "runs" / "_misc" / "ami_emb"
PERCENTILES = (80, 90, 95, 97, 98, 99)  # AMI 회의는 화제 드뭄 → 높은 p 필요
TOL = 2  # 경계 매칭 허용 turn 거리


def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider",
                      "file_name": "onnx/model_quint8_avx2.onnx"})


def encode_meeting(enc, mid: str, turns: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{mid}.pkl"
    if cp.exists():
        with open(cp, "rb") as f:
            return pickle.load(f)
    emb = np.asarray(enc.encode(turns, normalize_embeddings=True,
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


def granularity(pred_idx: list[int], gold_top: set[int], gold_sub: set[int],
                tol: int) -> dict:
    """예측 경계 분류: top-hit / sub-only-hit / false-alarm."""
    def near(i, gold):
        return any(abs(i - g) <= tol for g in gold)
    top_hit = sub_hit = fa = 0
    for i in pred_idx:
        if near(i, gold_top):
            top_hit += 1
        elif near(i, gold_sub):
            sub_hit += 1
        else:
            fa += 1
    # recall
    def covered(g, preds):
        return sum(1 for x in g if any(abs(x - p) <= tol for p in preds))
    preds = pred_idx
    return {
        "n_pred": len(pred_idx),
        "top_hit": top_hit, "sub_hit": sub_hit, "false_alarm": fa,
        "top_recall": covered(gold_top, preds) / len(gold_top) if gold_top else 0.0,
        "sub_recall": covered(gold_sub, preds) / len(gold_sub) if gold_sub else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=None, help="제한(디버그)")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--name", default="2026-06-08_ami_topic")
    args = ap.parse_args()

    man = json.load(open(TOPIC / "manifest.json"))
    mids = [m["meeting"] for m in man]
    if args.meetings:
        mids = [m for m in mids if m in args.meetings]
    mids.sort()

    # calib/test split (deterministic)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(mids))
    cut = int(round(len(idx) * args.calib_frac))
    calib_ids = sorted(mids[i] for i in idx[:cut])
    test_ids = sorted(mids[i] for i in idx[cut:])
    print(f"미팅 {len(mids)} → calib {len(calib_ids)} / test {len(test_ids)}", flush=True)

    enc = encoder()

    # 로드 + 인코딩 + delta_eff
    data: dict[str, dict] = {}
    t0 = time.perf_counter()
    for k, mid in enumerate(mids):
        d = json.load(open(TOPIC / f"{mid}.json"))
        emb = encode_meeting(enc, mid, [t["text"] for t in d["turns"]])
        deff = delta_eff_seq(emb)
        data[mid] = {"deff": deff, "bnd_top": d["bnd_top"], "bnd_all": d["bnd_all"]}
        if (k + 1) % 20 == 0:
            print(f"  encoded {k+1}/{len(mids)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # delta* = calib pooled delta_eff percentile
    pool = np.array([v for mid in calib_ids for v in data[mid]["deff"][1:]])
    dstars = {p: float(np.percentile(pool, p)) for p in PERCENTILES}
    print("delta*:", {p: round(v, 4) for p, v in dstars.items()}, flush=True)

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"dstars": dstars, "calib_n": len(calib_ids),
                     "test_n": len(test_ids), "tol": TOL, "per_p": {}}

    for p in PERCENTILES:
        dstar = dstars[p]
        rows_top, rows_all = [], []
        gran_tot = {"top_hit": 0, "sub_hit": 0, "false_alarm": 0, "n_pred": 0}
        recalls_top, recalls_sub = [], []
        for mid in test_ids:
            d = data[mid]
            yp = boundaries(d["deff"], dstar)
            rows_top.append(score_one(d["bnd_top"], yp))
            rows_all.append(score_one(d["bnd_all"], yp))
            pred_idx = [i for i, b in enumerate(yp) if b]
            gtop = {i for i, b in enumerate(d["bnd_top"]) if b}
            gall = {i for i, b in enumerate(d["bnd_all"]) if b}
            gsub = gall - gtop
            g = granularity(pred_idx, gtop, gsub, TOL)
            for kk in ("top_hit", "sub_hit", "false_alarm", "n_pred"):
                gran_tot[kk] += g[kk]
            recalls_top.append(g["top_recall"])
            recalls_sub.append(g["sub_recall"])

        npred = max(1, gran_tot["n_pred"])
        results["per_p"][p] = {
            "dstar": dstar,
            "top": {k: round(agg(rows_top, k), 4) for k in ("pk", "wd", "f1", "score")},
            "all": {k: round(agg(rows_all, k), 4) for k in ("pk", "wd", "f1", "score")},
            "gran": {
                "n_pred": gran_tot["n_pred"],
                "top_hit_frac": round(gran_tot["top_hit"] / npred, 4),
                "sub_hit_frac": round(gran_tot["sub_hit"] / npred, 4),
                "false_alarm_frac": round(gran_tot["false_alarm"] / npred, 4),
                "top_recall": round(float(np.mean(recalls_top)), 4),
                "sub_recall": round(float(np.mean(recalls_sub)), 4),
            },
        }
        r = results["per_p"][p]
        print(f"\n[p{p}] delta*={dstar:.4f}", flush=True)
        print(f"  top : Pk={r['top']['pk']:.3f} WD={r['top']['wd']:.3f} "
              f"F1={r['top']['f1']:.3f} Score={r['top']['score']:.3f}", flush=True)
        print(f"  all : Pk={r['all']['pk']:.3f} WD={r['all']['wd']:.3f} "
              f"F1={r['all']['f1']:.3f} Score={r['all']['score']:.3f}", flush=True)
        print(f"  gran: top-hit {r['gran']['top_hit_frac']:.2f} / "
              f"sub-hit {r['gran']['sub_hit_frac']:.2f} / "
              f"FA {r['gran']['false_alarm_frac']:.2f} | "
              f"recall top {r['gran']['top_recall']:.2f} sub {r['gran']['sub_recall']:.2f}",
              flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_report(out_dir, results, args)
    print(f"\nDONE → {out_dir}/REPORT.md", flush=True)


def write_report(out_dir: Path, R: dict, args) -> None:
    L = [
        f"# AMI Topic Segmentation — Hi-OnTop robustness + 입도 분석 ({args.name})",
        "",
        "## 실험 setup",
        "- **목적**: DTS 3벤치(TIAGE/Dialseg711/SuperDialseg) 밖 도메인(비즈니스 회의)"
        " 에서 Hi-OnTop 성능 유지 + AMI 계층 topic 라벨로 분절 입도 진단.",
        "- **데이터**: AMI scenario meetings manual annotation (NITE-XML). "
        f"turn=AMI segment, 경계=topic onset. calib {R['calib_n']} / test {R['test_n']} 미팅.",
        "- **인코더**: MiniLM-int8 (ONNX quint8_avx2, 384d). Hi-OnTop m=2,ρ=0.7,a=0.5.",
        "- **delta***: calib 미팅 delta_eff percentile (무라벨). "
        f"p60/p70/p80 = {R['dstars'][60]:.3f}/{R['dstars'][70]:.3f}/{R['dstars'][80]:.3f}.",
        f"- **경계 매칭 허용**: ±{R['tol']} turn.",
        "- **Score** = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).",
        "",
        "## 결과 — top-level 경계 (굵은 화제, 표준)",
        "",
        "| p | δ* | Pk↓ | WD↓ | F1↑ | Score↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in PERCENTILES:
        t = R["per_p"][p]["top"]
        L.append(f"| p{p} | {R['per_p'][p]['dstar']:.4f} | {t['pk']:.3f} | "
                 f"{t['wd']:.3f} | {t['f1']:.3f} | **{t['score']:.3f}** |")
    L += [
        "",
        "## 결과 — all-level 경계 (하위 포함, 촘촘)",
        "",
        "| p | Pk↓ | WD↓ | F1↑ | Score↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for p in PERCENTILES:
        a = R["per_p"][p]["all"]
        L.append(f"| p{p} | {a['pk']:.3f} | {a['wd']:.3f} | {a['f1']:.3f} | "
                 f"**{a['score']:.3f}** |")
    L += [
        "",
        "## 분절 입도 프로파일 (예측 경계의 정체)",
        "",
        "각 예측 경계가 top-level 정답과 일치(top-hit) / 하위 정답과만 일치(sub-hit) /"
        " 어느 것도 아님(false-alarm) 인지. percentile 낮을수록 잘게 쪼갬.",
        "",
        "| p | #예측 | top-hit | sub-hit | false-alarm | top-recall | sub-recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in PERCENTILES:
        g = R["per_p"][p]["gran"]
        L.append(f"| p{p} | {g['n_pred']} | {g['top_hit_frac']:.2f} | "
                 f"{g['sub_hit_frac']:.2f} | {g['false_alarm_frac']:.2f} | "
                 f"{g['top_recall']:.2f} | {g['sub_recall']:.2f} |")
    L += [
        "",
        "## 해석 포인트",
        "- **top-level Score** 가 DTS 3벤치 수준이면 → 도메인 무관 robustness 입증.",
        "- **percentile↓ 시 sub-hit/false-alarm↑, top-recall 유지** → 낮은 p 가 하위 화제까지"
        " 내려가며 분절. **percentile 로 분절 입도(top↔하위) 조절 가능** 실증.",
        "- top-recall 이 sub-recall 보다 일관되게 높으면 → Hi-OnTop 은 굵은 화제를 우선"
        " 포착하고 하위는 p 를 낮춰야 잡힌다는 위계적 동작.",
        "",
        "## 한계 / 검증 미해결",
        "- turn=AMI segment(화자 turn). 멀티파티 인터리브 스트림이라 DTS 단일대화와 구조 상이.",
        "- 경계=topic onset turn 매핑(±tol). overlap 으로 onset turn 모호한 경우 있음.",
        "- delta* 는 AMI calib 으로 재calibration (도메인 적응). cross-domain 고정 delta* 는 별도.",
        "- top-level 경계 수가 미팅당 평균 ~7개로 적어 F1 분산 큼 (미팅 수로 평균).",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
