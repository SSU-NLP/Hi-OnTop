#!/usr/bin/env python3
"""Hi-OnTop vs online 베이스라인 비교표 (Def-DTS 번들, 3 벤치 각각).

베이스라인(TextTiling/GreedySeg/GraphSeg)은 Def-DTS 번들 test set 에서
segeval Pk/WD + boundary-set F1 로 측정됨 → Hi-OnTop 도 **같은 데이터·같은
metric** 으로 평가해야 표가 유효하다 (superdialseg_data 와 turn 수가 달라
섞으면 안 됨).

Hi-OnTop δ* 두 가지:
- **p80**  : 각 벤치 test set 의 δ_eff 80-percentile (label 불필요 →
  full-test 에서 뽑아도 누수 없음, 베이스라인과 동일 데이터로 공정).
- **best-Score** : labeled train split 에서 Score 최대화 (supervised
  ceiling, 참고행). tiage/superseg = 번들 train split. dialseg711 은
  train split 부재 → test 에서 보정(‡ in-domain upper-bound, 공정비교 아님).

인코더 = multi-qa-mpnet (Hi-OnTop 기본, v4.1.x/Hi-OnTop 보고 수치와 동일 계열).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.baselines._seg_utils import (  # noqa: E402
    boundary_set_f1, latency_stats, parse_defdts_dialogue, pk_wd)
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

DEFDTS = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"
CACHE = REPO / "outputs" / "runs" / "_misc"
M, RHO, A = 2, 0.7, 0.5  # causal-window HP (Hi-OnTop default)

# 인코더 후보 — encode 시 모델 교체 가능. δ* 는 인코더별로 재보정.
# minilm-int8 = int8 양자화 ONNX (이 CPU 는 avx2 → quint8_avx2 파일).
ENCODERS = {
    "mpnet": {"model": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "minilm-int8": {"model": "sentence-transformers/all-MiniLM-L6-v2",
                    "backend": "onnx",
                    "file_name": "onnx/model_quint8_avx2.onnx"},
}
ENCODER = "mpnet"          # set by --encoder
DSTAR_LO, DSTAR_HI = 0.35, 0.95  # best-Score 탐색 범위 (인코더 적응형, mpnet·MiniLM 최적 둘 다 포함)

# 베이스라인 측정치 (각 REPORT.md 에서 그대로; Pk/WD/F1/Score/lat_ms)
BASELINES = {
    "tiage": {
        "TextTiling-style": (0.5266, 0.5476, 0.2252, 0.3441, 0.008),
        "GreedySeg-style":  (0.5370, 0.5535, 0.1420, 0.2984, 13.81),
        "GraphSeg-style":   (0.4925, 0.5159, 0.2517, 0.3738, 0.85),
    },
    "dialseg711": {
        "TextTiling-style": (0.4713, 0.4896, 0.2708, 0.3952, 0.010),
        "GreedySeg-style":  (0.4164, 0.4434, 0.4120, 0.4911, 16.27),
        "GraphSeg-style":   (0.4485, 0.4816, 0.3656, 0.4503, 1.15),
    },
    "superseg": {
        "TextTiling-style": (0.4623, 0.4667, 0.2618, 0.3987, 0.011),
        "GreedySeg-style":  (0.5072, 0.5109, 0.2775, 0.3842, 9.14),
        "GraphSeg-style":   (0.5387, 0.5423, 0.1644, 0.3120, 0.62),
    },
}


def load_split(ds: str, split: str):
    """Def-DTS 번들 {ds}_{split}.jsonl → [(id, utts, gold_bnds_1based)]."""
    path = DEFDTS / f"{ds}_{split}.jsonl"
    if not path.exists():
        return None
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        utts, bnds = parse_defdts_dialogue(row["dialogue"])
        if len(utts) >= 2:
            out.append((row["id"], utts, bnds))
    return out


_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        cfg = ENCODERS[ENCODER]
        if cfg.get("backend") == "onnx":
            _ENC = SentenceTransformer(
                cfg["model"], backend="onnx",
                model_kwargs={"provider": "CPUExecutionProvider",
                              "file_name": cfg["file_name"]})
        else:
            _ENC = SentenceTransformer(cfg["model"], device="cpu")
    return _ENC


def encode_cached(ds: str, split: str, dialogs):
    cp = CACHE / f"emb_defdts_{ds}_{split}_{ENCODER}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    # batch encode: 전체 발화를 한 번에 (대화 1개씩 호출이 병목이었음)
    flat, lens = [], []
    for _, utts, _ in dialogs:
        flat.extend(utts)
        lens.append(len(utts))
    allv = np.asarray(_encoder().encode(
        flat, normalize_embeddings=True, show_progress_bar=False))
    embs, i = [], 0
    for L in lens:
        embs.append(allv[i:i + L])
        i += L
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump(embs, fh)
    return embs


def delta_eff_seq(emb):
    """HiOnTop 의 turn별 δ_eff (δ* 무관). history()[i]['delta_eff']."""
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def score_at(dialogs, deff_seqs, delta_star):
    """주어진 δ* 로 Pk/WD/F1(per-dialog mean) + Score."""
    pk_v, wd_v, f1_v = [], [], []
    for (did, utts, gold), seq in zip(dialogs, deff_seqs):
        n = len(utts)
        pred = [i + 1 for i in range(1, n) if seq[i] >= delta_star]
        pk, wd = pk_wd(pred, gold, n)
        pk_v.append(pk); wd_v.append(wd); f1_v.append(boundary_set_f1(pred, gold))
    pk_m, wd_m, f1_m = np.mean(pk_v), np.mean(wd_v), np.mean(f1_v)
    return dict(pk=pk_m, wd=wd_m, f1=f1_m,
                score=0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m))


def best_score_delta_star(dialogs, deff_seqs):
    allv = np.array([d for seq in deff_seqs for d in seq[1:]])
    # best-Score δ* 탐색 — [DSTAR_LO, DSTAR_HI] 를 60점(해상도 ~0.01).
    # 범위는 인코더 적응형 wide band: degenerate δ* 는 Score 가 낮아
    # sweep 이 알아서 배제 (boundary 에 핀되지 않게 충분히 넓게).
    lo, hi = DSTAR_LO, DSTAR_HI
    best = (-1.0, 0.0)
    for ds in np.linspace(lo, hi, 60):
        sc = score_at(dialogs, deff_seqs, ds)["score"]
        if sc > best[0]:
            best = (sc, float(ds))
    return best[1]


def eval_latency(dialogs, embs, delta_star):
    """δ* 로 평가 + turn별 latency(ms) 측정 (turn idx≥1)."""
    pk_v, wd_v, f1_v, lat = [], [], [], []
    for (did, utts, gold), emb in zip(dialogs, embs):
        seg = HiOnTop(dim=emb.shape[1], delta_star=delta_star,
                     ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
        pred = []
        for i, s in enumerate(emb):
            t0 = time.perf_counter()
            _, isb = seg.assign(s.astype(np.float64))
            ms = (time.perf_counter() - t0) * 1000.0
            if i >= 1:
                lat.append(ms)
            if isb:
                pred.append(i + 1)
        n = len(utts)
        pk, wd = pk_wd(pred, gold, n)
        pk_v.append(pk); wd_v.append(wd); f1_v.append(boundary_set_f1(pred, gold))
    pk_m, wd_m, f1_m = np.mean(pk_v), np.mean(wd_v), np.mean(f1_v)
    return dict(pk=pk_m, wd=wd_m, f1=f1_m,
                score=0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m),
                lat=latency_stats(lat))


def main():
    global ENCODER
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=list(ENCODERS), default="mpnet")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    ENCODER = args.encoder
    name = args.name or (
        "2026-05-22_hiontop_baseline_table" if ENCODER == "mpnet"
        else f"2026-05-23_hiontop_table_{ENCODER}")
    out_dir = REPO / "outputs" / "experiments" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[encoder={ENCODER}  {ENCODERS[ENCODER]['model']}]", flush=True)
    results = {}

    for ds in ("tiage", "dialseg711", "superseg"):
        print(f"\n=== {ds} ===", flush=True)
        test = load_split(ds, "test")
        test_emb = encode_cached(ds, "test", test)
        test_deff = [delta_eff_seq(e) for e in test_emb]
        print(f"  test: {len(test)} dial", flush=True)

        # p80: label-free → test set 의 δ_eff 분포에서 (누수 없음)
        allv = np.array([d for seq in test_deff for d in seq[1:]])
        dstar_p80 = float(np.percentile(allv, 80))

        # best-Score: train split (tiage/superseg) / 없으면 test (dialseg711 ‡)
        train = load_split(ds, "train")
        if train:
            # 단일 scalar δ* 보정엔 600 dialog 면 충분 — 거대 train
            # (superseg 6947) 전수는 과함. seeded 서브샘플.
            n_full = len(train)
            if n_full > 600:
                idx = sorted(np.random.default_rng(0).permutation(n_full)[:600])
                train = [train[i] for i in idx]
            train_emb = encode_cached(ds, "train", train)
            train_deff = [delta_eff_seq(e) for e in train_emb]
            dstar_bs = best_score_delta_star(train, train_deff)
            bs_note = (f"train split ({len(train)}/{n_full} dial"
                       f"{' subsample' if n_full > 600 else ''})")
        else:
            dstar_bs = best_score_delta_star(test, test_deff)
            bs_note = "‡ train split 부재 → test 보정 (in-domain upper-bound)"
        print(f"  δ* p80={dstar_p80:.4f}  best-Score={dstar_bs:.4f} [{bs_note}]",
              flush=True)

        r_p80 = eval_latency(test, test_emb, dstar_p80)
        r_bs = eval_latency(test, test_emb, dstar_bs)
        results[ds] = dict(dstar_p80=dstar_p80, dstar_bs=dstar_bs,
                           bs_note=bs_note, p80=r_p80, bs=r_bs,
                           n_dial=len(test))
        print(f"  Ours(p80):       Pk={r_p80['pk']:.4f} WD={r_p80['wd']:.4f} "
              f"F1={r_p80['f1']:.4f} Score={r_p80['score']:.4f} "
              f"lat={r_p80['lat']['mean']:.4f}ms", flush=True)
        print(f"  Ours(best-Score):Pk={r_bs['pk']:.4f} WD={r_bs['wd']:.4f} "
              f"F1={r_bs['f1']:.4f} Score={r_bs['score']:.4f}", flush=True)

    # ---- REPORT ----
    L = [
        "# Hi-OnTop vs online 베이스라인 비교표 (Def-DTS 번들, 벤치별)",
        "",
        "전부 **online (past-only)**. metric = segeval Pk/WD + boundary-set F1, "
        "Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD). 데이터 = Def-DTS 번들 "
        "test set (베이스라인과 동일).",
        "",
        "- **Ours (p80)** : δ* = test δ_eff 80-percentile (label-free, 누수 "
        "없음). 베이스라인과 동일 데이터·무감독 → **메인 공정 비교 행**.",
        "- **Ours (best-Score)** : δ* = labeled split 에서 Score 최대화 "
        "(supervised). **참고: supervised ceiling** (Ours 만 train 튜닝 → "
        "베이스라인과 1:1 공정비교 아님).",
        f"- Hi-OnTop HP: m={M}, ρ={RHO}, a={A}, 인코더 {ENCODER} ({ENCODERS[ENCODER]['model']}).",
        f"- best-Score δ* 탐색 범위: [{DSTAR_LO}, {DSTAR_HI}] (사용자 지정).",
        "- 베이스라인 수치 = 각 baseline REPORT.md (TextTiling-streaming / "
        "GreedySeg-online-delay2 / GraphSeg-window-d) 그대로 인용.",
        "- CSM-style : 별도 실행 대기 (csm_online.py, 본 표에 추후 추가).",
        "",
    ]
    for ds in ("tiage", "dialseg711", "superseg"):
        r = results[ds]
        L += [
            f"## {ds}  (test {r['n_dial']} dialog)",
            "",
            f"δ* — p80={r['dstar_p80']:.4f} · best-Score={r['dstar_bs']:.4f} "
            f"({r['bs_note']})",
            "",
            "| Method (전부 Online) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
            "Avg turn latency (ms) ↓ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, (pk, wd, f1, sc, lat) in BASELINES[ds].items():
            L.append(f"| {name} | {pk:.4f} | {wd:.4f} | {f1:.4f} | {sc:.4f} "
                     f"| {lat:.3f} |")
        L.append("| CSM-style | _대기_ | _대기_ | _대기_ | _대기_ | _대기_ |")
        p, b = r["p80"], r["bs"]
        L.append(f"| **Ours (p80)** | {p['pk']:.4f} | {p['wd']:.4f} | "
                 f"{p['f1']:.4f} | **{p['score']:.4f}** | "
                 f"{p['lat']['mean']:.4f} |")
        L.append(f"| Ours (best-Score)‡ref | {b['pk']:.4f} | {b['wd']:.4f} | "
                 f"{b['f1']:.4f} | {b['score']:.4f} | {b['lat']['mean']:.4f} |")
        L.append("")
    L += [
        "## 한계 / 검증 미해결",
        "- 데이터 = Def-DTS 번들. Hi-OnTop 의 superdialseg_data 보고 수치"
        "(0.4675 등)와는 dialseg711/superseg turn 수가 달라 직접 일치 안 함.",
        "- dialseg711 은 번들에 train split 부재 → best-Score δ* 를 test 에서 "
        "보정(‡). 이 행은 in-domain upper-bound 로, 공정 비교는 p80 행.",
        "- best-Score 행 전반: Ours 만 라벨로 튜닝 → 베이스라인과 비대칭. "
        "메인 비교는 p80 행으로 읽을 것.",
        "- CSM-style 행 미측정 (csm_online.py 별도 실행 예정).",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out_dir/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
