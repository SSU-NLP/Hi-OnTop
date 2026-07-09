#!/usr/bin/env python3
"""인코더 비교 — mpnet / MiniLM / MiniLM-int8, superdialseg_data 공식 셋업.

데이터 = superdialseg_data (headline 0.4631 그 셋업). metric = 공식
SuperDialseg (nltk Pk/WD k=auto + sklearn binary F1, Score=0.5F1+
0.25(1-Pk)+0.25(1-WD)).

δ* = 인코더별, **train split 보정** (p80 + best-Score 둘 다):
  tiage→tiage-train, superseg→superseg-train(서브샘플),
  dialseg711→train split 부재라 test 70:30 분할(70%를 튜닝).
"""

from __future__ import annotations

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
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

SDS = REPO / "benchmarks" / "superdialseg_data"
CACHE = REPO / "outputs" / "runs" / "_misc"
M, RHO, A = 2, 0.7, 0.5
TRAIN_CAP = 400          # δ* 보정용 train 서브샘플 (단일 scalar → 충분)
DSTAR_GRID = np.linspace(0.35, 0.95, 60)  # best-Score sweep (인코더 적응형)

ENCODERS = {
    "mpnet": {"model": "sentence-transformers/multi-qa-mpnet-base-dot-v1"},
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "minilm-int8": {"model": "sentence-transformers/all-MiniLM-L6-v2",
                    "backend": "onnx",
                    "file_name": "onnx/model_quint8_avx2.onnx"},
}
# mpnet 은 기존 superdialseg 캐시 재사용 (load_dialogs 정렬 동일).
MPNET_REUSE = {
    ("tiage", "test"): "sds_emb_tiage_test.pkl",
    ("tiage", "train"):
        "sds_emb_tiage_train_sentence-transformers_multi-qa-mpnet-base-dot-v1.pkl",
    ("dialseg711", "test"): "sds_emb_dialseg711_test.pkl",
    ("superseg", "test"): "sds_emb_superseg_test.pkl",
}


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


def official_pk_wd(yt, yp):
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


_ENC = None
_ENC_NAME = None


def _encoder(enc):
    global _ENC, _ENC_NAME
    if _ENC is None or _ENC_NAME != enc:
        from sentence_transformers import SentenceTransformer
        import torch  # onnxruntime io_binding 가 torch.int4 참조 — 없는 torch 빌드용 shim
        if not hasattr(torch, "int4"):  # int8(quint8) 모델이라 int4 경로는 실제 미사용, 참조만 충족
            torch.int4 = torch.uint8
        cfg = ENCODERS[enc]
        _ENC_NAME = enc
        if cfg.get("backend") == "onnx":
            _ENC = SentenceTransformer(
                cfg["model"], backend="onnx",
                model_kwargs={"provider": "CPUExecutionProvider",
                              "file_name": cfg["file_name"]})
        else:
            _ENC = SentenceTransformer(cfg["model"], device="cpu")
    return _ENC


def encode(enc, ds, split, dialogs):
    if enc == "mpnet" and (ds, split) in MPNET_REUSE:
        p = CACHE / MPNET_REUSE[(ds, split)]
        if p.exists():
            with open(p, "rb") as fh:
                cached = pickle.load(fh)
            if len(cached) == len(dialogs):
                return [np.asarray(e) for e in cached]
    cp = CACHE / f"enccmp_{ds}_{split}_{enc}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    flat, lens = [], []
    for utts, _ in dialogs:
        flat.extend(utts)
        lens.append(len(utts))
    allv = np.asarray(_encoder(enc).encode(
        flat, normalize_embeddings=True, show_progress_bar=False))
    embs, i = [], 0
    for L in lens:
        embs.append(allv[i:i + L]); i += L
    with open(cp, "wb") as fh:
        pickle.dump(embs, fh)
    return embs


def delta_eff_seq(emb):
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def boundaries(seq, dstar):
    return [1 if seq[i] >= dstar else 0 for i in range(1, len(seq))] + [0]


def score_set(dialogs, deffs, dstar):
    # 공식 SuperDialseg 규약: F1 도 Pk/WD 와 같이 **대화별 계산 후 평균** (per-dialogue).
    # (이전엔 F1 만 corpus 풀링이라 비공식이었음 — HANDOFF_04. 풀링/per-dialogue 는
    # dialseg711 에선 거의 동일하나 짧은-대화 데이터셋에서 갈릴 수 있어 공식으로 통일.)
    pks, wds, f1s = [], [], []
    for (utts, yt), seq in zip(dialogs, deffs):
        yp = boundaries(seq, dstar)
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd)
        f1s.append(float(f1_score(yt, yp, zero_division=0)))
    pk_m, wd_m, f1_m = float(np.mean(pks)), float(np.mean(wds)), float(np.mean(f1s))
    return dict(pk=pk_m, wd=wd_m, f1=f1_m,
                score=0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m))


def best_score_dstar(dialogs, deffs):
    """train split 에서 Score(0.5F1+0.25(1−Pk)+0.25(1−WD)) 최대화하는 δ*."""
    best = (-1.0, 0.0)
    for d in DSTAR_GRID:
        sc = score_set(dialogs, deffs, d)["score"]
        if sc > best[0]:
            best = (sc, float(d))
    return best[1]


def main():
    rng = np.random.default_rng(0)
    rows = {}   # enc -> ds -> {...}
    for enc in ENCODERS:
        print(f"\n########## encoder = {enc} ##########", flush=True)
        rows[enc] = {}
        for ds in ("tiage", "dialseg711", "superseg"):
            # ---- calib / test split ----
            if ds == "dialseg711":
                dia = load_dialogs(ds, "test")
                emb = encode(enc, ds, "test", dia)
                idx = rng.permutation(len(dia))
                cut = int(round(len(idx) * 0.70))  # 70% 튜닝 / 30% test
                ci, ti = sorted(idx[:cut]), sorted(idx[cut:])
                calib = ([dia[i] for i in ci], [emb[i] for i in ci])
                test = ([dia[i] for i in ti], [emb[i] for i in ti])
                calib_note = f"test 70:30 split (calib {len(ci)} / test {len(ti)})"
            else:
                tr = load_dialogs(ds, "train")
                if len(tr) > TRAIN_CAP:
                    sub = sorted(rng.permutation(len(tr))[:TRAIN_CAP])
                    tr = [tr[i] for i in sub]
                tr_emb = encode(enc, ds, "train", tr)
                te = load_dialogs(ds, "test")
                te_emb = encode(enc, ds, "test", te)
                calib = (tr, tr_emb)
                test = (te, te_emb)
                calib_note = f"train split (calib {len(tr)} / test {len(te)})"

            calib_deff = [delta_eff_seq(e) for e in calib[1]]
            test_deff = [delta_eff_seq(e) for e in test[1]]
            allv = np.array([d for s in calib_deff for d in s[1:]])
            dstar_p80 = float(np.percentile(allv, 80))
            dstar_bs = best_score_dstar(calib[0], calib_deff)

            r_p80 = score_set(test[0], test_deff, dstar_p80)
            r_bs = score_set(test[0], test_deff, dstar_bs)
            rows[enc][ds] = dict(p80=dstar_p80, bs=dstar_bs,
                                 r_p80=r_p80, r_bs=r_bs, note=calib_note)
            print(f"  [{ds}] {calib_note}", flush=True)
            print(f"    δ*  p80={dstar_p80:.4f}  best-Score={dstar_bs:.4f}",
                  flush=True)
            print(f"    p80    : Pk={r_p80['pk']:.4f} WD={r_p80['wd']:.4f} "
                  f"F1={r_p80['f1']:.4f} Score={r_p80['score']:.4f}", flush=True)
            print(f"    best-Score: Pk={r_bs['pk']:.4f} WD={r_bs['wd']:.4f} "
                  f"F1={r_bs['f1']:.4f} Score={r_bs['score']:.4f}", flush=True)

    # ---- REPORT ----
    out = REPO / "outputs" / "experiments" / "2026-05-23_encoder_comparison"
    out.mkdir(parents=True, exist_ok=True)
    L = [
        "# 인코더 비교 — mpnet / MiniLM / MiniLM-int8 (superdialseg_data)",
        "",
        "데이터 = superdialseg_data, 공식 SuperDialseg metric "
        "(Score=0.5F1+0.25(1−Pk)+0.25(1−WD)).",
        f"δ* = 인코더별 train split 보정 (서브샘플 ≤{TRAIN_CAP}). "
        "dialseg711 은 train split 부재 → test 70:30 (70%를 보정).",
        "Hi-OnTop HP: m=2, ρ=0.7, a=0.5.",
        "",
        "## Ours (p80)  — δ* = train δ_eff 80-percentile",
        "",
        "| 인코더 | tiage | dialseg711 | superseg | **mean-3** |",
        "|---|---:|---:|---:|---:|",
    ]
    for enc in ENCODERS:
        sc = [rows[enc][d]["r_p80"]["score"]
              for d in ("tiage", "dialseg711", "superseg")]
        L.append(f"| {enc} | {sc[0]:.4f} | {sc[1]:.4f} | {sc[2]:.4f} | "
                 f"**{np.mean(sc):.4f}** |")
    L += ["", "## Ours (best-Score)  — δ* = train split F1 최대화", "",
          "| 인코더 | tiage | dialseg711 | superseg | **mean-3** |",
          "|---|---:|---:|---:|---:|"]
    for enc in ENCODERS:
        sc = [rows[enc][d]["r_bs"]["score"]
              for d in ("tiage", "dialseg711", "superseg")]
        L.append(f"| {enc} | {sc[0]:.4f} | {sc[1]:.4f} | {sc[2]:.4f} | "
                 f"**{np.mean(sc):.4f}** |")
    L += ["", "## δ* 값 (인코더 × 벤치)", "",
          "| 인코더 | 벤치 | δ* p80 | δ* best-Score | calib |",
          "|---|---|---:|---:|---|"]
    for enc in ENCODERS:
        for d in ("tiage", "dialseg711", "superseg"):
            r = rows[enc][d]
            L.append(f"| {enc} | {d} | {r['p80']:.4f} | {r['bs']:.4f} | "
                     f"{r['note']} |")
    L += ["", "## 한계",
          "- δ* 는 인코더별로 다름 (mpnet ~0.56, MiniLM ~0.77) — 인코더마다 "
          "재보정 필수.",
          "- dialseg711 은 superdialseg_data 에 train split 부재 → test "
          "70:30 seeded 분할 (no leakage).",
          "- 베이스라인(TextTiling/GreedySeg/GraphSeg) 비교는 별도 — 그들도 "
          "superdialseg_data 에서 재측정 필요."]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nDONE → {out/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
