#!/usr/bin/env python3
"""SuperDialseg calib N=2000 — MiniLM-fp32 + MiniLM-int8 (MPNet 자매판).

`diag_superseg_calib_2000.py` 의 MPNet 버전을 MiniLM 두 인코더로 확장.
세 인코더 모두 같은 결론 (N=400→2000 변화 < noise) 이 나오면 "calib 크기가
superseg 천장의 원인 아님" 이 단일 인코더 우연이 아님이 확정.

인코더별 처리:
- minilm   : SentenceTransformer('all-MiniLM-L6-v2', device=cpu)
- minilm-int8 : ONNX backend, model_quint8_avx2.onnx

기존 train 400-dialog 캐시 (enccmp_superseg_train_{enc}.pkl) 재사용 + 추가 1600
인코딩 → N∈{400,1000,2000} 측정. 결과는 같은 REPORT.md 에 인코더별 append.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hi_ontop.hi_ontop import HiOnTop  # noqa: E402
from run_encoder_comparison import (  # noqa: E402
    DSTAR_GRID, M, RHO, A, ENCODERS, load_dialogs,
    score_set, best_score_dstar,
)

CACHE = REPO / "outputs" / "runs" / "_misc"
CALIB_NS = [400, 1000, 2000]


def delta_eff_seq(emb):
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def load_encoder(enc: str):
    from sentence_transformers import SentenceTransformer
    cfg = ENCODERS[enc]
    if cfg.get("backend") == "onnx":
        return SentenceTransformer(
            cfg["model"], backend="onnx",
            model_kwargs={"provider": "CPUExecutionProvider",
                          "file_name": cfg["file_name"]})
    return SentenceTransformer(cfg["model"], device="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    choices=("minilm", "minilm-int8"))
    args = ap.parse_args()
    enc = args.encoder

    existing_cache = CACHE / f"enccmp_superseg_train_{enc}.pkl"
    extended_cache = CACHE / f"enccmp_superseg_train_{enc}_n2000.pkl"
    test_cache = CACHE / f"enccmp_superseg_test_{enc}.pkl"

    print(f"\n########## encoder = {enc} ##########", flush=True)

    # ---- 인덱스 (mpnet 패스 RNG 재현; encoder 마다 RNG 상태 다름) ----
    # run_encoder_comparison.main() 에서 ENCODERS 순서로 루프 →
    # mpnet → minilm → minilm-int8. rng 는 하나만 쓰고 인코더마다 갱신됨.
    # mpnet 패스: perm(711) + perm(6948)[:400]  ← rng 2 step
    # minilm 패스: perm(711) + perm(6948)[:400]  ← 추가 2 step
    # minilm-int8 패스: perm(711) + perm(6948)[:400]  ← 추가 2 step
    rng = np.random.default_rng(0)
    order = list(ENCODERS.keys())  # ('mpnet','minilm','minilm-int8')
    for e in order:
        _ = rng.permutation(711)        # dialseg711
        perm = rng.permutation(6948)    # superseg
        if e == enc:
            break
    idx_400 = list(map(int, perm[:400]))
    idx_2000 = list(map(int, perm[:2000]))

    tr_dia_all = load_dialogs("superseg", "train")

    # ---- 기존 400 cache 매핑 검증 ----
    with open(existing_cache, "rb") as fh:
        emb_400_raw = pickle.load(fh)
    sorted_400 = sorted(idx_400)
    for i, idx in enumerate(sorted_400):
        n_utt = len(tr_dia_all[idx][0])
        n_emb = emb_400_raw[i].shape[0] if hasattr(emb_400_raw[i], "shape") \
            else len(emb_400_raw[i])
        assert n_utt == n_emb, \
            f"[{enc}] 400-cache mismatch at {i} (idx {idx}): {n_utt} vs {n_emb}"
    print(f"[{enc}] 기존 400 cache 매핑 OK", flush=True)
    sorted_400_to_emb = {sorted_400[i]: np.asarray(emb_400_raw[i])
                         for i in range(400)}

    # ---- 추가 1600 인코딩 ----
    extra_idx = sorted(set(idx_2000) - set(idx_400))
    if extended_cache.exists():
        with open(extended_cache, "rb") as fh:
            sorted_2000_to_emb = pickle.load(fh)
        print(f"[{enc}] extended cache hit ({len(sorted_2000_to_emb)})",
              flush=True)
    else:
        print(f"[{enc}] encoding {len(extra_idx)} 추가 dialog ...", flush=True)
        model = load_encoder(enc)
        flat, lens = [], []
        for idx in extra_idx:
            utts = tr_dia_all[idx][0]
            flat.extend(utts)
            lens.append(len(utts))
        t0 = time.perf_counter()
        allv = model.encode(flat, normalize_embeddings=True,
                            show_progress_bar=True, batch_size=64)
        dt = time.perf_counter() - t0
        print(f"[{enc}] {len(flat)} utts in {dt/60:.2f} min "
              f"({1000*dt/len(flat):.1f} ms/utt)", flush=True)
        extra_embs = {}
        cur = 0
        for idx, L in zip(extra_idx, lens):
            extra_embs[idx] = np.asarray(allv[cur:cur + L])
            cur += L
        sorted_2000_to_emb = {**sorted_400_to_emb, **extra_embs}
        with open(extended_cache, "wb") as fh:
            pickle.dump(sorted_2000_to_emb, fh)
        print(f"[{enc}] save → {extended_cache.name}", flush=True)

    # ---- test δ_eff ----
    te_dia = load_dialogs("superseg", "test")
    with open(test_cache, "rb") as fh:
        te_emb = [np.asarray(e) for e in pickle.load(fh)]
    print(f"[{enc}] test δ_eff (1322 dialog) ...", flush=True)
    te_deff = [delta_eff_seq(e) for e in te_emb]

    # ---- calib pool δ_eff ----
    sorted_2000 = sorted(idx_2000)
    pool_dia = [tr_dia_all[i] for i in sorted_2000]
    pool_emb = [sorted_2000_to_emb[i] for i in sorted_2000]
    for i, idx in enumerate(sorted_2000):
        assert len(tr_dia_all[idx][0]) == pool_emb[i].shape[0]
    print(f"[{enc}] pool δ_eff (2000 dialog) ...", flush=True)
    pool_deff = [delta_eff_seq(e) for e in pool_emb]

    # ---- N sweep ----
    pool_idx_to_pos = {idx: pos for pos, idx in enumerate(sorted_2000)}
    rows = []
    for N in CALIB_NS:
        sub_sorted = sorted(map(int, perm[:N]))
        positions = [pool_idx_to_pos[i] for i in sub_sorted]
        ce_dia = [pool_dia[p] for p in positions]
        ce_deff = [pool_deff[p] for p in positions]
        allv = np.array([d for s in ce_deff for d in s[1:]])
        dstar_p80 = float(np.percentile(allv, 80))
        dstar_bs = best_score_dstar(ce_dia, ce_deff)
        r_p80 = score_set(te_dia, te_deff, dstar_p80)
        r_bs = score_set(te_dia, te_deff, dstar_bs)
        rows.append((N, dstar_p80, dstar_bs, r_p80["score"], r_bs["score"]))
        print(f"[{enc}] N={N:5d}  δ*_p80={dstar_p80:.4f} δ*_bs={dstar_bs:.4f}  "
              f"Score p80={r_p80['score']:.4f}  best={r_bs['score']:.4f}",
              flush=True)

    # ---- REPORT append ----
    rep = (REPO / "outputs" / "experiments"
           / "2026-05-23_superseg_calib_size_check" / "REPORT.md")
    txt = rep.read_text() if rep.exists() else ""
    header = f"### 확장 N=2000 — {enc.upper()}"
    if header in txt:
        print(f"[{enc}] REPORT 에 이미 {enc} 섹션 있음 — skip append",
              flush=True)
    else:
        append = ["", "", header, "",
                  "| calib N | δ*_p80 | δ*_best | Score (p80) | "
                  "Score (best) |",
                  "|---:|---:|---:|---:|---:|"]
        for N, p80, bs, sp, sb in rows:
            append.append(f"| {N} | {p80:.4f} | {bs:.4f} | "
                          f"{sp:.4f} | {sb:.4f} |")
        d400, d2000 = rows[0], rows[-1]
        dsp = d2000[3] - d400[3]; dsb = d2000[4] - d400[4]
        append.append("")
        append.append(f"**N=400 → N=2000 변화량**: Score(p80) {dsp:+.4f}, "
                      f"Score(best) {dsb:+.4f}.")
        append.append("")
        rep.write_text(txt + "\n".join(append) + "\n")
        print(f"\n[{enc}] REPORT append → {rep}", flush=True)


if __name__ == "__main__":
    main()
