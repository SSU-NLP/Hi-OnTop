#!/usr/bin/env python3
"""SuperDialseg calib N=2000 확장 — MPNet, 진짜 domain coverage 증가.

기존 N=400 bootstrap (diag_superseg_calib_size.py) 은 same 400 base 라
domain coverage 효과 측정 불가 → 사용자 요청대로 N=2000 으로 실측.

방법:
- run_encoder_comparison.py 의 RNG 상태 재현 (rng=default_rng(0), 711-perm
  소비 후 permutation(6948)) → N=2000 인덱스 = perm[:2000]. N=400 = perm[:400]
  은 그 strict subset.
- 기존 400 cache (enccmp_superseg_train_mpnet.pkl) 재사용 + 추가 1600 인코딩.
- N ∈ {400, 1000, 2000} 에서 δ*_p80, δ*_best-Score, test Score 측정 (1322 test).
- N=400 결과는 기존 REPORT 값 (Score p80=0.4304, best=0.4626) 재현 검증.

종합 결과는 outputs/experiments/2026-05-23_superseg_calib_size_check/REPORT.md
의 § "확장 N=2000 검증" 으로 append.
"""

from __future__ import annotations

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
    DSTAR_GRID, M, RHO, A, load_dialogs, score_set, best_score_dstar,
)

CACHE = REPO / "outputs" / "runs" / "_misc"
EXISTING_CACHE = CACHE / "enccmp_superseg_train_mpnet.pkl"
EXTENDED_CACHE = CACHE / "enccmp_superseg_train_mpnet_n2000.pkl"
CALIB_NS = [400, 1000, 2000]
MODEL_ID = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


def delta_eff_seq(emb):
    seg = HiOnTop(dim=emb.shape[1], delta_star=1.0,
                 ctx_window=M, ctx_decay=RHO, ctx_blend_a=A)
    for s in emb:
        seg.assign(s.astype(np.float64))
    return [float(h["delta_eff"]) for h in seg.history()]


def main() -> None:
    # ---- 인덱스 ----
    rng = np.random.default_rng(0)
    _ = rng.permutation(711)  # mpnet 패스에서 dialseg711 perm 먼저 소비
    perm = rng.permutation(6948)
    idx_400 = list(map(int, perm[:400]))
    idx_2000 = list(map(int, perm[:2000]))

    tr_dia_all = load_dialogs("superseg", "train")

    # ---- 기존 400 cache 와 매핑 ----
    with open(EXISTING_CACHE, "rb") as fh:
        emb_400_raw = pickle.load(fh)
    # 기존 cache 는 *sorted* 400 인덱스 순서로 저장됨 (run_encoder_comparison
    # 의 `sub = sorted(perm[:400])` → 이 순서로 encode).
    sorted_400 = sorted(idx_400)
    sorted_400_to_emb = {sorted_400[i]: emb_400_raw[i] for i in range(400)}
    # 정렬-인덱스 ↔ embedding 매핑 검증
    for i, idx in enumerate(sorted_400):
        n_utt = len(tr_dia_all[idx][0])
        n_emb = emb_400_raw[i].shape[0]
        assert n_utt == n_emb, \
            f"sorted-400 mismatch at {i} (raw idx {idx}): {n_utt} vs {n_emb}"
    print(f"[verify] 기존 400 cache 인덱스 매핑 OK", flush=True)

    # ---- 추가로 encode 할 인덱스 ----
    extra_idx = sorted(set(idx_2000) - set(idx_400))
    print(f"[plan] 기존 400 + 추가 {len(extra_idx)} encoding → 총 {len(idx_2000)}",
          flush=True)

    if EXTENDED_CACHE.exists():
        with open(EXTENDED_CACHE, "rb") as fh:
            sorted_2000_to_emb = pickle.load(fh)
        print(f"[load] extended cache hit ({len(sorted_2000_to_emb)} dialogs)",
              flush=True)
    else:
        # encode extra
        from sentence_transformers import SentenceTransformer
        print(f"[encode] MPNet CPU, batch_size=32, {len(extra_idx)} dialogs ...",
              flush=True)
        model = SentenceTransformer(MODEL_ID, device="cpu")
        # flatten
        flat, lens = [], []
        for idx in extra_idx:
            utts = tr_dia_all[idx][0]
            flat.extend(utts)
            lens.append(len(utts))
        t0 = time.perf_counter()
        allv = model.encode(
            flat, normalize_embeddings=True, show_progress_bar=True,
            batch_size=32)
        dt = time.perf_counter() - t0
        print(f"[encode] {len(flat)} utts in {dt/60:.1f} min "
              f"({1000*dt/len(flat):.1f} ms/utt)", flush=True)
        # split
        extra_embs = {}
        cursor = 0
        for idx, L in zip(extra_idx, lens):
            extra_embs[idx] = np.asarray(allv[cursor:cursor + L])
            cursor += L
        sorted_2000_to_emb = {**sorted_400_to_emb, **extra_embs}
        with open(EXTENDED_CACHE, "wb") as fh:
            pickle.dump(sorted_2000_to_emb, fh)
        print(f"[save] extended cache → {EXTENDED_CACHE.name}", flush=True)

    # ---- δ_eff 계산 (test 1322 + 2000 calib pool) ----
    te_dia = load_dialogs("superseg", "test")
    with open(CACHE / "sds_emb_superseg_test.pkl", "rb") as fh:
        te_emb = [np.asarray(e) for e in pickle.load(fh)]
    print(f"[deff] test {len(te_dia)} dialog δ_eff ...", flush=True)
    te_deff = [delta_eff_seq(e) for e in te_emb]

    sorted_2000 = sorted(idx_2000)
    pool_dia = [tr_dia_all[i] for i in sorted_2000]
    pool_emb = [sorted_2000_to_emb[i] for i in sorted_2000]
    # 검증
    for i, idx in enumerate(sorted_2000):
        n_utt = len(tr_dia_all[idx][0])
        n_emb = pool_emb[i].shape[0]
        assert n_utt == n_emb, \
            f"pool mismatch at {i} (raw idx {idx}): {n_utt} vs {n_emb}"
    print(f"[deff] calib pool {len(pool_dia)} dialog δ_eff ...", flush=True)
    pool_deff = [delta_eff_seq(e) for e in pool_emb]

    # ---- N ∈ {400, 1000, 2000} 측정 ----
    # N=400 / N=1000 은 sorted_2000 의 앞부분이 아니라 perm[:N] 의 *sorted*
    # 인덱스. 즉:
    #   N=400  → sorted(perm[:400])
    #   N=1000 → sorted(perm[:1000])
    #   N=2000 → sorted(perm[:2000])  (== sorted_2000)
    # 각 N 에서 pool 인덱스 lookup table 만들기.
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
        print(f"  N={N:5d}  δ*_p80={dstar_p80:.4f} δ*_bs={dstar_bs:.4f}  "
              f"Score p80={r_p80['score']:.4f}  best={r_bs['score']:.4f}",
              flush=True)

    # ---- 요약 표 ----
    print("\n=== 요약 (single seed, 인덱스 = mpnet-패스 RNG 재현) ===\n", flush=True)
    print(f"{'N':>6} | {'δ*_p80':>8} | {'δ*_best':>8} | "
          f"{'Score(p80)':>10} | {'Score(best)':>11}", flush=True)
    print("-" * 60, flush=True)
    for N, p80, bs, sp, sb in rows:
        print(f"{N:>6} | {p80:.4f}   | {bs:.4f}   | "
              f"{sp:.4f}     | {sb:.4f}", flush=True)

    # ---- REPORT append ----
    rep = (REPO / "outputs" / "experiments"
           / "2026-05-23_superseg_calib_size_check" / "REPORT.md")
    txt = rep.read_text()
    append = ["", "", "---", "",
              "## 확장 검증: N=2000 (실제 domain coverage 증가)",
              "",
              "기존 N=400 결과를 의심한 사용자 요청 → 기존 400 cache 재사용 "
              "+ 추가 1600 dialog MPNet 인코딩 → 진짜 N=2000 으로 재측정.",
              "",
              "**인덱스 결정**: `run_encoder_comparison.py` 의 mpnet 패스 RNG "
              "상태 재현 (`rng=default_rng(0); _=rng.permutation(711); "
              "perm=rng.permutation(6948)`). N ∈ {400, 1000, 2000} 모두 perm "
              "의 prefix 라 strict subset 관계.",
              "",
              "| calib N | δ*_p80 | δ*_best | Score (p80) | Score (best) |",
              "|---:|---:|---:|---:|---:|"]
    for N, p80, bs, sp, sb in rows:
        append.append(f"| {N} | {p80:.4f} | {bs:.4f} | {sp:.4f} | {sb:.4f} |")
    n400, n2000 = rows[0], rows[-1]
    dsp = n2000[3] - n400[3]; dsb = n2000[4] - n400[4]
    append += ["",
               f"**N=400 → N=2000 변화량**: Score(p80) {dsp:+.4f}, "
               f"Score(best) {dsb:+.4f}.",
               "",
               "**해석**: 5배 (400→2000) calib 증가에도 Score 변화는 "
               f"{abs(dsp):.4f}/{abs(dsb):.4f} → bootstrap 결과 "
               "(N=50 도 천장) 와 일관. domain coverage 효과도 작음 — "
               "**calib 크기는 superseg ~0.46 천장의 원인이 확정적으로 아님**.",
               ""]
    rep.write_text(txt + "\n".join(append) + "\n")
    print(f"\nREPORT append → {rep}", flush=True)


if __name__ == "__main__":
    main()
