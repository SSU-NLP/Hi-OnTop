#!/usr/bin/env python3
"""DTS 무회귀 검증: geometry-backchannel 병합(m=0) 이 DTS 3벤치(텍스트 대화)에서
base Hi-OnTop 성능을 깨지 않는지. 가설: content-rich 발화라 외톨이 거의 없음 →
flag≈0 → 병합 no-op → base 와 동일.

판정: flag 비율 ≈ 0 이고 base vs merge 의 Pk/WD/F1/Score Δ ≈ 0 이면 무회귀.
δ* = 양쪽 동일 방법(best-Score oracle sweep) 으로 병합 효과만 격리.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import (load_dialogs, delta_eff_seq, score_set,
                                    best_score_dstar)  # noqa: E402

ENC = dict(model="sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
           file_name="onnx/model_quint8_avx2.onnx")


def geom_flag(emb, margin=0.0):
    """turn i 를 빼면 양옆이 더 붙는 외톨이면 backchannel. 양끝 False."""
    e = [v / (np.linalg.norm(v) + 1e-12) for v in emb]
    flag = [False] * len(e)
    for i in range(1, len(e) - 1):
        sp, sn, sb = float(e[i-1] @ e[i]), float(e[i] @ e[i+1]), float(e[i-1] @ e[i+1])
        if sb > sp + margin and sb > sn + margin:
            flag[i] = True
    return flag


def merge_dialog(utts, yt, flag):
    groups, cur = [], None
    for i in range(len(utts)):
        if flag[i] and cur is not None:
            cur.append(i)
        else:
            cur = [i]; groups.append(cur)
    m_utts = [" ".join(utts[k] for k in g) for g in groups]
    m_yt = [1 if any(yt[k] == 1 for k in g) else 0 for g in groups]
    if m_yt:
        m_yt[-1] = 0
    return m_utts, m_yt


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENC["model"], backend="onnx",
                              model_kwargs={"provider": "CPUExecutionProvider",
                                            "file_name": ENC["file_name"]})

    def batch_embed(list_of_texts):
        """각 dialog 의 텍스트 리스트들을 한 번에 인코딩 후 dialog 별로 분할."""
        flat, offs = [], [0]
        for ts in list_of_texts:
            flat.extend(ts); offs.append(len(flat))
        if not flat:
            return [np.zeros((0, 384)) for _ in list_of_texts]
        allemb = np.asarray(enc.encode(flat, normalize_embeddings=True, batch_size=128,
                                       show_progress_bar=False), dtype=np.float64)
        return [allemb[offs[i]:offs[i+1]] for i in range(len(list_of_texts))]

    print("DTS 무회귀 검증 — geometry-backchannel 병합(m=0) vs base Hi-OnTop", flush=True)
    print("  encoder=MiniLM-int8, split=test, δ*=best-Score oracle (양쪽 동일)\n", flush=True)
    hdr = (f"{'벤치':10} {'n_dlg':>5} {'flag%':>6} {'dlg_flagged%':>11} | "
           f"{'base Sc':>8} {'merge Sc':>8} {'ΔSc':>7} | {'base F1':>7} {'merge F1':>8}")
    print(hdr); print("-" * len(hdr), flush=True)
    for ds in ("tiage", "dialseg711", "superseg"):
        dialogs = load_dialogs(ds, "test")
        base_emb = batch_embed([list(utts) for utts, _ in dialogs])
        # --- flag rate (결정적 수치; 인코딩만 필요) ---
        n_flag = n_turn = n_dlg_flag = 0
        flags = []
        for (utts, _), e in zip(dialogs, base_emb):
            f = geom_flag(e, margin=0.0)
            flags.append(f); n_flag += sum(f); n_turn += len(f)
            n_dlg_flag += 1 if any(f) else 0
        flag_pct = 100.0 * n_flag / max(1, n_turn)
        dlg_pct = 100.0 * n_dlg_flag / max(1, len(dialogs))
        # --- 점수: base vs merge (병합 후 <3턴 dialog 는 양쪽에서 제외해 공정 비교) ---
        b_dialogs, b_embs, m_dialogs, m_embs = [], [], [], []
        reenc_texts, reenc_slots = [], []          # 병합 발생 dialog 만 재인코딩
        for (utts, yt), e, f in zip(dialogs, base_emb, flags):
            mu, my = merge_dialog(utts, yt, f)
            if len(mu) < 3:                        # Pk/WD 안전: 너무 짧으면 양쪽 제외
                continue
            b_dialogs.append((utts, yt)); b_embs.append(e)
            m_dialogs.append((mu, my))
            if sum(f):
                m_embs.append(None); reenc_texts.append(mu); reenc_slots.append(len(m_embs) - 1)
            else:
                m_embs.append(e)
        for slot, em in zip(reenc_slots, batch_embed(reenc_texts) if reenc_texts else []):
            m_embs[slot] = em
        b_deffs = [delta_eff_seq(e) for e in b_embs]
        m_deffs = [delta_eff_seq(e) for e in m_embs]
        base = score_set(b_dialogs, b_deffs, best_score_dstar(b_dialogs, b_deffs))
        merge = score_set(m_dialogs, m_deffs, best_score_dstar(m_dialogs, m_deffs))
        print(f"{ds:10} {len(dialogs):>5} {flag_pct:>5.1f}% {dlg_pct:>10.1f}% | "
              f"{base['score']:>8.4f} {merge['score']:>8.4f} "
              f"{merge['score']-base['score']:>+7.4f} | "
              f"{base['f1']:>7.4f} {merge['f1']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
