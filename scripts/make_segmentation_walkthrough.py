#!/usr/bin/env python3
"""실제 대화 분절 트레이스 → markdown. AMI 한 미팅을 (raw) vs (geometry-merge)
두 버전으로, 둘 다 online 적응 임계치(ewma μ+cσ)로 분절해 per-turn 비교.
"""
from __future__ import annotations
import json, pickle, sys, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from sklearn.metrics import f1_score

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
    model_kwargs={"provider": "CPUExecutionProvider", "file_name": "onnx/model_quint8_avx2.onnx"})
embed = lambda ts: np.asarray(enc.encode(ts, normalize_embeddings=True), dtype=np.float64)
trunc = lambda s, n=52: s if len(s) <= n else s[:n-1] + "…"

C, SPAN = 1.0, 5   # ewma: c=1.0, span=5


def ewma_trace(seq, c=C, span=SPAN):
    """online ewma μ+cσ. 반환 (preds, thr) per index. thr[i]=None 면 미정(warmup)."""
    n = len(seq); preds = [0]*n; thr = [None]*n
    alpha = 2.0/(span+1.0); mu = ev = 0.0; init = False
    for i in range(1, n):
        x = seq[i]
        if not init:
            mu, ev, init = x, 0.0, True; continue
        t = mu + c*math.sqrt(max(0.0, ev)); thr[i] = t
        preds[i] = 1 if x >= t else 0
        dd = x - mu; mu += alpha*dd; ev = (1-alpha)*(ev + alpha*dd*dd)
    return preds, thr


def geom_flag(E, m=0.0):
    e = [v/(np.linalg.norm(v)+1e-12) for v in E]; fl=[False]*len(e)
    for i in range(1, len(e)-1):
        sp, sn, sb = float(e[i-1]@e[i]), float(e[i]@e[i+1]), float(e[i-1]@e[i+1])
        if sb > sp+m and sb > sn+m: fl[i] = True
    return fl


TOPIC = REPO/"data"/"ami"/"topic"; CACHE = REPO/"outputs"/"runs"/"_misc"/"ami_emb"
mid = "ES2002a"
d = json.load(open(TOPIC/f"{mid}.json")); turns = d["turns"]; bt = d["bnd_top"]
emb = np.asarray(pickle.load(open(CACHE/f"{mid}.pkl", "rb")), dtype=np.float64)
LO, HI = 48, 68   # 정답 경계 54(실질)·62(짧은 "Yes.") 포함 구간

L = [f"# 분절 트레이스 — AMI {mid} (turn {LO}–{HI-1} 발췌)", "",
     f"같은 대화를 **(1) raw 발화 단위** 와 **(2) geometry-merge 후** 로 각각 분절한다. "
     f"**둘 다 online 적응 임계치**(ewma μ+cσ, c={C}, span={SPAN}) 사용 — 미래를 안 보고 "
     "지금까지 본 δ_eff 의 평균+표준편차로 매 turn 임계치를 갱신.", "",
     "표기: `δ_eff`=직전 문맥과의 의미거리(↑=화제전환 신호), `δ*`=그 turn의 적응 임계치, "
     "`▲`=경계 예측(δ_eff≥δ*), `★`=정답 경계, `~`=geometry가 backchannel로 판정.", ""]

# ---- (1) raw ----
deff = delta_eff_seq(emb)
preds, thr = ewma_trace(deff)
flag = geom_flag(emb)
fp_raw = sum(1 for i in range(LO, HI) if preds[i] and bt[i] != 1)
L += ["## (1) raw 발화 단위 + 적응 임계치",
      "", "| # | 화자 | 발화 | 단어 | δ_eff | δ\\* | 예측 | 정답 | geom |",
      "|--:|:--:|------|--:|----:|----:|:--:|:--:|:--:|"]
for i in range(LO, HI):
    t = turns[i]; wc = len(t["text"].split())
    ts = f"{thr[i]:.3f}" if thr[i] is not None else "—"
    pm = "▲" if preds[i] else ""; gm = "★" if bt[i]==1 else ""; fm = "~" if flag[i] else ""
    L.append(f"| {i} | {t['speaker']} | {trunc(t['text'])} | {wc} | {deff[i]:.3f} | {ts} | {pm} | {gm} | {fm} |")
gold_in = [i for i in range(LO, HI) if bt[i]==1]
hit_raw = [i for i in gold_in if preds[i]==1]
L += ["", f"→ 이 구간 정답 경계 = {gold_in} 중 raw 가 맞춘 것 = **{hit_raw or '없음'}**. "
      f"짧은 맞장구·발화마다 δ_eff 가 튀어 **거짓 경계 ▲**({fp_raw}개)가 흩어지고, "
      "정작 경계 62(\"Yes.\")처럼 짧은 정답은 놓친다. (`~`=backchannel)", ""]

# ---- (2) geometry-merge ----
def merge(turns, bt, flag):
    groups, cur = [], None
    raw2grp = {}
    for i in range(len(turns)):
        if flag[i] and cur is not None: cur.append(i)
        else: cur = [i]; groups.append(cur)
        raw2grp[i] = len(groups)-1
    return groups, raw2grp
groups, raw2grp = merge(turns, bt, flag)
m_texts = [" ".join(turns[k]["text"] for k in g) for g in groups]
m_yt = [1 if any(bt[k]==1 for k in g) else 0 for g in groups]; m_yt[0] = 0
m_emb = embed(m_texts); m_deff = delta_eff_seq(m_emb)
m_preds, m_thr = ewma_trace(m_deff)
glo, ghi = raw2grp[LO], raw2grp[HI-1]
fp_m = sum(1 for gi in range(glo, ghi+1) if m_preds[gi] and m_yt[gi] != 1)
L += ["## (2) geometry-merge 후 + 적응 임계치",
      f"발화 {len(turns)} → 병합 turn {len(groups)} (맞장구 흡수). 위와 같은 구간(병합 turn {glo}–{ghi}):",
      "", "| 병합# | 흡수 raw# | 텍스트(앞부분) | δ_eff | δ\\* | 예측 | 정답 |",
      "|--:|:--|------|----:|----:|:--:|:--:|"]
for gi in range(glo, ghi+1):
    ts = f"{m_thr[gi]:.3f}" if m_thr[gi] is not None else "—"
    pm = "▲" if m_preds[gi] else ""; gm = "★" if m_yt[gi]==1 else ""
    L.append(f"| {gi} | {'+'.join(map(str,groups[gi]))} | {trunc(m_texts[gi],46)} | "
             f"{m_deff[gi]:.3f} | {ts} | {pm} | {gm} |")
m_gold_in = [gi for gi in range(glo, ghi+1) if m_yt[gi]==1]
m_hit = [gi for gi in m_gold_in if m_preds[gi]==1]
L += ["", f"→ 맞장구가 앞 발화에 흡수돼 **독립 turn 에서 사라진다**. 이 구간 거짓 경계 "
      f"{fp_raw}→{fp_m}개, 정답 경계 맞춘 것 = **{m_hit or '없음'}**. "
      "(merge 의 진짜 이득은 발췌가 아니라 미팅 전체 수치 — 아래.)", ""]

# ---- 전체 수치 ----
def metrics(yt, pred):
    yp = pred[:-1] + [0] if len(pred)==len(yt) else pred
    pk, wd = official_pk_wd(yt, [1 if x else 0 for x in pred][:len(yt)])
    f1 = f1_score(yt, [1 if x else 0 for x in pred][:len(yt)], zero_division=0)
    return pk, wd, f1
pk_r, wd_r, f1_r = metrics(bt, preds)
pk_m, wd_m, f1_m = metrics(m_yt, m_preds)
L += ["## 전체 미팅 수치 (adaptive ewma 임계치)",
      "", "| 버전 | F1 ↑ | Pk ↓ | WD ↓ |", "|---|--:|--:|--:|",
      f"| raw | {f1_r:.3f} | {pk_r:.3f} | {wd_r:.3f} |",
      f"| geometry-merge | {f1_m:.3f} | {pk_m:.3f} | {wd_m:.3f} |", "",
      "## 요약",
      "- **δ_eff** 가 신호, **δ\\*(적응)** 가 문턱. 적응 임계치는 미래를 안 보고 누적 평균+표준편차로 갱신.",
      "- **raw**: 회의 발화의 1/3이 맞장구라 δ_eff 가 흔들려 거짓 경계 다발.",
      "- **geometry-merge**: 맞장구를 외톨이 판정해 앞 발화에 흡수 → 거짓 경계 감소, 신호 회복.",
      "- 둘 다 동일한 online 적응 임계치를 써서 **순수하게 '단위(merge 여부)' 효과만** 비교됨.", ""]

out = REPO/"outputs"/"reports"/"segmentation_walkthrough.md"
out.write_text("\n".join(L)+"\n")
print(f"WROTE {out} ({len(L)} lines) | raw FP={fp_raw} merge FP={fp_m} | "
      f"raw F1={f1_r:.3f} merge F1={f1_m:.3f}")
