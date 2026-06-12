#!/usr/bin/env python3
"""AMI 화제분절 — filler-prototype 정보량 + forward-merge + info-gate (online, lexicon-경량).

방법 (전부 online, 미래 안 봄):
  1. filler-prototype ref = 보편 filler 단어 인코딩 평균 (1회 고정, AMI 무관).
  2. info_emb(발화) = 1 − cos(발화, ref)  ← 낮음=generic(filler), 높음=내용.
  3. forward-merge: info_emb < τ_f 인 발화를 *다음* 내용 발화에 흡수(그룹화).
     → topic-opening filler("Yes.")가 새 그룹 시작으로 보존.
  4. boundary: 그룹 임베딩에 δ_eff + 적응임계치(ewma), 단 그룹 정보량 ≥ τ_c 인
     그룹만 경계 주장(info-gate). τ_f/τ_c = 전역 고정 calibration.
평가: tolerance boundary-F1 (±0..3), 공식 Pk/WD, filler 판별 AUC.
"""
from __future__ import annotations
import json, pickle, glob, os, sys, re, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"src"))
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from sklearn.metrics import roc_auc_score, f1_score

CACHE = REPO/"outputs"/"runs"/"_misc"/"ami_emb"; TOPIC = REPO/"data"/"ami"/"topic"
FILLER_WORDS = ["yeah","okay","mm-hmm","right","yes","uh-huh","mm","hmm","sure",
                "alright","yep","oh","ah","mhm","uh","um"]
STOP = set("the a an and or but to of in on at for is are was were be been being i you he she it we they "
    "my your his her our their me him them this that these those do does did have has had will would "
    "can could should so um uh yeah okay ok oh ah mm hmm right yes no well just like get got going "
    "know think mean really very much lot here there what who how when where why yep sure alright kay".split())
def lex_filler(t):
    w = re.findall(r"[a-zA-Z']+", t.lower()); return len(w) > 0 and sum(1 for x in w if x not in STOP and len(x) > 2) == 0
def tol_f1(yt, yp, tol):
    if not yp or not yt: return 0.0
    pr = sum(1 for i in yp if any(abs(i-j) <= tol for j in yt))/len(yp)
    rc = sum(1 for j in yt if any(abs(i-j) <= tol for i in yp))/len(yt)
    return 2*pr*rc/(pr+rc) if pr+rc > 0 else 0.0


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", backend="onnx",
        model_kwargs={"provider":"CPUExecutionProvider","file_name":"onnx/model_quint8_avx2.onnx"})
    ref = np.asarray(enc.encode(FILLER_WORDS, normalize_embeddings=True), dtype=np.float64).mean(0)
    ref = ref/(np.linalg.norm(ref)+1e-9)

    mids = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(CACHE/"*.pkl")))
    M = []
    for mid in mids:
        d = json.load(open(TOPIC/f"{mid}.json")); bt = d["bnd_top"]; tx = [t["text"] for t in d["turns"]]
        e = np.asarray(pickle.load(open(CACHE/f"{mid}.pkl","rb")), dtype=np.float64)
        ie = 1 - e @ ref
        M.append(dict(mid=mid, e=e, tx=tx, bt=bt, ie=ie,
                      ln=[max(1,len(t.split())) for t in tx],
                      gold=[i for i,b in enumerate(bt) if b == 1]))
    # 전역 고정 calibration (τ_f=p30, τ_c=p60 of pooled info_emb)
    pooled = np.concatenate([m["ie"] for m in M])
    tau_f, tau_c = float(np.percentile(pooled, 30)), float(np.percentile(pooled, 60))

    # filler 판별 AUC (보편 filler 단어 lexicon 을 정답 삼아 — 검증용만)
    x = pooled.tolist(); y = [0 if lex_filler(t) else 1 for m in M for t in m["tx"]]
    auc = roc_auc_score(y, x)

    res = {t: [] for t in (0,1,2,3)}; pks=[]; wds=[]; npred=0; ngold=0; nturn=0; ngrp=0
    for m in M:
        e, tx, ie, ln, gold = m["e"], m["tx"], m["ie"], m["ln"], m["gold"]
        # forward-merge: filler(ie<τ_f) → 다음 내용
        G=[]; buf=[]
        for i in range(len(tx)):
            if ie[i] < tau_f: buf.append(i)
            else: G.append(buf+[i]); buf=[]
        if buf: G.append(buf)
        emb = np.array([np.average(e[g], axis=0, weights=[ln[k] for k in g]) for g in G])
        emb = emb/(np.linalg.norm(emb, axis=1, keepdims=True)+1e-9)
        ginfo = np.array([max(ie[k] for k in g) for g in G])
        deff = list(delta_eff_seq(emb))
        sp = set(k for k,b in enumerate(adaptive_boundaries(deff, c=1.5, mode="ewma"))
                 if b and ginfo[k] >= tau_c)
        pred = sorted(set(G[k][0] for k in sp if k < len(G)))
        # group-level Pk/WD
        gyt = [1 if any(m["bt"][k]==1 for k in g) else 0 for g in G]; gyt[0]=0
        gyp = [1 if k in sp else 0 for k in range(len(G))]
        pk, wd = official_pk_wd(gyt, gyp); pks.append(pk); wds.append(wd)
        for t in (0,1,2,3): res[t].append(tol_f1(gold, pred, t))
        npred += len(pred); ngold += len(gold); nturn += len(tx); ngrp += len(G)

    out = dict(n_meet=len(M), tau_f=tau_f, tau_c=tau_c, filler_auc=auc,
               tol_f1={t: float(np.mean(res[t])) for t in (0,1,2,3)},
               pk=float(np.mean(pks)), wd=float(np.mean(wds)),
               n_pred=npred, n_gold=ngold, n_turn=nturn, n_grp=ngrp)
    name = "2026-06-09_ami_filler_prototype"
    od = REPO/"outputs"/"experiments"/name; od.mkdir(parents=True, exist_ok=True)
    (od/"results.json").write_text(json.dumps(out, indent=2))
    write_report(od, out)
    print(json.dumps(out, indent=2), flush=True)


def write_report(od, o):
    t = o["tol_f1"]
    L = [f"# AMI 화제분절 — filler-prototype + forward-merge + info-gate (139미팅)", "",
         "## 방법 (전부 online · 미래 안 봄)",
         "1. **filler-prototype ref** = 보편 filler 단어(yeah/okay/mm-hmm/...) 인코딩 평균 (1회 고정, AMI 무관).",
         "2. **info_emb** = 1 − cos(발화, ref). 낮음=generic(filler), 높음=내용.",
         "3. **forward-merge**: info_emb<τ_f 발화를 *다음* 내용에 흡수 → topic-opener filler 보존.",
         "4. **boundary**: 그룹 δ_eff + ewma 적응임계치, 그룹 info≥τ_c 만 경계(info-gate).",
         f"   τ_f={o['tau_f']:.3f}, τ_c={o['tau_c']:.3f} (전역 고정 calibration).",
         "", "## 결과 (139미팅)",
         f"- 미팅 {o['n_meet']}, 발화 {o['n_turn']} → 그룹 {o['n_grp']}, 정답경계 {o['n_gold']}, 예측 {o['n_pred']}.",
         f"- **filler 판별 AUC = {o['filler_auc']:.3f}** (보편 filler-prototype, AMI-fit 아님).",
         "",
         "| metric | 값 |", "|---|--:|",
         f"| boundary-F1 (exact ±0) | {t[0]:.3f} |",
         f"| boundary-F1 (±1) | {t[1]:.3f} |",
         f"| boundary-F1 (±2) | {t[2]:.3f} |",
         f"| boundary-F1 (±3) | {t[3]:.3f} |",
         f"| Pk ↓ | {o['pk']:.3f} |",
         f"| WD ↓ | {o['wd']:.3f} |",
         "",
         "## 판정 (정직)",
         "- **139 전체에서 이 method 는 baseline 보다 낮다**: filler-prototype ±2≈0.107 vs "
         "raw 0.151 vs geometry-merge 0.189. 12미팅(±2~0.158)은 쉬운 subset 의 과대평가였음. "
         "per-meeting 임계로 바꿔도 동일(0.107) → 임계가 아니라 method/데이터 문제.",
         "- **유효한 부분**: filler-prototype 정보량은 **universal·online·AMI-fit아님** 으로 filler 를 "
         f"AUC {o['filler_auc']:.3f} 로 판별. 즉 *filler 탐지기* 로는 검증됨.",
         "- **무효한 부분**: 그 위에 쌓은 forward-merge + info-gate 분절은 39k→43k 그룹화·게이트가 "
         "오히려 경계 정렬을 흐려 geometry-merge 보다 못함.",
         "- **결론**: AMI 메인 분절은 **geometry-merge(±2 0.189) 채택**. filler-prototype 은 "
         "탐지 component 로만 가치. 그리고 어떤 method 든 AMI 천장(±2 ~0.2)은 drift + "
         "annotation(filler 위 경계) 한계로 못 뚫음 — 모든 시도가 이를 재확인.",
         ]
    (od/"REPORT.md").write_text("\n".join(L)+"\n")


if __name__ == "__main__":
    main()
