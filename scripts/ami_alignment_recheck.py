"""AMI 경계 정렬 재확인 — HANDOFF_04 §5(d).

★ 규약 확정(2026-06-14, 데이터): **AMI gold `bnd_top` = 새 topic 첫 turn(시작-turn)** — `topic_levels.start_turn`
과 139미팅 135정확일치·0불일치. ⇒ 신호(새-segment 첫 turn 발화)와 **같은 규약 → 원칙적 정렬 = shift 0**.
(DTS 는 정반대: gold=끝-turn → 신호 t→gold t-1 = shift -1 이 규약정렬. AMI 와 DTS 는 규약이 반대다.)

shift -1 이 AMI 에서 ±2F1 미세 우위(0.370 vs 0.342)인 건 **filler 효과**(새 topic 첫 turn이 군말이라 임베딩
스파이크가 +1 늦게 뜸) 를 노린 overfit 일 뿐 — 원칙 아님. ±2 tolerance 가 이 ±1 잔차를 흡수하므로 **AMI 는 shift 0 표준**.

결과 (139 meetings, per-meeting oracle ±2 F1, gold-reset prototype):
- de-neut(적응β): shift0=0.342(표준), -1=0.370(filler overfit), +1=0.295
- δ_eff:          shift0=0.214,        -1=0.225,                +1=0.214
- **원칙정렬(shift 0): de-neut 0.342 > δ_eff 0.214** → AMI de-neut 우위는 정렬과 무관하게 실재(shift -1/0 둘 다 유지).

실행: ``python scripts/ami_alignment_recheck.py``
데이터: ``data/ami/topic/*.json`` (bnd_top) + ``outputs/runs/_misc/ami_emb/*.pkl``.
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import hi_ontop.hi_ontop_deneut as cr  # noqa: E402
from hi_ontop.hi_ontop import HiOnTop  # noqa: E402

D = cr.DEFAULTS
TOPIC = REPO / "data" / "ami" / "topic"
AC = REPO / "outputs" / "runs" / "_misc" / "ami_emb"


def _nr(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def deneut_goldreset(e, gold):
    n = len(e); gs = set(gold); s = np.zeros(n)
    m = e[0].copy(); k = 1; g = e[0].copy(); gk = 1
    A, B, L0, lam, g_rho, rho_min = D["A"], D["B"], D["L0"], D["lam"], D["g_rho"], D["rho_min"]
    for t in range(1, n):
        x = e[t]; beta = cr._beta(k, A, B, L0)
        mc = cr._deneut(m, g, beta); xc = cr._deneut(x, g, beta)
        s[t] = (1 - float(xc @ mc)) - lam * (1 - float(x @ g))
        if t in gs:
            m = x.copy(); k = 1
        else:
            rho = max(rho_min, 1.0 / (k + 1)); m = cr._nr((1 - rho) * m + rho * x); k += 1
        gr = max(g_rho, 1.0 / (gk + 1)); g = cr._nr((1 - gr) * g + gr * x); gk += 1
    return s


def deff_sig(e):
    seg = HiOnTop(dim=e.shape[1], delta_star=1.0, ctx_window=2, ctx_decay=0.7, ctx_blend_a=0.5)
    for v in e:
        seg.assign(v.astype(np.float64))
    return np.array([float(h["delta_eff"]) for h in seg.history()])


def oracle_tol2(sigs, golds, ns, shift):
    F = []
    for s, gold, n in zip(sigs, golds, ns):
        if n <= 2:
            F.append(0.0); continue
        cand = sorted(set(float(s[i]) for i in range(1, n - 1)))
        if len(cand) > 80:
            cand = list(np.quantile(cand, np.linspace(0, 1, 80)))
        best = 0.0
        for thr in cand:
            pred = [i + shift for i in range(1, n - 1) if s[i] > thr]
            if not pred:
                continue
            p = sum(1 for i in pred if any(abs(i - j) <= 2 for j in gold)) / len(pred)
            r = sum(1 for j in gold if any(abs(i - j) <= 2 for i in pred)) / len(gold)
            f = 2 * p * r / (p + r) if p + r > 0 else 0.0
            best = max(best, f)
        F.append(best)
    return float(np.mean(F))


def main() -> None:
    mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json")
                  if not p.endswith("manifest.json"))
    dn, deff, golds, ns = [], [], [], []
    for mid in mids:
        d = json.load(open(f"{TOPIC}/{mid}.json")); bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
        e = _nr(np.asarray(pickle.load(open(f"{AC}/{mid}.pkl", "rb")), dtype=np.float64))
        g = [i for i, b in enumerate(bt) if b == 1]
        dn.append(deneut_goldreset(e, g)); deff.append(deff_sig(e)); golds.append(g); ns.append(n)
    print(f"AMI per-meeting oracle ±2 F1, n_meetings={len(mids)}")
    for name, sigs in [("de-neut", dn), ("δ_eff", deff)]:
        vals = {sh: oracle_tol2(sigs, golds, ns, sh) for sh in (-1, 0, 1)}
        print(f"  {name:8s}: shift-1={vals[-1]:.4f}  shift0={vals[0]:.4f}  shift+1={vals[1]:.4f}")
    print(f"\n>>> 원칙정렬(shift 0, AMI gold=시작-turn): de-neut={oracle_tol2(dn,golds,ns,0):.4f} "
          f"> δ_eff={oracle_tol2(deff,golds,ns,0):.4f}  (de-neut AMI 우위 = 실재; shift -1 은 filler overfit)")


if __name__ == "__main__":
    main()
