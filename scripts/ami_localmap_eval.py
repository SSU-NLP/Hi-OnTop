#!/usr/bin/env python3
"""Hi-OnTop active-event local MAP 분절 (codex 설계) — AMI 139미팅 전체 비교.
δ_eff magnitude threshold 폐기 → boundary = active event가 x_t를 설명 못 하고
새 event가 posterior상 더 그럴듯할 때. ablation A1~A5 + baseline + LLM 한 표.
metric: ±2 tol F1 + Pk + WD + Score(=0.5F1+0.25(1-Pk)+0.25(1-WD))."""
from __future__ import annotations
import json, pickle, glob, sys, math
from dataclasses import dataclass, field
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import delta_eff_seq, official_pk_wd
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from hi_ontop.baselines.texttiling_streaming import StreamingTextTiling

TOPIC = "data/ami/topic"; CACHE = "outputs/runs/_misc/ami_emb"; LLMP = "outputs/runs/_misc/ami_llm_pred"
mids = sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))


@dataclass
class HP:
    rho_min: float = 0.05; eta_r: float = 0.05; eta_g: float = 0.01; eta_spk: float = 0.02
    sig_min: float = 0.05; alpha: float = 0.05; L_min: int = 6; R: int = 4
    beta_short: float = 0.7; beta_age: float = 0.4; L_ref: float = 30.0
    g_spk: float = 0.5; g_coh: float = 1.0; g_single: float = 1.0; g_iso: float = 0.7
    n_var: float = 8.0; warmup: int = 8


class RunStat:
    """adaptive-rate online mean/var (count→EWMA)."""
    def __init__(self, eta: float):
        self.eta = eta; self.n = 0; self.mean = 0.0; self.var = 0.0
    def push(self, x: float) -> None:
        self.n += 1; rate = max(self.eta, 1.0/self.n)
        d = x - self.mean; self.mean += rate*d
        self.var = (1-rate)*(self.var + rate*d*d)


def segment_localmap(X: np.ndarray, spk: list, hp: HP, stage: int = 5,
                     tau: float = 0.0, collect=None) -> list[int]:
    """X: (n,d) unit-normalized turn embeddings. 반환: boundary turn index 리스트.
    tau = 결정 임계치 (boundary iff score>tau). collect: score 수집 리스트(진단용)."""
    n = len(X)
    g = RunStat(hp.eta_g); adj = RunStat(hp.eta_g)
    same = RunStat(hp.eta_spk); switch = RunStat(hp.eta_spk)
    m = X[0].copy(); nE = 1
    ev = RunStat(hp.eta_r)               # within-event turn-to-prototype distance
    last_x = X[0]; last_spk = spk[0]
    bounds = []
    for t in range(1, n):
        x = X[t]
        r_active = 1.0 - float(x @ m)
        d_prev = 1.0 - float(x @ last_x)
        mu_g = g.mean if g.n > 0 else 0.5
        sig_g = math.sqrt(max(g.var, hp.sig_min**2))
        mu_adj = adj.mean if adj.n > 0 else 0.5
        sig_adj = math.sqrt(max(adj.var, hp.sig_min**2))
        lam = min(1.0, nE/hp.n_var)
        sE_local = ev.mean if ev.n > 0 else mu_g
        varE_local = max(ev.var, hp.sig_min**2)
        mu_E = lam*sE_local + (1-lam)*mu_g
        var_E = max(lam*varE_local + (1-lam)*max(g.var, hp.sig_min**2), hp.sig_min**2)
        sig_E = math.sqrt(var_E)
        # --- LLR (active event 설명 실패) ---
        excess = max(0.0, r_active - mu_E)
        LLR = max(0.0, 0.5*(excess*excess)/(sig_E*sig_E) + math.log(sig_E/sig_g))
        # --- sCRP prior ---
        if stage <= 1:
            prior = math.log(hp.alpha) - math.log(nE) + hp.beta_age*math.log(1+nE/hp.L_ref)
        else:
            prior = (math.log(hp.alpha) - math.log(nE)
                     - hp.beta_short*max(0, hp.L_min - nE)
                     + hp.beta_age*math.log(1+nE/hp.L_ref))
        # --- penalties (stage별 점증) ---
        pen = 0.0
        if stage >= 3:
            pen += hp.g_coh*max(0.0, d_prev - r_active)/sig_adj
        if stage >= 4 and (same.n > 0 and switch.n > 0):
            pen += hp.g_spk*(1.0 if spk[t] != last_spk else 0.0)*max(0.0, switch.mean - same.mean)/sig_adj
        if stage >= 5:
            z_prev = (d_prev - mu_adj)/sig_adj
            z_active = (r_active - mu_E)/sig_E
            pen += hp.g_single*(1.0 if nE < hp.L_min else 0.0) + hp.g_iso*max(0.0, z_prev - z_active)
        score = LLR + prior - pen
        if collect is not None and t >= hp.warmup and nE >= (1 if stage <= 1 else hp.R):
            collect.append(score)
        Rgate = 1 if stage <= 1 else hp.R
        is_bnd = (score > tau) and (nE >= Rgate) and (t >= hp.warmup)
        # --- 통계 갱신 (과거만) ---
        adj.push(d_prev)
        (same if spk[t] == last_spk else switch).push(d_prev)
        if is_bnd:
            bounds.append(t)
            m = x.copy(); nE = 1; ev = RunStat(hp.eta_r)
        else:
            g.push(r_active); ev.push(r_active)
            rho = max(hp.rho_min, 1.0/(nE+1))
            m = (1-rho)*m + rho*x; m = m/(np.linalg.norm(m)+1e-12)
            nE += 1
        last_x = x; last_spk = spk[t]
    return bounds


# ---------------- 평가 ----------------
def tol_f1(gold, pred, t=2):
    if not pred or not gold: return 0.0
    p = sum(1 for i in pred if any(abs(i-j) <= t for j in gold))/len(pred)
    r = sum(1 for j in gold if any(abs(i-j) <= t for i in pred))/len(gold)
    return 2*p*r/(p+r) if p+r > 0 else 0.0

DATA = []
for mid in mids:
    d = json.load(open(f"{TOPIC}/{mid}.json")); turns = d["turns"]; bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
    gold = [i for i, b in enumerate(bt) if b == 1]
    e = np.asarray(pickle.load(open(f"{CACHE}/{mid}.pkl", "rb")), dtype=np.float64)
    e = e/(np.linalg.norm(e, axis=1, keepdims=True)+1e-9)
    spk = [t.get("speaker", "?") for t in turns]
    llm = [p for p in json.load(open(f"{LLMP}/{mid}.json"))["pred"] if 0 < p < n] if glob.glob(f"{LLMP}/{mid}.json") else []
    DATA.append((mid, turns, bt, n, gold, e, spk, llm))

def evalp(pred_fn):
    F2, PK, WD, SC = [], [], [], []; tot = 0
    for mid, turns, bt, n, gold, e, spk, llm in DATA:
        pred = sorted(set(p for p in pred_fn(mid, turns, n, e, spk, llm) if 0 < p < n-1))
        tot += len(pred); f2 = tol_f1(gold, pred)
        yt = [int(b) for b in bt]; yp = [1 if i in set(pred) else 0 for i in range(n)]
        pk, wd = official_pk_wd(yt, yp)
        F2.append(f2); PK.append(pk); WD.append(wd); SC.append(0.5*f2+0.25*(1-pk)+0.25*(1-wd))
    return tot, np.mean(F2), np.mean(PK), np.mean(WD), np.mean(SC)

def even(mid, turns, n, e, spk, llm):
    g = len([1 for b in DATA[mids.index(mid)][2] if b == 1])
    return [int(round((i+1)*n/(g+1))) for i in range(g)] if g else []

def tt(mid, turns, n, e, spk, llm):
    seg = StreamingTextTiling(); out = []
    for t in turns: out += seg.push(t["text"])
    out += seg.flush(); return out

def hi_ewma(mid, turns, n, e, spk, llm):
    return [i for i, b in enumerate(adaptive_boundaries(list(delta_eff_seq(e)), c=1.5, mode="ewma")) if b]

hp = HP()
ngold = sum(len(g) for _, _, _, _, g, _, _, _ in DATA)

# --- score 분포 진단 (τ 범위 잡기): A5 ---
sc_all = []
for mid, turns, bt, n, gold, e, spk, llm in DATA:
    segment_localmap(e, spk, hp, 5, tau=-1e9, collect=sc_all)
sc_all = np.array(sc_all)
print(f"[진단] A5 per-turn score 분포 (n={len(sc_all)}): "
      f"p50={np.percentile(sc_all,50):.2f} p90={np.percentile(sc_all,90):.2f} "
      f"p99={np.percentile(sc_all,99):.2f} max={sc_all.max():.2f}")
print()

print(f"AMI 전체 {len(mids)}미팅, gold={ngold}.  F1=±2tol, Pk/WD↓good, Score=0.5F1+0.25(1-Pk)+0.25(1-WD)")
print(f"{'method':<32}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}")
base_rows = [
    ("Even-spacing (count oracle)", even),
    ("TextTiling (streaming)", tt),
    ("Hi-OnTop δ_eff ewma [기존]", hi_ewma),
]
for name, fn in base_rows:
    tot, f2, pk, wd, sc = evalp(fn)
    print(f"{name:<32}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
# local-MAP: stage A5 를 τ 글로벌 sweep
taus = [round(float(np.percentile(sc_all, p)), 2) for p in (50, 75, 90, 95, 98)]
for tau in taus:
    fn = lambda mid,turns,n,e,spk,llm,tau=tau: segment_localmap(e, spk, hp, 5, tau=tau)
    tot, f2, pk, wd, sc = evalp(fn)
    print(f"{('local-MAP A5 (τ='+str(tau)+')'):<32}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
# 고정 τ(중앙 p90)에서 ablation A1~A5
tau_fix = round(float(np.percentile(sc_all, 90)), 2)
print(f"-- ablation @ τ={tau_fix} (A5 p90) --")
for stg in (1, 2, 3, 4, 5):
    fn = lambda mid,turns,n,e,spk,llm,stg=stg: segment_localmap(e, spk, hp, stg, tau=tau_fix)
    tot, f2, pk, wd, sc = evalp(fn)
    print(f"{('local-MAP A'+str(stg)):<32}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
tot, f2, pk, wd, sc = evalp(lambda mid,turns,n,e,spk,llm: llm)
print(f"{'LLM full-context (cache)':<32}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")
