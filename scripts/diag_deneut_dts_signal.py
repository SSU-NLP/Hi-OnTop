import sys, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from sklearn.metrics import f1_score
from run_encoder_comparison import CACHE, load_dialogs
from hi_ontop import dts_scoring as S
import hi_ontop.hi_ontop_deneut as dn
D=dn.DEFAULTS
def nrm(v): return v/(np.linalg.norm(v)+1e-9)

def signals(e, gold):
    """여러 신호를 한 번에 (gold-reset prototype 공유)."""
    n=len(e); gs=set(gold)
    out={k:np.zeros(n) for k in ["dprev","dctx","deff","V_cur","deneut_b1","deneut_prev","blend"]}
    m=e[0].copy(); k=1; g=e[0].copy(); gk=1
    recent=[e[0]]
    for t in range(1,n):
        x=e[t]; xp=e[t-1]
        # δ_prev / δ_ctx / δ_eff (HiOnTop, m=2 ρ=0.7 a=0.5)
        dprev=1-float(x@xp)
        c=np.zeros_like(x)
        for i,s in enumerate(reversed(recent[-2:])): c=c+(0.7**i)*s
        c=nrm(c); dctx=1-float(x@c); deff=0.5*dprev+0.5*dctx
        out["dprev"][t]=dprev; out["dctx"][t]=dctx; out["deff"][t]=deff
        # de-neut 변형들 (현 prototype m, global g, 적응 β)
        beta=dn._beta(k,D['A'],D['B'],D['L0'])
        mc=dn._deneut(m,g,beta); xc=dn._deneut(x,g,beta)
        r_active=1-float(xc@mc); r_global=1-float(x@g)
        out["V_cur"][t]=r_active-D['lam']*r_global                 # 현행 DeNeut
        # β=1 full de-neut, global 항 없음, prototype 비교
        mc1=dn._deneut(m,g,1.0); xc1=dn._deneut(x,g,1.0)
        out["deneut_b1"][t]=1-float(xc1@mc1)
        # de-neut(β=1) 인데 prototype 대신 직전 발화와 비교
        xpc=dn._deneut(xp,g,1.0)
        out["deneut_prev"][t]=1-float(xc1@xpc)
        # blend δ_prev + 현행 V
        out["blend"][t]=0.5*dprev+0.5*out["V_cur"][t]
        # 업데이트 (gold-reset)
        if t in gs: m=x.copy(); k=1; recent=[x]
        else:
            rho=max(D['rho_min'],1.0/(k+1)); m=nrm((1-rho)*m+rho*x); k+=1; recent.append(x)
        gr=max(D['g_rho'],1.0/(gk+1)); g=nrm((1-gr)*g+gr*x); gk+=1
    return out

def oracle(sigs, golds):
    ev=S.new_evaluation()
    for sig,yt in zip(sigs,golds):
        n=len(yt); cand=sorted(set(float(sig[i]) for i in range(1,n)))
        if len(cand)>120: cand=list(np.quantile(cand,np.linspace(0,1,120)))
        ytv=list(yt); ytv[-1]=0; best=None; bf=-1
        for thr in cand:
            yp=S.signal_to_pred(sig,thr); f=f1_score(ytv,yp,zero_division=0)
            if f>bf: bf=f; best=yp
        ev.add(ytv,best)
    return ev.compute()['total_score']

for ds in ["dialseg711","tiage"]:
    dl=load_dialogs(ds,"test")
    embs=[np.asarray(e,dtype=np.float64) for e in pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))]
    golds=[[i for i,b in enumerate(yt) if b==1] for _,yt in dl]
    gylist=[list(yt) for _,yt in dl]
    allsig={k:[] for k in ["dprev","dctx","deff","V_cur","deneut_b1","deneut_prev","blend"]}
    for e,g in zip(embs,golds):
        s=signals(e,g)
        for k in allsig: allsig[k].append(s[k])
    print(f"\n### {ds} oracle Score (official per-dialogue, gold-reset) ###")
    for k in ["deff","dprev","dctx","V_cur","deneut_b1","deneut_prev","blend"]:
        print(f"  {k:12s}: {oracle(allsig[k],gylist):.3f}", flush=True)
