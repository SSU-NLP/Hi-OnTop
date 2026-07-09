import sys, glob, math, pickle, numpy as np
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
from run_encoder_comparison import CACHE
def nrm(v): return v/(np.linalg.norm(v)+1e-9)
def alpha(hl): return 1.0-2.0**(-1.0/hl)
a_f,a_s,a_g=alpha(8),alpha(64),alpha(16); eps=1e-8
def sbg_stream(e):
    m_f=e[0].copy(); m_s=e[0].copy(); q_f=q_s=q_fs=1.0; v_perp=0.0; E_s=0.0; out=[]
    for t in range(1,len(e)):
        x=e[t]; rho_p=np.linalg.norm(m_s); u=m_s/max(rho_p,eps)
        v_perp=(1-a_s)*v_perp+a_s*(1-float(x@u)**2)
        m_f=(1-a_f)*m_f+a_f*x; m_s=(1-a_s)*m_s+a_s*x
        q_f=(1-a_f)**2*q_f+a_f**2; q_s=(1-a_s)**2*q_s+a_s**2; q_fs=(1-a_f)*(1-a_s)*q_fs+a_f*a_s
        q_d=max(q_f+q_s-2*q_fs,eps)
        g_f=m_f/max(np.linalg.norm(m_f),eps); g_s=m_s/max(np.linalg.norm(m_s),eps); rho=np.linalg.norm(m_s)
        d_obs=1-float(g_f@g_s); d_null=0.5*q_d*v_perp/(rho*rho+eps)
        R=d_obs/(d_null+eps); E=max(0.0,math.log(R+eps)); E_s=(1-a_g)*E_s+a_g*E
        out.append(min(max(math.exp(-E_s)*(rho*rho)/(rho*rho+q_s*v_perp+eps),0),1))
    return np.array(out)
def dom(embs, warm=8):
    vals=[];lens=[]
    for e in embs:
        e=nrm(np.asarray(e,dtype=np.float64))
        if len(e)<warm+5: continue
        s=sbg_stream(e); vals.append(float(np.mean(s[warm:]))); lens.append(len(e))
    return np.array(vals),np.array(lens)
def show(v,l,name):
    if len(v)==0: print(f"  {name}: (length filter로 표본 0)"); return
    corr=np.corrcoef(v,l)[0,1] if len(v)>3 else float('nan')
    print(f"  {name:14s}: S_bg={v.mean():.3f}±{v.std():.3f}  n={len(v)}  len {l.min()}-{l.max()}  corr(S_bg,len)={corr:+.2f}")
AC="outputs/runs/_misc/ami_emb"; TOPIC="data/ami/topic"
mids=sorted(p.split("/")[-1][:-5] for p in glob.glob(f"{TOPIC}/*.json") if not p.endswith("manifest.json"))
av,al=dom([pickle.load(open(f"{AC}/{m}.pkl","rb")) for m in mids[:40]])
print("도메인별 S_bg (높을수록 '안정배경 있음' → de-neut 켬):")
show(av,al,"AMI(단일회의)")
for ds in ["dialseg711","tiage","superseg"]:
    embs=pickle.load(open(CACHE/f"enccmp_{ds}_test_minilm-int8.pkl","rb"))
    v,l=dom(embs[:80]); show(v,l,ds)
