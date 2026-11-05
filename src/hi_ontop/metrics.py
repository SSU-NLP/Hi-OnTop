"""Dialogue-mean DTS metrics matching SuperDialseg (pinned in PAPER_ALIGNMENT.md)."""
import numpy as np
from nltk.metrics import pk, windowdiff
from sklearn.metrics import f1_score

def score_dialogues(golds, preds):
    if not golds or len(golds)!=len(preds): raise ValueError("Expected matching nonempty dialogue lists")
    rows=[]
    for gold,pred in zip(golds,preds):
        if len(gold)!=len(pred) or len(gold)<2: raise ValueError("Each dialogue requires at least two turns")
        if not set(list(gold)+list(pred))<= {0,1}: raise ValueError("Boundary labels must be binary")
        g=list(gold);p=list(pred);g[-1]=p[-1]=0
        k=max(2,round(len(g)/(sum(g)+1)/2))
        gs=''.join(map(str,g));ps=''.join(map(str,p))
        rows.append([pk(gs,ps,k=k),windowdiff(gs,ps,k=k),f1_score(g,p,zero_division=0)])
    p,w,f=np.mean(rows,axis=0).tolist()
    return dict(pk=p,wd=w,f1=f,score=.5*f+.25*(1-p)+.25*(1-w),n=len(rows))

def signal_to_pred(signal, threshold):
    return [int(v>=threshold) for v in signal[1:]]+[0] if len(signal) else []
