"""Segmentation and calibration without situation-tracking dependencies."""
import numpy as np
from .hi_ontop import HiOnTop
from .metrics import score_dialogues, signal_to_pred

ENCODER='sentence-transformers/multi-qa-mpnet-base-dot-v1'

def segment_embeddings(embeddings, delta_star=.5594, **kwargs):
    z=np.asarray(embeddings,dtype=float)
    if z.ndim!=2 or z.shape[1]<1: raise ValueError("Expected [turns, dimensions]")
    model=HiOnTop(z.shape[1],delta_star=delta_star,**kwargs)
    boundaries=[]
    for i,row in enumerate(z):
        if model.assign(row)[1]:boundaries.append(i)
    cuts=[0]+boundaries+[len(z)]
    return dict(boundaries=boundaries,segments=[list(v) for v in zip(cuts[:-1],cuts[1:])] if len(z) else [],
                scores=[float(x['delta_eff']) for x in model.history()],graded_scores=model.graded_scores())

def segment(texts, delta_star=.5594, encoder=ENCODER, device='cpu', **kwargs):
    if not texts:return dict(boundaries=[],segments=[],scores=[],graded_scores=[])
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(encoder,device=device,revision='17997f24dca0df1a4fed68894fb0e1e133e60482' if encoder==ENCODER else None)
    z=model.encode(texts,normalize_embeddings=True)
    return segment_embeddings(z,delta_star,**kwargs)

def calibrate(documents, pseudo_labels, percentiles=range(5,100,5), metric='score', **kwargs):
    """Fit only on calibration documents; pseudo labels use end-turn alignment."""
    if metric not in ('score','f1'):raise ValueError('metric must be score or f1')
    signals=[segment_embeddings(z,**kwargs)['scores'] for z in documents]
    if len(signals)!=len(pseudo_labels) or not signals:raise ValueError('Calibration inputs must align')
    if any(len(x)!=len(y) or len(x)<2 for x,y in zip(signals,pseudo_labels)):raise ValueError('Invalid calibration lengths')
    pool=np.concatenate([x[1:] for x in signals]); results=[]
    for percentile in percentiles:
        if not 0<percentile<100:raise ValueError('Percentile must be between 0 and 100')
        threshold=float(np.percentile(pool,percentile))
        if threshold<=0:continue
        score=score_dialogues(pseudo_labels,[signal_to_pred(x,threshold) for x in signals])
        results.append(dict(percentile=percentile,delta_star=threshold,**score))
    if not results:raise ValueError('No positive threshold in calibration pool')
    best=max(results,key=lambda row:row[metric])
    return dict(selected=best,selection_metric=metric,candidates=results,parameters=kwargs,
                label_source='user-supplied pseudo labels; provenance required')
