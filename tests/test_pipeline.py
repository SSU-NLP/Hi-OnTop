import numpy as np
import pytest
from hi_ontop.pipeline import segment_embeddings,calibrate
from hi_ontop.hi_ontop import HiOnTop
from hi_ontop.metrics import score_dialogues

def test_boundary_alignment_and_strength():
    r=segment_embeddings([[1,0],[1,0],[0,1],[0,1]],delta_star=.5,ctx_window=1)
    assert r['boundaries']==[2] and r['segments']==[[0,2],[2,4]]
    assert r['graded_scores']==[0,0,2,0]

def test_causality_and_warmup():
    z=np.random.default_rng(31).normal(size=(12,8))
    full=segment_embeddings(z,ctx_window=4)
    for n in range(1,12):
        prefix=segment_embeddings(z[:n],ctx_window=4)
        assert prefix['scores']==full['scores'][:n]
    normalized=z/np.linalg.norm(z,axis=1,keepdims=True)
    assert np.isclose(full['scores'][2],1-normalized[1]@normalized[2])

def test_calibration_and_validation():
    z=[[1,0],[1,0],[0,1],[0,1]]
    r=calibrate([z],[[0,1,0,0]],percentiles=[50,75],ctx_window=1)
    assert r['selected']['score']==1.
    assert score_dialogues([[0,1,0,0]],[[0,1,0,0]])['score']==1.
    with pytest.raises(ValueError):HiOnTop(2,delta_star=0)
    with pytest.raises(ValueError):HiOnTop(2).assign(np.zeros(2))
