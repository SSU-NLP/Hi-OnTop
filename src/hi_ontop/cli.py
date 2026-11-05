import argparse,json
from pathlib import Path
import numpy as np
from .pipeline import segment,segment_embeddings,calibrate,ENCODER
from .metrics import score_dialogues

def main():
    p=argparse.ArgumentParser(description='Hi-OnTop cosine dialogue segmentation')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('segment');q.add_argument('input');q.add_argument('--embeddings',action='store_true')
    q.add_argument('--delta-star',type=float,default=.5594);q.add_argument('--encoder',default=ENCODER)
    q.add_argument('--device',default='cpu');q.add_argument('--window',type=int,default=2)
    q.add_argument('--decay',type=float,default=.7);q.add_argument('--blend',type=float,default=.5)
    q.add_argument('--warmup',choices=['paper','legacy'],default='paper')
    q=sub.add_parser('calibrate');q.add_argument('input',help='JSON list: embeddings, pseudo_labels per dialogue')
    q.add_argument('--metric',choices=['score','f1'],default='score')
    q=sub.add_parser('evaluate');q.add_argument('input',help='JSON object with golds, preds end-turn labels')
    a=p.parse_args()
    if a.command=='segment':
        kw=dict(delta_star=a.delta_star,ctx_window=a.window,ctx_decay=a.decay,ctx_blend_a=a.blend,warmup=a.warmup)
        out=segment_embeddings(np.load(a.input,allow_pickle=False),**kw) if a.embeddings else segment(
            Path(a.input).read_text().splitlines(),encoder=a.encoder,device=a.device,**kw)
    elif a.command=='calibrate':
        rows=json.loads(Path(a.input).read_text());out=calibrate([r['embeddings'] for r in rows],[r['pseudo_labels'] for r in rows],metric=a.metric)
    else:
        data=json.loads(Path(a.input).read_text());out=score_dialogues(data['golds'],data['preds'])
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
