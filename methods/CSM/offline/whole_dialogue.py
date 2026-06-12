#!/usr/bin/env python3
"""CSM **offline** (원본 SuperDialseg `CSMSegmenter`, whole-dialogue).

원본 코드는 `benchmarks/superdialseg/src/super_dialseg/models/csm/`
(Coldog2333/SuperDialseg, read-only) 의 `CSMSegmenter` — BERT-NSP 기반
coherence-scoring + TextTiling-style depth, **대화 전체** 에 적용
(미래 포함, prefix 아님). 코드 복사 없이 sys.path 로 import 만 (CLAUDE.md:
benchmarks read-only).

online 판(`methods/CSM/online/delay2.py`) 과 **같은 데이터(Def-DTS 번들)
·같은 metric (segeval Pk/WD + boundary-set F1, Score=0.5F1+0.25(1-Pk)
+0.25(1-WD))** 로 산출 → offline vs online 을 Hi-OnTop 내 apples-to-apples.

ckpt 호환: lxing532 `CoherenceNet` state-dict (우리 `methods/CSM/
cpt_277000.pth`) 와 SuperDialseg `CoherenceScoringModel` 은 가중치
shape 동일·attribute name 만 다르다. 본 runner 의 `_load_lxing532_ckpt`
가 키를 remap 해 둘이 호환되게 한다:

  bert.<rest>                 → bert.bert.<rest>      (BertForNSP wrap)
  coherence_decoder.<n>.<w/b> → coherence_prediction_decoder.<n>.<w/b>

NSP head (`bert.cls.*`) 는 우리 ckpt 에 없지만 CoherenceScoringModel
forward 가 안 쓰므로 `strict=False` 로 로드 (pretrained 값 유지).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
SDS_SRC = REPO / "benchmarks" / "superdialseg" / "src"
DEFDTS_DIR = REPO / "benchmarks" / "Def-DTS" / "data" / "DTS_session_datasets"
DATASETS = ("tiage", "dialseg711", "superseg")


def _utts_and_gold(dialogue: str):
    """Def-DTS 의 `dialogue` 문자열 → (utterances, gold_bnds_1based)."""
    from hi_ontop.baselines._seg_utils import parse_defdts_dialogue
    return parse_defdts_dialogue(dialogue)


def _load_lxing532_ckpt(seg, ckpt_path: str) -> None:
    """lxing532 `CoherenceNet` ckpt 의 key 를 remap 해 SuperDialseg
    `CoherenceScoringModel` 에 로드. 두 모델은 가중치 shape 동일,
    attribute name 만 다르다:

      bert.<rest>                 → bert.bert.<rest>
      coherence_decoder.<n>.<w/b> → coherence_prediction_decoder.<n>.<w/b>

    NSP head (`bert.cls.*`) 는 lxing532 ckpt 에 없으므로 strict=False —
    BertForNSP 가 from_pretrained 로 채운 pretrained 값을 그대로 둠.
    """
    import torch
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    remapped = {}
    for k, v in raw.items():
        if k.startswith("bert."):
            nk = "bert.bert." + k[len("bert."):]
        elif k.startswith("coherence_decoder."):
            nk = "coherence_prediction_decoder." + k[len("coherence_decoder."):]
        else:
            nk = k
        remapped[nk] = v
    missing, unexpected = seg.model.load_state_dict(remapped, strict=False)
    # NSP head 키만 missing 이어야 정상 (pretrained 값 유지). 그 외엔 경고.
    nsp_missing = [m for m in missing if m.startswith("bert.cls.")
                   or "pooler" in m]
    other_missing = [m for m in missing if m not in nsp_missing]
    if other_missing:
        print(f"  [load] WARN missing keys (NSP head 외): "
              f"{len(other_missing)} (샘플 3: {other_missing[:3]})")
    if unexpected:
        print(f"  [load] WARN unexpected keys: {len(unexpected)} "
              f"(샘플 3: {unexpected[:3]})")
    print(f"  [load] OK — remapped {len(remapped)} keys, "
          f"NSP/pooler missing {len(nsp_missing)} (정상)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-23_csm_offline")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument("--limit", type=int, default=0,
                    help="0=full test set; N=앞 N 대화")
    ap.add_argument("--backbone", default="bert-base-uncased")
    ap.add_argument("--max-utterance-len", type=int, default=50)
    ap.add_argument("--cut-rate", type=float, default=1.0)
    ap.add_argument("--ckpt-path", default="methods/CSM/cpt_277000.pth",
                    help="lxing532 CoherenceNet ckpt (우리 학습) — 자동 "
                         "remap 으로 CoherenceScoringModel 에 로드. 미지정/"
                         "''이면 pretrained backbone 만 (Score 낮음).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # super_dialseg/__init__.py 가 textseg(gensim)·baselines(anthropic) 등을
    # eager import 함 → 우리는 CSM 만 쓰므로 미설치 모듈은 stub 으로 통과.
    # openai 는 실제 설치돼 있으므로 stub 안 함 (transformers 가 spec 확인).
    import types
    import importlib.machinery as _mch
    def _stub(name, attrs=None):
        m = types.ModuleType(name)
        m.__spec__ = _mch.ModuleSpec(name, loader=None)
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)
        return m
    _stub("anthropic")
    _gensim = _stub("gensim")
    # textseg/utils_textseg 가 class 정의 시점에 gensim.models.KeyedVectors
    # 를 참조 (실제 사용 X, 타입 힌트). 빈 클래스 stub 으로 통과.
    _gensim_models = _stub("gensim.models",
                           {"KeyedVectors": type("KeyedVectors", (), {})})
    _gensim.models = _gensim_models
    sys.path.insert(0, str(SDS_SRC))
    from super_dialseg.models.csm.modeling_csm import CSMSegmenter  # noqa: E402
    from hi_ontop.baselines._seg_utils import (
        boundary_set_f1, latency_stats, load_defdts, pk_wd)

    print(f"[setup] backbone={args.backbone} device={args.device} "
          f"ckpt={args.ckpt_path or '(none, pretrained-only)'}")
    seg = CSMSegmenter(backbone=args.backbone,
                       max_utterance_len=args.max_utterance_len,
                       cut_rate=args.cut_rate)
    if args.ckpt_path:
        _load_lxing532_ckpt(seg, args.ckpt_path)
    seg.to(args.device)

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        full = load_defdts(ds, DEFDTS_DIR)
        if args.limit:
            full = full[: args.limit]
        print(f"\n=== {ds}: {len(full)} dialogues ===")

        pk_v, wd_v, f1_v, lat = [], [], [], []
        n_pred_total = n_gold_total = 0
        miss = 0
        for did, utts, gold in full:
            n = len(utts)
            t0 = time.perf_counter()
            try:
                preds = seg.forward({"utterances": utts})
            except Exception as e:
                miss += 1
                print(f"  miss {did}: {type(e).__name__}: {e}")
                continue
            lat.append((time.perf_counter() - t0) * 1000.0)
            # preds[i]=1 ⇒ 경계 AFTER utt i (0-based) ⇒ utt i+2 (1-based)
            # 가 새 segment 시작. preds[-1] 은 항상 0.
            pred_bs = [i + 2 for i, p in enumerate(preds[:-1]) if p == 1]
            pk, wd = pk_wd(pred_bs, gold, n)
            f1 = boundary_set_f1(pred_bs, gold)
            pk_v.append(pk); wd_v.append(wd); f1_v.append(f1)
            n_pred_total += len(pred_bs); n_gold_total += len(gold)

        pk_m = float(np.mean(pk_v)) if pk_v else float("nan")
        wd_m = float(np.mean(wd_v)) if wd_v else float("nan")
        f1_m = float(np.mean(f1_v)) if f1_v else float("nan")
        score = 0.5 * f1_m + 0.25 * (1 - pk_m) + 0.25 * (1 - wd_m)
        st = latency_stats(lat)
        rows.append(dict(ds=ds, n_dial=len(full), pk=pk_m, wd=wd_m,
                         f1=f1_m, score=score, n_pred=n_pred_total,
                         n_gold=n_gold_total, miss=miss, **st))
        r = rows[-1]
        print(f"  Pk={pk_m:.4f} WD={wd_m:.4f} F1={f1_m:.4f} Score={score:.4f}"
              f" | lat/dial mean={st['mean']:.1f}ms | miss={miss}")

    L = [
        "# CSM **offline** (SuperDialseg `CSMSegmenter`, whole-dialogue)",
        "",
        f"backbone={args.backbone} max_utt_len={args.max_utterance_len} "
        f"cut_rate={args.cut_rate} ckpt={args.ckpt_path or '(none)'} "
        f"device={args.device}",
        "데이터=Def-DTS 번들 test, metric=segeval Pk/WD + boundary-set F1, "
        "Score=0.5F1+0.25(1-Pk)+0.25(1-WD).",
        "online 판 (methods/CSM/online/delay2.py) 과 동일 harness.",
        "",
        "| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/dial(ms) | pred bs | gold bs | miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n_dial']} | {r['pk']:.4f} | "
                 f"{r['wd']:.4f} | {r['f1']:.4f} | {r['score']:.4f} | "
                 f"{r['mean']:.1f} | {r['n_pred']} | {r['n_gold']} | "
                 f"{r['miss']} |")
    L += ["",
          "## 한계",
          "- offline = 대화 전체(미래 포함). online 판은 "
          "`methods/CSM/online/delay2.py` (delay-2, AUXILIARY).",
          "- ckpt 미지정 시 BERT-NSP pretrained 가중치만 사용 → 점수 낮음. "
          "동작 검증용.",
          "- SuperDialseg `CSMSegmenter` 의 `CoherenceScoringModel` 은 "
          "lxing532 `CoherenceNet` 와 다른 모듈 구조 — ckpt 호환 주의."]
    (exp_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"\nreport → {exp_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
