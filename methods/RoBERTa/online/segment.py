#!/usr/bin/env python3
"""Supervised RoBERTa dialogue segmenter — **online (strict causal)**.

``methods/RoBERTa/offline/train.py`` 로 학습한 supervised RoBERTa 분절기를
**미래 발화를 전혀 보지 않는** online 스타일로 추론한다. 모델 가중치·입력
인코딩·metric 은 offline 코드를 그대로 재사용(import) — 바뀌는 건 *추론
방식* 하나뿐이다.

offline vs online — 무엇이 바뀌나
---------------------------------
- **offline** (`train.py` ``eval_set``): 발화 ``g`` 의 경계를 ``g`` 를 포함
  하는 *모든* stride-1 윈도우(미래로 최대 19 발화까지 뻗는 윈도우 포함)의
  logit 평균으로 결정. → 미래 발화 관측.
- **online (본 파일)**: 경계 "turn ``t-1`` 와 ``t`` 사이" 는 **turn ``t`` 가
  도착한 시점에** 단 한 번 결정한다. 그 시점의 입력 = 직전 ≤``W`` 발화
  ``u_{t-W+1..t}`` (현재 turn ``t`` 까지, 그 *이후*는 미관측). 모델은 이
  causal 윈도우에서 발화 ``t-1`` 의 첫 ``</s>`` 위치를 분류한다. 발화
  ``t-1`` 은 윈도우의 끝에서 두 번째 — 학습 시 라벨이 살아있던(non-last)
  위치라 모델이 익숙한 자리. **look-ahead 0, 재수정 없음, turn 당 윈도우
  1개(O(1)/turn).**

즉 경계 ``(t-1, t)`` 는 turn ``t`` 가 와야만 결정 가능(turn ``t`` 가 새
topic 을 여는지는 ``t`` 가 존재해야 안다) — 이 결정에 미래(``t`` 이후)는
일절 쓰지 않으므로 strict causal 이다.

성능 해석 (구현 점검 기준)
--------------------------
offline 은 경계마다 ~20 윈도우를 평균, online 은 1 윈도우만 쓴다. 모델·
윈도우 단위 연산은 동일하므로 **성능이 크게 벌어지면 구현 문제**로 본다
(사용자 기준). offline 의 다중-윈도우 평균이 주는 smoothing 만큼의 소폭
차이는 정상.

실행::

    uv run python methods/RoBERTa/online/segment.py            # Run-1 체크포인트로 평가
    uv run python methods/RoBERTa/online/segment.py --model_dir <경로>

산출 = ``outputs/experiments/<name>/REPORT.md`` (online vs offline(Run-1) vs
논문 비교). 학습은 하지 않는다 — offline 의 학습된 가중치 그대로.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, RobertaForTokenClassification

REPO = Path(__file__).resolve().parents[3]

# --- offline 코드 재사용 (로직 분기 방지: 동일 함수 그대로 import) --------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from train import (encode_window, load_dialogs,  # noqa: E402
                   official_pk_wd, score_dialogs, PAPER_REF)

# offline run-1 test 결과 (outputs/runs/_misc/.../results.json, seed 42).
# 사용자 지정 "첫 번째 run" — online 비교 기준.
RUN1_OFFLINE = {
    "superseg":   dict(pk=0.1945, wd=0.2046, f1=0.8311, score=0.8158),
    "tiage":      dict(pk=0.4002, wd=0.4225, f1=0.4879, score=0.5383),
    "dialseg711": dict(pk=0.2728, wd=0.3395, f1=0.6417, score=0.6678),
}
# Run-1 학습 체크포인트 기본 경로 (사용자가 methods/RoBERTa/ 로 이동)
DEFAULT_MODEL_DIR = (REPO / "methods" / "RoBERTa" / "_roberta_unzip"
                     / "roberta_seg_out" / "roberta_supervised" / "model")


@torch.no_grad()
def eval_online(model, tok, dialogs, cfg, device) -> dict:
    """Strict-causal online 추론.

    경계 ``(t-1, t)`` 를 turn ``t`` 도착 시점의 causal 윈도우
    ``u_{max(0,t-W+1)..t}`` 만으로 결정 (미래 미관측). 윈도우 안에서 발화
    ``t-1`` 의 ``</s>`` 분류 logit 의 argmax. 한 dialogue 의 윈도우들은
    속도를 위해 한 번에 forward 하지만 *각 윈도우는 독립적으로 causal* —
    turn-by-turn 스트리밍과 결과가 동일하다 (turn 당 윈도우 1개).
    """
    model.eval()
    W, pad = cfg["sliding_window"], tok.pad_token_id
    preds = []
    for utts, _ in dialogs:
        n = len(utts)
        yp = [0] * n
        win_ids, meta = [], []
        for t in range(1, n):                       # 경계 (t-1, t), turn t 에서 결정
            ws = max(0, t - W + 1)                  # causal 윈도우 시작
            wu = utts[ws:t + 1]                     # u_{ws..t} — 미래(>t) 없음
            ids, _, cls_pos = encode_window(
                tok, wu, [0] * len(wu), "test",
                cfg["max_utt_len"], cfg["max_seq_len"])
            local = (t - 1) - ws                    # 발화 t-1 의 윈도우 내 index
            win_ids.append(torch.tensor(ids))
            meta.append((t, cls_pos[local]))
        if win_ids:
            batch = pad_sequence(win_ids, batch_first=True,
                                 padding_value=pad).to(device)
            logits = model(input_ids=batch,
                            attention_mask=batch.ne(pad).long()).logits.cpu().numpy()
            for w, (t, j) in enumerate(meta):
                yp[t - 1] = int(logits[w, j].argmax())
        yp[-1] = 0                                  # 마지막 발화는 경계 아님
        preds.append(yp)
    return score_dialogs(dialogs, preds)


def write_report(exp_dir: Path, cfg: dict, results: dict, lat_ms: float) -> None:
    L = [
        "# Supervised RoBERTa (SuperDialseg) — online (strict causal)",
        "",
        "`methods/RoBERTa/online/segment.py` 산출. `methods/RoBERTa/offline/"
        "train.py` 학습 체크포인트를 **미래 발화 미관측** online 스타일로 추론.",
        "",
        "## 1. 설정",
        "",
        f"- **체크포인트**: `{cfg['model_dir']}` (offline Run-1, seed 42, "
        "재학습 없음).",
        "- **추론**: strict-causal. 경계 (t-1, t) 를 turn t 도착 시점의 causal "
        f"윈도우 `u_{{max(0,t-{cfg['sliding_window']}+1)..t}}` 만으로 1회 결정 "
        "— look-ahead 0, 재수정 없음, turn 당 윈도우 1개.",
        "- **offline 대비 유일한 차이**: offline 은 발화별 경계를 ~20 stride-1 "
        "윈도우(미래 최대 19발화 포함) logit 평균으로 결정. online 은 미래를 "
        "안 보는 단일 causal 윈도우. 모델·입력 인코딩·metric 동일 (offline "
        "코드 import 재사용).",
        f"- **데이터**: `benchmarks/superdialseg_data/` 의 tiage/dialseg711/"
        "superseg test. metric = official Pk/WD (window=평균 segment 길이/2) "
        "+ binary F1, `Score=(2·F1+(1−Pk)+(1−WD))/4`.",
        f"- 평균 turn 추론 latency ≈ {lat_ms:.2f} ms (윈도우 1개 forward/turn).",
        "",
        "## 2. 결과 — online vs offline(Run-1) vs 논문",
        "",
        "| 평가셋 | online Pk↓ | online WD↓ | online F1↑ | **online Score↑** "
        "| offline Score | ΔScore (on−off) | 논문 Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = []
    for n in ("superseg", "tiage", "dialseg711"):
        r, off = results[n], RUN1_OFFLINE[n]
        d = r["score"] - off["score"]
        deltas.append(d)
        L.append(f"| {n} | {r['pk']:.4f} | {r['wd']:.4f} | {r['f1']:.4f} "
                 f"| **{r['score']:.4f}** | {off['score']:.4f} | {d:+.4f} "
                 f"| {PAPER_REF[n]['score']:.3f} |")
    on_m = float(np.mean([results[n]["score"] for n in results]))
    off_m = float(np.mean([RUN1_OFFLINE[n]["score"] for n in RUN1_OFFLINE]))
    L.append(f"| **mean-3** |  |  |  | **{on_m:.4f}** | {off_m:.4f} "
             f"| {on_m-off_m:+.4f} |  |")
    maxgap = max(abs(d) for d in deltas)
    L += [
        "",
        "## 3. 해석 / 판정",
        "",
        f"- online mean-3 Score = **{on_m:.4f}** vs offline(Run-1) {off_m:.4f} "
        f"→ ΔScore {on_m-off_m:+.4f}. 최대 벤치별 |Δ| = {maxgap:.4f}.",
        "- offline 은 경계마다 ~20 윈도우 평균(미래 포함), online 은 미래를 "
        "안 보는 단일 윈도우 — 그 차이만큼의 소폭 하락은 정상. **큰 하락이면 "
        "구현 문제** (사용자 기준).",
        "- online 은 발화 t-1 을 윈도우 끝에서 두 번째(학습 시 라벨 살아있던 "
        "non-last 위치)로 분류 → 모델 학습 분포와 정합. 구조적 mismatch 없음.",
        "",
        "## 4. 한계 / 검증 미해결",
        "",
        "- 단일 seed(Run-1) 체크포인트. offline 자체가 run 간 변동 있음 "
        "(특히 dialseg711) — 비교는 *동일 가중치* 기준 추론 차이만 격리.",
        "- offline Score 는 Run-1 (`outputs/runs/_misc/...results.json`) 값 "
        "하드코딩. 다른 체크포인트로 바꾸면 offline 기준도 갱신 필요.",
        "- per-turn latency 는 윈도우 1회 forward 기준 — batch 평균 측정값, "
        "엄밀한 단건 측정 아님.",
        "",
    ]
    (exp_dir / "REPORT.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised RoBERTa segmenter (online, strict causal)")
    ap.add_argument("--name", default="2026-05-23_roberta_online")
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR),
                    help="offline 학습 체크포인트 경로 (default = Run-1)")
    ap.add_argument("--sliding_window", type=int, default=20)
    ap.add_argument("--max_utt_len", type=int, default=25)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"[error] 체크포인트 없음: {model_dir}\n"
                         "  먼저 methods/RoBERTa/offline/train.py 로 학습하거나 "
                         "--model_dir 로 경로 지정.")
    device = torch.device(args.device)
    print(f"[device] {device} | checkpoint {model_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = RobertaForTokenClassification.from_pretrained(str(model_dir)).to(device)
    cfg = dict(model_dir=str(model_dir), sliding_window=args.sliding_window,
               max_utt_len=args.max_utt_len, max_seq_len=args.max_seq_len)

    results = {}
    lat = []
    for ds in ("superseg", "tiage", "dialseg711"):
        dialogs = load_dialogs(ds, "test")
        n_turns = sum(len(u) for u, _ in dialogs)
        t0 = time.perf_counter()
        results[ds] = eval_online(model, tok, dialogs, cfg, device)
        lat.append((time.perf_counter() - t0) * 1000.0 / max(1, n_turns))
        r, off = results[ds], RUN1_OFFLINE[ds]
        print(f"  {ds:11s}: Score={r['score']:.4f} (Pk={r['pk']:.4f} "
              f"F1={r['f1']:.4f})  offline={off['score']:.4f}  "
              f"Δ={r['score']-off['score']:+.4f}", flush=True)

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results.json").write_text(json.dumps(
        dict(config=cfg, test=results, offline_run1=RUN1_OFFLINE,
             paper_ref=PAPER_REF), indent=2))
    write_report(exp_dir, cfg, results, float(np.mean(lat)))
    print(f"\nDONE -> {exp_dir/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
