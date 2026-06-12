#!/usr/bin/env python3
"""Supervised RoBERTa dialogue segmenter — **offline**, 논문 충실 재현.

Coldog2333/SuperDialseg (EMNLP 2023, ``2023.emnlp-main.249``) 의 Table 3
``RoBERTa`` (plain supervised) 분절기를, 논문 Appendix A + repo
``data_collator.py`` (`__getitem_input__` roberta 분기) 그대로 학습·평가한다.

**offline** — supervised, 학습 필요, 슬라이딩 윈도우 안에서 발화 ``i+1`` 이후를
관측. Hi-OnTop 의 online segmenter (Hi-OnTop 등) 와 비교 시 비대칭임을 명시할 것.

repo 기반 실행 (Hi-OnTop repo clone 후 그대로):

    uv run python methods/RoBERTa/offline/train.py                 # 본 재현 (superseg/train)
    uv run python methods/RoBERTa/offline/train.py --limit 200 --epochs 2   # smoke
    uv run python methods/RoBERTa/offline/train.py --train_ds tiage --epochs 40

데이터 = ``benchmarks/superdialseg_data/{ds}/segmentation_file_{split}.json``
(repo 내장). 산출 = ``outputs/experiments/<name>/{REPORT.md, results.json}`` +
best 체크포인트 ``model/`` (gitignored). metric = 논문 5.3 의 official
SuperDialseg Pk/WD (sliding window = 평균 segment 길이의 절반) + binary F1,
``Score = (2*F1 + (1-Pk) + (1-WD)) / 4``.

재현 스펙 (논문 Appendix A.1/A.2):
- 입력: ``<s> u1 </s></s> u2 </s></s> ... uN </s>`` — 발화당 BPE 23 토큰
  (max_utterance_len 25 − 2), 마지막 ``</s>`` 제거, max_seq_len 512. 각 발화
  뒤 **첫 ``</s>``** 위치에서 token-classification.
- 모델: ``RobertaForTokenClassification`` (num_labels=2). plain — da/role
  출력 헤드(MT)·입력 임베딩(MV) 없음.
- 슬라이딩 윈도우 ``|T|=20`` 발화, stride 1. 학습은 임의 19-발화 윈도우,
  추론은 stride-1 전체 윈도우 → 발화별 logit 평균 (집계는 논문/repo 미명시
  → 본 재현의 명시적 결정).
- 학습: AdamW, lr=1e-5, batch=8, weight_decay=1e-3, grad_clip=1.0,
  20 epochs (SuperDialseg) / 40 (TIAGE), early stopping (val Score 가
  patience=epochs//2 epoch 미개선 시 중단, best 보존). LR 스케줄러 미사용.

한계: byte 재현 불가 — 학습 코드/seed 미공개(논문은 다회 평균),
데이터가 ``superseg-v2`` 개정판(논문 Table 2 와 dialogue 수 상이). 자세히는
``outputs/experiments/2026-05-23_roberta_supervised/REPORT.md`` 참조.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from nltk.metrics import pk as _nltk_pk
from nltk.metrics import windowdiff as _nltk_wd
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, RobertaForTokenClassification

REPO = Path(__file__).resolve().parents[3]
SDS = REPO / "benchmarks" / "superdialseg_data"
IGNORE = -100

# 논문 Table 3 (SuperDialseg 학습) RoBERTa 보고치 — 재현 목표
PAPER_REF = {
    "superseg":   dict(pk=0.185, wd=0.192, f1=0.784, score=0.798),
    "tiage":      dict(pk=0.401, wd=0.443, f1=0.373, score=0.482),
    "dialseg711": dict(pk=0.241, wd=0.272, f1=0.660, score=0.702),
}


# --------------------------------------------------------------------------
# data + metric
# --------------------------------------------------------------------------

def load_dialogs(ds: str, split: str) -> list[tuple[list[str], list[int]]]:
    """SuperDialseg JSON -> [(utterances, segmentation_labels)]. yt[-1]=0 강제."""
    path = SDS / ds / f"segmentation_file_{split}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    arr = raw["dial_data"][list(raw["dial_data"])[0]]
    out = []
    for d in arr:
        utts = [t["utterance"] for t in d["turns"]]
        yt = [int(t.get("segmentation_label", 0)) for t in d["turns"]]
        if yt:
            yt[-1] = 0
        if len(utts) >= 2:
            out.append((utts, yt))
    return out


def official_pk_wd(yt: list[int], yp: list[int]) -> tuple[float, float]:
    """논문 5.3: sliding window = 평균 segment 길이의 절반."""
    n_seg = sum(yt) + 1
    k = max(2, int(round(len(yt) / n_seg / 2)))
    ts, ps = "".join(map(str, yt)), "".join(map(str, yp))
    return float(_nltk_pk(ts, ps, k=k)), float(_nltk_wd(ts, ps, k=k))


def score_dialogs(dialogs, preds: list[list[int]]) -> dict:
    pks, wds, g, p = [], [], [], []
    for (_, yt), yp in zip(dialogs, preds):
        pk, wd = official_pk_wd(yt, yp)
        pks.append(pk); wds.append(wd); g += yt; p += yp
    f1 = float(f1_score(g, p, zero_division=0))
    pk_m, wd_m = float(np.mean(pks)), float(np.mean(wds))
    return dict(pk=pk_m, wd=wd_m, f1=f1,
                score=(2 * f1 + (1 - pk_m) + (1 - wd_m)) / 4)


# --------------------------------------------------------------------------
# 입력 구성 — data_collator.py __getitem_input__ roberta 분기 재현
# --------------------------------------------------------------------------

def encode_window(tok, utts, labels, mode, max_utt_len, max_seq_len):
    """<s> u1 </s></s> u2 </s></s> ... uN </s>  (마지막 </s> 제거).
    첫 </s> 위치(classification)에 발화 라벨, 그 외 -100.
    반환: (input_ids, token_labels, cls_positions)."""
    input_tokens = [tok.cls_token]                  # '<s>'
    cmask = [0]
    for u in utts:
        ut = tok.tokenize(u)[:max_utt_len - 2]       # 25-2 = 23 (use_mask=False)
        input_tokens += ut + ["</s>", "</s>"]
        cmask += [0] * len(ut) + [1, 0]              # 첫 </s> 에서 분류
    input_tokens = input_tokens[:-1]                 # 마지막 </s> 제거
    cmask = cmask[:-1]

    input_ids = tok.convert_tokens_to_ids(input_tokens)[:max_seq_len]
    cmask = cmask[:max_seq_len]

    lab = list(labels)
    if mode == "train" and lab:
        lab[-1] = IGNORE                             # 윈도우 마지막 발화 제외

    token_labels = [IGNORE] * len(input_ids)
    cls_pos: list[int] = []
    i = 0
    for j, m in enumerate(cmask):
        if m:
            if i < len(lab):
                token_labels[j] = lab[i]
            cls_pos.append(j)
            i += 1
    return input_ids, token_labels, cls_pos


class TrainDS(Dataset):
    """__getitem__ 마다 임의 윈도우 (repo train 분기 재현)."""

    def __init__(self, dialogs, tok, cfg):
        self.dialogs, self.tok, self.cfg = dialogs, tok, cfg

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        utts, yt = self.dialogs[idx]
        W = self.cfg["sliding_window"]
        if len(utts) > W:
            s = random.randint(0, len(utts) - W)
            utts, yt = utts[s:s + W - 1], yt[s:s + W - 1]   # 19 발화
        ids, lab, _ = encode_window(self.tok, utts, yt, "train",
                                    self.cfg["max_utt_len"], self.cfg["max_seq_len"])
        return torch.tensor(ids), torch.tensor(lab)


def make_collate(pad_id):
    def collate(batch):
        ids, labs = zip(*batch)
        ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
        labs = pad_sequence(labs, batch_first=True, padding_value=IGNORE)
        return ids, ids.ne(pad_id).long(), labs
    return collate


@torch.no_grad()
def eval_set(model, tok, dialogs, cfg, device) -> dict:
    """stride-1 윈도우 전부 → 발화별 logit 평균 → argmax."""
    model.eval()
    W, pad = cfg["sliding_window"], tok.pad_token_id
    preds = []
    for utts, _ in dialogs:
        n = len(utts)
        logit_sum = np.zeros((n, 2), dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        win_ids, win_meta = [], []
        for ws in range(max(1, n - W + 1)):
            wu = utts[ws:ws + W]
            ids, _, cls_pos = encode_window(tok, wu, [0] * len(wu), "test",
                                            cfg["max_utt_len"], cfg["max_seq_len"])
            win_ids.append(torch.tensor(ids))
            win_meta.append((ws, cls_pos))
        batch = pad_sequence(win_ids, batch_first=True, padding_value=pad).to(device)
        logits = model(input_ids=batch,
                        attention_mask=batch.ne(pad).long()).logits.cpu().numpy()
        for w, (ws, cls_pos) in enumerate(win_meta):
            for u_local, j in enumerate(cls_pos):
                g = ws + u_local
                if g < n:
                    logit_sum[g] += logits[w, j]
                    cnt[g] += 1
        cnt[cnt == 0] = 1.0
        yp = list((logit_sum / cnt[:, None]).argmax(1))
        yp[-1] = 0                                   # 마지막 발화는 경계 아님
        preds.append([int(x) for x in yp])
    return score_dialogs(dialogs, preds)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def train(cfg: dict) -> dict:
    device = torch.device(cfg["device"])
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    print(f"[device] {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    tr = cfg["train_ds"]
    train_d = load_dialogs(tr, "train")
    val_d = load_dialogs(tr, "validation")
    if cfg["limit"]:
        train_d = train_d[:cfg["limit"]]
        val_d = val_d[:max(1, cfg["limit"] // 4)]
    test_sets = {ds: load_dialogs(ds, "test")
                 for ds in ("tiage", "dialseg711", "superseg")}
    print(f"[data] train {len(train_d)} / val {len(val_d)} dial (train_ds={tr})",
          flush=True)

    model = RobertaForTokenClassification.from_pretrained(
        cfg["backbone"], num_labels=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                              weight_decay=cfg["weight_decay"])
    loader = DataLoader(TrainDS(train_d, tok, cfg), batch_size=cfg["batch_size"],
                        shuffle=True, collate_fn=make_collate(tok.pad_token_id))

    exp_dir = REPO / "outputs" / "experiments" / cfg["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "model"
    patience = max(1, cfg["epochs"] // 2)
    best, since, history = -1.0, 0, []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.perf_counter(); tot = 0.0
        for ids, attn, labs in loader:
            out = model(input_ids=ids.to(device), attention_mask=attn.to(device),
                        labels=labs.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); optim.zero_grad()
            tot += out.loss.item()
        vr = eval_set(model, tok, val_d, cfg, device)
        history.append(dict(epoch=epoch, train_loss=tot / len(loader), val=vr))
        print(f"[epoch {epoch:>2}] loss={tot/len(loader):.4f} "
              f"val Score={vr['score']:.4f} (Pk={vr['pk']:.4f} F1={vr['f1']:.4f}) "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
        if vr["score"] > best:
            best, since = vr["score"], 0
            model.save_pretrained(model_dir); tok.save_pretrained(model_dir)
            print(f"   -> best 저장 (val Score={best:.4f})", flush=True)
        else:
            since += 1
            if since >= patience:
                print(f"   early stopping (patience {patience} epoch 미개선)",
                      flush=True)
                break

    print("\n[final] best 체크포인트로 test 평가", flush=True)
    best_model = RobertaForTokenClassification.from_pretrained(model_dir).to(device)
    results = {}
    for n, d in test_sets.items():
        r = eval_set(best_model, tok, d, cfg, device)
        results[n] = r
        ref = PAPER_REF.get(n, {})
        print(f"  {n:11s}: Pk={r['pk']:.4f} WD={r['wd']:.4f} F1={r['f1']:.4f} "
              f"Score={r['score']:.4f}  (논문 {ref.get('score')})", flush=True)

    out = dict(config=cfg, history=history, test=results, paper_ref=PAPER_REF)
    (exp_dir / "results.json").write_text(json.dumps(out, indent=2))
    write_report(exp_dir, cfg, history, results, len(train_d), len(val_d))
    print(f"\nDONE -> {exp_dir/'REPORT.md'}", flush=True)
    return results


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def write_report(exp_dir: Path, cfg, history, results, n_train, n_val) -> None:
    best_ep = max(history, key=lambda h: h["val"]["score"])["epoch"]
    L = [
        "# Supervised RoBERTa (SuperDialseg) — offline, 논문 충실 재현",
        "",
        "`methods/RoBERTa/offline/train.py` 산출. Coldog2333/SuperDialseg "
        "(EMNLP 2023) Table 3 `RoBERTa` (plain supervised) 재현.",
        "",
        "## 1. 실험 setup",
        "",
        f"- **모델**: `RobertaForTokenClassification` (`{cfg['backbone']}`, "
        "num_labels=2). 입력 `<s> u1 </s></s> … uN </s>`, 각 발화 뒤 첫 `</s>` "
        "에서 token-classification (논문 Appendix A + repo `data_collator.py`).",
        f"- **데이터**: `benchmarks/superdialseg_data/`. 학습 "
        f"`{cfg['train_ds']}/train` ({n_train} dial), model selection "
        f"`{cfg['train_ds']}/validation` ({n_val} dial) val Score, 평가 "
        "tiage/dialseg711/superseg test.",
        f"- **학습 HP**: AdamW lr={cfg['lr']}, batch={cfg['batch_size']}, "
        f"weight_decay={cfg['weight_decay']}, grad_clip=1.0, epochs="
        f"{cfg['epochs']}, early stopping(patience={max(1,cfg['epochs']//2)}, "
        f"best 보존), 슬라이딩 윈도우 |T|={cfg['sliding_window']} stride 1, "
        f"max_utt_len={cfg['max_utt_len']}, max_seq_len={cfg['max_seq_len']}, "
        f"seed={cfg['seed']} (단일 run)."
        + ("  ⚠ **smoke run** (`--limit` 사용 — 재현 아님)." if cfg["limit"] else ""),
        "- **metric**: official SuperDialseg Pk/WD (window=평균 segment 길이/2) "
        "+ binary F1, `Score=(2·F1+(1−Pk)+(1−WD))/4`.",
        "",
        "## 2. 결과 — test split vs 논문 Table 3 (`RoBERTa`)",
        "",
        "| 평가셋 | in-domain? | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | 논문 Score | ΔScore |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    deltas = []
    for n in ("superseg", "tiage", "dialseg711"):
        r = results[n]
        ref = PAPER_REF[n]
        d = r["score"] - ref["score"]
        deltas.append(d)
        indom = "✔" if n == cfg["train_ds"] else "✘ zero-shot"
        L.append(f"| {n} | {indom} | {r['pk']:.4f} | {r['wd']:.4f} | "
                 f"{r['f1']:.4f} | **{r['score']:.4f}** | {ref['score']:.3f} "
                 f"| {d:+.4f} |")
    mean_s = float(np.mean([results[n]["score"] for n in results]))
    mean_ref = float(np.mean([PAPER_REF[n]["score"] for n in PAPER_REF]))
    L.append(f"| **mean-3** |  |  |  |  | **{mean_s:.4f}** | {mean_ref:.3f} "
             f"| {mean_s-mean_ref:+.4f} |")
    L += [
        "",
        "## 3. 학습 곡선 (val = {}/validation Score)".format(cfg["train_ds"]),
        "",
        "| epoch | " + " | ".join(str(h["epoch"]) for h in history) + " |",
        "|---|" + "---:|" * len(history),
        "| train loss | " + " | ".join(f"{h['train_loss']:.3f}" for h in history) + " |",
        "| val Score | " + " | ".join(f"{h['val']['score']:.3f}" for h in history) + " |",
        "",
        f"best = epoch {best_ep} → test 평가는 그 체크포인트 사용.",
        "",
        "## 4. 판정",
        "",
        f"- mean-3 ΔScore = {mean_s-mean_ref:+.4f}. "
        + ("smoke run — 재현 판정 불가 (`--limit` 제거 후 본 런 필요)."
           if cfg["limit"] else
           "논문 RoBERTa 행 근방이면 방법 충실 재현 성공."),
        "- **Hi-OnTop 대비 비대칭**: 본 모델은 supervised·offline(윈도우 안 "
        "미래 발화 관측)·학습 필요. Hi-OnTop 는 무감독 online past-only. "
        "RoBERTa 우위는 supervised offline 상한 — online 무감독과 1:1 비교 아님.",
        "",
        "## 5. 한계 / 검증 미해결",
        "",
        "- **byte 재현 불가**: repo 에 학습 코드·체크포인트·seed 미공개 "
        "(`main.py` 평가 전용). 논문은 다회 평균. 단일 seed run.",
        "- **추론 윈도우 집계 미명시**: 겹치는 stride-1 윈도우의 발화별 예측을 "
        "logit 평균으로 통합 (논문/repo 미명시 → 본 재현의 결정).",
        "- **데이터 버전**: `superseg-v2` (개정판) — 논문 Table 2 의 dialogue "
        "수와 상이.",
        "- **plain RoBERTa 만**: `MVRoBERTa`(role/DA 입력 임베딩)·`RobertaMultiTask`"
        "(da/role 출력 헤드)는 미구현.",
        "- model 체크포인트는 `outputs/experiments/<name>/model/` (gitignored).",
        "",
    ]
    (exp_dir / "REPORT.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised RoBERTa segmenter (offline)")
    ap.add_argument("--name", default="2026-05-23_roberta_supervised")
    ap.add_argument("--backbone", default="roberta-base")
    ap.add_argument("--train_ds", default="superseg", choices=["superseg", "tiage"])
    ap.add_argument("--epochs", type=int, default=20,
                    help="논문: SuperDialseg 20 / TIAGE 40")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--sliding_window", type=int, default=20)
    ap.add_argument("--max_utt_len", type=int, default=25)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="학습 대화 N개로 제한 (0=전체; smoke 용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not SDS.exists():
        raise SystemExit(f"[error] SuperDialseg 데이터 없음: {SDS}")
    train(vars(args))


if __name__ == "__main__":
    main()
