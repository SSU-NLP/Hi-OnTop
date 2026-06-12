#!/usr/bin/env python3
r"""v4.3.2 train: frozen Sentence-T5 + NextEmbedHeadMLP on DailyDialog (원본 zip).

데이터 소스 — **원본 `ijcnlp_dailydialog.zip` (yanran.li)** 사용. HF
``daily_dialog`` loading script 의 trust_remote_code 의존 / format 변동
회피. ``colab_csm_train.ipynb`` [2] 셀의 zip 탐색 + 재귀 추출 패턴 그대로.

Pipeline:
  1. PROJECT_ROOT 에서 ``ijcnlp_dailydialog*.zip`` 자동 탐색 + 재귀 추출
     (이미 ``<repo>/data/train/dailydialog/dialogues_text.txt`` 있으면 skip).
     - Colab: setup_colab.ipynb 의 업로드 셀 또는 `!cp /content/*.zip <repo>/`
     - 로컬:  `<repo>/ijcnlp_dailydialog.zip` 에 두면 자동 탐지
  2. ``dialogues_text.txt`` 파싱 (``__eou__`` 구분자). split 별 파일
     (`dialogues_train/validation/test.txt`) 이 있으면 그걸 사용, 없으면
     전체 corpus 를 90 / 5 / 5 split (seed=0).
  3. Frozen Sentence-T5 (`sentence-transformers/sentence-t5-base`) 로
     모든 utterance 임베딩 1회 캐시
     → ``outputs/runs/_misc/ijcnlp_dailydialog_emb_sentence-t5-base.pkl``.
  4. 캐시에서 (context=s_{t-m..t-1}, target=s_t) 쌍 in-memory 구성.
     short prefix 는 zero-pad. 첫 turn (t=0) 제외 (context 0).
  5. NextEmbedHeadMLP (m=5, hidden=1024, out_dim=768, L2-norm) 학습.
     Loss = mean ``1 - cos(\hat{s}, s)``. AdamW, linear warmup.
  6. valid loss 기준 best epoch 의 ckpt 저장
     → ``outputs/runs/_misc/next_embed_head_<tag>.pt``.

사용 예:
    # 로컬: <repo>/ijcnlp_dailydialog.zip 두고
    uv run python scripts/train_next_embed_head.py \\
        --epochs 10 --batch-size 256 --lr 1e-3 \\
        --tag st5_m5_mlp1024

    # smoke (1000 dialog 만)
    uv run python scripts/train_next_embed_head.py --smoke --epochs 2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from hi_ontop.next_embed_head import NextEmbedHeadMLP, make_head  # noqa: E402

CACHE = REPO / "outputs" / "runs" / "_misc"
DD_DIR = REPO / "data" / "train" / "dailydialog"
EXTRACT_DIR = REPO / "dd_extract"
ENC_NAME = "sentence-transformers/sentence-t5-base"
DEFAULT_M = 5
DEFAULT_HIDDEN = 1024
EOU = "__eou__"

SPLIT_FILE_NAMES = {
    "train": ("dialogues_train.txt", "train/dialogues_train.txt"),
    "validation": ("dialogues_validation.txt", "validation/dialogues_validation.txt"),
    "test": ("dialogues_test.txt", "test/dialogues_test.txt"),
}


# ----------------------------------------------------------------------
# Data — ijcnlp_dailydialog 원본 zip (colab_csm_train.ipynb [2] 패턴)
# ----------------------------------------------------------------------

def _parse_dd_file(path: Path) -> list[list[str]]:
    """Parse ``dialogues_*.txt`` (각 line = 한 dialog, ``__eou__`` 구분).

    빈 utterance 와 길이 < 2 dialog 제거. utterance 는 strip.
    """
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = [u.strip() for u in line.split(EOU)]
            parts = [u for u in parts if u]
            if len(parts) >= 2:
                out.append(parts)
    return out


def _ensure_dd_extracted() -> None:
    """``ijcnlp_dailydialog*.zip`` 자동 탐색 + 재귀 추출 (DD_DIR 준비)."""
    import glob
    import shutil
    import zipfile

    DD_DIR.mkdir(parents=True, exist_ok=True)
    needed_any = [DD_DIR / "dialogues_text.txt"] + [
        DD_DIR / fn[0] for fn in SPLIT_FILE_NAMES.values()
    ]
    if any(p.exists() and p.stat().st_size > 0 for p in needed_any):
        return  # 이미 추출/배치됨

    patterns = [
        str(REPO / "**" / "ijcnlp_dailydialog*.zip"),
        str(REPO / "**" / "*dailydialog*.zip"),
        str(REPO / "*.zip"),
    ]
    raw = []
    for pat in patterns:
        raw.extend(glob.glob(pat, recursive=True))
    cand = []
    for p in dict.fromkeys(raw):
        if zipfile.is_zipfile(p):
            cand.append(p)
        else:
            print(f"[skip] invalid zip (HTML/손상): {p}")

    if not cand:
        raise SystemExit(
            f"!! ijcnlp_dailydialog*.zip 을 {REPO}/ 또는 그 하위에 두세요.\n"
            f"   원본 (yanran.li): http://yanran.li/files/ijcnlp_dailydialog.zip\n"
            f"   Colab: setup_colab.ipynb 에서 업로드 → repo root 로 복사.\n"
            f"   추출 결과는 {EXTRACT_DIR}/ 와 {DD_DIR}/ 에 배치됩니다."
        )

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {cand[0]} → {EXTRACT_DIR}")
    with zipfile.ZipFile(cand[0]) as z:
        z.extractall(EXTRACT_DIR)
    # 안쪽에 또 zip (train.zip/validation.zip/test.zip) 이 있을 수 있음
    for inner in glob.glob(str(EXTRACT_DIR / "**" / "*.zip"), recursive=True):
        if zipfile.is_zipfile(inner):
            target = Path(inner).parent
            print(f"[extract] (inner) {inner} → {target}")
            with zipfile.ZipFile(inner) as z:
                z.extractall(target)

    # 필요한 파일들을 DD_DIR 로 복사
    wanted = ["dialogues_text.txt"]
    for split_files in SPLIT_FILE_NAMES.values():
        wanted.extend(split_files)
    for name in wanted:
        hits = sorted(
            glob.glob(str(EXTRACT_DIR / "**" / Path(name).name), recursive=True),
            key=lambda p: (("train" in p) + ("valid" in p) + ("test" in p), len(p)),
        )
        for h in hits:
            dest = DD_DIR / Path(name).name
            if dest.exists() and dest.stat().st_size > 0:
                break
            shutil.copy(h, dest)
            print(f"[place] {Path(name).name} ← {h}")
            break


def load_dailydialog(seed: int = 0) -> dict[str, list[list[str]]]:
    """Return ``{split: List[List[str]]}`` for train/validation/test.

    Priority:
      1. ``DD_DIR/dialogues_<split>.txt`` (공식 split 파일)
      2. ``DD_DIR/dialogues_text.txt`` 전체 → 90/5/5 random split (seed=0)
    """
    _ensure_dd_extracted()

    out = {}
    have_official = all(
        (DD_DIR / fn[0]).exists() for fn in SPLIT_FILE_NAMES.values()
    )
    if have_official:
        for split, (fname, _) in SPLIT_FILE_NAMES.items():
            out[split] = _parse_dd_file(DD_DIR / fname)
        print(f"[data] official split files used  "
              f"(train={len(out['train'])}, valid={len(out['validation'])}, "
              f"test={len(out['test'])})")
        return out

    all_path = DD_DIR / "dialogues_text.txt"
    if not all_path.exists():
        raise SystemExit(f"!! 추출 후에도 {all_path} 없음. zip 내용 확인 필요.")
    all_dialogs = _parse_dd_file(all_path)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_dialogs))
    n = len(all_dialogs)
    n_va = max(1, n // 20)
    n_te = max(1, n // 20)
    n_tr = n - n_va - n_te
    out["train"] = [all_dialogs[i] for i in idx[:n_tr]]
    out["validation"] = [all_dialogs[i] for i in idx[n_tr:n_tr + n_va]]
    out["test"] = [all_dialogs[i] for i in idx[n_tr + n_va:]]
    print(f"[data] split 파일 없음 → 90/5/5 random split (seed={seed})  "
          f"(train={n_tr}, valid={n_va}, test={n_te})")
    return out


def encode_all(dialogs: list[list[str]], encoder, batch: int = 128):
    """모든 utterance 1회 encode. Returns list[np.ndarray (n_turns, dim)]."""
    out = []
    flat = []
    sizes = []
    for d in dialogs:
        flat.extend(d)
        sizes.append(len(d))
    # sentence-transformers SentenceTransformer.encode supports list
    embs = encoder._model.encode(
        flat,
        normalize_embeddings=True,
        batch_size=batch,
        show_progress_bar=True,
    )
    embs = np.asarray(embs, dtype=np.float32)
    i = 0
    for n in sizes:
        out.append(embs[i:i + n])
        i += n
    return out


def cache_or_encode(split_dialogs: dict[str, list[list[str]]], device: str):
    cache_path = CACHE / f"ijcnlp_dailydialog_emb_{ENC_NAME.split('/')[-1]}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
        if all(s in data for s in split_dialogs):
            print(f"[cache] reuse {cache_path.name}")
            return data
    print(f"[encode] {ENC_NAME} (device={device})")
    from hi_ontop.embedding import QueryEncoder
    encoder = QueryEncoder(device=device, model_name=ENC_NAME)
    data = {}
    for split, dialogs in split_dialogs.items():
        print(f"  [encode] {split}: n_dial={len(dialogs)}, "
              f"n_utt={sum(len(d) for d in dialogs)}")
        data[split] = encode_all(dialogs, encoder)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(data, fh)
    print(f"  → {cache_path}")
    return data


def build_pairs(embs_per_dialog: list[np.ndarray], m: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ctx, tgt)`` arrays shape ``(N, m, d)`` / ``(N, d)``.

    For each dialog of length n, generate t=1..n-1 pairs. ``ctx[i]`` =
    embeddings of u_{max(0,t-m)..t-1} with zero left-pad. ``tgt[i]`` =
    embedding of u_t.
    """
    ctx_list = []
    tgt_list = []
    for emb in embs_per_dialog:
        n, d = emb.shape
        if n < 2:
            continue
        for t in range(1, n):
            start = max(0, t - m)
            win = emb[start:t]  # (k, d), k = t - start ≤ m
            k = win.shape[0]
            if k < m:
                pad = np.zeros((m - k, d), dtype=emb.dtype)
                win = np.concatenate([pad, win], axis=0)  # (m, d)
            ctx_list.append(win)
            tgt_list.append(emb[t])
    ctx = np.stack(ctx_list, axis=0)  # (N, m, d)
    tgt = np.stack(tgt_list, axis=0)  # (N, d)
    return ctx, tgt


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def make_loader(ctx: np.ndarray, tgt: np.ndarray, batch_size: int, shuffle: bool):
    """L2-normalize target as guard (cosine loss 가정). ctx 는 원본 유지."""
    ctx_t = torch.from_numpy(ctx).float()
    tgt_t = torch.from_numpy(tgt).float()
    tgt_t = F.normalize(tgt_t, p=2, dim=-1, eps=1e-12)
    ds = TensorDataset(ctx_t, tgt_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ----------------------------------------------------------------------
# Diagnostic thresholds (codex 권고 근거 + 사용자 검토 2026-05-21)
# 4-tier: SUCCESS / ACCEPTABLE / RETRY / FAIL.
# 'ACCEPTABLE' = 경고는 아니지만 downstream segmentation 통과해야 신뢰.
# ----------------------------------------------------------------------
THR = {
    # gain_vs_mean: mean-baseline (= constant predictor) 대비 이득.
    # codex success ≥ +0.035, FAIL ≤ 0. 그 사이는 RETRY/ACCEPTABLE 회색지대.
    "gain_mean_success": 0.035,
    "gain_mean_acceptable": 0.020,
    "gain_mean_retry": 0.0,
    # gain_vs_last: identity (s_{t-1}) 대비 이득.
    # ≤ 0 이면 last-utterance copy 만 외움 = corpus prior. FAIL.
    # ≥ +0.005 면 head 가 의미 추가.
    "gain_last_fail": 0.0,
    "gain_last_acceptable": 0.005,
    # gain_vs_mean_full: zero-pad 없는 full-context subset 의 gain.
    # codex success ≥ +0.025. 0 이면 5-turn 능력 0 = FAIL.
    "gain_full_success": 0.025,
    "gain_full_acceptable": 0.010,
    "gain_full_retry": 0.0,
    # delta_std: per-sample δ_model 의 std. 작으면 PE channel 정보량 0.
    # codex success ≥ 0.045. < 0.02 면 거의 constant.
    "delta_std_success": 0.045,
    "delta_std_acceptable": 0.030,
    "delta_std_retry": 0.020,
    # pred_norm_mean: L2-norm 적용 후 1.0 ± 0.005 안이어야 함. 밖이면 코드 버그.
    "pred_norm_lo": 0.995,
    "pred_norm_hi": 1.005,
}


def classify_diag(diag: dict[str, float]) -> tuple[str, list[str]]:
    """Return (overall_status, list_of_warnings).

    overall_status ∈ {SUCCESS, ACCEPTABLE, RETRY, FAIL, CRITICAL}.
    CRITICAL = pred_norm 이탈 (normalize 코드 버그 의심).
    FAIL = mean/last/full 중 하나가 0 이하, 또는 delta_std < retry 임계.
    RETRY = 0 이상이지만 ACCEPTABLE 미달.
    ACCEPTABLE = success 임계 미달이지만 RETRY 보다 위.
    SUCCESS = 4가지 모두 success 임계 통과.
    """
    pn = diag.get("pred_norm_mean", 1.0)
    if not (THR["pred_norm_lo"] <= pn <= THR["pred_norm_hi"]):
        return "CRITICAL", [f"pred_norm={pn:.4f} 이탈 (L2-norm 코드 의심)"]

    warns = []
    fail = False
    g_mean = diag.get("gain_vs_mean", 0.0)
    g_last = diag.get("gain_vs_last", 0.0)
    g_full = diag.get("gain_vs_mean_full", 0.0)
    d_std = diag.get("delta_std", 0.0)

    if g_mean <= THR["gain_mean_retry"]:
        warns.append(f"MEAN-COLLAPSE (gain_mean={g_mean:+.4f}≤0)"); fail = True
    if g_last <= THR["gain_last_fail"]:
        warns.append(f"LAST-IDENT (gain_last={g_last:+.4f}≤0)"); fail = True
    if g_full <= THR["gain_full_retry"]:
        warns.append(f"FULL-CTX-COLLAPSE (full_gain={g_full:+.4f}≤0)"); fail = True
    if d_std < THR["delta_std_retry"]:
        warns.append(f"low δ_std ({d_std:.4f}<{THR['delta_std_retry']})"); fail = True
    if fail:
        return "FAIL", warns

    # all positive — grade
    is_success = (
        g_mean >= THR["gain_mean_success"]
        and g_full >= THR["gain_full_success"]
        and g_last >= THR["gain_last_acceptable"]
        and d_std >= THR["delta_std_success"]
    )
    if is_success:
        return "SUCCESS", []

    is_retry = (
        g_mean < THR["gain_mean_acceptable"]
        or g_full < THR["gain_full_acceptable"]
        or d_std < THR["delta_std_acceptable"]
    )
    status = "RETRY" if is_retry else "ACCEPTABLE"
    # ACCEPTABLE 도 어느 metric 이 미달인지 표시
    notes = []
    if g_mean < THR["gain_mean_success"]:
        notes.append(f"gain_mean={g_mean:+.4f}<{THR['gain_mean_success']}")
    if g_full < THR["gain_full_success"]:
        notes.append(f"full_gain={g_full:+.4f}<{THR['gain_full_success']}")
    if d_std < THR["delta_std_success"]:
        notes.append(f"δ_std={d_std:.4f}<{THR['delta_std_success']}")
    return status, notes


def mixed_loss(pred: torch.Tensor, target: torch.Tensor,
               nce_weight: float = 0.0, temperature: float = 0.07) -> torch.Tensor:
    """cosine regression + (optional) batch-InfoNCE.

    nce_weight=0 → pure cosine regression (codex 권고: collapse 확인 전 default).
    """
    reg = NextEmbedHeadMLP.cosine_loss(pred, target)
    if nce_weight <= 0.0:
        return reg
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    nce = F.cross_entropy(logits, labels)
    return reg + nce_weight * nce


def train_one_epoch(model, loader, optimizer, scheduler, device,
                    nce_weight: float = 0.0):
    model.train()
    losses = []
    for ctx, tgt in loader:
        ctx = ctx.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        pred = model(ctx)
        loss = mixed_loss(pred, tgt, nce_weight=nce_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_diagnostics(model, loader, device) -> dict[str, float]:
    """Collapse 진단 metric (codex 권고: regression-to-mean + pad shortcut + corpus prior).

    Baselines:
        mean: 모든 target 의 평균 방향만 예측 (가장 약한 baseline)
        last: s_{t-1} (ctx 의 마지막 timestep) 을 그대로 예측 — corpus prior 회피 검증.
              head 가 next predictor 가 아니라 'identity ≈ last-utt' 만 외우는지 확인.
    Subset:
        full-context (t ≥ m, ctx 의 첫 timestep 이 zero 가 아닌 sample) —
        zero-pad shortcut 의 영향 없이 head 의 실제 m-turn 능력 측정.
    """
    model.eval()
    preds, tgts, ctxs = [], [], []
    for ctx, tgt in loader:
        ctx_d = ctx.to(device, non_blocking=True)
        tgt_d = tgt.to(device, non_blocking=True)
        preds.append(model(ctx_d).cpu())
        tgts.append(tgt_d.cpu())
        ctxs.append(ctx.cpu() if ctx.device.type != "cpu" else ctx)
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    ctx_all = torch.cat(ctxs, dim=0)  # (N, m, d)

    cos = (pred * tgt).sum(dim=-1)
    deltas = 1.0 - cos

    mean_tgt = F.normalize(tgt.mean(0, keepdim=True), p=2, dim=-1, eps=1e-12)
    mean_baseline_deltas = 1.0 - (mean_tgt * tgt).sum(dim=-1)

    last_utt = F.normalize(ctx_all[:, -1, :], p=2, dim=-1, eps=1e-12)
    last_baseline_deltas = 1.0 - (last_utt * tgt).sum(dim=-1)

    # padded: ctx 의 first timestep 이 zero (build_pairs 의 left zero-pad)
    is_padded = ctx_all[:, 0, :].abs().sum(dim=-1) < 1e-6
    full_mask = ~is_padded

    def _m(t):
        return float(t.mean()) if t.numel() > 0 else float("nan")

    def _s(t):
        return float(t.std(unbiased=False)) if t.numel() > 1 else float("nan")

    return {
        "loss": _m(deltas),
        "mean_baseline_loss": _m(mean_baseline_deltas),
        "gain_vs_mean": _m(mean_baseline_deltas) - _m(deltas),
        "last_baseline_loss": _m(last_baseline_deltas),
        "gain_vs_last": _m(last_baseline_deltas) - _m(deltas),
        "delta_std": _s(deltas),
        "pred_norm_mean": float(pred.norm(dim=-1).mean()),
        "n_padded": int(is_padded.sum().item()),
        "n_full": int(full_mask.sum().item()),
        "loss_full": _m(deltas[full_mask]),
        "mean_baseline_loss_full": _m(mean_baseline_deltas[full_mask]),
        "gain_vs_mean_full": _m(mean_baseline_deltas[full_mask]) - _m(deltas[full_mask]),
        "delta_std_full": _s(deltas[full_mask]),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--m", type=int, default=DEFAULT_M)
    ap.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true",
                    help="첫 1000 dialog 만 (학습 sanity)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None,
                    help="ckpt filename suffix. default = st5_m{m}_mlp{hidden}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nce-weight", type=float, default=0.0,
                    help="batch-InfoNCE auxiliary weight. 0 = pure cosine "
                         "regression. codex 권고: collapse 진단 후 조건부 "
                         "(예: 0.02~0.05) 도입.")
    ap.add_argument("--nce-temp", type=float, default=0.07,
                    help="InfoNCE temperature (only used if --nce-weight > 0)")
    ap.add_argument("--patience", type=int, default=7,
                    help="early stop: valid loss 가 N epoch 동안 개선 안 되면 중단. "
                         "0 = 비활성 (전체 epoch 다 돌림).")
    ap.add_argument("--head-type", default="mlp", choices=["mlp", "transformer"],
                    help="next-embedding head architecture. transformer = 1-layer "
                         "encoder + learned pos emb + pad-mask + mean pool.")
    ap.add_argument("--n-heads", type=int, default=8,
                    help="transformer attention heads (only used if --head-type=transformer)")
    ap.add_argument("--n-layers", type=int, default=1,
                    help="transformer encoder layers")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="transformer dropout")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v4.3.2 train] device={device}, m={args.m}, hidden={args.hidden_dim}, "
          f"lr={args.lr}, bs={args.batch_size}, epochs={args.epochs}, smoke={args.smoke}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Data
    splits = load_dailydialog()
    if args.smoke:
        for k in splits:
            splits[k] = splits[k][:1000]
    for k, v in splits.items():
        print(f"  [data] {k}: n_dial={len(v)}")

    embs = cache_or_encode(splits, device=device)

    # 2. Pairs
    print(f"[pairs] m={args.m}")
    ctx_tr, tgt_tr = build_pairs(embs["train"], args.m)
    ctx_va, tgt_va = build_pairs(embs["validation"], args.m)
    print(f"  train: {ctx_tr.shape[0]} pairs   valid: {ctx_va.shape[0]} pairs")

    train_loader = make_loader(ctx_tr, tgt_tr, args.batch_size, shuffle=True)
    valid_loader = make_loader(ctx_va, tgt_va, args.batch_size, shuffle=False)

    # 3. Model + optim
    model = make_head(
        args.head_type,
        emb_dim=ctx_tr.shape[-1],
        context_window=args.m,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] head_type={args.head_type}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)

    def lr_lambda(step):
        # codex 권고: 첫 step lr=0 회피 (step+1)/warmup
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Train loop + early stopping
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    tag = args.tag or f"st5_m{args.m}_mlp{args.hidden_dim}"
    ckpt_path = CACHE / f"next_embed_head_{tag}.pt"
    CACHE.mkdir(parents=True, exist_ok=True)

    history = []
    stopped_early = False
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                                  nce_weight=args.nce_weight)
        diag = eval_diagnostics(model, valid_loader, device)
        va_loss = diag["loss"]
        dt = time.perf_counter() - t0
        history.append({"epoch": ep, "train_loss": tr_loss, "valid_loss": va_loss,
                        "valid_mean_baseline_loss": diag["mean_baseline_loss"],
                        "valid_gain_vs_mean": diag["gain_vs_mean"],
                        "valid_delta_std": diag["delta_std"],
                        "valid_pred_norm_mean": diag["pred_norm_mean"],
                        "wall_s": dt})
        marker = ""
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = ep
            patience_counter = 0
            marker = "  ★ saved"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "head_type": args.head_type,
                        "emb_dim": model.emb_dim,
                        "context_window": model.context_window,
                        "hidden_dim": getattr(model, "hidden_dim", args.hidden_dim),
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout,
                        "encoder": ENC_NAME,
                    },
                    "best_epoch": ep,
                    "best_valid_loss": va_loss,
                    "best_valid_diag": diag,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
        # 4-tier 분류 (SUCCESS/ACCEPTABLE/RETRY/FAIL/CRITICAL) — 명시 임계값 기반
        status, notes = classify_diag(diag)
        tag_str = f"[{status}]"
        note_str = ("  " + " | ".join(notes)) if notes else ""
        print(f"[ep {ep:02d}] tr={tr_loss:.4f}  va={va_loss:.4f}  "
              f"gain_mean={diag['gain_vs_mean']:+.4f}  "
              f"gain_last={diag['gain_vs_last']:+.4f}  "
              f"full_gain={diag['gain_vs_mean_full']:+.4f}  "
              f"δ_std={diag['delta_std']:.4f}  "
              f"pred_norm={diag['pred_norm_mean']:.3f}  "
              f"(n_pad={diag['n_padded']}, n_full={diag['n_full']}, {dt:.1f}s)"
              f"{marker} {tag_str}{note_str}")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            stopped_early = True
            print(f"\n[early stop] valid 가 {args.patience} epoch 동안 개선 없음 "
                  f"(best ep={best_epoch}, valid={best_val:.4f}). 학습 중단.")
            break

    # 5. Test (load best)
    if stopped_early:
        print(f"[summary] stopped at ep {ep+1}/{args.epochs} (early stop)")
    print(f"[best] epoch={best_epoch}, valid={best_val:.4f}")
    ctx_te, tgt_te = build_pairs(embs["test"], args.m)
    test_loader = make_loader(ctx_te, tgt_te, args.batch_size, shuffle=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    te_diag = eval_diagnostics(model, test_loader, device)
    te_status, te_notes = classify_diag(te_diag)
    te_note_str = ("  " + " | ".join(te_notes)) if te_notes else ""
    print(f"[test] loss={te_diag['loss']:.4f}  "
          f"gain_mean={te_diag['gain_vs_mean']:+.4f}  "
          f"gain_last={te_diag['gain_vs_last']:+.4f}  "
          f"full_gain={te_diag['gain_vs_mean_full']:+.4f}  "
          f"δ_std={te_diag['delta_std']:.4f}  "
          f"(n_pad={te_diag['n_padded']}, n_full={te_diag['n_full']})  "
          f"[{te_status}]{te_note_str}  (DailyDialog test)")
    print()
    print("=" * 70)
    print("⚠ 학습 진단은 *필요조건*. 진짜 판정은 v4.3.2 segmentation sweep")
    print("  (`precompute_v432_delta.py` + `run_v432_smoke.py`) 결과로.")
    print("  학습 진단 ✓ + seg sweep ✗  → negative result (정직 보고, contrastive")
    print("  켜지 말고 head 자체가 task misaligned 로 판정).")
    print("=" * 70)

    # 6. dump history
    hist_path = ckpt_path.with_suffix(".history.json")
    hist_path.write_text(json.dumps({
        "history": history,
        "best_epoch": best_epoch,
        "best_valid": best_val,
        "test_diag": te_diag,
        "ckpt": str(ckpt_path),
        "config": ckpt["config"],
        "args": vars(args),
    }, indent=2))
    print(f"[ok] ckpt={ckpt_path.name}  history={hist_path.name}")


if __name__ == "__main__":
    main()
