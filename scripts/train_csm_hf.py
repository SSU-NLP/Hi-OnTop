"""CSM (lxing532/Dialogue-Topic-Segmenter) training via HuggingFace Trainer.

원본 notebook (`colab_csm_train.ipynb`) 의 `[2e]` 셀이 가진 custom 학습 루프
(step-resume + heartbeat + linear LR + AdamW) 를 HF `Trainer` 기반으로 재작성한다.
알고리즘 / 데이터 / loss / optimizer 수식은 그대로:

- model     : `external/Dialogue-Topic-Segmenter/model_utils.CoherenceNet`
              (bert encoder + 768→768→2 decoder, softmax 2-way)
- loss      : marginal ranking loss across (pos, neg1, neg2) — margin=1
- data      : `external/.../data_utils.UtteranceDataset` (NSP-style triplets,
              padding='max_length', max_length=128)
- optimizer : AdamW lr=2e-5 eps=1e-8 + linear schedule (warmup=0)

변경되는 건 runner 뿐. resume 은 Trainer 표준 `--resume-from-checkpoint` 사용 —
"auto" 로 주면 `output_dir` 안 최신 checkpoint 자동 로드 (optimizer / scheduler /
RNG state / global_step 모두 복원).

평가 호환: 학습 종료 시 `cpt_<gstep>.pth` 를 `output_dir` 에 저장 (순수
CoherenceNet state_dict). 노트북 `[4]` 의 `segment.py -m CM -e <path.pth>` 가
그대로 strict load.

사용 예:
    uv run python scripts/train_csm_hf.py \\
        --repo-dir external/Dialogue-Topic-Segmenter \\
        --encoder bert-base-uncased \\
        --output-dir checkpoints_batch64 \\
        --num-train-epochs 10 \\
        --per-device-train-batch-size 32 \\
        --learning-rate 2e-5 --margin 1.0 \\
        --save-steps 1000 --logging-steps 50

    # resume (중단된 학습 이어서)
    uv run python scripts/train_csm_hf.py ... --resume-from-checkpoint auto
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset


logger = logging.getLogger("train_csm_hf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CSM coherence model with HF Trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data / model ---
    p.add_argument(
        "--repo-dir",
        default=_env("CSM_REPO_DIR", "external/Dialogue-Topic-Segmenter"),
        help="외부 레포 경로 (CoherenceNet / UtteranceDataset import 용).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="DailyDialog 폴더 (dialogues_text/topic/act.txt). default = <repo-dir>/data/train/dailydialog.",
    )
    p.add_argument(
        "--encoder",
        default=_env("CSM_ENCODER", "bert-base-uncased"),
        help="backbone encoder (HF model id). README default = bert-base-uncased.",
    )
    p.add_argument("--max-length", type=int, default=128, help="tokenizer max_length")

    # --- HF TrainingArguments 직결 ---
    p.add_argument(
        "--output-dir",
        default=_env("CSM_CKPT_DIR", "checkpoints"),
        help="HF Trainer output_dir (체크포인트 + trainer_state).",
    )
    p.add_argument(
        "--num-train-epochs",
        type=float,
        default=float(_env("CSM_EPOCHS", "10") or 10),
    )
    p.add_argument("--max-steps", type=int, default=-1, help="설정 시 num-train-epochs 무시.")
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=int(_env("CSM_BATCH", "32") or 32),
    )
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--adam-epsilon", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--lr-scheduler-type", default="linear")
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--dataloader-num-workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument(
        "--report-to",
        default="none",
        help='HF report integrations. default "none" = 노트북 호환. wandb 쓰려면 "wandb".',
    )
    p.add_argument("--run-name", default=None, help="wandb / trainer 표시용 run name.")

    # --- resume ---
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help='checkpoint 경로 또는 "auto" (output_dir 안 latest 자동 탐색).',
    )

    # --- loss HP ---
    p.add_argument(
        "--margin",
        type=float,
        default=float(_env("CSM_MARGIN", "1.0") or 1.0),
        help="marginal ranking loss margin (논문 default = 1).",
    )

    # --- eval (default OFF — 노트북과 동일) ---
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="0.0 = 전체 train (노트북 default). >0 이면 그 비율을 random split 으로 val.",
    )
    p.add_argument(
        "--evaluation-strategy",
        default="no",
        choices=["no", "steps", "epoch"],
        help="HF eval strategy.",
    )
    p.add_argument("--eval-steps", type=int, default=1000)

    # --- 부가 ---
    p.add_argument(
        "--save-eval-pth",
        action="store_true",
        default=True,
        help="학습 후 segment.py 호환 cpt_<gstep>.pth 저장 (default True).",
    )
    p.add_argument(
        "--no-save-eval-pth",
        action="store_false",
        dest="save_eval_pth",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# External repo import
# ---------------------------------------------------------------------------

def _add_repo_to_path(repo_dir: Path) -> None:
    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"--repo-dir 가 없음: {repo_dir}. notebook [1] 셀이 clone 하는 위치 — "
            "git clone https://github.com/lxing532/Dialogue-Topic-Segmenter.git 로 받으세요."
        )
    repo_str = str(repo_dir.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def marginal_ranking_loss(out: torch.Tensor, margin: float) -> torch.Tensor:
    """`out`: [B, 3, 2] softmax probs. col 0 = "coherent" prob."""
    bt = out[:, :, 0]  # [B, 3]
    l1 = F.relu(margin - (bt[:, 0] - bt[:, 1]))
    l2 = F.relu(margin - (bt[:, 0] - bt[:, 2]))
    l3 = F.relu(margin - (bt[:, 1] - bt[:, 2]))
    return torch.mean((l1 + l2 + l3) / 3.0)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class CoherenceNetTrainable(nn.Module):
    """HF Trainer 호환 wrapper.

    - `self.core` = 원본 `CoherenceNet` (state_dict 키 1:1 보존 → eval-compat).
    - forward 는 collator 가 만든 9개 텐서 (3 view × 3 keys) 를 받아 한 번에 BERT
      를 통과시킨다 (분포 동등 batched forward).
    """

    def __init__(self, encoder_name: str, margin: float):
        super().__init__()
        from model_utils import CoherenceNet  # noqa: WPS433  (external repo)
        from transformers import AutoModel

        bert = AutoModel.from_pretrained(encoder_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.core = CoherenceNet(bert, device)
        self.margin = margin

    def forward(  # noqa: PLR0913 — Trainer 호환 위해 9개 텐서를 직접 받음
        self,
        prev_input_ids: torch.Tensor,
        prev_attention_mask: torch.Tensor,
        prev_token_type_ids: torch.Tensor,
        curr_input_ids: torch.Tensor,
        curr_attention_mask: torch.Tensor,
        curr_token_type_ids: torch.Tensor,
        next_input_ids: torch.Tensor,
        next_attention_mask: torch.Tensor,
        next_token_type_ids: torch.Tensor,
        labels: torch.Tensor | None = None,  # noqa: ARG002 — collator 호환용 placeholder
    ) -> dict[str, torch.Tensor]:
        bsz = prev_input_ids.size(0)
        all_input_ids = torch.cat([prev_input_ids, curr_input_ids, next_input_ids], dim=0)
        all_attn = torch.cat(
            [prev_attention_mask, curr_attention_mask, next_attention_mask], dim=0,
        )
        all_token_type = torch.cat(
            [prev_token_type_ids, curr_token_type_ids, next_token_type_ids], dim=0,
        )

        bert_out = self.core.bert(
            input_ids=all_input_ids,
            attention_mask=all_attn,
            token_type_ids=all_token_type,
        )
        h = bert_out.last_hidden_state[:, 0, :]  # [3B, 768]
        dec = self.core.coherence_decoder(h)  # [3B, 2]
        sm = F.softmax(dec, dim=-1)
        logits = sm.view(3, bsz, 2).permute(1, 0, 2).contiguous()  # [B, 3, 2]

        loss = marginal_ranking_loss(logits, self.margin)
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class CSMCollator:
    """`UtteranceDataset.__getitem__` 결과 (list[3] of dict, 각 dict 의 tensor
    shape `[1, max_length]`) 를 받아 Trainer 가 기대하는 dict[str → Tensor[B, L]] 로
    변환.
    """

    def __call__(self, batch: list[list[dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for r_idx, view in enumerate(("prev", "curr", "next")):
            view_dicts = [item[r_idx] for item in batch]
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                tensors = [d[key] for d in view_dicts]
                # UtteranceDataset 는 padding='max_length' → 모두 동일 길이
                stacked = torch.cat([t if t.dim() == 2 else t.unsqueeze(0) for t in tensors], dim=0)
                out[f"{view}_{key}"] = stacked
        # Trainer 가 prediction_step 에서 labels 키를 찾으므로 더미 placeholder
        out["labels"] = torch.zeros(len(batch), dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_dotenv_if_present(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    for cand in (project_root / ".env", Path.home() / ".env"):
        if cand.is_file():
            load_dotenv(cand, override=False)
            logger.info(".env loaded: %s", cand)
            break


def _resolve_data_dir(args: argparse.Namespace, repo_dir: Path) -> Path:
    if args.data_dir:
        d = Path(args.data_dir).expanduser().resolve()
    else:
        d = (repo_dir / "data" / "train" / "dailydialog").resolve()
    needed = ["dialogues_text.txt", "dialogues_topic.txt", "dialogues_act.txt"]
    missing = [n for n in needed if not (d / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DailyDialog 원본 파일 누락 in {d}: {missing}. "
            "notebook [2]/[2b] 가 ijcnlp_dailydialog.zip 을 풀어 배치하는 위치 — "
            "수동으로 같은 위치에 두거나 notebook 의 [2] 셀을 먼저 실행하세요."
        )
    return d


def _resolve_resume(arg: str | None, output_dir: str) -> Any:
    """`--resume-from-checkpoint` 값 → Trainer.train() 인자로 변환."""
    if arg in (None, "", "none", "false", "False"):
        return None
    if arg in ("auto", "true", "True", "latest"):
        # True → Trainer 가 output_dir 안 latest 자동 탐색
        cks = sorted(Path(output_dir).glob("checkpoint-*"))
        if not cks:
            logger.warning(
                "--resume-from-checkpoint=%s 였지만 %s 안 checkpoint-* 없음. fresh 시작.",
                arg, output_dir,
            )
            return None
        return True
    return arg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv_if_present(project_root)

    args = parse_args()

    # repo / data 경로 해소
    repo_dir = Path(args.repo_dir).expanduser()
    if not repo_dir.is_absolute():
        repo_dir = (project_root / repo_dir).resolve()
    _add_repo_to_path(repo_dir)
    data_dir = _resolve_data_dir(args, repo_dir)

    # output_dir 상대경로 → repo_dir 기준 (.env 호환)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (repo_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    logger.info("repo_dir   = %s", repo_dir)
    logger.info("data_dir   = %s", data_dir)
    logger.info("output_dir = %s", output_dir)
    logger.info("encoder    = %s", args.encoder)
    logger.info("HP         = bs=%d epochs=%g lr=%g margin=%g seed=%d",
                args.per_device_train_batch_size, args.num_train_epochs,
                args.learning_rate, args.margin, args.seed)

    # --- imports that depend on sys.path / transformers version ---
    from transformers import (  # noqa: WPS433
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from data_utils import UtteranceDataset  # noqa: WPS433  (external repo)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    full_ds = UtteranceDataset(
        str(data_dir / "dialogues_text.txt"),
        str(data_dir / "dialogues_topic.txt"),
        str(data_dir / "dialogues_act.txt"),
        tokenizer,
        max_length=args.max_length,
    )
    logger.info("dataset size = %d", len(full_ds))

    # split
    train_ds: Any = full_ds
    eval_ds: Any = None
    if args.val_frac > 0.0:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(full_ds), generator=g).tolist()
        n_val = max(1, int(len(full_ds) * args.val_frac))
        eval_ds = Subset(full_ds, perm[:n_val])
        train_ds = Subset(full_ds, perm[n_val:])
        logger.info("split: train=%d val=%d (val_frac=%.3f)",
                    len(train_ds), len(eval_ds), args.val_frac)

    # model
    model = CoherenceNetTrainable(encoder_name=args.encoder, margin=args.margin)

    # report_to: HF 는 list 또는 "none" / "all" / "wandb" 등 문자열 허용
    report_to = args.report_to
    if isinstance(report_to, str) and "," in report_to:
        report_to = [s.strip() for s in report_to.split(",") if s.strip()]

    # TrainingArguments — transformers 4.39.x 호환 (evaluation_strategy 사용)
    targs_kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        report_to=report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        label_names=["labels"],
        disable_tqdm=False,
    )
    # transformers 4.39.3 = evaluation_strategy; 4.46+ = eval_strategy.
    # 둘 다 시도해서 호환.
    try:
        targs = TrainingArguments(evaluation_strategy=args.evaluation_strategy,
                                  eval_steps=args.eval_steps, **targs_kwargs)
    except TypeError:
        targs = TrainingArguments(eval_strategy=args.evaluation_strategy,
                                  eval_steps=args.eval_steps, **targs_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CSMCollator(),
    )

    resume = _resolve_resume(args.resume_from_checkpoint, args.output_dir)
    if resume is True:
        logger.info("resume: auto — Trainer 가 %s 안 latest 탐색", args.output_dir)
    elif isinstance(resume, str):
        logger.info("resume: %s", resume)
    else:
        logger.info("resume: 없음 (fresh start)")

    trainer.train(resume_from_checkpoint=resume)

    # --- final save: HF format (Trainer 가 알아서) + eval-compat .pth ---
    trainer.save_state()
    if args.save_eval_pth:
        gstep = trainer.state.global_step
        pth_path = Path(args.output_dir) / f"cpt_{gstep}.pth"
        torch.save(model.core.state_dict(), pth_path)
        logger.info("eval-compat ckpt 저장: %s (gstep=%d)", pth_path, gstep)


if __name__ == "__main__":
    main()
