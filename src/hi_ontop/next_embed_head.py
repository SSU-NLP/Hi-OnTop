"""NextEmbedHead — frozen Sentence-T5 위에 학습하는 작은 next-embedding regressor.

v4.3.2-exp 의 δ_model 산출 구성 요소. DailyDialog 등 dialog corpus 에서
``(context = s_{t-m..t-1}, target = s_t)`` 쌍을 만들어 ``\\hat{s}_t =
head(s_{t-m..t-1})`` 를 학습. inference 시:

    δ_model(t) = 1 − cos(\\hat{s}_t, s_t)        # ST5 embedding-space cosine

Architecture (default = "small MLP"):

    input  = concat(s_{t-m..t-1}) ∈ R^{m·d}     (m=5, d=768 → 3840)
    h      = GELU(Linear(input → 1024))
    out    = Linear(h → d)
    \\hat{s}_t = L2-normalize(out)

Padding rule (stream start, t < m):
    부족한 prefix 는 0-벡터로 padding (mask 없음). 첫 t=0 은 context 가
    전혀 없으므로 head 호출 안 함 (caller 가 fallback).

Loss (training):
    L = 1 − cos(\\hat{s}_t, s_t)        # δ scale 과 자연 일치
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NextEmbedHeadTransformer(nn.Module):
    """Transformer encoder next-embedding head (1 layer, GELU FFN, learned pos emb).

    MLP head 의 zero-pad shortcut + concat 의 position 정보 부재를 해결할
    목적. 1 layer (codex 권고 — m=5 짧고 80k pair 작음, overfit 위험).

    Args:
        emb_dim: encoder embedding dim (Sentence-T5-base = 768).
        context_window: causal context length m (default 5).
        n_heads: multi-head attention heads (default 8).
        dim_feedforward: FFN hidden width (default 1024 — MLP head 와 비교 가능).
        n_layers: encoder layers (default 1).
        dropout: attention/FFN dropout (default 0.1).

    Forward:
        ctx: (B, m, d) or (B, m*d). Zero-padded prefix 는 attention mask
        로 ignore (concat MLP 의 pad-shortcut 해소). Mean pool over
        non-padded tokens → Linear → L2-norm.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        context_window: int = 5,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.context_window = context_window
        self.hidden_dim = dim_feedforward
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.pos_embed = nn.Parameter(torch.randn(context_window, emb_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.dim() == 2:
            ctx = ctx.view(-1, self.context_window, self.emb_dim)
        elif ctx.dim() != 3:
            raise ValueError(
                f"NextEmbedHeadTransformer expects (B,m,d) or (B,m*d), got {ctx.shape}"
            )
        # padding mask: True = padded position (ignore in attention)
        pad_mask = ctx.abs().sum(dim=-1) < 1e-6  # (B, m), bool
        x = ctx + self.pos_embed  # broadcast (B, m, d)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # mean pool over non-padded
        mask_f = (~pad_mask).float().unsqueeze(-1)  # (B, m, 1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        out = self.proj(pooled)
        return F.normalize(out, p=2, dim=-1, eps=1e-12)

    @staticmethod
    def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos = (pred * target).sum(dim=-1)
        return (1.0 - cos).mean()


def make_head(head_type: str, **kwargs) -> nn.Module:
    """Factory — 'mlp' / 'transformer'."""
    if head_type == "mlp":
        return NextEmbedHeadMLP(
            emb_dim=kwargs.get("emb_dim", 768),
            context_window=kwargs.get("context_window", 5),
            hidden_dim=kwargs.get("hidden_dim", 1024),
        )
    if head_type == "transformer":
        return NextEmbedHeadTransformer(
            emb_dim=kwargs.get("emb_dim", 768),
            context_window=kwargs.get("context_window", 5),
            n_heads=kwargs.get("n_heads", 8),
            dim_feedforward=kwargs.get("hidden_dim", 1024),
            n_layers=kwargs.get("n_layers", 1),
            dropout=kwargs.get("dropout", 0.1),
        )
    raise ValueError(f"unknown head_type={head_type!r}, expected 'mlp' or 'transformer'")


class NextEmbedHeadMLP(nn.Module):
    """Small MLP next-embedding head for frozen Sentence-T5.

    Args:
        emb_dim: encoder embedding dim (Sentence-T5-base = 768).
        context_window: causal context length m (default 5).
        hidden_dim: MLP hidden width (default 1024).

    Forward:
        Input ``ctx`` is either ``(B, m, d)`` or ``(B, m*d)``. Output is
        ``(B, d)`` L2-normalized predicted next embedding.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        context_window: int = 5,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.context_window = context_window
        self.hidden_dim = hidden_dim

        in_dim = emb_dim * context_window
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.dim() == 3:
            # (B, m, d) → (B, m*d)
            ctx = ctx.flatten(1, 2)
        elif ctx.dim() != 2:
            raise ValueError(
                f"NextEmbedHeadMLP expects (B,m,d) or (B,m*d), got {ctx.shape}"
            )
        out = self.net(ctx)
        return F.normalize(out, p=2, dim=-1, eps=1e-12)

    @staticmethod
    def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean ``1 − cos(pred, target)`` over the batch.

        Both ``pred`` and ``target`` should be L2-normalized along dim -1.
        """
        cos = (pred * target).sum(dim=-1)
        return (1.0 - cos).mean()


def pack_causal_window(
    history: list[torch.Tensor],
    context_window: int,
    emb_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Pack last ``context_window`` embeddings into ``(1, m, d)``.

    Pads with zero vectors on the *left* if ``history`` is shorter than
    ``context_window``. Returns ``None`` if ``history`` is empty (no
    context → caller should fallback, e.g. δ_model = 0).
    """
    if not history:
        return None
    win = history[-context_window:]
    pad_n = context_window - len(win)
    if pad_n > 0:
        pad = [torch.zeros(emb_dim, dtype=dtype, device=device) for _ in range(pad_n)]
        win = pad + list(win)
    # stack to (m, d) then (1, m, d)
    return torch.stack(win, dim=0).to(device=device, dtype=dtype).unsqueeze(0)
