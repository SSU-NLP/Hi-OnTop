"""Hi-OnTop-DeNeut — de-neut + run-length 적응-β 신호 + threshold(0-lag) deploy.

[[hi-ontop]] (δ_eff causal threshold) 의 후속. **drift/AMI 도메인용** 신호.
(DTS 는 δ_eff `class HiOnTop` 가 우세 — [[HANDOFF_04]] / decision-log 2026-06-13.)

신호 (universal, calibration-free):
    de-neut 화제 변별거리 ``r_active = 1 − cos(deneut(x), deneut(m))``,
    ``deneut(v) = normalize(v − β·(v·g)·g)`` (global 중립성분 β 제거),
    ``β_t = clip(A − B·log(1+k/L0), 0, 1)`` (run-length k 적응; 짧은 seg→β→1 de-neut,
    긴 seg→β↓ V_rel), ``V = r_active − λ·r_global``, 적응 임계치 ``μ + c·σ``.
    → AMI 정상정렬 oracle ±2F1 de-neut 0.370 > δ_eff 0.225 (`scripts/ami_alignment_recheck.py`).

Deploy = **`threshold` (0-lag) 단독 지원** — 경계를 다음 턴 즉시 emit(버퍼 없음).
detected-reset hard reset(V>θ → 즉시 prototype reset). AMI Score 0.372.

[DEPRECATED 2026-06-13] ``commit_refine`` (bounded-lag commit-and-refine) 는 **폐기**됨.
우위(AMI 0.401 vs 0.372)가 전적으로 lag(L=8≈26s) 매입분이고 0-lag 요구와 충돌(decision-log 2026-06-12).
참고/재현용으로 ``archive/legacy_commit_refine/seg_commit_refine.py`` 에 보존(decision-log 2026-06-13).

검증/한계 (정직): threshold AMI deploy Score 0.372 / ±2F1 0.131 ≪ **oracle 천장 ±2F1 0.554** —
online 운영의 구조적 한계(놓친 경계가 prototype 을 두 topic 으로 오염, 발화 단위 복구 불가).
정식 config (c=1.0, A=2.0, B=1.0): c=calibration-free, (A,B)=LOO 검증.
"""

from __future__ import annotations

import math

import numpy as np

# 정식 (검증된) 기본 config — 신호 + threshold deploy
DEFAULTS = dict(c=1.0, A=2.0, B=1.0, L0=8, lam=0.6, g_rho=0.15, rho_min=0.05,
                R=4, warmup=8)


def _nr(v: np.ndarray) -> np.ndarray:
    """단위벡터 정규화."""
    return v / (np.linalg.norm(v) + 1e-12)


def _deneut(x: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    """global(중립) 성분을 β 만큼 제거한 단위벡터 — '화제의 변별적 방향'."""
    xc = x - beta * float(x @ g) * g
    return xc / (np.linalg.norm(xc) + 1e-9)


def _beta(k: int, A: float, B: float, L0: int) -> float:
    """run-length 적응 β = clip(A − B·log(1+k/L0), 0, 1)."""
    return min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)


def _seg_threshold(e, c, A, B, L0, lam, g_rho, rho_min, R, warmup, **_):
    """sharp-seam (DTS) deploy — detected-reset hard reset. de-neut V + 적응 β + μ+cσ."""
    n = len(e); m = e[0].copy(); k = 1; g = e[0].copy()
    Wm = 0.0; WM2 = 0.0; Wn = 0; pred = []; last = -999
    for t in range(1, n):
        x = e[t]; beta = _beta(k, A, B, L0)
        mc = _deneut(m, g, beta); xc = _deneut(x, g, beta)
        V = (1 - float(xc @ mc)) - lam * (1 - float(x @ g))
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= warmup else None
        if thr is not None and V > thr and k >= R and t - last >= R:
            pred.append(t); m = x.copy(); k = 1; last = t
        else:
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            # NOTE(2026-06-15): deploy 는 매-step 정규화. clean 신호/oracle 측정에선 raw-EWMA(사용시점만
            # 정규화) 필수 — 매-step 정규화하면 oracle ±2F1 0.69→0.32 붕괴(버그). 단 deploy 는 reset 오염이
            # 지배해 raw-EWMA 로 바꿔도 0.14→0.16 (미미). raw-EWMA 전환 = 소폭 open win, methodology §1 참조.
            rho = max(rho_min, 1.0 / (k + 1)); m = _nr((1 - rho) * m + rho * x); k += 1
        g = _nr((1 - g_rho) * g + g_rho * x)
    return pred


def segment(emb, **overrides) -> list[int]:
    """Hi-OnTop-DeNeut online 분절 (threshold deploy 단독) — 경계 turn index 리스트.

    de-neut + 적응 β 신호 V + 적응 임계치 μ+cσ, detected-reset hard reset (0-lag,
    다음 턴 즉시 emit). [DEPRECATED] commit_refine 는 폐기 — `archive/legacy_commit_refine/`.

    Args:
        emb: (n, d) 발화 임베딩 (단위 정규화 가정). online·0-look-ahead·무학습.
        **overrides: ``DEFAULTS`` 의 HP override (c/A/B/L0/lam/g_rho/rho_min/R/warmup).

    Returns:
        경계로 판정된 turn index 의 정렬 리스트(0<i<n).
    """
    e = np.asarray(emb, dtype=np.float64)
    cfg = {**DEFAULTS, **overrides}
    pred = _seg_threshold(e, **cfg)
    return sorted(p for p in set(pred) if 0 < p < len(e))
