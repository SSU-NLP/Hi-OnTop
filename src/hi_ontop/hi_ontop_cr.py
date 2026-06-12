"""Hi-OnTop-CR — de-neut + run-length 적응-β 신호 + commit-and-refine deploy.

**승격 main 분절 모델 (2026-06-11).** [[hi-ontop]] (δ_eff causal threshold) 의 후속.

신호 (universal, 두 도메인, calibration-free):
    de-neut 화제 변별거리 ``r_active = 1 − cos(deneut(x), deneut(m))``,
    ``deneut(v) = normalize(v − β·(v·g)·g)`` (global 중립성분 β 제거),
    ``β_t = clip(A − B·log(1+k/L0), 0, 1)`` (run-length k 적응; 짧은 seg→β→1 de-neut,
    긴 seg→β↓ V_rel), ``V = r_active − λ·r_global``, 적응 임계치 ``μ + c·σ``.
    → axis-1 에서 oracle 천장이 LLM 초과(AMI 0.687>0.543), DTS superseg 벽 돌파. LOO 검증.

Deploy reset (`reset=`, **default ``threshold``**):
- ``threshold`` (default, 0-lag): detected-reset hard reset — 경계를 **다음 턴 즉시 emit**(버퍼 없음).
    Score 0.372. sharp-seam(DTS)도 이게 맞음(commit-refine 은 짧은 segment 억제로 회귀).
- ``commit_refine`` (drift / AMI, 버퍼 옵션): hard reset 을 bounded-lag commit-and-refine 으로 교체.
    V>θ 는 split 후보 생성만; shock 으로 armed → 윈도우 후보 split 위치별 gain
    ``SSE(a..t)−[SSE(a..b−1)+SSE(b..t)]`` 의 **argmax b\* 에 경계 emit**(위치 정정), 확정 시 왼쪽 prototype 을
    b\* 오른쪽으로 reset. Score 0.401(±0.029) — **단 lag L=8≈26s 매입분**(L≤3 은 threshold 와 동급).

검증/한계 (정직): commit_refine AMI deploy Score 0.401 / ±2F1 0.140 (hard-reset 0.372/0.131 대비
+0.029/+0.009). **그러나 oracle 천장 ±2F1 0.554 와 격차는 미해결** — online 운영의 구조적 한계
(놓친 경계가 prototype 을 두 topic 으로 오염, 발화 단위 복구 불가; reset 기법 5각도 v1~v5 모두
격차 못 메움). 정식 config (c=1.0, L=8, A=2.0, B=1.0, m_min=2): c=calibration-free, L=8=AMI k-fold
CV 검증(5-fold L\*=8 만장일치), (A,B)=LOO 검증.
상세: `outputs/experiments/2026-06-11_ami_commit_refine/REPORT.md`, decision-log 2026-06-11, [[hi-ontop-cr]].
"""

from __future__ import annotations

import math

import numpy as np

# 정식 (검증된) 기본 config — 신호 + deploy
DEFAULTS = dict(c=1.0, A=2.0, B=1.0, L0=8, lam=0.6, g_rho=0.15, rho_min=0.05,
                R=4, warmup=8, L=8, m_min=2, Mc=0.0)


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
            rho = max(rho_min, 1.0 / (k + 1)); m = _nr((1 - rho) * m + rho * x); k += 1
        g = _nr((1 - g_rho) * g + g_rho * x)
    return pred


def _seg_commit_refine(e, c, A, B, L0, lam, g_rho, rho_min, R, warmup, L, m_min, Mc, **_):
    """drift (AMI) deploy — bounded-lag commit-and-refine + local split-gain b* refinement.

    shock(V>θ)으로 armed → 오른쪽이 m_min 차거나 윈도우 L 소진 시, 윈도우 후보 split 위치 b 별
    gain 의 argmax b\* 에 경계 emit. 확정 시 왼쪽 prototype 을 b\* 오른쪽으로 reset.
    """
    n = len(e); g = e[0].copy()
    Wm = 0.0; WM2 = 0.0; Wn = 0; m = e[0].copy(); k = 1
    pref = [np.zeros_like(e[0]), e[0].copy()]
    pred = []; last = -999; armed_at = -1
    for t in range(1, n):
        x = e[t]; beta = _beta(k, A, B, L0)
        mc = _deneut(m, g, beta); xc = _deneut(x, g, beta)
        V = (1 - float(xc @ mc)) - lam * (1 - float(x @ g))
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= warmup else None
        pref.append(pref[-1] + x)
        seglen = len(pref) - 1
        if armed_at < 0 and thr is not None and V > thr and k >= R and t - last >= R:
            armed_at = t
        committed = False
        if armed_at >= 0:
            right_len_now = t - armed_at + 1
            window_done = right_len_now >= L
            if right_len_now >= m_min or window_done:
                a = t - seglen + 1
                lo = max(a + 1, armed_at - (L - 1)); hi = t - m_min + 1
                SSE = lambda i, j: (j - i + 1) - float(((pref[j - a + 1] - pref[i - a]) ** 2).sum()) / (j - i + 1)
                tot = SSE(a, t); best_b = -1; best_g = -1e9; second = -1e9
                for b in range(lo, hi + 1):
                    gn = tot - (SSE(a, b - 1) + SSE(b, t))
                    if gn > best_g:
                        second = best_g; best_g = gn; best_b = b
                    elif gn > second:
                        second = gn
                ok = (best_b > 0 and (best_g - max(second, 0.0)) >= Mc)
                if ok or window_done:
                    if ok:
                        pred.append(best_b); last = best_b
                        seg_r = [e[i] for i in range(best_b, t + 1)]
                        k = 0; pref = [np.zeros_like(e[0])]
                        for z in seg_r:
                            pref.append(pref[-1] + z); k += 1
                        m = pref[-1] / (np.linalg.norm(pref[-1]) + 1e-9)
                        Wm = 0.0; WM2 = 0.0; Wn = 0; committed = True
                    armed_at = -1
        if not committed and armed_at < 0:
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            rho = max(rho_min, 1.0 / (k + 1)); m = _nr((1 - rho) * m + rho * x); k += 1
        elif not committed:
            k += 1
        g = _nr((1 - g_rho) * g + g_rho * x)
    return pred


def segment(emb, reset: str = "threshold", **overrides) -> list[int]:
    """Hi-OnTop-CR online 분절 — 경계 turn index 리스트.

    Args:
        emb: (n, d) 발화 임베딩 (단위 정규화 가정). online·0-look-ahead·무학습.
        reset: **default ``threshold``** = 즉시 reset, **0-lag(다음 턴 emit)**, Score 0.372.
            ``commit_refine`` = bounded-lag(L=8≈26s) 로 +0.029 Score(0.401) 사는 옵션 — 버퍼 허용 시만.
            (commit_refine 의 우위는 전적으로 lag 매입분: L≤3 에선 threshold 와 동급.)
        **overrides: ``DEFAULTS`` 의 HP override (c/A/B/L0/lam/g_rho/rho_min/R/warmup/L/m_min/Mc).

    Returns:
        경계로 판정된 turn index 의 정렬 리스트(0<i<n).
    """
    e = np.asarray(emb, dtype=np.float64)
    cfg = {**DEFAULTS, **overrides}
    if reset == "commit_refine":
        pred = _seg_commit_refine(e, **cfg)
    elif reset == "threshold":
        pred = _seg_threshold(e, **cfg)
    else:
        raise ValueError(f"reset must be 'commit_refine' or 'threshold', got {reset!r}")
    return sorted(p for p in set(pred) if 0 < p < len(e))
