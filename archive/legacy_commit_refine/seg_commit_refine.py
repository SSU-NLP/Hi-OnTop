"""[ARCHIVED 2026-06-13] Hi-OnTop-DeNeut 의 commit-and-refine deploy (폐기).

폐기 사유: 사용자 결정 — Hi-OnTop-DeNeut 은 **`threshold`(0-lag) deploy 만 지원**한다.
commit_refine 의 우위(AMI Score 0.401 vs threshold 0.372)는 **전적으로 lag 매입분**
(L=8≈26s; L≤3 은 threshold 와 동급/이하, decision-log 2026-06-12). 0-lag 요구와 충돌해
"사실상 폐기". 본 파일은 삭제 아님 — **재현/참고용 보존** (decision-log 2026-06-13).

원위치: `src/hi_ontop/hi_ontop_deneut.py::_seg_commit_refine` (제거됨).
신호·threshold deploy 는 현행 `hi_ontop_deneut.segment(emb)` 그대로 (de-neut + 적응 β + μ+cσ).
helper(`_deneut`,`_beta`,`_nr`)는 현행 모듈에서 import.
"""

from __future__ import annotations

import numpy as np

from hi_ontop.hi_ontop_deneut import _beta, _deneut, _nr


def seg_commit_refine(e, c, A, B, L0, lam, g_rho, rho_min, R, warmup, L, m_min, Mc, **_):
    """drift (AMI) deploy — bounded-lag commit-and-refine + local split-gain b* refinement.

    shock(V>θ)으로 armed → 오른쪽이 m_min 차거나 윈도우 L 소진 시, 윈도우 후보 split 위치 b 별
    gain ``SSE(a..t)-[SSE(a..b-1)+SSE(b..t)]`` 의 argmax b* 에 경계 emit. 확정 시 왼쪽 prototype 을
    b* 오른쪽으로 reset.
    """
    n = len(e); g = e[0].copy()
    Wm = 0.0; WM2 = 0.0; Wn = 0; m = e[0].copy(); k = 1
    pref = [np.zeros_like(e[0]), e[0].copy()]
    pred = []; last = -999; armed_at = -1
    for t in range(1, n):
        x = e[t]; beta = _beta(k, A, B, L0)
        mc = _deneut(m, g, beta); xc = _deneut(x, g, beta)
        V = (1 - float(xc @ mc)) - lam * (1 - float(x @ g))
        sd = (WM2 / (Wn - 1)) ** 0.5 if Wn > 1 else 0.0
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
