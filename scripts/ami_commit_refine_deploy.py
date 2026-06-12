#!/usr/bin/env python3
"""Local Commit-and-Refine Splitter — DEPLOY (online reset 부트스트랩 돌파 시도).

신호(de-neut V + 적응 β + μ+cσ)는 ami_adaptive_deneut_deploy.segment 와 동일하게 유지.
바뀌는 것은 **reset 메커니즘 하나**: hard reset(즉시·불가역) → commit-and-refine.
  핵심: 경계 *감지*(V>θ)는 'split 후보 생성'으로만 쓰고, reset *확정*은 오른쪽 후보 segment가
  자기 prototype 으로 응집(persistence)할 때만. 미확정 구간은 bounded lag(L) 안에서 정정/취소 가능.
경계는 spike 위치 b 에 emit(±2 tol 보존). 비교: hard-reset deploy / δ_eff / oracle 천장.
codex 자문 2026-06-11 (outputs/runs/_misc/codex_bootstrap_consult.md), decision-log 2026-06-11.
"""
from __future__ import annotations
import sys, math
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from run_encoder_comparison import delta_eff_seq
from hi_ontop.hi_ontop_v2 import adaptive_boundaries
from ami_adaptive_deneut_deploy import load_ami, ev, nr1, segment as segment_hard


def _deneut(x: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    """global(중립) 성분을 β만큼 제거한 단위벡터 — '화제의 변별적 방향'."""
    xc = x - beta * float(x @ g) * g
    return xc / (np.linalg.norm(xc) + 1e-9)


def segment_cr(e, c=1.0, A=2.0, B=1.0, L0=8, lam=0.6, g_rho=0.15, rho_min=0.05,
               R=4, warmup=8, L=3, m_min=2, h=1, Pc=0.0):
    """Local Commit-and-Refine Splitter.

    상태: 확정-왼쪽 prototype m(run-length k) / global g / 거리신호 stats(Wm,WM2,Wn) /
          잠정 경계 prov{b, pr=오른쪽 shadow prototype, rk, buf, ok=연속 persistence 횟수}.
    파라미터: L=lag 윈도우, m_min=확정 최소 오른쪽 길이, h=persistence 연속 충족, Pc=persistence 마진.
    """
    n = len(e)
    m = e[0].copy(); k = 1            # 확정 segment 왼쪽 prototype
    g = e[0].copy()
    Wm = 0.0; WM2 = 0.0; Wn = 0       # 거리신호 running stats (μ,σ)
    pred = []
    prov = None                       # 활성 잠정 경계 (없으면 None)
    last = -999
    for t in range(1, n):
        x = e[t]
        beta = min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)
        mc = _deneut(m, g, beta); xc = _deneut(x, g, beta)
        r_act = 1 - float(xc @ mc)
        V = r_act - lam * (1 - float(x @ g))
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= warmup else None

        if prov is None:
            # 후보 생성: V>θ 면 잠정 경계만 연다 (reset 확정 안 함, spike 는 stats 오염 안 시킴)
            if thr is not None and V > thr and k >= R and t - last >= R:
                prov = {"b": t, "pr": x.copy(), "rk": 1, "buf": [x.copy()], "ok": 0}
            else:
                Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
                rho = max(rho_min, 1.0 / (k + 1)); m = nr1((1 - rho) * m + rho * x); k += 1
        else:
            # bounded-lag 윈도우 안: 오른쪽 shadow prototype 키우며 응집(persistence) 검사
            b = prov["b"]; prov["buf"].append(x.copy())
            rk = prov["rk"]; rrho = max(rho_min, 1.0 / (rk + 1))
            prov["pr"] = nr1((1 - rrho) * prov["pr"] + rrho * x); prov["rk"] = rk + 1
            # persistence = 버퍼 발화들이 왼쪽보다 오른쪽 prototype 에 더 붙는가 (단일 outlier vs 새 응집)
            pr_c = _deneut(prov["pr"], g, beta); m_c2 = _deneut(m, g, beta)
            pers = float(np.mean([(1 - float(_deneut(z, g, beta) @ m_c2))
                                  - (1 - float(_deneut(z, g, beta) @ pr_c)) for z in prov["buf"]]))
            right_len = t - b + 1
            prov["ok"] = prov["ok"] + 1 if pers > Pc else 0
            if right_len >= m_min and prov["ok"] >= h:
                # 확정: 경계는 spike 위치 b 에 emit, 왼쪽 prototype 을 깨끗한 오른쪽 shadow 로 교체
                pred.append(b); last = b
                m = prov["pr"].copy(); k = prov["rk"]
                Wm = 0.0; WM2 = 0.0; Wn = 0        # 새 segment: stats 재초기화
                prov = None
            elif right_len >= L:
                # 거절(윈도우 소진): outlier 로 보고 버퍼를 왼쪽에 흡수, 경계 안 냄
                for z in prov["buf"]:
                    rho = max(rho_min, 1.0 / (k + 1)); m = nr1((1 - rho) * m + rho * z); k += 1
                prov = None
        g = nr1((1 - g_rho) * g + g_rho * x)
    return pred


def segment_cr2(e, c=1.0, A=2.0, B=1.0, L0=8, lam=0.6, g_rho=0.15, rho_min=0.05,
                R=4, warmup=8, L=4, m_min=2, Gc=0.0, Mc=0.0,
                rho_fix=None, gate=None, damp=0.3):
    """CR v2 — local split-gain b* refinement 추가.

    v5 오염완화 옵션(기본 off → v2 동일): rho_fix=고정 rho(recency, self-heal), gate=σ-배수(V>μ+gate·σ 인
    ambiguous/outlier 발화는 prototype 업데이트 rho 를 damp 배로 축소 → 오염 차단).

    shock(V>θ)으로 armed → 오른쪽이 m_min 차거나 윈도우 L 소진될 때, 최근 윈도우의 모든
    후보 split 위치 b 에 대해 SSE 기반 split gain 을 계산해 **argmax b* 에 경계 emit**(위치 최적화).
    SSE(i..j)=Σ‖z‖²−‖Σz‖²/cnt = cnt−‖Σz‖²/cnt (unit norm). 확정 시 왼쪽 prototype 을 b* 오른쪽으로 reset.
    """
    n = len(e); g = e[0].copy()
    Wm = 0.0; WM2 = 0.0; Wn = 0
    m = e[0].copy(); k = 1                 # shock 감지용 EWMA 왼쪽 prototype
    pref = [np.zeros_like(e[0]), e[0].copy()]   # 확정 segment(a..) 누적합 prefix; pref[i]=앞 i개 합
    pred = []; last = -999
    armed_at = -1                          # shock 발생 위치(없으면 -1)
    for t in range(1, n):
        x = e[t]
        beta = min(max(A - B * math.log(1 + k / L0), 0.0), 1.0)
        mc = _deneut(m, g, beta); xc = _deneut(x, g, beta)
        V = (1 - float(xc @ mc)) - lam * (1 - float(x @ g))
        sd = math.sqrt(WM2 / (Wn - 1)) if Wn > 1 else 0.0
        thr = (Wm + c * sd) if Wn >= warmup else None
        # 항상 segment 에 누적 (split 은 위치만 고르지 발화를 버리지 않음)
        pref.append(pref[-1] + x)
        seglen = len(pref) - 1             # 확정 시작 a 부터 현재까지 발화 수
        # shock 이면 arm (아직 armed 아니면)
        if armed_at < 0 and thr is not None and V > thr and k >= R and t - last >= R:
            armed_at = t
        committed = False
        if armed_at >= 0:
            right_from = armed_at           # shock 위치 = 오른쪽 후보 시작 하한
            right_len_now = t - armed_at + 1
            window_done = (t - armed_at + 1) >= L
            if right_len_now >= m_min or window_done:
                # 후보 split 위치(global) b ∈ [armed_at-? , t-m_min+1]; 우측 ≥ m_min, 좌측 ≥1
                a = t - seglen + 1          # 확정 segment 시작 global index
                lo = max(a + 1, armed_at - (L - 1)); hi = t - m_min + 1
                SSE = lambda i, j: (j - i + 1) - float(((pref[j - a + 1] - pref[i - a]) ** 2).sum()) / (j - i + 1)
                tot = SSE(a, t)
                best_b = -1; best_g = -1e9; second = -1e9
                for b in range(lo, hi + 1):
                    gn = tot - (SSE(a, b - 1) + SSE(b, t))
                    if gn > best_g:
                        second = best_g; best_g = gn; best_b = b
                    elif gn > second:
                        second = gn
                ok = (best_b > 0 and best_g > Gc and (best_g - max(second, 0.0)) >= Mc)
                if ok or window_done:
                    if ok:
                        pred.append(best_b); last = best_b
                        # 왼쪽 prototype/누적을 b* 오른쪽으로 reset
                        seg_r = [e[i] for i in range(best_b, t + 1)]
                        m = seg_r[-1].copy() if seg_r else x.copy()
                        mm = np.zeros_like(e[0]); k = 0; pref = [np.zeros_like(e[0])]
                        for z in seg_r:
                            pref.append(pref[-1] + z); k += 1
                        m = pref[-1] / (np.linalg.norm(pref[-1]) + 1e-9)
                        Wm = 0.0; WM2 = 0.0; Wn = 0
                        committed = True
                    armed_at = -1
        if not committed and armed_at < 0:
            # 평상시: stats + 왼쪽 EWMA 갱신 (오염완화 옵션 적용)
            Wn += 1; dd = V - Wm; Wm += dd / Wn; WM2 += dd * (V - Wm)
            rho = rho_fix if rho_fix else max(rho_min, 1.0 / (k + 1))
            if gate is not None and Wn > 1:               # outlier-gated: ambiguous 발화는 업데이트 축소
                sd_now = math.sqrt(WM2 / (Wn - 1))
                if V > Wm + gate * sd_now:
                    rho *= damp
            m = nr1((1 - rho) * m + rho * x); k += 1
        elif not committed:
            k += 1                          # armed 대기 중엔 run-length 만 증가
        g = nr1((1 - g_rho) * g + g_rho * x)
    return pred


def segment_cr3(e, c=1.0, W=8, m_min=2, R=4, warmup=8, deneut=False, lam=0.6,
                g_rho=0.15, A=2.0, B=1.0, L0=8):
    """CR v3 — shock 감지 제거, sliding split-gain 자체가 경계 신호.

    매 t 에서 현 segment(a..t)의 최근 윈도우(W) 후보 split 위치별 gain 계산 → best_g.
    best_g 가 적응 임계치 μ+cσ(gain history) 초과 시 b*(argmax)에 경계 emit. 오염된 EWMA
    prototype/V-signal 에 의존하지 않으므로 '감지 자체가 놓치던' 경계를 잡는 것이 목표(recall).
    deneut=True 면 윈도우 내 벡터를 de-neut(global g 제거) 후 gain 계산(global 지배 완화).
    """
    n = len(e); a = 0
    g = e[0].copy()
    pref = [np.zeros_like(e[0]), e[0].copy()]
    Gm = 0.0; GM2 = 0.0; Gn = 0
    pred = []; last = -999

    def rebuild(a, t):
        p = [np.zeros_like(e[0])]
        for i in range(a, t + 1):
            p.append(p[-1] + (_deneut(e[i], g, _beta(t - a + 1)) if deneut else e[i]))
        return p

    def _beta(L):
        return min(max(A - B * math.log(1 + L / L0), 0.0), 1.0) if deneut else 0.0

    for t in range(1, n):
        seglen = t - a + 1
        vec = _deneut(e[t], g, _beta(seglen)) if deneut else e[t]
        if deneut:
            pref = rebuild(a, t)            # de-neut 은 g/β 변해 prefix 재계산 필요 (윈도우만이면 비싸지만 정확)
        else:
            pref.append(pref[-1] + vec)
        if seglen >= m_min + 1:
            SSE = lambda i, j: (j - i + 1) - float(((pref[j - a + 1] - pref[i - a]) ** 2).sum()) / (j - i + 1)
            tot = SSE(a, t); lo = max(a + 1, t - W + 1); hi = t - m_min + 1
            if hi >= lo:
                best_b = -1; best_g = -1e9
                for b in range(lo, hi + 1):
                    gn = tot - (SSE(a, b - 1) + SSE(b, t))
                    if gn > best_g:
                        best_g = gn; best_b = b
                sd = math.sqrt(GM2 / (Gn - 1)) if Gn > 1 else 0.0
                thr = (Gm + c * sd) if Gn >= warmup else None
                if thr is not None and best_g > thr and (t - last) >= R:
                    pred.append(best_b); last = best_b; a = best_b
                    pref = rebuild(a, t)
                else:
                    Gn += 1; dd = best_g - Gm; Gm += dd / Gn; GM2 += dd * (best_g - Gm)
        g = nr1((1 - g_rho) * g + g_rho * e[t])
    return pred


def segment_cr4(e, c=1.0, A=2.0, B=1.0, L0=8, lam=0.6, g_rho=0.15, rho_min=0.05,
                R=4, warmup=8, L=8, m_min=2, Mc=0.0, Lmax=48, min_gain=0.0):
    """CR v4 — 2-stage: cr2(precision) + 긴 segment recall booster.

    1) cr2 로 high-precision 경계(`base`) 획득.
    2) base 로 나뉜 각 segment 가 Lmax 보다 길면(=놓친 경계 흔적), de-neut split-gain argmax 위치에서
       분할(gain>min_gain). 분할 후 양쪽이 여전히 길면 재귀. → shock 이 놓친 경계의 recall 보강.
    de-neut split-gain 은 segment-local global(g=segment 평균)로 중립 제거 후 SSE 기반.
    """
    base = sorted(set(p for p in segment_cr2(e, c=c, A=A, B=B, L0=L0, lam=lam, g_rho=g_rho,
                                             rho_min=rho_min, R=R, warmup=warmup, L=L,
                                             m_min=m_min, Mc=Mc) if 0 < p < len(e)))
    n = len(e); out = set(base)

    def split_seg(lo, hi):
        # [lo, hi) 구간, 길면 best de-neut split 추가 후 재귀
        if hi - lo <= Lmax:
            return
        seg = e[lo:hi]
        g = seg.mean(0); g = g / (np.linalg.norm(g) + 1e-9)
        sc = np.stack([_deneut(z, g, 1.0) for z in seg])      # full de-neut (긴 segment → β≈1 영역)
        pref = np.vstack([np.zeros_like(sc[0]), np.cumsum(sc, 0)])
        m = len(seg)
        SSE = lambda i, j: (j - i) - float((pref[j] - pref[i]) @ (pref[j] - pref[i])) / (j - i)
        tot = SSE(0, m); best_p = -1; best_g = -1e9
        for p in range(m_min, m - m_min + 1):
            gn = tot - (SSE(0, p) + SSE(p, m))
            if gn > best_g:
                best_g = gn; best_p = p
        if best_p > 0 and best_g > min_gain:
            b = lo + best_p; out.add(b)
            split_seg(lo, b); split_seg(b, hi)

    pts = [0] + base + [n]
    for i in range(len(pts) - 1):
        split_seg(pts[i], pts[i + 1])
    return sorted(p for p in out if 0 < p < n)


if __name__ == "__main__":
    AMI = load_ami(); ng = sum(len(d[2]) for d in AMI)
    print(f"AMI {len(AMI)}미팅 gold={ng}. Local Commit-and-Refine Splitter DEPLOY.")
    print("참조: δ_eff Score~0.203 / hard-reset deneut deploy Score~0.37(±2F1~0.115) / "
          "clean+μcσ oracle 천장 ±2F1 0.554 / adaptive-deneut oracle 0.341")
    hdr = f"{'method':<34}{'pred':>6}{'  F1(±2)':>9}{'  Pk':>8}{'  WD':>8}{'  Score':>9}"
    print(hdr)

    # baseline 1: δ_eff
    tot, f2, pk, wd, sc = ev(AMI, lambda e: [i for i, b in enumerate(
        adaptive_boundaries(list(delta_eff_seq(e)), c=1.5, mode="ewma")) if b])
    print(f"{'δ_eff ewma [기존]':<34}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")

    # baseline 2: hard-reset de-neut deploy (현 best)
    for c in (1.5, 1.2, 1.0):
        tot, f2, pk, wd, sc = ev(AMI, lambda e, c=c: segment_hard(e, c=c))
        print(f"{('hard-reset deneut c='+str(c)):<34}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}")

    print("-" * len(hdr))
    # commit-and-refine grid (L, m_min, h, Pc) × c
    for L, m_min, h in [(3, 2, 1), (4, 2, 1), (5, 3, 1), (3, 2, 2)]:
        for c in (1.2, 1.0, 0.8):
            for Pc in (0.0, 0.01):
                tot, f2, pk, wd, sc = ev(AMI, lambda e, c=c, L=L, m_min=m_min, h=h, Pc=Pc:
                                         segment_cr(e, c=c, L=L, m_min=m_min, h=h, Pc=Pc))
                lab = f"CR L={L} m={m_min} h={h} c={c} Pc={Pc}"
                print(f"{lab:<34}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}", flush=True)

    print("-" * len(hdr) + "  [v2: local split-gain b* refinement]")
    for L, m_min in [(4, 2), (6, 2), (4, 3), (8, 2)]:
        for c in (1.5, 1.2, 1.0):
            for Mc in (0.0, 0.02):
                tot, f2, pk, wd, sc = ev(AMI, lambda e, c=c, L=L, m_min=m_min, Mc=Mc:
                                         segment_cr2(e, c=c, L=L, m_min=m_min, Mc=Mc))
                lab = f"CR2 L={L} m={m_min} c={c} Mc={Mc}"
                print(f"{lab:<34}{tot:>6}{f2:>9.3f}{pk:>8.3f}{wd:>8.3f}{sc:>9.3f}", flush=True)
