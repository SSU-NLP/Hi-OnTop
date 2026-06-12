#!/usr/bin/env python3
"""Latency 전용 미팅 subset 선정 (논문-서술 가능한 기준이 N 을 결정).

전제: LLM latency ≈ f(입력 window 크기, 출력 크기, 모델) — **content-independent**(같은 크기면 내용 무관 비슷).
따라서 37 dev 전수(모델당 ~10h)를 다 잴 필요 없이, **실제 워크로드의 window-크기 분포를 재현**하고 **극단을
포함**하는 최소 미팅 subset 으로 동일 latency 분포를 얻는다(입력버킷 샘플 ❌ — 실제 미팅·실제 콜 전수, 미팅 수만↓).

선택 기준(이 셋을 만족하는 **최소** subset, greedy):
  (1) 극단 포함: 최장·최단 회의 + 최대·최소 boundary density 회의 → tail(p95/p99/max) 결정 크기 극단 커버.
  (2) 분포 일치: 모든 버퍼 B 에서 pooled window turn-count 분포가 full-37 과 KS D≤D_MAX & p≥0.05 (구별 불가).
  (3) tail 안정성: 작은 버퍼는 percentile 안정화 위한 최소 window 수 MIN_WIN 확보.
N 은 사전 지정이 아니라 위 기준의 결과. 산출 + KS 표(논문 justification) 를 출력.

입력 = `ami_subset.json`(37 dev). 산출 = `outputs/runs/_misc/ami_latency_subset.json` + 근거 통계.
handoff HANDOFF_0612 §1A/§3z-C(latency). secom_latency_full.py 가 이 subset 을 쓰면 됨.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

REPO = Path(__file__).resolve().parent.parent
TOPIC = REPO / "data" / "ami" / "topic"
SEG_BUFFERS = ["full", 120, 60, 30, 10]
OFFSETS = [0.0, 0.5]
D_MAX = 0.15        # 버퍼별 KS 거리 상한
P_MIN = 0.05        # 버퍼별 KS p 하한(분포 구별 불가)
MIN_WIN = 300       # 작은 버퍼 tail 안정화 최소 window 수(full 제외)


def load(subset):
    out = {}
    for mid in sorted(json.load(open(subset))):
        d = json.load(open(TOPIC / f"{mid}.json")); turns = d["turns"]; n = len(turns)
        bt = list(d["bnd_top"]); nb = sum(1 for i, b in enumerate(bt) if b == 1 and i < n - 1)
        out[mid] = {"turns": turns, "n": n, "dens": nb / n if n else 0.0, "nb": nb}
    return out


def win_sizes(meta, B):
    """한 미팅이 버퍼 B 에서 만드는 window 들의 turn-count 리스트(secom 와 동일 규칙)."""
    turns = meta["turns"]; n = meta["n"]
    if B == "full":
        return [n]
    t0 = turns[0]["start"]; t_last = turns[-1]["start"]; sizes = []
    for off in OFFSETS:
        w_start = t0; w_end = t0 + (off * B if off > 0 else B)
        while w_start <= t_last:
            c = sum(1 for k in range(n) if w_start <= turns[k]["start"] < w_end)
            if c >= 2:
                sizes.append(c)
            w_start = w_end; w_end += B
    return sizes


def pooled(mids, by, B):
    out = []
    for m in mids:
        out += by[m][B]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(REPO / "outputs/runs/_misc/ami_subset.json"))
    ap.add_argument("--out", default=str(REPO / "outputs/runs/_misc/ami_latency_subset.json"))
    args = ap.parse_args()
    meta = load(args.subset)
    mids_all = list(meta)
    # 미팅별 버퍼별 window-크기 미리 계산
    by = {m: {B: win_sizes(meta[m], B) for B in SEG_BUFFERS} for m in mids_all}
    full = {B: pooled(mids_all, by, B) for B in SEG_BUFFERS}

    # (1) 극단 4종 seed
    seed = set()
    seed.add(max(mids_all, key=lambda m: meta[m]["n"]))      # 최장
    seed.add(min(mids_all, key=lambda m: meta[m]["n"]))      # 최단
    seed.add(max(mids_all, key=lambda m: meta[m]["dens"]))   # 최대 density
    seed.add(min(mids_all, key=lambda m: meta[m]["dens"]))   # 최소 density
    chosen = list(seed)

    def buf_ks(mids):
        """버퍼별 (D, p). window<2 면 None."""
        r = {}
        for B in SEG_BUFFERS:
            sub = pooled(mids, by, B)
            if len(sub) < 2 or len(full[B]) < 2:
                r[B] = (1.0, 0.0); continue
            d, p = ks_2samp(sub, full[B]); r[B] = (float(d), float(p))
        return r

    def satisfied(mids):
        ks = buf_ks(mids)
        for B in SEG_BUFFERS:
            d, p = ks[B]
            if B != "full":
                if len(pooled(mids, by, B)) < MIN_WIN:
                    return False, ks
            if d > D_MAX or p < P_MIN:
                return False, ks
        return True, ks

    # greedy: KS 최악-버퍼 D 를 가장 줄이는 미팅 추가
    rest = [m for m in mids_all if m not in chosen]
    ok, ks = satisfied(chosen)
    while not ok and rest:
        def worst_d(mids):
            return max(buf_ks(mids)[B][0] for B in SEG_BUFFERS if B != "full")
        best = min(rest, key=lambda m: worst_d(chosen + [m]))
        chosen.append(best); rest.remove(best)
        ok, ks = satisfied(chosen)

    chosen = sorted(chosen)
    json.dump(chosen, open(args.out, "w"), indent=1)

    # --- 근거 출력 (논문 Table) ---
    print(f"=== latency subset N={len(chosen)} / {len(mids_all)} (기준이 N 결정) ===")
    print(f"기준: 극단4 포함 + 버퍼별 KS D≤{D_MAX}, p≥{P_MIN} + 작은버퍼 window≥{MIN_WIN}\n")
    print(f"{'buffer':>8}{'sub_win':>9}{'full_win':>9}{'KS_D':>8}{'KS_p':>8}{'pass':>6}")
    for B in SEG_BUFFERS:
        d, p = ks[B]; sw = len(pooled(chosen, by, B)); fw = len(full[B])
        ps = "OK" if (d <= D_MAX and p >= P_MIN and (B == "full" or sw >= MIN_WIN)) else "—"
        print(f"{str(B):>8}{sw:>9}{fw:>9}{d:>8.3f}{p:>8.3f}{ps:>6}")

    def rng(key):
        a = np.array([meta[m][key] for m in chosen]); fa = np.array([meta[m][key] for m in mids_all])
        return f"sub [{a.min():.3g},{a.max():.3g}] vs full [{fa.min():.3g},{fa.max():.3g}]"
    print(f"\n극단 커버: n_turns {rng('n')}\n          density {rng('dens')}")
    inc_sub = sum(meta[m]["n"] - 1 for m in chosen); inc_full = sum(meta[m]["n"] - 1 for m in mids_all)
    seg_sub = sum(len(pooled(chosen, by, B)) for B in SEG_BUFFERS)
    seg_full = sum(len(full[B]) for B in SEG_BUFFERS)
    print(f"\n콜 수(모델당): segment {seg_sub} (full {seg_full}) + incremental {inc_sub} (full {inc_full}) "
          f"= {seg_sub+inc_sub} (full {seg_full+inc_full}, {(seg_sub+inc_sub)/(seg_full+inc_full)*100:.0f}%)")
    print(f"저장: {args.out}")
    print("subset:", chosen)


if __name__ == "__main__":
    main()
