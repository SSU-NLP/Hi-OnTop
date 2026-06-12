#!/usr/bin/env python3
"""AMI 버퍼-sweep 용 stratified subset 선정 + 분포 검증.

codex 권고: ES/IS/TS(IB) 시리즈 비율 + 회의 길이(n_turns) + boundary density 를 전체 139개와 맞춘
30~45개 subset. 방법: 시리즈 비례 할당 → 시리즈 내 n_turns 정렬 후 등간격 추출(길이 범위 span).
seed-free 결정론(등간격)이라 재현 가능. 선정 후 subset vs full 분포(시리즈%/길이/밀도/경계수) 비교 출력.
산출: outputs/runs/_misc/ami_subset.json (meeting id 리스트).
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

TOPIC = "data/ami/topic"
N_TARGET = 36


def feats():
    rows = []
    for p in sorted(glob.glob(f"{TOPIC}/*.json")):
        if p.endswith("manifest.json"):
            continue
        d = json.load(open(p)); mid = d["meeting"]; nt = d["n_turns"]
        nb = int(sum(d["bnd_top"])) - (1 if d["bnd_top"] and d["bnd_top"][-1] == 1 else 0)
        nb = sum(1 for i, b in enumerate(d["bnd_top"]) if b == 1 and i < nt - 1)
        rows.append({"mid": mid, "series": mid[:2], "n_turns": nt,
                     "n_bnd": nb, "dens": nb / nt if nt else 0.0})
    return rows


def pick_even(sorted_rows, k):
    """정렬 리스트에서 k개 등간격 추출(min~max span)."""
    if k <= 0:
        return []
    if k >= len(sorted_rows):
        return list(sorted_rows)
    idx = [round(i * (len(sorted_rows) - 1) / (k - 1)) for i in range(k)] if k > 1 else [len(sorted_rows) // 2]
    return [sorted_rows[i] for i in sorted(set(idx))]


def alloc_prop(groups, quota):
    """그룹 크기에 비례해 quota 배분(합 보정)."""
    tot = sum(len(g) for g in groups.values()) or 1
    a = {k: min(len(g), round(len(g) / tot * quota)) for k, g in groups.items()}
    while sum(a.values()) != quota:
        diff = quota - sum(a.values())
        # 가장 큰(또는 여유 있는) 셀에서 가감
        cand = [k for k in groups if (a[k] < len(groups[k]) if diff > 0 else a[k] > 0)]
        if not cand:
            break
        tgt = max(cand, key=lambda k: len(groups[k]))
        a[tgt] += 1 if diff > 0 else -1
    return a


def main():
    rows = feats()
    by_ser = {}
    for r in rows:
        by_ser.setdefault(r["series"], []).append(r)
    total = len(rows)
    ser_alloc = alloc_prop(by_ser, N_TARGET)
    sub = []
    for s, v in by_ser.items():
        q = ser_alloc[s]
        if q >= len(v):
            sub += v; continue
        # 시리즈 내 길이-반 × density-반 = 4 사분면 (2D 층화)
        med_l = float(np.median([r["n_turns"] for r in v]))
        med_d = float(np.median([r["dens"] for r in v]))
        cells = {(li, di): [] for li in (0, 1) for di in (0, 1)}
        for r in v:
            cells[(0 if r["n_turns"] < med_l else 1, 0 if r["dens"] < med_d else 1)].append(r)
        cell_alloc = alloc_prop(cells, q)
        for ck, cv in cells.items():
            sub += pick_even(sorted(cv, key=lambda r: r["dens"]), cell_alloc[ck])  # 셀 내 density span
    # 길이 극단(최장 회의) force-include → 길이 [min,max] 완전 span (N=37)
    longest = max(rows, key=lambda r: r["n_turns"])
    if not any(r["mid"] == longest["mid"] for r in sub):
        sub.append(longest)
    sub_mids = sorted(r["mid"] for r in sub)
    Path("outputs/runs/_misc").mkdir(parents=True, exist_ok=True)
    json.dump(sub_mids, open("outputs/runs/_misc/ami_subset.json", "w"), indent=1)

    def stat(rs, key):
        a = np.array([r[key] for r in rs])
        return f"mean {a.mean():.1f} med {np.median(a):.0f} [{a.min():.0f},{a.max():.0f}]"

    print(f"=== stratified subset N={len(sub_mids)} / 139 (seed-free 등간격) ===")
    print(f"{'series':<8}{'full':>10}{'subset':>10}{'full%':>8}{'sub%':>8}")
    for s in sorted(by_ser):
        fn = len(by_ser[s]); sn = sum(1 for r in sub if r['series'] == s)
        print(f"{s:<8}{fn:>10}{sn:>10}{fn/total*100:>7.1f}%{sn/len(sub)*100:>7.1f}%")
    print("\n분포 검증 (full vs subset):")
    print(f"  n_turns : full {stat(rows,'n_turns')}\n            sub  {stat(sub,'n_turns')}")
    print(f"  density : full {stat(rows,'dens')}\n            sub  {stat(sub,'dens')}")
    print(f"  n_bnd   : full {stat(rows,'n_bnd')}\n            sub  {stat(sub,'n_bnd')}")
    print(f"  총 경계 : full {sum(r['n_bnd'] for r in rows)}  subset {sum(r['n_bnd'] for r in sub)}")
    print(f"\n저장: outputs/runs/_misc/ami_subset.json ({len(sub_mids)}개)")
    print("subset:", sub_mids)


if __name__ == "__main__":
    main()
