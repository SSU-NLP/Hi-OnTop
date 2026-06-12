#!/usr/bin/env python3
"""**TextTiling-prefix (online, past-only) — AUXILIARY baseline.**

NOT a 5-row core-table method. Online classical/unsupervised reference
(codex decision-log 2026-05-20), same status as BayesSeg-prefix.

Original SuperDialseg TextTiling is offline whole-document. Here, at
turn ``t`` we feed ONLY ``U1..Ut`` (no future) to nltk
``TextTilingTokenizer`` (w=10,k=6, the SuperDialseg toolkit setting) and
read whether a tile boundary falls right before ``Ut`` → the online
boundary decision for turn ``t``. Unlike BayesSeg, TextTiling needs **no
K** (depth-score thresholding decides boundaries itself), so it does NOT
degenerate via native-K over-segmentation; short prefixes simply yield
few/no boundaries (under-seg) — not catastrophic.

Columns (aux row): Pk/WD/F1/Score (INDICATIVE — prefix-causal ≠ offline
paper setting), Avg latency/turn (ms, pure-CPU nltk), LLM calls/turn = 0,
Tokens/turn = 0.

Sample = whole dialogues until cumulative utts ≥ --target-turns (same
convention as the other online runners; small-n indicative). Crash-safe
sidecar + resume. Non-LLM ⇒ no quota/cost, runtime = seconds–minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
DATASETS = ["tiage", "dialseg711", "superseg"]


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None
    sys.modules["anthropic"] = m


def _utterance_lines(dialogue: str) -> list[str]:
    return [seg for seg in dialogue.split("[NEWLINE]")
            if seg.strip() != "[BOUNDARY]" and seg.strip() != ""]


def _tile_end_flags(utts: list[str], tt) -> list[int] | None:
    """Mirror SuperDialseg TexttilingSegmenter.forward: per-utterance
    1 = last utterance of a tile (boundary AFTER it). len == len(utts)
    or None if TextTiling fails / length mismatch (too short)."""
    doc = "\n\n".join((u.strip() or ".") for u in utts)
    try:
        tiles = tt.tokenize(doc)
    except Exception:
        return None
    pred: list[int] = []
    for tile in tiles:
        lines = [x for x in tile.strip().split("\n\n") if x != ""]
        if not lines:
            continue
        pred += [0] * len(lines)
        pred[-1] = 1
    if not pred or len(pred) != len(utts):
        return None
    pred[-1] = 0
    return pred


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {k: float("nan") for k in
                ("n", "mean", "std", "min", "p50", "p90",
                 "p95", "p99", "max")}
    a = np.asarray(xs, float)
    return dict(
        n=len(a), mean=float(a.mean()), std=float(a.std()),
        min=float(a.min()), p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)), max=float(a.max()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-20_texttiling_prefix")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--target-turns", type=int, default=100)
    ap.add_argument("-w", type=int, default=10, help="TextTiling w")
    ap.add_argument("-k", type=int, default=6, help="TextTiling k")
    args = ap.parse_args()

    import nltk
    for pkg in ("stopwords", "punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    from nltk.tokenize import TextTilingTokenizer

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)
    import src.autoseg as A  # noqa: E402

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in args.datasets:
        full = A.alternative_load_dataset(ds, "test")
        sample, cum = [], 0
        for d in full:
            utts = _utterance_lines(d["dialogue"])
            if len(utts) < 2:
                continue
            sample.append((d["id"], d["dialogue"], utts))
            cum += len(utts)
            if cum >= args.target_turns:
                break
        print(f"\n=== {ds}: {len(sample)} dialogues, {cum} turns ===")

        sidecar = exp_dir / f"turns_{ds}.jsonl"
        done = {}
        if sidecar.exists():
            for ln in sidecar.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    done[(r["id"], r["t"])] = r

        tt = TextTilingTokenizer(w=args.w, k=args.k)
        lat_ms, preds, labels, miss = [], [], [], 0
        fh = open(sidecar, "a")
        for did, dialogue, utts in sample:
            n = len(utts)
            uttr = []
            for t in range(1, n + 1):
                kk = (did, t)
                if kk in done:
                    r = done[kk]
                    if r["t"] != 1:
                        lat_ms.append(r["ms"])
                    miss += int(r.get("miss", False))
                    uttr.append(bool(r["shift"]))
                    continue
                if t == 1:
                    rec = {"ds": ds, "id": did, "t": 1, "ms": 0.0,
                           "shift": False, "miss": False}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    uttr.append(False)
                    continue
                t0 = time.perf_counter()
                flags = _tile_end_flags(utts[:t], tt)
                ms = (time.perf_counter() - t0) * 1000.0
                # online decision for the just-arrived Ut: boundary
                # between U(t-1) and Ut  <=>  U(t-1) ended a tile
                if flags is None:
                    shift, mis = False, True
                else:
                    shift, mis = bool(flags[t - 2] == 1), False
                rec = {"ds": ds, "id": did, "t": t, "ms": ms,
                       "shift": shift, "miss": mis}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                lat_ms.append(ms); miss += int(mis)
                uttr.append(shift)
            uttr[0] = False
            pred = A.extract_pred(uttr)
            lbl, _ = A.extract_label(dialogue.split("[NEWLINE]"), True)
            pred, lbl = A.align_pred_label(pred, lbl)
            preds.append(pred)
            labels.append(lbl)
        fh.close()

        m = A.compute_metrics(preds, labels)
        score = 0.5 * m["f1"] + 0.25 * (1 - m["pk"]) + 0.25 * (1 - m["wd"])
        st = _stats(lat_ms)
        rows.append(dict(ds=ds, n_dial=len(sample), n_turns=cum,
                         pk=m["pk"], wd=m["wd"], f1=m["f1"], score=score,
                         parse_miss=miss, **st))
        r = rows[-1]
        print(f"  Pk={r['pk']:.4f} WD={r['wd']:.4f} F1={r['f1']:.4f} "
              f"Score={r['score']:.4f} | lat/turn mean={r['mean']:.2f}ms "
              f"p95={r['p95']:.2f}ms miss={miss}/{r['n_turns']}")

    _write_report(exp_dir, args, rows)


def _write_report(exp_dir: Path, args, rows) -> None:
    def g(v, p=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    L = [
        "# TextTiling-prefix (online, past-only) — AUXILIARY baseline",
        "",
        "> **5행 핵심표 아님.** BayesSeg-prefix 와 같은 'online "
        "classical/unsupervised 참고선'. codex decision-log 2026-05-20.",
        "",
        "## 1. Setup",
        f"- **Method**: nltk TextTilingTokenizer(w={args.w},k={args.k}, "
        "SuperDialseg toolkit 설정). turn t 에 `U1..Ut`(미래 미관측)만 → "
        "U(t-1) 이 tile 끝이면 t 에 경계. **K 불필요**(depth-score 자체 "
        "임계) → BayesSeg native-K 식 degenerate 없음(짧으면 under-seg).",
        "- **non-LLM, CPU-only**: LLM calls/turn = 0, Tokens/turn = 0.",
        "- **표본**: 데이터셋별 누적 발화 ≥ "
        f"{args.target_turns} (다른 online 러너와 동일 정의, small-n "
        "indicative). prefix-causal ≠ 원 offline TextTiling 점수.",
        "- metric = autoseg(segeval) Pk/WD + F1; Score = 0.5·F1 + "
        "0.25·(1−Pk) + 0.25·(1−WD).",
        "- **Crash-safe**: `turns_<ds>.jsonl` 턴마다 append, resume.",
        "",
        "## 2. 보조표 행 (TextTiling-prefix | past-only, non-LLM)",
        "",
        "| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) ↓ | calls/turn | tok/turn | parse_miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_dial']}/{r['n_turns']} | {g(r['pk'])} | "
            f"{g(r['wd'])} | {g(r['f1'])} | {g(r['score'])} | "
            f"{g(r['mean'], 2)} | 0 | 0 | {r['parse_miss']} |")
    L += [
        "",
        "## 3. latency 분포 (ms/turn, pure-CPU nltk)",
        "",
        "| dataset | n | mean | p50 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {g(r['n'],0)} | {g(r['mean'],2)} | "
                 f"{g(r['p50'],2)} | {g(r['p95'],2)} | {g(r['max'],2)} |")
    L += [
        "",
        "## 4. 한계 / 정직성",
        "- **AUXILIARY only** (codex). 핵심 5행표 비교 아님.",
        "- **Pk/WD/F1 = INDICATIVE**: prefix-causal online = 원 offline "
        "TextTiling(논문 0.471/0.363/0.382)과 다른 setting → 직접 비교 불가. "
        "단 K 불필요라 BayesSeg-prefix(WD≈1 degenerate)와 달리 정상 범위 기대.",
        "- parse_miss = 짧은 prefix 에서 TextTiling 실패/길이불일치 → 그 턴 "
        "경계 False 처리(과소분할 쪽, degenerate 아님).",
        "- latency = 순수 CPU nltk (LLM 무관). small-n 표본.",
        "- 원 offline 논문값 검증은 별도(offline 전체대화 실행) 필요.",
    ]
    out = exp_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
