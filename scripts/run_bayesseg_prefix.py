#!/usr/bin/env python3
"""**BayesSeg-prefix (online, past-only) — AUXILIARY baseline.**

NOT a 5-row core-table method. Online classical/unsupervised reference,
same status as TextTiling-prefix (codex decision-log 2026-05-20).

BayesSeg (Eisenstein & Barzilay 2008) is offline whole-doc. Here, at
turn ``t`` we feed ONLY ``U1..Ut`` (no future) to a **persistent JVM**
(``OnlineSegServer``, I/O-harness only, algorithm unchanged) → per-turn
latency reflects the segmentation compute, NOT JVM cold-start (~5 ms vs
~700 ms subprocess). num-segs is **native Bayesian K** (config
``num-segs-known=false``); the passed value is only a causal duration
prior (round(nSent/seg-len)), never gold/oracle K → no future leak
(option C; A=gold-K leak rejected, decision-log 2026-05-20).

Columns for the comparison (aux row): Pk/WD/F1/Score (INDICATIVE — K
policy sensitive, native-K over-segments short prefixes), Avg
latency/turn (ms, meaningful via persistent JVM), LLM calls/turn = 0,
Tokens/turn = 0 (non-LLM, CPU-only).

Sample = whole dialogues until cumulative utts ≥ --target-turns
(same convention as the other online runners; small-n indicative).
Crash-safe: per-turn {sec_ms, shift} appended to sidecar; resume skips
done turns. No LLM ⇒ no quota/cost; runtime = minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
BSEG = (REPO / "benchmarks" / "superdialseg" / "src" / "super_dialseg"
        / "models" / "bayesseg")
CP = ("classes:lib/colt.jar:lib/lingpipe-3.4.0.jar:lib/MinCutSeg.jar:"
      "lib/mtj.jar:lib/options.jar:lib/log4j-1.2.14.jar")
DATASETS = ["tiage", "dialseg711", "superseg"]


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None
    sys.modules["anthropic"] = m


def _utterance_lines(dialogue: str) -> list[str]:
    return [seg for seg in dialogue.split("[NEWLINE]")
            if seg.strip() != "[BOUNDARY]" and seg.strip() != ""]


def _clean(u: str) -> str:
    """BayesSeg input = one utterance per line; strip newlines/tabs and the
    'speaker:' prefix is kept as plain text (lexical-cohesion model)."""
    return u.replace("\t", " ").replace("\n", " ").strip() or "."


class Server:
    """Persistent OnlineSegServer JVM (one per run)."""

    def __init__(self, seg_len: float):
        self.p = subprocess.Popen(
            ["java", f"-Duser.dir={BSEG}", "-cp", CP,
             "edu.mit.nlp.segmenter.OnlineSegServer",
             "-config", "config/dp_online.config", "-seg-len", str(seg_len)],
            cwd=str(BSEG), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        # wait for ready (server prints to stderr; just give it a beat)
        time.sleep(1.0)

    def seg(self, utts: list[str]) -> tuple[list[int], float]:
        for u in utts:
            self.p.stdin.write(_clean(u) + "\n")
        self.p.stdin.write("===END===\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline().rstrip("\n")
        if "\t" not in line:
            return [], float("nan")
        body, ms = line.rsplit("\t", 1)
        try:
            ms = float(ms)
        except ValueError:
            ms = float("nan")
        if body.startswith("ERR") or not body.strip().startswith("["):
            return [], ms
        try:
            idx = [int(x) for x in body.strip()[1:-1].split(",")
                   if x.strip() != ""]
        except ValueError:
            idx = []
        return idx, ms

    def close(self):
        try:
            self.p.stdin.write("===QUIT===\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


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
    ap.add_argument("--name", default="2026-05-20_bayesseg_prefix")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--target-turns", type=int, default=100)
    ap.add_argument("--seg-len", type=float, default=5.0,
                    help="causal duration-prior hint (NOT gold K)")
    ap.add_argument("--warmup-drop", type=int, default=1,
                    help="drop first N calls from latency stats (JIT warmup)")
    args = ap.parse_args()

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)
    import src.autoseg as A  # noqa: E402  (metric + data helpers only)

    if not (BSEG / "classes" / "edu" / "mit" / "nlp" / "segmenter"
            / "OnlineSegServer.class").exists():
        sys.exit("OnlineSegServer.class 없음 — bayesseg ant build 먼저")

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

        srv = Server(args.seg_len)
        lat_ms, preds, labels, call_i = [], [], [], 0
        fh = open(sidecar, "a")
        try:
            for did, dialogue, utts in sample:
                n = len(utts)
                uttr = []
                for t in range(1, n + 1):
                    kk = (did, t)
                    if kk in done:
                        r = done[kk]
                        if r["t"] != 1 and not (isinstance(r["ms"], float)
                                                and np.isnan(r["ms"])):
                            call_i += 1
                            if call_i > args.warmup_drop:
                                lat_ms.append(r["ms"])
                        uttr.append(bool(r["shift"]))
                        continue
                    if t == 1:
                        rec = {"ds": ds, "id": did, "t": 1, "ms": 0.0,
                               "shift": False}
                        fh.write(json.dumps(rec) + "\n"); fh.flush()
                        uttr.append(False)
                        continue
                    idx, ms = srv.seg(utts[:t])
                    shift = t in idx          # is Ut a segment endpoint?
                    rec = {"ds": ds, "id": did, "t": t, "ms": ms,
                           "shift": bool(shift)}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    call_i += 1
                    if not np.isnan(ms) and call_i > args.warmup_drop:
                        lat_ms.append(ms)
                    uttr.append(bool(shift))
                uttr[0] = False
                pred = A.extract_pred(uttr)
                lbl, _ = A.extract_label(dialogue.split("[NEWLINE]"), True)
                pred, lbl = A.align_pred_label(pred, lbl)
                preds.append(pred)
                labels.append(lbl)
        finally:
            fh.close()
            srv.close()

        m = A.compute_metrics(preds, labels)
        score = 0.5 * m["f1"] + 0.25 * (1 - m["pk"]) + 0.25 * (1 - m["wd"])
        st = _stats(lat_ms)
        rows.append(dict(ds=ds, n_dial=len(sample), n_turns=cum,
                         pk=m["pk"], wd=m["wd"], f1=m["f1"], score=score,
                         **st))
        r = rows[-1]
        print(f"  Pk={r['pk']:.4f} F1={r['f1']:.4f} Score={r['score']:.4f}"
              f" | lat/turn mean={r['mean']:.1f}ms p95={r['p95']:.1f}ms "
              f"(n={r['n']}, warmup-drop={args.warmup_drop})")

    _write_report(exp_dir, args, rows)


def _write_report(exp_dir: Path, args, rows) -> None:
    def g(v, p=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    L = [
        "# BayesSeg-prefix (online, past-only) — AUXILIARY baseline",
        "",
        "> **5행 핵심표 아님.** TextTiling-prefix 와 같은 'online "
        "classical/unsupervised 참고선'. codex decision-log 2026-05-20.",
        "",
        "## 1. Setup",
        "- **Method**: BayesSeg (Eisenstein&Barzilay 2008) offline whole-doc"
        " → turn t 에 `U1..Ut`(미래 미관측)만. **persistent JVM**"
        " (`OnlineSegServer`, I/O-harness only, 알고리즘 불변) → per-turn "
        "latency = 분할 compute (JVM cold-start 제외, ~ms).",
        "- **num-segs = native Bayesian K** (`dp_online.config` "
        "`num-segs-known=false`, `DPSeg.segmentUnknown`). 전달 K 는 causal "
        f"duration prior round(nSent/{args.seg_len}) 뿐, **gold/oracle K 아님"
        "** (option C; A=gold-K 미래누출 배제).",
        "- **non-LLM, CPU-only**: LLM calls/turn = 0, Tokens/turn = 0.",
        "- **표본**: 데이터셋별 누적 발화 ≥ "
        f"{args.target_turns} (다른 online 러너와 동일 정의, small-n "
        "indicative). warmup-drop={}.".format(args.warmup_drop),
        "- metric = autoseg(segeval) Pk/WD + F1; Score = 0.5·F1 + "
        "0.25·(1−Pk) + 0.25·(1−WD).",
        "",
        "## 2. 보조표 행 (BayesSeg-prefix | past-only, non-LLM)",
        "",
        "| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) ↓ | calls/turn | tok/turn |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_dial']}/{r['n_turns']} | {g(r['pk'])} | "
            f"{g(r['wd'])} | {g(r['f1'])} | {g(r['score'])} | "
            f"{g(r['mean'], 1)} | 0 | 0 |")
    L += [
        "",
        "## 3. latency 분포 (ms/turn, persistent JVM, warmup 제외)",
        "",
        "| dataset | n | mean | p50 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {g(r['n'],0)} | {g(r['mean'],1)} | "
                 f"{g(r['p50'],1)} | {g(r['p95'],1)} | {g(r['max'],1)} |")
    L += [
        "",
        "## 4. 한계 / 정직성",
        "- **AUXILIARY only**: 핵심 5행표 비교용 아님 (codex).",
        "- **Pk/WD/F1 = INDICATIVE**: native-K 가 짧은 prefix 에서 과분할"
        "(거의 매 발화 경계) 경향 → 분할 점수 신뢰 낮음. K 정책 민감.",
        "- **latency 는 의미 있음**: persistent JVM 로 JVM cold-start "
        "(~700ms subprocess) 제거 → per-turn ≈ 분할 compute (수 ms). "
        "warmup-drop 으로 JIT 첫 콜 제외.",
        "- prefix-causal: 원 offline BayesSeg 점수/조건 아님 (Plain/"
        "TextTiling online 과 동일 원칙).",
        "- 알고리즘(DPSeg/BayesWrapper) 코드 불변 — OnlineSegServer 는 "
        "I/O 루프 harness 추가일 뿐. ant 컴파일 (java 11).",
        "- small-n 표본. 비결정성: DP 자체는 결정적이나 native-K prior "
        "민감.",
    ]
    out = exp_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
