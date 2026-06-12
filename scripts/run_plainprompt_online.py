#!/usr/bin/env python3
"""**Plain LLM prompting (online, past-only)** — a SEPARATE baseline.

Original SuperDialseg "Plain Text Prompting" (benchmarks/superdialseg
.../models/llm: ChatGPTSegmenter) is OFFLINE: whole dialogue → 1 LLM
call → "Part i: Ua-Ub". Here we make it ONLINE: at turn ``t`` feed ONLY
``U1..Ut`` (past+current, **no future**) with the *same plain prompt*
and read whether a new part begins at ``Ut`` → the online boundary
decision for turn ``t``. One LLM call per turn.

Because the conditioning is causal (prefix) not joint (full dialogue),
predictions differ from the offline SuperDialseg LLM baseline — this is
a DISTINCT baseline (codex decision-log 2026-05-19). Do NOT merge its
numbers with the offline Plain Text Prompting results.

Fills the comparison-table row  *Plain LLM prompting | past-only* with:
Pk, WD, F1, Score(=0.5·F1+0.25·(1-Pk)+0.25·(1-WD), official SuperDialseg
aggregate), avg latency/turn (ms), LLM calls/turn, tokens/turn.

Goal = per-dataset representative numbers over a ~100-turn sample (NOT a
full benchmark): sample whole dialogues in order until cumulative
utterances ≥ --target-turns. Pk/WD/F1 are *indicative* (small n).

Crash-safe: every turn's {sec, tok, shift, miss} appended to a sidecar
jsonl immediately; re-run resumes (skip done turns, rebuild predictions)
→ no re-cost, survives quota 429. LLM = qwen/qwen3.5-27b via Crts
(--model override).
workers=1 ⇒ one request in flight ⇒ uncontended per-turn latency.

Parsing rule (codex caution #4, must be explicit): model output lines
containing 'Part' → all ``U<n>`` ints; part-start = min(n) of the line.
boundary positions = {start | start>1}. boundary_at(t) = t in that set.
Unparseable output (no 'Part'/'U<n>') → decision False + counted as
parse-miss (reported).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
DEFAULT_MODEL = "qwen/qwen3.5-27b"  # Crts slug; --model override
DATASETS = ["tiage", "dialseg711", "superseg"]


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None
    sys.modules["anthropic"] = m


def _utterance_lines(dialogue: str) -> list[str]:
    """Raw 'speaker: text' lines, gold [BOUNDARY] tokens removed."""
    return [seg for seg in dialogue.split("[NEWLINE]")
            if seg.strip() != "[BOUNDARY]" and seg.strip() != ""]


def _plain_messages(utts: list[str]) -> list[dict]:
    """SuperDialseg ChatGPTSegmenter.create_prompt, verbatim style,
    applied to the PREFIX U1..Ut."""
    content = (
        "Dialogue Segmentation aims to segment a dialogue "
        "D = {U1, U2, ..., Un} into several parts according to their "
        "discussing topics.\nPlease help me to segment the following "
        "dialogue: \n")
    for i, u in enumerate(utts):
        content += f"U{i + 1}: {u}\n"
    content += "\nOutput format: Part i: Ua-Ub\n"
    content += ("\n=====\nOutput example:\nPart 1: U1-U4\nPart 2: U5-U6\n"
                "=====\n")
    return [
        {"role": "system",
         "content": ("You are a helpful assistance to segment give "
                     "dialogues.\nPlease follow the output format.\n"
                     "DO NOT explain.")},
        {"role": "user", "content": content},
    ]


def _boundary_at(raw: str, t: int) -> bool | None:
    """Does a new part begin exactly at utterance t? None=unparseable."""
    starts = []
    for line in raw.splitlines():
        if "part" not in line.lower():
            continue
        nums = [int(x) for x in re.findall(r"U\s*(\d+)", line)]
        if nums:
            starts.append(min(nums))
    if not starts:
        return None
    bnds = {a for a in sorted(set(starts)) if a > 1}
    return t in bnds


def _infer(client, messages, slug):
    last = None
    for i in range(3):
        try:
            return client.chat.completions.create(
                messages=messages, temperature=0.0, model=slug)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {k: float("nan") for k in
                ("n", "mean", "std", "min", "p50", "p90",
                 "p95", "p99", "max", "sum")}
    a = np.asarray(xs, float)
    return dict(
        n=len(a), mean=float(a.mean()), std=float(a.std()),
        min=float(a.min()), p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)), max=float(a.max()),
        sum=float(a.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-19_plainprompt_online")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--target-turns", type=int, default=100)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Crts model slug (default qwen/qwen3.5-27b)")
    args = ap.parse_args()
    MODEL = args.model

    load_dotenv(REPO / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not key or not base_url:
        sys.exit("OPENAI_API_KEY / OPENAI_BASE_URL missing (.env)")

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)
    import src.autoseg as A  # noqa: E402  (metric + data helpers only)
    import openai as _o

    client = _o.OpenAI(api_key=key, base_url=base_url)
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
        print(f"\n=== {ds}: {len(sample)} dialogues, {cum} turns "
              f"(target {args.target_turns}) ===")

        sidecar = exp_dir / f"turns_{ds}.jsonl"
        done = {}
        if sidecar.exists():
            for ln in sidecar.read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    done[(r["id"], r["t"])] = r
        print(f"  resume: {len(done)} turns already done")

        lat, toks, calls = [], [], 0
        preds, labels = [], []
        miss = 0
        fh = open(sidecar, "a")
        for did, dialogue, utts in sample:
            n = len(utts)
            uttr = []
            for t in range(1, n + 1):
                kk = (did, t)
                if kk in done:
                    r = done[kk]
                    if r["t"] != 1:
                        lat.append(r["sec"])
                        toks.append(r["tok"])
                        calls += 1
                    miss += int(r.get("miss", False))
                    uttr.append(bool(r["shift"]))
                    continue
                if t == 1:                       # no boundary by convention
                    rec = {"ds": ds, "id": did, "t": 1, "sec": 0.0,
                           "tok": 0, "shift": False, "miss": False}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    uttr.append(False)
                    continue
                msgs = _plain_messages(utts[:t])
                t0 = time.perf_counter()
                res = _infer(client, msgs, MODEL)
                sec = time.perf_counter() - t0
                raw = res.choices[0].message.content or ""
                u = getattr(res, "usage", None)
                tok = int(getattr(u, "total_tokens", 0) or 0)
                b = _boundary_at(raw, t)
                rec = {"ds": ds, "id": did, "t": t, "sec": sec,
                       "tok": tok, "shift": bool(b) if b is not None
                       else False, "miss": b is None}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                lat.append(sec); toks.append(tok); calls += 1
                miss += int(b is None)
                uttr.append(bool(b) if b is not None else False)
            uttr[0] = False
            pred = A.extract_pred(uttr)
            lbl, _ = A.extract_label(dialogue.split("[NEWLINE]"), True)
            pred, lbl = A.align_pred_label(pred, lbl)
            preds.append(pred)
            labels.append(lbl)
        fh.close()

        m = A.compute_metrics(preds, labels)
        score = 0.5 * m["f1"] + 0.25 * (1 - m["pk"]) + 0.25 * (1 - m["wd"])
        st = _stats(lat)
        n_turn = cum
        rows.append(dict(
            ds=ds, n_dial=len(sample), n_turns=n_turn,
            pk=m["pk"], wd=m["wd"], f1=m["f1"], score=score,
            parse_miss=miss, calls_per_turn=calls / max(1, n_turn),
            tok_per_turn=float(np.mean(toks)) if toks else float("nan"),
            lat_ms_mean=st["mean"] * 1000.0,
            lat_ms_p50=st["p50"] * 1000.0, lat_ms_p95=st["p95"] * 1000.0,
            lat_ms_max=st["max"] * 1000.0, n_calls=st["n"]))
        r = rows[-1]
        print(f"  Pk={r['pk']:.4f} WD={r['wd']:.4f} F1={r['f1']:.4f} "
              f"Score={r['score']:.4f} | lat/turn={r['lat_ms_mean']:.0f}ms "
              f"tok/turn={r['tok_per_turn']:.0f} miss={miss}")

    _write_report(exp_dir, args, rows)


def _write_report(exp_dir: Path, args, rows) -> None:
    def g(v, p=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    L = [
        f"# Plain LLM prompting (online, past-only) — Crts `{args.model}`",
        "",
        "> **별도 baseline.** 원 SuperDialseg Plain Text Prompting(offline,"
        " 대화당 1콜)을 past-only online 으로 개조한 것 — 예측 조건이 "
        "joint→causal 로 달라 **offline 결과와 혼합 금지**. codex "
        "decision-log 2026-05-19.",
        "",
        "## 1. Setup",
        "- **Method**: turn t 에 `U1..Ut`(미래 미관측)만 SuperDialseg "
        "plain 프롬프트(ChatGPTSegmenter 스타일)로 1콜 → Ut 에서 새 part "
        "시작 여부 = online 경계 결정. 턴당 1 LLM 콜.",
        f"- **LLM**: `{args.model}` via Crts (`api.ssunlp.co.kr/v1`). "
        "temperature=0. workers=1 (비경합 per-turn latency). "
        "⚠ Qwen3.5 는 reasoning model → 내부 reasoning 으로 **턴당 토큰·"
        "latency 가 큼**(표에 그대로 반영). 표의 다른 LLM 행과 모델이 "
        "다르면(예: Def-DTS offline=gpt-4o, Ours=qwen3.5-9b) **동일-LLM "
        "비교 아님**을 표 주석에 명시할 것.",
        f"- **표본**: 데이터셋별 누적 발화 ≥ {args.target_turns} 까지 "
        "대화 통째 (small-n indicative, 정식 벤치 아님).",
        "- **Score** = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD) (공식 "
        "SuperDialseg aggregate). metric = autoseg(segeval) Pk/WD + F1.",
        "- **Parsing**: 'Part' 포함 줄의 `U<n>` 최소값=part 시작, "
        ">1 인 시작 = 경계. 파싱 불가 → 경계 False + parse-miss 집계.",
        "- **Crash-safe**: `turns_<ds>.jsonl` 턴마다 append, 재실행 resume.",
        "",
        "## 2. 비교표 행 (Plain LLM prompting | past-only)",
        "",
        "| dataset | n(dial/turn) | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | "
        "lat/turn(ms) ↓ | calls/turn ↓ | tok/turn ↓ | parse_miss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_dial']}/{r['n_turns']} | {g(r['pk'])} "
            f"| {g(r['wd'])} | {g(r['f1'])} | {g(r['score'])} "
            f"| {g(r['lat_ms_mean'], 0)} | {g(r['calls_per_turn'], 2)} "
            f"| {g(r['tok_per_turn'], 0)} | {r['parse_miss']} |")
    L += [
        "",
        "## 3. latency 분포 (ms, per-turn)",
        "",
        "| dataset | n_calls | mean | p50 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n_calls']} | {g(r['lat_ms_mean'],0)} "
                 f"| {g(r['lat_ms_p50'],0)} | {g(r['lat_ms_p95'],0)} "
                 f"| {g(r['lat_ms_max'],0)} |")
    L += [
        "",
        "## 4. 한계 / 정직성",
        "- ≠ 원 SuperDialseg Plain Text Prompting(offline). past-only "
        "causal prefix → 예측·점수 다름. 논문 수치와 비교 불가.",
        "- Pk/WD/F1/Score 는 ~100턴 표본 → **경향 참고용**, 정식 점수 아님.",
        "- calls/turn ≈ 1 (t=1 은 콜 없이 경계=0 규약 → 정확히는 "
        "(#turns−#dial)/#turns). tok/turn = prefix 누적이라 후반 턴 ↑.",
        "- Crts quota 초과 시 429 → resume 로 재개. 1-run, "
        "LLM 비결정성 잔존(temp=0, API seed 없음).",
        "- 표의 다른 행(Offline CSM/Def-DTS, Online CSM, Ours)과 동일 "
        "metric/표본 정의로 채워야 공정 비교.",
    ]
    out = exp_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
