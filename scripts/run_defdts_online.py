#!/usr/bin/env python3
"""**Def-DTS-based-online** — a SEPARATE baseline derived from the
Def-DTS prompt, NOT Def-DTS itself. Do not mix its numbers with
``run_defdts_crts.py`` output.

Difference from Def-DTS (offline, whole-dialogue, 1 LLM call/dialogue):
at turn ``t`` we feed ONLY ``dialogue[0:t]`` (past+current, no future
utterances) into the same Def-DTS template and read the topic_shift of
the **last** utterance = the online decision for turn ``t``. → genuinely
online, one LLM call per turn. Because the conditioning is causal
(prefix) not joint (full dialogue), predictions differ from Def-DTS —
hence a distinct baseline name (decision-log 2026-05-19).

Goal here = per-dataset representative **per-turn latency** over a
~100-turn sample (NOT a full benchmark). We sample whole dialogues in
order until cumulative utterances ≥ ``--target-turns``. Pk/WD/F1 on the
sample are reported as *indicative only* (small n).

Crash-safe: every turn's {sec, shift} is appended to a sidecar jsonl
immediately; re-running resumes (skips done turns, rebuilds predictions
from the sidecar) → no re-cost, survives quota 429 mid-run.

LLM = ``openai/gpt-4o`` via Crts (project endpoint). workers=1 ⇒ one
request in flight ⇒ uncontended per-turn latency.
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
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
REAL_SLUG = "openai/gpt-4o"

DOMAINS = {
    "tiage": dict(template="defdts_tiage", change_speaker=True),
    "dialseg711": dict(template="defdts_711", change_speaker=False),
    "superseg": dict(template="defdts_superseg", change_speaker=False),
}


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None
    sys.modules["anthropic"] = m


def _patch_client(A, key: str, base_url: str):
    import openai as _o

    A.API_KEY = {"crts": key}
    A.OpenAI = lambda *a, **k: _o.OpenAI(api_key=key, base_url=base_url)

    def safe_cost(self, usage):
        return 0.0

    A.SegAnnotator.compute_cost = safe_cost


def _utterance_lines(dialogue: str) -> list[str]:
    """Raw 'speaker: text' lines, gold [BOUNDARY] tokens removed
    (same filter Def-DTS fill_prompt applies)."""
    return [seg for seg in dialogue.split("[NEWLINE]")
            if seg.strip() != "[BOUNDARY]" and seg.strip() != ""]


def _parse_last_shift(raw: str) -> bool | None:
    """Mirror autoseg.parse_output's per-utterance topic_shift scan;
    return the LAST utterance's shift (= decision for turn t).
    None = unparseable (counted as parse-miss → default NO)."""
    v = raw
    if v.startswith("```"):
        v = v.replace("```", "")
    while v and not v.startswith("<") and len(v) > 1:
        v = v[1:]
    shifts: list[bool] = []
    for line in v.strip().split("\n"):
        if "<topic_shift" in line or "<preceding_topical_relation" in line:
            shifts.append("YES" in line.upper())
    if not shifts:
        return None
    return shifts[-1]


def _infer(agent, prompt: str, slug: str):
    """One Crts chat call (retry x3), timed by caller."""
    msgs = []
    if agent.template == "":
        msgs.append({"role": "system", "content": "segment."})
    msgs += agent.fewshot_list
    msgs.append({"role": "user", "content": prompt})
    last = None
    for i in range(3):
        try:
            return agent.gpt.chat.completions.create(
                messages=msgs, temperature=0.0, model=slug)
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
    ap.add_argument("--name", default="2026-05-19_defdts_based_online")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--target-turns", type=int, default=100,
                    help="sample whole dialogues until cumulative "
                         "utterances ≥ this (per dataset)")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not key or not base_url:
        sys.exit("OPENAI_API_KEY / OPENAI_BASE_URL missing (.env)")

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)
    if not (DEFDTS / "usage.json").exists():
        (DEFDTS / "usage.json").write_text("{}")
    import src.autoseg as A  # noqa: E402

    _patch_client(A, key, base_url)

    exp_dir = REPO / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in args.datasets:
        cfg = DOMAINS[ds]
        agent = A.SegAnnotator(key_owner="crts", template=cfg["template"],
                               model="gpt-4o",
                               change_speaker=cfg["change_speaker"])
        # set speaker_map (load_data side effect) without consuming data
        agent.domain = ds
        agent.speaker_map = {"user": "user", "agent": "agent"}
        if ds == "tiage" and cfg["change_speaker"]:
            agent.speaker_map = {"user": "speaker1", "agent": "speaker2"}

        full = A.alternative_load_dataset(ds, "test")
        # sample whole dialogues until cumulative turns ≥ target
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

        sidecar = exp_dir / f"latency_{ds}.jsonl"
        done = {}
        if sidecar.exists():
            for ln in sidecar.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                done[(r["id"], r["t"])] = r
        print(f"  resume: {len(done)} turns already done")

        lat, preds, labels = [], [], []
        fh = open(sidecar, "a")
        for did, dialogue, utts in sample:
            n = len(utts)
            uttr = []
            for t in range(1, n + 1):
                kkey = (did, t)
                if kkey in done:
                    rec = done[kkey]
                    lat.append(rec["sec"])
                    uttr.append(bool(rec["shift"]))
                    continue
                if t == 1:  # first utterance: no boundary by convention
                    rec = {"ds": ds, "id": did, "t": 1, "sec": 0.0,
                           "shift": False, "miss": False}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    uttr.append(False)
                    continue
                prefix = "[NEWLINE]".join(utts[:t])
                prompt = agent.fill_prompt(prefix)
                t0 = time.perf_counter()
                res = _infer(agent, prompt, REAL_SLUG)
                sec = time.perf_counter() - t0
                raw = res.choices[0].message.content
                sh = _parse_last_shift(raw)
                rec = {"ds": ds, "id": did, "t": t, "sec": sec,
                       "shift": bool(sh) if sh is not None else False,
                       "miss": sh is None}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                lat.append(sec)
                uttr.append(bool(sh) if sh is not None else False)
            uttr[0] = False
            pred = A.extract_pred(uttr)
            lbl, _ = A.extract_label(dialogue.split("[NEWLINE]"), True)
            pred, lbl = A.align_pred_label(pred, lbl)
            preds.append(pred)
            labels.append(lbl)
        fh.close()

        metrics = A.compute_metrics(preds, labels)
        st = _stats(lat)
        miss = sum(1 for v in done.values() if v.get("miss")) if done else 0
        rows.append(dict(ds=ds, n_dial=len(sample), n_turns=cum,
                         pk=metrics["pk"], wd=metrics["wd"],
                         f1=metrics["f1"], parse_miss=miss, **st))
        print(f"  Pk={metrics['pk']:.4f} F1={metrics['f1']:.4f} | "
              f"per-turn lat mean={st['mean']:.2f}s p95={st['p95']:.2f}s "
              f"(n={st['n']})")

    _write_report(exp_dir, args, rows)


def _write_report(exp_dir: Path, args, rows) -> None:
    def g(v, p=2):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    out = exp_dir / "REPORT.md"
    L = [
        "# Def-DTS-based-online — per-turn latency (Crts gpt-4o)",
        "",
        "> **별도 baseline. Def-DTS(offline) 와 다른 방법 — 결과 혼합 금지.** "
        "decision-log 2026-05-19 참조.",
        "",
        "## 1. Setup",
        "",
        "- **Method**: turn `t` 에서 `dialogue[0:t]`(과거+현재, 미래 미관측)만 "
        "Def-DTS 프롬프트(`defdts_{ds}.prompt`)에 넣어 1콜 → 마지막 발화의 "
        "`<topic_shift>` 채택. 턴당 1 LLM 콜 = 진짜 online.",
        "- **vs Def-DTS**: Def-DTS = 대화 전체 1콜 joint 추론(offline). 본 "
        "baseline = causal prefix → 예측·점수 다름. 논문 Def-DTS 수치와 "
        "비교 불가.",
        f"- **LLM**: `{REAL_SLUG}` via Crts (`api.ssunlp.co.kr/v1`). "
        "temperature=0. workers=1 (비경합 per-turn latency).",
        f"- **표본**: 데이터셋별 누적 발화 ≥ {args.target_turns} 까지 "
        "대화 통째 샘플(점수는 small-n indicative, 벤치 아님). "
        "목적 = 데이터셋별 대표 per-turn latency.",
        "- **Crash-safe**: 턴마다 `latency_<ds>.jsonl` 즉시 append; "
        "재실행 시 resume(추가비용 0).",
        "",
        "## 2. Per-turn latency (초)",
        "",
        "| dataset | n_turns | mean | std | min | p50 | p90 | p95 "
        "| p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(
            f"| {r['ds']} | {int(r['n']) if not np.isnan(r['n']) else '—'} "
            f"| {g(r['mean'])} | {g(r['std'])} | {g(r['min'])} "
            f"| {g(r['p50'])} | {g(r['p90'])} | {g(r['p95'])} "
            f"| {g(r['p99'])} | {g(r['max'])} |")
    L += [
        "",
        "## 3. Segmentation (indicative, small-n)",
        "",
        "| dataset | n_dial | n_turns | Pk ↓ | WD ↓ | F1 ↑ | parse_miss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n_dial']} | {r['n_turns']} | "
                 f"{g(r['pk'],4)} | {g(r['wd'],4)} | {g(r['f1'],4)} | "
                 f"{r['parse_miss']} |")
    L += [
        "",
        "## 4. 해석 / 한계",
        "",
        "- 본 표 Pk/WD/F1 은 ~100턴 표본 → **경향 참고용, 벤치 점수 아님**. "
        "정식 점수는 별도 full 실행 필요.",
        "- per-turn latency 는 prefix 가 길어질수록(후반 턴) 토큰↑ 로 증가 "
        "경향 — mean 외 p90/p99 동반 해석.",
        "- Def-DTS-based-online ≠ Def-DTS: causal prefix decoding 으로 "
        "예측이 joint 와 달라짐(보통 열화). Hi-OnTop(online) 과의 latency "
        "비교에만 동일 조건(턴단위) 충족.",
        "- Crps USD quota 초과 시 신규 콜 429 — resume 로 복구 후 재개.",
        "- temperature=0 이나 gpt-4o 비결정성 잔존, API seed 미지원, 1-run.",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
