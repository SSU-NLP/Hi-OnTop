#!/usr/bin/env python3
"""Run Def-DTS (ElPlaguister/Def-DTS, ACL'25) on tiage / dialseg711 /
superseg via the project's **Crts** OpenAI-compatible proxy with
``openai/gpt-4o`` — the paper's LLM, reached through the single
project endpoint (no OpenAI-direct, no OpenRouter).

Def-DTS is LLM-prompting (deductive reasoning → utterance-level intent
classification). No checkpoint / no training: one LLM call per dialogue,
parse the structured XML output, score Pk/WD/F1 with the repo's own
``segeval`` pipeline (kept verbatim for literature comparability).

We do **not** edit ``benchmarks/Def-DTS`` (read-only external). All
behavioural fixes are runtime monkeypatches here:

1. ``OpenAI(api_key=...)`` → Crts client (base_url + key injected).
2. ``data_process`` dispatches the API path only when
   ``self.model == 'gpt-4o'`` → we construct with ``model='gpt-4o'``
   (so dispatch + cost branch work) but ``infer`` is overridden to send
   the real Crts slug ``openai/gpt-4o`` and to record per-call latency.
3. ``compute_cost`` / end-of-run ``usage.json`` made defensive (Crts
   may omit token usage).
4. ``anthropic`` stubbed (sonnet path never used).

Per-dialogue latency is collected and reported (mean/p50/p95/max/total).

REPORT → ``outputs/experiments/<name>/REPORT.md`` (CLAUDE.md rule).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DEFDTS = REPO / "benchmarks" / "Def-DTS"
REAL_SLUG = "openai/gpt-4o"  # Crts proxied (verified); paper LLM

# tiage example in main.ipynb uses change_speaker=True; others default.
DOMAINS = {
    "tiage": dict(template="defdts_tiage", change_speaker=True),
    "dialseg711": dict(template="defdts_711", change_speaker=False),
    "superseg": dict(template="defdts_superseg", change_speaker=False),
}

# per-call latency (seconds), filled by patched infer; reset per domain.
_LAT: list[float] = []

# per-dataset context for the resumable data_process patch (set in main()
# before each A.process). save_abs = result json, sidecar_abs = per-call
# latency jsonl (crash-safe, also enables resume).
_CTX: dict = {}


def _sidecar_secs(path: Path) -> list[float]:
    """Read a latency sidecar jsonl → per-call seconds, deduped by id
    (resume appends; last write wins). Authoritative latency source
    (survives crash, unlike in-memory _LAT)."""
    if not path.exists():
        return []
    by_id: dict = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            by_id[r["id"]] = float(r["sec"])
    return list(by_id.values())


def _stub_anthropic() -> None:
    m = types.ModuleType("anthropic")
    m.Anthropic = lambda **kw: None  # constructed, never called
    sys.modules["anthropic"] = m


def _patch(A, real_key: str, base_url: str, workers: int):
    """Monkeypatch the imported autoseg module/class in place."""
    import openai as _o
    from tqdm import tqdm

    A.API_KEY = {"crts": real_key}

    def _crts_client(*a, **k):
        return _o.OpenAI(api_key=real_key, base_url=base_url)

    A.OpenAI = _crts_client

    def patched_infer(self, prompt):
        messages = []
        if self.template == "":
            messages += [{
                "role": "system",
                "content": ("You are a helpful assistance to segment give "
                            "dialogues.\nPlease follow the output format.\n"
                            "DO NOT explain."),
            }]
        messages += self.fewshot_list
        messages += [{"role": "user", "content": prompt}]
        last = None
        for attempt in range(3):  # robustness over a long run
            t0 = time.perf_counter()
            try:
                cc = self.gpt.chat.completions.create(
                    messages=messages, temperature=0.0, model=REAL_SLUG)
                _LAT.append(time.perf_counter() - t0)
                return cc
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        raise last

    A.SegAnnotator.infer = patched_infer

    # Resume- + crash-safe data_process (gpt-4o path — the model we run).
    # - preload existing save json → skip done ids (no re-cost on resume)
    # - per call: append {id,sec,n_turns} to sidecar jsonl (survives crash;
    #   source of truth for latency stats — fixes prior in-mem loss)
    # - periodic atomic save of result json every save_every calls
    # - ThreadPoolExecutor(max_workers=workers); workers=1 ⇒ exactly one
    #   request in flight ⇒ ISOLATED (uncontended) latency.
    _lock = threading.Lock()

    def _n_turns(dialogue: str) -> int:
        return len([l for l in dialogue.split("[NEWLINE]")
                    if l.strip() != "[BOUNDARY]"])

    def resumable_data_process(self, dataset_name, dataset_split,
                               ratio=1.0, token_check_only=False, start=0):
        dataset = self.load_data(dataset_name, dataset_split, ratio, start)
        if token_check_only:
            return
        ctx = _CTX  # {save_abs, sidecar_abs, save_every}
        save_abs = Path(ctx["save_abs"])
        sidecar = Path(ctx["sidecar_abs"])
        every = ctx["save_every"]

        done = set()
        if save_abs.exists():
            try:
                prev = json.loads(save_abs.read_text())
                self.result.update(prev)
                done = set(prev.keys())
            except Exception:
                pass
        rows = [d for d in dataset if d["id"] not in done]
        print(f"  resume: {len(done)} done, {len(rows)} to do")
        sidecar.parent.mkdir(parents=True, exist_ok=True)

        def _atomic_save():
            tmp = save_abs.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.result, ensure_ascii=False,
                                      indent=2))
            tmp.replace(save_abs)

        def work(data):
            prompt = self.fill_prompt(data["dialogue"])
            t0 = time.perf_counter()
            res = self.infer(prompt)
            sec = time.perf_counter() - t0
            return data["id"], res, sec, _n_turns(data["dialogue"])

        n_new = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for did, res, sec, nt in tqdm(ex.map(work, rows),
                                          total=len(rows)):
                content = res.choices[0].message.content
                with _lock:
                    self.result[did] = content
                    self.compute_cost(getattr(res, "usage", None))
                    with open(sidecar, "a") as fh:
                        fh.write(json.dumps(
                            {"id": did, "sec": sec, "n_turns": nt}) + "\n")
                    n_new += 1
                    if n_new % every == 0:
                        _atomic_save()
        with _lock:
            _atomic_save()

    A.SegAnnotator.data_process = resumable_data_process

    _orig_cost = A.SegAnnotator.compute_cost

    def safe_cost(self, usage):
        try:
            if usage is None:
                return 0.0
            return _orig_cost(self, usage)
        except Exception:
            return 0.0

    A.SegAnnotator.compute_cost = safe_cost


def _percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def _latency_from_log(path: str) -> dict:
    """Salvage: parse per-dataset wall / lat mean / p95 from a prior run
    log (the crashed run). p50/max/sum were not persisted → left NaN and
    reported as such (honest gap)."""
    import re

    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    cur = None
    hdr = re.compile(r"^=== (\w+)\s+\(template=")
    res = re.compile(
        r"Pk=[\d.]+ WD=[\d.]+ F1=[\d.]+ \| drop=\d+/(\d+) \| "
        r"wall=(\d+)s lat mean=([\d.]+)s p95=([\d.]+)s")
    for line in p.read_text(errors="ignore").splitlines():
        m = hdr.match(line.strip())
        if m:
            cur = m.group(1)
            continue
        m = res.search(line)
        if m and cur:
            out[cur] = dict(
                n_calls=int(m.group(1)), wall=float(m.group(2)),
                mean=float(m.group(3)), p95=float(m.group(4)),
                p50=float("nan"), max=float("nan"), sum=float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="2026-05-19_defdts_gpt4o_crts")
    ap.add_argument("--datasets", nargs="+",
                    default=["tiage", "dialseg711", "superseg"])
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: cap dialogues per dataset (0 = full)")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent Crts calls (1 = sequential)")
    ap.add_argument("--eval-only", action="store_true",
                    help="no LLM calls: score existing saved result jsons "
                         "+ regenerate REPORT (zero cost; salvage path).")
    ap.add_argument("--latency-log", default="/tmp/defdts_full.log",
                    help="eval-only: parse per-dataset latency from this "
                         "run log if present.")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not args.eval_only and (not key or not base_url):
        sys.exit("OPENAI_API_KEY / OPENAI_BASE_URL missing (.env)")

    log_lat = _latency_from_log(args.latency_log) if args.eval_only else {}

    _stub_anthropic()
    sys.path.insert(0, str(DEFDTS))
    os.chdir(DEFDTS)  # autoseg uses relative prompts/ results/ data/ paths
    (DEFDTS / "results").mkdir(exist_ok=True)
    if not (DEFDTS / "usage.json").exists():
        (DEFDTS / "usage.json").write_text("{}")

    import src.autoseg as A  # noqa: E402  (after stub + path + chdir)

    if not args.eval_only:
        _patch(A, key, base_url, args.workers)

    rows = []
    for ds in args.datasets:
        cfg = DOMAINS[ds]
        tmpl = cfg["template"]
        n_file = sum(1 for _ in open(
            DEFDTS / "data" / "DTS_session_datasets" / f"{ds}_test.jsonl"))
        n = min(args.limit, n_file) if args.limit else n_file
        save_path = f"results/{args.name}_{tmpl}_{ds}.json"

        if args.eval_only and not (DEFDTS / save_path).exists():
            print(f"  [skip] {ds}: no saved result json — not run yet")
            continue

        print(f"\n=== {ds}  (template={tmpl}, n={n}/{n_file}, "
              f"change_speaker={cfg['change_speaker']}, "
              f"eval_only={args.eval_only}) ===")
        _LAT.clear()
        sidecar = (REPO / "outputs" / "experiments" / args.name
                   / f"latency_{ds}.jsonl")
        global _CTX
        _CTX = {"save_abs": str(DEFDTS / save_path),
                "sidecar_abs": str(sidecar), "save_every": 25}
        t0 = time.perf_counter()
        if not args.eval_only:
            A.process(ds, tmpl, n, 0, save_path, "crts",
                      cfg["change_speaker"], [], "gpt-4o")
        wall = (time.perf_counter() - t0) if not args.eval_only else None

        # eval (no LLM): preds/labels → drop count + official metrics
        if args.eval_only:
            saved = json.loads((DEFDTS / save_path).read_text())
            score_len = 1 + max(
                int(k.split("_")[-1]) for k in saved)  # cover all saved ids
        else:
            score_len = n
        preds, labels = A.compute_gpt_performance(
            ds, "test", tmpl, length=score_len, start=0,
            specified_path=save_path, compute_metric=False)
        n_total = len(preds)
        n_drop = sum(1 for p, l in zip(preds, labels)
                     if sum(p) != sum(l))
        metrics = A.compute_metrics(preds, labels)  # repo verbatim

        # latency: sidecar jsonl is authoritative (crash-safe, survives
        # resume); fall back to in-mem _LAT, then to crashed-run log parse.
        lat = _sidecar_secs(sidecar) or list(_LAT)
        ll = log_lat.get(ds, {}) if (args.eval_only and not lat) else {}
        rows.append(dict(
            ds=ds, n_total=n_total, n_scored=n_total - n_drop,
            n_drop=n_drop, pk=metrics["pk"], wd=metrics["wd"],
            f1=metrics["f1"],
            wall=ll.get("wall", wall) if ll else wall,
            n_calls=ll.get("n_calls", len(lat)) if ll else len(lat),
            lat_mean=ll.get("mean", float("nan")) if ll
            else (float(np.mean(lat)) if lat else float("nan")),
            lat_p50=ll.get("p50", float("nan")) if ll
            else _percentile(lat, 50),
            lat_p95=ll.get("p95", float("nan")) if ll
            else _percentile(lat, 95),
            lat_max=ll.get("max", float("nan")) if ll
            else (float(np.max(lat)) if lat else float("nan")),
            lat_sum=ll.get("sum", float("nan")) if ll
            else (float(np.sum(lat)) if lat else 0.0)))
        r = rows[-1]
        wtxt = f"{r['wall']:.0f}s" if r['wall'] is not None else "n/a"
        print(f"  Pk={r['pk']:.4f} WD={r['wd']:.4f} F1={r['f1']:.4f} "
              f"| drop={n_drop}/{n_total} | wall={wtxt} "
              f"lat mean={r['lat_mean']:.2f}s p95={r['lat_p95']:.2f}s")

    _write_report(REPO, args, rows)


def _write_report(repo: Path, args, rows) -> None:
    out = repo / "outputs" / "experiments" / args.name / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"# Def-DTS @ {REAL_SLUG} (Crts) — tiage / dialseg711 / superseg",
        "",
        "## 1. Setup",
        "",
        "- **Method**: Def-DTS (ElPlaguister/Def-DTS, ACL 2025) — LLM "
        "deductive-reasoning prompting, DTS as utterance-level intent "
        "classification. No checkpoint / no training: 1 LLM call/dialogue.",
        f"- **LLM**: `{REAL_SLUG}` via **Crts** proxy "
        "(`api.ssunlp.co.kr/v1`, OpenAI-compatible). Single project "
        "endpoint (not OpenAI-direct / not OpenRouter). temperature=0.0.",
        "- **Data**: bundled `benchmarks/Def-DTS/data/DTS_session_datasets/"
        "<ds>_test.jsonl`. Prompts: `prompts/defdts_{tiage,711,superseg}"
        ".prompt` (full Def-DTS template).",
        "- **Metric**: repo-verbatim `segeval` Pk/WD + sklearn F1 "
        "(`autoseg.compute_metrics`), kept unmodified for literature "
        "comparability. tiage uses change_speaker=True (main.ipynb).",
        "- **Code**: `benchmarks/Def-DTS` untouched (read-only); behaviour "
        "fixes are runtime monkeypatches in `scripts/run_defdts_crts.py`.",
        "",
        "## 2. Results",
        "",
        "| dataset | n (scored/total) | Pk ↓ | WD ↓ | F1 ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        L.append(f"| {r['ds']} | {r['n_scored']}/{r['n_total']} "
                 f"(drop {r['n_drop']}) | {r['pk']:.4f} | {r['wd']:.4f} "
                 f"| {r['f1']:.4f} |")
    L += [
        "",
        "## 3. Latency",
        "",
        f"Per-dialogue LLM call wall time (Crts round-trip incl. queue). "
        f"workers={args.workers} → end-to-end wall ≈ sum/workers; "
        f"mean/p50/p95/max/sum are per-call (concurrency-independent).",
        "",
        "| dataset | n_calls | mean (s) | p50 (s) | p95 (s) | max (s) "
        "| sum (s) | end-to-end wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def g(v, p=2):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) \
            else f"{v:.{p}f}"

    for r in rows:
        L.append(
            f"| {r['ds']} | {r['n_calls']} | {g(r['lat_mean'])} | "
            f"{g(r['lat_p50'])} | {g(r['lat_p95'])} | {g(r['lat_max'])} "
            f"| {g(r['lat_sum'], 0)} | {g(r['wall'], 0)} |")
    L += [
        "",
        "## 4. 해석",
        "",
        "- (실행 후 작성) 숫자가 Def-DTS 논문 보고치(gpt-4o) 와 정합한지, "
        "데이터셋별 패턴, drop 률이 결과에 주는 영향.",
        "",
        "## 5. 한계 / 검증 미해결",
        "",
        "- **LLM 조건**: Def-DTS 는 gpt-4o, Hi-OnTop 은 Qwen3.5-9B. 같은 "
        "Crts 엔드포인트지만 **LLM 이 달라** \"Def-DTS@gpt-4o (논문 설정)\" "
        "대 \"Hi-OnTop@Qwen\" 비교임 (동일-LLM 비교 아님).",
        "- **compute_metrics 의 관용성**: `sum(pred)!=sum(label)` 대화는 "
        "**점수에서 제외**(drop)됨 — 파싱/포맷 실패가 페널티가 아니라 "
        "표본에서 빠짐. drop 률을 함께 보고(위 표) 해 해석 시 반드시 고려.",
        "- **비용/usage**: Crts 가 token usage 를 안 줄 수 있어 비용 추정은 "
        "skip(0). gpt-4o 단가 기준 추정은 별도.",
        "- **재현성**: temperature=0 이나 gpt-4o 비결정성 잔존 가능. "
        "seed 미지원(API). 1-run.",
        "- 데이터: tiage n=99, dialseg711 n=710, superseg n=1321 "
        "(번들 test jsonl 줄 수).",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
