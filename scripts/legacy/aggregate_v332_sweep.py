#!/usr/bin/env python3
"""Aggregate v3.3.1 HP sweep + v3.3.2 PE sweep + RAG/sliding baselines
into a single Markdown report.

Sources:
    - outputs/sweeps/2026-05-08_v331_hpsweep/<tag>/        — 5 configs
    - outputs/sweeps/2026-05-08_v332_pesweep/<tag>/        — 13 configs
    - outputs/sweeps/2026-05-07_v331_compare/locomo/<m>/   — rag/rag-summary/rag-observation/sliding

Output:
    outputs/sweeps/2026-05-08_v332_pesweep/REPORT.md

Columns per config:
    cos α β [pe] | acc | mh | sh | T1μ T2μ T3μ T1max T2max T1var | gen_p50 | wall_s
plus RAG/sliding baseline rows.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HP_DIR = REPO / "outputs" / "sweeps/2026-05-08_v331_hpsweep"
PE_DIR = REPO / "outputs" / "sweeps/2026-05-08_v332_pesweep"
RAG_DIR = REPO / "outputs" / "sweeps/2026-05-07_v331_compare" / "locomo"
OUT = PE_DIR / "REPORT.md"


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.strip())
    except Exception:
        return None


def wall_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    starts = []
    ends = []
    for line in log_path.open():
        m = re.match(r"=== START (\S+) ===", line)
        if m:
            t = _parse_iso(m.group(1))
            if t: starts.append(t)
            continue
        m = re.match(r"=== END (\S+) ===", line)
        if m:
            t = _parse_iso(m.group(1))
            if t: ends.append(t)
    if starts and ends:
        return (ends[-1] - starts[0]).total_seconds()
    return None


def load_summary(run_dir: Path) -> dict:
    sp = list((run_dir / "results" / "experiments").glob("*/summary.json"))
    return json.loads(sp[0].read_text()) if sp else {}


def load_topk(run_dir: Path) -> dict:
    tk = run_dir / "stm_topk.json"
    return json.loads(tk.read_text()) if tk.exists() else {}


def n_topics_avg(run_dir: Path) -> float:
    rj = run_dir / "stm_topk.rounds.jsonl"
    if not rj.exists():
        return 0.0
    n = []
    for line in rj.open():
        if line.strip():
            try:
                n.append(json.loads(line).get("n_stm_topics", 0))
            except Exception:
                pass
    return sum(n) / len(n) if n else 0.0


def f(v, dp=3):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{dp}f}"
    except Exception:
        return str(v)


def parse_hp_from_log(run_dir: Path) -> dict[str, str]:
    """Parse HP values from the run.log header lines written by the sweep
    script. Handles both formats:
        cos_threshold=0.85  alpha=10  beta=0.5
        cos=0.9 alpha=100 beta=0.25 pe=0.5 lmda=10 prom=0.5 tau=50 train_steps=1
    Returns dict of name → string value (only those found).
    """
    out: dict[str, str] = {}
    log = run_dir / "run.log"
    if not log.exists():
        return out
    aliases = {
        "cos_threshold": "cos", "cos": "cos",
        "alpha": "alpha", "beta": "beta",
        "pe_threshold": "pe", "pe": "pe",
        "lmda": "lmda", "prom": "prom",
        "promotion_threshold": "prom",
        "tau": "tau",
        "train_steps": "train", "rnn_train_steps": "train",
    }
    for line in log.open():
        if "===" in line:
            continue
        # parse any '=' tokens — HP header lines contain them; benign elsewhere.
        for tok in line.replace(",", " ").split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                k = k.strip()
                if k in aliases and aliases[k] not in out:
                    out[aliases[k]] = v.strip().rstrip(",")
    return out


def parse_hp_tag(tag: str) -> tuple[str, str, str, str | None]:
    """Best-effort parse of legacy tags like 'cos0p85_a10_b0p5[_pe0p5]'.
    Returns ('-', '-', '-', None) when nothing matches.
    """
    cos = al = be = "-"
    pe: str | None = None
    if tag.startswith("cos"):
        try:
            cos = tag.split("_")[0].replace("cos", "").replace("p", ".")
        except Exception:
            pass
    if "_a" in tag:
        try:
            al = tag.split("_a")[1].split("_")[0]
        except Exception:
            pass
    if "_b" in tag:
        try:
            be = tag.split("_b")[1].split("_")[0].replace("p", ".")
        except Exception:
            pass
    if "_pe" in tag:
        try:
            pe = tag.split("_pe")[1].split("_")[0].replace("p", ".")
        except Exception:
            pass
    return cos, al, be, pe


def row(run_dir: Path, label: str | None = None) -> str:
    s = load_summary(run_dir)
    tk = load_topk(run_dir)
    t1 = tk.get("top1_per_round", {})
    t2 = tk.get("top2_per_round", {})
    t3 = tk.get("top3_per_round", {})
    # Prefer log-parsed HP (works for new tags); fall back to tag parsing.
    hp = parse_hp_from_log(run_dir)
    if not hp:
        cos, al, be, pe = parse_hp_tag(run_dir.name)
        hp = {"cos": cos, "alpha": al, "beta": be, "pe": pe or "—"}
    cos = hp.get("cos", "-")
    al = hp.get("alpha", "-")
    be = hp.get("beta", "-")
    pe = hp.get("pe", "—")
    lmda = hp.get("lmda", "—")
    prom = hp.get("prom", "—")
    tau = hp.get("tau", "—")
    train = hp.get("train", "—")
    acc = s.get("accuracy_overall", "-")
    mh = s.get("accuracy_by_qtype/multi-hop", "-")
    sh = s.get("accuracy_by_qtype/single-hop", "-")
    tr = s.get("accuracy_by_qtype/temporal-reasoning", "-")
    adv = s.get("accuracy_by_qtype/adversarial", "-")
    od = s.get("accuracy_by_qtype/open-domain", "-")
    gen_p50 = s.get("gen_sec_p50", "-")
    wall = wall_seconds(run_dir / "run.log")
    nt = n_topics_avg(run_dir)
    return (
        f"| {label or run_dir.name} | {cos} | {al} | {be} | {pe} | {lmda} | {prom} | {tau} | {train} | "
        f"{f(acc)} | {f(mh)} | {f(sh)} | {f(tr)} | {f(adv)} | {f(od)} | "
        f"{f(t1.get('mean'),1)} | {f(t2.get('mean'),1)} | {f(t3.get('mean'),1)} | "
        f"{t1.get('max','-')} | {t2.get('max','-')} | {f(t1.get('variance'),1)} | "
        f"{f(nt,2)} | {f(gen_p50,2)} | {f(wall,1)} |"
    )


def baseline_row(run_dir: Path, label: str) -> str:
    s = load_summary(run_dir)
    acc = s.get("accuracy_overall", "-")
    mh = s.get("accuracy_by_qtype/multi-hop", "-")
    sh = s.get("accuracy_by_qtype/single-hop", "-")
    tr = s.get("accuracy_by_qtype/temporal-reasoning", "-")
    adv = s.get("accuracy_by_qtype/adversarial", "-")
    od = s.get("accuracy_by_qtype/open-domain", "-")
    gen_p50 = s.get("gen_sec_p50", "-")
    wall = wall_seconds(run_dir / "run.log")
    return (
        f"| {label} | — | — | — | — | — | — | — | — | "
        f"{f(acc)} | {f(mh)} | {f(sh)} | {f(tr)} | {f(adv)} | {f(od)} | "
        f"— | — | — | — | — | — | — | {f(gen_p50,2)} | {f(wall,1)} |"
    )


def main() -> None:
    PE_DIR.mkdir(parents=True, exist_ok=True)

    header = (
        "| variant | cos | α | β | pe | λ | prom | τ | train | "
        "acc | mh | sh | tr | adv | od | "
        "T1μ | T2μ | T3μ | T1max | T2max | T1var | n_topics_avg | gen_p50(s) | wall(s) |"
    )
    sep = "|" + "---|" * 25

    lines: list[str] = []
    lines.append("# v3.3.1 HP sweep + v3.3.2 PE sweep — LoCoMo (limit 50, stratify)\n")
    lines.append("Generated by `scripts/aggregate_v332_sweep.py`. Date: 2026-05-08.\n")
    lines.append("Sweep wall-clock and gen_p50 are per single config run; T1/T2/T3 are STM "
                 "top-1/2/3 topic turn-counts averaged over rounds. Accuracy is judge F1 "
                 "over 49 questions (`--limit 50 --stratify`).\n")

    # v3.3.1 HP sweep
    lines.append("## v3.3.1 HP sweep — 'mega-topic 깨질 수 있는 HP 영역 탐색'\n")
    lines.append(header)
    lines.append(sep)
    if HP_DIR.exists():
        for d in sorted(HP_DIR.iterdir()):
            if not d.is_dir() or not (d / "exit_code.txt").exists():
                continue
            lines.append(row(d))
    else:
        lines.append("| (empty) |" + " — |" * 19)
    lines.append("")

    # v3.3.2 PE sweep
    lines.append("## v3.3.2 PE sweep — 'surprise hard boundary 효과'\n")
    lines.append(header)
    lines.append(sep)
    if PE_DIR.exists():
        for d in sorted(PE_DIR.iterdir()):
            if not d.is_dir() or not (d / "exit_code.txt").exists():
                continue
            lines.append(row(d))
    else:
        lines.append("| (empty) |" + " — |" * 19)
    lines.append("")

    # RAG / sliding baselines (default HP only — they don't use seg HPs)
    lines.append("## RAG / sliding baselines — 동일 LoCoMo (limit 50 stratify, 2026-05-07)\n")
    lines.append(header)
    lines.append(sep)
    for m, lbl in [
        ("rag", "rag"),
        ("rag_summary", "rag-summary"),
        ("rag_observation", "rag-observation"),
        ("sliding", "sliding"),
    ]:
        d = RAG_DIR / m
        if d.exists():
            lines.append(baseline_row(d, lbl))
    lines.append("")

    # Best summary
    lines.append("## 자동 추출: 각 sweep 의 best (acc 기준)\n")
    def best_in(directory: Path) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        if not directory.exists():
            return None
        for d in sorted(directory.iterdir()):
            if not d.is_dir() or not (d / "exit_code.txt").exists():
                continue
            s = load_summary(d)
            acc = s.get("accuracy_overall")
            if acc is None: continue
            if best is None or float(acc) > best[1]:
                best = (d.name, float(acc))
        return best

    bh = best_in(HP_DIR)
    bp = best_in(PE_DIR)
    if bh:
        lines.append(f"- **v3.3.1 HP best**: `{bh[0]}` — acc = {bh[1]:.3f}")
    if bp:
        lines.append(f"- **v3.3.2 PE best**: `{bp[0]}` — acc = {bp[1]:.3f}")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"report → {OUT}")


if __name__ == "__main__":
    main()
