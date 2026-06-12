"""Aggregate per-method JSON outputs into the final REPORT.md tables.

Reads:
- delta_star_calibration.json
- latency_ours.json, latency_baseline.json
- metrics_ours.json, metrics_baseline.json
- (optional) segment_compare.json

Writes a markdown snippet that can be pasted into REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def fmt(v, p=2):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{p}f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp_dir",
        default="outputs/experiments/2026-05-21_v413_secom_swap",
    )
    args = ap.parse_args()

    d = Path(args.exp_dir)
    lat_o = load(d / "latency_ours.json")
    lat_b = load(d / "latency_baseline.json")
    met_o = load(d / "metrics_ours.json")
    met_b = load(d / "metrics_baseline.json")
    calib = load(d / "delta_star_calibration.json")
    seg_cmp = load(d / "segment_compare.json")

    lines = []
    lines.append("## Auto-generated summary\n")

    # δ*
    if calib:
        lines.append("### δ* calibration (mpnet)\n")
        cs = calib.get("delta_prev", {})
        cand = calib.get("candidate_delta_star", {})
        lines.append(
            f"- n_sessions={calib.get('n_sessions')}, "
            f"n_delta_samples={calib.get('n_delta_samples')}, "
            f"recommended δ*={fmt(calib.get('recommended_initial'), 4)} (p80)"
        )
        lines.append(
            "- δ_prev: mean=" + fmt(cs.get("mean"), 4)
            + ", p50=" + fmt(cs.get("p50"), 4)
            + ", p70=" + fmt(cand.get("p70"), 4)
            + ", p80=" + fmt(cand.get("p80"), 4)
            + ", p85=" + fmt(cand.get("p85"), 4)
            + ", p90=" + fmt(cand.get("p90"), 4) + "\n"
        )

    # Segment + latency table
    lines.append("### Latency\n")
    lines.append("| method | n_seg | avg ex/seg | encode (s) | total (s) | ms/ex |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, lat in [("baseline (gpt-4o-mini)", lat_b), ("ours (v4.1.3)", lat_o)]:
        if not lat:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        n_seg = lat.get("n_segments")
        n_ex = lat.get("n_exchanges")
        avg = (n_ex / n_seg) if n_seg else None
        enc = lat.get("encode_sec", 0)
        tot = lat.get("total_sec")
        ms = lat.get("ms_per_exchange") or lat.get("ms_per_exchange_total")
        lines.append(
            f"| {label} | {fmt(n_seg, 0)} | {fmt(avg, 2)} | "
            f"{fmt(enc, 1)} | {fmt(tot, 1)} | {fmt(ms, 2)} |"
        )

    # Speedup
    if lat_o and lat_b:
        b_ms = lat_b.get("ms_per_exchange") or 0
        o_ms = lat_o.get("ms_per_exchange") or lat_o.get("ms_per_exchange_total") or 0
        if o_ms > 0:
            lines.append(f"\n**Speedup**: {b_ms / o_ms:.1f}× (baseline {b_ms:.1f} ms/ex → ours {o_ms:.2f} ms/ex)\n")

    # Downstream QA
    lines.append("\n### Downstream QA\n")
    lines.append("| method | QA F1 | Subspan EM | ROUGE-L | BERTScore-F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, m in [("baseline", met_b), ("ours", met_o)]:
        if not m:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | "
            f"{fmt(m.get('qa_f1_score'), 2)} | "
            f"{fmt(m.get('best_subspan_em'), 2)} | "
            f"{fmt(m.get('rouge_l_f1'), 2)} | "
            f"{fmt(m.get('bertscore_f1'), 2)} |"
        )
    if met_o and met_b:
        deltas = []
        for k in ("qa_f1_score", "best_subspan_em", "rouge_l_f1", "bertscore_f1"):
            a = met_o.get(k); b = met_b.get(k)
            if a is None or b is None:
                deltas.append("—")
            else:
                deltas.append(f"{a - b:+.2f}")
        lines.append(f"| **Δ (ours − baseline)** | {deltas[0]} | {deltas[1]} | {deltas[2]} | {deltas[3]} |")

    # Segment compare
    if seg_cmp.get("overall"):
        ov = seg_cmp["overall"]
        lines.append("\n### Segment placement agreement (ours vs baseline)\n")
        lines.append(f"- position_agreement = {fmt(ov.get('position_agreement'), 3)}")
        lines.append(f"- boundary F1 = {fmt(ov.get('boundary_f1'), 3)}")
        lines.append(f"- n_segments: baseline={ov.get('n_seg_a_total')}, ours={ov.get('n_seg_b_total')}")

    out = "\n".join(lines)
    print(out)
    snippet_path = d / "summary_snippet.md"
    snippet_path.write_text(out)
    print(f"\n\nwrote {snippet_path}")


if __name__ == "__main__":
    main()
