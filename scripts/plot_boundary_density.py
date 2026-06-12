"""EMNLP paper-ready boundary density figures.

Produces:
  outputs/figures/boundary_density_downstream.pdf  (Long-MT-Bench+)
  outputs/figures/boundary_density_dts_tiage.pdf
  outputs/figures/boundary_density_dts_dialseg711.pdf
  outputs/figures/boundary_density_dts_superdialseg.pdf

Strategy: manual radial label placement informed by cluster geometry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
})

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLOR = {
    "unsup":    "#888888",
    "sup":      "#e08214",
    "llm":      "#4a90d9",
    "ours":     "#d62728",
    "ours_mp":  "#d62728",
    "ours_int": "#d62728",
}
CAT_MARKER = {
    "unsup":    "s",
    "sup":      "D",
    "llm":      "o",
    "ours":     "*",
    "ours_mp":  "*",
    "ours_int": "P",
}
CAT_LABEL = {
    "unsup":    "Unsup. baseline",
    "sup":      "Sup. baseline",
    "llm":      "LLM-based",
    "ours":     "Ours (Hi-OnTop)",
    "ours_mp":  "Ours — MPNet",
    "ours_int": "Ours — MiniLM-int8",
}

# ──────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────

def _draw_label(ax, x, y, label, dx, dy, color, weight, fontsize=8):
    """Place a label at offset (dx, dy) from (x, y) with a leader line."""
    ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
    va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
    ax.annotate(
        label, xy=(x, y), xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize, color=color, weight=weight,
        ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.22",
                    facecolor="white", edgecolor="none", alpha=0.94),
        arrowprops=dict(arrowstyle="-", color="#888",
                          lw=0.5, alpha=0.75,
                          shrinkA=2, shrinkB=2),
        zorder=5,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1) Downstream — Long-MT-Bench+ (16 points, GPT4Score y-axis)
# ──────────────────────────────────────────────────────────────────────────

# (label, density, GPT4Score, category, dx, dy)
DOWNSTREAM = [
    ("CSM-Style",        0.100, 64.24, "unsup",  ( 9,  4)),
    ("GraphSeg-Style",   0.106, 62.53, "unsup",  ( 9, -4)),
    ("GreedySeg-Style",  0.197, 68.58, "unsup",  ( 9,  4)),
    ("TextTiling-Style", 0.297, 73.47, "unsup",  (-9, -10)),
    ("RoBERTa",          0.372, 74.44, "sup",    ( 9, -2)),
    ("GPT-4o-mini",      0.426, 78.13, "llm",    ( 9, -2)),
    ("GPT-5",            0.310, 80.62, "llm",    (-9,  4)),
    ("Qwen3.5-122B",     0.322, 80.83, "llm",    ( -2, 14)),
    ("Qwen3.5-27B",      0.328, 81.28, "llm",    ( 9, 8)),
    ("Qwen3.5-4B",       0.302, 76.77, "llm",    ( -9, -10)),
    ("Qwen3.5-2B",       0.592, 72.81, "llm",    (-9, 4)),
    ("Llama3.2-3B",      0.340, 71.60, "llm",    ( 9, -2)),
    ("Mistral3-3B",      0.372, 76.91, "llm",    (-9,  4)),
    ("Ours (p60)",       0.426, 78.75, "ours",   ( 11, 6)),
    ("Ours (p70)",       0.336, 79.90, "ours",   ( 14, -10)),
    ("Ours (p80)",       0.242, 75.87, "ours",   (-11,  6)),
]


def plot_downstream_scatter() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # best-p (highest GPT4Score among Ours percentile variants)
    best_label = max(
        (m for m in DOWNSTREAM if m[3] == "ours"),
        key=lambda m: m[2],
    )[0]

    for cat in ("unsup", "sup", "llm", "ours"):
        xs = [m[1] for m in DOWNSTREAM if m[3] == cat and m[0] != best_label]
        ys = [m[2] for m in DOWNSTREAM if m[3] == cat and m[0] != best_label]
        ax.scatter(xs, ys, c=CAT_COLOR[cat], marker=CAT_MARKER[cat],
                    s=190 if cat == "ours" else 65,
                    edgecolor="black", linewidth=0.6,
                    label=CAT_LABEL[cat], zorder=4, alpha=0.95)

    # best-p halo
    for m in DOWNSTREAM:
        if m[0] != best_label:
            continue
        x, y, cat = m[1], m[2], m[3]
        ax.scatter([x], [y], s=380, c="none",
                    edgecolor="#ffb700", linewidth=2.4, zorder=5)
        ax.scatter([x], [y], s=190, c=CAT_COLOR[cat],
                    marker=CAT_MARKER[cat], edgecolor="black",
                    linewidth=0.7, zorder=6, alpha=0.98)

    for label, x, y, cat, (dx, dy) in DOWNSTREAM:
        weight = "bold" if cat == "ours" else "normal"
        color = CAT_COLOR[cat] if cat == "ours" else "#202020"
        _draw_label(ax, x, y, label, dx, dy, color, weight, fontsize=7.6)

    # sweet-spot ellipse
    cx, cy = 0.324, 80.66
    ell = Ellipse((cx, cy), width=0.043, height=2.0,
                   facecolor="none", edgecolor="#d62728",
                   linewidth=1.1, linestyle=(0, (4, 2)), alpha=0.6, zorder=1)
    ax.add_patch(ell)

    ax.annotate("Sweet-spot cluster\n(LLM top-3 + Ours-p70)",
                xy=(cx + 0.022, cy - 0.4), xytext=(0.44, 83.0),
                fontsize=7.5, color="#d62728", style="italic", weight="bold",
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                  linewidth=0.7, alpha=0.85),
                zorder=6)

    ax.set_xlabel("Predicted boundary density  (# boundaries / # turns)")
    ax.set_ylabel("Downstream QA  (GPT4Score)")
    ax.set_xlim(0.06, 0.65)
    ax.set_ylim(60, 84.5)
    ax.grid(True, alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    best_handle = Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor="white",
                          markeredgecolor="#ffb700",
                          markeredgewidth=2.0,
                          markersize=11,
                          label="Best $p_x$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(best_handle)
    labels.append(best_handle.get_label())
    ax.legend(handles, labels,
               loc="lower right", frameon=True, framealpha=0.97,
               edgecolor="#bbb", borderpad=0.85, handletextpad=0.55,
               labelspacing=1.4)
    ax.set_title("Boundary density vs downstream QA on Long-MT-Bench+",
                  pad=8)

    plt.tight_layout()
    out = FIG_DIR / "figure_D_boundary_density_downstream.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────
# 2) DTS — per-bench scatter with manual radial label placement
# ──────────────────────────────────────────────────────────────────────────

# data: (label, category, T_dens, T_score, D_dens, D_score, S_dens, S_score)
DTS = [
    ("TextTiling-Style", "unsup",
     0.195, 0.344,  0.226, 0.395,  0.202, 0.399),
    ("GraphSeg-Style",   "unsup",
     0.164, 0.374,  0.213, 0.450,  0.097, 0.312),
    ("GreedySeg-Style",  "unsup",
     0.147, 0.298,  0.188, 0.491,  0.153, 0.384),
    ("CSM-Style",        "unsup",
     0.176, 0.430,  0.201, 0.476,  0.176, 0.443),
    ("Ours (MPNet, p60)",    "ours_mp",
     0.373, 0.457,  0.384, 0.509,  0.388, 0.463),
    ("Ours (MPNet, p70)",    "ours_mp",
     0.265, 0.458,  0.289, 0.575,  0.289, 0.458),
    ("Ours (MPNet, p80)",    "ours_mp",
     0.170, 0.431,  0.193, 0.616,  0.187, 0.424),
    ("Ours (MPNet, sup)",    "ours_mp",
     0.235, 0.453,  0.152, 0.630,  0.460, 0.463),
    ("Ours (MPNet, oracle)", "ours_mp",
     0.320, 0.473,  0.152, 0.630,  0.355, 0.464),
    ("Ours (MiniLM-int8, p60)",    "ours_int",
     0.380, 0.470,  0.382, 0.503,  0.365, 0.423),
    ("Ours (MiniLM-int8, p70)",    "ours_int",
     0.285, 0.489,  0.286, 0.566,  0.272, 0.401),
    ("Ours (MiniLM-int8, p80)",    "ours_int",
     0.196, 0.493,  0.191, 0.607,  0.174, 0.365),
    ("Ours (MiniLM-int8, sup)",    "ours_int",
     0.263, 0.484,  0.150, 0.615,  0.470, 0.434),
    ("Ours (MiniLM-int8, oracle)", "ours_int",
     0.284, 0.489,  0.150, 0.616,  0.512, 0.436),
]

GOLD = {"TIAGE": 0.201, "Dialseg711": 0.147, "SuperDialseg": 0.251}

# manual offsets per (label, bench_idx) — fan placement around clusters
# (dx, dy) in points; positive dx → right, positive dy → up
DTS_OFFSET = {
    # ─── TIAGE (idx 0) ─────────────────────────────────────────────
    # baselines (lower left)
    ("TextTiling-Style", 0): ( -8,  -16),
    ("GraphSeg-Style",   0): ( -8,    6),
    ("GreedySeg-Style",  0): (  8,   -8),
    ("CSM-Style",        0): (-14,    8),
    # Ours cluster ~(0.17-0.38, 0.43-0.49) — spread radially
    ("Ours (MPNet, p80)",          0): (-50,    8),
    ("Ours (MiniLM-int8, p80)",    0): ( -10,  28),
    ("Ours (MPNet, sup)",          0): (-22,   -22),
    ("Ours (MPNet, p70)",          0): (-12,   -22),
    ("Ours (MiniLM-int8, sup)",    0): (-30,   38),
    ("Ours (MiniLM-int8, p70)",    0): ( 18,   34),
    ("Ours (MiniLM-int8, oracle)", 0): (-50,   28),
    ("Ours (MPNet, oracle)",       0): (-15,   22),
    ("Ours (MPNet, p60)",          0): ( 35,  -22),
    ("Ours (MiniLM-int8, p60)",    0): ( 50,   22),
    # ─── Dialseg711 (idx 1) ────────────────────────────────────────
    # baselines (low right)
    ("TextTiling-Style", 1): ( 30,  -10),
    ("GraphSeg-Style",   1): ( 25,    8),
    ("GreedySeg-Style",  1): (-30,    8),
    ("CSM-Style",        1): ( 30,    0),
    # Ours - heavy cluster at (0.15, 0.61-0.63) — 4 stacked: M-sup, M-oracle, I-sup, I-oracle
    ("Ours (MPNet, sup)",          1): ( 40,   30),
    ("Ours (MPNet, oracle)",       1): ( 35,   12),
    ("Ours (MiniLM-int8, sup)",    1): (-30,   38),
    ("Ours (MiniLM-int8, oracle)", 1): (-50,   12),
    # Ours p80 (0.19, 0.61)
    ("Ours (MPNet, p80)",          1): ( 50,    8),
    ("Ours (MiniLM-int8, p80)",    1): (-22,   -22),
    # Ours p70 (0.29, 0.57)
    ("Ours (MPNet, p70)",          1): (-22,   22),
    ("Ours (MiniLM-int8, p70)",    1): ( 30,  -12),
    # Ours p60 (0.38, 0.51)
    ("Ours (MPNet, p60)",          1): ( 18,   18),
    ("Ours (MiniLM-int8, p60)",    1): ( 20,  -18),
    # ─── SuperDialseg (idx 2) ───────────────────────────────────────
    ("TextTiling-Style", 2): (-12,   -8),
    ("GraphSeg-Style",   2): ( 8,    4),
    ("GreedySeg-Style",  2): (-20,    8),
    ("CSM-Style",        2): (  8,    8),
    # Ours points are well-spread on SuperDialseg
    ("Ours (MPNet, p80)",          2): (-22,    8),
    ("Ours (MiniLM-int8, p80)",    2): ( -8,  -22),
    ("Ours (MPNet, p70)",          2): ( -8,   18),
    ("Ours (MiniLM-int8, p70)",    2): ( 18,  -12),
    ("Ours (MiniLM-int8, p60)",    2): ( -8,   -22),
    ("Ours (MPNet, p60)",          2): ( 18,    8),
    ("Ours (MPNet, oracle)",       2): (-12,   18),
    ("Ours (MiniLM-int8, sup)",    2): ( -8,   -22),
    ("Ours (MPNet, sup)",          2): ( 22,    8),
    ("Ours (MiniLM-int8, oracle)", 2): ( 18,  -12),
}

DTS_AXIS = {
    "TIAGE":        dict(xlim=(0.05, 0.55), ylim=(0.25, 0.58)),
    "Dialseg711":   dict(xlim=(0.05, 0.50), ylim=(0.35, 0.72)),
    "SuperDialseg": dict(xlim=(0.03, 0.60), ylim=(0.25, 0.55)),
}

# best percentile per (bench, encoder) — from data within {p60, p70, p80}
DTS_BEST_P = {
    ("TIAGE",        "MPNet"): "Ours (MPNet, p70)",
    ("TIAGE",        "int8"):  "Ours (MiniLM-int8, p80)",
    ("Dialseg711",   "MPNet"): "Ours (MPNet, p80)",
    ("Dialseg711",   "int8"):  "Ours (MiniLM-int8, p80)",
    ("SuperDialseg", "MPNet"): "Ours (MPNet, p60)",
    ("SuperDialseg", "int8"):  "Ours (MiniLM-int8, p60)",
}


def _place_labels_greedy(
    ax,
    fig,
    items,
    *,
    fontsize=7.8,
    marker_radius_px=14,
    label_pad_px=3,
    cluster_radius_px=55,
    forbidden_x_lines=None,
):
    """Greedy short-leader-line placement, *cluster-aware*.

    Key behaviour:
      - Cluster nearby data points (within `cluster_radius_px`).
      - For each cluster, pick ONE escape direction (toward emptiest side)
        and stack all cluster labels along that direction so nearby points
        → nearby labels.
      - Isolated points keep individual short leaders.
      - Strict collision checks against markers, legend, other labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    ax_bbox = ax.bbox

    pts = [ax.transData.transform((x, y)) for _, x, y, _, _ in items]
    n = len(items)

    leg = ax.get_legend()
    legend_box = None
    if leg is not None:
        lb = leg.get_window_extent(renderer)
        legend_box = (lb.x0 - 6, lb.y0 - 6, lb.x1 + 6, lb.y1 + 6)

    # forbidden vertical zones (e.g., gold-density line) — convert to display
    forbidden_boxes = []
    if forbidden_x_lines:
        for x_data, half_width_px in forbidden_x_lines:
            xpix, _ = ax.transData.transform((x_data, 0))
            forbidden_boxes.append(
                (xpix - half_width_px, ax_bbox.y0,
                 xpix + half_width_px, ax_bbox.y1)
            )

    def rect_overlap(a, b):
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def inside_axes(bx):
        return (bx[0] >= ax_bbox.x0 + 2 and bx[2] <= ax_bbox.x1 - 2
                 and bx[1] >= ax_bbox.y0 + 2 and bx[3] <= ax_bbox.y1 - 2)

    sizes = []
    for (label, _, _, _, weight) in items:
        probe = ax.text(0, 0, label, fontsize=fontsize, weight=weight,
                          ha="center", va="center")
        bb = probe.get_window_extent(renderer)
        sizes.append((bb.width + 2 * label_pad_px,
                       bb.height + 2 * label_pad_px))
        probe.remove()

    # Cluster via simple union-find on cluster_radius_px proximity
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            if dx * dx + dy * dy < cluster_radius_px ** 2:
                union(i, j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    groups = list(clusters.values())

    DIRS = list(range(0, 360, 15))
    DISTS_INDIV = [16, 22, 28, 36, 44, 54, 66, 80, 98, 120, 150]
    placed = []  # placed label bboxes
    placements = [None] * n

    # marker boxes (all)
    all_marker_boxes = [
        (mx - marker_radius_px, my - marker_radius_px,
         mx + marker_radius_px, my + marker_radius_px)
        for (mx, my) in pts
    ]

    def best_individual(idx, exclude_label_boxes):
        px, py = pts[idx]
        w, h = sizes[idx]
        own = (px - 6, py - 6, px + 6, py + 6)
        for dist in DISTS_INDIV:
            for ang in DIRS:
                rad = np.deg2rad(ang)
                cx = px + dist * np.cos(rad)
                cy = py + dist * np.sin(rad)
                bx = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if not inside_axes(bx):
                    continue
                if rect_overlap(bx, own):
                    continue
                if legend_box is not None and rect_overlap(bx, legend_box):
                    continue
                if any(rect_overlap(bx, fb) for fb in forbidden_boxes):
                    continue
                if any(rect_overlap(bx, mb)
                       for k, mb in enumerate(all_marker_boxes) if k != idx):
                    continue
                if any(rect_overlap(bx, pb) for pb in exclude_label_boxes):
                    continue
                return (cx, cy, bx)
        return None

    def stack_cluster(indices, exclude_label_boxes):
        """Place all labels of one cluster, stacked along one escape side.

        Tries 8 fan directions × multiple base distances + small jitters.
        Returns dict idx -> (cx, cy, bx) or None.
        """
        if len(indices) == 1:
            res = best_individual(indices[0], exclude_label_boxes)
            return {indices[0]: res} if res else None

        cx0 = np.mean([pts[i][0] for i in indices])
        cy0 = np.mean([pts[i][1] for i in indices])

        # Order labels: tallest/widest first centered, then alternate
        ord_idx = sorted(indices, key=lambda i: -sizes[i][0])

        # 8 fan directions to try, prefer first
        FAN_DIRS = [180, 90, 0, 270, 135, 225, 45, 315]
        for fan_ang in FAN_DIRS:
            rad = np.deg2rad(fan_ang)
            ux, uy = np.cos(rad), np.sin(rad)

            # gap depends on max label height + a few px
            max_h = max(sizes[i][1] for i in indices)
            max_w = max(sizes[i][0] for i in indices)

            for base_dist in (40, 56, 72, 92, 116):
                # candidate placements: each label stacked perpendicular to fan
                # along axis perpendicular to (ux, uy)
                px_perp, py_perp = -uy, ux
                placements_try = {}
                ok = True
                local_placed = list(exclude_label_boxes)
                count = len(ord_idx)
                offsets = np.arange(count) - (count - 1) / 2
                # iterate in order: closest-to-center label first
                offsets = sorted(range(count), key=lambda k: abs(offsets[k]))
                for k_idx in offsets:
                    i = ord_idx[k_idx]
                    w, h = sizes[i]
                    # vertical (perpendicular) offset
                    off_perp = (k_idx - (count - 1) / 2) * (max_h + 4)
                    cx = cx0 + base_dist * ux + off_perp * px_perp
                    cy = cy0 + base_dist * uy + off_perp * py_perp
                    bx = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    if not inside_axes(bx):
                        ok = False; break
                    if legend_box is not None and rect_overlap(bx, legend_box):
                        ok = False; break
                    if any(rect_overlap(bx, fb) for fb in forbidden_boxes):
                        ok = False; break
                    if any(rect_overlap(bx, mb) for mb in all_marker_boxes):
                        ok = False; break
                    if any(rect_overlap(bx, pb) for pb in local_placed):
                        ok = False; break
                    placements_try[i] = (cx, cy, bx)
                    local_placed.append(bx)
                if ok:
                    return placements_try
        # fallback: individual placement
        out = {}
        local_placed = list(exclude_label_boxes)
        for i in sorted(indices, key=lambda k: -(sizes[k][0] * sizes[k][1])):
            res = best_individual(i, local_placed)
            if res is None:
                cx, cy = pts[i][0] + 22, pts[i][1] + 22
                w, h = sizes[i]
                res = (cx, cy, (cx - w / 2, cy - h / 2,
                                   cx + w / 2, cy + h / 2))
            out[i] = res
            local_placed.append(res[2])
        return out

    # Place clusters in order: largest cluster first (it needs most room)
    groups_sorted = sorted(groups, key=lambda g: -len(g))
    for g in groups_sorted:
        result = stack_cluster(g, placed)
        if result is None:
            continue
        for i, (cx, cy, bx) in result.items():
            placements[i] = (cx, cy)
            placed.append(bx)

    # any missing -> fallback short leader
    for i in range(n):
        if placements[i] is None:
            res = best_individual(i, placed)
            if res is None:
                cx, cy = pts[i][0] + 24, pts[i][1] + 24
                w, h = sizes[i]
                bx = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            else:
                cx, cy, bx = res
            placements[i] = (cx, cy)
            placed.append(bx)

    for i, ((label, x, y, color, weight), (cx, cy)) in enumerate(zip(items, placements)):
        tx, ty = inv.transform((cx, cy))
        ax.annotate(
            label, xy=(x, y), xytext=(tx, ty),
            textcoords="data",
            fontsize=fontsize, color=color, weight=weight,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.20",
                        facecolor="white", edgecolor="none", alpha=1.0),
            arrowprops=dict(arrowstyle="-", color="#888",
                              lw=0.5, alpha=0.85,
                              shrinkA=2, shrinkB=2),
            zorder=6,
        )


def plot_dts_per_bench(bench_idx: int, bench_name: str, file_suffix: str) -> None:
    """Per-bench scatter with greedy short-leader-line label placement."""
    d_idx = 2 + bench_idx * 2
    s_idx = 3 + bench_idx * 2

    fig, ax = plt.subplots(figsize=(9.0, 6.4))

    # best-p per encoder for this bench
    best_mp  = DTS_BEST_P[(bench_name, "MPNet")]
    best_int = DTS_BEST_P[(bench_name, "int8")]
    best_set = {best_mp, best_int}

    # plot non-best Ours + baselines normally, then best-p with gold halo
    # split Ours into "best" vs "rest" so we can highlight
    for cat in ("unsup", "ours_mp", "ours_int"):
        xs, ys = [], []
        for row in DTS:
            if row[1] != cat:
                continue
            if row[0] in best_set:
                continue
            xs.append(row[d_idx])
            ys.append(row[s_idx])
        s = 190 if cat.startswith("ours") else 70
        ax.scatter(xs, ys, c=CAT_COLOR[cat], marker=CAT_MARKER[cat],
                    s=s, edgecolor="black", linewidth=0.6,
                    label=CAT_LABEL[cat], zorder=4, alpha=0.95)

    # best-p markers with gold halo (drawn ON TOP)
    for row in DTS:
        if row[0] not in best_set:
            continue
        x, y = row[d_idx], row[s_idx]
        cat = row[1]
        # gold halo behind
        ax.scatter([x], [y], s=380, c="none",
                    edgecolor="#ffb700", linewidth=2.4, zorder=5)
        ax.scatter([x], [y], s=190, c=CAT_COLOR[cat],
                    marker=CAT_MARKER[cat], edgecolor="black",
                    linewidth=0.7, zorder=6, alpha=0.98)

    # one shared legend entry for "best-p per encoder"
    from matplotlib.lines import Line2D
    best_handle = Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor="white",
                          markeredgecolor="#ffb700",
                          markeredgewidth=2.0,
                          markersize=11,
                          label="Best $p_x$ (per encoder)")

    gold = GOLD[bench_name]
    ax.axvline(gold, linestyle="--", linewidth=1.1,
                color="#2ca02c", alpha=0.85, zorder=1)
    lims = DTS_AXIS[bench_name]
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["ylim"])

    # Gold label — at top, 3pt below previous position
    ymax = lims["ylim"][1]
    ax.annotate(f"Gold Boundary Density = {gold:.3f}",
                  xy=(gold, ymax), xytext=(0, -8),
                  textcoords="offset points",
                  ha="center", va="top", fontsize=8.5,
                  color="#2ca02c", weight="bold",
                  bbox=dict(boxstyle="round,pad=0.22",
                              facecolor="white", edgecolor="#2ca02c",
                              linewidth=0.6, alpha=0.98),
                  zorder=10)

    ax.set_xlabel("Predicted boundary density  (# boundaries / # turns)")
    ax.set_ylabel("Segmentation Score")
    ax.grid(True, alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(best_handle)
    labels.append(best_handle.get_label())
    ax.legend(handles, labels,
               loc="lower right", frameon=True, framealpha=0.97,
               edgecolor="#bbb", borderpad=1.0, handletextpad=0.6,
               labelspacing=1.8)
    ax.set_title(
        f"Boundary density vs Segmentation Score — {bench_name}",
        pad=8,
    )

    items = []
    for row in DTS:
        label, cat = row[0], row[1]
        x, y = row[d_idx], row[s_idx]
        weight = "bold" if cat.startswith("ours") else "normal"
        color = CAT_COLOR[cat] if cat.startswith("ours") else "#202020"
        items.append((label, x, y, color, weight))

    # forbid placing labels on top of the gold dashed line
    _place_labels_greedy(
        ax, fig, items, fontsize=7.8,
        forbidden_x_lines=[(gold, 12)],  # ±12 px around vertical line
    )

    plt.tight_layout()
    out = FIG_DIR / f"figure_D_boundary_density_dts_{file_suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


def main() -> None:
    plot_downstream_scatter()
    plot_dts_per_bench(0, "TIAGE",        "tiage")
    plot_dts_per_bench(1, "Dialseg711",   "dialseg711")
    plot_dts_per_bench(2, "SuperDialseg", "superdialseg")


if __name__ == "__main__":
    main()
