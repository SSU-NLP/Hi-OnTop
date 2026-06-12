# TIAGE test HP sweep — v3.3.9 (target: WD/F1/Pk)

n_convs=100 · n_turns=1564 · n_shifts=315

grid: {'eta_prev': [1.0, 0.7, 0.5, 0.0]}

swept HP = ['eta_prev'] · seeds=[0] · 4/4 configs aggregated (incremental — partial-safe).

GT n_topics/conv ≈ 4.15. **Target = WD↓/F1↑/Pk↓** (literature-comparable; user 2026-05-18, supersedes ARI-primary). Rows sorted by F1↓. **ARI/n_topics/collapse = guard** (F1 gamed by every-turn split; † = degenerate collapse≥50% EXCLUDED from best).

| method | eta_prev | F1 ↑ (m±s) | WD ↓ | Pk ↓ | ARI (guard) | n_topics | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3.3.9 | 1 | 0.437 ± 0.000 | 0.605 | 0.415 | 0.408 | 6.9 | 0% |
| v3.3.9 | 0.7 | 0.427 ± 0.000 | 0.616 | 0.421 | 0.400 | 6.9 | 0% |
| v3.3.9 | 0.5 | 0.423 ± 0.000 | 0.629 | 0.434 | 0.378 | 6.9 | 1% |
| v3.3.9 | 0 | 0.421 ± 0.000 | 0.636 | 0.437 | 0.369 | 7.1 | 0% |

**best F1 (target)**: `v3.3.9` {'eta_prev': 1.0} — 0.437 (WD 0.605, Pk 0.415, n_topics 6.9, ARI 0.408)
**best WD (target)**: `v3.3.9` {'eta_prev': 1.0} — 0.605 (F1 0.437, n_topics 6.9)
**best Pk (target)**: `v3.3.9` {'eta_prev': 1.0} — 0.415
