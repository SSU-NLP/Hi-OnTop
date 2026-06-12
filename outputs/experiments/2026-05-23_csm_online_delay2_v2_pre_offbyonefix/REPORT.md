# CSM-online-delay2 (3-benchmark)

backbone=bert-base-uncased · ckpt=methods/CSM/cpt_277000.pth · alpha=1.0 · delay=2 · min_gap=2 · device=cpu
데이터 = Def-DTS 번들 test. metric = segeval Pk/WD + boundary-set F1. Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).
cold-start (CSM/BERT load) = 22.68s

| dataset | n_dial | n_turn | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/turn(ms) | p95 | pred bs | gold bs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 1564 | 0.5214 | 0.5497 | 0.0981 | 0.2813 | 267.20 | 360.24 | 275 | 315 |
| dialseg711 | 711 | 18639 | 0.4713 | 0.5035 | 0.0924 | 0.3025 | 293.67 | 469.90 | 3753 | 2743 |
| superseg | 1322 | 16006 | 0.5334 | 0.5387 | 0.0778 | 0.2709 | 288.98 | 445.99 | 2821 | 4017 |

## 한계
- ckpt 호환성 — 본 wrapper 는 lxing532 `CoherenceNet` 가중치 (`bert-base-uncased` backbone) 전제. SuperDialseg 의 CSM ckpt 와는 모듈 구조가 달라 호환 안 됨.
- strict prefix-causal 아님 — `delay=2` right context 의존. 원본 score/HP 보존, *emission* 만 delay-2 (decision-log 2026-05-21 GreedySeg-online-delay2 entry 와 동일 framing).
