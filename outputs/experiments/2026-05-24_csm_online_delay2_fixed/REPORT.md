# CSM-online-delay2 (3-benchmark)

backbone=bert-base-uncased · ckpt=methods/CSM/cpt_277000.pth · alpha=1.0 · delay=2 · min_gap=2 · device=cpu
데이터 = Def-DTS 번들 test. metric = segeval Pk/WD + boundary-set F1. Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).
cold-start (CSM/BERT load) = 28.77s

| dataset | n_dial | n_turn | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/turn(ms) | p95 | pred bs | gold bs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 1564 | 0.4644 | 0.4920 | 0.3385 | 0.4302 | 297.90 | 519.47 | 275 | 315 |
| dialseg711 | 711 | 18639 | 0.4210 | 0.4580 | 0.3904 | 0.4755 | 267.52 | 373.97 | 3753 | 2743 |
| superseg | 1322 | 16006 | 0.4709 | 0.4769 | 0.3606 | 0.4433 | 251.63 | 303.68 | 2821 | 4017 |

## 한계
- ckpt 호환성 — 본 wrapper 는 lxing532 `CoherenceNet` 가중치 (`bert-base-uncased` backbone) 전제. SuperDialseg 의 CSM ckpt 와는 모듈 구조가 달라 호환 안 됨.
- strict prefix-causal 아님 — `delay=2` right context 의존. 원본 score/HP 보존, *emission* 만 delay-2 (decision-log 2026-05-21 GreedySeg-online-delay2 entry 와 동일 framing).
