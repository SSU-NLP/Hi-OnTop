# Hi-OnTop per-variant Seg latency (HiOnTop.assign timing, per-bench δ*)

## 정의
- 각 (variant × bench) 의 *실제 δ\** 로 `HiOnTop.assign()` 만 perf_counter.
- 3-seed × 1500-turn sample per bench → 3 bench → 1 variant 당 ~13500 sample.
- 보고값 = cross-bench mean (3 bench 평균).
- 200-turn warmup, first turn 제외. cached embeddings.

## 결과
| variant | tiage Seg | ds711 Seg | sds Seg | cross-bench mean |
|---|---:|---:|---:|---:|
| mpnet_p60 | 0.0572 | 0.0520 | 0.0641 | **0.0578** |
| mpnet_p70 | 0.0526 | 0.0469 | 0.0677 | **0.0559** |
| mpnet_p80 | 0.0535 | 0.0549 | 0.0477 | **0.0520** |
| mpnet_sup | 0.0417 | 0.0422 | 0.0432 | **0.0424** |
| mpnet_oracle | 0.0513 | 0.0434 | 0.0359 | **0.0431** |
| int8_p60 | 0.0564 | 0.0470 | 0.0444 | **0.0489** |
| int8_p70 | 0.0404 | 0.0962 | 0.0650 | **0.0687** |
| int8_p80 | 0.0622 | 0.0471 | 0.0436 | **0.0504** |
| int8_sup | 0.0413 | 0.0393 | 0.0357 | **0.0386** |
| int8_oracle | 0.0425 | 0.0509 | 0.0394 | **0.0444** |
