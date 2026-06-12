# 인코더 후보 per-utterance latency (Phase 1+2)

표본 = Def-DTS 번들 test 발화 600개. 단건(turn당 1발화) 인코딩 — 온라인 배포 조건. CPU. warm-up 15회 제외.

| 인코더 | mean (ms) | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| mpnet  PyTorch | **599.66** | 365.20 | 1388.02 | 3110.91 |
| mpnet  ONNX | FAILED | — | — | — |
| MiniLM PyTorch | **110.87** | 67.48 | 256.89 | 600.37 |
| MiniLM ONNX | FAILED | — | — | — |

## ONNX ↔ PyTorch 임베딩 동등성 (표본 평균 cos)

- mpnet : 측정 불가
- MiniLM: 측정 불가

cos ≈ 1.0 이면 ONNX 변형의 segmentation Score 는 PyTorch 결과를 그대로 상속 가능 (재평가 불필요).
