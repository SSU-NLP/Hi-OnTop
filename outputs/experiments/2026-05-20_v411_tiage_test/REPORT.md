# v4.1.1 ctx_window sweep × 3 SuperDialseg-family benches

fixed: ρ=0.7 a=0.5 δ*=0.5594 α=1 λ=10 (TIAGE-cfg) · sweep ctx_window only.
metric = official SuperDialseg (Pk/WD k=auto, F1 binary), Score = 0.5·F1 + 0.25·(1−Pk) + 0.25·(1−WD).
split=test · datasets=['tiage'] · windows=[2]

## Score 매트릭스 (행=ctx_window, 열=dataset, **bold**=ds best)

| m | tiage | mean |
|---:|:---:|---:|
| 2 | **0.4675** | 0.4675 |

## 세부 (F1/Pk/WD)

| dataset | m | F1 | Pk | WD | Score |
|---|---:|---:|---:|---:|---:|
| tiage | 2 | 0.4102 | 0.4421 | 0.5082 | 0.4675 |

## 한계
- δ* 는 TIAGE-train calibration(0.5594) 고정 — ctx_window 만 sweep. 다른 m 에서 δ* 재calibration 시 결과 달라질 수 있음.
- 인코더 = `multi-qa-mpnet`(Hi-OnTop 기본). split=test 직접 사용 (δ* 도 TIAGE-train 전이값 → leakage 없음).
- metric/Score 정의는 공식 SuperDialseg (literature-comparable).
