# Hi-OnTop oracle / best per-metric (superdialseg_data, NLTK Pk/WD)

δ\* 두 가지 + per-metric Pk/WD/F1/Score:
- **best** = labeled train split sweep → test 적용 (run_encoder_comparison.py 와 동일 정의).
  tiage/superseg = train split, dialseg711 = test 70:30 split.
- **oracle** = test 에서 δ\* sweep (supervised upper bound, label leakage 인정 — not deployable).

harness: `run_encoder_comparison.py` 함수 직접 import. δ\* grid = linspace(0.35, 0.95, 60).
HP: m=2, ρ=0.7, a=0.5. cached embeddings 만 사용 (인코더 forward 없음).

## encoder = mpnet

| 벤치 | type | δ\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |
|---|---|---:|---:|---:|---:|---:|---|
| tiage | best | 0.5737 | 0.4457 | 0.4987 | 0.3777 | **0.4528** | train split (calib 300 / test 100) |
| tiage | oracle | 0.5432 | 0.4386 | 0.5288 | 0.4294 | **0.4729** | test sweep |
| dialseg711 | best | 0.6144 | 0.2846 | 0.3205 | 0.5622 | **0.6299** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | oracle | 0.6144 | 0.2846 | 0.3205 | 0.5622 | **0.6299** | test sweep |
| superseg | best | 0.5534 | 0.4710 | 0.5462 | 0.4338 | **0.4626** | train split (calib 400 / test 1322) |
| superseg | oracle | 0.5432 | 0.4691 | 0.5557 | 0.4411 | **0.4643** | test sweep |

## encoder = minilm

| 벤치 | type | δ\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |
|---|---|---:|---:|---:|---:|---:|---|
| tiage | best | 0.8178 | 0.4210 | 0.4692 | 0.3941 | **0.4745** | train split (calib 300 / test 100) |
| tiage | oracle | 0.7669 | 0.4248 | 0.5241 | 0.4442 | **0.4849** | test sweep |
| dialseg711 | best | 0.8178 | 0.3053 | 0.3474 | 0.5436 | **0.6086** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | oracle | 0.8178 | 0.3053 | 0.3474 | 0.5436 | **0.6086** | test sweep |
| superseg | best | 0.6856 | 0.4906 | 0.6093 | 0.4193 | **0.4347** | train split (calib 400 / test 1322) |
| superseg | oracle | 0.6449 | 0.4808 | 0.6297 | 0.4317 | **0.4382** | test sweep |

## encoder = minilm-int8

| 벤치 | type | δ\* | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | calib note |
|---|---|---:|---:|---:|---:|---:|---|
| tiage | best | 0.7873 | 0.4186 | 0.4965 | 0.4264 | **0.4844** | train split (calib 300 / test 100) |
| tiage | oracle | 0.7771 | 0.4181 | 0.5065 | 0.4401 | **0.4889** | test sweep |
| dialseg711 | best | 0.8280 | 0.2940 | 0.3315 | 0.5427 | **0.6150** | test 70:30 split (calib 498 / test 213) |
| dialseg711 | oracle | 0.8178 | 0.2949 | 0.3419 | 0.5498 | **0.6157** | test sweep |
| superseg | best | 0.6042 | 0.4786 | 0.6660 | 0.4396 | **0.4336** | train split (calib 400 / test 1322) |
| superseg | oracle | 0.6246 | 0.4799 | 0.6484 | 0.4361 | **0.4360** | test sweep |

## 한계
- oracle: test 에 직접 fit → not deployable. supervised upper bound 로만 의미.
- best (dialseg711): train 부재로 test 70:30 분할 → 30% test 평가. ‡ supervised 라 베이스라인과 비대칭 — 표에서 참고용 행으로 표기.
- HP m/ρ/a fixed (Hi-OnTop default), encoder 만 교체.
