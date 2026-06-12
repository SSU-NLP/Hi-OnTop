# Hi-OnTop ↔ v4.1.3 output-parity 검증

config: m=2 · ρ=0.7 · a=0.5 · δ*=0.5594 (canonical TIAGE-cfg)
v4.1.3 의 segmentation 결정 = HiOnTopSegmenterV411 (v4.1.3 는 출력 attribute 만 추가) → V411 로 대조.
metric = 공식 SuperDialseg (Pk/WD k=auto, F1 binary, Score=0.5F1+0.25(1−Pk)+0.25(1−WD)).

| dataset | n_dial | n_turn | diff turns | Score v411 | Score Hi-OnTop | F1 | Pk | WD | parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| tiage | 100 | 1564 | 0 | 0.4675 | 0.4675 | 0.4102 | 0.4421 | 0.5082 | ✅ |
| dialseg711 | 711 | 19350 | 0 | 0.5897 | 0.5897 | 0.5493 | 0.3248 | 0.4151 | ✅ |
| superseg | 1322 | 17328 | 0 | 0.4631 | 0.4631 | 0.4323 | 0.4711 | 0.5410 | ✅ |

## 판정

**모든 데이터셋에서 byte-identical** — Hi-OnTop 출력이 v4.1.3 와 완전히 동일.

diff turns = v411 과 Hi-OnTop 의 boundary 예측이 다른 turn 수. 0 이면 두 segmenter 의 turn-level 출력이 완전히 같다는 뜻.
