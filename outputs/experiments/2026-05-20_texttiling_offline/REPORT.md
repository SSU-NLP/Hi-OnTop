# TextTiling **offline** (원본 SuperDialseg 설정, whole-dialogue)

원본=benchmarks/superdialseg TexttilingSegmenter 동일 알고리즘(nltk w/k, 코드 복사 없이 호출). data=Def-DTS 번들, metric=autoseg Pk/WD+F1, Score=0.5F1+0.25(1-Pk)+0.25(1-WD).
online(prefix-causal)판과 동일 harness → offline↔online 직접 비교.
limit=full · w=10 k=6

| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/dial(ms) | miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 0.5500 | 0.5657 | 0.1741 | 0.3081 | 9.4 | 0 |
| dialseg711 | 711 | 0.4903 | 0.5145 | 0.2632 | 0.3804 | 29.0 | 0 |
| superseg | 1322 | 0.4883 | 0.5044 | 0.3875 | 0.4456 | 9.9 | 0 |

## 한계
- 원 SuperDialseg 논문 보고치(tiage .363/superseg .471/dialseg711 .382)는 *그쪽 데이터·공식 metric* — 본 표는 Hi-OnTop harness(Def-DTS 데이터+autoseg+Score)라 정확 일치 아님, 방향·정상동작 검증용.
- offline = 대화 전체(미래 포함). online 판은 methods/texttiling/online.py (prefix-causal, AUXILIARY).
- non-LLM CPU, calls/turn=0 tok/turn=0.
