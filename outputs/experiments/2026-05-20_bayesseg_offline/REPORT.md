# BayesSeg **offline** (원본 SuperDialseg BayesSegmenter)

원본 modeling_bayesseg.BayesSegmenter 동작(`segment config/dp.config`, `-num-segs 7`, 대화 전체) 호출(코드 복사 X). data=Def-DTS 번들, metric=autoseg Pk/WD+F1, Score=0.5F1+0.25(1-Pk)+0.25(1-WD). online 판과 동일 harness.
limit=full

| dataset | n_dial | Pk ↓ | WD ↓ | F1 ↑ | Score ↑ | lat/dial(ms) | miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiage | 100 | 0.4881 | 0.6197 | 0.3519 | 0.3990 | 459 | 0 |
| dialseg711 | 711 | 0.3082 | 0.3828 | 0.5093 | 0.5819 | 465 | 0 |
| superseg | 1322 | 0.4452 | 0.6624 | 0.4506 | 0.4484 | 435 | 0 |

## 한계
- 원 SuperDialseg 보고치(tiage .419/superseg .463/dialseg711 .614)는 그쪽 데이터·공식 metric — 본 표는 Hi-OnTop harness, 정확 일치 아님(방향·정상동작 검증용).
- `-num-segs 7` 은 SuperDialseg 가 박은 원본 설정 그대로(항상 7분할). 대화 8문장 미만이면 segment 크래시→miss 처리.
- offline = 전체대화(미래 포함, JVM 대화당 1회). online 판 = methods/bayesseg/online.py (persistent JVM, native-K, prefix, AUXILIARY).
- non-LLM CPU(+JVM), calls/turn=0 tok/turn=0.
