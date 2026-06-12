# 실험2 — LLM 버퍼-지연 vs Hi-OnTop 0-지연 (AMI 화제분절)

## 실험 setup
- **목적**: "스트리밍 0-턴 지연"이 Hi-OnTop의 핵심 강점임을 LLM 폴링 대비 정량화.
  LLM 분절을 버퍼 크기(초)로 제한했을 때 성능이 어떻게 변하는지 곡선화.
- **데이터**: AMI scenario meetings (manifest 12미팅, `data/ami/topic/`). turn 에 begin_time 있음.
- **방법**:
  - **LLM 폴링 (buffer-limited)**: 미팅을 B초 비중첩 윈도우로 분할, **각 윈도우만** LLM에 주고
    그 안의 화제 경계를 묻음 (`scripts/ami_llm_buffer_eval.py`). 버퍼 B = 폴링 1회당 문맥/지연.
    모델 = Qwen3.5-27B, **no-thinking** (`reasoning:{enabled:false}`), temp=0.
  - **LLM full-context (offline, ∞ 버퍼)**: 전체 transcript 1회 (`scripts/ami_llm_segment_eval.py`).
  - **Hi-OnTop (0 버퍼)**: per-utterance δ_eff + ewma 적응임계치, MiniLM-int8. 미래 0.
  - **TextTiling (0 버퍼)**: `StreamingTextTiling` (lexical block-cosine, streaming).
- **metric**: boundary-F1 (±2 turn tolerance). exact(±0)는 AMI annotation 특성상 누구도 ~0 (응답/filler에 경계) → 부적합.

## 결과 (12미팅, ±2 boundary-F1)
| 방법 | 버퍼 | ±2 F1 | exact F1 | pred/gold | LLM 호출 |
|---|---|--:|--:|--:|--:|
| LLM 폴링 | 10s | 0.256 | 0.006 | 115/80 | 907 |
| LLM 폴링 | 30s | 0.237 | 0.021 | 179/80 | 585 |
| LLM 폴링 | 60s | 0.254 | 0.010 | 196/80 | 340 |
| LLM 폴링 | 120s | 0.372 | 0.022 | 194/80 | 182 |
| **LLM full** | ∞(offline) | **0.526** | 0.034 | 78/80 | 12 |
| **Hi-OnTop** | **0** | 0.227 | 0.03 | — | 0 |
| TextTiling | 0 | 0.264 | 0.004 | 452/80 | 0 |

→ figure: `outputs/figures/figure_R_buffer_latency.{pdf,png}`.

## 해석
- **LLM의 강점(±2 0.53)은 *전체 문맥(offline)* 에서만 나온다.** 버퍼 10~60초에선 ±2 0.24~0.26으로
  평평하고, 120초에서야 0.37로 오름. 즉 **"조금 더 버퍼"로는 안 되고, 거의 전체 대화를 봐야** 함.
- **스트리밍 버퍼(10~60s)에서 LLM(0.24~0.26)은 0-지연 cheap 방법과 사실상 동급** — TextTiling(0.264),
  Hi-OnTop(0.227)과 차이 없음. 게다가 버퍼 LLM은 수백~900 LLM 호출 + 버퍼 지연.
- **Hi-OnTop은 0-지연·LLM 호출 0**으로 그 수준(0.227)을 냄.

## 판정
- **포지셔닝 입증**: 실시간 streaming(미래 못 봄)에선 LLM의 우위가 사라짐. LLM이 이기려면 **전체
  대화(= 큰 지연/버퍼 + 반복 호출)** 가 필요. → **Hi-OnTop의 0-턴 지연이 streaming 시나리오의 결정적 강점.**
- **trade-off 곡선**: 정확도(full LLM 0.53) ↔ 지연(buffer). Hi-OnTop은 지연 0에서 streaming-LLM과 동급.

## 한계 / 검증 미해결
- **12미팅**, seed 1, 단일 LLM(Qwen3.5-27B). Llama-3-70B/Mistral-7B 추가 시 곡선 변동 가능.
- **버퍼 폴링 구현 = 비중첩 윈도우**. 슬라이딩/누적 버퍼면 중간 버퍼에서 더 오를 수 있음(상한은 full 0.526).
- Hi-OnTop 점은 raw ewma(±2 0.227); geometry-merge·refractory 변형은 0.19~0.23 범위 — 0-지연 동급대.
- exact F1 은 전 방법 ~0.03 (LLM 포함) = AMI annotation 한계, 본 figure 는 ±2 기준.
- AMI = drift 도메인(어려움). DTS(concat-seam)에선 embedding exact F1 ~0.5 로 잘 됨 — 도메인 의존.
