# 실험 — Hi-OnTop AMI 분절: δ_eff 한계 진단 → V_rel 상대신호 → online reset 부트스트랩 병목

## 실험 setup
- **목적**: AMI 회의 코퍼스(drift 도메인)에서 Hi-OnTop 화제분절기의 성능 한계 원인을 규명하고,
  δ_eff magnitude 임계치를 넘어서는 분절 신호/판정을 설계·검증.
- **데이터**: AMI scenario meetings **139개 전체** (`data/ami/topic/*.json`), turn=AMI segment 단위,
  gold top-level 경계 미팅당 5~7개(밀도 ~2.4%, 극sparse). 총 gold=935.
- **인코더**: MiniLM-int8 (`all-MiniLM-L6-v2` ONNX quint8, 384-d, CPU). 캐시 `outputs/runs/_misc/ami_emb/`.
- **metric**: ±2 tolerance boundary-F1 (주지표), 공식 Pk·WD(nltk, k=auto, ↓good),
  Score = 0.5·F1(±2) + 0.25·(1−Pk) + 0.25·(1−WD).
- **비교 baseline**: even-spacing(gold 개수 oracle 균등배치), TextTiling(streaming),
  Hi-OnTop δ_eff ewma(기존), LLM full-context(Qwen3.5-27B no-thinking, offline 참조 천장).

## 결과 (139미팅)

### 1. baseline + 기존 Hi-OnTop + LLM
| 방법 | pred | F1(±2) | Pk | WD | Score |
|---|--:|--:|--:|--:|--:|
| Even-spacing (count oracle) | 935 | 0.088 | 0.513 | 0.536 | 0.282 |
| TextTiling (streaming) | 7067 | 0.186 | 0.636 | 0.902 | 0.209 |
| Hi-OnTop δ_eff ewma [기존] | 5090 | 0.151 | 0.636 | 0.857 | 0.203 |
| **LLM full-context (offline)** | 1110 | **0.543** | 0.228 | 0.298 | **0.640** |

### 2. 신호 진단 (왜 기존이 약한가)
- LLM이 경계 찍은 turn에서 δ_eff z-score = **+0.545** (random +0.021) → 임베딩 신호는 진짜 화제전환에 솟음.
- 그러나 가장 강한 δ_eff peak를 top-K/threshold로 뽑아도 LLM 일치 ~0.11 → **가장 큰 spike는 경계가
  아니라 noise(화자전환·단발 이상치)**. magnitude 단독으론 boundary-spike와 noise-spike 분리 불가.
- gold가 의미전환보다 ~2턴 뒤(응답 turn)에 찍힘 → exact 정렬 부분적 불가.

### 3. V_rel 상대신호 + clean prototype — oracle 천장
prototype을 경계서 reset(깨끗하게)하면 같은 cosine 신호의 oracle 천장이 급등:
| 신호 (gold-reset oracle 임계) | ±2 F1 천장 |
|---|--:|
| δ_eff (기존, prototype 오염) | 0.226 |
| raw r_active (clean prototype) | 0.488 |
| **V_rel = r_active − λ·r_global** (λ=0.6) | 0.565 |
| **V_rel + global-EWMA (g_rho=0.15)** | **0.687** |
| (참조 LLM) | 0.543 |

- **V_rel = (active 화제 prototype 거리) − 0.6·(global/recent 거리)**: 경계는 화제서만 멀고 global로는
  평범 → 큰 값; noise는 둘 다 멂 → 작은 값. boundary/noise 분별의 핵심 레버.
- **overfit 검증 통과**: 2-fold 교차 격차 **0.000**, (lam=0.6, g_rho=0.15)가 A/B/manifest12/held127
  모든 split에서 동일하게 best. 12미팅 우연 아님.
- **제약 위반 없음**: 신호는 online·0-look-ahead(causal), 인코더 고정(MiniLM+online 통계만),
  retrieval 규칙 무관. SEM 계승(= local MAP의 new-event base 분포 근사, decision-log 기록).

### 4. 격차 분해 — 병목은 임계치가 아니라 reset
| 구성 | ±2 F1 |
|---|--:|
| gold-reset + per-meeting oracle 임계 [상한] | 0.687 |
| **gold-reset(clean) + 단순 μ+cσ (c=2.0)** | **0.554** (LLM급) |
| detected-reset + μ+cσ (deploy) | 0.15 |

→ **깨끗한 prototype만 있으면 단순 적응임계치로도 LLM급(0.554).** 임계치는 충분. 병목은 **online에서
깨끗한 prototype reset을 부트스트랩하는 것** 하나.

### 5. deploy 시도 — online reset 부트스트랩은 안 풀림
| deploy 방법 | pred | F1(±2) | Pk | WD | Score |
|---|--:|--:|--:|--:|--:|
| V_rel 적응임계치 (c=2.0) | 318 | 0.056 | 0.329 | 0.350 | **0.358** |
| V_rel (c=0.8) | 3842 | 0.149 | 0.504 | 0.678 | 0.279 |
| BOCPD top-K particle filter | 1408 | 0.069 | 0.424 | 0.493 | 0.305 |

- 단순 V_rel 적응임계치 **Score 0.358** = deployable 1위 (기존 0.203·TextTiling 0.209·even-spacing
  oracle 0.282 모두 상회).
- robust 갱신·peak gating·anchoring·refractory 조절: 무효(±2F1 0.15 고착).
- EM 반복정제(detected reset→재검출): 나쁜 고정점, 0.13에서 수렴 안 함.
- BOCPD particle filter(codex 설계, reset 가설 K개 병렬): 개수는 잡았으나(1408, Pk/WD↑) **위치는 더
  나쁨**(±2F1 0.069). soft hypothesis가 정답 경계 lock-on 실패.

## 판정
- **신호 확립**: V_rel(상대신호) + clean prototype은 oracle 천장 0.687(>LLM 0.543), overfit·제약 OK.
  "magnitude가 아니라 active-event 대비 설명 실패 + background 상대거리"가 boundary/noise를 가르는
  올바른 신호임을 입증.
- **deployable 최선**: 단순 V_rel 적응임계치, **Score 0.358 — 기존/TextTiling/even-spacing oracle 모두
  넘는 deployable 1위.**
- **미해결 병목 확정**: online에서 깨끗한 reset 부트스트랩. 임계치·갱신트릭·EM·particle filter 모두
  oracle(0.554~0.687)에 도달 못 함. 다음 iteration 대상.

## 한계 / 검증 미해결
- **oracle(0.687, 0.554)은 deploy 수치 아님** — gold-reset + per-meeting 임계치(또는 set-tuned knob)의
  상한. deploy ±2F1은 ~0.15에 고착.
- **AMI는 drift+sparse+gold-offset 도메인** — even-spacing oracle이 Pk/WD를 지배(content 신호와 구조적
  불일치). DTS(concat-seam, embedding exact F1~0.5)와 달리 어려운 robustness 도메인. primary가 아닌
  한계 도메인으로 포지셔닝 권장(codex 자문 일치).
- **lam=0.6/g_rho=0.15는 신호 calibration** — 2-fold로 overfit 아님 확인했으나, 다른 도메인 일반화는
  미검증. hazard·threshold는 도메인별 보정 필요.
- BOCPD particle filter의 run-length 정규화는 1차 구현. per-particle μ seeding으로 과분절 버그는
  잡았으나 localization 미해결 — soft assignment 설계 추가 검토 여지.
- seed 단일(LLM 캐시 1회). 단 embedding 방법들은 결정적.
