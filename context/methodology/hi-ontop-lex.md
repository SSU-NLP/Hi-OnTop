# Hi-OnTop-Lex — Hi-OnTop + lexical-overlap correction

`src/hi_ontop/hi_ontop_lex.py` · `class HiOnTopLex(HiOnTop)` · 2026-05-23 신설

## 한 줄

[[hi-ontop]] (`HiOnTop`) 의 의미거리 boundary 메커니즘은 **그대로 두고**,
TextTiling 식 단어-빈도 겹침(lexical overlap) 신호를 *보조 관측 feature*
로 더해 δ_eff 를 보정한 변형. v1 의 알려진 실패 모드 — wording 은
비슷한데 topic 이 바뀌는 경계 — 를 어휘 신호로 보완하는 게 목적.

## 왜 만들었나

[[hi-ontop]] 한계 §: "wording 은 비슷한데 topic 이 바뀌는 경우(δ≈낮은데
GT 경계)는 못 잡음" — 인접 임베딩 cosine 신호 자체의 구조적 한계.
역으로 어휘적으로 응집된 구간이 임베딩 noise 로 과분절될 수도 있음.
어휘 겹침 신호는 두 방향 모두를 보정한다 (사용자 확정 방향:
**겹침↓ → δ_eff↑ / 겹침↑ → δ_eff↓**).

설계는 codex:rescue 위임 (2026-05-23) — 결합 형태·인과적 lexical 신호
정의·SEM 계승 정당화 모두 codex 권고를 따름.

## 수식

L2-정규화 발화 임베딩 stream `s_1, s_2, …` 와 원문 텍스트 `u_1, u_2, …`
에 대해, 매 turn:

```
δ_base(t)  = a·δ_prev(t) + (1−a)·δ_ctx(t)        # = HiOnTop δ_eff (불변)
L_{t-1}    = Σ_{i=1..min(m_lex,t-1)} ρ_lex^{i-1} · subtf(u_{t-i})
overlap(t) = cos_tf( L_{t-1}, subtf(u_t) )        # 단어-빈도 겹침
lexdist(t) = 1 − overlap(t)
r_t        = min(1, min(n_ctx, n_t) / min_tokens) # 짧은-turn 신뢰 게이트
δ_eff_v2(t)= clip_[0,2]( δ_base(t) + w_lex · r_t · (lexdist(t) − μ_lex) )
g_t        = δ_eff_v2(t) / δ*
boundary(t) ⇔ δ_eff_v2(t) ≥ δ*
```

- `subtf(·)` = stopword 제거 후 content token 의 sublinear TF 벡터,
  `subtf(w) = 1 + log(count(w))`. dialogue 의 단어 반복 과대평가 방지.
- `cos_tf` = 두 sparse TF dict 의 cosine. 한쪽이라도 비면 0.
- `lexdist` 는 train median `μ_lex` 로 **centering** → 보정은 residual:
  평소보다 어휘가 더 갈라지면 δ_eff↑, 더 응집되면 δ_eff↓.
- `r_t` 는 token 수가 적은 짧은 발화에서 lexical 항을 약화 (TF 벡터
  불안정 구간 down-weight). `n_t` = 현재 turn content token 수,
  `n_ctx` = lexical window 내 총 content token 수.
- `clip_[0,2]` — cosine-distance 자연 범위. **w_lex=0 이면 보정항이 0
  이고 clip 도 no-op → δ_eff_v2 가 [[hi-ontop]] δ_eff 와 정확히 일치**
  (parity 검증됨, smoke test).
- lexical causal window `(m_lex, ρ_lex)` 는 [[hi-ontop]] 의 임베딩 causal
  window `(m, ρ)` 와 같은 구조 — 기하감쇠·past-only·O(m)/turn·look-ahead
  없음. streaming TextTiling 의 block-cosine 과 달리 right-block 닫힘
  lag 가 없어 turn `t` 에서 즉시 `overlap(t)` 사용 가능.

### Edge cases

- turn 0 (`_prev_s` 없음): δ_base=0, lexical context 없음 → r_t=0 →
  δ_eff_v2=0, boundary False, topic 0. ([[hi-ontop]] turn-0 동일.)
- `text=""` 전달 시 그 turn lexical 보정 비활성 (δ_eff_v2=δ_base).
- lexical window/현재 turn 의 content token 이 비면 r_t=0 → 보정 inert.

## State 필드 (HiOnTop 대비 추가분)

- `_lex_recent` : lexical causal window — 직전 ≤m_lex turn 의 raw token
  Counter 리스트.
- `last_delta_base` / `last_lex_overlap` / `last_lexdist` / `last_r` :
  직전 turn lexical read-out. `_history` 각 record 에 `delta_base`,
  `lex_overlap`, `lexdist`, `r` 추가.
- 나머지 (`_prev_s`, `_recent`, `_topic_id`, `last_*`) 는 [[hi-ontop]] 상속.

## HP

HiOnTop 임베딩 HP 4개 (`delta_star`, `ctx_window`, `ctx_decay`,
`ctx_blend_a`) 는 의미·default 불변. 추가 lexical HP:

| HP | 의미 | default |
|---|---|---|
| `w_lex` | lexical residual 가중치 (0 ⇒ v1 환원) | 0.15 |
| `m_lex` | lexical causal window turn 수 | 2 |
| `rho_lex` | lexical window 기하감쇠 | 0.7 |
| `mu_lex` | lexdist centering 값 (= train median) | 0.0 (calibration 필수) |
| `min_tokens` | 짧은-turn 게이트 분모 | 3 |
| `stop_words` | stopword set | 내장 영문 set |

`tf_mode` 는 sublinear 고정 (HP 아님). codex 권고 default:
`w_lex=0.15, m_lex=2, ρ_lex=0.7, min_tokens=3`.

### calibration

corpus·encoder 의존, label-free 2-pass (test leakage 없음):
1. calib split 에서 `μ_lex = median(lexdist)` — w_lex 무관, 1회.
2. 그 μ_lex 로 `δ* = p80(δ_eff_v2)` — w_lex 별 재산출.

`scripts/run_hiontop_v2.py` 가 세 벤치 calib split (tiage/train,
superseg/validation, dialseg711 30% tune split) 에서 자동 수행.

## SEM 계승 3-step (codex 2026-05-23)

1. **SEM 에 있나?** — **없음**. SEM 은 structured scene dynamics 의
   예측가능성으로 event boundary 를 설명하는 인지 모델이지, 단어 표면형
   lexical cohesion 을 직접 추적하는 텍스트 분절 모델이 아님. lexical
   overlap 부재는 "금지된 신호"가 아니라 SEM 의 추상화 수준(word-level
   surface cohesion 아님)에서 자연히 빠진 것.
2. **충돌?** — 제한적. lexical 항은 sCRP prior 도 local-MAP inference 도
   아니므로 *SEM 원형 충실 재현* 목표와는 이질적. 그러나 Hi-OnTop-Lex 의
   목표는 SEM 원형 재현이 아니라 SEM-inspired online dialogue
   segmentation — 이 관점에선 충돌 없음. lexical overlap 은 현재 관측
   `s_t` 주변 local boundary evidence 를 보강하는 domain-specific
   *observation correction*.
   우선순위: SEM 계승성(online·causal·local dynamics 변화) > 도메인
   적합성 > SEM 원형 충실 재현. → lexical 항은 SEM core mechanism 이
   아니라 텍스트 대화 도메인용 보조 관측 feature 로 기록.
3. **decision-log**: `context/06-decision-log.md` 2026-05-23 entry.

핵심 제약은 SEM 계승 §.

## 결과 (3-벤치, official SuperDialseg Score)

`outputs/experiments/2026-05-23_hiontop_v2/REPORT.md` (2026-05-23).
calib=train(tiage)/validation(superseg)/30%-tune(dialseg711), test split,
w_lex sweep, multi-qa-mpnet.

| w_lex | tiage | superseg | dialseg711 | mean-3 |
|---:|---:|---:|---:|---:|
| 0.00 (=v1) | 0.4331 | 0.4283 | 0.6167 | 0.4927 |
| 0.15 | 0.4376 | 0.4314 | 0.6267 | 0.4986 |
| 0.30 | 0.4324 | 0.4329 | **0.6314** | **0.4989** |

- **dialseg711**: w_lex 에 대해 Score·F1 단조 증가 (+0.005~+0.015) →
  lexical 보정 효과 실재 (robust 양).
- **superseg**: w_lex≥0.10 에서 +0.003~+0.005 (표본 크나 약한 양).
- **tiage**: 부호 불안정·단조성 없음 → noise (표본 100, 향상 주장 불가).
- 정직한 단일 default = `w_lex≈0.15~0.20` (벤치별 test 튜닝 없이).
  mean-3 +0.6 Score point — 약함.

## 판정 / 현재 상태

**v1 대체 승격 보류** (REPORT §4). mean-3 향상이 약하고 tiage 가
noise — default 모델 교체 근거 부족. 단 dialseg711 단조 추세가 lexical
신호의 원리적 유효성을 보이므로 폐기도 아님. 승격 재검토 전제 3개:
(a) multi-seed 로 superseg·tiage Δ 유의성, (b) lexical HP 2차 grid,
(c) v1 실패 모드 turn case-level 검증.

→ 현재 main 모델은 [[hi-ontop]] (`HiOnTop`) 유지. `HiOnTopLex` 는 검증
대기 변형.

## 한계 / 검증 미해결

- seed 1개 — superseg/tiage Δ 의 noise 대비 유의성 미확인.
- superseg calib = validation (cached train 임베딩 부재·92k-turn 로컬
  인코딩 비현실적). label-free p80/median 이라 leakage 아니나 엄밀히
  "학습 데이터" 아님.
- dialseg711 train split 부재 → 30/70 seeded split.
- lexical HP(m_lex/ρ_lex/min_tokens) 미sweep — codex default 고정.
- μ_lex = median; mean 대비 robust 하나 p80-δ* 와의 상호작용 미분석.
- stopword = 내장 영문 set, 도메인 특화 미조정.
- v1 의 "wording 유사·topic 전환" 실패 turn 을 v2 가 실제로 잡는지
  case-level 미검증 (집계 지표만 봄).

## 변경 이력

- **2026-05-23 신설** — [[hi-ontop]] 에 lexical-overlap 보정 추가한
  `HiOnTopLex` 도입. 설계 codex:rescue 위임. w_lex=0 parity 검증.
  3-벤치 평가 → v1 대체 승격 보류 (검증 대기 변형).
