# REPORT — SeCom prompt ablation (baseline vs verbatim), AMI 18-subset, gpt-4o:nitro

**작성일**: 2026-06-12 · **상태**: 완료(단일 모델 **예비**, 예산 아티팩트 정정 후 clean 재측정) · **관련 handoff**: `HANDOFF_02_realtime-llm-comparison_0612.md` §1A

> ⚠️ **정정 이력**: 초판은 verbatim 이 버퍼(120/60)에서 "붕괴"(err 131/274, Score 0.428/0.372)한다고 보고했으나, 이는
> **프롬프트 취약이 아니라 API 예산 소진 아티팩트**였다(verbatim run 이 baseline 뒤에 돌며 도중 예산 고갈 → 콜 실패 →
> 빈 응답 캐시). 예산 재충전 + 빈 캐시 항목(405개) 제거 + 재실행 → **err 0 으로 정상화**. 본 REPORT 는 그 clean 수치.
> 코드도 **실패 응답은 캐시하지 않도록**(`secom_llm_eval.py` Caller, `ok` 플래그) 수정 + 실패 시 `[call-fail]` stderr 로깅 추가.

## 1. 실험 setup

- **목적**: LLM 분절 baseline 프롬프트로 **우리 적응본(`baseline_segment_v1.md`)** vs **SeCom 원본 verbatim(`segment_exchange.md`)**
  중 어느 것을 main 으로 쓸지 결정. 비용 절감 위해 **단일 모델(gpt-4o:nitro)** 로 먼저 가린다(사용자 지정 2026-06-12).
- **데이터**: AMI latency 18-subset (`outputs/runs/_misc/ami_latency_subset.json`), 18미팅, gold=`bnd_top` 합산 **121 경계**.
  (ES2002c, ES2004b, ES2005a/c/d, ES2009a/d, ES2010b, ES2012d, ES2014c/d, ES2016b, IB4005, IS1000c, IS1005a, TS3005a/d, TS3007a.)
- **방법(프롬프트 2종)** — `scripts/secom_llm_eval.py --mode segment --prompt {baseline|verbatim}`:
  - `baseline` → `secom_prompts/baseline_segment_v1.md` (SeCom `segment_turn` 구조를 turn 기반·멀티파티 일반화, LABEL=Turn).
  - `verbatim` → `secom_prompts/segment_exchange.md` (SeCom 원본 `segment_with_exchange_number` 그대로, LABEL=Exchange).
- **버퍼**: full / 120s / 60s, **B-mode(누적 prefix, `--context B`)**. (프롬프트 결정엔 full/120/60 충분 판단, seg-30/10 생략.)
- **모델**: `openrouter/openai/gpt-4o:nitro` 단일 (Crts 경유, non-reasoning).
- **HP/디코딩**: temperature **0**(greedy), no-thinking(`reasoning.enabled=false`), workers 32, 출력예산 segment 1024 tok. 응답 캐시 사용.
- **지표**: ±2-tol F1(primary), Pk, WindowDiff(raw), **Score=0.5·F1+0.25·(1−Pk)+0.25·(1−WD)**(summary), pred/gold, err(=4회 retry 후 실패 콜 수).
- **비교**: 동일 데이터·모델·decoding·후처리, 차이는 프롬프트뿐 = **controlled prompt ablation**.

## 2. 결과 표 (gpt-4o:nitro, 18-subset, 전 조건 err 0)

| 버퍼 | 프롬프트 | ±2F1 | Pk | WD | **Score** | pred/gold | err |
|---|---|--:|--:|--:|--:|--:|--:|
| full | baseline | 0.250 | 0.313 | 0.369 | 0.455 | 115/121 | 0 |
| full | **verbatim** | 0.298 | 0.302 | 0.354 | **0.485** | 111/121 | 0 |
| 120 | **baseline** | 0.288 | 0.303 | 0.361 | **0.478** | 119/121 | 0 |
| 120 | verbatim | 0.265 | 0.302 | 0.345 | 0.471 | 99/121 | 0 |
| 60 | baseline | 0.247 | 0.330 | 0.376 | 0.447 | 116/121 | 0 |
| 60 | **verbatim** | 0.250 | 0.319 | 0.360 | **0.455** | 96/121 | 0 |

**평균 Score (full/120/60)**: baseline **0.460** vs verbatim **0.470** (Δ = **+0.010 verbatim**).

## 3. 해석

- **두 프롬프트는 사실상 동급.** 평균 Score 차 0.010, 버퍼별 차도 0.007~0.030 → **단일 모델·단일 run 에서 noise 범위로 추정.**
  - full(offline): verbatim 우위(0.485 vs 0.455) — SeCom 원본의 native 설정(전체 대화 1콜 offline)이라 자연스럽다.
  - 120: baseline 근소 우위(0.478 vs 0.471). 60: verbatim 근소 우위(0.455 vs 0.447).
- **초판의 "verbatim 버퍼 붕괴"는 철회** — 예산 아티팩트였고, clean 재측정에선 verbatim 도 버퍼에서 정상 동작(err 0).
- **남는 실질 차이 = boundary-count calibration**: baseline 은 pred 가 버퍼 무관 안정(115/119/116 ≈ gold 121). verbatim 은
  버퍼 작아질수록 **under-predict**(111 → 99 → 96). 즉 composite Score 는 비슷해도 **baseline 이 경계 개수를 gold 에 더 잘 맞춤**
  (verbatim 은 버퍼에서 경계를 덜 찍는 경향). streaming 설정에서 baseline 의 약한 이점.

## 3.1 진단 — 실제 예측 분석 (왜 안 좋은가, full 기준, `scripts/analyze_prompt_ablation.py`)

캐시된 LLM 응답에서 실제 예측 경계를 추출해 분석(추가 API 0). buffer=full(offline, 전체 1콜) 기준:

- **(A) 경계 위치가 틀림 (near-miss 아님)**: 예측 경계의 **~57%(baseline)/53%(verbatim)가 노이즈**(gold 에서 >5 turn).
  pred→gold 중앙거리 **8/7 turn**. tolerance 를 ±5 로 풀어도 micro-F1 **0.428/0.461** → 위치 자체가 틀린 것이지
  ±2 밖 살짝 빗나간 게 아님. (tolerance curve baseline ±0/1/2/3/5 = 0.085/0.233/0.280/0.331/0.428, verbatim = 0.112/0.263/0.319/0.384/0.461.)
- **(B) 긴 미팅 후반부 포기 (지배적 recall 실패)**: LLM 이 경계를 앞쪽에 몰아 찍고 뒤를 비움. **후반부(>50% 지점) gold 경계
  46개 중 예측 13개(72% 놓침)**, 18미팅 중 **8개는 후반부 예측 0개**. 예: TS3005d(1194턴) gold 가 turn 1149 까지인데
  baseline/verbatim 둘 다 turn 560 이후 아무 경계도 안 찍음. → full-context 단일콜이 긴 회의의 뒷부분을 사실상 한 덩어리로 처리.
- **(C) 두 프롬프트가 실패모드를 공유 → baseline≈verbatim**: 정확히 같은 위치 경계 **66개**, baseline 115개 중 **78개가
  verbatim 과 ±2 내 일치**. verbatim 이 전 tolerance 에서 근소 우위(노이즈 53 vs 57%, 약간 정확한 위치)지만 구조적 차이 아님
  → §2 의 Score 차 0.01 의 정체. **프롬프트가 병목이 아니라 LLM full-context 분절 능력 자체가 한계.**
- 함의: LLM-offline(upper-bound)조차 AMI 에서 mediocre(±2F1 ~0.25–0.30, 후반부 붕괴) → Hi-OnTop 비교 서사에 유리.

## 4. 판정

- **composite Score 기준 baseline vs verbatim = near-tie**(Δ0.010, verbatim 이 mean 근소 위, 단일모델·단일run = noise 추정).
  → **"baseline 이 AMI 에서 verbatim 보다 낫다"는 강한 경험 주장은 이 데이터로 성립하지 않음**(오히려 mean 은 verbatim 근소 우위).
- 단 baseline 은 **버퍼에서 boundary-count 안정**(verbatim under-seg) → streaming 적합성에서 약한 plus.
- **결론 = 미결(예비)**. 프롬프트 선택을 AMI-nitro 성능만으로 정할 수 없음. 결정하려면 추가 근거 필요:
  ① 다른 모델(qwen/haiku/mistral/gemma×2/gpt-4o-mini)로 일반화, ② DTS 무저하 확인, ③ variance(N회 재실행 std),
  ④ principled 근거(멀티파티 turn 필연 적응 — codex 자문, handoff §1A 정당화 프레이밍).
  → **baseline·verbatim 둘 다 살려두고 추가 모델·DTS 로 재판단.**

## 5. 한계 / 검증 미해결

- **단일 모델(gpt-4o:nitro)만** (사용자 지정). 프롬프트 우열이 모델 불변인지 미확인 → 결론은 nitro 한정 예비.
- **18-subset only**(139 전수 아님), **단일 run**(temp 0 캐시, API 비결정성 std 미측정), **버퍼 full/120/60 만**(30/10 생략).
- **incremental 미비교**: `baseline_incremental_v1` vs `segment_incremental` 은 계보가 달라 controlled A/B 부적합 → 단일 operationalization 처리(§1A).
- **예산 아티팩트 발견·정정 완료**: 초판 "붕괴"는 예산 소진. clean 재측정으로 대체. 코드에 실패-미캐시 + `[call-fail]` 로깅 추가해
  재발 시 즉시 가시화(다른 잠재 아티팩트=rate-limit/context-length 도 이제 로그로 확인 가능).

## 산출물

- 결과 원천: `outputs/runs/_misc/secom_sweep_results.tsv` (태그 `ami_latency_subset|seg|*|{baseline,verbatim}|...gpt-4o:nitro`).
- 응답 캐시: `outputs/runs/_misc/secom_cache_openrouter_openai_gpt-4o:nitro.jsonl` (빈 응답 제거 후 3751개).
- 실행: `SEG_ONLY=1 BUFS="full 120 60" ONLY_MODEL="openrouter/openai/gpt-4o:nitro" bash scripts/secom_sweep.sh outputs/runs/_misc/ami_latency_subset.json`.
- 로그: `outputs/runs/_misc/secom_nitro_only.log`, `secom_nitro_rerun.log`, `secom_sweep_logs/`.
- 진단 스크립트: `scripts/analyze_prompt_ablation.py` (캐시 재사용, precision/recall·변위·tolerance curve·후반부 coverage·프롬프트 일치도).
