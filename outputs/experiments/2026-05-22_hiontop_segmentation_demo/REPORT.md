# Hi-OnTop 실제 분절 시연 — 3개 벤치마크 진짜 대화 + 신뢰도

Hi-OnTop default config (`δ*=0.5594, m=2, ρ=0.7, a=0.5`) 를 각 벤치
test set 의 실제 대화에 turn 단위로 돌린 결과. 발화는 데이터셋 원문
(긴 것은 `…` 절단).

## 신뢰도(비율)란

Hi-OnTop 는 매 turn `graded = δ_eff / δ*` 를 낸다. 이걸 4 밴드로 나눈다:
`매우약`(<0.7) · `약`(0.7~1.0) · `정상`(1.0~1.3) · `강`(≥1.3).
**신뢰도(비율) = 그 밴드에 속한 turn 중 실제 GT 경계인 비율**(= 밴드
precision). 각 벤치 test set *전체*에서 계산한다 — 즉 "graded 가 이
밴드면, 진짜 경계일 확률이 경험적으로 몇 %"인가.

→ 예측 경계(▶, graded≥1.0)는 `정상` 또는 `강` 밴드. 같은 ▶ 라도
밴드 신뢰도가 다르므로, downstream 은 신뢰도로 약/강 경계를 구분
처리할 수 있다.

**표 읽는 법** — 행 = 한 turn.

- `topic` : Hi-OnTop segment id (바뀌면 새 주제로 분절).
- `δ_eff` : 그 turn 의 놀람 `a·δ_prev + (1−a)·δ_ctx`.
- `graded` : `δ_eff / δ*`. **≥ 1.0 이면 경계 판정.**
- `신뢰도` : 그 turn 의 graded 밴드 precision (벤치 전체 기준 비율).
- `예측` : Hi-OnTop 가 새 주제 시작이라 판정한 turn → `▶`.
- `정답` : 데이터셋 GT 경계 → `●`.  `평가` : ✓hit / ✗FP / ✗FN.

turn 0 은 첫 segment 시작이라 경계로 세지 않음(δ_eff 미정의).

## TIAGE — 인간 주석 잡담(PersonaChat 계열)

### 신뢰도 보정표 (tiage test 전체 100 dialog)

| 밴드 | graded 범위 | turn 수 | GT 경계 수 | **신뢰도(경계 비율)** |
|:--:|:--:|--:|--:|--:|
| 매우약 | <0.7 | 296 | 26 | **8.8%** |
| 약 | 0.7~1.0 | 737 | 136 | **18.5%** |
| 정상 | 1.0~1.3 | 406 | 140 | **34.5%** |
| 강 | ≥1.3 | 25 | 13 | **52.0%** |

### 예시 1 — dialog #0 (16 turns)

GT 경계 4개 · Hi-OnTop 예측 8개 · hit 4 / FP 4 / FN 0

| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |
|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|
| 0 | 0 | user: hello , how are you doing tonight ? | — | — | — |  |  |  |
| 1 | 0 | agent: i am well an loving this interaction how are you ? | 0.228 | 0.41 | 9% |  |  |  |
| 2 | 0 | user: i am great . i just got back from the club . | 0.399 | 0.71 | 18% |  |  |  |
| 3 | 1 | agent: this is my favorite time of the year season wise | 0.616 | 1.10 | 34% | ▶ | ● | ✓ |
| 4 | 1 | user: i would rather eat chocolate cake during this season . | 0.359 | 0.64 | 9% |  |  |  |
| 5 | 2 | agent: what club did you go to ? me an timothy watched tv | 0.742 | 1.33 | 52% | ▶ |  | ✗FP |
| 6 | 2 | user: i went to club chino . what show are you watching ? | 0.377 | 0.67 | 9% |  |  |  |
| 7 | 2 | agent: lol oh okay kind of random | 0.530 | 0.95 | 18% |  |  |  |
| 8 | 3 | user: do you live in a house or apartment ? | 0.588 | 1.05 | 34% | ▶ | ● | ✓ |
| 9 | 4 | agent: we watched a show about animals like him | 0.668 | 1.19 | 34% | ▶ |  | ✗FP |
| 10 | 5 | user: i love those shows . i am really craving cake . | 0.678 | 1.21 | 34% | ▶ |  | ✗FP |
| 11 | 6 | agent: why does that matter any ? i went outdoors to play frisbee | 0.824 | 1.47 | 52% | ▶ | ● | ✓ |
| 12 | 6 | user: it matters because i have a sweet tooth . | 0.500 | 0.89 | 18% |  |  |  |
| 13 | 7 | agent: so ? lol i want to meet my family at home in bora | 0.647 | 1.16 | 34% | ▶ | ● | ✓ |
| 14 | 7 | user: my family lives in alaska . it is freezing down there . | 0.538 | 0.96 | 18% |  |  |  |
| 15 | 8 | user: i bet it is oh i could not | 0.624 | 1.11 | 34% | ▶ |  | ✗FP |

### 예시 2 — dialog #2 (16 turns)

GT 경계 3개 · Hi-OnTop 예측 3개 · hit 1 / FP 2 / FN 2

| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |
|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|
| 0 | 0 | user: hey , are you a student , i traveled a lot , i even studied… | — | — | — |  |  |  |
| 1 | 1 | agent: no , i work full time at a nursing home . i am a nurses ai… | 0.624 | 1.12 | 34% | ▶ |  | ✗FP |
| 2 | 1 | user: nice , i just got a advertising job myself . do you like yo… | 0.512 | 0.92 | 18% |  |  |  |
| 3 | 1 | agent: nice . yes i do . caring for people is the joy of my life . | 0.456 | 0.82 | 18% |  |  |  |
| 4 | 1 | user: nice my best friend is a nurse , i knew him since kindergar… | 0.481 | 0.86 | 18% |  |  |  |
| 5 | 1 | agent: very cool . do you have pets ? | 0.464 | 0.83 | 18% |  | ● | ✗FN |
| 6 | 1 | user: no i do not , do you ? | 0.535 | 0.96 | 18% |  |  |  |
| 7 | 1 | agent: yes . two dogs and a cat . they are my babies . | 0.467 | 0.83 | 18% |  |  |  |
| 8 | 2 | user: nice , where do you live ? i resign in north dakota | 0.661 | 1.18 | 34% | ▶ | ● | ✓ |
| 9 | 2 | agent: i live in texas . i love riding my bike here . | 0.462 | 0.83 | 18% |  |  |  |
| 10 | 2 | user: are you a christian ? i am jewish | 0.559 | 1.00 | 18% |  | ● | ✗FN |
| 11 | 2 | agent: yes i am . i have always been a christian . | 0.307 | 0.55 | 9% |  |  |  |
| 12 | 2 | user: i am jewish but i am not religious all that much | 0.405 | 0.72 | 18% |  |  |  |
| 13 | 2 | agent: oh really ? so you do not go to church often ? | 0.329 | 0.59 | 9% |  |  |  |
| 14 | 2 | user: i mostly go to the mosque on my free time | 0.349 | 0.62 | 9% |  |  |  |
| 15 | 3 | user: i see . that is very cool . i enjoy hearing about other pra… | 0.583 | 1.04 | 34% | ▶ |  | ✗FP |

## Dialseg711 — 합성(서로 다른 task 대화 concat), 경계 선명

### 신뢰도 보정표 (dialseg711 test 전체 711 dialog)

| 밴드 | graded 범위 | turn 수 | GT 경계 수 | **신뢰도(경계 비율)** |
|:--:|:--:|--:|--:|--:|
| 매우약 | <0.7 | 5436 | 13 | **0.2%** |
| 약 | 0.7~1.0 | 8001 | 556 | **6.9%** |
| 정상 | 1.0~1.3 | 4737 | 1813 | **38.3%** |
| 강 | ≥1.3 | 465 | 372 | **80.0%** |

### 예시 1 — dialog #0 (24 turns)

GT 경계 4개 · Hi-OnTop 예측 5개 · hit 3 / FP 2 / FN 1

| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |
|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|
| 0 | 0 | user: check the weather for the 7 day forecast | — | — | — |  |  |  |
| 1 | 1 | agent: What city are you interested in? | 0.566 | 1.01 | 38% | ▶ |  | ✗FP |
| 2 | 1 | user: Los Angeles, please. Will it be hot? | 0.379 | 0.68 | 0% |  |  |  |
| 3 | 1 | agent: It will be hot today in Los Angeles. | 0.269 | 0.48 | 0% |  |  |  |
| 4 | 2 | user: Yes, can you give me the information on the Huntingdon Marr… | 0.754 | 1.35 | 80% | ▶ | ● | ✓ |
| 5 | 2 | agent: Absolutely. It is an expensive hotel located in the west p… | 0.365 | 0.65 | 0% |  |  |  |
| 6 | 2 | user: Yes please, I need a reservation for 6 people for 5 nights … | 0.377 | 0.67 | 0% |  |  |  |
| 7 | 2 | agent: Sorry, there are not enough rooms available for that time … | 0.336 | 0.60 | 0% |  |  |  |
| 8 | 2 | user: How about for 1 night? If that works, I'll need a reference… | 0.303 | 0.54 | 0% |  |  |  |
| 9 | 2 | agent: Booking was successful. Your reference number is : OO8QDA6… | 0.411 | 0.73 | 7% |  |  |  |
| 10 | 2 | user: I need a train from London Kings Cross to Cambridge. | 0.533 | 0.95 | 7% |  | ● | ✗FN |
| 11 | 2 | agent: I have 70 trains travelling that route. To narrow it down,… | 0.460 | 0.82 | 7% |  |  |  |
| 12 | 2 | user: sure, I would like to go on Saturday, and arrive by 20:20. … | 0.444 | 0.79 | 7% |  |  |  |
| 13 | 2 | agent: I have a 19:17 from London that arrives at 20:08. Would th… | 0.359 | 0.64 | 0% |  |  |  |
| 14 | 2 | user: Yes, let me have 7 tickets, please. | 0.490 | 0.88 | 7% |  |  |  |
| 15 | 2 | agent: Booking was successful, the total fee is 132.16 GBP payabl… | 0.536 | 0.96 | 7% |  |  |  |
| 16 | 3 | user: I need to find a shopping center. | 0.749 | 1.34 | 80% | ▶ | ● | ✓ |
| 17 | 3 | agent: The Stanford Shopping Center at 773 Alger Dr is 3 miles aw… | 0.387 | 0.69 | 0% |  |  |  |
| 18 | 4 | user: Yes please. | 0.567 | 1.01 | 38% | ▶ |  | ✗FP |
| 19 | 4 | agent: I sent all the info on the screen, please drive carefully! | 0.447 | 0.80 | 7% |  |  |  |
| 20 | 5 | user: Schedule meeting. | 0.624 | 1.11 | 38% | ▶ | ● | ✓ |
| 21 | 5 | agent: What details shall I add to your meeting reminder? | 0.348 | 0.62 | 0% |  |  |  |
| 22 | 5 | user: Please set it for the 13th at 11am. It is going to be with … | 0.451 | 0.81 | 7% |  |  |  |
| 23 | 5 | agent: Reminder set for your meeting at 11am on the 13th with man… | 0.252 | 0.45 | 0% |  |  |  |

### 예시 2 — dialog #1 (26 turns)

GT 경계 4개 · Hi-OnTop 예측 9개 · hit 3 / FP 6 / FN 1

| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |
|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|
| 0 | 0 | user: Can you help me find a restaurant that serves greek food an… | — | — | — |  |  |  |
| 1 | 0 | agent: I apologize but there are no expensive greek restaurants l… | 0.131 | 0.23 | 0% |  |  |  |
| 2 | 0 | user: Do you have any Indian restaurants? | 0.456 | 0.82 | 7% |  |  |  |
| 3 | 0 | agent: We have 14 indian restaurants in the expensive category. D… | 0.238 | 0.43 | 0% |  |  |  |
| 4 | 1 | user: I do not. Is there one you can recommend? | 0.636 | 1.14 | 38% | ▶ |  | ✗FP |
| 5 | 1 | agent: I can recommend one for you, but first is there a particul… | 0.490 | 0.88 | 7% |  |  |  |
| 6 | 1 | user: Any part of town is fine, I just need the address and postc… | 0.323 | 0.58 | 0% |  |  |  |
| 7 | 1 | agent: I would recommend the Maharajah Tandoori Restaurant. It's … | 0.477 | 0.85 | 7% |  |  |  |
| 8 | 1 | user: Car I need to get at my friends house near my current locat… | 0.538 | 0.96 | 7% |  | ● | ✗FN |
| 9 | 1 | agent: Tom lives at 580 Van Ness Ave. | 0.485 | 0.87 | 7% |  |  |  |
| 10 | 2 | user: looking for a place to dine. The restaurant should serve br… | 0.660 | 1.18 | 38% | ▶ | ● | ✓ |
| 11 | 2 | agent: Fitzbillies restaurant at 51 Trumpington Street City Centr… | 0.317 | 0.57 | 0% |  |  |  |
| 12 | 3 | user: I would. It's a party of 6 for Monday at 15:45. Can I get t… | 0.560 | 1.00 | 38% | ▶ |  | ✗FP |
| 13 | 3 | agent: Your reservation has been made. The reference number is GK… | 0.450 | 0.80 | 7% |  |  |  |
| 14 | 4 | user: No, that's everything. Thanks for your help. | 0.625 | 1.12 | 38% | ▶ |  | ✗FP |
| 15 | 5 | agent: welcome next time | 0.578 | 1.03 | 38% | ▶ |  | ✗FP |
| 16 | 6 | user: Im looking for a shopping mall around here. | 0.708 | 1.26 | 38% | ▶ | ● | ✓ |
| 17 | 6 | agent: The Topanga mall is 4 miles away. | 0.356 | 0.64 | 0% |  |  |  |
| 18 | 6 | user: What is the address? | 0.546 | 0.98 | 7% |  |  |  |
| 19 | 6 | agent: The address to Topanga mall is 171 Oak Rd. | 0.424 | 0.76 | 7% |  |  |  |
| 20 | 7 | user: Set GPS for the quickest route to get there. | 0.621 | 1.11 | 38% | ▶ |  | ✗FP |
| 21 | 7 | agent: GPS set for quickest route to Topanga Mall. | 0.253 | 0.45 | 0% |  |  |  |
| 22 | 8 | user: Is there rain on Sunday | 0.754 | 1.35 | 80% | ▶ | ● | ✓ |
| 23 | 8 | agent: What city would you like the forecast for? | 0.513 | 0.92 | 7% |  |  |  |
| 24 | 9 | user: Compton. | 0.724 | 1.29 | 38% | ▶ |  | ✗FP |
| 25 | 9 | agent: It will not rain on Saturday in Compton. | 0.392 | 0.70 | 7% |  |  |  |

## SuperDialseg — 문서 기반 task 대화

### 신뢰도 보정표 (superseg test 전체 1322 dialog)

| 밴드 | graded 범위 | turn 수 | GT 경계 수 | **신뢰도(경계 비율)** |
|:--:|:--:|--:|--:|--:|
| 매우약 | <0.7 | 3883 | 385 | **9.9%** |
| 약 | 0.7~1.0 | 6655 | 1584 | **23.8%** |
| 정상 | 1.0~1.3 | 4915 | 1791 | **36.4%** |
| 강 | ≥1.3 | 553 | 260 | **47.0%** |

### 예시 1 — dialog #13 (15 turns)

GT 경계 3개 · Hi-OnTop 예측 3개 · hit 3 / FP 0 / FN 0

| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |
|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|
| 0 | 0 | user: Do I need documents for my foreign born child to receive th… | — | — | — |  |  |  |
| 1 | 0 | agent: Yes, you will need documentation pertaining to their proof… | 0.267 | 0.48 | 10% |  |  |  |
| 2 | 0 | agent: Do you have documents that pertain to their citizenship? | 0.290 | 0.52 | 10% |  |  |  |
| 3 | 0 | user: Yes, I have documents. | 0.365 | 0.65 | 10% |  |  |  |
| 4 | 0 | agent: If you don't, there are alternative forms that we can cons… | 0.393 | 0.70 | 24% |  |  |  |
| 5 | 1 | user: Are foreign born U.S. children eligible for social security… | 0.646 | 1.15 | 36% | ▶ | ● | ✓ |
| 6 | 1 | agent: Yes, as long as they have the correct important documentat… | 0.512 | 0.91 | 24% |  |  |  |
| 7 | 2 | user: What kind of proof of identity does social security accept? | 0.568 | 1.02 | 36% | ▶ | ● | ✓ |
| 8 | 2 | agent: They can accept a U.S. driver's license, a state issued no… | 0.430 | 0.77 | 24% |  |  |  |
| 9 | 2 | user: Can you accept any alternative documents for the child? | 0.475 | 0.85 | 24% |  |  |  |
| 10 | 2 | agent: We can accept their state issued non-driver id card or the… | 0.263 | 0.47 | 10% |  |  |  |
| 11 | 2 | user: What records do I need if my child lived outside the U.S. f… | 0.454 | 0.81 | 24% |  |  |  |
| 12 | 2 | agent: A current or previous passport and school/employment recor… | 0.354 | 0.63 | 10% |  |  |  |
| 13 | 3 | user: Can I turn in copies of the documentation? | 0.684 | 1.22 | 36% | ▶ | ● | ✓ |
| 14 | 3 | agent: No, it must be the current and original documents, no phot… | 0.419 | 0.75 | 24% |  |  |  |

### 예시 2 — dialog #28 (14 turns)

GT 경계 4개 · Hi-OnTop 예측 3개 · hit 1 / FP 2 / FN 3

| # | topic | 발화 | δ_eff | graded | 신뢰도 | 예측 | 정답 | 평가 |
|--:|:--:|:--|--:|--:|--:|:--:|:--:|:--:|
| 0 | 0 | user: Good Morning, i would like to know details about retirement. | — | — | — |  |  |  |
| 1 | 1 | agent: Do you mean for your family, that is? | 0.629 | 1.12 | 36% | ▶ |  | ✗FP |
| 2 | 1 | user: Correct, that's what i'm looking for. | 0.497 | 0.89 | 24% |  |  |  |
| 3 | 2 | agent: Very Well. Have you started to receive retirement benefits… | 0.594 | 1.06 | 36% | ▶ |  | ✗FP |
| 4 | 2 | user: Yeah, i'm receiving the benefits. | 0.402 | 0.72 | 24% |  |  |  |
| 5 | 2 | agent: Excellent, because in that case, some members of your fami… | 0.533 | 0.95 | 24% |  |  |  |
| 6 | 2 | user: That would be great. Now, what about if my spouse start rec… | 0.412 | 0.74 | 24% |  | ● | ✗FN |
| 7 | 2 | agent: Well in that case, the amount received will have a reducti… | 0.512 | 0.92 | 24% |  |  |  |
| 8 | 2 | user: And what happens if any of my family members apply for bene… | 0.431 | 0.77 | 24% |  | ● | ✗FN |
| 9 | 2 | agent: Not much. We'll just ask for your SS numbers and birth cer… | 0.429 | 0.77 | 24% |  |  |  |
| 10 | 3 | user: So my family can qualify for benefits and those payments wo… | 0.566 | 1.01 | 36% | ▶ | ● | ✓ |
| 11 | 3 | agent: That's correct, they won't affect your benefit amount. | 0.338 | 0.60 | 10% |  |  |  |
| 12 | 3 | user: Thanks, just wanted to be sure. now, is there any limitatio… | 0.523 | 0.94 | 24% |  | ● | ✗FN |
| 13 | 3 | agent: Well, yeah. there is a limit in the amount we pay to a fam… | 0.305 | 0.55 | 10% |  |  |  |

## 해석

- 신뢰도(밴드 precision)가 `강` 밴드로 갈수록 올라가면 graded 가
  calibrated 된 신호라는 뜻 — 보정표에서 `매우약`→`강` 순으로 비율이
  단조 증가하는지 확인.
- 같은 예측 경계(▶)라도 신뢰도가 다르다: `강` 밴드 ▶ 는 믿고 commit,
  `정상` 밴드 ▶ 는 약한 경계 — downstream 이 비율로 구분 처리 가능.
- TIAGE 잡담형은 신뢰도 자체가 낮고(밴드 precision 낮음) FP 가 많다.
  Dialseg711 은 `강` 밴드 신뢰도가 높아 hit 비율이 높다 — 벤치 Score
  차이가 신뢰도 보정표에서 그대로 드러난다.

## 한계

- 신뢰도(밴드 precision)는 해당 벤치 test set 전체에서 계산 — 같은
  test 로 보정·표시라 held-out 신뢰도 주장은 아님(시연용 calibration).
- 예시 대화는 '길이 14~26 + GT 경계 3~7개' 앞쪽을 기계 선별
  (cherry-pick 아님, 잘된 것·못된 것 그대로).
- 밴드 경계값(0.7/1.0/1.3)은 휴리스틱.
