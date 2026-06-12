# v4 그래프 구조 실제 덤프 — AMI ES2002a (turn 0–45)

## 그래프 만드는 법 (레시피)
- **노드** = 발화 1개. **임베딩** = MiniLM-int8(384d).
- **엣지** = 두 발화의 cosine 유사도 `cos > τ(0.4)` 이면 연결.
- **엣지 가중** = `cos × exp(−|i−j|/T)`, T=40 → **멀리 떨어진 발화일수록 약하게**(시간감쇠).
- **community** = Louvain(resolution=0.5) 로 가중그래프를 토픽 덩어리로 분할 → 그 뒤 median smooth(2).
- **경계** = 이웃 발화의 community 가 바뀌는 곳.

전체 미팅: 노드 155, 엣지 1149, community 9개 (정답 top화제 6).

## community(토픽 덩어리) 구성 — 어느 발화들이 묶였나

- **T0** (10 발화): [0, 2, 3, 4, 5, 6, 24, 77, 128, 134]
- **T1** (76 발화): [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 31, 32, 33, 34, 35, 51, 53, 62, 63, 64] …
- **T2** (42 발화): [7, 8, 9, 54, 55, 56, 57, 58, 59, 60, 61, 74, 75, 76, 78, 79, 80, 81, 82, 83, 85, 96, 97, 98, 103] …
- **T3** (3 발화): [25, 26, 27]
- **T4** (1 발화): [28]
- **T5** (2 발화): [29, 30]
- **T6** (16 발화): [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52]
- **T7** (1 발화): [89]
- **T8** (4 발화): [115, 116, 117, 118]

→ **같은 T 안에 멀리 떨어진 발화가 섞여 있으면 = 그래프가 비연속(revisit) 링크를 만든 것.**

## 발화별 엣지 (turn 0–45) — 누가 누구랑 연결됐나

`연결`= 그 발화와 엣지로 이어진 turn (cos). **★far**=10턴 이상 떨어진 long-range 링크(가짜 revisit 의심).

| # | (화자) `T` | 발화 | 연결된 turn (cos) |
|--:|:--|------|------|
| 0 | (B) `T0` | Okay. Right. Um well this is the kick-off… | 97(0.55)★far, 151(0.50)★far, 136(0.44)★far, 132(0.42)★far, 11(0.42)★far |
| 1 | (D) `T1` | Mm-hmm. Great. | 9(0.83), 16(0.83)★far, 88(0.83)★far, 94(0.83)★far, 95(0.83)★far, 107(0.83)★far |
| 2 | (A) `T0` | Hi, I'm David and I'm supposed to be an i… | 127(0.52)★far, 132(0.51)★far, 4(0.42) |
| 3 | (B) `T0` | Okay. | 23(1.00)★far, 135(1.00)★far, 147(1.00)★far, 149(1.00)★far, 32(0.86)★far, 152(0.77)★far |
| 4 | (D) `T0` | And I'm Andrew and I'm uh our marketing | 5(0.45), 2(0.42), 77(0.40)★far |
| 5 | (C) `T0` | Um I'm Craig and I'm User Interface. | 4(0.45) |
| 6 | (D) `T0` | expert. | —(고립) |
| 7 | (B) `T2` | Great. Okay. Um so we're designing a new … | 115(0.65)★far, 8(0.64), 124(0.52)★far, 108(0.52)★far, 96(0.50)★far, 136(0.45)★far |
| 8★ | (A) `T2` | Um, I just got the project announcement a… | 7(0.64), 124(0.54)★far, 115(0.53)★far, 96(0.53)★far, 81(0.51)★far, 53(0.50)★far |
| 9 | (B) `T2` | Mm-hmm. | 16(1.00), 88(1.00)★far, 94(1.00)★far, 95(1.00)★far, 107(1.00)★far, 111(1.00)★far |
| 10 | (D) `T1` | Mm-hmm. Mm-hmm. Yeah, that's that's it. | 64(0.78)★far, 138(0.78)★far, 9(0.77), 16(0.77), 88(0.77)★far, 94(0.77)★far |
| 11 | (B) `T1` | Is that what everybody got? Okay. Um. So … | 38(0.54)★far, 19(0.48), 0(0.42)★far, 17(0.40) |
| 12 | (C) `T1` | Yeah. | 72(1.00)★far, 99(1.00)★far, 100(1.00)★far, 146(1.00)★far, 122(0.87)★far, 39(0.82)★far |
| 13 | (D) `T1` | Yeah. I will go. That's fine. | 142(0.45)★far, 55(0.42)★far |
| 14★ | (B) `T1` | Very good. | 32(0.42)★far |
| 15 | (D) `T1` | Alright. So This one here, right? | 142(0.54)★far, 144(0.52)★far, 71(0.50)★far, 152(0.49)★far, 32(0.45)★far, 122(0.45)★far |
| 16 | (B) `T1` | Mm-hmm. | 9(1.00), 88(1.00)★far, 94(1.00)★far, 95(1.00)★far, 107(1.00)★far, 111(1.00)★far |
| 17 | (D) `T1` | Okay. Very nice. Alright. My favourite an… | 38(0.56)★far, 19(0.52), 24(0.48), 11(0.40) |
| 18 | (B) `T1` | Yeah. Yeah. Right. Lovely. | 71(0.60)★far, 122(0.59)★far, 142(0.54)★far, 86(0.53)★far, 12(0.50), 72(0.50)★far |
| 19 | (C) `T1` | Well, my favourite animal would be a monk… | 38(0.61)★far, 17(0.52), 11(0.48) |
| 20 | (B) `T1` | Right. | 12(0.62), 72(0.62)★far, 99(0.62)★far, 100(0.62)★far, 146(0.62)★far, 3(0.57)★far |
| 21 | (A) `T1` | Cool. There's too much gear. | 154(0.47)★far |
| 22 | (B) `T1` | You can take as long over this as you lik… | 97(0.50)★far, 141(0.42)★far |
| 23 | (A) `T1` | Okay. | 3(1.00)★far, 135(1.00)★far, 147(1.00)★far, 149(1.00)★far, 32(0.86), 152(0.77)★far |
| 24 | (D) `T0` | I coulda told you a whole lot more about … | 17(0.48) |
| 25 | (B) `T3` | Ach why not We might have to get you up a… | —(고립) |
| 26 | (D) `T3` | Impressionist. | —(고립) |
| 27 | (A) `T3` | Can't draw. | —(고립) |
| 28 | (B) `T4` | Is that a whale? | 31(0.55) |
| 29 | (A) `T5` | Um. Yeah. Um, well anyway, I don't know, … | 38(0.41) |
| 30 | (B) `T5` | Ah. | 44(0.60)★far, 131(0.60)★far, 49(0.59)★far, 65(0.59)★far, 68(0.59)★far, 93(0.59)★far |
| 31 | (A) `T1` | Um, yeah, and I kind of like whales. They… | 28(0.55) |
| 32 | (D) `T1` | Alright. | 3(0.86)★far, 23(0.86), 135(0.86)★far, 147(0.86)★far, 149(0.86)★far, 152(0.85)★far |
| 33 | (A) `T1` | And they're quite harmless and mild and i… | —(고립) |
| 34 | (D) `T1` | Mm. | 118(1.00)★far, 79(0.82)★far, 9(0.79)★far, 16(0.79)★far, 88(0.79)★far, 94(0.79)★far |
| 35 | (B) `T1` | Okay. God, I still don't know what I'm go… | 60(0.41)★far |
| 36 | (D) `T6` | Superb sketch, by the way. | —(고립) |
| 37 | (A) `T6` | Tail's a bit big, I think. | 47(0.48)★far |
| 38 | (B) `T6` | I was gonna choose a dog as well. But I'l… | 19(0.61)★far, 17(0.56)★far, 11(0.54)★far, 50(0.41)★far, 29(0.41), 41(0.40) |
| 39 | (D) `T6` | Yep. | 133(1.00)★far, 12(0.82)★far, 72(0.82)★far, 99(0.82)★far, 100(0.82)★far, 146(0.82)★far |
| 40 | (B) `T6` | That doesn't really look like him, actual… | —(고립) |
| 41 | (D) `T6` | I see a dog in there. Yep. | 50(0.49), 43(0.49), 38(0.40) |
| 42 | (B) `T6` | Do you? Oh that's very good of you. | —(고립) |
| 43 | (D) `T6` | Now I see a rooster. | 41(0.49) |
| 44 | (B) `T6` | Uh. | 131(1.00)★far, 82(0.78)★far, 12(0.73)★far, 72(0.73)★far, 99(0.73)★far, 100(0.73)★far |
| 45 | (D) `T6` | What kind is it? | —(고립) |

## 읽는 법 / 왜 과분절되나
- 한 발화가 **여러 다른 T 의 발화와 연결**되거나, **★far(먼) 링크**가 있으면 community 배정이 흔들림.
- 예: 동물그리기 구간에서 beagle 발화가 intro(T0) 발화와 cos 높아 엮이면 → 그 turn 이 T0 로 튐 → 경계 오발.
- 즉 **그래프 엣지가 '같은 토픽'이 아니라 '비슷한 표현/주제어'로 생겨서**, 한 토픽 안 내용 변화·먼 우연 유사도가 community 를 쪼갬.
