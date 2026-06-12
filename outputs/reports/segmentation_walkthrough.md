# 분절 트레이스 — AMI ES2002a (turn 48–67 발췌)

같은 대화를 **(1) raw 발화 단위** 와 **(2) geometry-merge 후** 로 각각 분절한다. **둘 다 online 적응 임계치**(ewma μ+cσ, c=1.0, span=5) 사용 — 미래를 안 보고 지금까지 본 δ_eff 의 평균+표준편차로 매 turn 임계치를 갱신.

표기: `δ_eff`=직전 문맥과의 의미거리(↑=화제전환 신호), `δ*`=그 turn의 적응 임계치, `▲`=경계 예측(δ_eff≥δ*), `★`=정답 경계, `~`=geometry가 backchannel로 판정.

## (1) raw 발화 단위 + 적응 임계치

| # | 화자 | 발화 | 단어 | δ_eff | δ\* | 예측 | 정답 | geom |
|--:|:--:|------|--:|----:|----:|:--:|:--:|:--:|
| 48 | B | It is. I think it is. He only does it after he's ha… | 35 | 0.428 | 0.858 |  |  |  |
| 49 | D | Hmm. | 1 | 0.864 | 0.836 | ▲ |  | ~ |
| 50 | A | It's an after dinner dog then. | 6 | 0.687 | 0.901 |  |  | ~ |
| 51 | B | Yeah, so uh | 3 | 0.785 | 0.857 |  |  | ~ |
| 52 | D | Probably when he was little he got lots of attentio… | 18 | 0.928 | 0.860 | ▲ |  | ~ |
| 53 | B | Yeah, maybe. Maybe. Right, um where did you find th… | 93 | 0.965 | 0.935 | ▲ |  |  |
| 54 | D | 'Kay. Um, can we just go over that again? | 9 | 0.852 | 0.990 |  | ★ |  |
| 55 | B | Sure. | 1 | 0.724 | 0.964 |  |  |  |
| 56 | D | Uh, so bas at twel Alright, yeah. Okay. So cost lik… | 29 | 0.954 | 0.919 | ▲ |  | ~ |
| 57 | B | All together. Um I dunno. I imagine That's a good q… | 11 | 0.808 | 0.970 |  |  | ~ |
| 58 | D | Our sale our sale anyway. Yeah, okay okay. | 8 | 0.817 | 0.936 |  |  |  |
| 59 | B | I imagine it probably is our sale actually because … | 28 | 0.516 | 0.911 |  |  |  |
| 60 | D | Okay. Mm-hmm. Alright. | 3 | 0.822 | 0.890 |  |  | ~ |
| 61 | B | But I I don't know, I mean do you think the fact th… | 30 | 0.871 | 0.899 |  |  | ~ |
| 62 | D | Yes. | 1 | 0.834 | 0.922 |  | ★ | ~ |
| 63 | B | Think it will? Um. | 4 | 0.681 | 0.913 |  |  |  |
| 64 | D | Mm-hmm. Mm-hmm. | 2 | 0.777 | 0.870 |  |  |  |
| 65 | B | Hmm. | 1 | 0.485 | 0.855 |  |  |  |
| 66 | D | Well right away I'm wondering if there's um th th u… | 38 | 0.953 | 0.826 | ▲ |  |  |
| 67 | B | Oh yeah, regions and stuff, yeah. Yeah. Okay. Yeah.… | 31 | 0.652 | 0.947 |  |  |  |

→ 이 구간 정답 경계 = [54, 62] 중 raw 가 맞춘 것 = **없음**. 짧은 맞장구·발화마다 δ_eff 가 튀어 **거짓 경계 ▲**(5개)가 흩어지고, 정작 경계 62("Yes.")처럼 짧은 정답은 놓친다. (`~`=backchannel)

## (2) geometry-merge 후 + 적응 임계치
발화 155 → 병합 turn 92 (맞장구 흡수). 위와 같은 구간(병합 turn 30–40):

| 병합# | 흡수 raw# | 텍스트(앞부분) | δ_eff | δ\* | 예측 | 정답 |
|--:|:--|------|----:|----:|:--:|:--:|
| 30 | 48+49+50+51+52 | It is. I think it is. He only does it after h… | 0.486 | 0.862 |  |  |
| 31 | 53 | Yeah, maybe. Maybe. Right, um where did you f… | 0.909 | 0.823 | ▲ |  |
| 32 | 54 | 'Kay. Um, can we just go over that again? | 0.860 | 0.918 |  | ★ |
| 33 | 55+56+57 | Sure. Uh, so bas at twel Alright, yeah. Okay.… | 0.804 | 0.933 |  |  |
| 34 | 58 | Our sale our sale anyway. Yeah, okay okay. | 0.590 | 0.913 |  |  |
| 35 | 59+60+61+62 | I imagine it probably is our sale actually be… | 0.558 | 0.862 |  | ★ |
| 36 | 63 | Think it will? Um. | 0.719 | 0.806 |  |  |
| 37 | 64 | Mm-hmm. Mm-hmm. | 0.824 | 0.800 | ▲ |  |
| 38 | 65 | Hmm. | 0.423 | 0.846 |  |  |
| 39 | 66 | Well right away I'm wondering if there's um t… | 0.981 | 0.802 | ▲ |  |
| 40 | 67 | Oh yeah, regions and stuff, yeah. Yeah. Okay.… | 0.636 | 0.964 |  |  |

→ 맞장구가 앞 발화에 흡수돼 **독립 turn 에서 사라진다**. 이 구간 거짓 경계 5→3개, 정답 경계 맞춘 것 = **없음**. (merge 의 진짜 이득은 발췌가 아니라 미팅 전체 수치 — 아래.)

## 전체 미팅 수치 (adaptive ewma 임계치)

| 버전 | F1 ↑ | Pk ↓ | WD ↓ |
|---|--:|--:|--:|
| raw | 0.000 | 0.586 | 0.910 |
| geometry-merge | 0.160 | 0.605 | 0.837 |

## 요약
- **δ_eff** 가 신호, **δ\*(적응)** 가 문턱. 적응 임계치는 미래를 안 보고 누적 평균+표준편차로 갱신.
- **raw**: 회의 발화의 1/3이 맞장구라 δ_eff 가 흔들려 거짓 경계 다발.
- **geometry-merge**: 맞장구를 외톨이 판정해 앞 발화에 흡수 → 거짓 경계 감소, 신호 회복.
- 둘 다 동일한 online 적응 임계치를 써서 **순수하게 '단위(merge 여부)' 효과만** 비교됨.

