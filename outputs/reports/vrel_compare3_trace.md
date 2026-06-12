# 3-way 분절 비교 trace — ES2002a (turn 0–61)

| 방법 | reset 기준 | 신호 | 상태 |
|---|---|---|---|
| **clean** | **정답경계(gold)** | V_rel, μ+2σ | 천장 ±2F1 0.554 (배포불가, 컨닝) |
| **V_rel c2.0** | 자기 검출 | V_rel, μ+2σ | deploy best Score 0.358 |
| **δ_eff** | (없음, 2턴윈도우) | δ_eff, ewma | 공식 main, Score 0.203 |

`★`=정답 · `▲`=그 방법이 친 경계(=reset) · `요약[a..b]`=현재 prototype 범위.

**핵심 관전**: clean은 요약이 *정답에서만* 새로 시작해 안 섞임 → V_rel이 경계서 또렷. V_rel c2.0은 자기 추측으로 reset해 요약 오염. δ_eff는 요약 없이 2턴만 봄.

---

**[1] (D)** Mm-hmm. Great.
  · **clean**   요약[0..0] r_act=0.67 V=0.27/—  
  · **V_rel2.0** 요약[0..0] r_act=0.67 V=0.27/—  
  · **δ_eff**    δ=0.67                     ▲

**[2] (A)** Hi, I'm David and I'm supposed to be an industrial designer.
  · **clean**   요약[0..1] r_act=0.64 V=0.28/—  
  · **V_rel2.0** 요약[0..1] r_act=0.64 V=0.28/—  
  · **δ_eff**    δ=0.73                     ▲

**[3] (B)** Okay.
  · **clean**   요약[0..2] r_act=0.65 V=0.21/—  
  · **V_rel2.0** 요약[0..2] r_act=0.65 V=0.21/—  
  · **δ_eff**    δ=0.75                      

**[4] (D)** And I'm Andrew and I'm uh our marketing
  · **clean**   요약[0..3] r_act=0.54 V=0.20/—  
  · **V_rel2.0** 요약[0..3] r_act=0.54 V=0.20/—  
  · **δ_eff**    δ=0.66                      

**[5] (C)** Um I'm Craig and I'm User Interface.
  · **clean**   요약[0..4] r_act=0.52 V=0.18/—  
  · **V_rel2.0** 요약[0..4] r_act=0.52 V=0.18/—  
  · **δ_eff**    δ=0.54                     ▲

**[6] (D)** expert.
  · **clean**   요약[0..5] r_act=0.67 V=0.19/—  
  · **V_rel2.0** 요약[0..5] r_act=0.67 V=0.19/—  
  · **δ_eff**    δ=0.81                      

**[7] (B)** Great. Okay. Um so we're designing a new remote control and u...
  · **clean**   요약[0..6] r_act=0.53 V=0.21/—  
  · **V_rel2.0** 요약[0..6] r_act=0.53 V=0.21/—  
  · **δ_eff**    δ=0.78                      

**[8] (A)** Um, I just got the project announcement about what the projec...  ★정답
  · **clean**   요약[0..7] r_act=0.64 V=0.29/—  
  · **V_rel2.0** 요약[0..7] r_act=0.64 V=0.29/—  
  · **δ_eff**    δ=0.42                     ▲

**[9] (B)** Mm-hmm.
  · **clean**   요약[8..8] r_act=0.92 V=0.57/—  
  · **V_rel2.0** 요약[0..8] r_act=0.44 V=0.10/0.31  
  · **δ_eff**    δ=0.90                      

**[10] (D)** Mm-hmm. Mm-hmm. Yeah, that's that's it.
  · **clean**   요약[8..9] r_act=0.42 V=0.12/0.52  
  · **V_rel2.0** 요약[0..9] r_act=0.46 V=0.15/0.33  
  · **δ_eff**    δ=0.29                      

**[11] (B)** Is that what everybody got? Okay. Um. So we're gonna have lik...
  · **clean**   요약[8..10] r_act=0.81 V=0.45/0.51  
  · **V_rel2.0** 요약[0..10] r_act=0.64 V=0.28/0.33  
  · **δ_eff**    δ=0.86                      

**[12] (C)** Yeah.
  · **clean**   요약[8..11] r_act=0.65 V=0.28/0.55  
  · **V_rel2.0** 요약[0..11] r_act=0.59 V=0.22/0.33  
  · **δ_eff**    δ=0.81                      

**[13] (D)** Yeah. I will go. That's fine.
  · **clean**   요약[8..12] r_act=0.75 V=0.34/0.53  
  · **V_rel2.0** 요약[0..12] r_act=0.70 V=0.30/0.33  
  · **δ_eff**    δ=0.68                      

**[14] (B)** Very good.  ★정답
  · **clean**   요약[8..13] r_act=0.74 V=0.35/0.53  
  · **V_rel2.0** 요약[0..13] r_act=0.62 V=0.23/0.34  
  · **δ_eff**    δ=0.67                      

**[15] (D)** Alright. So This one here, right?
  · **clean**   요약[14..14] r_act=0.80 V=0.46/0.53  
  · **V_rel2.0** 요약[0..14] r_act=0.60 V=0.26/0.34  
  · **δ_eff**    δ=0.75                      

**[16] (B)** Mm-hmm.
  · **clean**   요약[14..15] r_act=0.75 V=0.49/0.56  
  · **V_rel2.0** 요약[0..15] r_act=0.36 V=0.10/0.34  
  · **δ_eff**    δ=0.77                      

**[17] (D)** Okay. Very nice. Alright. My favourite animal is like A beagl...
  · **clean**   요약[14..16] r_act=0.77 V=0.33/0.58  
  · **V_rel2.0** 요약[0..16] r_act=0.75 V=0.31/0.34  
  · **δ_eff**    δ=0.87                      

**[18] (B)** Yeah. Yeah. Right. Lovely.
  · **clean**   요약[14..17] r_act=0.53 V=0.20/0.57  
  · **V_rel2.0** 요약[0..17] r_act=0.60 V=0.27/0.35  
  · **δ_eff**    δ=0.73                      

**[19] (C)** Well, my favourite animal would be a monkey. Then they're sma...
  · **clean**   요약[14..18] r_act=0.76 V=0.31/0.56  
  · **V_rel2.0** 요약[0..18] r_act=0.82 V=0.37/0.35 ▲
  · **δ_eff**    δ=0.77                      

**[20] (B)** Right.
  · **clean**   요약[14..19] r_act=0.47 V=0.18/0.56  
  · **V_rel2.0** 요약[19..19] r_act=0.92 V=0.62/0.35  
  · **δ_eff**    δ=0.80                      

**[21] (A)** Cool. There's too much gear.
  · **clean**   요약[14..20] r_act=0.76 V=0.34/0.55  
  · **V_rel2.0** 요약[19..20] r_act=0.85 V=0.43/0.47  
  · **δ_eff**    δ=0.80                     ▲

**[22] (B)** You can take as long over this as you like, because we haven'...
  · **clean**   요약[14..21] r_act=0.82 V=0.38/0.54  
  · **V_rel2.0** 요약[19..21] r_act=0.92 V=0.48/0.48  
  · **δ_eff**    δ=0.91                      

**[23] (A)** Okay.
  · **clean**   요약[14..22] r_act=0.42 V=0.18/0.54  
  · **V_rel2.0** 요약[19..22] r_act=0.50 V=0.25/0.51  
  · **δ_eff**    δ=0.76                      

**[24] (D)** I coulda told you a whole lot more about beagles. Boy, let me...
  · **clean**   요약[14..23] r_act=0.69 V=0.25/0.54  
  · **V_rel2.0** 요약[19..23] r_act=0.76 V=0.32/0.50  
  · **δ_eff**    δ=0.82                      

**[25] (B)** Ach why not We might have to get you up again then. I don't k...
  · **clean**   요약[14..24] r_act=0.67 V=0.31/0.53  
  · **V_rel2.0** 요약[19..24] r_act=0.68 V=0.32/0.50  
  · **δ_eff**    δ=0.82                     ▲

**[26] (D)** Impressionist.
  · **clean**   요약[14..25] r_act=0.76 V=0.27/0.53  
  · **V_rel2.0** 요약[19..25] r_act=0.86 V=0.37/0.50  
  · **δ_eff**    δ=0.98                      

**[27] (A)** Can't draw.
  · **clean**   요약[14..26] r_act=0.78 V=0.33/0.52  
  · **V_rel2.0** 요약[19..26] r_act=0.77 V=0.32/0.50  
  · **δ_eff**    δ=0.74                      

**[28] (B)** Is that a whale?
  · **clean**   요약[14..27] r_act=0.71 V=0.27/0.52  
  · **V_rel2.0** 요약[19..27] r_act=0.74 V=0.29/0.50  
  · **δ_eff**    δ=0.81                      

**[29] (A)** Um. Yeah. Um, well anyway, I don't know, it's just the first ...
  · **clean**   요약[14..28] r_act=0.69 V=0.29/0.51  
  · **V_rel2.0** 요약[19..28] r_act=0.64 V=0.23/0.50  
  · **δ_eff**    δ=0.68                      

**[30] (B)** Ah.
  · **clean**   요약[14..29] r_act=0.53 V=0.20/0.51  
  · **V_rel2.0** 요약[19..29] r_act=0.58 V=0.25/0.49  
  · **δ_eff**    δ=0.86                      

**[31] (A)** Um, yeah, and I kind of like whales. They come in and go eat ...
  · **clean**   요약[14..30] r_act=0.65 V=0.28/0.50  
  · **V_rel2.0** 요약[19..30] r_act=0.62 V=0.25/0.49  
  · **δ_eff**    δ=0.83                      

**[32] (D)** Alright.
  · **clean**   요약[14..31] r_act=0.37 V=0.12/0.50  
  · **V_rel2.0** 요약[19..31] r_act=0.42 V=0.17/0.48  
  · **δ_eff**    δ=0.79                      

**[33] (A)** And they're quite harmless and mild and interesting.
  · **clean**   요약[14..32] r_act=0.78 V=0.32/0.50  
  · **V_rel2.0** 요약[19..32] r_act=0.74 V=0.29/0.48  
  · **δ_eff**    δ=0.85                      

**[34] (D)** Mm.
  · **clean**   요약[14..33] r_act=0.49 V=0.19/0.50  
  · **V_rel2.0** 요약[19..33] r_act=0.54 V=0.24/0.48  
  · **δ_eff**    δ=0.77                      

**[35] (B)** Okay. God, I still don't know what I'm gonna write about. Um.
  · **clean**   요약[14..34] r_act=0.59 V=0.25/0.49  
  · **V_rel2.0** 요약[19..34] r_act=0.60 V=0.26/0.47  
  · **δ_eff**    δ=0.74                     ▲

**[36] (D)** Superb sketch, by the way.
  · **clean**   요약[14..35] r_act=0.72 V=0.25/0.49  
  · **V_rel2.0** 요약[19..35] r_act=0.80 V=0.33/0.47  
  · **δ_eff**    δ=0.91                      

**[37] (A)** Tail's a bit big, I think.
  · **clean**   요약[14..36] r_act=0.69 V=0.26/0.49  
  · **V_rel2.0** 요약[19..36] r_act=0.69 V=0.25/0.47  
  · **δ_eff**    δ=0.85                      

**[38] (B)** I was gonna choose a dog as well. But I'll just draw a differ...
  · **clean**   요약[14..37] r_act=0.57 V=0.23/0.48  
  · **V_rel2.0** 요약[19..37] r_act=0.53 V=0.19/0.47  
  · **δ_eff**    δ=0.81                      

**[39] (D)** Yep.
  · **clean**   요약[14..38] r_act=0.49 V=0.17/0.48  
  · **V_rel2.0** 요약[19..38] r_act=0.53 V=0.22/0.46  
  · **δ_eff**    δ=0.87                      

**[40] (B)** That doesn't really look like him, actually. He looks more li...
  · **clean**   요약[14..39] r_act=0.66 V=0.27/0.48  
  · **V_rel2.0** 요약[19..39] r_act=0.66 V=0.26/0.46  
  · **δ_eff**    δ=0.84                      

**[41] (D)** I see a dog in there. Yep.
  · **clean**   요약[14..40] r_act=0.62 V=0.28/0.47  
  · **V_rel2.0** 요약[19..40] r_act=0.62 V=0.27/0.46  
  · **δ_eff**    δ=0.65                      

**[42] (B)** Do you? Oh that's very good of you.
  · **clean**   요약[14..41] r_act=0.66 V=0.24/0.47  
  · **V_rel2.0** 요약[19..41] r_act=0.72 V=0.31/0.45  
  · **δ_eff**    δ=0.85                      

**[43] (D)** Now I see a rooster.
  · **clean**   요약[14..42] r_act=0.64 V=0.29/0.47  
  · **V_rel2.0** 요약[19..42] r_act=0.60 V=0.26/0.45  
  · **δ_eff**    δ=0.76                      

**[44] (B)** Uh.
  · **clean**   요약[14..43] r_act=0.50 V=0.19/0.47  
  · **V_rel2.0** 요약[19..43] r_act=0.50 V=0.20/0.45  
  · **δ_eff**    δ=0.76                     ▲

**[45] (D)** What kind is it?
  · **clean**   요약[14..44] r_act=0.80 V=0.31/0.46  
  · **V_rel2.0** 요약[19..44] r_act=0.84 V=0.35/0.45  
  · **δ_eff**    δ=0.92                      

**[46] (B)** Um he's a mixture of uh various things. Um and what do I like...
  · **clean**   요약[14..45] r_act=0.65 V=0.27/0.46  
  · **V_rel2.0** 요약[19..45] r_act=0.66 V=0.28/0.45  
  · **δ_eff**    δ=0.81                      

**[47] (D)** Is he aware that th it's his own cha tail he's chasing?
  · **clean**   요약[14..46] r_act=0.69 V=0.31/0.46  
  · **V_rel2.0** 요약[19..46] r_act=0.70 V=0.31/0.45  
  · **δ_eff**    δ=0.69                      

**[48] (B)** It is. I think it is. He only does it after he's had his dinn...
  · **clean**   요약[14..47] r_act=0.70 V=0.33/0.46  
  · **V_rel2.0** 요약[19..47] r_act=0.74 V=0.36/0.45  
  · **δ_eff**    δ=0.43                      

**[49] (D)** Hmm.
  · **clean**   요약[14..48] r_act=0.49 V=0.17/0.46  
  · **V_rel2.0** 요약[19..48] r_act=0.53 V=0.21/0.45  
  · **δ_eff**    δ=0.86                      

**[50] (A)** It's an after dinner dog then.
  · **clean**   요약[14..49] r_act=0.57 V=0.27/0.46  
  · **V_rel2.0** 요약[19..49] r_act=0.59 V=0.29/0.45  
  · **δ_eff**    δ=0.69                      

**[51] (B)** Yeah, so uh
  · **clean**   요약[14..50] r_act=0.53 V=0.19/0.45  
  · **V_rel2.0** 요약[19..50] r_act=0.56 V=0.22/0.45  
  · **δ_eff**    δ=0.79                     ▲

**[52] (D)** Probably when he was little he got lots of attention for doin...
  · **clean**   요약[14..51] r_act=0.85 V=0.36/0.45  
  · **V_rel2.0** 요약[19..51] r_act=0.88 V=0.38/0.44  
  · **δ_eff**    δ=0.93                      

**[53] (B)** Yeah, maybe. Maybe. Right, um where did you find this? Just d...
  · **clean**   요약[14..52] r_act=0.84 V=0.33/0.45  
  · **V_rel2.0** 요약[19..52] r_act=0.84 V=0.33/0.45  
  · **δ_eff**    δ=0.97                      

**[54] (D)** 'Kay. Um, can we just go over that again?  ★정답
  · **clean**   요약[14..53] r_act=0.69 V=0.27/0.45  
  · **V_rel2.0** 요약[19..53] r_act=0.69 V=0.26/0.45  
  · **δ_eff**    δ=0.85                      

**[55] (B)** Sure.
  · **clean**   요약[54..54] r_act=0.73 V=0.39/0.45  
  · **V_rel2.0** 요약[19..54] r_act=0.52 V=0.18/0.45  
  · **δ_eff**    δ=0.72                      

**[56] (D)** Uh, so bas at twel Alright, yeah. Okay. So cost like producti...
  · **clean**   요약[54..55] r_act=0.95 V=0.46/0.46  
  · **V_rel2.0** 요약[19..55] r_act=0.90 V=0.40/0.44  
  · **δ_eff**    δ=0.95                      

**[57] (B)** All together. Um I dunno. I imagine That's a good question.
  · **clean**   요약[54..56] r_act=0.72 V=0.35/0.47  
  · **V_rel2.0** 요약[19..56] r_act=0.68 V=0.30/0.45  
  · **δ_eff**    δ=0.81                      

**[58] (D)** Our sale our sale anyway. Yeah, okay okay.
  · **clean**   요약[54..57] r_act=0.61 V=0.22/0.47  
  · **V_rel2.0** 요약[19..57] r_act=0.68 V=0.29/0.45  
  · **δ_eff**    δ=0.82                      

**[59] (B)** I imagine it probably is our sale actually because it's proba...
  · **clean**   요약[54..58] r_act=0.64 V=0.23/0.46  
  · **V_rel2.0** 요약[19..58] r_act=0.82 V=0.41/0.45  
  · **δ_eff**    δ=0.52                      

**[60] (D)** Okay. Mm-hmm. Alright.
  · **clean**   요약[54..59] r_act=0.54 V=0.25/0.46  
  · **V_rel2.0** 요약[19..59] r_act=0.48 V=0.19/0.45  
  · **δ_eff**    δ=0.82                      

**[61] (B)** But I I don't know, I mean do you think the fact that it's go...
  · **clean**   요약[54..60] r_act=0.79 V=0.31/0.46  
  · **V_rel2.0** 요약[19..60] r_act=0.84 V=0.36/0.45  
  · **δ_eff**    δ=0.87                      

---

**구간 요약** (정답 [8, 14, 54]):
- clean    경계 [] — ±2 적중 0/3
- V_rel2.0 경계 [19] — ±2 적중 0/3
- δ_eff    경계 [1, 2, 5, 8, 21, 25, 35, 44, 51] — ±2 적중 1/3

