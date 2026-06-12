# V_rel deploy 분절 trace — ES2002a (turn 0–61)

**설정**: prototype=화제 EWMA 요약(경계서 reset) · global=최근맥락 EWMA(g_rho=0.15) · V_rel=r_active−0.6·r_global · 적응임계치 μ+1.0σ · 최소화제길이 R=4.

**기호**: `★`=정답경계, `▲reset`=우리가 친 경계(여기서 prototype 새로 시작), `요약[a..b]`=현재 prototype이 담은 화제 발화 범위.

**읽는 법**: `r_act`=이 발화가 *지금 화제 요약*에서 먼 정도(클수록 새 화제스러움). `r_glob`=최근 전체 맥락에서 먼 정도. `V_rel`=둘의 차(임계치 넘으면 경계). 정답(★)과 우리(▲)가 어긋나고, 한 번 어긋나면 요약[a..b]에 딴 화제가 섞이는 걸 보라.

---

**[1] (D)** Mm-hmm. Great.
  · 요약[0..0] (len 1) | r_act=0.67 r_glob=0.67 **V_rel=0.27** vs 임계 (warmup)

**[2] (A)** Hi, I'm David and I'm supposed to be an industrial designer.
  · 요약[0..1] (len 2) | r_act=0.64 r_glob=0.60 **V_rel=0.28** vs 임계 (warmup)

**[3] (B)** Okay.
  · 요약[0..2] (len 3) | r_act=0.65 r_glob=0.73 **V_rel=0.21** vs 임계 (warmup)

**[4] (D)** And I'm Andrew and I'm uh our marketing
  · 요약[0..3] (len 4) | r_act=0.54 r_glob=0.56 **V_rel=0.20** vs 임계 (warmup)

**[5] (C)** Um I'm Craig and I'm User Interface.
  · 요약[0..4] (len 5) | r_act=0.52 r_glob=0.57 **V_rel=0.18** vs 임계 (warmup)

**[6] (D)** expert.
  · 요약[0..5] (len 6) | r_act=0.67 r_glob=0.80 **V_rel=0.19** vs 임계 (warmup)

**[7] (B)** Great. Okay. Um so we're designing a new remote control and um Oh I...
  · 요약[0..6] (len 7) | r_act=0.53 r_glob=0.53 **V_rel=0.21** vs 임계 (warmup)

**[8] (A)** Um, I just got the project announcement about what the project is. ...
  · 요약[0..7] (len 8) | r_act=0.64 r_glob=0.58 **V_rel=0.29** vs 임계 (warmup)  ← ★정답경계

**[9] (B)** Mm-hmm.
  · 요약[0..8] (len 9) | r_act=0.44 r_glob=0.57 **V_rel=0.10** vs 임계 0.27

**[10] (D)** Mm-hmm. Mm-hmm. Yeah, that's that's it.
  · 요약[0..9] (len 10) | r_act=0.46 r_glob=0.51 **V_rel=0.15** vs 임계 0.27

**[11] (B)** Is that what everybody got? Okay. Um. So we're gonna have like indi...
  · 요약[0..10] (len 11) | r_act=0.64 r_glob=0.60 **V_rel=0.28** vs 임계 0.27  ← ▲reset

**[12] (C)** Yeah.
  · 요약[11..11] (len 1) | r_act=0.89 r_glob=0.62 **V_rel=0.52** vs 임계 0.27

**[13] (D)** Yeah. I will go. That's fine.
  · 요약[11..12] (len 2) | r_act=0.68 r_glob=0.68 **V_rel=0.28** vs 임계 0.34

**[14] (B)** Very good.
  · 요약[11..13] (len 3) | r_act=0.66 r_glob=0.66 **V_rel=0.27** vs 임계 0.34  ← ★정답경계

**[15] (D)** Alright. So This one here, right?
  · 요약[11..14] (len 4) | r_act=0.52 r_glob=0.57 **V_rel=0.18** vs 임계 0.34

**[16] (B)** Mm-hmm.
  · 요약[11..15] (len 5) | r_act=0.68 r_glob=0.43 **V_rel=0.42** vs 임계 0.33  ← ▲reset

**[17] (D)** Okay. Very nice. Alright. My favourite animal is like A beagle. Um ...
  · 요약[16..16] (len 1) | r_act=0.91 r_glob=0.73 **V_rel=0.47** vs 임계 0.33

**[18] (B)** Yeah. Yeah. Right. Lovely.
  · 요약[16..17] (len 2) | r_act=0.71 r_glob=0.56 **V_rel=0.37** vs 임계 0.36

**[19] (C)** Well, my favourite animal would be a monkey. Then they're small cut...
  · 요약[16..18] (len 3) | r_act=0.65 r_glob=0.76 **V_rel=0.19** vs 임계 0.37

**[20] (B)** Right.
  · 요약[16..19] (len 4) | r_act=0.59 r_glob=0.50 **V_rel=0.30** vs 임계 0.37

**[21] (A)** Cool. There's too much gear.
  · 요약[16..20] (len 5) | r_act=0.80 r_glob=0.69 **V_rel=0.38** vs 임계 0.36  ← ▲reset

**[22] (B)** You can take as long over this as you like, because we haven't got ...
  · 요약[21..21] (len 1) | r_act=0.92 r_glob=0.72 **V_rel=0.49** vs 임계 0.36

**[23] (A)** Okay.
  · 요약[21..22] (len 2) | r_act=0.71 r_glob=0.41 **V_rel=0.46** vs 임계 0.39

**[24] (D)** I coulda told you a whole lot more about beagles. Boy, let me tell ...
  · 요약[21..23] (len 3) | r_act=0.87 r_glob=0.73 **V_rel=0.43** vs 임계 0.40

**[25] (B)** Ach why not We might have to get you up again then. I don't know wh...
  · 요약[21..24] (len 4) | r_act=0.67 r_glob=0.61 **V_rel=0.31** vs 임계 0.41

**[26] (D)** Impressionist.
  · 요약[21..25] (len 5) | r_act=0.97 r_glob=0.81 **V_rel=0.48** vs 임계 0.41  ← ▲reset

**[27] (A)** Can't draw.
  · 요약[26..26] (len 1) | r_act=0.74 r_glob=0.75 **V_rel=0.30** vs 임계 0.41

**[28] (B)** Is that a whale?
  · 요약[26..27] (len 2) | r_act=0.74 r_glob=0.74 **V_rel=0.30** vs 임계 0.40

**[29] (A)** Um. Yeah. Um, well anyway, I don't know, it's just the first animal...
  · 요약[26..28] (len 3) | r_act=0.75 r_glob=0.67 **V_rel=0.34** vs 임계 0.40

**[30] (B)** Ah.
  · 요약[26..29] (len 4) | r_act=0.77 r_glob=0.55 **V_rel=0.44** vs 임계 0.40  ← ▲reset

**[31] (A)** Um, yeah, and I kind of like whales. They come in and go eat everyt...
  · 요약[30..30] (len 1) | r_act=0.90 r_glob=0.62 **V_rel=0.53** vs 임계 0.40

**[32] (D)** Alright.
  · 요약[30..31] (len 2) | r_act=0.62 r_glob=0.41 **V_rel=0.38** vs 임계 0.42

**[33] (A)** And they're quite harmless and mild and interesting.
  · 요약[30..32] (len 3) | r_act=0.74 r_glob=0.76 **V_rel=0.29** vs 임계 0.42

**[34] (D)** Mm.
  · 요약[30..33] (len 4) | r_act=0.60 r_glob=0.50 **V_rel=0.30** vs 임계 0.42

**[35] (B)** Okay. God, I still don't know what I'm gonna write about. Um.
  · 요약[30..34] (len 5) | r_act=0.69 r_glob=0.57 **V_rel=0.35** vs 임계 0.42

**[36] (D)** Superb sketch, by the way.
  · 요약[30..35] (len 6) | r_act=0.88 r_glob=0.78 **V_rel=0.41** vs 임계 0.42

**[37] (A)** Tail's a bit big, I think.
  · 요약[30..36] (len 7) | r_act=0.80 r_glob=0.72 **V_rel=0.36** vs 임계 0.42

**[38] (B)** I was gonna choose a dog as well. But I'll just draw a different ki...
  · 요약[30..37] (len 8) | r_act=0.71 r_glob=0.58 **V_rel=0.37** vs 임계 0.42

**[39] (D)** Yep.
  · 요약[30..38] (len 9) | r_act=0.52 r_glob=0.53 **V_rel=0.20** vs 임계 0.42

**[40] (B)** That doesn't really look like him, actually. He looks more like a p...
  · 요약[30..39] (len 10) | r_act=0.69 r_glob=0.66 **V_rel=0.30** vs 임계 0.42

**[41] (D)** I see a dog in there. Yep.
  · 요약[30..40] (len 11) | r_act=0.62 r_glob=0.57 **V_rel=0.28** vs 임계 0.42

**[42] (B)** Do you? Oh that's very good of you.
  · 요약[30..41] (len 12) | r_act=0.67 r_glob=0.70 **V_rel=0.25** vs 임계 0.41

**[43] (D)** Now I see a rooster.
  · 요약[30..42] (len 13) | r_act=0.62 r_glob=0.57 **V_rel=0.28** vs 임계 0.41

**[44] (B)** Uh.
  · 요약[30..43] (len 14) | r_act=0.44 r_glob=0.51 **V_rel=0.14** vs 임계 0.41

**[45] (D)** What kind is it?
  · 요약[30..44] (len 15) | r_act=0.85 r_glob=0.82 **V_rel=0.36** vs 임계 0.41

**[46] (B)** Um he's a mixture of uh various things. Um and what do I like about...
  · 요약[30..45] (len 16) | r_act=0.66 r_glob=0.63 **V_rel=0.28** vs 임계 0.41

**[47] (D)** Is he aware that th it's his own cha tail he's chasing?
  · 요약[30..46] (len 17) | r_act=0.70 r_glob=0.65 **V_rel=0.31** vs 임계 0.41

**[48] (B)** It is. I think it is. He only does it after he's had his dinner and...
  · 요약[30..47] (len 18) | r_act=0.71 r_glob=0.63 **V_rel=0.34** vs 임계 0.40

**[49] (D)** Hmm.
  · 요약[30..48] (len 19) | r_act=0.46 r_glob=0.53 **V_rel=0.14** vs 임계 0.40

**[50] (A)** It's an after dinner dog then.
  · 요약[30..49] (len 20) | r_act=0.58 r_glob=0.50 **V_rel=0.28** vs 임계 0.40

**[51] (B)** Yeah, so uh
  · 요약[30..50] (len 21) | r_act=0.54 r_glob=0.57 **V_rel=0.20** vs 임계 0.40

**[52] (D)** Probably when he was little he got lots of attention for doing it a...
  · 요약[30..51] (len 22) | r_act=0.88 r_glob=0.83 **V_rel=0.38** vs 임계 0.40

**[53] (B)** Yeah, maybe. Maybe. Right, um where did you find this? Just down he...
  · 요약[30..52] (len 23) | r_act=0.87 r_glob=0.84 **V_rel=0.37** vs 임계 0.40

**[54] (D)** 'Kay. Um, can we just go over that again?
  · 요약[30..53] (len 24) | r_act=0.73 r_glob=0.71 **V_rel=0.31** vs 임계 0.40  ← ★정답경계

**[55] (B)** Sure.
  · 요약[30..54] (len 25) | r_act=0.58 r_glob=0.57 **V_rel=0.24** vs 임계 0.40

**[56] (D)** Uh, so bas at twel Alright, yeah. Okay. So cost like production cos...
  · 요약[30..55] (len 26) | r_act=0.88 r_glob=0.82 **V_rel=0.38** vs 임계 0.40

**[57] (B)** All together. Um I dunno. I imagine That's a good question.
  · 요약[30..56] (len 27) | r_act=0.65 r_glob=0.63 **V_rel=0.28** vs 임계 0.40

**[58] (D)** Our sale our sale anyway. Yeah, okay okay.
  · 요약[30..57] (len 28) | r_act=0.70 r_glob=0.65 **V_rel=0.31** vs 임계 0.40

**[59] (B)** I imagine it probably is our sale actually because it's probably up...
  · 요약[30..58] (len 29) | r_act=0.84 r_glob=0.68 **V_rel=0.43** vs 임계 0.40  ← ▲reset

**[60] (D)** Okay. Mm-hmm. Alright.
  · 요약[59..59] (len 1) | r_act=0.89 r_glob=0.48 **V_rel=0.60** vs 임계 0.40

**[61] (B)** But I I don't know, I mean do you think the fact that it's going to...
  · 요약[59..60] (len 2) | r_act=0.78 r_glob=0.79 **V_rel=0.30** vs 임계 0.41

---

**이 구간 요약**: 정답경계 [8, 14, 54], 우리경계(▲) [11, 16, 21, 26, 30, 59].
정답 3개 중 ±2 적중 1개.

