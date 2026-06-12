# 요약(prototype) 구성 trace — ES2002a (turn 0–61, V_rel deploy c=1.0)

**요약(prototype) = 한 화제 안 발화들의 EWMA 가중평균.** 새 발화가 들어올 때마다 `prototype ← (1−ρ)·prototype + ρ·새발화`. ρ=mixing 가중(처음엔 큼, 길어질수록 1/(k+1)→작아짐). 경계(▲)를 치면 그 요약을 *버리고* 새 발화 하나로 다시 시작.

`★`=정답경계. 각 발화 옆 `ρ=`가 그 발화가 *그 시점 요약*에 섞인 비율. `r_act`=직전까지의 요약에서 먼 정도.

---

## ═══ 요약 #0 시작 (turn 1 부터) ═══

**[1] (D)**
> Mm-hmm. Great.
  ↳ r_act=0.67 V_rel=0.27 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[2] (A)**
> Hi, I'm David and I'm supposed to be an industrial designer.
  ↳ r_act=0.64 V_rel=0.28 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[3] (B)**
> Okay.
  ↳ r_act=0.65 V_rel=0.21 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[4] (D)**
> And I'm Andrew and I'm uh our marketing
  ↳ r_act=0.54 V_rel=0.20 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[5] (C)**
> Um I'm Craig and I'm User Interface.
  ↳ r_act=0.52 V_rel=0.18 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[6] (D)**
> expert.
  ↳ r_act=0.67 V_rel=0.19 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[7] (B)**
> Great. Okay. Um so we're designing a new remote control and um Oh I have to record who's here actually. So that's David, Andrew and Craig, isn't it? And you all arrived on time. Um yeah so des uh design a new remote control. Um, as you can see it's supposed to be original, trendy and user friendly. Um so that's kind of our our brief, as it were. Um and so there are three different stages to the design. Um I'm not really sure what what you guys have already received um in your emails. What did you get?
  ↳ r_act=0.53 V_rel=0.21 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[8] (A)** ★정답경계
> Um, I just got the project announcement about what the project is. Designing a remote control. That's about it, didn't get anything else. Did you get the same thing?
  ↳ r_act=0.64 V_rel=0.29 (임계 warmup, 경계아님) → 이 발화를 요약에 섞음

**[9] (B)**
> Mm-hmm.
  ↳ r_act=0.44 V_rel=0.10 (임계 0.27, 경계아님) → 이 발화를 요약에 섞음

**[10] (D)**
> Mm-hmm. Mm-hmm. Yeah, that's that's it.
  ↳ r_act=0.46 V_rel=0.15 (임계 0.27, 경계아님) → 이 발화를 요약에 섞음

**[11] (B)**
> Is that what everybody got? Okay. Um. So we're gonna have like individual work and then a meeting about it. And repeat that process three times. Um and at this point we get try out the whiteboard over there. Um. So uh you get to draw your favourite animal and sum up your favourite characteristics of it. So who would like to go first?
  ↳ r_act=0.64 → **V_rel=0.28 > 임계 0.27 ⟹ ▲경계!** 위 요약 버림.

## ═══ 요약 #1 시작 (turn 11 이 발화가 새 요약의 첫 재료) ═══

**[12] (C)**
> Yeah.
  ↳ r_act=0.89 V_rel=0.52 (임계 0.27, 경계아님) → 이 발화를 요약에 섞음

**[13] (D)**
> Yeah. I will go. That's fine.
  ↳ r_act=0.68 V_rel=0.28 (임계 0.34, 경계아님) → 이 발화를 요약에 섞음

**[14] (B)** ★정답경계
> Very good.
  ↳ r_act=0.66 V_rel=0.27 (임계 0.34, 경계아님) → 이 발화를 요약에 섞음

**[15] (D)**
> Alright. So This one here, right?
  ↳ r_act=0.52 V_rel=0.18 (임계 0.34, 경계아님) → 이 발화를 요약에 섞음

**[16] (B)**
> Mm-hmm.
  ↳ r_act=0.68 → **V_rel=0.42 > 임계 0.33 ⟹ ▲경계!** 위 요약 버림.

## ═══ 요약 #2 시작 (turn 16 이 발화가 새 요약의 첫 재료) ═══

**[17] (D)**
> Okay. Very nice. Alright. My favourite animal is like A beagle. Um charac favourite characteristics of it? Is that right? Uh, right, well basically um high priority for any animal for me is that they be willing to take a lot of physical affection from their family. And, yeah that they have lots of personality and uh be fit and in robust good health. So this is blue. Blue beagle. My family's beagle.
  ↳ r_act=0.91 V_rel=0.47 (임계 0.33, 경계아님) → 이 발화를 요약에 섞음

**[18] (B)**
> Yeah. Yeah. Right. Lovely.
  ↳ r_act=0.71 V_rel=0.37 (임계 0.36, 경계아님) → 이 발화를 요약에 섞음

**[19] (C)**
> Well, my favourite animal would be a monkey. Then they're small cute and furry, and uh when planet of the apes becomes real, I'm gonna be up there with them.
  ↳ r_act=0.65 V_rel=0.19 (임계 0.37, 경계아님) → 이 발화를 요약에 섞음

**[20] (B)**
> Right.
  ↳ r_act=0.59 V_rel=0.30 (임계 0.37, 경계아님) → 이 발화를 요약에 섞음

**[21] (A)**
> Cool. There's too much gear.
  ↳ r_act=0.80 → **V_rel=0.38 > 임계 0.36 ⟹ ▲경계!** 위 요약 버림.

## ═══ 요약 #3 시작 (turn 21 이 발화가 새 요약의 첫 재료) ═══

**[22] (B)**
> You can take as long over this as you like, because we haven't got an awful lot to discuss. Ok oh we do we do. Don't feel like you're in a rush, anyway.
  ↳ r_act=0.92 V_rel=0.49 (임계 0.36, 경계아님) → 이 발화를 요약에 섞음

**[23] (A)**
> Okay.
  ↳ r_act=0.71 V_rel=0.46 (임계 0.39, 경계아님) → 이 발화를 요약에 섞음

**[24] (D)**
> I coulda told you a whole lot more about beagles. Boy, let me tell you.
  ↳ r_act=0.87 V_rel=0.43 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[25] (B)**
> Ach why not We might have to get you up again then. I don't know what mine is. I'm gonna have to think on the spot now.
  ↳ r_act=0.67 V_rel=0.31 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[26] (D)**
> Impressionist.
  ↳ r_act=0.97 → **V_rel=0.48 > 임계 0.41 ⟹ ▲경계!** 위 요약 버림.

## ═══ 요약 #4 시작 (turn 26 이 발화가 새 요약의 첫 재료) ═══

**[27] (A)**
> Can't draw.
  ↳ r_act=0.74 V_rel=0.30 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[28] (B)**
> Is that a whale?
  ↳ r_act=0.74 V_rel=0.30 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[29] (A)**
> Um. Yeah. Um, well anyway, I don't know, it's just the first animal I can think off the top of my head. Um. Yes. Big reason is 'cause I'm allergic to most animals. Allergic to animal fur, so um fish was a natural choice.
  ↳ r_act=0.75 V_rel=0.34 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[30] (B)**
> Ah.
  ↳ r_act=0.77 → **V_rel=0.44 > 임계 0.40 ⟹ ▲경계!** 위 요약 버림.

## ═══ 요약 #5 시작 (turn 30 이 발화가 새 요약의 첫 재료) ═══

**[31] (A)**
> Um, yeah, and I kind of like whales. They come in and go eat everything in sight.
  ↳ r_act=0.90 V_rel=0.53 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[32] (D)**
> Alright.
  ↳ r_act=0.62 V_rel=0.38 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[33] (A)**
> And they're quite harmless and mild and interesting.
  ↳ r_act=0.74 V_rel=0.29 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[34] (D)**
> Mm.
  ↳ r_act=0.60 V_rel=0.30 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[35] (B)**
> Okay. God, I still don't know what I'm gonna write about. Um.
  ↳ r_act=0.69 V_rel=0.35 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[36] (D)**
> Superb sketch, by the way.
  ↳ r_act=0.88 V_rel=0.41 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[37] (A)**
> Tail's a bit big, I think.
  ↳ r_act=0.80 V_rel=0.36 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[38] (B)**
> I was gonna choose a dog as well. But I'll just draw a different kind of dog. M my favourite animal is my own dog at home. Um
  ↳ r_act=0.71 V_rel=0.37 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[39] (D)**
> Yep.
  ↳ r_act=0.52 V_rel=0.20 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[40] (B)**
> That doesn't really look like him, actually. He looks more like a pig, actually. Ah well.
  ↳ r_act=0.69 V_rel=0.30 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[41] (D)**
> I see a dog in there. Yep.
  ↳ r_act=0.62 V_rel=0.28 (임계 0.42, 경계아님) → 이 발화를 요약에 섞음

**[42] (B)**
> Do you? Oh that's very good of you.
  ↳ r_act=0.67 V_rel=0.25 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[43] (D)**
> Now I see a rooster.
  ↳ r_act=0.62 V_rel=0.28 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[44] (B)**
> Uh.
  ↳ r_act=0.44 V_rel=0.14 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[45] (D)**
> What kind is it?
  ↳ r_act=0.85 V_rel=0.36 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[46] (B)**
> Um he's a mixture of uh various things. Um and what do I like about him, um That's just to suggest that his tail wags. Um he's very friendly and cheery and always pleased to see you, and very kind of affectionate and um uh and he's quite quite wee as well so you know he can doesn't take up too much space. Um and uh And he does a funny thing where he chases his tail as well, which is quite amusing, so
  ↳ r_act=0.66 V_rel=0.28 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[47] (D)**
> Is he aware that th it's his own cha tail he's chasing?
  ↳ r_act=0.70 V_rel=0.31 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

**[48] (B)**
> It is. I think it is. He only does it after he's had his dinner and um he'll just all of a sudden just get up and start chasing his tail 'round the living room.
  ↳ r_act=0.71 V_rel=0.34 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[49] (D)**
> Hmm.
  ↳ r_act=0.46 V_rel=0.14 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[50] (A)**
> It's an after dinner dog then.
  ↳ r_act=0.58 V_rel=0.28 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[51] (B)**
> Yeah, so uh
  ↳ r_act=0.54 V_rel=0.20 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[52] (D)**
> Probably when he was little he got lots of attention for doing it and has forever been conditioned.
  ↳ r_act=0.88 V_rel=0.38 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[53] (B)**
> Yeah, maybe. Maybe. Right, um where did you find this? Just down here? Yeah. Okay. Um what are we doing next? Uh um. Okay, uh we now need to discuss the project finance. Um so according to the brief um we're gonna be selling this remote control for twenty five Euro, um and we're aiming to make fifty million Euro. Um so we're gonna be selling this on an international scale. And uh we don't want it to cost any more than uh twelve fifty Euros, so fifty percent of the selling price.
  ↳ r_act=0.87 V_rel=0.37 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[54] (D)** ★정답경계
> 'Kay. Um, can we just go over that again?
  ↳ r_act=0.73 V_rel=0.31 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[55] (B)**
> Sure.
  ↳ r_act=0.58 V_rel=0.24 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[56] (D)**
> Uh, so bas at twel Alright, yeah. Okay. So cost like production cost is twelve fifty, but selling price is is that wholesale or retail? Like on the shelf.
  ↳ r_act=0.88 V_rel=0.38 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[57] (B)**
> All together. Um I dunno. I imagine That's a good question.
  ↳ r_act=0.65 V_rel=0.28 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[58] (D)**
> Our sale our sale anyway. Yeah, okay okay.
  ↳ r_act=0.70 V_rel=0.31 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[59] (B)**
> I imagine it probably is our sale actually because it's probably up to the the um the retailer to uh sell it for whatever price they want. Um.
  ↳ r_act=0.84 → **V_rel=0.43 > 임계 0.40 ⟹ ▲경계!** 위 요약 버림.

## ═══ 요약 #6 시작 (turn 59 이 발화가 새 요약의 첫 재료) ═══

**[60] (D)**
> Okay. Mm-hmm. Alright.
  ↳ r_act=0.89 V_rel=0.60 (임계 0.40, 경계아님) → 이 발화를 요약에 섞음

**[61] (B)**
> But I I don't know, I mean do you think the fact that it's going to be sold internationally will have a bearing on how we design it at all?
  ↳ r_act=0.78 V_rel=0.30 (임계 0.41, 경계아님) → 이 발화를 요약에 섞음

---

**정리**: 위에서 `═══ 요약 #n ═══` 블록 하나가 prototype 하나야. 그 블록 안 발화들의 EWMA 평균이 그 화제의 '요약'이고, 다음 발화가 그 요약에서 충분히 멀면(V_rel>임계) 경계를 치고 새 블록 시작.

