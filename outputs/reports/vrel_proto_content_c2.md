# prototype(요약벡터)이 담은 내용 — ES2002a (V_rel deploy **c=2.0 best**)

**총 155발화 → 검출 경계 1개 → 요약(segment) 2개.** 정답경계 [8, 14, 54, 62, 97, 133].

⚠️ **prototype은 텍스트 요약이 아니라 발화 임베딩들의 EWMA 평균 *벡터*.** 사람이 읽게, 각 segment의 **최종 prototype 벡터에 가장 가까운 실제 발화 = 『대표발화』**로 그 요약이 '대략 무슨 내용인지' 표시. 그 아래 그 요약을 *구성한 전체 발화*를 prototype과의 유사도(cos)와 함께 나열.

---

## ═══ 요약 #0 : turn 0–18 (19발화)  ★정답경계포함: [8, 14] ═══

**『대표발화』 (prototype에 가장 가까움, cos=0.73) — [1] (D):**
> Mm-hmm. Great.

**이 요약을 구성한 전체 발화 (prototype과의 cos 유사도):**
- `[0]` cos=0.65 (B): Okay. Right. Um well this is the kick-off meeting for our our project. Um and um this is just what we're gonna be doing over the next twenty five minutes. Um so first of all, just to kind of make sure that we all know each other, I'm Laura and I'm the project manager. Do you want to introduce yourself again?
- `[1]` cos=0.73 ◀대표 (D): Mm-hmm. Great.
- `[2]` cos=0.55 (A): Hi, I'm David and I'm supposed to be an industrial designer.
- `[3]` cos=0.60 (B): Okay.
- `[4]` cos=0.54 (D): And I'm Andrew and I'm uh our marketing
- `[5]` cos=0.55 (C): Um I'm Craig and I'm User Interface.
- `[6]` cos=0.45 (D): expert.
- `[7]` cos=0.55 (B): Great. Okay. Um so we're designing a new remote control and um Oh I have to record who's here actually. So that's David, Andrew and Craig, isn't it? And you all arrived on time. Um yeah so des uh design a new remote control. Um, as you can see it's supposed to be original, trendy and user friendly. Um so that's kind of our our brief, as it were. Um and so there are three different stages to the design. Um I'm not really sure what what you guys have already received um in your emails. What did you get?
- `[8]` cos=0.38 ★ (A): Um, I just got the project announcement about what the project is. Designing a remote control. That's about it, didn't get anything else. Did you get the same thing?
- `[9]` cos=0.67 (B): Mm-hmm.
- `[10]` cos=0.62 (D): Mm-hmm. Mm-hmm. Yeah, that's that's it.
- `[11]` cos=0.44 (B): Is that what everybody got? Okay. Um. So we're gonna have like individual work and then a meeting about it. And repeat that process three times. Um and at this point we get try out the whiteboard over there. Um. So uh you get to draw your favourite animal and sum up your favourite characteristics of it. So who would like to go first?
- `[12]` cos=0.52 (C): Yeah.
- `[13]` cos=0.39 (D): Yeah. I will go. That's fine.
- `[14]` cos=0.44 ★ (B): Very good.
- `[15]` cos=0.46 (D): Alright. So This one here, right?
- `[16]` cos=0.67 (B): Mm-hmm.
- `[17]` cos=0.31 (D): Okay. Very nice. Alright. My favourite animal is like A beagle. Um charac favourite characteristics of it? Is that right? Uh, right, well basically um high priority for any animal for me is that they be willing to take a lot of physical affection from their family. And, yeah that they have lots of personality and uh be fit and in robust good health. So this is blue. Blue beagle. My family's beagle.
- `[18]` cos=0.45 (B): Yeah. Yeah. Right. Lovely.

## ═══ 요약 #1 : turn 19–154 (136발화)  ★정답경계포함: [54, 62, 97, 133] ═══

**『대표발화』 (prototype에 가장 가까움, cos=0.72) — [144] (D):**
> Okay, yeah.

**이 요약을 구성한 전체 발화 (prototype과의 cos 유사도):**
- `[19]` cos=0.15 (C): Well, my favourite animal would be a monkey. Then they're small cute and furry, and uh when planet of the apes becomes real, I'm gonna be up there with them.
- `[20]` cos=0.63 (B): Right.
- `[21]` cos=0.40 (A): Cool. There's too much gear.
- `[22]` cos=0.30 (B): You can take as long over this as you like, because we haven't got an awful lot to discuss. Ok oh we do we do. Don't feel like you're in a rush, anyway.
- `[23]` cos=0.71 (A): Okay.
- `[24]` cos=0.22 (D): I coulda told you a whole lot more about beagles. Boy, let me tell you.
- `[25]` cos=0.34 (B): Ach why not We might have to get you up again then. I don't know what mine is. I'm gonna have to think on the spot now.
- `[26]` cos=0.19 (D): Impressionist.
- `[27]` cos=0.20 (A): Can't draw.
- `[28]` cos=0.16 (B): Is that a whale?
- `[29]` cos=0.22 (A): Um. Yeah. Um, well anyway, I don't know, it's just the first animal I can think off the top of my head. Um. Yes. Big reason is 'cause I'm allergic to most animals. Allergic to animal fur, so um fish was a natural choice.
- `[30]` cos=0.56 (B): Ah.
- `[31]` cos=0.23 (A): Um, yeah, and I kind of like whales. They come in and go eat everything in sight.
- `[32]` cos=0.69 (D): Alright.
- `[33]` cos=0.17 (A): And they're quite harmless and mild and interesting.
- `[34]` cos=0.56 (D): Mm.
- `[35]` cos=0.44 (B): Okay. God, I still don't know what I'm gonna write about. Um.
- `[36]` cos=0.16 (D): Superb sketch, by the way.
- `[37]` cos=0.23 (A): Tail's a bit big, I think.
- `[38]` cos=0.21 (B): I was gonna choose a dog as well. But I'll just draw a different kind of dog. M my favourite animal is my own dog at home. Um
- `[39]` cos=0.64 (D): Yep.
- `[40]` cos=0.18 (B): That doesn't really look like him, actually. He looks more like a pig, actually. Ah well.
- `[41]` cos=0.22 (D): I see a dog in there. Yep.
- `[42]` cos=0.31 (B): Do you? Oh that's very good of you.
- `[43]` cos=0.25 (D): Now I see a rooster.
- `[44]` cos=0.62 (B): Uh.
- `[45]` cos=0.06 (D): What kind is it?
- `[46]` cos=0.19 (B): Um he's a mixture of uh various things. Um and what do I like about him, um That's just to suggest that his tail wags. Um he's very friendly and cheery and always pleased to see you, and very kind of affectionate and um uh and he's quite quite wee as well so you know he can doesn't take up too much space. Um and uh And he does a funny thing where he chases his tail as well, which is quite amusing, so
- `[47]` cos=0.20 (D): Is he aware that th it's his own cha tail he's chasing?
- `[48]` cos=0.20 (B): It is. I think it is. He only does it after he's had his dinner and um he'll just all of a sudden just get up and start chasing his tail 'round the living room.
- `[49]` cos=0.63 (D): Hmm.
- `[50]` cos=0.26 (A): It's an after dinner dog then.
- `[51]` cos=0.57 (B): Yeah, so uh
- `[52]` cos=0.07 (D): Probably when he was little he got lots of attention for doing it and has forever been conditioned.
- `[53]` cos=0.37 (B): Yeah, maybe. Maybe. Right, um where did you find this? Just down here? Yeah. Okay. Um what are we doing next? Uh um. Okay, uh we now need to discuss the project finance. Um so according to the brief um we're gonna be selling this remote control for twenty five Euro, um and we're aiming to make fifty million Euro. Um so we're gonna be selling this on an international scale. And uh we don't want it to cost any more than uh twelve fifty Euros, so fifty percent of the selling price.
- `[54]` cos=0.38 ★ (D): 'Kay. Um, can we just go over that again?
- `[55]` cos=0.54 (B): Sure.
- `[56]` cos=0.24 (D): Uh, so bas at twel Alright, yeah. Okay. So cost like production cost is twelve fifty, but selling price is is that wholesale or retail? Like on the shelf.
- `[57]` cos=0.39 (B): All together. Um I dunno. I imagine That's a good question.
- `[58]` cos=0.42 (D): Our sale our sale anyway. Yeah, okay okay.
- `[59]` cos=0.25 (B): I imagine it probably is our sale actually because it's probably up to the the um the retailer to uh sell it for whatever price they want. Um.
- `[60]` cos=0.64 (D): Okay. Mm-hmm. Alright.
- `[61]` cos=0.26 (B): But I I don't know, I mean do you think the fact that it's going to be sold internationally will have a bearing on how we design it at all?
- `[62]` cos=0.60 ★ (D): Yes.
- `[63]` cos=0.38 (B): Think it will? Um.
- `[64]` cos=0.56 (D): Mm-hmm. Mm-hmm.
- `[65]` cos=0.63 (B): Hmm.
- `[66]` cos=0.24 (D): Well right away I'm wondering if there's um th th uh, like with D_V_D_ players, if there are zones. Um f frequencies or something um as well as uh characters, um different uh keypad styles and s symbols.
- `[67]` cos=0.39 (B): Oh yeah, regions and stuff, yeah. Yeah. Okay. Yeah. Well for a remote control, do you think that will be I suppose it's depends on how complicated our remote control is.
- `[68]` cos=0.63 (A): Hmm.
- `[69]` cos=0.51 (D): Um. I don't know. Yeah.
- `[70]` cos=0.27 (A): It does make sense from maybe the design point of view 'cause you have more complicated characters like European languages, then you need more buttons.
- `[71]` cos=0.68 (B): Yeah, yeah. Okay.
- `[72]` cos=0.70 (D): Yeah.
- `[73]` cos=0.55 (A): So, possibly.
- `[74]` cos=0.20 (D): Yeah. And then a and then al the other thing international is on top of the price. I'm thinking the price might might appeal to a certain market in one region, whereas in another it'll be different, so
- `[75]` cos=0.19 (B): What, just like in terms of like the wealth of the country? Like how much money people have to spend on things like?
- `[76]` cos=0.29 (D): Just a chara just a characteristic of the Just Or just like, basic product podi positioning, the twenty five Euro remote control might be a big hit in London, might not be such a big hit in Greece, who knows, something like that, yeah.
- `[77]` cos=0.44 (B): Aye, I see what you mean, yeah. Marketing. Good marketing thoughts. Oh gosh, I should be writing all this down. Um.
- `[78]` cos=0.26 (D): Yep. Right away I'm making some kind of assumptions about what what information we're given here, thinking, 'kay trendy probably means something other than just basic, something other than just standard. Um
- `[79]` cos=0.67 (B): Mm. Yeah.
- `[80]` cos=0.23 (D): so I'm wondering right away, is selling twenty five Euros, is that sort of the thi is this gonna to be like the premium product kinda thing or
- `[81]` cos=0.40 (B): Yeah, yeah. Like how much does, you know, a remote control cost.
- `[82]` cos=0.67 (D): Uh-huh.
- `[83]` cos=0.21 (B): Well twenty five Euro, I mean that's um that's about like eighteen pounds or something, isn't it? Or no, is it as much as that? Sixteen seventeen eighteen pounds.
- `[84]` cos=0.63 (D): Mm-hmm. Yep. Yeah, I'd say so, yeah.
- `[85]` cos=0.18 (B): Um, I dunno, I've never bought a remote control, so I don't know how how good a remote control that would get you. Um.
- `[86]` cos=0.56 (D): No. Yeah, yeah.
- `[87]` cos=0.33 (B): But yeah, I suppose it has to look kind of cool and gimmicky.
- `[88]` cos=0.61 (D): Mm-hmm.
- `[89]` cos=0.40 (B): Um right, okay. Let me just scoot on ahead here. Okay. Um well d Does anybody have anything to add to uh to the finance issue at all?
- `[90]` cos=0.24 (D): Do we have any other background information on like how that compares to other
- `[91]` cos=0.33 (B): Thin No, actually. That would be useful, though, wouldn't it, if you knew like what your money would get you now.
- `[92]` cos=0.53 (D): other Yeah.
- `[93]` cos=0.63 (A): Hmm.
- `[94]` cos=0.61 (D): Mm-hmm.
- `[95]` cos=0.61 (B): Mm-hmm.
- `[96]` cos=0.31 (D): Yeah, interesting thing about discussing um production of a remote control for me is that l as you point out, I just don't think of remote controls as somethin something people consciously assess in their purchasing habits. It's just like getting shoelaces with shoes or something. It just comes along.
- `[97]` cos=0.44 ★ (B): Yeah, yeah. Oh. Five minutes to end of meeting. Oh, okay. We're a bit behind.
- `[98]` cos=0.30 (D): Do you know what I mean? Like so sort of like how do you I I mean one one way of looking at it would be, well the people producing television sets, maybe they have to buy remote controls. Or another way is maybe people who have T_V_ sets are really fed up with their remote control and they really want a better one or something.
- `[99]` cos=0.70 (C): Yeah.
- `[100]` cos=0.70 (A): Yeah.
- `[101]` cos=0.51 (C): I know um
- `[102]` cos=0.42 (D): But
- `[103]` cos=0.23 (C): My parents went out and bought um remote controls because um they got fed up of having four or five different remote controls for each things the house. So um for them it was just how many devices control.
- `[104]` cos=0.37 (D): Right. Right. Okay so Right, so in function one of the priorities might be to combine as many uses
- `[105]` cos=0.35 (B): Yeah. Right, so do you think that should be like a main design aim of our remote control d you know, do your your satellite and your regular telly and your V_C_R_ and everything?
- `[106]` cos=0.33 (D): I think so. Yeah, yeah. Yeah. Well like um, maybe what we could use is a sort of like a example of a successful other piece technology is palm palm pilots. They're gone from being just like little sort of scribble boards to cameras, M_P_ three players, telephones,
- `[107]` cos=0.61 (B): Mm-hmm.
- `[108]` cos=0.41 (D): everything, agenda. So, like, I wonder if we might add something new to the to the remote control market, such as the lighting in your house, or um
- `[109]` cos=0.35 (B): Yeah. Or even like, you know, notes about um what you wanna watch. Like you might put in there oh I want to watch such and such and look a Oh that's a good idea. So extra functionalities.
- `[110]` cos=0.28 (D): Yeah, yeah. An Yeah. Like, p personally for me, at home I've I've combined the um the audio video of my television set and my D_V_D_ player and my C_D_ player. So they w all work actually function together but I have different remote controls for each of them. So it's sort of ironic that that then they're in there
- `[111]` cos=0.61 (B): Mm-hmm.
- `[112]` cos=0.30 (D): um you know, the sound and everything it's just one system. But each one's got its own little
- `[113]` cos=0.63 (B): Hmm.
- `[114]` cos=0.54 (D): part.
- `[115]` cos=0.40 (B): Um okay, uh I'd wel we're gonna have to wrap up pretty quickly in the next couple of minutes. Um I'll just check we've nothing else. Okay. Um so anything else anybody wants to add about what they don't like about remote controls they've used, what they would really like to be part of this new one at all?
- `[116]` cos=0.26 (A): And you keep losing them.
- `[117]` cos=0.32 (B): You keep losing them. Okay.
- `[118]` cos=0.56 (D): Mm.
- `[119]` cos=0.18 (A): Finding them is really a pain, you know. I mean it's usually quite small, or when you want it right, it slipped behind the couch or it's kicked under the table.
- `[120]` cos=0.47 (D): Mm. Mm. Mm-hmm. Mm-hmm.
- `[121]` cos=0.31 (B): Yeah. W You get those ones where you can, if you like, whistle or make a really high pitched noise they beep. There I mean is that something we'd want to include, do you think?
- `[122]` cos=0.68 (D): Yeah. Yeah.
- `[123]` cos=0.58 (A): You know.
- `[124]` cos=0.37 (D): That's just really good id Yep. Uh, sure. I remember when the first remote control my my family had was on a cable. Actually had a cable between it and the T_V_ and big like buttons that sort of like, like on a blender or something. And um, you know, when I think about what they are now, it's better, but actually it's still kind of, I dunno, like a massive junky thing on the table. Maybe we could think about how, could be more, you know, streamlined. S
- `[125]` cos=0.33 (B): Dunno. Okay maybe. My goodness. Still feels quite primitive. Maybe like a touch screen or something?
- `[126]` cos=0.35 (D): Something like that, yeah. Or whatever would be technologically reasonable.
- `[127]` cos=0.51 (B): Okay. Uh-huh, okay. Well I guess that's up to our industrial designer.
- `[128]` cos=0.30 (D): 'Cause it could b it could it could be that f it could be that functionally that doesn't make it any better, but that just the appeal of of not having You know, these days there's a r pe things in people's homes are becoming more and more like chic, you know. Um, nicer materials and might be
- `[129]` cos=0.45 (B): It looks better. Yeah. Okay. Okay.
- `[130]` cos=0.31 (D): be worth exploring anyway.
- `[131]` cos=0.62 (C): Uh.
- `[132]` cos=0.39 (B): Right, well um so just to wrap up, the next meeting's gonna be in thirty minutes. So that's about um about ten to twelve by my watch. Um so inbetween now and then, um as the industrial designer, you're gonna be working on you know the actual working design of it so y you know what you're doing there. Um for user interface, technical functions, I guess that's you know like what we've been talking about, what it'll actually do. Um and uh marketing executive,
- `[133]` cos=0.64 ★ (A): Yep.
- `[134]` cos=0.18 (B): you'll be just thinking about what it actually what, you know, what requirements it has to has to fulfil and you'll all get instructions emailed to you, I guess.
- `[135]` cos=0.71 (D): Okay.
- `[136]` cos=0.45 (B): Um. Yeah, so it's th the functional design stage is next, I guess. And uh and that's the end of the meeting. So I got that little message a lot sooner than I thought I would, so
- `[137]` cos=0.40 (D): Um. Before we wrap up, just to make sure we're all on the same page here, um, do we We were given sort of an example of a coffee machine or something, right? Well,
- `[138]` cos=0.63 (B): Mm-hmm. Uh-huh, yeah.
- `[139]` cos=0.61 (A): Mm-hmm.
- `[140]` cos=0.33 (D): um are we at ma right now on the assumption that our television remote control may have features which go beyond the television? Or are we keeping sort of like a a design commitment to television features? I I don't know.
- `[141]` cos=0.49 (B): Th Okay, well just very quickly 'cause this we're supposed to finish now. Um I guess that's up to us, I mean you probably want some kind of unique selling point of it, so um, you know
- `[142]` cos=0.61 (D): Yep. Yeah, sure. Okay.
- `[143]` cos=0.24 (A): I think one factor would be production cost.
- `[144]` cos=0.72 ◀대표 (D): Okay, yeah.
- `[145]` cos=0.24 (A): Because there's a cap there, so um depends on how much you can cram into that price. Um.
- `[146]` cos=0.70 (B): Yeah.
- `[147]` cos=0.71 (D): Okay.
- `[148]` cos=0.61 (B): Mm-hmm.
- `[149]` cos=0.71 (D): Okay.
- `[150]` cos=0.31 (A): I think that that's the main factor.
- `[151]` cos=0.55 (B): Yeah. Okay. Right, okay, we'll that's that's the end of the meeting, then. Um.
- `[152]` cos=0.63 (D): Okay. Alright.
- `[153]` cos=0.36 (B): So, uh thank you all for coming.
- `[154]` cos=0.62 (A): Cool.

