# AMI ES2002a — 발화별 임베딩-정보량(info_emb) + 전문 (turn 0–104)

**info_emb = 1 − cos(발화, 미팅 전체 평균 임베딩)** — 평균에 가까우면 낮음(generic=filler), 멀면 높음(독특=내용). 사전 0개.
이 미팅 임계: **p30=0.416** (이하=🔵FILLER, forward-merge 대상) · **p60=0.658** (이상=🟢내용, 경계탐지 대상).
info_emb 범위: min 0.310 / median 0.599 / max 0.910. `★`=정답경계.

| # | (화자) | **info_emb** | tier | 발화 |
|--:|:--|--:|:--|------|
| 0 | (B) | **0.565** | ⚪약함 | Okay. Right. Um well this is the kick-off meeting for our our project. Um and um this is just what we're gonna be doing over the next twenty five minutes. Um so first of all, just to kind of make sure that we all know each other, I'm Laura and I'm the project manager. Do you want to introduce yourself again? |
| 1 | (D) | **0.388** | 🔵FILLER | Mm-hmm. Great. |
| 2 | (A) | **0.658** | ⚪약함 | Hi, I'm David and I'm supposed to be an industrial designer. |
| 3 | (B) | **0.338** | 🔵FILLER | Okay. |
| 4 | (D) | **0.619** | ⚪약함 | And I'm Andrew and I'm uh our marketing |
| 5 | (C) | **0.571** | ⚪약함 | Um I'm Craig and I'm User Interface. |
| 6 | (D) | **0.566** | ⚪약함 | expert. |
| 7 | (B) | **0.549** | ⚪약함 | Great. Okay. Um so we're designing a new remote control and um Oh I have to record who's here actually. So that's David, Andrew and Craig, isn't it? And you all arrived on time. Um yeah so des uh design a new remote control. Um, as you can see it's supposed to be original, trendy and user friendly. Um so that's kind of our our brief, as it were. Um and so there are three different stages to the design. Um I'm not really sure what what you guys have already received um in your emails. What did you get? |
| 8 ★ | (A) | **0.674** | 🟢내용 | Um, I just got the project announcement about what the project is. Designing a remote control. That's about it, didn't get anything else. Did you get the same thing? |
| 9 | (B) | **0.363** | 🔵FILLER | Mm-hmm. |
| 10 | (D) | **0.401** | 🔵FILLER | Mm-hmm. Mm-hmm. Yeah, that's that's it. |
| 11 | (B) | **0.649** | ⚪약함 | Is that what everybody got? Okay. Um. So we're gonna have like individual work and then a meeting about it. And repeat that process three times. Um and at this point we get try out the whiteboard over there. Um. So uh you get to draw your favourite animal and sum up your favourite characteristics of it. So who would like to go first? |
| 12 | (C) | **0.311** | 🔵FILLER | Yeah. |
| 13 | (D) | **0.680** | 🟢내용 | Yeah. I will go. That's fine. |
| 14 ★ | (B) | **0.594** | ⚪약함 | Very good. |
| 15 | (D) | **0.489** | ⚪약함 | Alright. So This one here, right? |
| 16 | (B) | **0.363** | 🔵FILLER | Mm-hmm. |
| 17 | (D) | **0.719** | 🟢내용 | Okay. Very nice. Alright. My favourite animal is like A beagle. Um charac favourite characteristics of it? Is that right? Uh, right, well basically um high priority for any animal for me is that they be willing to take a lot of physical affection from their family. And, yeah that they have lots of personality and uh be fit and in robust good health. So this is blue. Blue beagle. My family's beagle. |
| 18 | (B) | **0.498** | ⚪약함 | Yeah. Yeah. Right. Lovely. |
| 19 | (C) | **0.809** | 🟢내용 | Well, my favourite animal would be a monkey. Then they're small cute and furry, and uh when planet of the apes becomes real, I'm gonna be up there with them. |
| 20 | (B) | **0.386** | 🔵FILLER | Right. |
| 21 | (A) | **0.608** | ⚪약함 | Cool. There's too much gear. |
| 22 | (B) | **0.718** | 🟢내용 | You can take as long over this as you like, because we haven't got an awful lot to discuss. Ok oh we do we do. Don't feel like you're in a rush, anyway. |
| 23 | (A) | **0.338** | 🔵FILLER | Okay. |
| 24 | (D) | **0.743** | 🟢내용 | I coulda told you a whole lot more about beagles. Boy, let me tell you. |
| 25 | (B) | **0.644** | ⚪약함 | Ach why not We might have to get you up again then. I don't know what mine is. I'm gonna have to think on the spot now. |
| 26 | (D) | **0.762** | 🟢내용 | Impressionist. |
| 27 | (A) | **0.767** | 🟢내용 | Can't draw. |
| 28 | (B) | **0.772** | 🟢내용 | Is that a whale? |
| 29 | (A) | **0.747** | 🟢내용 | Um. Yeah. Um, well anyway, I don't know, it's just the first animal I can think off the top of my head. Um. Yes. Big reason is 'cause I'm allergic to most animals. Allergic to animal fur, so um fish was a natural choice. |
| 30 | (B) | **0.431** | ⚪약함 | Ah. |
| 31 | (A) | **0.718** | 🟢내용 | Um, yeah, and I kind of like whales. They come in and go eat everything in sight. |
| 32 | (D) | **0.352** | 🔵FILLER | Alright. |
| 33 | (A) | **0.799** | 🟢내용 | And they're quite harmless and mild and interesting. |
| 34 | (D) | **0.409** | 🔵FILLER | Mm. |
| 35 | (B) | **0.546** | ⚪약함 | Okay. God, I still don't know what I'm gonna write about. Um. |
| 36 | (D) | **0.804** | 🟢내용 | Superb sketch, by the way. |
| 37 | (A) | **0.723** | 🟢내용 | Tail's a bit big, I think. |
| 38 | (B) | **0.723** | 🟢내용 | I was gonna choose a dog as well. But I'll just draw a different kind of dog. M my favourite animal is my own dog at home. Um |
| 39 | (D) | **0.352** | 🔵FILLER | Yep. |
| 40 | (B) | **0.768** | 🟢내용 | That doesn't really look like him, actually. He looks more like a pig, actually. Ah well. |
| 41 | (D) | **0.715** | 🟢내용 | I see a dog in there. Yep. |
| 42 | (B) | **0.675** | 🟢내용 | Do you? Oh that's very good of you. |
| 43 | (D) | **0.720** | 🟢내용 | Now I see a rooster. |
| 44 | (B) | **0.385** | 🔵FILLER | Uh. |
| 45 | (D) | **0.882** | 🟢내용 | What kind is it? |
| 46 | (B) | **0.753** | 🟢내용 | Um he's a mixture of uh various things. Um and what do I like about him, um That's just to suggest that his tail wags. Um he's very friendly and cheery and always pleased to see you, and very kind of affectionate and um uh and he's quite quite wee as well so you know he can doesn't take up too much space. Um and uh And he does a funny thing where he chases his tail as well, which is quite amusing, so |
| 47 | (D) | **0.756** | 🟢내용 | Is he aware that th it's his own cha tail he's chasing? |
| 48 | (B) | **0.755** | 🟢내용 | It is. I think it is. He only does it after he's had his dinner and um he'll just all of a sudden just get up and start chasing his tail 'round the living room. |
| 49 | (D) | **0.339** | 🔵FILLER | Hmm. |
| 50 | (A) | **0.681** | 🟢내용 | It's an after dinner dog then. |
| 51 | (B) | **0.425** | ⚪약함 | Yeah, so uh |
| 52 | (D) | **0.910** | 🟢내용 | Probably when he was little he got lots of attention for doing it and has forever been conditioned. |
| 53 | (B) | **0.608** | ⚪약함 | Yeah, maybe. Maybe. Right, um where did you find this? Just down here? Yeah. Okay. Um what are we doing next? Uh um. Okay, uh we now need to discuss the project finance. Um so according to the brief um we're gonna be selling this remote control for twenty five Euro, um and we're aiming to make fifty million Euro. Um so we're gonna be selling this on an international scale. And uh we don't want it to cost any more than uh twelve fifty Euros, so fifty percent of the selling price. |
| 54 ★ | (D) | **0.616** | ⚪약함 | 'Kay. Um, can we just go over that again? |
| 55 | (B) | **0.463** | ⚪약함 | Sure. |
| 56 | (D) | **0.742** | 🟢내용 | Uh, so bas at twel Alright, yeah. Okay. So cost like production cost is twelve fifty, but selling price is is that wholesale or retail? Like on the shelf. |
| 57 | (B) | **0.589** | ⚪약함 | All together. Um I dunno. I imagine That's a good question. |
| 58 | (D) | **0.583** | ⚪약함 | Our sale our sale anyway. Yeah, okay okay. |
| 59 | (B) | **0.737** | 🟢내용 | I imagine it probably is our sale actually because it's probably up to the the um the retailer to uh sell it for whatever price they want. Um. |
| 60 | (D) | **0.353** | 🔵FILLER | Okay. Mm-hmm. Alright. |
| 61 | (B) | **0.742** | 🟢내용 | But I I don't know, I mean do you think the fact that it's going to be sold internationally will have a bearing on how we design it at all? |
| 62 ★ | (D) | **0.380** | 🔵FILLER | Yes. |
| 63 | (B) | **0.621** | ⚪약함 | Think it will? Um. |
| 64 | (D) | **0.415** | 🔵FILLER | Mm-hmm. Mm-hmm. |
| 65 | (B) | **0.339** | 🔵FILLER | Hmm. |
| 66 | (D) | **0.756** | 🟢내용 | Well right away I'm wondering if there's um th th uh, like with D_V_D_ players, if there are zones. Um f frequencies or something um as well as uh characters, um different uh keypad styles and s symbols. |
| 67 | (B) | **0.616** | ⚪약함 | Oh yeah, regions and stuff, yeah. Yeah. Okay. Yeah. Well for a remote control, do you think that will be I suppose it's depends on how complicated our remote control is. |
| 68 | (A) | **0.339** | 🔵FILLER | Hmm. |
| 69 | (D) | **0.474** | ⚪약함 | Um. I don't know. Yeah. |
| 70 | (A) | **0.740** | 🟢내용 | It does make sense from maybe the design point of view 'cause you have more complicated characters like European languages, then you need more buttons. |
| 71 | (B) | **0.347** | 🔵FILLER | Yeah, yeah. Okay. |
| 72 | (D) | **0.311** | 🔵FILLER | Yeah. |
| 73 | (A) | **0.456** | ⚪약함 | So, possibly. |
| 74 | (D) | **0.795** | 🟢내용 | Yeah. And then a and then al the other thing international is on top of the price. I'm thinking the price might might appeal to a certain market in one region, whereas in another it'll be different, so |
| 75 | (B) | **0.813** | 🟢내용 | What, just like in terms of like the wealth of the country? Like how much money people have to spend on things like? |
| 76 | (D) | **0.688** | 🟢내용 | Just a chara just a characteristic of the Just Or just like, basic product podi positioning, the twenty five Euro remote control might be a big hit in London, might not be such a big hit in Greece, who knows, something like that, yeah. |
| 77 | (B) | **0.567** | ⚪약함 | Aye, I see what you mean, yeah. Marketing. Good marketing thoughts. Oh gosh, I should be writing all this down. Um. |
| 78 | (D) | **0.720** | 🟢내용 | Yep. Right away I'm making some kind of assumptions about what what information we're given here, thinking, 'kay trendy probably means something other than just basic, something other than just standard. Um |
| 79 | (B) | **0.310** | 🔵FILLER | Mm. Yeah. |
| 80 | (D) | **0.743** | 🟢내용 | so I'm wondering right away, is selling twenty five Euros, is that sort of the thi is this gonna to be like the premium product kinda thing or |
| 81 | (B) | **0.599** | ⚪약함 | Yeah, yeah. Like how much does, you know, a remote control cost. |
| 82 | (D) | **0.329** | 🔵FILLER | Uh-huh. |
| 83 | (B) | **0.759** | 🟢내용 | Well twenty five Euro, I mean that's um that's about like eighteen pounds or something, isn't it? Or no, is it as much as that? Sixteen seventeen eighteen pounds. |
| 84 | (D) | **0.352** | 🔵FILLER | Mm-hmm. Yep. Yeah, I'd say so, yeah. |
| 85 | (B) | **0.815** | 🟢내용 | Um, I dunno, I've never bought a remote control, so I don't know how how good a remote control that would get you. Um. |
| 86 | (D) | **0.435** | ⚪약함 | No. Yeah, yeah. |
| 87 | (B) | **0.673** | 🟢내용 | But yeah, I suppose it has to look kind of cool and gimmicky. |
| 88 | (D) | **0.363** | 🔵FILLER | Mm-hmm. |
| 89 | (B) | **0.624** | ⚪약함 | Um right, okay. Let me just scoot on ahead here. Okay. Um well d Does anybody have anything to add to uh to the finance issue at all? |
| 90 | (D) | **0.761** | 🟢내용 | Do we have any other background information on like how that compares to other |
| 91 | (B) | **0.674** | 🟢내용 | Thin No, actually. That would be useful, though, wouldn't it, if you knew like what your money would get you now. |
| 92 | (D) | **0.468** | ⚪약함 | other Yeah. |
| 93 | (A) | **0.339** | 🔵FILLER | Hmm. |
| 94 | (D) | **0.363** | 🔵FILLER | Mm-hmm. |
| 95 | (B) | **0.363** | 🔵FILLER | Mm-hmm. |
| 96 | (D) | **0.689** | 🟢내용 | Yeah, interesting thing about discussing um production of a remote control for me is that l as you point out, I just don't think of remote controls as somethin something people consciously assess in their purchasing habits. It's just like getting shoelaces with shoes or something. It just comes along. |
| 97 ★ | (B) | **0.563** | ⚪약함 | Yeah, yeah. Oh. Five minutes to end of meeting. Oh, okay. We're a bit behind. |
| 98 | (D) | **0.717** | 🟢내용 | Do you know what I mean? Like so sort of like how do you I I mean one one way of looking at it would be, well the people producing television sets, maybe they have to buy remote controls. Or another way is maybe people who have T_V_ sets are really fed up with their remote control and they really want a better one or something. |
| 99 | (C) | **0.311** | 🔵FILLER | Yeah. |
| 100 | (A) | **0.311** | 🔵FILLER | Yeah. |
| 101 | (C) | **0.475** | ⚪약함 | I know um |
| 102 | (D) | **0.592** | ⚪약함 | But |
| 103 | (C) | **0.758** | 🟢내용 | My parents went out and bought um remote controls because um they got fed up of having four or five different remote controls for each things the house. So um for them it was just how many devices control. |
| 104 | (D) | **0.666** | 🟢내용 | Right. Right. Okay so Right, so in function one of the priorities might be to combine as many uses |

## 정답 경계 발화의 info_emb (★가 어느 tier?)

- turn 8 **0.674** 🟢내용 — "Um, I just got the project announcement about what" → project goals and design process
- turn 14 **0.594** ⚪약함 — "Very good." → drawing animals on the whiteboard
- turn 54 **0.616** ⚪약함 — "'Kay. Um, can we just go over that again?" → project budget
- turn 62 **0.380** 🔵FILLER — "Yes." → possible issues with project goals
- turn 97 **0.563** ⚪약함 — "Yeah, yeah. Oh. Five minutes to end of meeting. Oh" → initial ideas about RC design

→ 경계가 🔵FILLER tier(낮은 info)에 걸리면 = forward-merge로 *뒤* 내용에 흡수되어 그 그룹 시작으로 보존. 🟢내용 tier면 그 자체가 경계 후보.
