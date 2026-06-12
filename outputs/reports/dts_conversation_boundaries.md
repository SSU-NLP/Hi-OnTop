# DTS 대화 전문 + 정답 화제 경계 (AMI와 convention 비교)

`segmentation_label=1` = 정답 경계(★). `topic_id` 도 같이 표시해 경계가 **화제 도입 발화**에 찍히는지 **그 응답**에 찍히는지 본다. 발화 전문(자르지 않음).

## dialseg711

### dialseg711 dialog #0 — 24 turns, 경계 4개

| # | role | topic | 경계 | 발화 |
|--:|:--|--:|:--:|------|
| 0 | user | 0 |  | check the weather for the 7 day forecast |
| 1 | agent | 0 |  | What city are you interested in? |
| 2 | user | 0 |  | Los Angeles, please. Will it be hot? |
| 3 | agent | 0 | ★ | It will be hot today in Los Angeles. |
| 4 | user | 1 |  | Yes, can you give me the information on the Huntingdon Marriott Hotel? |
| 5 | agent | 1 |  | Absolutely. It is an expensive hotel located in the west part of town. It has 4 starts and includes free wifi and parking. Would you like help booking a room? |
| 6 | user | 1 |  | Yes please, I need a reservation for 6 people for 5 nights starting on Saturday. |
| 7 | agent | 1 |  | Sorry, there are not enough rooms available for that time period. Perhaps a different day or a shorter stay might yield better results. |
| 8 | user | 1 |  | How about for 1 night? If that works, I'll need a reference number of course. |
| 9 | agent | 1 | ★ | Booking was successful. Your reference number is : OO8QDA62. |
| 10 | user | 2 |  | I need a train from London Kings Cross to Cambridge. |
| 11 | agent | 2 |  | I have 70 trains travelling that route. To narrow it down, what day would you like to leave and what time would you like to depart/arrive? |
| 12 | user | 2 |  | sure, I would like to go on Saturday, and arrive by 20:20. As close to that time as I can arrive. |
| 13 | agent | 2 |  | I have a 19:17 from London that arrives at 20:08. Would that work for you? |
| 14 | user | 2 |  | Yes, let me have 7 tickets, please. |
| 15 | agent | 2 | ★ | Booking was successful, the total fee is 132.16 GBP payable at the station. Reference number is : QMD5P3EG. Is there anything else I can help you with? |
| 16 | user | 3 |  | I need to find a shopping center. |
| 17 | agent | 3 |  | The Stanford Shopping Center at 773 Alger Dr is 3 miles away in no traffic. Would you like directions there? |
| 18 | user | 3 |  | Yes please. |
| 19 | agent | 3 | ★ | I sent all the info on the screen, please drive carefully! |
| 20 | user | 4 |  | Schedule meeting. |
| 21 | agent | 4 |  | What details shall I add to your meeting reminder? |
| 22 | user | 4 |  | Please set it for the 13th at 11am. It is going to be with management and to discuss our company picnic. Thank you. |
| 23 | agent | 4 |  | Reminder set for your meeting at 11am on the 13th with management to discuss your company picnic. Is there anything else? |

### dialseg711 dialog #1 — 26 turns, 경계 4개

| # | role | topic | 경계 | 발화 |
|--:|:--|--:|:--:|------|
| 0 | user | 0 |  | Can you help me find a restaurant that serves greek food and is expensive? |
| 1 | agent | 0 |  | I apologize but there are no expensive greek restaurants listed, would you like to try a different food type or price range? |
| 2 | user | 0 |  | Do you have any Indian restaurants? |
| 3 | agent | 0 |  | We have 14 indian restaurants in the expensive category. Do you have any more information to narrow down the search? |
| 4 | user | 0 |  | I do not. Is there one you can recommend? |
| 5 | agent | 0 |  | I can recommend one for you, but first is there a particular part of town you would like to travel too? |
| 6 | user | 0 |  | Any part of town is fine, I just need the address and postcode of one you'd recommend. |
| 7 | agent | 0 | ★ | I would recommend the Maharajah Tandoori Restaurant. It's in the west end. The address is 41518 Castle Street City Centre. The postcode is cb30ah. Shall I make a booking? |
| 8 | user | 1 |  | Car I need to get at my friends house near my current location, please send me their address on the screen |
| 9 | agent | 1 | ★ | Tom lives at 580 Van Ness Ave. |
| 10 | user | 2 |  | looking for a place to dine. The restaurant should serve british food and should be in the expensive price range. |
| 11 | agent | 2 |  | Fitzbillies restaurant at 51 Trumpington Street City Centre fits your request. Would you like me to book a table for you? |
| 12 | user | 2 |  | I would. It's a party of 6 for Monday at 15:45. Can I get the reference number when you're done? |
| 13 | agent | 2 |  | Your reservation has been made. The reference number is GK4F8AFS. Is there anything else I can help you with? |
| 14 | user | 2 |  | No, that's everything. Thanks for your help. |
| 15 | agent | 2 | ★ | welcome next time |
| 16 | user | 3 |  | Im looking for a shopping mall around here. |
| 17 | agent | 3 |  | The Topanga mall is 4 miles away. |
| 18 | user | 3 |  | What is the address? |
| 19 | agent | 3 |  | The address to Topanga mall is 171 Oak Rd. |
| 20 | user | 3 |  | Set GPS for the quickest route to get there. |
| 21 | agent | 3 | ★ | GPS set for quickest route to Topanga Mall. |
| 22 | user | 4 |  | Is there rain on Sunday |
| 23 | agent | 4 |  | What city would you like the forecast for? |
| 24 | user | 4 |  | Compton. |
| 25 | agent | 4 |  | It will not rain on Saturday in Compton. |

## tiage

### tiage dialog #0 — 16 turns, 경계 5개

| # | role | topic | 경계 | 발화 |
|--:|:--|--:|:--:|------|
| 0 | user | 0 |  | hello , how are you doing tonight ? |
| 1 | agent | 0 |  | i am well an loving this interaction how are you ? |
| 2 | user | 0 | ★ | i am great . i just got back from the club . |
| 3 | agent | 1 |  | this is my favorite time of the year season wise |
| 4 | user | 1 |  | i would rather eat chocolate cake during this season . |
| 5 | agent | 1 |  | what club did you go to ? me an timothy watched tv |
| 6 | user | 1 |  | i went to club chino . what show are you watching ? |
| 7 | agent | 1 | ★ | lol oh okay kind of random |
| 8 | user | 2 |  | do you live in a house or apartment ? |
| 9 | agent | 2 |  | we watched a show about animals like him |
| 10 | user | 2 | ★ | i love those shows . i am really craving cake . |
| 11 | agent | 3 |  | why does that matter any ? i went outdoors to play frisbee |
| 12 | user | 3 | ★ | it matters because i have a sweet tooth . |
| 13 | agent | 4 |  | so ? lol i want to meet my family at home in bora |
| 14 | user | 4 |  | my family lives in alaska . it is freezing down there . |
| 15 | user | 4 | ★ | i bet it is oh i could not |

### tiage dialog #1 — 16 turns, 경계 2개

| # | role | topic | 경계 | 발화 |
|--:|:--|--:|:--:|------|
| 0 | user | 0 |  | how are you ? being an old man , i am slowing down these days |
| 1 | agent | 0 |  | hi , my dad is old as well , they live close to me and i see them often |
| 2 | user | 0 |  | that is a great thing honor your dad with your presence |
| 3 | agent | 0 | ★ | sure , i pick him up for church every sunday with my ford pickup |
| 4 | user | 1 |  | sounds wonderful my wheelchair can go very fast on various terrains |
| 5 | agent | 1 |  | i guess that means you do not go hunting often ? i love hunting , i own 3 guns |
| 6 | user | 1 |  | hunting ? i served in the marines , yes i hunt |
| 7 | agent | 1 |  | yeah me too , i am conservative so i love church and hunting |
| 8 | user | 1 |  | what do you like to hunt ? do you ever fish ? |
| 9 | agent | 1 |  | fishing is good . i love fishing as well |
| 10 | user | 1 |  | fishing is a better choice sometimes for my one leg |
| 11 | agent | 1 |  | yes that must be hard , i hope things get better for you |
| 12 | user | 1 |  | i enjoy life , it is what it is these days . |
| 13 | agent | 1 |  | yes i agree . i try to enjoy life too , whenever i am not working |
| 14 | user | 1 |  | well , you better enjoying working so you can enjoy more of your time . |
| 15 | user | 1 | ★ | yeah but i go to church every sunday so my weekends are usually booked |

## superseg

### superseg dialog #0 — 12 turns, 경계 4개

| # | role | topic | 경계 | 발화 |
|--:|:--|--:|:--:|------|
| 0 | user | 0 |  | I need help planning my Social Security benefits for my survivors. |
| 1 | agent | 0 |  | Are you currently planning for your future? |
| 2 | user | 0 |  | Yes, I am. |
| 3 | agent | 0 | ★ | you should think about what your family would need if you die now. Social Security can help your family if you have earned enough credits through your work. |
| 4 | user | 1 |  | Could the benefits received by my family be affected if my widow receives a pension based on work? |
| 5 | agent | 1 | ★ | If it not covered by Social Security, her benefits as a survivor may be affected. |
| 6 | user | 2 |  | and what about my ex-wife. Would the benefits she'd receive affect the amount my widow gets? |
| 7 | agent | 2 | ★ | It won't, the benefits your ex-wife would receive won't affect the benefits the rest of your survivors would receive. |
| 8 | user | 3 |  | And for my parents, what benefits can they receive? |
| 9 | agent | 3 | ★ | They can get the survivor benefits if you provide at least half of your parents' support and they are not eligible for retirement benefits higher than the benefit they would get as survivors from your record. |
| 10 | user | 4 |  | About my ex-wife, would she stop receiving the survivor benefit if she remarries after my death? |
| 11 | agent | 4 |  | She won't if she remarries after turning 60 (50 if disabled). |

### superseg dialog #1 — 14 turns, 경계 2개

| # | role | topic | 경계 | 발화 |
|--:|:--|--:|:--:|------|
| 0 | user | 0 |  | I want to know when I should apply for disability benefits |
| 1 | agent | 0 |  | You should apply for disability benefits as soon as you become disabled |
| 2 | agent | 0 |  | Are you ready to apply for disability benefits? |
| 3 | user | 0 |  | no, not yet but maybe soon. so can I call or walk in somewhere to do this other than online? |
| 4 | agent | 0 |  | yes there are other ways to apply. Will you need any help? are you deaf or hard of hearing? |
| 5 | user | 0 |  | yes,I am deaf |
| 6 | agent | 0 | ★ | when you are ready to apply you can call us at TTY  1-800-325-0778 , or Call or visit your local Social Security office . |
| 7 | user | 1 |  | what are some of the kinds of things you will ask for when I apply? |
| 8 | agent | 1 | ★ | Your Social Security number and proof of your age ; Names , addresses , and phone numbers of doctors , caseworkers , hospitals , and clinics that took care of you and the dates of your visits; Names and dosages of all the medications you are taking |
| 9 | user | 2 |  | and all I do is give that info and I get it for sure or what? |
| 10 | agent | 2 |  | There is a possibility that after we review your application and the information you provided, we may decide you do not meet the qualifications for disability benefits . |
| 11 | agent | 2 |  | would you like to know what to do in case you are denied? |
| 12 | user | 2 |  | yes please |
| 13 | agent | 2 |  | If you don't qualify you may ask us to look at your application again. You will receive a notice from us if you don't qualify  and it will explain how to appeal our decision and the time period in which you must make the request . |

## convention 분석 (전체 test, label=1 위치)
- **tiage**: 경계 turn 의 topic_id 가 *다음 turn 과 같음* 0/315 (0%) → 같으면 '경계 turn = 새 화제의 첫 발화', 다르면 '경계 turn = 옛 화제 마지막(경계 후 다음부터 새 화제)'.
- **dialseg711**: 경계 turn 의 topic_id 가 *다음 turn 과 같음* 0/2754 (0%) → 같으면 '경계 turn = 새 화제의 첫 발화', 다르면 '경계 turn = 옛 화제 마지막(경계 후 다음부터 새 화제)'.
- **superseg**: 경계 turn 의 topic_id 가 *다음 turn 과 같음* 0/4020 (0%) → 같으면 '경계 turn = 새 화제의 첫 발화', 다르면 '경계 turn = 옛 화제 마지막(경계 후 다음부터 새 화제)'.
