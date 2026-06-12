# v4.1.3 graded_score / re-entry 분석 — false-positive bucket + case study

**Setup**:
- v4.1.3 default ({f0_min_starts=1, m=2, ρ=0.7, a=0.5, δ*=0.5594})
- Encoder: `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- Boundary 매핑: history[i+1].is_boundary ↔ y_t[i] (i,i+1 사이 경계)
- Re-entry sub-types:
  - `same_label_restart`: boundary AND topic_id == prev_k (V411 _is_restart 경로)
  - `cross_topic_reentry`: boundary AND topic_id != prev_k AND counts[new]>1 (옛 topic 복귀)
  - `is_reentry` = OR of the two (backward compat)

## Overall — re-entry precision per dataset (sub-type breakdown)

| dataset | n_trans | GT_bnd | pred_bnd | total re | same_label_restart (TP/FP/prec) | cross_topic_reentry (TP/FP/prec) | total re-prec |
|---|---:|---:|---:|---:|---|---|---:|
| tiage | 1464 | 315 | 696 | 603 | 146/327/0.309 (n=473) | 41/89/0.315 (n=130) | **0.310** |
| dialseg711 | 18639 | 2754 | 8367 | 7854 | 2102/4484/0.319 (n=6586) | 280/988/0.221 (n=1268) | **0.303** |

## tiage

### Per-band confusion

| band | n_trans | n_pred_bnd | TP | FP | precision | recall (of band) |
|---|---:|---:|---:|---:|---:|---:|
| very_weak | 296 | 0 | 0 | 0 | 0.000 | 0.000 |
| weak | 737 | 265 | 63 | 202 | 0.238 | 0.463 |
| normal | 406 | 406 | 140 | 266 | 0.345 | 1.000 |
| strong | 25 | 25 | 13 | 12 | 0.520 | 1.000 |

### Re-entry precision (sub-type 분리)

- **Total re-entries**: 603
  - TP: **187** (31.0%)
  - FP: **416** (69.0%)

- **same_label_restart** (V411 _is_restart 경로, topic_id 유지): 473
  - TP: 146 / FP: 327 / **prec 0.309**

- **cross_topic_reentry** (옛 topic 으로 복귀, 진짜 non-linear): 130
  - TP: 41 / FP: 89 / **prec 0.315**

### Case study — TP-rich (schema reinstatement 성공 예시)

### Dialog 64 (14 turns, GT boundaries: 5)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 |   |   |  | 0.543 | very_weak | 0→0 | `hi i am from wisconsin and i do man…` → `hello , i am a web designer .` |
| 1 | **B** ✓ | **B** | 🔁 | 1.111 | normal | 0→0 | `hello , i am a web designer .` → `in wisconsin i enjoy the cold and s…` |
| 2 |   |   |  | 0.883 | weak | 0→0 | `in wisconsin i enjoy the cold and s…` → `no , i do not . i like it hot .` |
| 3 | **B** ✓ | **B** | 🔁 | 1.211 | normal | 0→0 | `no , i do not . i like it hot .` → `what work do you do ?` |
| 4 |   |   |  | 0.780 | weak | 0→0 | `what work do you do ?` → `i am a web designer , i do coding f…` |
| 5 | **B** ✓ | **B** | 🔁 | 1.119 | normal | 0→0 | `i am a web designer , i do coding f…` → `nice . i love the band metallica . …` |
| 6 |   |   |  | 0.885 | weak | 0→0 | `nice . i love the band metallica . …` → `yes , i am . i only listen to count…` |
| 7 | **B** ✓ | **B** | 🔁 | 1.271 | normal | 0→0 | `yes , i am . i only listen to count…` → `i want to move south , but i will m…` |
| 8 |   |   |  | 0.384 | very_weak | 0→0 | `i want to move south , but i will m…` → `i hate snow , i am so glad i made t…` |
| 9 |   |   |  | 0.229 | very_weak | 0→0 | `i hate snow , i am so glad i made t…` → `i want to move south , but i will m…` |
| 10 |   |   |  | 0.731 | weak | 0→0 | `i want to move south , but i will m…` → `where at in the south ? i am in flo…` |
| 11 |   |   |  | 0.474 | very_weak | 0→0 | `where at in the south ? i am in flo…` → `i want to move south , but i will m…` |
| 12 | **B** ✓ | **B** | 🔁 | 1.200 | normal | 0→0 | `i want to move south , but i will m…` → `do you drink iced tea ?` |

### Dialog 29 (16 turns, GT boundaries: 6)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 |   |   |  | 0.735 | weak | 0→0 | `how are you today ? i like you` → `i am great . just relaxing and knit…` |
| 1 |   (miss) | **B** |  | 0.880 | weak | 0→0 | `i am great . just relaxing and knit…` → `where did you use to work ?` |
| 2 |   |   |  | 0.940 | weak | 0→0 | `where did you use to work ?` → `childrens hospital until i became d…` |
| 3 |   |   |  | 0.987 | weak | 0→0 | `childrens hospital until i became d…` → `i rent out houses to people` |
| 4 | **B** ✓ | **B** | 🔁 | 1.103 | normal | 0→0 | `i rent out houses to people` → `that is a great industry . what kin…` |
| 5 | **B** ✗ |   | 🔁 | 1.017 | normal | 0→0 | `that is a great industry . what kin…` → `like people who are handy ?` |
| 6 | **B** ✓ | **B** | 🔁 | 1.019 | normal | 0→0 | `like people who are handy ?` → `yes . i love to cook and i am looki…` |
| 7 | **B** ✓ | **B** | 🔁 | 1.205 | normal | 0→0 | `yes . i love to cook and i am looki…` → `lol , i like classic cars` |
| 8 |   |   |  | 0.594 | very_weak | 0→0 | `lol , i like classic cars` → `me too what is your favorite ? chev…` |
| 9 | **B** ✓ | **B** | 🔁 | 1.232 | normal | 0→0 | `me too what is your favorite ? chev…` → `i get sick on seafood` |
| 10 |   |   |  | 0.768 | weak | 0→0 | `i get sick on seafood` → `i prefer to cook vegan food . what …` |
| 11 |   |   |  | 0.475 | very_weak | 0→0 | `i prefer to cook vegan food . what …` → `i like to eat fried foods` |
| 12 | **B** ✗ |   | 🔁 | 0.941 | weak | 0→0 | `i like to eat fried foods` → `well they are not good for you but …` |
| 13 | **B** ✗ |   | 🔁 | 0.895 | weak | 0→0 | `well they are not good for you but …` → `i do they so good too` |
| 14 | **B** ✓ | **B** | 🔁 | 1.165 | normal | 0→0 | `i do they so good too` → `do you have any plans for the weeke…` |

### Case study — FP-rich (false re-entry, generic opener 등)

### Dialog 49 (16 turns, GT boundaries: 0)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 |   |   |  | 0.829 | weak | 0→0 | `hi there ! just finished shopping !…` → `hello how are you today` |
| 1 | **B** ✗ |   | 🔁 | 1.242 | normal | 0→0 | `hello how are you today` → `i love clothes and models ! cant wa…` |
| 2 | **B** ✗ |   | 🔁 | 1.047 | normal | 0→0 | `i love clothes and models ! cant wa…` → `why are you taking so long to answe…` |
| 3 |   |   |  | 0.929 | weak | 0→0 | `why are you taking so long to answe…` → `why are others ? sorry . i love to …` |
| 4 | **B** ✗ |   | 🔁 | 1.056 | normal | 0→0 | `why are others ? sorry . i love to …` → `i like to get to know new people` |
| 5 |   |   |  | 0.914 | weak | 0→0 | `i like to get to know new people` → `i know plenty in high school i am 1…` |
| 6 | **B** ✗ |   | 🔁 | 1.199 | normal | 0→0 | `i know plenty in high school i am 1…` → `autumn is my favorite time of the y…` |
| 7 |   |   |  | 0.547 | very_weak | 0→0 | `autumn is my favorite time of the y…` → `nice ! this is my senior year in sc…` |
| 8 | **B** ✗ |   | 🔁 | 0.970 | weak | 0→0 | `nice ! this is my senior year in sc…` → `that is great . i like to play ulti…` |
| 9 | **B** ✗ |   | 🔁 | 1.044 | normal | 0→0 | `that is great . i like to play ulti…` → `that took a while . i was going thr…` |
| 10 | **B** ✗ |   | 🔁 | 1.189 | normal | 0→0 | `that took a while . i was going thr…` → `my parents do not live with me and …` |
| 11 | **B** ✗ |   | 🔁 | 1.022 | normal | 0→0 | `my parents do not live with me and …` → `taking you a while . do you play fr…` |
| 12 | **B** ✗ |   | 🔁 | 1.080 | normal | 0→0 | `taking you a while . do you play fr…` → `they moved to bora bora last week` |
| 13 | **B** ✗ |   | 🔁 | 0.941 | weak | 0→0 | `they moved to bora bora last week` → `why did not you go with them ?` |
| 14 | **B** ✗ |   | 🔁 | 1.021 | normal | 0→0 | `why did not you go with them ?` → `i could not leave my pet turtle tim…` |

### Dialog 25 (16 turns, GT boundaries: 3)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 | **B** ✓ | **B** |  | 1.105 | normal | 0→1 | `i really need to make this chat qui…` → `ok lets do it . movies ?` |
| 1 | **B** ✗ |   | 🔁 | 1.177 | normal | 1→0 | `ok lets do it . movies ?` → `i get up early and brightly but no …` |
| 2 | **B** ✓ | **B** |  | 1.098 | normal | 0→2 | `i get up early and brightly but no …` → `i was born in south carolina you ? …` |
| 3 | **B** ✗ |   | 🔁 | 1.177 | normal | 2→0 | `i was born in south carolina you ? …` → `no i m allergic to the fair of cats…` |
| 4 | **B** ✗ |   | 🔁 | 1.104 | normal | 0→1 | `no i m allergic to the fair of cats…` → `my name is joanna and i like horror…` |
| 5 | **B** ✗ |   | 🔁 | 1.189 | normal | 1→0 | `my name is joanna and i like horror…` → `i do crafts for ordering the pham` |
| 6 | **B** ✓ | **B** | 🔁 | 0.909 | weak | 0→1 | `i do crafts for ordering the pham` → `interesting , what do you do for wo…` |
| 7 | **B** ✗ |   | 🔁 | 1.083 | normal | 1→0 | `interesting , what do you do for wo…` → `in a very small town that s where i…` |
| 8 | **B** ✗ |   | 🔁 | 0.969 | weak | 0→1 | `in a very small town that s where i…` → `i can understand that .` |
| 9 |   |   |  | 0.639 | very_weak | 1→1 | `i can understand that .` → `i mean that s where i live how abou…` |
| 10 | **B** ✗ |   | 🔁 | 0.905 | weak | 1→2 | `i mean that s where i live how abou…` → `big city now . helps with my career…` |
| 11 | **B** ✗ |   | 🔁 | 0.922 | weak | 2→1 | `big city now . helps with my career…` → `so what do you see any music you si…` |
| 12 | **B** ✗ |   | 🔁 | 0.921 | weak | 1→2 | `so what do you see any music you si…` → `mostly country . like the way it ma…` |
| 13 | **B** ✗ |   | 🔁 | 1.286 | normal | 2→1 | `mostly country . like the way it ma…` → `you re awesome i hope i can do that` |
| 14 | **B** ✗ |   | 🔁 | 0.881 | weak | 1→0 | `you re awesome i hope i can do that` → `took a very long time for me to lan…` |


## dialseg711

### Per-band confusion

| band | n_trans | n_pred_bnd | TP | FP | precision | recall (of band) |
|---|---:|---:|---:|---:|---:|---:|
| very_weak | 5436 | 29 | 0 | 29 | 0.000 | 0.000 |
| weak | 8001 | 3136 | 404 | 2732 | 0.129 | 0.727 |
| normal | 4737 | 4737 | 1813 | 2924 | 0.383 | 1.000 |
| strong | 465 | 465 | 372 | 93 | 0.800 | 1.000 |

### Re-entry precision (sub-type 분리)

- **Total re-entries**: 7854
  - TP: **2382** (30.3%)
  - FP: **5472** (69.7%)

- **same_label_restart** (V411 _is_restart 경로, topic_id 유지): 6586
  - TP: 2102 / FP: 4484 / **prec 0.319**

- **cross_topic_reentry** (옛 topic 으로 복귀, 진짜 non-linear): 1268
  - TP: 280 / FP: 988 / **prec 0.221**

### Case study — TP-rich (schema reinstatement 성공 예시)

### Dialog 119 (11 turns, GT boundaries: 4)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 |   |   |  | 0.354 | very_weak | 0→0 | `will it hail in oakland` → `Hail is predicted to fall in oaklan…` |
| 1 | **B** ✓ | **B** | 🔁 | 1.404 | strong | 0→0 | `Hail is predicted to fall in oaklan…` → `Please provide the location of the …` |
| 2 |   |   |  | 0.850 | weak | 0→0 | `Please provide the location of the …` → `The nearest police station in Parks…` |
| 3 | **B** ✓ | **B** | 🔁 | 1.374 | strong | 0→0 | `The nearest police station in Parks…` → `Schedule a doctor appointment and m…` |
| 4 |   |   |  | 0.843 | weak | 0→0 | `Schedule a doctor appointment and m…` → `Okay, when would you like to schedu…` |
| 5 |   |   |  | 0.421 | very_weak | 0→0 | `Okay, when would you like to schedu…` → `Schedule it for 4pm today please.` |
| 6 | **B** ✓ | **B** | 🔁 | 0.927 | weak | 0→0 | `Schedule it for 4pm today please.` → `Get me the address to the police st…` |
| 7 | **B** ✗ |   | 🔁 | 0.906 | weak | 0→0 | `Get me the address to the police st…` → `The nearest Police Station is Parks…` |
| 8 | **B** ✓ | **B** | 🔁 | 1.341 | strong | 0→0 | `The nearest Police Station is Parks…` → `Check the next 48 hours and tell me…` |
| 9 |   |   |  | 0.425 | very_weak | 0→0 | `Check the next 48 hours and tell me…` → `It will not be stormy in San Franci…` |

### Dialog 649 (18 turns, GT boundaries: 4)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 |   |   |  | 0.691 | very_weak | 0→0 | `Hello! I'm looking for information …` → `Sure, restaurant one seven serves b…` |
| 1 |   |   |  | 0.652 | very_weak | 0→0 | `Sure, restaurant one seven serves b…` → `Yes please, could you book me a tab…` |
| 2 |   |   |  | 0.709 | weak | 0→0 | `Yes please, could you book me a tab…` → `Unfortunately there is nothing avai…` |
| 3 |   |   |  | 0.750 | weak | 0→0 | `Unfortunately there is nothing avai…` → `Can we try the same day for 13:15?` |
| 4 | **B** ✗ |   | 🔁 | 1.092 | normal | 0→0 | `Can we try the same day for 13:15?` → `Your booking was successful. Your R…` |
| 5 | **B** ✓ | **B** | 🔁 | 0.956 | weak | 0→0 | `Your booking was successful. Your R…` → `Find me a parking garage around me.` |
| 6 |   |   |  | 0.739 | weak | 0→0 | `Find me a parking garage around me.` → `There is a parking garage 5 miles a…` |
| 7 |   |   |  | 0.791 | weak | 0→0 | `There is a parking garage 5 miles a…` → `OK, please give me directions to tr…` |
| 8 |   |   |  | 0.864 | weak | 0→0 | `OK, please give me directions to tr…` → `OK, setting navigation to the Palo …` |
| 9 | **B** ✓ | **B** | 🔁 | 1.202 | normal | 0→0 | `OK, setting navigation to the Palo …` → `I need to book a taxi from Kings Co…` |
| 10 |   |   |  | 0.803 | weak | 0→0 | `I need to book a taxi from Kings Co…` → `The contact number is 07502362913. …` |
| 11 | **B** ✓ | **B** | 🔁 | 1.198 | normal | 0→0 | `The contact number is 07502362913. …` → `what time is my meeting at` |
| 12 |   |   |  | 0.379 | very_weak | 0→0 | `what time is my meeting at` → `meeting at 3pm` |
| 13 | **B** ✓ | **B** | 🔁 | 1.171 | normal | 0→0 | `meeting at 3pm` → `tell me what the temperature will b…` |
| 14 |   |   |  | 0.589 | very_weak | 0→0 | `tell me what the temperature will b…` → `For what city are you interested in…` |
| 15 |   |   |  | 0.885 | weak | 0→0 | `For what city are you interested in…` → `Alameda, please` |
| 16 |   |   |  | 0.651 | very_weak | 0→0 | `Alameda, please` → `The temperature in Alameda will be …` |

### Case study — FP-rich (false re-entry, generic opener 등)

### Dialog 195 (46 turns, GT boundaries: 4)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 |   |   |  | 0.662 | very_weak | 0→0 | `where's the nearest shopping mall` → `The Midtown Shopping Center is loca…` |
| 1 | **B** ✗ |   |  | 1.161 | normal | 0→1 | `The Midtown Shopping Center is loca…` → `Yes sounds great.` |
| 2 | **B** ✗ |   | 🔁 | 1.070 | normal | 1→0 | `Yes sounds great.` → `Its address is 383 University Ave, …` |
| 3 |   |   |  | 0.557 | very_weak | 0→0 | `Its address is 383 University Ave, …` → `OK, please give me directions via t…` |
| 4 |   |   |  | 0.422 | very_weak | 0→0 | `OK, please give me directions via t…` → `The current route is the fastest on…` |
| 5 | **B** ✓ | **B** | 🔁 | 1.129 | normal | 0→0 | `The current route is the fastest on…` → `I'm looking for an expensive restau…` |
| 6 |   |   |  | 0.869 | weak | 0→0 | `I'm looking for an expensive restau…` → `I have many. Could we narrow it dow…` |
| 7 |   |   |  | 0.643 | very_weak | 0→0 | `I have many. Could we narrow it dow…` → `I'd like to try Indian cuisine. I a…` |
| 8 | **B** ✗ |   | 🔁 | 0.989 | weak | 0→0 | `I'd like to try Indian cuisine. I a…` → `I can't seem to find a table that i…` |
| 9 |   |   |  | 0.694 | very_weak | 0→0 | `I can't seem to find a table that i…` → `Yes, let's try for 14:45 instead.` |
| 10 | **B** ✗ |   | 🔁 | 0.977 | weak | 0→0 | `Yes, let's try for 14:45 instead.` → `Booking at Curry Garden was success…` |
| 11 | **B** ✓ | **B** | 🔁 | 0.978 | weak | 0→0 | `Booking at Curry Garden was success…` → `I am looking for a train leaving Lo…` |
| 12 |   |   |  | 0.630 | very_weak | 0→0 | `I am looking for a train leaving Lo…` → `Where are you departing from and wh…` |
| 13 |   |   |  | 0.553 | very_weak | 0→0 | `Where are you departing from and wh…` → `I am leaving from london kings cros…` |
| 14 | **B** ✗ |   | 🔁 | 0.965 | weak | 0→0 | `I am leaving from london kings cros…` → `Sure thing. What will your destinat…` |
| 15 |   |   |  | 0.795 | weak | 0→0 | `Sure thing. What will your destinat…` → `I'm going to Cambridge` |
| 16 | **B** ✗ |   | 🔁 | 0.962 | weak | 0→0 | `I'm going to Cambridge` → `you have plenty of trains leaving f…` |
| 17 | **B** ✗ |   | 🔁 | 0.863 | weak | 0→1 | `you have plenty of trains leaving f…` → `yes please book me something around…` |
| 18 | **B** ✗ |   | 🔁 | 0.767 | weak | 1→0 | `yes please book me something around…` → `booking was done for 165.2 GBP, you…` |
| 19 | **B** ✓ | **B** | 🔁 | 1.111 | normal | 0→0 | `booking was done for 165.2 GBP, you…` → `Where in the east can I find a rest…` |
| 20 |   |   |  | 0.320 | very_weak | 0→0 | `Where in the east can I find a rest…` → `I'm sorry there are no results for …` |
| 21 |   |   |  | 0.856 | weak | 0→0 | `I'm sorry there are no results for …` → `I prefer the east but I am open to …` |
| 22 |   |   |  | 0.756 | weak | 0→0 | `I prefer the east but I am open to …` → `Do you want me to look at the cente…` |
| 23 | **B** ✗ |   | 🔁 | 0.929 | weak | 0→0 | `Do you want me to look at the cente…` → `Yes see if there are any venetian r…` |
| 24 | **B** ✗ |   | 🔁 | 0.928 | weak | 0→0 | `Yes see if there are any venetian r…` → `No there are not. Want to try a dif…` |
| 25 | **B** ✗ |   | 🔁 | 0.988 | weak | 0→0 | `No there are not. Want to try a dif…` → `How about indian food?` |
| 26 |   |   |  | 0.574 | very_weak | 0→0 | `How about indian food?` → `I have 9 restaurants in the cetre t…` |
| 27 | **B** ✗ |   | 🔁 | 0.976 | weak | 0→0 | `I have 9 restaurants in the cetre t…` → `I would like expensive.` |
| 28 | **B** ✗ |   | 🔁 | 0.904 | weak | 0→0 | `I would like expensive.` → `There are five expensive indian pla…` |
| 29 |   |   |  | 0.535 | very_weak | 0→0 | `There are five expensive indian pla…` → `I'm sorry to trouble you but is the…` |
| 30 | **B** ✗ |   | 🔁 | 1.191 | normal | 0→0 | `I'm sorry to trouble you but is the…` → `There are five, would you like me t…` |
| 31 | **B** ✗ |   | 🔁 | 0.938 | weak | 0→0 | `There are five, would you like me t…` → `Yes, book one of the expensive Indi…` |
| 32 | **B** ✗ |   | 🔁 | 1.141 | normal | 0→0 | `Yes, book one of the expensive Indi…` → `Which day would you like to dine?` |
| 33 | **B** ✗ |   | 🔁 | 0.920 | weak | 0→0 | `Which day would you like to dine?` → `I would like to go on Monday at 18:…` |
| 34 | **B** ✗ |   | 🔁 | 1.008 | normal | 0→0 | `I would like to go on Monday at 18:…` → `I would love to help! how many peop…` |
| 35 | **B** ✗ |   | 🔁 | 0.872 | weak | 0→1 | `I would love to help! how many peop…` → `It will be a party of 6.` |
| 36 | **B** ✗ |   | 🔁 | 1.079 | normal | 1→0 | `It will be a party of 6.` → `None of the restaurants were availa…` |
| 37 | **B** ✗ |   | 🔁 | 0.785 | weak | 0→1 | `None of the restaurants were availa…` → `Yes. Please try 17:00.` |
| 38 | **B** ✗ |   | 🔁 | 0.711 | weak | 1→0 | `Yes. Please try 17:00.` → `THANKS VERY MUCH` |
| 39 | **B** ✗ |   | 🔁 | 0.778 | weak | 0→1 | `THANKS VERY MUCH` → `Thank you also. Could I receive the…` |
| 40 |   |   |  | 0.516 | very_weak | 1→1 | `Thank you also. Could I receive the…` → `I've booked you a table at pipasha …` |
| 41 | **B** ✓ | **B** | 🔁 | 0.866 | weak | 1→0 | `I've booked you a table at pipasha …` → `I want to book the el shaddai hotel…` |
| 42 | **B** ✗ |   | 🔁 | 0.978 | weak | 0→1 | `I want to book the el shaddai hotel…` → `No problem. Just give me a date, ho…` |
| 43 | **B** ✗ |   | 🔁 | 0.608 | very_weak | 1→0 | `No problem. Just give me a date, ho…` → `I am looking to book it for 4 night…` |
| 44 |   |   |  | 0.585 | very_weak | 0→0 | `I am looking to book it for 4 night…` → `I have made that booking for you. Y…` |

### Dialog 74 (34 turns, GT boundaries: 4)

| trans | pred | GT | re-entry | graded | band | topic→ | utterance pair |
|---:|:---:|:---:|:---:|---:|---|---|---|
| 0 | **B** ✗ |   |  | 1.165 | normal | 0→1 | `I need to book a train leaving leic…` → `Where would your destination be?` |
| 1 | **B** ✗ |   | 🔁 | 0.989 | weak | 1→0 | `Where would your destination be?` → `I would like to go to Cambridge on …` |
| 2 | **B** ✗ |   | 🔁 | 0.838 | weak | 0→1 | `I would like to go to Cambridge on …` → `There are 11 choices that are avail…` |
| 3 | **B** ✗ |   | 🔁 | 1.029 | normal | 1→0 | `There are 11 choices that are avail…` → `the sooner the better but now rush …` |
| 4 | **B** ✗ |   | 🔁 | 0.968 | weak | 0→1 | `the sooner the better but now rush …` → `First train leaves at 13:09 and the…` |
| 5 | **B** ✗ |   | 🔁 | 0.836 | weak | 1→0 | `First train leaves at 13:09 and the…` → `yep, for 5 people at 13:09` |
| 6 | **B** ✗ |   | 🔁 | 0.959 | weak | 0→1 | `yep, for 5 people at 13:09` → `tickets booked for 151.19 GBP, your…` |
| 7 | **B** ✓ | **B** | 🔁 | 1.177 | normal | 1→0 | `tickets booked for 151.19 GBP, your…` → `I am looking for a place to stay. T…` |
| 8 |   |   |  | 0.480 | very_weak | 0→0 | `I am looking for a place to stay. T…` → `There is one hotel and 6 guesthouse…` |
| 9 | **B** ✗ |   | 🔁 | 1.012 | normal | 0→0 | `There is one hotel and 6 guesthouse…` → `no, I just need it on the east side…` |
| 10 | **B** ✗ |   | 🔁 | 1.000 | normal | 0→1 | `no, I just need it on the east side…` → `how about the leverton house?` |
| 11 | **B** ✗ |   | 🔁 | 0.924 | weak | 1→0 | `how about the leverton house?` → `Sure, I'd like to book it for 2 peo…` |
| 12 | **B** ✗ |   | 🔁 | 0.836 | weak | 0→1 | `Sure, I'd like to book it for 2 peo…` → `Sorry, but the hotel isn't availabl…` |
| 13 |   |   |  | 0.586 | very_weak | 1→1 | `Sorry, but the hotel isn't availabl…` → `how about 2 nights instead?` |
| 14 | **B** ✗ |   | 🔁 | 0.965 | weak | 1→1 | `how about 2 nights instead?` → `Booking was successful.Reference nu…` |
| 15 | **B** ✗ |   | 🔁 | 0.964 | weak | 1→0 | `Booking was successful.Reference nu…` → `No, that's all I need. Thank you ve…` |
| 16 | **B** ✗ |   | 🔁 | 0.864 | weak | 0→1 | `No, that's all I need. Thank you ve…` → `You are welcome. I can help with at…` |
| 17 | **B** ✓ | **B** | 🔁 | 0.882 | weak | 1→0 | `You are welcome. I can help with at…` → `Please give me an address and direc…` |
| 18 |   |   |  | 0.843 | weak | 0→0 | `Please give me an address and direc…` → `Coupa is 7 miles away from here in …` |
| 19 | **B** ✓ | **B** | 🔁 | 0.994 | weak | 0→0 | `Coupa is 7 miles away from here in …` → `I need a train from london liverpoo…` |
| 20 | **B** ✗ |   | 🔁 | 1.038 | normal | 0→1 | `I need a train from london liverpoo…` → `What is your destination, please?` |
| 21 | **B** ✗ |   | 🔁 | 0.772 | weak | 1→0 | `What is your destination, please?` → `I am heading to Cambridge for an im…` |
| 22 | **B** ✗ |   | 🔁 | 0.770 | weak | 0→1 | `I am heading to Cambridge for an im…` → `There is one train leaving at 7:39 …` |
| 23 | **B** ✗ |   | 🔁 | 0.796 | weak | 1→0 | `There is one train leaving at 7:39 …` → `That should be perfect. How many ti…` |
| 24 |   |   |  | 0.445 | very_weak | 0→0 | `That should be perfect. How many ti…` → `You can buy up to eight tickets.` |
| 25 |   |   |  | 0.885 | weak | 0→0 | `You can buy up to eight tickets.` → `Just one is fine.` |
| 26 | **B** ✗ |   | 🔁 | 1.066 | normal | 0→0 | `Just one is fine.` → `Ok. The price is 16.60 pounds for 1…` |
| 27 | **B** ✗ |   | 🔁 | 0.827 | weak | 0→1 | `Ok. The price is 16.60 pounds for 1…` → `Could you tell me the travel time p…` |
| 28 | **B** ✗ |   | 🔁 | 0.832 | weak | 1→0 | `Could you tell me the travel time p…` → `Yes, it's 88 minutes. May I help wi…` |
| 29 | **B** ✓ | **B** | 🔁 | 1.557 | strong | 0→0 | `Yes, it's 88 minutes. May I help wi…` → `what is the weather in new york cit…` |
| 30 |   |   |  | 0.519 | very_weak | 0→0 | `what is the weather in new york cit…` → `What can I tell you about the weath…` |
| 31 |   |   |  | 0.809 | weak | 0→0 | `What can I tell you about the weath…` → `I'd like to know the weather overca…` |
| 32 |   |   |  | 0.338 | very_weak | 0→0 | `I'd like to know the weather overca…` → `It will be overcast on Sunday for t…` |


## 해석

(자동 채울 것)
