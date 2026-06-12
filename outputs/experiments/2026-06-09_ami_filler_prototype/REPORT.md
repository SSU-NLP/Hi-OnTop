# AMI 화제분절 — filler-prototype + forward-merge + info-gate (139미팅)

## 방법 (전부 online · 미래 안 봄)
1. **filler-prototype ref** = 보편 filler 단어(yeah/okay/mm-hmm/...) 인코딩 평균 (1회 고정, AMI 무관).
2. **info_emb** = 1 − cos(발화, ref). 낮음=generic(filler), 높음=내용.
3. **forward-merge**: info_emb<τ_f 발화를 *다음* 내용에 흡수 → topic-opener filler 보존.
4. **boundary**: 그룹 δ_eff + ewma 적응임계치, 그룹 info≥τ_c 만 경계(info-gate).
   τ_f=0.486, τ_c=0.770 (전역 고정 calibration).

## 결과 (139미팅)
- 미팅 139, 발화 61381 → 그룹 43016, 정답경계 935, 예측 2605.
- **filler 판별 AUC = 0.977** (보편 filler-prototype, AMI-fit 아님).

| metric | 값 |
|---|--:|
| boundary-F1 (exact ±0) | 0.028 |
| boundary-F1 (±1) | 0.060 |
| boundary-F1 (±2) | 0.107 |
| boundary-F1 (±3) | 0.149 |
| Pk ↓ | 0.579 |
| WD ↓ | 0.701 |

## 판정 (정직)
- **139 전체에서 이 method 는 baseline 보다 낮다**: filler-prototype ±2≈0.107 vs raw 0.151 vs geometry-merge 0.189. 12미팅(±2~0.158)은 쉬운 subset 의 과대평가였음. per-meeting 임계로 바꿔도 동일(0.107) → 임계가 아니라 method/데이터 문제.
- **유효한 부분**: filler-prototype 정보량은 **universal·online·AMI-fit아님** 으로 filler 를 AUC 0.977 로 판별. 즉 *filler 탐지기* 로는 검증됨.
- **무효한 부분**: 그 위에 쌓은 forward-merge + info-gate 분절은 39k→43k 그룹화·게이트가 오히려 경계 정렬을 흐려 geometry-merge 보다 못함.
- **결론**: AMI 메인 분절은 **geometry-merge(±2 0.189) 채택**. filler-prototype 은 탐지 component 로만 가치. 그리고 어떤 method 든 AMI 천장(±2 ~0.2)은 drift + annotation(filler 위 경계) 한계로 못 뚫음 — 모든 시도가 이를 재확인.
