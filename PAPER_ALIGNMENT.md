# 논문·구현 대조

기준: 사용자가 제공한 _EMNLP_2026.pdf, 14페이지.

| 항목 | 현재 구현 |
|---|---|
| 정규화된 동결 임베딩 | 기본 multi-qa-mpnet-base-dot-v1; 비영 유한 벡터 검증 |
| 인접 거리 | 1−cos(previous,current) |
| 최근 m개 문맥 | rho^(i−1) 가중합을 정규화 |
| 초기 문맥 Eq.3 | t≤m에는 인접 거리. --warmup legacy는 원본 부분 문맥 |
| 혼합·경계 | a*adjacent+(1−a)*context; 거리/양수 임계값≥1 |
| 보정 | 첫 발화 제외 거리 풀, 의사 라벨에 대한 Score 또는 명시적 F1 선택 |
| 배포 | 미래 발화·학습 모델·LLM 호출 없음 |
| SeCom | 세션별 상태 초기화 및 구간 연결 어댑터 |

m=2에서는 초기 문맥 차이가 없습니다. m>2의 paper 모드 변경은 논문 식을 따르기 위한 명시적 차이입니다. 이전 기본 threshold=0도 허용하던 입력은 graded score가 정의되지 않아 거부합니다. 기본 입력을 정규화하므로 원본과의 수치 비교는 정규화 임베딩 기준입니다.

평가는 SuperDialseg commit 3e8de3d9c87b45904f7e2985f144cae4eef46212의 정의에 맞춘 NLTK/scikit-learn 호출입니다. 대화별 평균, 마지막 라벨 0, window=max(2,round(n/(boundaries+1)/2)), Score=.5F1+.25(1−Pk)+.25(1−WD). 길이 2 미만은 공식 지표가 정의되지 않으므로 거부합니다.

원문: https://github.com/Coldog2333/SuperDialseg/blob/3e8de3d9c87b45904f7e2985f144cae4eef46212/src/super_dialseg/metrics/segmentation.py

MiniLM/INT8 비교 실험, 데이터 다운로드, 기초모델 의사 라벨 생성과 전체 SeCom 실험 하네스는 보존 브랜치에 있습니다. 현재 실행 경로는 코어·보정·평가·SeCom 분절 어댑터입니다. 고정 문맥 벡터 버퍼는 O(md)이지만 history 기록은 발화 수에 따라 증가합니다.
