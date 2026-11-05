# Hi-OnTop

EMNLP 2026 논문 「Hi-OnTop: Human-inspired Online Topic Shift Detection for Efficient Dialogue Management」의 코사인 유사도 기반 온라인 대화 분절입니다. SITUATE 학습 모델이나 체크포인트를 가져오지 않습니다.

## 설치 및 실행

Python 3.11, uv를 사용합니다. 각 프로젝트는 자체 가상환경과 uv.lock을 가집니다.

```sh
uv sync --frozen --extra text --extra test
uv run hi-ontop segment examples/input.txt
uv run pytest -q
```

입력 텍스트의 한 줄이 발화 하나입니다. 출력 `boundaries`는 새 구간의 첫 발화 인덱스(0부터), `segments`는 `[start, end_exclusive]`, `scores`는 원거리, `graded_scores`는 거리/임계값입니다. 문장 인코더는 최초 실행 때 다운로드합니다. API 키는 필요 없습니다.

이미 계산한 임베딩으로는 인코더 없이 실행할 수 있습니다.

```sh
uv run hi-ontop segment embeddings.npy --embeddings --delta-star 0.55
```

.npy는 [발화수, 차원]의 유한한 비영 벡터 배열입니다. m=2, rho=0.7, a=0.5가 기본이며 CLI의 --window, --decay, --blend로 지정합니다. 각 입력 파일은 독립 대화입니다. 기본 임계값 0.5594는 원본 코드의 기존 동작점이며 모든 인코더·도메인에 최적이라는 뜻이 아닙니다.

## 임계값 보정과 평가

`calibration.json`은 `[{"embeddings": [[...], ...], "pseudo_labels": [0,1,...,0]}, ...]` 형식입니다. pseudo_labels는 이전 구간의 마지막 발화에 1을 붙이는 끝-turn 규약입니다. 최종 평가셋과 분리된 보정 대화에만 기초모델의 의사 라벨을 준비합니다. 실제 분절 시 기초모델을 호출하지 않습니다.

```sh
uv run hi-ontop calibrate calibration.json --metric score > calibration-result.json
uv run hi-ontop segment examples/input.txt --delta-star <selected.delta_star>
uv run hi-ontop evaluate evaluation.json
```

각 대화의 첫 발화를 제외한 거리 풀에서 5~95 백분위를 탐색하고 대화별 평균 Score로 선택합니다. --metric f1은 논문의 하류 실험 선택 규약용입니다. 평가 JSON은 `{"golds": [[...]], "preds": [[...]]}`입니다. 라벨 출처·보정 데이터 ID는 실험에서 별도 보관해야 합니다. 이 저장소가 LLM 의사 라벨을 자동 생성하지는 않습니다.

## 연결과 범위

Python: `hi_ontop.hi_ontop.HiOnTop.assign(vector)`는 한 발화를 소비하고 `(topic_id, is_boundary)`를 반환합니다. `hi_ontop.secom_adapter.HiOnTopSecomSegmenter`는 SeCom의 세션→분절 문자열 계약을 제공합니다. 검색·압축·답변 LLM은 외부 SeCom 구성입니다.

원본 전체 실험, 이전 SEM 계열과 결과는 보존용 Git 브랜치에 있습니다. 현재 main은 독립 실행용 분리본입니다. 대규모 벤치마크 결과 재현을 이번 검증으로 주장하지 않습니다. 세부 차이는 PAPER_ALIGNMENT.md, 출처는 SOURCE_PROVENANCE.json을 확인하세요.
