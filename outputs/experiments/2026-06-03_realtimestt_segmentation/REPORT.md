# Figure U — Hi-OnTop g_t on Single-Topic Audio (2026-06-03)

## 실험 setup
- 입력: `KoljaB/RealtimeSTT` `tests/unit/audio/asr-reference.wav`
  의 VAD-split 전사 결과 (Whisper large-v2, 51.9초, 9 utterances)
- 인코더: MiniLM-int8 (ONNX quint8_avx2, dim=384)
- δ* = 0.771 (MiniLM-int8 p70 cross-benchmark 평균)
  - TIAGE 0.7763 / DS711 0.7519 / SDS 0.7839
- HP: m=2, ρ=0.7, a=0.5

## g_t 값 (turn별)

| turn | utterance (앞 30자) | δ_eff | g_t | boundary? |
|---|---|---:|---:|---|
| 0 | Hey guys.… | 0.0000 | 0.0000 | no |
| 1 | Welcome to the new demo of my real-time … | 0.8984 | 1.1652 | **YES** |
| 2 | As you'll see, speech is transcribed alm… | 0.5310 | 0.6887 | no |
| 3 | I've put a lot of effort into making thi… | 0.8817 | 1.1436 | **YES** |
| 4 | This library is completely open source a… | 0.5872 | 0.7616 | no |
| 5 | Visibility and community feedback are ke… | 0.7848 | 1.0179 | **YES** |
| 6 | Whether you are working on a small proje… | 0.5183 | 0.6723 | no |
| 7 | So feel free to download it, use it in y… | 0.5686 | 0.7375 | no |
| 8 | Thanks for watching and I hope you enjoy… | 0.6718 | 0.8713 | no |

## 관찰
- 9개 utterance 중 boundary 판정: 3개
- g_t 최대값: 1.1652
- g_t mean (turn 1~8): 0.8823

## 한계
- δ* = cross-benchmark 평균값 사용 (이 오디오로 calibration 불가 — 레이블 없음)
- 9 utterance 로 통계적으로 의미있는 결론 내리기 어려움 (관찰용)
