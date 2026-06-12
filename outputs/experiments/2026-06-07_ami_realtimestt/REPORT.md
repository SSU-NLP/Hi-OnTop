# AMI × RealtimeSTT 실시간 스트리밍 벤치마크 (2026-06-07_ami_realtimestt)

## 실험 setup
- **목적**: Hi-OnTop 이 전제하는 실시간 스트리밍 STT 파이프라인
  (RealtimeSTT + faster-whisper) 의 지연/비용/정확도 지표를 AMI 회의 음성으로 검증.
- **데이터**: AMI Meeting Corpus IHM (edinburghcstr/ami, test split).
  발화 단위 parquet 을 begin_time 타임라인에 합성해 연속 단일 스트림 WAV 로 재구성 (`scripts/ami_prep.py`).
  미팅 1개, 각 앞 전체분 (16kHz mono).
- **STT**: faster-whisper `tiny`, device=cpu, compute_type=int8, beam_size=5.
- **스트리밍 시뮬레이션**: WAV 를 32ms 청크로 **실시간 페이스**(마이크 모방) `feed_audio()`. final + realtime 모델을 **in-process executor** 로 주입해 모든 모델 호출의 추론 시간을 정밀 계측.
- **환경 주의**: GPU 미가용(CPU only) → Peak VRAM 대신 process RSS 보고. GPU 에서는 RTF/지연 모두 크게 개선됨 (본 수치는 CPU 보수 상한).

## 결과 (per meeting)

| meeting | dur(s) | WER | RTF_final | RTF_rt | RTF_wall | first_rt(s) | final_lat(s) | sched p50/p95(ms) | arm→spch(ms) | flicker | calls(f/rt) | RSS(MB) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| EN2002a | 601 | 0.414 | 0.222 | 0.966 | 1.061 | 2.413 | 8.537 | 1340/12482 | 1000 | 0.240 | 40/428 | 1405 |

### 평균

- WER: **0.414**
- RTF_final: **0.222** (< 1.0 = 확정 전사 실시간 충족)
- RTF_realtime: 0.966 (라이브 프리뷰 재연산 비용/audio)
- First realtime latency (발화 onset 기준): 2.413 s
- Final latency (transcription gap): 8.537 s
- Scheduler p50/p95: 1340 / 12482 ms
- Arm→speech gap: 1000 ms (VAD 처리지연 아님 — 도입부 침묵 반영)
- Flicker rate: 0.240
- Peak RSS: 1405 MB

## 지표 정의
- **WER**: jiwer, 소문자·구두점 제거 정규화 후 reference(해당 cap 구간 발화) vs 최종 전사.
- **RTF_final** = Σ(최종 모델 추론시간)/오디오길이. 스트리밍 keep-up 핵심 지표.
- **RTF_realtime** = Σ(realtime 모델 추론시간)/오디오길이. 늘어나는 버퍼 반복 재연산.
- **RTF_wall** = 전체 wall/오디오길이 (실시간 페이스라 ~1.0 부근, 백로그 시 상승).
- **First realtime latency**: 발화 onset(첫 on_recording_start) → 그 직후 첫 realtime partial.
- **Final latency**: feed 완료 → 마지막 final 확정 (= final transcription gap; 음수=백로그 0).
- **Scheduler p50/p95**: 최종 모델 호출당 추론시간 분포.
- **Arm→speech gap**: on_vad_detect_start("listening" armed) → on_recording_start. ※ 진짜 VAD 처리지연 아님 — 도입부 침묵 길이 반영.
- **Flicker rate**: realtime partial 텍스트 변경 횟수 / 최종 단어 수.
- **Model calls (f/rt)**: 최종/realtime 모델 호출 수.

## 한계 / 검증 미해결
- **CPU only**: GPU 부재로 RTF·지연은 보수적 상한. Peak VRAM 미측정(RSS 대용).
- `tiny` 모델: 회의 음성(중첩·잡음)에서 WER 높음 — STT 품질은 본 연구 범위 밖(상위 모델 교체로 개선 가능), Hi-OnTop 검증 목적은 지연/비용 축.
- IHM 채널 합성 재구성: 실제 단일 원거리 마이크(SDM)와 음향 특성 다름.
- **Concurrency(동시성)**: 본 스크립트는 단일 스트림. WebSocket 서버 기반 N-동시 클라이언트 부하 시험은 별도 실험으로 분리 (RealtimeSTT_server 필요).
