#!/usr/bin/env python3
"""AMI × RealtimeSTT 실시간 스트리밍 벤치마크.

이미 녹음된 AMI 미팅 WAV 를 **마이크처럼 실시간 페이스로 feed_audio()** 하여
RealtimeSTT (faster-whisper backend) 의 스트리밍 전사 파이프라인을 그대로
구동하고, 논문용 지표를 한 번에 측정한다.

측정 지표
---------
1. WER                      jiwer (reference vs 최종 전사, 정규화 후)
2. RTF_final                Σ(최종 모델 추론시간) / 오디오 길이   < 1.0 = 실시간 충족
   RTF_realtime             Σ(realtime 모델 추론시간) / 오디오 길이 (라이브 프리뷰 재연산)
   RTF_wall                 전체 wall time / 오디오 길이
3. First realtime latency   **발화 onset(첫 on_recording_start) → 그 직후 첫 realtime partial**
4. Final latency            feed 완료 → 마지막 final 확정 (transcription gap; 음수=백로그 0)
5. Scheduler p50/p95        최종 모델 호출당 추론시간 분포
6. Flicker rate             realtime partial 변경 횟수 / 최종 단어 수
7. Arm→speech gap           on_vad_detect_start("listening" armed) → on_recording_start.
                            ※ 진짜 VAD 처리지연 아님(도입부 침묵 길이 반영). 정직 보고용.
8. Peak memory              process RSS peak (GPU 없으면 VRAM 대용)
9. Model-call counts        final / realtime 모델 호출 수 (분리)

Concurrency 는 별도 (서버/WebSocket) — 본 스크립트 범위 밖, REPORT 에 명시.

사용
----
  uv run python scripts/ami_realtimestt_benchmark.py \
      --meetings EN2002a EN2002b EN2002c \
      --model tiny --device cpu --compute-type int8 \
      --cap-sec 0        # 0 = manifest 전체 길이
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import jiwer
import numpy as np
import psutil
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
# vendored RealtimeSTT clone (feed_audio + lazy pyaudio). env 로 override 가능.
RTSTT_ROOT = os.environ.get("RTSTT_ROOT", str(REPO / "benchmarks" / "RealtimeSTT"))
sys.path.insert(0, RTSTT_ROOT)
DATA = REPO / "data" / "ami"
SR = 16000

# ── 모델 호출 계측 — engine 인스턴스별 .transcribe 래핑 ─────────────────────
# final(확정 전사) / realtime(라이브 미리보기) 모델을 분리 계측한다. realtime 은
# 늘어나는 버퍼를 반복 재연산하므로 RTF 합산 시 별도 보고가 정확.
_LOCK = threading.Lock()


def _wrap_engine_transcribe(engine, bucket: list[float]):
    orig = engine.transcribe

    def wrapped(audio, *a, **kw):
        t0 = time.perf_counter()
        res = orig(audio, *a, **kw)   # engine.transcribe 가 segments 소비 완료
        with _LOCK:
            bucket.append(time.perf_counter() - t0)
        return res

    engine.transcribe = wrapped


def _trust_torch_hub() -> None:
    """torch.hub.load 의 trust 프롬프트(input())를 비대화형에서 우회."""
    import torch

    _orig = torch.hub.load

    def _patched(*a, **kw):
        kw.setdefault("trust_repo", True)
        return _orig(*a, **kw)

    torch.hub.load = _patched


# ── 텍스트 정규화 (WER 용, AMI/Whisper 관행) ────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _feed_chunks(recorder, samples, chunk, pace):
    for s in range(0, samples.size, chunk):
        c = samples[s:s + chunk]
        recorder.feed_audio(c, original_sample_rate=SR)
        if pace:
            time.sleep((c.size / SR))


def _make_inproc_engines(args):
    """final + realtime 모델을 메인 프로세스에서 실행하는 in-process executor.

    이렇게 하면 faster_whisper monkeypatch 가 모든 모델 호출(추론 시간 포함)을
    캡처 가능 (default child-process 워커는 parent monkeypatch 미적용)."""
    from RealtimeSTT.transcription_engines.base import TranscriptionEngineConfig
    from RealtimeSTT.transcription_engines.factory import create_transcription_engine

    cfg = TranscriptionEngineConfig(
        model=args.model, device=args.device, compute_type=args.compute_type,
        beam_size=args.beam_size, suppress_tokens=[-1], vad_filter=False,
    )
    final_engine = create_transcription_engine("faster_whisper", cfg)
    rt_engine = create_transcription_engine("faster_whisper", cfg)
    return final_engine, rt_engine


def run_meeting(mid: str, args) -> dict:
    from RealtimeSTT import AudioToTextRecorder

    wav_p = DATA / "audio" / f"{mid}.wav"

    audio, srate = sf.read(wav_p, dtype="float32")
    assert srate == SR, f"{mid}: sr={srate}"
    if args.cap_sec > 0:
        audio = audio[: int(args.cap_sec * SR)]
    audio_dur = len(audio) / SR
    samples = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    # reference 를 오디오 cap 구간에 맞춰 자름 (utterance start < audio_dur)
    utt_path = DATA / "transcripts" / f"{mid}.jsonl"
    ref_utts = [json.loads(l) for l in utt_path.read_text().splitlines() if l.strip()]
    ref = " ".join(u["text"] for u in ref_utts
                   if u["text"] and u["start"] < audio_dur).strip()

    # ── 이벤트 타임스탬프 수집 ──────────────────────────────────────────────
    ev: dict = {"realtime_updates": [], "recording_starts": [],
                "vad_starts": [], "finals": [], "realtime_texts": []}
    t_start = [None]

    def on_rt_update(text):
        ev["realtime_updates"].append(time.perf_counter())
        ev["realtime_texts"].append(text)

    def on_vad_start():
        ev["vad_starts"].append(time.perf_counter())

    def on_rec_start():
        ev["recording_starts"].append(time.perf_counter())

    final_engine, rt_engine = _make_inproc_engines(args)
    final_times: list[float] = []
    rt_times: list[float] = []
    _wrap_engine_transcribe(final_engine, final_times)
    _wrap_engine_transcribe(rt_engine, rt_times)
    recorder = AudioToTextRecorder(
        model=args.model,
        realtime_model_type=args.model,
        language="en",
        device=args.device,
        compute_type=args.compute_type,
        use_microphone=False,
        spinner=False,
        enable_realtime_transcription=True,
        realtime_processing_pause=args.realtime_pause,
        on_realtime_transcription_update=on_rt_update,
        on_vad_detect_start=on_vad_start,
        on_recording_start=on_rec_start,
        post_speech_silence_duration=args.silence,
        min_length_of_recording=0.0,
        min_gap_between_recordings=0.0,
        pre_recording_buffer_duration=1.0,
        silero_use_onnx=True,
        silero_deactivity_detection=True,
        faster_whisper_vad_filter=False,
        no_log_file=True,
        transcription_executor=final_engine,
        realtime_transcription_executor=rt_engine,
    )

    chunk = max(1, args.chunk_ms * SR // 1000)
    feed_done = threading.Event()
    text_done = threading.Event()
    feed_done_at = [None]

    def recorder_idle():
        pend = getattr(recorder, "has_pending_recordings", None)
        pend = pend() if callable(pend) else False
        return (not recorder.is_recording and not recorder.frames and not pend)

    def feeder():
        try:
            _feed_chunks(recorder, np.zeros(int(0.5 * SR), np.int16), chunk, args.realtime_pace)
            _feed_chunks(recorder, samples, chunk, args.realtime_pace)
            _feed_chunks(recorder, np.zeros(int(args.silence * 2 * SR), np.int16),
                         chunk, args.realtime_pace)
            fb = getattr(recorder, "flush_buffered_audio", None)
            if callable(fb):
                fb()
        finally:
            feed_done_at[0] = time.perf_counter()
            feed_done.set()

    text_err: list[str] = []

    def text_loop():
        try:
            while True:
                txt = recorder.text()
                if txt:
                    ev["finals"].append((time.perf_counter(), txt))
                if feed_done.is_set() and recorder_idle():
                    break
        except Exception as exc:  # 묵살 금지 — REPORT/콘솔에 남김
            text_err.append(repr(exc))
            print(f"  [text_loop error] {exc!r}", flush=True)
        finally:
            text_done.set()

    proc = psutil.Process()
    peak_rss = [proc.memory_info().rss]

    def mem_watch():
        while not text_done.is_set():
            peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
            time.sleep(0.2)

    wall0 = time.perf_counter()
    t_start[0] = wall0
    ft = threading.Thread(target=feeder, daemon=True)
    ct = threading.Thread(target=text_loop, daemon=True)
    mt = threading.Thread(target=mem_watch, daemon=True)
    ft.start(); ct.start(); mt.start()

    # idle watchdog (공식 regression 패턴)
    while not text_done.is_set():
        if feed_done.is_set() and feed_done_at[0] is not None:
            idle_for = time.perf_counter() - feed_done_at[0]
            if (idle_for >= args.idle_timeout and recorder_idle()
                    and getattr(recorder, "transcribe_count", 0) == 0):
                recorder.interrupt_stop_event.set()
                break
        time.sleep(0.05)

    ct.join(timeout=30)
    ft.join(timeout=5)
    recorder.shutdown()
    wall = time.perf_counter() - wall0

    # ── 지표 계산 ───────────────────────────────────────────────────────────
    hyp = " ".join(t for _, t in ev["finals"]).strip()
    wer = jiwer.wer(normalize(ref), normalize(hyp)) if hyp else 1.0

    with _LOCK:
        fin = list(final_times)
        rt = list(rt_times)
    # final-model RTF = 확정 전사 throughput (실시간 충족 핵심 지표)
    rtf_final = float(np.sum(fin)) / audio_dur if audio_dur else 0.0
    # realtime overhead = 라이브 미리보기 재연산 비용 (audio 당)
    rtf_realtime = float(np.sum(rt)) / audio_dur if audio_dur else 0.0
    rtf_wall = wall / audio_dur if audio_dur else 0.0
    # scheduler latency = final 모델 호출당 처리 시간 분포
    sched_p50 = float(np.percentile(fin, 50)) * 1000 if fin else 0.0
    sched_p95 = float(np.percentile(fin, 95)) * 1000 if fin else 0.0

    # first realtime latency = 발화 onset(첫 recording_start) → 그 직후 첫 realtime partial.
    # (이전 정의는 feed 시작 기준이라 도입부 침묵을 포함 → 보정)
    first_rt = None
    if ev["recording_starts"] and ev["realtime_updates"]:
        onset = min(ev["recording_starts"])
        after = [u for u in ev["realtime_updates"] if u >= onset]
        if after:
            first_rt = min(after) - onset
    # final latency = feed 완료 → 마지막 final (음수 = feed 완료 전 확정 = 백로그 0)
    final_lat = None
    if ev["finals"] and feed_done_at[0] is not None:
        final_lat = max(t for t, _ in ev["finals"]) - feed_done_at[0]
    # arm→speech gap: on_vad_detect_start("listening" armed) → on_recording_start.
    # ※ 진짜 VAD 처리지연 아님 — 도입부 침묵 길이를 반영. 정직 보고용.
    arm_gaps = []
    for vs in ev["vad_starts"]:
        later = [rs for rs in ev["recording_starts"] if rs >= vs]
        if later:
            arm_gaps.append(min(later) - vs)
    arm_gap = float(np.mean(arm_gaps)) * 1000 if arm_gaps else None
    # flicker: realtime partial text 변경 횟수 / 최종 단어 수
    changes = sum(1 for i in range(1, len(ev["realtime_texts"]))
                  if ev["realtime_texts"][i] != ev["realtime_texts"][i - 1])
    n_words = max(1, len(hyp.split()))
    flicker = changes / n_words

    # ── 전사 텍스트 덤프 (품질 검토용) ──────────────────────────────────────
    dump_dir = REPO / "outputs" / "experiments" / args.name / "transcripts"
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / f"{mid}.ref.txt").write_text(ref + "\n")
    (dump_dir / f"{mid}.hyp.txt").write_text(hyp + "\n")
    # jiwer word-level alignment (S/D/I 가시화)
    try:
        out = jiwer.process_words(normalize(ref), normalize(hyp))
        (dump_dir / f"{mid}.align.txt").write_text(jiwer.visualize_alignment(out))
    except Exception as exc:
        (dump_dir / f"{mid}.align.txt").write_text(f"[align failed] {exc!r}\n")

    return {
        "meeting_id": mid,
        "audio_dur_sec": round(audio_dur, 2),
        "n_words_ref": len(ref.split()),
        "n_words_hyp": len(hyp.split()),
        "wer": round(wer, 4),
        "rtf_final": round(rtf_final, 4),
        "rtf_realtime": round(rtf_realtime, 4),
        "rtf_wall": round(rtf_wall, 4),
        "first_realtime_lat_s": round(first_rt, 3) if first_rt is not None else None,
        "final_latency_s": round(final_lat, 3) if final_lat is not None else None,
        "sched_p50_ms": round(sched_p50, 2),
        "sched_p95_ms": round(sched_p95, 2),
        "arm_to_speech_gap_ms": round(arm_gap, 2) if arm_gap is not None else None,
        "flicker_rate": round(flicker, 4),
        "final_calls": len(fin),
        "realtime_calls": len(rt),
        "peak_rss_mb": round(peak_rss[0] / 1e6, 1),
        "wall_sec": round(wall, 1),
        "text_errors": text_err,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=["EN2002a", "EN2002b", "EN2002c"])
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--cap-sec", type=float, default=0.0, help="0=전체")
    ap.add_argument("--chunk-ms", type=int, default=32)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--realtime-pause", type=float, default=0.2)
    ap.add_argument("--silence", type=float, default=0.6)
    ap.add_argument("--idle-timeout", type=float, default=3.0)
    ap.add_argument("--realtime-pace", action="store_true", default=True)
    ap.add_argument("--no-realtime-pace", dest="realtime_pace", action="store_false")
    ap.add_argument("--name", default="2026-06-07_ami_realtimestt")
    args = ap.parse_args()

    _trust_torch_hub()

    out_dir = REPO / "outputs" / "experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mid in args.meetings:
        print(f"\n===== {mid} =====", flush=True)
        r = run_meeting(mid, args)
        rows.append(r)
        print(json.dumps(r, indent=2), flush=True)
        (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
        write_report(out_dir, rows, args)

    print(f"\nDONE → {out_dir}/REPORT.md", flush=True)


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def write_report(out_dir: Path, rows: list[dict], args) -> None:
    def fmt(x, n=3):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"

    L = [
        f"# AMI × RealtimeSTT 실시간 스트리밍 벤치마크 ({args.name})",
        "",
        "## 실험 setup",
        "- **목적**: Hi-OnTop 이 전제하는 실시간 스트리밍 STT 파이프라인",
        "  (RealtimeSTT + faster-whisper) 의 지연/비용/정확도 지표를 AMI 회의 음성으로 검증.",
        "- **데이터**: AMI Meeting Corpus IHM (edinburghcstr/ami, test split).",
        "  발화 단위 parquet 을 begin_time 타임라인에 합성해 연속 단일 스트림 WAV 로 재구성"
        " (`scripts/ami_prep.py`).",
        f"  미팅 {len(rows)}개, 각 앞 {args.cap_sec/60 if args.cap_sec else '전체'}분 (16kHz mono).",
        f"- **STT**: faster-whisper `{args.model}`, device={args.device}, "
        f"compute_type={args.compute_type}, beam_size={args.beam_size}.",
        "- **스트리밍 시뮬레이션**: WAV 를 32ms 청크로 **실시간 페이스**(마이크 모방)"
        " `feed_audio()`. final + realtime 모델을 **in-process executor** 로 주입해"
        " 모든 모델 호출의 추론 시간을 정밀 계측.",
        "- **환경 주의**: GPU 미가용(CPU only) → Peak VRAM 대신 process RSS 보고."
        " GPU 에서는 RTF/지연 모두 크게 개선됨 (본 수치는 CPU 보수 상한).",
        "",
        "## 결과 (per meeting)",
        "",
        "| meeting | dur(s) | WER | RTF_final | RTF_rt | RTF_wall | "
        "first_rt(s) | final_lat(s) | sched p50/p95(ms) | arm→spch(ms) | flicker | "
        "calls(f/rt) | RSS(MB) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        L.append(
            f"| {r['meeting_id']} | {r['audio_dur_sec']:.0f} | {fmt(r['wer'])} | "
            f"{fmt(r['rtf_final'])} | {fmt(r['rtf_realtime'])} | {fmt(r['rtf_wall'])} | "
            f"{fmt(r['first_realtime_lat_s'])} | {fmt(r['final_latency_s'])} | "
            f"{fmt(r['sched_p50_ms'],0)}/{fmt(r['sched_p95_ms'],0)} | "
            f"{fmt(r['arm_to_speech_gap_ms'],0)} | {fmt(r['flicker_rate'])} | "
            f"{r['final_calls']}/{r['realtime_calls']} | {r['peak_rss_mb']:.0f} |")

    L += [
        "",
        "### 평균",
        "",
        f"- WER: **{fmt(_mean(rows,'wer'))}**",
        f"- RTF_final: **{fmt(_mean(rows,'rtf_final'))}** "
        f"(< 1.0 = 확정 전사 실시간 충족)",
        f"- RTF_realtime: {fmt(_mean(rows,'rtf_realtime'))} (라이브 프리뷰 재연산 비용/audio)",
        f"- First realtime latency (발화 onset 기준): {fmt(_mean(rows,'first_realtime_lat_s'))} s",
        f"- Final latency (transcription gap): {fmt(_mean(rows,'final_latency_s'))} s",
        f"- Scheduler p50/p95: {fmt(_mean(rows,'sched_p50_ms'),0)} / "
        f"{fmt(_mean(rows,'sched_p95_ms'),0)} ms",
        f"- Arm→speech gap: {fmt(_mean(rows,'arm_to_speech_gap_ms'),0)} ms "
        f"(VAD 처리지연 아님 — 도입부 침묵 반영)",
        f"- Flicker rate: {fmt(_mean(rows,'flicker_rate'))}",
        f"- Peak RSS: {fmt(_mean(rows,'peak_rss_mb'),0)} MB",
        "",
        "## 지표 정의",
        "- **WER**: jiwer, 소문자·구두점 제거 정규화 후 reference(해당 cap 구간 발화) vs 최종 전사.",
        "- **RTF_final** = Σ(최종 모델 추론시간)/오디오길이. 스트리밍 keep-up 핵심 지표.",
        "- **RTF_realtime** = Σ(realtime 모델 추론시간)/오디오길이. 늘어나는 버퍼 반복 재연산.",
        "- **RTF_wall** = 전체 wall/오디오길이 (실시간 페이스라 ~1.0 부근, 백로그 시 상승).",
        "- **First realtime latency**: 발화 onset(첫 on_recording_start) → 그 직후 첫 realtime partial.",
        "- **Final latency**: feed 완료 → 마지막 final 확정 (= final transcription gap; 음수=백로그 0).",
        "- **Scheduler p50/p95**: 최종 모델 호출당 추론시간 분포.",
        "- **Arm→speech gap**: on_vad_detect_start(\"listening\" armed) → on_recording_start."
        " ※ 진짜 VAD 처리지연 아님 — 도입부 침묵 길이 반영.",
        "- **Flicker rate**: realtime partial 텍스트 변경 횟수 / 최종 단어 수.",
        "- **Model calls (f/rt)**: 최종/realtime 모델 호출 수.",
        "",
        "## 한계 / 검증 미해결",
        "- **CPU only**: GPU 부재로 RTF·지연은 보수적 상한. Peak VRAM 미측정(RSS 대용).",
        f"- `{args.model}` 모델: 회의 음성(중첩·잡음)에서 WER 높음 — STT 품질은 본 연구"
        " 범위 밖(상위 모델 교체로 개선 가능), Hi-OnTop 검증 목적은 지연/비용 축.",
        "- IHM 채널 합성 재구성: 실제 단일 원거리 마이크(SDM)와 음향 특성 다름.",
        "- **Concurrency(동시성)**: 본 스크립트는 단일 스트림. WebSocket 서버 기반 N-동시"
        " 클라이언트 부하 시험은 별도 실험으로 분리 (RealtimeSTT_server 필요).",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
