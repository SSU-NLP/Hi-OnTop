#!/usr/bin/env python3
"""AMI IHM → 연속 미팅 WAV + reference transcript 재구성.

edinburghcstr/ami 의 IHM test parquet 은 발화(utterance) 단위로 저장돼 있다
(meeting_id, audio_id, text, audio[16kHz wav bytes], begin_time, end_time,
speaker_id). RealtimeSTT 스트리밍 시뮬레이션을 위해 한 미팅의 모든 발화를
begin_time 기준 타임라인에 배치해 **연속 단일 스트림** WAV 로 재구성한다.

- 겹치는 헤드셋 채널은 합산(sum) 후 [-1,1] 클립 (IHM cross-talk 낮아 무해).
- reference transcript = begin_time 정렬 발화 텍스트 join.
- --cap-min 으로 앞부분만 잘라 CPU 런타임 제어.

출력:
  data/ami/audio/<meeting>.wav          16kHz mono
  data/ami/transcripts/<meeting>.txt    공백 정규화 reference (대문자, AMI 관행)
  data/ami/transcripts/<meeting>.jsonl  발화별 (start,end,text) — latency 정렬용
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ami"
PARQUET = DATA / "_parquet" / "ihm"
SR = 16000


def reconstruct(df_m: pd.DataFrame, cap_sec: float | None) -> tuple[np.ndarray, list[dict]]:
    df_m = df_m.sort_values("begin_time").reset_index(drop=True)
    t0 = float(df_m["begin_time"].min())
    end = float(df_m["end_time"].max()) - t0
    if cap_sec is not None:
        end = min(end, cap_sec)
    n = int(np.ceil(end * SR)) + SR
    timeline = np.zeros(n, dtype=np.float32)

    utts: list[dict] = []
    for _, row in df_m.iterrows():
        start = float(row["begin_time"]) - t0
        if cap_sec is not None and start >= cap_sec:
            continue
        arr, srate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if srate != SR:  # AMI IHM 는 이미 16k, 방어적
            continue
        s = int(round(start * SR))
        e = min(s + len(arr), n)
        timeline[s:e] += arr[: e - s]
        utts.append({"start": round(start, 3),
                     "end": round(float(row["end_time"]) - t0, 3),
                     "text": str(row["text"]).strip()})

    np.clip(timeline, -1.0, 1.0, out=timeline)
    # trim trailing pad
    last = int(np.ceil((utts[-1]["end"] if utts else end) * SR)) + SR // 2
    timeline = timeline[: min(last, n)]
    return timeline, utts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="meeting_id 목록 (생략 시 shard 내 전체)")
    ap.add_argument("--cap-min", type=float, default=10.0,
                    help="미팅당 앞부분 분 단위 cap (CPU 런타임 제어). 0=전체")
    ap.add_argument("--shard", default="test-00000-of-00004.parquet")
    args = ap.parse_args()

    cap_sec = None if args.cap_min <= 0 else args.cap_min * 60.0

    pq = PARQUET / args.shard
    df = pd.read_parquet(pq)
    meetings = args.meetings or sorted(df["meeting_id"].unique())

    (DATA / "audio").mkdir(parents=True, exist_ok=True)
    (DATA / "transcripts").mkdir(parents=True, exist_ok=True)

    manifest = []
    for mid in meetings:
        df_m = df[df["meeting_id"] == mid]
        if df_m.empty:
            print(f"[skip] {mid}: shard 에 없음")
            continue
        audio, utts = reconstruct(df_m, cap_sec)
        dur = len(audio) / SR

        wav_p = DATA / "audio" / f"{mid}.wav"
        sf.write(wav_p, audio, SR, subtype="PCM_16")

        ref_text = " ".join(u["text"] for u in utts if u["text"])
        (DATA / "transcripts" / f"{mid}.txt").write_text(ref_text + "\n")
        with open(DATA / "transcripts" / f"{mid}.jsonl", "w") as fh:
            for u in utts:
                fh.write(json.dumps(u) + "\n")

        n_words = len(ref_text.split())
        manifest.append({"meeting_id": mid, "wav": str(wav_p.relative_to(REPO)),
                         "duration_sec": round(dur, 2), "n_utts": len(utts),
                         "n_words": n_words, "cap_min": args.cap_min})
        print(f"[ok] {mid}: {dur/60:.1f}min, {len(utts)} utts, {n_words} words "
              f"→ {wav_p.name}")

    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest → {DATA/'manifest.json'}")


if __name__ == "__main__":
    main()
