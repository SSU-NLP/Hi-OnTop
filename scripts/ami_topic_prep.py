#!/usr/bin/env python3
"""AMI manual annotation → topic-segmentation 평가용 데이터.

AMI(scenario meetings) 의 NITE-XML 주석에서 turn 시퀀스 + 계층적 topic 경계를
추출한다. DTS 3벤치(TIAGE/Dialseg711/SuperDialseg)와 다른 도메인(비즈니스
회의)에서 Hi-OnTop robustness + 분절 입도(granularity) 분석에 사용.

단위
----
- turn = AMI `segment` (한 화자의 연속 발화 단위). 전 화자 segment 를
  transcriber_start 순으로 linearize → 멀티파티 인터리브 turn 스트림.
- topic 경계 = 각 topic 의 첫 단어가 속한 turn = "새 topic 의 첫 turn".
  경계 레이블 yt[i]=1 (i>0) 는 turn i 가 새 topic 의 시작임을 의미
  (run_encoder_comparison 컨벤션과 동일: 분절기가 그 turn 에서 shift 감지).

두 레벨
-------
- **top**: depth-1 topic 만 (굵은 화제 경계, 표준)
- **all**: 모든 topic (하위 포함, 촘촘한 경계)

출력: data/ami/topic/<meeting>.json
  {meeting, turns:[{start,speaker,text}], bnd_top:[0/1...], bnd_all:[0/1...],
   topic_levels:[{start_turn, depth, label}]}
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANN = REPO / "data" / "ami" / "annotations"
OUT = REPO / "data" / "ami" / "topic"
NID = "{http://nite.sourceforge.net/}id"
NCHILD = "{http://nite.sourceforge.net/}child"
NPTR = "{http://nite.sourceforge.net/}pointer"

_HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def _load_onto() -> dict[str, str]:
    onto: dict[str, str] = {}
    root = ET.parse(ANN / "ontologies" / "default-topics.xml").getroot()

    def rec(el):
        nid, nm = el.get(NID), el.get("name")
        if nid and nm:
            onto[nid] = nm
        for c in el:
            rec(c)

    rec(root)
    return onto


def _wid_index(s: str) -> int:
    """ES2002a.A.words13 → 13 (정렬용 보조; 실제 시간은 words 에서)."""
    m = re.search(r"words(\d+)$", s)
    return int(m.group(1)) if m else -1


def load_words(meeting: str) -> dict[str, tuple[float, str, str]]:
    """wid → (starttime, text, speaker)."""
    words: dict[str, tuple[float, str, str]] = {}
    for f in glob(str(ANN / "words" / f"{meeting}.*.words.xml")):
        spk = Path(f).name.split(".")[1]
        for el in ET.parse(f).getroot():
            wid = el.get(NID)
            if not wid:
                continue
            st = el.get("starttime")
            if el.get("punc") == "true":   # 구두점은 텍스트만, 시각 옵션
                txt = (el.text or "").strip()
                words[wid] = (float(st) if st else -1.0, txt, spk)
            else:
                words[wid] = (float(st) if st else -1.0, (el.text or "").strip(), spk)
    return words


def _expand_range(a: str, b: str | None) -> list[str]:
    """words13 .. words48 → [words13..words48] (같은 화자 가정)."""
    if not b:
        return [a]
    pre = a.rsplit("words", 1)[0]
    ia, ib = _wid_index(a), _wid_index(b)
    if ia < 0 or ib < 0:
        return [a]
    return [f"{pre}words{i}" for i in range(ia, ib + 1)]


def load_segments(meeting: str, words: dict) -> list[dict]:
    """turn 리스트 (transcriber_start 순). 각 turn = AMI segment."""
    turns: list[dict] = []
    for f in glob(str(ANN / "segments" / f"{meeting}.*.segments.xml")):
        spk = Path(f).name.split(".")[1]
        for seg in ET.parse(f).getroot():
            st = seg.get("transcriber_start")
            if st is None:
                continue
            wids: list[str] = []
            for ch in seg.findall(NCHILD):
                m = _HREF_RE.search(ch.get("href", ""))
                if m:
                    wids += _expand_range(m.group(1), m.group(2))
            toks = [words[w][1] for w in wids if w in words and words[w][1]]
            text = " ".join(toks)
            text = re.sub(r"\s+([,.?!;:])", r"\1", text).strip()
            if text:
                turns.append({"start": float(st), "speaker": spk,
                              "text": text, "wids": set(wids)})
    turns.sort(key=lambda t: t["start"])
    return turns


def load_topics(meeting: str, words: dict, onto: dict) -> list[dict]:
    """topic 리스트: {start_time, depth, label, first_wid}. 계층 보존."""
    root = ET.parse(ANN / "topics" / f"{meeting}.topic.xml").getroot()
    topics: list[dict] = []

    def label(el) -> str:
        d = el.get("other_description")
        if d:
            return d
        for p in el.findall(NPTR):
            if p.get("role") == "scenario_topic_type":
                m = re.search(r"id\(([^)]+)\)", p.get("href", ""))
                if m:
                    return onto.get(m.group(1), m.group(1))
        return "(unlabeled)"

    def topic_words(el) -> list[str]:
        out: list[str] = []
        for ch in el.findall(NCHILD):
            m = _HREF_RE.search(ch.get("href", ""))
            if m:
                out += _expand_range(m.group(1), m.group(2))
        return out

    def walk(el, depth):
        if el.tag == "topic":
            wids = topic_words(el)   # 직속 child word (하위 topic 제외)
            starts = [words[w][0] for w in wids if w in words and words[w][0] >= 0]
            st = min(starts) if starts else None
            if st is not None:
                topics.append({"start_time": st, "depth": depth,
                               "label": label(el)})
            for c in el:
                walk(c, depth + 1 if el.tag == "topic" else depth)
        else:
            for c in el:
                walk(c, depth)

    for top in root:
        walk(top, 1)
    topics.sort(key=lambda x: x["start_time"])
    return topics


def build_boundaries(turns: list[dict], topics: list[dict]) -> dict:
    """topic start_time → 첫 turn index. top/all 두 레벨 yt 생성."""
    n = len(turns)
    starts = [t["start"] for t in turns]

    def first_turn_at(t0: float) -> int:
        # t0 이상에서 시작하는 첫 turn
        for i, s in enumerate(starts):
            if s >= t0 - 1e-6:
                return i
        return n - 1

    bnd_top = [0] * n
    bnd_all = [0] * n
    levels: list[dict] = []
    seen_top = False   # 첫 depth-1 topic onset = 대화 시작(전환 아님) → skip
    seen_any = False   # 첫 topic onset (all 레벨) skip
    for tp in topics:
        ti = first_turn_at(tp["start_time"])
        levels.append({"start_turn": ti, "depth": tp["depth"], "label": tp["label"]})
        if 0 < ti < n - 1:               # turn 0 / 마지막은 경계 아님
            if seen_any:
                bnd_all[ti] = 1
            if tp["depth"] == 1 and seen_top:
                bnd_top[ti] = 1
        seen_any = True
        if tp["depth"] == 1:
            seen_top = True
    return {"bnd_top": bnd_top, "bnd_all": bnd_all, "topic_levels": levels}


def merge_speaker_turns(turns: list[dict]) -> list[dict]:
    """시간순 인접 segment 를 **같은 화자면 병합** → speaker-turn.

    AMI segment 는 억양구 단위(짧은 backchannel 다수)라 인접-임베딩 분절에
    노이즈. 한 화자의 연속 발화를 한 turn 으로 묶어 신호 안정화."""
    if not turns:
        return turns
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["text"] = (merged[-1]["text"] + " " + t["text"]).strip()
            merged[-1]["wids"] |= t["wids"]
        else:
            merged.append({"start": t["start"], "speaker": t["speaker"],
                           "text": t["text"], "wids": set(t["wids"])})
    return merged


def process(meeting: str, onto: dict, merge: bool) -> dict | None:
    words = load_words(meeting)
    if not words:
        return None
    turns = load_segments(meeting, words)
    if merge:
        turns = merge_speaker_turns(turns)
    if len(turns) < 5:
        return None
    topics = load_topics(meeting, words, onto)
    b = build_boundaries(turns, topics)
    return {
        "meeting": meeting,
        "n_turns": len(turns),
        "n_topic_top": sum(1 for t in topics if t["depth"] == 1),
        "n_topic_all": len(topics),
        "turns": [{"start": round(t["start"], 2), "speaker": t["speaker"],
                   "text": t["text"]} for t in turns],
        "bnd_top": b["bnd_top"],
        "bnd_all": b["bnd_all"],
        "topic_levels": b["topic_levels"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="미팅 id (생략 시 전체)")
    ap.add_argument("--merge", action="store_true",
                    help="같은 화자 연속 segment 를 speaker-turn 으로 병합")
    args = ap.parse_args()

    onto = _load_onto()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.meetings:
        meetings = args.meetings
    else:
        meetings = sorted({Path(f).name.split(".")[0]
                           for f in glob(str(ANN / "topics" / "*.topic.xml"))})

    manifest = []
    for mid in meetings:
        try:
            d = process(mid, onto, args.merge)
        except Exception as exc:
            print(f"[skip] {mid}: {exc!r}", flush=True)
            continue
        if d is None:
            print(f"[skip] {mid}: 데이터 부족", flush=True)
            continue
        (OUT / f"{mid}.json").write_text(json.dumps(d, ensure_ascii=False))
        manifest.append({"meeting": mid, "n_turns": d["n_turns"],
                         "n_topic_top": d["n_topic_top"],
                         "n_topic_all": d["n_topic_all"],
                         "n_bnd_top": sum(d["bnd_top"]),
                         "n_bnd_all": sum(d["bnd_all"])})
        print(f"[ok] {mid}: {d['n_turns']} turns, "
              f"top {d['n_topic_top']}({sum(d['bnd_top'])} bnd) / "
              f"all {d['n_topic_all']}({sum(d['bnd_all'])} bnd)", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} meetings → {OUT}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
