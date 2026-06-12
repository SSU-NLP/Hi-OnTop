#!/usr/bin/env python3
"""DTS 대화 전문 + 정답 경계(topic_id 변화 포함) 보기 → markdown. AMI와 경계 convention 비교용."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
L = ["# DTS 대화 전문 + 정답 화제 경계 (AMI와 convention 비교)", "",
     "`segmentation_label=1` = 정답 경계(★). `topic_id` 도 같이 표시해 경계가 "
     "**화제 도입 발화**에 찍히는지 **그 응답**에 찍히는지 본다. 발화 전문(자르지 않음).", ""]

for ds, n_dialogs in [("dialseg711", 2), ("tiage", 2), ("superseg", 2)]:
    f = REPO/"benchmarks"/"superdialseg_data"/ds/"segmentation_file_test.json"
    raw = json.load(open(f)); arr = raw["dial_data"][list(raw["dial_data"])[0]]
    L.append(f"## {ds}")
    for di in range(n_dialogs):
        d = arr[di]; turns = d["turns"]
        nb = sum(t.get("segmentation_label", 0) for t in turns)
        L.append(f"\n### {ds} dialog #{di} — {len(turns)} turns, 경계 {nb}개\n")
        L.append("| # | role | topic | 경계 | 발화 |")
        L.append("|--:|:--|--:|:--:|------|")
        for i, t in enumerate(turns):
            sl = t.get("segmentation_label", 0)
            mark = "★" if sl == 1 else ""
            role = t.get("role", "")
            L.append(f"| {i} | {role} | {t.get('topic_id','')} | {mark} | {t['utterance']} |")
    L.append("")

# convention 분석: 경계(label=1) turn 의 topic_id 가 다음 turn 과 같나 다른가
L.append("## convention 분석 (전체 test, label=1 위치)")
for ds in ("tiage", "dialseg711", "superseg"):
    f = REPO/"benchmarks"/"superdialseg_data"/ds/"segmentation_file_test.json"
    arr = json.load(open(f))["dial_data"]; arr = arr[list(arr)[0]]
    same_next = diff_next = 0   # label=1 turn 의 topic_id == 다음 turn topic_id ?
    for d in arr:
        ts = d["turns"]
        for i in range(len(ts)-1):
            if ts[i].get("segmentation_label", 0) == 1:
                if ts[i]["topic_id"] == ts[i+1]["topic_id"]:
                    same_next += 1
                else:
                    diff_next += 1
    tot = same_next + diff_next
    L.append(f"- **{ds}**: 경계 turn 의 topic_id 가 *다음 turn 과 같음* {same_next}/{tot} "
             f"({same_next/max(1,tot)*100:.0f}%) → 같으면 '경계 turn = 새 화제의 첫 발화', "
             f"다르면 '경계 turn = 옛 화제 마지막(경계 후 다음부터 새 화제)'.")

out = REPO/"outputs"/"reports"/"dts_conversation_boundaries.md"
out.write_text("\n".join(L)+"\n")
print(f"WROTE {out}")
print("\n".join(L[-4:]))
