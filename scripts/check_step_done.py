#!/usr/bin/env python3
"""Step 완료 검증.

Step을 [x]로 표시하기 직전 반드시 실행. 통과할 때까지 수정-재실행 반복.

Usage:
    python scripts/check_step_done.py              # 현재 진행 중인 Step 자동 감지
    python scripts/check_step_done.py --step 0-1   # 특정 Step 검증

Exit codes:
    0 = 모든 검사 통과 (Step 완료 가능)
    1 = FAIL 있음 (수정 후 재실행 필요)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True, cwd=_SCRIPT_DIR
    ).strip()
)

Result = tuple[str, str]  # (level, message), level in {OK, WARN, FAIL}


def _ok(r: list[Result], m: str) -> None:   r.append(("OK", m))
def _warn(r: list[Result], m: str) -> None: r.append(("WARN", m))
def _fail(r: list[Result], m: str) -> None: r.append(("FAIL", m))


def detect_current_step() -> str | None:
    """plan.md에서 첫 번째 미완료 '- [ ]' 항목이 속한 Step ID (예: '0-1') 반환."""
    text = (ROOT / "plan.md").read_text(encoding="utf-8")
    section = None
    for line in text.splitlines():
        m = re.match(r"###\s*(\d+-\d+)\.", line)
        if m:
            section = m.group(1)
            continue
        if section and re.match(r"\s*-\s*\[\s\]\s", line):
            return section
    return None


def check_common(results: list[Result]) -> None:
    """모든 Step 공통 검증."""
    handoff = (ROOT / "handoff.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*마지막 업데이트\*\*:\s*(\d{4}-\d{2}-\d{2})", handoff)
    if not m:
        _fail(results, "handoff.md: '마지막 업데이트: YYYY-MM-DD' 항목 없음")
    else:
        stamp = m.group(1)
        today = date.today().isoformat()
        if stamp != today:
            _fail(results, f"handoff.md 마지막 업데이트 {stamp} != 오늘({today}) — 갱신 필요")
        else:
            _ok(results, f"handoff.md 최신 ({stamp})")

    for section in ["## 현재 상태", "## 다음 할 일"]:
        if section not in handoff:
            _fail(results, f"handoff.md: '{section}' 섹션 없음")

    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, cwd=str(ROOT)
    )
    dirty = [ln for ln in status.splitlines() if ln.strip() and "setup_colab.ipynb" not in ln]
    if dirty:
        _warn(results, f"uncommitted 변경 {len(dirty)}건 — Step 완료 후 commit 필요")
    else:
        _ok(results, "git working tree clean")


STEP_CHECKS: dict[str, callable] = {}


def step(step_id: str):
    def deco(fn):
        STEP_CHECKS[step_id] = fn
        return fn
    return deco


@step("0-1")
def _check_0_1(results: list[Result]) -> None:
    """SEM 논문 정독 — context/00-sem-paper.md 실질 내용 확인."""
    path = ROOT / "context" / "00-sem-paper.md"
    if not path.exists():
        _fail(results, f"{path.relative_to(ROOT)} 없음")
        return
    text = path.read_text(encoding="utf-8")
    if len(text) < 2000:
        _fail(results, f"{path.relative_to(ROOT)}: {len(text)}자 — 정독 후 정리로 보기엔 부족")
    else:
        _ok(results, f"{path.relative_to(ROOT)}: {len(text)}자")
    for kw in ["sticky-CRP", "prediction error"]:
        if kw.lower() not in text.lower():
            _warn(results, f"{path.relative_to(ROOT)}에 '{kw}' 언급 없음")


@step("0-2")
def _check_0_2(results: list[Result]) -> None:
    """벤치마크 데이터 분석 — outputs/benchmark-analysis.md 확인."""
    path = ROOT / "outputs" / "benchmark-analysis.md"
    if not path.exists():
        _fail(results, f"{path.relative_to(ROOT)} 없음")
        return
    text = path.read_text(encoding="utf-8")
    for bench in ["LoCoMo", "TopiOCQA", "LongMemEval"]:
        if bench not in text:
            _fail(results, f"benchmark-analysis.md에 {bench} 섹션 없음")
    if len(text) < 1500:
        _fail(results, f"benchmark-analysis.md: {len(text)}자 — 분석으로 보기엔 부족")
    else:
        _ok(results, f"benchmark-analysis.md: {len(text)}자")


@step("1-3")
def _check_1_3(results: list[Result]) -> None:
    """Phase 1-3 — outputs/phase-1-topiocqa.md + Gate 결과 기록 확인."""
    path = ROOT / "outputs" / "phase-1-topiocqa.md"
    if not path.exists():
        _fail(results, f"{path.relative_to(ROOT)} 없음")
        return
    text = path.read_text(encoding="utf-8")
    for kw in ["all-boundary", "cosine", "Hi-OnTop", "Latency", "Gate"]:
        if kw not in text:
            _warn(results, f"phase-1-topiocqa.md에 '{kw}' 섹션/키워드 없음")
    if len(text) < 1500:
        _fail(results, f"phase-1-topiocqa.md 내용 부족 ({len(text)}자)")
    else:
        _ok(results, f"phase-1-topiocqa.md: {len(text)}자")


@step("1-4")
def _check_1_4(results: list[Result]) -> None:
    """Phase 1-4 Gate — outputs/phase-1-topiocqa.md에 'Gate 결과: PASS' 또는 FAIL 처리 기록."""
    path = ROOT / "outputs" / "phase-1-topiocqa.md"
    if not path.exists():
        _fail(results, f"{path.relative_to(ROOT)} 없음 — Step 1-3 먼저 수행")
        return
    text = path.read_text(encoding="utf-8")
    if "Gate 결과: PASS" in text:
        _ok(results, "phase-1-topiocqa.md: Gate PASS 기록됨")
    elif "Gate 결과: FAIL" in text:
        _ok(results, "phase-1-topiocqa.md: Gate FAIL — decision-log append + 옵션 D 전환 확인 필요")
        log = (ROOT / "context" / "06-decision-log.md").read_text(encoding="utf-8")
        if "옵션 A 번복" not in log:
            _fail(results, "FAIL인데 decision-log에 '옵션 A 번복' 기록 없음")
    else:
        _fail(results, "phase-1-topiocqa.md에 'Gate 결과' 명시 없음")


@step("0-3")
def _check_0_3(results: list[Result]) -> None:
    """사건 모델 설계 확정."""
    design_path = ROOT / "context" / "01-hi-ontop-design.md"
    design = design_path.read_text(encoding="utf-8")
    stale_markers = ["미확정", "TBD", "TODO", "???"]
    hits = [m for m in stale_markers if m in design]
    if hits:
        _fail(results, f"01-hi-ontop-design.md에 미해결 마커 존재: {hits}")
    else:
        _ok(results, "01-hi-ontop-design.md 미확정 마커 없음")

    math = (ROOT / "context" / "02-math-model.md").read_text(encoding="utf-8")
    if any(m in math for m in stale_markers):
        _fail(results, "02-math-model.md에 미해결 마커 존재")
    else:
        _ok(results, "02-math-model.md 미확정 마커 없음")

    log = (ROOT / "context" / "06-decision-log.md").read_text(encoding="utf-8")
    today = date.today().isoformat()
    if today not in log:
        _warn(results, f"06-decision-log.md에 오늘({today}) 결정 기록 없음")
    else:
        _ok(results, "06-decision-log.md 오늘 결정 기록됨")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", help="검증할 Step ID (예: 0-1). 생략 시 자동 감지.")
    args = parser.parse_args()

    step_id = args.step or detect_current_step()

    print("=" * 48)
    print(f"  Step 완료 검증  /  Step: {step_id or '(감지 불가)'}")
    print("=" * 48)

    results: list[Result] = []
    check_common(results)

    if step_id and step_id in STEP_CHECKS:
        STEP_CHECKS[step_id](results)
    elif step_id:
        _warn(results, f"Step {step_id}: 전용 검증 없음 — STEP_CHECKS에 추가하거나 수동 확인")
    else:
        _fail(results, "현재 Step 자동 감지 실패 — --step <id>로 명시하거나 plan.md 확인")

    fail_count = 0
    for level, msg in results:
        marker = {"OK": "  [OK]  ", "WARN": "  [WARN]", "FAIL": "  [FAIL]"}[level]
        print(f"{marker} {msg}")
        if level == "FAIL":
            fail_count += 1

    print("=" * 48)
    if fail_count == 0:
        print(f"통과. Step {step_id} 완료 처리 가능.")
        return 0
    print(f"FAIL {fail_count}건. 수정 후 재실행.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
