#!/usr/bin/env python3
"""SeCom/baseline **실제 워크로드** LLM latency 전수 측정 (fresh, N=1/모델, RESUME).

품질 sweep(`secom_llm_eval.py`)이 *실제로 보내는 그 호출들*을 그대로 재현해 round-trip latency 를 잰다.
→ home-made `llm_latency_full.py`(WIN_PROMPT·max_tokens=256, 실제와 불일치) **대체**.

왜 이 스크립트가 필요한가 (재측정 0 보장 조건):
- **프롬프트 일치**: 품질과 동일한 SeCom 프롬프트(`--prompt baseline|verbatim`). 출력토큰이 LLM latency 를 지배하므로
  **모드별 정확한 max_tokens**(incremental=16 / segment=1024)로 재야 대표성 있음.
- **fresh·캐시금지**: 응답캐시 재사용은 즉답(0ms) → latency 오염. 매 호출 fresh.
- **N=1/모델(동시부하 0)**: 모델 내부는 **완전 순차**(self-contention 0 = 순수 per-call latency).
  모델끼리는 병렬(다른 endpoint). cf. §3b 동시부하=상한 문제 회피.
- **전수**(사용자 결정 2026-06-12): incremental 17,397콜 포함 실제 호출 전부. segment=모든 window, full=회의당 1콜.
- **RESUME**: (model,mode,buffer,callkey) 단위 checkpoint(append+flush). 재시작 시 done skip.
  incremental 은 stateful(prev_session 이 과거 Yes/No 에 의존) → checkpoint 에 **answer 도 저장**, resume 시 그걸로
  상태 재구성하며 un-done 턴만 fresh 호출. 출력토큰수·retry·invalid 도 기록(end-to-end 지표).

usage:
  python scripts/secom_latency_full.py --prompt baseline                 # 측정(이어하기)
  python scripts/secom_latency_full.py --prompt baseline --models qwen…  # 일부 모델
  python scripts/secom_latency_full.py --report                          # 모델×모드/버퍼 p50/95/99 집계
handoff HANDOFF_0612 §3z-C(latency). latency 는 품질 sweep 종료 후(깨끗한 API/CPU) 단독 실행할 것.
"""
from __future__ import annotations
import sys, os, json, time, hashlib, threading, argparse, random
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import numpy as np
import secom_llm_eval as S   # load_prompts, load_meetings, fmt_exchanges, make_client, P_SEG/P_INC

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "outputs" / "runs" / "_misc" / "secom_latency_full.jsonl"

# 모델셋 확정 2026-06-12 (사용자, 전 실험 공통).
MODELS = ["openrouter/openai/gpt-5-mini", "openrouter/openai/gpt-5-nano",
          "openrouter/qwen/qwen3.5-27b", "openrouter/anthropic/claude-haiku-4.5",
          "openrouter/mistralai/mistral-small-3.1-24b-instruct",
          "openrouter/google/gemma-3-12b-it", "openrouter/google/gemma-3n-e4b-it"]
# 실제 sweep 과 동일한 segment 버퍼들 + incremental
SEG_BUFFERS = ["full", 120, 60, 30, 10]
OFFSETS = [0.0, 0.5]


def seg_windows(meetings, B):
    """run_segment 과 동일한 window 구성. 반환 [(mid, base, tuple(idxs))]."""
    out = []
    for mid, turns, n, gold in meetings:
        if B == "full":
            out.append((mid, 0, tuple(range(n))))
            continue
        t0 = turns[0]["start"]; t_last = turns[-1]["start"]
        for off in OFFSETS:
            w_start = t0; w_end = t0 + (off * B if off > 0 else B)
            while w_start <= t_last:
                w = [k for k in range(n) if w_start <= turns[k]["start"] < w_end]
                if len(w) >= 2:
                    out.append((mid, w[0], tuple(w)))
                w_start = w_end; w_end += B
    return out


def fresh_call(client, model, prompt, mtok, response_format=None):
    """fresh round-trip. 성공한 create() 만 timing. 반환 (lat_ms, ok, retries, out_tok, text)."""
    rf = response_format
    for attempt in range(4):
        t0 = time.perf_counter()
        try:
            kw = {"response_format": rf} if rf else {}
            r = client.chat.completions.create(model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=mtok, temperature=0.0,
                extra_body={"reasoning": {"enabled": False}}, **kw)
            lat = (time.perf_counter() - t0) * 1000
            m = r.choices[0].message
            txt = m.content or getattr(m, "reasoning_content", None) or ""
            ot = getattr(getattr(r, "usage", None), "completion_tokens", None)
            return lat, True, attempt, ot, txt
        except Exception:
            if rf is not None:        # response_format 미지원/실패 → prompt-only strict 로 강등 후 재시도
                rf = None; continue
            if attempt < 3:
                time.sleep((2 ** attempt) * 0.5 + random.random())
    return None, False, 3, None, ""


def load_done():
    """checkpoint → {(model,mode,buffer,key): record}. answer 보존(incremental resume 용)."""
    done = {}
    if CKPT.exists():
        for line in open(CKPT):
            try:
                o = json.loads(line); done[(o["model"], o["mode"], str(o["buffer"]), o["key"])] = o
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(REPO / "outputs/runs/_misc/ami_latency_subset.json"),
                    help="latency 전용 12-미팅 subset(기준 도출, select_latency_subset.py). "
                         "37 전수는 ami_subset.json 지정.")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--prompt", choices=["baseline", "verbatim"], default="baseline",
                    help="품질 sweep 과 동일 프롬프트. latency 는 출력토큰 지배 → 보통 baseline 1세트면 대표적.")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(); return

    S.load_prompts(args.prompt)
    meetings = S.load_meetings(args.subset)
    models = args.models.split(",")
    client = S.make_client()
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done()
    fout = open(CKPT, "a"); lock = threading.Lock()

    def rec(model, mode, buf, key, lat, ok, nr, ot, text, meta):
        o = {"model": model, "mode": mode, "buffer": str(buf), "key": key, "ok": ok,
             "lat_ms": lat, "retries": nr, "out_tok": ot, "ans": text[:64], **meta}
        with lock:
            fout.write(json.dumps(o) + "\n"); fout.flush()
        return o

    def run_model(model):
        prog = 0; t_start = time.perf_counter()
        # --- segment 모드: 모든 버퍼 × 모든 window, 순차 fresh ---
        for B in SEG_BUFFERS:
            for mid, base, w in seg_windows(meetings, B):
                turns = next(t for m, t, n, g in meetings if m == mid)
                prompt = S.P_SEG.format(text_to_be_segmented=S.fmt_exchanges(turns, list(w)))
                key = hashlib.md5(("seg" + str(B) + mid + str(base) + str(w)).encode()).hexdigest()
                if (model, "segment", str(B), key) in done:
                    continue
                rf = S.SEG_RF if S.STRICT else None     # baseline=strict JSON (실워크로드 일치)
                lat, ok, nr, ot, txt = fresh_call(client, model, prompt, 1024, rf)
                rec(model, "segment", B, key, lat, ok, nr, ot, txt, {"mid": mid, "nw": len(w)})
                prog += 1
                if prog % 200 == 0:
                    with lock:
                        print(f"  [{model.split('/')[-1]}] {prog} calls "
                              f"({time.perf_counter()-t_start:.0f}s)", flush=True)
        # --- incremental: 미팅별 순차 replay(상태 의존), checkpoint answer 로 resume ---
        for mid, turns, n, gold in meetings:
            seg_start = 0
            for t in range(1, n):
                prev = S.fmt_exchanges(turns, list(range(seg_start, t)))
                new = f"({turns[t]['speaker']}) {turns[t]['text']}"
                prompt = S.P_INC.format(new_turn=new, prev_session=prev)
                key = hashlib.md5(("inc" + mid + str(t) + str(seg_start)).encode()).hexdigest()
                dk = (model, "incremental", "-", key)
                if dk in done:                      # resume: 저장된 answer 로 상태만 전진
                    ans = (done[dk].get("ans") or "").strip().lower()
                else:
                    lat, ok, nr, ot, txt = fresh_call(client, model, prompt, 16)
                    o = rec(model, "incremental", "-", key, lat, ok, nr, ot, txt,
                            {"mid": mid, "t": t}); done[dk] = o
                    ans = (txt or "").strip().lower(); prog += 1
                    if prog % 500 == 0:
                        with lock:
                            print(f"  [{model.split('/')[-1]}] {prog} calls "
                                  f"({time.perf_counter()-t_start:.0f}s)", flush=True)
                if ans.startswith("no") or ("no" in ans[:5] and "yes" not in ans[:5]):
                    seg_start = t
        with lock:
            print(f"  [{model.split('/')[-1]}] DONE {prog} fresh calls "
                  f"({time.perf_counter()-t_start:.0f}s)", flush=True)

    print(f"{len(meetings)}미팅 × {len(models)}모델. 이미 측정 {len(done)}건. fresh·N=1·RESUME.", flush=True)
    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        for m in models:
            ex.submit(run_model, m)
    print("=== latency 측정 완료 ===", flush=True); report()


def report():
    by = {}
    for line in open(CKPT):
        try:
            o = json.loads(line)
        except Exception:
            continue
        grp = (o["model"], o["mode"], str(o["buffer"]))
        by.setdefault(grp, {"lat": [], "fail": 0, "ot": []})
        if o.get("ok"):
            by[grp]["lat"].append(o["lat_ms"])
            if o.get("out_tok") is not None:
                by[grp]["ot"].append(o["out_tok"])
        else:
            by[grp]["fail"] += 1
    print(f"{'model':<40}{'mode':<12}{'buf':>5}{'n':>7}{'p50':>8}{'p95':>8}{'p99':>8}"
          f"{'out_tok_p50':>12}{'fail':>6}")
    for grp in sorted(by):
        m, mode, buf = grp; a = np.array(by[grp]["lat"]); ot = np.array(by[grp]["ot"])
        if len(a) == 0:
            print(f"{m.replace('openrouter/',''):<40}{mode:<12}{buf:>5}{0:>7}  all-fail {by[grp]['fail']}")
            continue
        otp = f"{np.percentile(ot,50):.0f}" if len(ot) else "-"
        print(f"{m.replace('openrouter/',''):<40}{mode:<12}{buf:>5}{len(a):>7}"
              f"{np.percentile(a,50):>8.0f}{np.percentile(a,95):>8.0f}{np.percentile(a,99):>8.0f}"
              f"{otp:>12}{by[grp]['fail']:>6}  (ms)")


if __name__ == "__main__":
    main()
