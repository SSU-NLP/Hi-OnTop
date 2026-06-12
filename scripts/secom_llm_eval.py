#!/usr/bin/env python3
"""SeCom (Microsoft, ICLR2025) canonical 프롬프트 LLM 분절 baseline.

두 모드 — SeCom 실제 instruction(`scripts/secom_prompts/`) 사용:
- **incremental** (online, 0-look-ahead): 턴마다 "직전 session + 새 turn = 같은 topic? Yes/No".
  No → 경계. 미팅 내부는 순차(prev_session 이 과거 결정에 의존), 미팅·모델 간 병렬.
- **segment** (exchange-number JSONL): 청크(전체 회의=oracle 또는 B초 window=buffer)를 한 번에 세그먼트.
  window 들은 독립 → 완전 병렬.

concurrency(ThreadPool) + prompt 캐시(재실행 skip) + retry/backoff. metric F1/Pk/WD/Score.
AMI 멀티파티 turn 을 SeCom "exchange" 로 매핑(도메인 적응 — speaker tag 포함). handoff HANDOFF_0612.
usage:
  python scripts/secom_llm_eval.py --mode incremental --subset .../ami_subset.json --model openrouter/qwen/qwen3.5-27b
  python scripts/secom_llm_eval.py --mode segment --buffer full --model ...     # offline oracle
  python scripts/secom_llm_eval.py --mode segment --buffer 30 --offsets 0,0.5 --model ...  # buffer 곡선
"""
from __future__ import annotations
import sys, os, json, re, time, hashlib, threading, argparse, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import f1_score
from run_encoder_comparison import official_pk_wd

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
TOPIC = REPO / "data" / "ami" / "topic"
PDIR = REPO / "scripts" / "secom_prompts"
# 우리 baseline(DTS 2-party + AMI 멀티파티 공통, turn 기반) 기본 — SeCom segment_turn 구조를
# online incremental 로 확장. --prompt verbatim 으로 SeCom 원본 그대로.
PROMPTS = {
    "baseline": ("baseline_segment.md", "baseline_incremental.md", "Turn"),
    "verbatim": ("segment_exchange.md", "segment_incremental.md", "Exchange"),
}
P_SEG = ""; P_INC = ""; LABEL = "Turn"; STRICT = False
# baseline 은 strict JSON object 출력(response_format) → mistral 류 구조화 실패 차단. verbatim 은 SeCom 원본 충실(미적용).
SEG_RF = {"type": "json_object"}


def load_prompts(which):
    global P_SEG, P_INC, LABEL, STRICT
    seg, inc, LABEL = PROMPTS[which]
    P_SEG = open(PDIR / seg).read(); P_INC = open(PDIR / inc).read()
    # strict JSON(response_format) 보류 2026-06-12 (사용자) — 버퍼 컨텍스트 설계 결정 후 재검토.
    # 재활성: 아래를 (which == "baseline") 로 + baseline_segment.md 를 JSON-object 판으로 교체.
    STRICT = False


def make_client():
    cf = ({"CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
           "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"]}
          if os.environ.get("CF_ACCESS_CLIENT_ID") and os.environ.get("CF_ACCESS_CLIENT_SECRET") else None)
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"],
                  default_headers=cf)


def tol_f1(gold, pred, t=2):
    pred = [p for p in pred if 0 < p]
    if not pred or not gold:
        return 0.0
    p = sum(1 for i in pred if any(abs(i - j) <= t for j in gold)) / len(pred)
    r = sum(1 for j in gold if any(abs(i - j) <= t for i in pred)) / len(gold)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


class Caller:
    """모델별 캐시 + retry concurrent caller."""
    def __init__(self, client, model, mtok):
        self.client = client; self.model = model; self.mtok = mtok
        self.cp = REPO / "outputs/runs/_misc" / f"secom_cache_{model.replace('/', '_')}.jsonl"
        self.cache = {}; self.lock = threading.Lock(); self.errs = 0
        if self.cp.exists():
            for line in open(self.cp):
                try:
                    o = json.loads(line); self.cache[o["k"]] = o["t"]
                except Exception:
                    pass
        self.cf = open(self.cp, "a")

    def __call__(self, prompt, response_format=None):
        key = hashlib.md5((self.model + "||" + (response_format and "rf||" or "") + prompt).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        txt = ""; rf = response_format
        for attempt in range(4):
            try:
                kw = {"response_format": rf} if rf else {}
                r = self.client.chat.completions.create(model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.mtok, temperature=0.0,
                    extra_body={"reasoning": {"enabled": False}}, **kw)
                m = r.choices[0].message
                txt = m.content or getattr(m, "reasoning_content", None) or ""
                break
            except Exception:
                if rf is not None:        # response_format 미지원/실패 → prompt-only strict 로 강등 후 재시도
                    rf = None; continue
                if attempt == 3:
                    with self.lock:
                        self.errs += 1
                else:
                    time.sleep((2 ** attempt) * 0.5 + random.random())
        with self.lock:
            self.cache[key] = txt; self.cf.write(json.dumps({"k": key, "t": txt}) + "\n"); self.cf.flush()
        return txt


def parse_seg(txt, base_idx, n):
    """exchange-number JSONL → 경계(global turn). start_exchange_number(>0)=경계, base_idx 오프셋."""
    m = re.search(r"<segmentation>(.*?)</segmentation>", txt or "", re.S)
    body = m.group(1) if m else (txt or "")
    bounds = set()
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line); s = _seg_start(o)
            g = base_idx + s
            if 0 < g < n:
                bounds.add(g)
        except Exception:
            continue
    return bounds


def fmt_exchanges(turns, idxs):
    return "\n\n".join(f"[{LABEL} {j}]: ({turns[k]['speaker']}) {turns[k]['text']}"
                       for j, k in enumerate(idxs))


def _seg_start(o):
    """적응본 start_turn_number 또는 원본 start_exchange_number."""
    for key in ("start_turn_number", "start_exchange_number"):
        if key in o and o[key] != "":
            return int(o[key])
    raise KeyError


def extract_starts(txt):
    """응답 → segment 시작 인덱스 리스트. strict JSON object({"segments":[...]}) / SeCom <segmentation> JSONL 둘 다 처리."""
    t = (txt or "").strip()
    if t.startswith("```"):                       # code fence 제거
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t); t = re.sub(r"\n?```$", "", t).strip()
    # 1) strict JSON object
    try:
        o = json.loads(t)
        if isinstance(o, dict) and isinstance(o.get("segments"), list):
            out = []
            for s in o["segments"]:
                try:
                    out.append(_seg_start(s))
                except Exception:
                    continue
            return out
    except Exception:
        pass
    # 2) <segmentation> JSONL (verbatim / fallback)
    m = re.search(r"<segmentation>(.*?)</segmentation>", txt or "", re.S)
    body = m.group(1) if m else (txt or "")
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(_seg_start(json.loads(line)))
        except Exception:
            continue
    return out


def load_meetings(subset):
    out = []
    for mid in sorted(json.load(open(subset))):
        d = json.load(open(TOPIC / f"{mid}.json")); bt = list(d["bnd_top"]); n = len(bt); bt[-1] = 0
        out.append((mid, d["turns"], n, [i for i, b in enumerate(bt) if b == 1]))
    return out


def metrics(gold, pred, n):
    yt = [1 if i in set(gold) else 0 for i in range(n)]
    yp = [1 if i in set(pred) else 0 for i in range(n)]
    f2 = tol_f1(gold, sorted(pred)); pk, wd = official_pk_wd(yt, yp)
    return f2, f1_score(yt, yp, zero_division=0), pk, wd, 0.5 * f2 + 0.25 * (1 - pk) + 0.25 * (1 - wd)


def seg_checkpoints(turns, n, B):
    """B-mode 누적 checkpoint. 반환 [(k_prev, k_i)] — prefix=range(0,k_i), emit 는 새 구간 [k_prev,k_i)."""
    t0 = turns[0]["start"]; t_last = turns[-1]["start"]
    ks = []
    T = t0 + B
    while T <= t_last:
        ks.append(sum(1 for j in range(n) if turns[j]["start"] < T)); T += B
    ks.append(n)                              # 마지막 prefix = 전체
    cuts = []; prev = 0
    for k in ks:
        if k > prev and k >= 2:
            cuts.append((prev, k)); prev = k
    return cuts


def run_segment(meetings, caller, B, offs, workers, context="B"):
    """B='full'=offline oracle(전체 1콜). 버퍼: context 'B'=누적 prefix(채택, §1B) / 'A'=독립 window(구).
    채택 = B(누적). 콜당 입력=prefix[0,k_i], emit=새 구간 경계만, commit(과거 미수정)."""
    if B == "full" or context == "A":
        return _run_segment_windows(meetings, caller, B, offs, workers)
    # --- B-mode: 누적 prefix ---
    plan = []; cuts_by = {}
    for mid, turns, n, gold in meetings:
        cuts = seg_checkpoints(turns, n, B); cuts_by[mid] = cuts
        for kp, ki in cuts:
            plan.append((mid, ki))            # prefix=range(0,ki)
    uniq = {}
    for mid, ki in set(plan):
        turns = next(t for m, t, n, g in meetings if m == mid)
        uniq[(mid, ki)] = P_SEG.format(text_to_be_segmented=fmt_exchanges(turns, list(range(ki))))
    resp = {}
    rf = SEG_RF if STRICT else None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(caller, p, rf): k for k, p in uniq.items()}
        for fut in as_completed(futs):
            resp[futs[fut]] = fut.result()
    rows = []
    for mid, turns, n, gold in meetings:
        pred = set()
        for kp, ki in cuts_by[mid]:           # prefix 인덱스=global, emit 는 새 구간 [kp,ki) 의 경계만(commit)
            for s in extract_starts(resp.get((mid, ki), "")):
                if 0 < s < n and kp <= s < ki:
                    pred.add(s)
        rows.append(metrics(gold, pred, n) + (len(pred), len(gold)))
    return rows


def _run_segment_windows(meetings, caller, B, offs, workers):
    """구 A-mode: B='full' 전체 1콜, 아니면 B초 독립 window(맥락 없음). 참고/비교용."""
    plan = []; win_by = {}
    for mid, turns, n, gold in meetings:
        wins = []
        if B == "full":
            wins.append((0, list(range(n))))
        else:
            t0 = turns[0]["start"]; t_last = turns[-1]["start"]
            for off in offs:
                w_start = t0; w_end = t0 + (off * B if off > 0 else B)
                while w_start <= t_last:
                    w = [k for k in range(n) if w_start <= turns[k]["start"] < w_end]
                    if len(w) >= 2:
                        wins.append((w[0], w))
                    w_start = w_end; w_end += B
        win_by[mid] = wins
        for base, w in wins:
            plan.append((mid, base, tuple(w)))
    uniq = {}
    for mid, base, w in plan:
        turns = next(t for m, t, n, g in meetings if m == mid)
        uniq[(mid, base, w)] = P_SEG.format(text_to_be_segmented=fmt_exchanges(turns, w))
    resp = {}
    rf = SEG_RF if STRICT else None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(caller, p, rf): k for k, p in uniq.items()}
        for fut in as_completed(futs):
            resp[futs[fut]] = fut.result()
    rows = []
    for mid, turns, n, gold in meetings:
        pred = set()
        for base, w in win_by[mid]:
            for s in extract_starts(resp.get((mid, base, tuple(w)), "")):
                if 0 < s < len(w):
                    pred.add(w[s])
        rows.append(metrics(gold, pred, n) + (len(pred), len(gold)))
    return rows


def run_incremental(meetings, caller, workers):
    """턴마다 Yes/No. 미팅 내부 순차, 미팅 병렬."""
    def one(meeting):
        mid, turns, n, gold = meeting
        seg_start = 0; pred = set()
        for t in range(1, n):
            prev = fmt_exchanges(turns, list(range(seg_start, t)))
            new = f"({turns[t]['speaker']}) {turns[t]['text']}"
            ans = caller(P_INC.format(new_turn=new, prev_session=prev)).strip().lower()
            if ans.startswith("no") or ("no" in ans[:5] and "yes" not in ans[:5]):
                pred.add(t); seg_start = t
        return metrics(gold, pred, n) + (len(pred), len(gold))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, meetings):
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["incremental", "segment"], required=True)
    ap.add_argument("--subset", default=str(REPO / "outputs/runs/_misc/ami_subset.json"))
    ap.add_argument("--model", default="openrouter/qwen/qwen3.5-27b")
    ap.add_argument("--buffer", default="full", help="segment 모드: 'full' 또는 B초")
    ap.add_argument("--offsets", default="0,0.5")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prompt", choices=["baseline", "verbatim"], default="baseline",
                    help="baseline=우리 적응본(DTS+AMI 공통, turn 기반) / verbatim=SeCom 원본(user-bot exchange)")
    ap.add_argument("--context", choices=["B", "A"], default="B",
                    help="버퍼드 LLM 컨텍스트(§1B): B=누적 prefix(채택) / A=독립 window(구, 참고용)")
    args = ap.parse_args()
    load_prompts(args.prompt)
    meetings = load_meetings(args.subset)
    client = make_client()
    mtok = 16 if args.mode == "incremental" else 1024
    caller = Caller(client, args.model, mtok)
    t0 = time.perf_counter()
    if args.mode == "incremental":
        rows = run_incremental(meetings, caller, args.workers)
        tag = f"SeCom-incremental [{args.prompt}]"
    else:
        B = "full" if args.buffer == "full" else int(args.buffer)
        offs = [float(x) for x in args.offsets.split(",")]
        rows = run_segment(meetings, caller, B, offs, args.workers, args.context)
        tag = f"SeCom-segment buffer={args.buffer} ctx={args.context} [{args.prompt}]"
    a = np.array([r[:5] for r in rows]); npred = sum(r[5] for r in rows); ngold = sum(r[6] for r in rows)
    print(f"=== {tag} | {args.model} | {len(meetings)}미팅 ===")
    print(f"  ±2F1 {a[:,0].mean():.3f} | exactF1 {a[:,1].mean():.3f} | Pk {a[:,2].mean():.3f} | "
          f"WD {a[:,3].mean():.3f} | Score {a[:,4].mean():.3f} | pred {npred}/{ngold} | "
          f"err {caller.errs} | {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
