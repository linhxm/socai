"""Phát lại ngân hàng ví dụ: chạy lại từng biểu thức ĐÃ KIỂM CHỨNG qua đúng đường chạy thật của agent (LLM thay bằng
bộ trả lời ghi sẵn), so với đáp án chuẩn. Không cần GPU. Dùng làm phép thử hồi quy cho mọi sửa đổi tầng dữ liệu/view:
trước khi sửa mọi biểu thức đều ra đúng, sau khi sửa câu nào hỏng là sửa đó làm hỏng.

    PYTHONPATH=src python scripts/replay_bank.py --train ../train --cache /tmp/cache [--shops biz_121,biz_122] [--out f.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socai.agent import Agent, AgentConfig  # noqa: E402
from socai.answer.decide import Policy  # noqa: E402
from socai.eval.scorer import judge, load_jsonl  # noqa: E402
from socai.llm.client import FakeLLM  # noqa: E402
from socai.rules.calendar import LunarCalendar  # noqa: E402
from socai.rules.rulebook import Rulebook  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--bank", type=Path, default=ROOT / "src" / "socai" / "v2" / "bank.json")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--shops", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    Q, A = load_jsonl(a.train / "questions.jsonl"), load_jsonl(a.train / "answers.jsonl")
    shots = json.loads(a.bank.read_text(encoding="utf-8"))
    want = set(filter(None, a.shops.split(",")))
    by_q = {s["q"]: s["expr"] for s in shots}
    llm = FakeLLM(lambda msgs: json.dumps(by_q.get(msgs[1]["content"], {"op": "none"}), ensure_ascii=False))
    ag = Agent(Rulebook.load(a.train / "rulebook.md"), LunarCalendar.load(a.train / "calendar.csv"),
               Policy.load(ROOT / "src" / "socai" / "calibration.json"), llm,
               AgentConfig(use_llm=True, use_v2=True, second_opinion=False, verify_fastpath=True, max_repairs=0),
               a.cache, a.train / "calendar.csv", bank=None)
    by_biz = collections.defaultdict(list)
    for s in shots:
        if s["id"] in Q and (not want or s["biz"] in want):
            by_biz[s["biz"]].append(s)
    rows, t0 = [], time.time()
    for b in sorted(by_biz):
        sess = ag.open(a.train / "data" / b)
        c = collections.Counter()
        for s in by_biz[b]:
            q = Q[s["id"]]
            _, log = ag.answer(sess, q["id"], q["question"], q["as_of_date"])
            cand = next((x for x in (log.get("decision") or {}).get("candidates") or []
                         if x["source"].startswith("v2")), None)
            v = judge(cand["answer"], A[q["id"]]) if cand else "abstain"
            c[v] += 1
            if v != "correct":
                rows.append({"id": q["id"], "biz": b, "q": q["question"], "verdict": v, "expr": s["expr"],
                             "got": cand and cand["answer"], "gold": A[q["id"]].get("answer"),
                             "problem": (log.get("v2") or {}).get("problem")})
        ag.close(sess)
        print(f"{b}: {dict(c)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"HỎNG {len(rows)} biểu thức", flush=True)
    for r in rows[:30]:
        print(f"  {r['biz']} {r['verdict']}: {r['q'][:70]} | got {json.dumps(r['got'], ensure_ascii=False)[:60]} "
              f"| gold {json.dumps(r['gold'], ensure_ascii=False)[:50]} | {r['problem'] or ''}"[:300], flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
