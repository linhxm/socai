"""Đo hold-out đúng cách chạy thật (đường luật → v2 → chính sách), có hạn giờ; dùng cho mọi mô hình.

    PYTHONPATH=src python scripts/eval_holdout.py --train ../train --llm http://127.0.0.1:8088/v1 \
        --out runs/h_14b_bank.json --n-per-shop 20 --deadline-min 60 [--no-bank]

Ghi sau MỖI câu (dừng giữa chừng vẫn còn kết quả). Mỗi dòng có cả đáp án cuối (sau ngưỡng) lẫn phán xét của
ứng viên thô, để so mô hình mà không phụ thuộc ngưỡng hiệu chỉnh cho 4B. `--ids` dùng đúng bộ câu của lần khác.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socai.agent import Agent, AgentConfig  # noqa: E402
from socai.answer.decide import Policy  # noqa: E402
from socai.eval.scorer import judge, load_jsonl  # noqa: E402
from socai.llm.client import VllmClient  # noqa: E402
from socai.rules.calendar import LunarCalendar  # noqa: E402
from socai.rules.rulebook import Rulebook  # noqa: E402
from socai.v2.bank import Bank  # noqa: E402


def pick(Q: dict, shops: list[str], n: int, seed: int = 0) -> list[dict]:
    """n câu mỗi shop, cố định theo seed; xếp xen kẽ các shop để dừng sớm vẫn đủ đại diện."""
    per = {}
    for b in shops:
        qs = [q for q in Q.values() if q["business_id"] == b]
        per[b] = random.Random(f"{seed}:{b}").sample(qs, min(n, len(qs))) if n else qs
    return [q for b in shops for q in per[b]]


def summary(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in ("final", "raw"):
        c = collections.Counter(r[key] for r in rows)
        out[key] = {**c, "PA": round((c["correct"] - c["wrong"]) / max(1, len(rows)), 4)}
    src = collections.defaultdict(collections.Counter)
    for r in rows:
        src[r["src"] or "none"][r["raw"]] += 1
    out["by_source"] = {k: dict(v) for k, v in src.items()}
    out["sec_per_q"] = round(sum(r["sec"] for r in rows) / max(1, len(rows)), 2)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--llm", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--shops", default=",".join(json.loads((ROOT / "configs" / "holdout.json").read_text(encoding="utf-8"))["holdout"]))
    ap.add_argument("--n-per-shop", type=int, default=20)
    ap.add_argument("--ids", type=Path, default=None, help="file JSON danh sách id câu (thay cho --n-per-shop)")
    ap.add_argument("--deadline-min", type=float, default=0, help="dừng nhận câu mới sau ngần ấy phút (0 = không)")
    ap.add_argument("--no-bank", action="store_true", help="không truy hồi ví dụ: chỉ ví dụ viết tay (đo câu kiểu mới)")
    ap.add_argument("--no-second", action="store_true")
    a = ap.parse_args()

    t_end = time.time() + a.deadline_min * 60 if a.deadline_min else float("inf")
    Q, A = load_jsonl(a.train / "questions.jsonl"), load_jsonl(a.train / "answers.jsonl")
    shops = a.shops.split(",")
    qs = [Q[i] for i in json.loads(a.ids.read_text(encoding="utf-8"))] if a.ids else pick(Q, shops, a.n_per_shop)
    bank = None if a.no_bank else Bank.load(ROOT / "src" / "socai" / "v2" / "bank.json")
    ag = Agent(Rulebook.load(a.train / "rulebook.md"), LunarCalendar.load(a.train / "calendar.csv"),
               Policy.load(ROOT / "src" / "socai" / "calibration.json"), VllmClient(base_url=a.llm, timeout=600),
               AgentConfig(use_llm=True, use_v2=True, second_opinion=not (a.no_second or a.no_bank)), a.cache,
               a.train / "calendar.csv", bank=bank)
    rows, sessions, t0 = [], {}, time.time()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for q in qs:
        if time.time() > t_end:
            print("HẾT GIỜ: dừng nhận câu mới", flush=True)
            break
        b = q["business_id"]
        if b not in sessions:
            ts = time.time()
            sessions[b] = ag.open(a.train / "data" / b)
            print(f"mở {b} ({time.time() - ts:.0f}s)", flush=True)
        t = time.time()
        sub, log = ag.answer(sessions[b], q["id"], q["question"], q["as_of_date"])
        if any(k in n for n in log.get("notes") or [] for k in ("URLError", "Connection", "Remote", "timed out")):
            print("LLM FAIL", log["notes"], flush=True)
            break
        d = log.get("decision") or {}
        cands = d.get("candidates") or []
        fin = None if sub["type"] == "abstain" else {k: v for k, v in sub.items() if k != "id"}
        best = next((c for c in cands if c["source"] == d.get("source")), cands[0] if cands else None)
        rows.append({"id": q["id"], "biz": b, "q": q["question"], "src": d.get("source"), "p": d.get("p"),
                     "final": judge(fin, A[q["id"]]), "raw": judge(best["answer"], A[q["id"]]) if best else "abstain",
                     "sig": (log.get("v2") or {}).get("sig"), "expr": (log.get("v2") or {}).get("expr"),
                     "sec": round(time.time() - t, 2)})
        a.out.write_text(json.dumps({"summary": summary(rows), "rows": rows}, ensure_ascii=False, indent=0),
                         encoding="utf-8")
        r = rows[-1]
        print(f"[{len(rows)}/{len(qs)} {time.time() - t0:.0f}s] {r['raw']:8s} {r['final']:8s} {r['sec']:5.1f}s "
              f"{q['question'][:60]}", flush=True)
    for s in sessions.values():
        ag.close(s)
    print("TỔNG", json.dumps(summary(rows), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
