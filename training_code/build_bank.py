"""Dựng ngân hàng ví dụ `src/socai/v2/bank.json` từ bộ TRAIN (không dùng public/private).

Cách làm: LLM (cùng mô hình lúc suy luận) viết biểu thức cho từng câu train; biểu thức được chạy và chấm với
đáp án chuẩn; CHỈ biểu thức ra đúng mới vào ngân hàng. Khi làm câu của shop X, ví dụ truy hồi không lấy từ X.
Chạy nhiều vòng thì vòng sau có nhiều ví dụ hơn nên làm được thêm câu.

    python training_code/build_bank.py --train ../train --out src/socai/v2/bank.json \
        --llm http://127.0.0.1:8088/v1 --exclude-biz biz_105,biz_111 [--seed kết_quả_cũ.json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from socai.eval.scorer import judge, load_jsonl
from socai.llm.client import VllmClient
from socai.nlq.plan import _series_periods
from socai.profile.business import load_business
from socai.profile.llm_tagger import tag_with_llm
from socai.rules.calendar import LunarCalendar
from socai.rules.timeexpr import Resolver
from socai.semantic.canonical import build
from socai.v2.algebra import Ctx, V2Error, execute
from socai.v2.bank import Bank
from socai.v2.brain import think
from socai.v2.views import build_views
from socai.exec.metrics import Unsupported


def verdict(expr: dict, views: dict, q: dict, ref: dict, cal) -> str:
    today = date.fromisoformat(q["as_of_date"])
    r = Resolver(today, cal)
    try:
        res = execute(expr, Ctx(views, r.resolve(q["question"]), r.resolve, _series_periods, today=today))
    except (V2Error, Unsupported):
        return "abstain"
    except Exception:  # noqa: BLE001 — biểu thức cũ không còn chạy được thì bỏ
        return "abstain"
    return judge({"type": res.type, "answer": res.value}, ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--llm", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--exclude-biz", default="")
    ap.add_argument("--only-biz", default="")
    ap.add_argument("--seed", nargs="*", default=[], help="file kết quả cũ có trường id, expr: kiểm lại rồi nạp")
    ap.add_argument("--limit", type=int, default=0, help="số câu tối đa mỗi shop (0 = tất cả)")
    ap.add_argument("--no-llm", action="store_true", help="chỉ kiểm lại seed/ngân hàng hiện có")
    a = ap.parse_args()

    Q, A = load_jsonl(a.train / "questions.jsonl"), load_jsonl(a.train / "answers.jsonl")
    cal = LunarCalendar.load(a.train / "calendar.csv")
    llm = VllmClient(base_url=a.llm, timeout=300)
    skip = set(filter(None, a.exclude_biz.split(",")))
    only = set(filter(None, a.only_biz.split(",")))
    old = Bank.load(a.out)
    seeds: dict[str, list[dict]] = {}
    for s in old.shots:
        seeds.setdefault(s.id, []).append(s.expr)
    for f in a.seed:
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            if r.get("expr") and r.get("id"):
                seeds.setdefault(r["id"], []).append(r["expr"])
    bank, done, log = Bank(), set(), []
    pool = old  # ví dụ để truy hồi: ngân hàng vòng trước + ví dụ mới của vòng này
    known = {s.id for s in old.shots}
    stat = {"seed": 0, "llm": 0, "fail": 0}
    for biz_dir in sorted(p for p in (a.train / "data").iterdir() if p.is_dir()):
        b = biz_dir.name
        if b in skip or (only and b not in only):
            for s in old.shots:  # shop không chạy lại: giữ nguyên ví dụ cũ
                if s.biz == b and b not in skip:
                    bank.add(s.q, s.expr, s.biz, s.id, reindex=False)
            continue
        biz = load_business(biz_dir, a.cache)
        if not a.no_llm:
            try:
                tag_with_llm(biz, llm)  # cùng cách gán nghĩa cột như lúc suy luận
            except Exception as e:  # noqa: BLE001
                print("tag lỗi", b, e, flush=True)
        views = build_views(biz, build(biz))
        qs = [q for q in Q.values() if q["business_id"] == b]
        qs = qs[:a.limit] if a.limit else qs
        t0, n_ok = time.time(), 0
        for q in qs:
            ref = A[q["id"]]
            hit = next((e for e in seeds.get(q["id"], []) if verdict(e, views, q, ref, cal) == "correct"), None)
            src = "seed"
            if hit is None and not a.no_llm:
                today = date.fromisoformat(q["as_of_date"])
                r = Resolver(today, cal)
                ctx = Ctx(views, r.resolve(q["question"]), r.resolve, _series_periods, today=today)
                th = think(llm, views, q["question"], ctx, bank=pool, exclude_biz=b)
                if th.result is not None and judge({"type": th.result.type, "answer": th.result.value}, ref) == "correct":
                    hit, src = th.expr, "llm"
                else:
                    log.append({"id": q["id"], "biz": b, "q": q["question"], "ref": ref, "expr": th.expr,
                                "problem": th.problem,
                                "got": None if th.result is None else th.result.value})
            if hit is not None:
                bank.add(q["question"], hit, b, q["id"], reindex=False)
                if q["id"] not in known:
                    pool.add(q["question"], hit, b, q["id"], reindex=False)
                stat[src] += 1
                n_ok += 1
            else:
                stat["fail"] += 1
            done.add(q["id"])
        bank.reindex()
        pool.reindex()
        bank.save(a.out)
        Path(str(a.out) + ".fail.json").write_text(json.dumps(log, ensure_ascii=False, indent=0, default=str),
                                                   encoding="utf-8")
        print(f"{b}: {n_ok}/{len(qs)} vào ngân hàng ({time.time() - t0:.0f}s) | tổng {stat}", flush=True)
    print("XONG", stat, "ví dụ:", len(bank.shots))


if __name__ == "__main__":
    sys.exit(main())
