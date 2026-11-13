"""CLI v5.

  python -m socai.v5 predict   --data /data --out /result --llm http://127.0.0.1:8000/v1[,http://…:8001/v1]
  python -m socai.v5 eval      --pred runs/x/submission.jsonl --ref train/answers.jsonl [--questions train/questions.jsonl]
  python -m socai.v5 bank      --log runs/x/logs/log.jsonl --data train          # ví dụ đã kiểm chứng (chỉ TRAIN)
  python -m socai.v5 calibrate --log runs/x/logs/log.jsonl --data train          # xác suất đúng theo nhóm tin cậy
  python -m socai.v5 shop      --data train --shop biz_101                       # xem L1 + L2 của một shop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def _llms(urls: str | None, model: str):
    from socai.llm.client import VllmClient

    out = []
    for u in (urls or "").split(","):
        if u.strip():
            c = VllmClient(base_url=u.strip(), model=model)
            if c.ready():
                out.append(c)
            else:  # vẫn chạy để ghi đủ submission.jsonl (câu không có LLM thì bỏ trống)
                print(f"[v5] không kết nối được LLM ở {u}, bỏ server này", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="socai.v5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llm", default=None, help="một hoặc nhiều URL, phẩy ngăn cách (mỗi GPU một server)")
    p.add_argument("--model", default="local")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--shops", default=None)
    p.add_argument("--limit", type=int, default=None, help="số câu tối đa mỗi shop")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--policy", type=Path, default=HERE / "calibration.json")
    p.add_argument("--bank", type=Path, default=HERE / "bank.json")
    p.add_argument("--bank-exclude-own", action="store_true", help="đo trên train: không lấy ví dụ của chính shop")
    p.add_argument("--budget-min", type=float, default=5.4 * 60)
    p.add_argument("--schema-budget", type=int, default=9000, help="ký tự lược đồ; ~5000 cho ngữ cảnh 8K token")
    p.add_argument("--base", type=Path, default=None,
                   help="submission.jsonl của hệ khác (v2): giữ câu nó đã trả lời, v5 chỉ làm câu nó bỏ trống")
    e = sub.add_parser("eval")
    e.add_argument("--pred", type=Path, required=True)
    e.add_argument("--ref", type=Path, required=True)
    e.add_argument("--questions", type=Path, default=None)
    for name in ("bank", "calibrate"):
        b = sub.add_parser(name)
        b.add_argument("--log", type=Path, required=True)
        b.add_argument("--data", type=Path, required=True, help="thư mục TRAIN (có questions.jsonl, answers.jsonl)")
        b.add_argument("--out", type=Path, default=HERE / ("bank.json" if name == "bank" else "calibration.json"))
    s = sub.add_parser("shop")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--shop", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "predict":
        from socai.v5.pipeline import Config, run

        cfg = Config(n_samples=a.samples, thinking=a.thinking, shops=a.shops.split(",") if a.shops else None,
                     limit=a.limit, bank_exclude_own=a.bank_exclude_own, time_budget_s=a.budget_min * 60,
                     schema_budget=a.schema_budget,
                     base={r["id"]: r for r in map(json.loads, a.base.read_text(encoding="utf-8").splitlines()) if r}
                     if a.base and a.base.exists() else None)
        st = run(a.data, a.out, _llms(a.llm, a.model), cfg, a.cache, a.policy, a.bank, time.time())
        print(json.dumps(st, ensure_ascii=False))
    elif a.cmd == "eval":
        from socai.eval.scorer import load_jsonl, score

        preds, refs = load_jsonl(a.pred), load_jsonl(a.ref)
        refs = {k: v for k, v in refs.items() if k in preds}
        r = score(preds, refs)
        print(json.dumps({k: v for k, v in r.items() if k != "verdicts"}, ensure_ascii=False))
        if a.questions:
            qs = load_jsonl(a.questions)
            by: dict = defaultdict(Counter)
            for qid, v in r["verdicts"].items():
                by[qs[qid]["business_id"]][v] += 1
            for b, c in sorted(by.items()):
                n = sum(c.values())
                print(f"{b}: đúng {c['correct']} sai {c['wrong']} trống {c['abstain']} "
                      f"PA {(c['correct'] - c['wrong']) / n:.3f}")
        bt: dict = defaultdict(Counter)
        for qid, v in r["verdicts"].items():
            bt[refs[qid]["type"]][v] += 1
        for t, c in sorted(bt.items()):
            print(f"  {t}: {dict(c)}")
    elif a.cmd == "bank":
        from socai.v5.bank import build

        n = build(a.log, a.data / "questions.jsonl", a.data / "answers.jsonl", a.out)
        print(f"ngân hàng có {n} ví dụ → {a.out}")
    elif a.cmd == "calibrate":
        from socai.v5.l6_decide import calibrate

        print(json.dumps(calibrate(a.log, a.data / "answers.jsonl", a.out), ensure_ascii=False, indent=1))
    elif a.cmd == "shop":
        from socai.v5.l1_read import read_shop
        from socai.v5.l2_model import build_shop, render_table
        from socai.v5.l3_rules import load_rules

        rules = load_rules(a.data / "rulebook.md")
        readme, tables = read_shop(a.data / "data" / a.shop)
        shop = build_shop(a.shop, readme, tables, rules.industries)
        print(f"ngành: {shop.industry} · dữ liệu: {shop.date_range} · {len(shop.index)} khoá giá trị")
        for t in shop.tables.values():
            print(render_table(t), "\n")


if __name__ == "__main__":
    main()
