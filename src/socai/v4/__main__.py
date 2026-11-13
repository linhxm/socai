"""python -m socai.v4 predict --data ../public_test --out runs/v4 --llm http://127.0.0.1:8088/v1
   python -m socai.v4 eval --pred runs/v4/submission.jsonl --ref ../submission.jsonl [--questions ../public_test/questions.jsonl]
   python -m socai.v4 shop --data ../public_test --shop dev_001 [--llm URL]   # chỉ dựng L1 + L2, in danh mục bảng
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from socai.eval.scorer import judge, load_jsonl, score
from socai.llm.client import VllmClient


def _llm(a):
    if not a.llm:
        return None
    c = VllmClient(base_url=a.llm, model=a.model)
    if not c.ready():
        sys.exit(f"không kết nối được LLM ở {a.llm}")
    return c


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="socai.v4")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llm", default=None)
    p.add_argument("--model", default="local")
    p.add_argument("--shops", default=None, help="dev_001,dev_002")
    p.add_argument("--limit", type=int, default=None, help="số câu tối đa mỗi shop")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--enrich-rules", action="store_true")
    p.add_argument("--cache", type=Path, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--pred", type=Path, required=True)
    e.add_argument("--ref", type=Path, required=True)
    e.add_argument("--questions", type=Path, default=None)
    s = sub.add_parser("shop")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--shop", required=True)
    s.add_argument("--llm", default=None)
    s.add_argument("--model", default="local")
    a = ap.parse_args(argv)

    if a.cmd == "predict":
        from socai.v4.pipeline import Config, run
        cfg = Config(n_samples=a.samples, thinking=a.thinking, enrich_rules=a.enrich_rules)
        st = run(a.data, a.out, _llm(a), cfg, a.shops.split(",") if a.shops else None, a.limit, a.cache)
        print(json.dumps(st, ensure_ascii=False))
    elif a.cmd == "eval":
        preds, refs = load_jsonl(a.pred), load_jsonl(a.ref)
        refs = {k: v for k, v in refs.items() if k in preds}
        r = score(preds, refs)
        print({k: v for k, v in r.items() if k != "verdicts"})
        if a.questions:
            qs = load_jsonl(a.questions)
            by: dict = defaultdict(Counter)
            for qid, v in r["verdicts"].items():
                by[qs[qid]["business_id"]][v] += 1
            for b, c in sorted(by.items()):
                n = sum(c.values())
                print(f"{b}: đúng {c['correct']} sai {c['wrong']} trống {c['abstain']} PA {(c['correct'] - c['wrong']) / n:.3f}")
            bt: dict = defaultdict(Counter)
            for qid, v in r["verdicts"].items():
                bt[refs[qid]["type"]][v] += 1
            for t, c in sorted(bt.items()):
                print(f"  {t}: {dict(c)}")
    elif a.cmd == "shop":
        from socai.v4.l1 import run_l1
        from socai.v4.l2 import build_model, catalog
        from socai.v4.l3 import load_rules
        llm = _llm(a)
        rules = load_rules(a.data / "rulebook.md")
        trace: list = []
        sess, tables = run_l1(a.data / "data" / a.shop, llm, trace)
        m = build_model(a.shop, tables, sess.readme.intro, rules.industries, llm, trace)
        print(f"ngành: {m.industry} · dữ liệu: {m.date_range}")
        print(catalog(m))


if __name__ == "__main__":
    main()
