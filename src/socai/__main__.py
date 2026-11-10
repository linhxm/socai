"""CLI SổCái.

  python -m socai predict  --data /data --out /result [--no-llm] [--limit N]
  python -m socai eval     --split train --out runs/train [--no-llm] [--calibrate configs/calibration.json]
  python -m socai profile  --biz train/data/biz_101
  python -m socai rulebook-diff train/rulebook.md public_test/rulebook.md
  python -m socai synth    --out /tmp/synth --n-biz 3 [--render-images]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from socai.agent import AgentConfig
from socai.runner import RunConfig, run


def _llm(args) -> object | None:
    if args.no_llm:
        return None
    from socai.llm.client import VllmClient

    c = VllmClient(base_url=os.environ.get("SOCAI_LLM_URL", "http://127.0.0.1:8000/v1"),
                   model=os.environ.get("SOCAI_LLM_MODEL", "local"))
    if not c.ready():
        print("[socai] không kết nối được vLLM → chạy chỉ đường nhanh", file=sys.stderr)
        return None
    return c


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="socai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path)
        p.add_argument("--split", type=Path)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--no-llm", action="store_true")
        p.add_argument("--limit", type=int)
        p.add_argument("--cache", type=Path, default=Path(os.environ.get("SOCAI_CACHE", "/tmp/socai_cache")))
        p.add_argument("--policy", type=Path, default=Path(__file__).parent / "calibration.json")
        p.add_argument("--calibrate", type=Path)
        p.add_argument("--verify-fastpath", action="store_true")
        p.add_argument("--thinking", action="store_true")
    p = sub.add_parser("profile")
    p.add_argument("--biz", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p = sub.add_parser("rulebook-diff")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p = sub.add_parser("synth")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-biz", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-images", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd in ("predict", "eval"):
        t0 = time.time()
        data = args.data or args.split
        cfg = AgentConfig(use_llm=not args.no_llm, verify_fastpath=args.verify_fastpath, thinking=args.thinking)
        stats = run(data, args.out, _llm(args), cfg, RunConfig(limit=args.limit), args.cache,
                    args.policy if args.policy.exists() else None, t0)
        print(json.dumps(stats, ensure_ascii=False))
        if args.cmd == "eval":
            from socai.eval.evaluate import evaluate

            rep = evaluate(data / "answers.jsonl", args.out / "logs" / "log.jsonl")
            (args.out / "eval_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            print(json.dumps({k: rep[k] for k in ("n", "PA_policy", "final", "PA_lbo_simulated")}, ensure_ascii=False))
            if args.calibrate:
                args.calibrate.write_text(json.dumps(rep["calibration"], ensure_ascii=False, indent=1), encoding="utf-8")
    elif args.cmd == "profile":
        from socai.profile.business import load_business
        from socai.semantic.canonical import build

        biz = load_business(args.biz, args.cache)
        can = build(biz)
        print(json.dumps({**biz.summary(), "canonical_ok": can.ok, "issues": can.issues,
                          "data_start": str(can.data_start)}, ensure_ascii=False, indent=1))
    elif args.cmd == "rulebook-diff":
        from socai.rules.rulebook import Rulebook, diff

        print(json.dumps(diff(Rulebook.load(args.a), Rulebook.load(args.b)), ensure_ascii=False, indent=1))
    elif args.cmd == "synth":
        from socai.synth.world import generate_split

        print(json.dumps(generate_split(args.out, args.n_biz, args.seed, args.render_images), ensure_ascii=False))


if __name__ == "__main__":
    main()
