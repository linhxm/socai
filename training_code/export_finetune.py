"""Xuất dữ liệu fine-tune từ ngân hàng ví dụ đã kiểm chứng (`src/socai/v2/bank.json`, chỉ từ TRAIN).

Mỗi dòng JSONL là một hội thoại ĐÚNG như lúc suy luận (`brain.messages`): system (luật + bảng của shop + ví dụ
truy hồi) → user (câu hỏi) → assistant (biểu thức đã chạy ra đúng đáp án chuẩn). Thiết kế:
- Cân bằng: tối đa `--per-skeleton` mẫu mỗi khung biểu thức (khung phổ biến nhất có 203 mẫu), chọn xoay vòng
  theo shop. Mẫu bị bỏ vẫn nằm trong ngân hàng để làm ví dụ truy hồi.
- Ba kiểu prompt, gán cố định theo id câu:
  `bank`   ví dụ truy hồi từ shop khác, như lúc chạy thật;
  `novel`  như `bank` nhưng bỏ mọi ví dụ CÙNG DẠNG với đáp án: cùng cấu trúc + phép + kiểu lấy ra, bất kể bảng,
           cột, kỳ, tên (mô phỏng câu kiểu mới ở private);
  `static` chỉ ví dụ viết tay (như lúc chạy không có ngân hàng).
- Đích được chuẩn hoá bằng `lint` (như lúc chạy); `Bank` đã ghi `$ky_so` ngoài vế so sánh thành `$ky`.
- Chia train/val THEO SHOP. 6 shop hold-out không có trong ngân hàng nên không bao giờ vào đây.

    PYTHONPATH=src python training_code/export_finetune.py --train ../train --bank src/socai/v2/bank.json \
        --out training_code/data/finetune --val-biz biz_109,biz_124
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from socai.profile.business import load_business
from socai.semantic.canonical import build
from socai.v2.bank import Bank, skeleton
from socai.v2.brain import EXAMPLES, lint, messages, pick_views
from socai.v2.views import build_views

VARIANTS = (("bank", 0.4), ("novel", 0.3), ("static", 0.3))


def variant(qid: str, seed: int) -> str:
    x = random.Random(f"{seed}:{qid}").random()
    for name, w in VARIANTS:
        if x < w:
            return name
        x -= w
    return VARIANTS[-1][0]


KEEP = ("op", "out", "agg")  # giữ khi so dạng; agg chỉ giữ tên phép, không giữ cột


def shape(e: object, key: str = "") -> str:
    """Dạng biểu thức: cấu trúc + phép + kiểu lấy ra; che bảng, cột, kỳ, tên, số."""
    if isinstance(e, dict):
        return "{" + ",".join(f"{k}:{shape(v, k)}" for k, v in sorted(e.items()) if k != "nghi") + "}"
    if isinstance(e, list):
        if key == "agg":
            return str(e[0]) if e else "_"
        if key in ("where", "having"):
            return f"[{len(e)}]"
        return "[" + ",".join(shape(x) for x in e) + "]"
    return str(e) if key in KEEP else "_"


def balanced(shots: list, cap: int) -> list:
    """Tối đa `cap` mẫu mỗi khung, lấy xoay vòng giữa các shop để mỗi khung có nhiều kiểu bảng."""
    by_sk: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in sorted(shots, key=lambda s: s.id):
        by_sk[skeleton(s.expr)][s.biz].append(s)
    out = []
    for sk in sorted(by_sk):
        queues = [list(v) for _, v in sorted(by_sk[sk].items())]
        took = 0
        while took < cap and any(queues):
            for qu in queues:
                if qu and took < cap:
                    out.append(qu.pop(0))
                    took += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--val-biz", default="")
    ap.add_argument("--per-skeleton", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bank = Bank.load(a.bank)
    for q0, e0 in EXAMPLES:
        bank.add(q0, e0, "_static", "static", reindex=False)
    bank.reindex()
    bank.seeded = True
    static = Bank()
    for q0, e0 in EXAMPLES:
        static.add(q0, e0, "_static", "static", reindex=False)
    static.reindex()

    val = set(filter(None, a.val_biz.split(",")))
    picked = balanced([s for s in bank.shots if s.biz != "_static"], a.per_skeleton)
    by_biz: dict[str, list] = defaultdict(list)
    for s in picked:
        by_biz[s.biz].append(s)
    a.out.mkdir(parents=True, exist_ok=True)
    files = {k: (a.out / f"{k}.jsonl").open("w", encoding="utf-8") for k in ("train", "val")}
    n, kinds, chars = Counter(), Counter(), []
    for b, shots in sorted(by_biz.items()):
        biz = load_business(a.train / "data" / b, a.cache)
        views = build_views(biz, build(biz))
        split = "val" if b in val else "train"
        for s in shots:
            kind = variant(s.id, a.seed)
            if kind == "static":
                near = static.nearest(s.q, a.k)
            else:
                sh = shape(s.expr)
                near = bank.nearest(s.q, a.k, exclude_biz=b,
                                    skip=(lambda x, sh=sh: shape(x.expr) == sh) if kind == "novel" else None)
            msgs = messages(pick_views(views, s.q), s.q, near)
            target = lint(s.expr, s.q, [])
            msgs.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))})
            files[split].write(json.dumps({"id": s.id, "biz": b, "kind": kind, "messages": msgs},
                                          ensure_ascii=False) + "\n")
            n[split] += 1
            kinds[kind] += 1
            chars.append(sum(len(m["content"]) for m in msgs))
        print(b, split, len(shots), flush=True)
    for f in files.values():
        f.close()
    chars.sort()
    manifest = {"source": "train questions + verified expressions (src/socai/v2/bank.json); no public/private data",
                "bank_sha256": hashlib.sha256(a.bank.read_bytes()).hexdigest(), "bank_shots": len(bank.shots),
                "per_skeleton": a.per_skeleton, "k": a.k, "seed": a.seed, "counts": dict(n), "kinds": dict(kinds),
                "chars_p50_p95_max": [chars[len(chars) // 2], chars[int(len(chars) * .95)], chars[-1]],
                "val_biz": sorted(val),
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.out.glob("*.jsonl"))}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
