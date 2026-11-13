"""Ngân hàng ví dụ: các cặp (câu hỏi, SQL) của tập TRAIN mà SQL đã chạy ra đúng đáp án chuẩn.

Dựng bằng `python -m socai.v5 bank` từ log của một lần chạy trên train (chỉ train: Rule 6). Lúc chạy thật, truy
hồi vài ví dụ giống câu hỏi nhất để LLM thấy cách tính; tên bảng/cột trong ví dụ là của shop khác.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socai.v5.l3_rules import tokens


def shape(question: str) -> list[str]:
    return tokens(re.sub(r"\d+", "0", question))


@dataclass
class Bank:
    items: list[dict] = field(default_factory=list)  # {q, sql, loai, don_vi, biz}
    idf: dict[str, float] = field(default_factory=dict)
    toks: list[Counter] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None) -> Bank:
        items = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else []
        toks = [Counter(shape(it["q"])) for it in items]
        n = len(items) or 1
        df = Counter(w for t in toks for w in t)
        return cls(items, {w: math.log(1 + n / d) for w, d in df.items()}, toks)

    def search(self, question: str, k: int = 4, exclude_biz: str | None = None) -> list[dict]:
        q = set(shape(question))
        scored = []
        for i, (it, t) in enumerate(zip(self.items, self.toks)):
            if exclude_biz and it.get("biz") == exclude_biz:
                continue
            s = sum(self.idf.get(w, 0.0) for w in q if w in t) / math.sqrt(len(t) + 1)
            if s > 0:
                scored.append((s, -i, it))
        out, seen = [], set()
        for _, _, it in sorted(scored, key=lambda x: (-x[0], -x[1])):
            key = re.sub(r"\s+", " ", re.sub(r"'[^']*'|\d+", "?", it["sql"]))  # không lấy hai ví dụ cùng khung SQL
            if key not in seen:
                seen.add(key)
                out.append(it)
            if len(out) == k:
                break
        return out


def build(log_path: Path, questions: Path, answers: Path, out: Path, merge: bool = True) -> int:
    """Giữ SQL của những mẫu chạy ra ĐÚNG đáp án chuẩn. `merge`: cộng dồn vào ngân hàng đã có (theo câu hỏi)."""
    from socai.eval.scorer import judge, load_jsonl

    qs, refs = load_jsonl(questions), load_jsonl(answers)
    items = {it["q"]: it for it in (json.loads(out.read_text(encoding="utf-8")) if merge and out.exists() else [])}
    for ln in log_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        ref, q = refs.get(rec["id"]), qs.get(rec["id"])
        if not ref or not q or ref.get("type") == "unanswerable":
            continue
        for smp in rec.get("samples") or []:
            if smp.get("sql") and smp.get("answer") and judge(smp["answer"], ref) == "correct":
                items[q["question"]] = {"q": q["question"], "sql": smp["sql"], "loai": smp["answer"]["type"],
                                        "don_vi": smp.get("don_vi", "so"), "biz": q["business_id"]}
                break
    out.write_text(json.dumps(list(items.values()), ensure_ascii=False, indent=0), encoding="utf-8")
    return len(items)
