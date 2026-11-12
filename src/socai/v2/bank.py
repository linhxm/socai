"""Ngân hàng ví dụ đã kiểm chứng: (câu hỏi train → biểu thức) mà biểu thức CHẠY RA ĐÚNG đáp án chuẩn.

Dùng để truy hồi vài ví dụ giống câu đang hỏi nhất đưa vào prompt (thay cho danh sách ví dụ cố định):
LLM nhỏ chỉ thấy cú pháp liên quan, không áp nhầm phép lạ vào câu thường. Cũng là dữ liệu fine-tune.
Chỉ dựng từ bộ train (`training_code/build_bank.py`); khi đo trên shop nào thì loại ví dụ của shop đó.
Độ giống: trùng từ có trọng số IDF trên câu đã bỏ dấu, bỏ Telex, số thay bằng '0'.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from socai.io.cells import fold
from socai.nlq.normalize import untelex


def tokens(q: str) -> list[str]:
    f = re.sub(r"\d+", "0", fold(q))
    return [untelex(w) for w in re.findall(r"[a-z0-9]+", f)]


def skeleton(e: object) -> str:
    """Dạng biểu thức đã che giá trị: hai ví dụ cùng khung chỉ khác tên/số."""
    if isinstance(e, dict):
        return "{" + ",".join(f"{k}:{skeleton(v)}" for k, v in sorted(e.items()) if k != "nghi") + "}"
    if isinstance(e, list):
        if len(e) == 3 and isinstance(e[0], str) and isinstance(e[1], str):
            return f"[{e[0]},{e[1]},_]"
        return "[" + ",".join(skeleton(x) for x in e) + "]"
    return "_" if isinstance(e, (int, float)) and not isinstance(e, bool) else str(e)


def canonical(e: dict) -> dict:
    """`$ky_so` chỉ có nghĩa ở vế b của phép so sánh; ở gốc hoặc vế a code vẫn hiểu là `$ky` → ghi lại cho đúng,
    để ví dụ trong prompt (và dữ liệu fine-tune) không dạy thói quen sai."""
    if not isinstance(e, dict):
        return e
    if e.get("time") == "$ky_so":
        e = {**e, "time": "$ky"}
    if isinstance(e.get("a"), dict) and e["a"].get("time") == "$ky_so":
        e = {**e, "a": {**e["a"], "time": "$ky"}}
    return e


@dataclass
class Shot:
    q: str
    expr: dict
    biz: str = ""
    id: str = ""
    toks: frozenset = frozenset()


@dataclass
class Bank:
    shots: list[Shot] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)
    seeded: bool = False

    @classmethod
    def load(cls, path: Path | None) -> Bank:
        b = cls()
        if path and Path(path).exists():
            for r in json.loads(Path(path).read_text(encoding="utf-8")):
                b.add(r["q"], r["expr"], r.get("biz", ""), r.get("id", ""), reindex=False)
            b.reindex()
        return b

    def save(self, path: Path) -> None:
        rows = [{"id": s.id, "biz": s.biz, "q": s.q, "expr": s.expr} for s in self.shots if s.biz != "_static"]
        Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")

    def add(self, q: str, expr: dict, biz: str = "", id: str = "", reindex: bool = True) -> None:
        self.shots.append(Shot(q, canonical(expr), biz, id, frozenset(tokens(q))))
        if reindex:
            self.reindex()

    def reindex(self) -> None:
        df = Counter(t for s in self.shots for t in s.toks)
        n = max(1, len(self.shots))
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def nearest(self, question: str, k: int = 8, exclude_biz: str | None = None, per_skeleton: int = 2,
                skip: Callable[[Shot], bool] | None = None) -> list[Shot]:
        """k ví dụ giống nhất, tối đa `per_skeleton` ví dụ cho mỗi khung biểu thức (để có đa dạng cách làm).
        `skip`: bỏ ví dụ thoả điều kiện (dữ liệu fine-tune mô phỏng câu kiểu mới)."""
        qt = set(tokens(question))
        if not qt or not self.shots:
            return []
        d = max(self.idf.values(), default=1.0)
        qn = math.sqrt(sum(self.idf.get(t, d) ** 2 for t in qt))
        scored = []
        for i, s in enumerate(self.shots):
            if (exclude_biz and s.biz == exclude_biz) or (skip and skip(s)):
                continue
            common = qt & s.toks
            if not common:
                continue
            sn = math.sqrt(sum(self.idf.get(t, d) ** 2 for t in s.toks))
            scored.append((sum(self.idf.get(t, d) ** 2 for t in common) / (qn * sn), -i, s))
        out, used = [], defaultdict(int)
        for _, _, s in sorted(scored, key=lambda x: (-x[0], -x[1])):
            sk = skeleton(s.expr)
            if used[sk] >= per_skeleton:
                continue
            used[sk] += 1
            out.append(s)
            if len(out) == k:
                break
        return out
