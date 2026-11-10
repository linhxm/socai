"""Kiểm format đáp án và quyết định TRẢ LỜI hay BỎ TRỐNG.

Điểm PA: đúng +1, sai −1, bỏ trống 0. Trả lời có lợi khi và chỉ khi P(đúng) > 0.5. Mỗi ứng viên mang
một ước lượng P(đúng) = độ chính xác đã hiệu chỉnh của (nguồn, metric) trên train theo
leave-business-out (`calibration.json`, sinh bởi `socai eval --calibrate`), điều chỉnh bởi đồng thuận
giữa các đường độc lập và chất lượng nguồn dữ liệu (OCR → confidence < 1).
Ngưỡng mặc định 0.5 + biên an toàn, vì hiệu chỉnh trên train lạc quan hơn private (ngành mới).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from socai.io.cells import round2

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"}
_RE_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def _is_num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _num(x: float) -> int | float:
    """Số nguyên giữ nguyên; số lẻ làm tròn 2 chữ số (đề: số là nguyên hoặc làm tròn 2 chữ số)."""
    return int(round(x)) if abs(x - round(x)) < 1e-9 else round2(x)


def normalize(raw: object) -> dict | None:
    """Đưa đáp án về đúng format nộp bài; sai format → None (để bỏ trống thay vì nộp sai)."""
    if not isinstance(raw, dict) or raw.get("type") not in TYPES:
        return None
    t, a = raw["type"], raw.get("answer")
    if t == "unanswerable":
        r = raw.get("reason")
        return {"type": t, **({"reason": r} if isinstance(r, str) else {})}
    if t == "number" and _is_num(a):
        return {"type": t, "answer": _num(a)}
    if t == "percent" and _is_num(a):
        return {"type": t, "answer": round2(a)}
    if t == "date" and isinstance(a, str) and _RE_DATE.match(a):
        return {"type": t, "answer": a}
    if t == "boolean" and isinstance(a, bool):
        return {"type": t, "answer": a}
    if t == "string" and isinstance(a, str) and a.strip():
        return {"type": t, "answer": a.strip()}
    if t == "list" and isinstance(a, list) and all(isinstance(x, str) or _is_num(x) for x in a):
        return {"type": t, "answer": [_num(x) if _is_num(x) else x for x in a]}
    if t == "table" and isinstance(a, dict) and isinstance(a.get("columns"), list) \
            and isinstance(a.get("rows"), list) and all(isinstance(r, list) for r in a["rows"]):
        rows = [[_num(x) if _is_num(x) else x for x in r] for r in a["rows"]]
        return {"type": t, "answer": {"columns": a["columns"], "rows": rows}}
    return None


@dataclass
class Candidate:
    answer: dict  # đã normalize
    source: str  # fastpath | llm_code | llm_ir ...
    metric: str = "other"
    p: float = 0.5  # ước lượng P(đúng)
    trace: dict = field(default_factory=dict)


@dataclass
class Policy:
    threshold: float = 0.6
    agree_bonus: float = 0.15
    disagree_cap: float = 0.3
    priors: dict[str, float] = field(default_factory=lambda: {
        # giá trị khởi tạo; `socai eval --calibrate` ghi đè bằng số đo leave-business-out
        "fastpath": 0.9, "llm_code": 0.55, "llm_code:unanswerable": 0.5,
    })

    @classmethod
    def load(cls, path: Path | None) -> Policy:
        pol = cls()
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            pol.priors.update(data.get("priors", {}))
            pol.threshold = data.get("threshold", pol.threshold)
        return pol

    def prior(self, c: Candidate) -> float:
        for k in (f"{c.source}:{c.metric}", f"{c.source}:{c.answer['type']}", c.source):
            if k in self.priors:
                return self.priors[k]
        return 0.5

    def decide(self, cands: list[Candidate], source_conf: float = 1.0) -> tuple[dict, dict]:
        """Trả (đáp án nộp, giải trình). Không có ứng viên đủ tin → abstain."""
        if not cands:
            return {"type": "abstain"}, {"why": "không có ứng viên"}
        for c in cands:
            c.p = self.prior(c) * source_conf
        keys = [json.dumps(c.answer, sort_keys=True, ensure_ascii=False) for c in cands]
        if len(set(keys)) == 1 and len(cands) > 1:
            best = max(cands, key=lambda c: c.p)
            p = min(0.99, best.p + self.agree_bonus)
        elif len(set(keys)) > 1:
            best = max(cands, key=lambda c: c.p)
            p = min(best.p, self.disagree_cap) if best.source != "fastpath" else best.p - self.agree_bonus
        else:
            best, p = cands[0], cands[0].p
        why = {"p": round(p, 3), "threshold": self.threshold, "source": best.source,
               "candidates": [{"source": c.source, "answer": c.answer, "p": round(c.p, 3)} for c in cands]}
        return (best.answer if p > self.threshold else {"type": "abstain"}), why
