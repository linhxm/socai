"""L6: kiểm và quyết định (v4.md §7). Lấy k mẫu spec, chạy hết, bỏ phiếu theo ĐÁP ÁN ĐÃ ĐỊNH DẠNG (không theo chữ SQL).

- Cổng thời gian (code): kỳ hỏi nằm hoàn toàn ngoài khoảng dữ liệu → unanswerable (§7).
- unanswerable do LLM đề xuất chỉ nhận khi đa số mẫu cùng đề xuất.
- Đồng thuận dưới ngưỡng → bỏ trống (sai bị trừ điểm).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from socai.rules.timeexpr import TimeSpec


@dataclass
class Decision:
    answer: dict
    source: str
    agree: float
    n: int


def gate_time(spec: TimeSpec, date_range: tuple[str, str] | None) -> str | None:
    if not date_range or not spec.periods:
        return None
    lo, hi = date_range
    for p in spec.periods + ([spec.compare_to] if spec.compare_to else []):
        if p.kind == "until":
            continue
        if str(p.end) < lo or str(p.start) > hi:
            return f"kỳ {p.start}→{p.end} nằm ngoài dữ liệu {lo}→{hi} (§7)"
    return None


def sanity(ans: dict) -> str | None:
    t, a = ans.get("type"), ans.get("answer")
    if t == "percent" and isinstance(a, (int, float)) and abs(a) > 100000:
        return "phần trăm bất thường"
    if t == "number" and isinstance(a, (int, float)) and a < 0 and abs(a) > 1e12:
        return "số bất thường"
    return None


def decide(results: list[dict], min_agree: float = 0.5) -> Decision:
    """results: mỗi mẫu {'answer': dict} hoặc {'error': str}. Trả đáp án đa số nếu tỉ lệ đồng thuận > min_agree."""
    ok = [r["answer"] for r in results if "answer" in r and not sanity(r["answer"])]
    n = len(results)
    if not ok:
        return Decision({"type": "abstain"}, "no_result", 0.0, n)
    keys = Counter(json.dumps(a, ensure_ascii=False, sort_keys=True) for a in ok)
    best, cnt = keys.most_common(1)[0]
    agree = cnt / n
    ans = json.loads(best)
    if ans["type"] == "unanswerable" and agree <= 0.5 and n > 1:
        return Decision({"type": "abstain"}, "unans_minority", agree, n)
    if n == 1 or agree > min_agree:
        return Decision(ans, "agree" if agree == 1 else "majority", agree, n)
    return Decision({"type": "abstain"}, "split", agree, n)
