"""QueryIR: biểu diễn trung gian độc lập ngôn ngữ của một câu hỏi.

Câu hỏi tiếng Việt có dấu, không dấu, teencode, tiếng Anh hay tiếng Trung đều phải ra CÙNG một IR
nếu cùng nghĩa: đây là bất biến được kiểm bằng test metamorphic (tests/test_metamorphic.py).
LLM sinh IR theo `IR_SCHEMA` (guided decoding JSON); ngày cụ thể KHÔNG nằm trong IR mà do
`rules.timeexpr` tính từ `time_text`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

METRICS = ["revenue", "order_count", "avg_order_value", "cancel_rate", "best_day_revenue",
           "qty_sold", "gross_profit", "net_profit", "debt", "stock", "purchase_amount", "other"]
COMPARE = [None, "pct_change", "diff"]


@dataclass
class QueryIR:
    metric: str
    time_text: str = ""  # cụm thời gian nguyên văn trong câu (để timeexpr giải)
    compare: str | None = None
    filters: dict[str, str] = field(default_factory=dict)  # product, product_group, customer, payment...
    group_by: str | None = None
    top_k: int | None = None
    answer_type: str = "number"
    source: str = "fastpath"  # fastpath | llm
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


IR_SCHEMA = {
    "type": "object",
    "properties": {
        "metric": {"type": "string", "enum": METRICS},
        "time_text": {"type": "string"},
        "compare": {"type": ["string", "null"], "enum": COMPARE},
        "filters": {"type": "object", "additionalProperties": {"type": "string"}},
        "group_by": {"type": ["string", "null"]},
        "top_k": {"type": ["integer", "null"]},
        "answer_type": {"type": "string", "enum": ["number", "percent", "string", "date", "boolean",
                                                   "list", "table", "unanswerable"]},
        "answerable_reason": {"type": "string"},
    },
    "required": ["metric", "time_text", "compare", "filters", "answer_type"],
}
