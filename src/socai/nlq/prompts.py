"""Dựng prompt cho đường LLM. Thứ tự cố định để tận dụng prefix caching của vLLM (CAG):
  [hệ thống + rulebook của split + API + format]  ← giống hệt nhau cho mọi câu của split
  [hồ sơ shop: README + bảng + cột + dòng mẫu]      ← giống nhau cho mọi câu của một shop
  [câu hỏi + hôm nay + kỳ đã giải + gợi ý luật]     ← riêng từng câu
Câu hỏi được xử lý theo nhóm shop nên tiền tố thứ hai cũng được cache.
"""

from __future__ import annotations

import json

from socai.exec.api import API_DOC
from socai.profile.business import Business
from socai.rules.rulebook import Rulebook
from socai.rules.timeexpr import TimeSpec

SYSTEM = """Bạn là trợ lý phân tích sổ sách cho chủ cửa hàng nhỏ ở Việt Nam. Bạn trả lời câu hỏi bằng cách
VIẾT CODE PYTHON chạy trên dữ liệu thật của shop, theo đúng RULEBOOK dưới đây (luật mặc định khi câu hỏi
không nói khác). Không bịa số. Không dùng kiến thức ngoài dữ liệu.

Viết đúng một hàm:
```python
def answer(ctx):
    ...
    return {"type": "<number|percent|string|date|boolean|list|table|unanswerable>", "answer": ...}
```
Quy tắc đầu ra: tiền là số nguyên VND; percent là số phần trăm làm tròn 2 chữ số (12.4 chứ không 0.124);
date "YYYY-MM-DD" (câu hỏi theo tháng: "YYYY-MM"); tên lấy NGUYÊN VĂN từ dữ liệu; list có thứ tự;
table {"columns": [...], "rows": [[...]]} cột đầu là tên đối tượng. Khi dữ liệu không đủ để trả lời
(rulebook §7) trả {"type": "unanswerable", "answer": null, "reason": "..."}.

API có sẵn trong code:
{api}

===== RULEBOOK =====
{rulebook}
"""

SHOP = """===== HỒ SƠ SHOP {biz_id} =====
README:
{readme}

BẢNG (tên → cột: nghĩa đã gán | dòng mẫu):
{tables}
Bảng chuẩn hoá: {canonical}
"""

QUESTION = """===== CÂU HỎI =====
Hôm nay: {today}
Câu hỏi: {question}
Mốc thời gian đã giải (tất định, dùng đúng các ngày này): {periods}
Luật có thể liên quan: {rules}
Trả lời bằng một khối ```python``` chứa hàm answer(ctx)."""


def _table_block(biz: Business, n_rows: int = 3, max_cols: int = 30) -> str:
    out = []
    for t in biz.tables:
        cols = t.raw.columns[:max_cols]
        desc = ", ".join(f"{c}" + (f":{t.tags[c]}" if c in t.tags else "") for c in cols)
        sample = t.raw.df[cols].head(n_rows).astype(str).values.tolist() if len(t.raw.df) else []
        out.append(f"- {t.name} [{t.role}, {len(t.raw.df)} dòng]: {desc}\n  mẫu: {json.dumps(sample, ensure_ascii=False)[:600]}")
    return "\n".join(out)


def system_prompt(rb: Rulebook) -> str:
    return SYSTEM.replace("{api}", API_DOC).replace("{rulebook}", rb.text)


def shop_prompt(biz: Business, canonical_ok: bool) -> str:
    return SHOP.format(biz_id=biz.biz_id, readme=biz.readme.text[:6000], tables=_table_block(biz),
                       canonical="có (ctx.lines/ctx.orders)" if canonical_ok else "KHÔNG — dùng ctx.tables")


def question_prompt(question: str, today: str, spec: TimeSpec, rb: Rulebook) -> str:
    periods = [{"label": p.label, "start": str(p.start), "end": str(p.end), "kind": p.kind} for p in spec.periods]
    if spec.compare_to is not None:
        periods.append({"label": "so sánh", "start": str(spec.compare_to.start), "end": str(spec.compare_to.end)})
    rules = [c.text for c in rb.relevant(question, k=6)]
    return QUESTION.format(today=today, question=question, periods=json.dumps(periods, ensure_ascii=False),
                           rules=json.dumps(rules, ensure_ascii=False))


def messages(rb: Rulebook, biz: Business, canonical_ok: bool, question: str, today: str, spec: TimeSpec
             ) -> list[dict]:
    return [{"role": "system", "content": system_prompt(rb)},
            {"role": "user", "content": shop_prompt(biz, canonical_ok) + "\n" +
             question_prompt(question, today, spec, rb)}]
