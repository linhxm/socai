"""API mà code do LLM sinh ra được dùng (`ctx`). Giữ API nhỏ, tất định, có docstring ngắn vì toàn bộ
phần này được đưa vào prompt. Code sinh ra phải định nghĩa `def answer(ctx) -> dict` trả
{"type": ..., "answer": ...} theo đúng format nộp bài.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from socai.io.cells import fold, parse_date, parse_number, parse_qty, round2, to_int_money
from socai.profile.business import Business
from socai.rules.calendar import LunarCalendar
from socai.rules.timeexpr import Period, Resolver
from socai.semantic.canonical import Canonical, norm_status

API_DOC = '''
ctx.today            -> datetime.date  (as_of_date của câu hỏi)
ctx.tables           -> dict[str, pandas.DataFrame]  bảng THÔ (giá trị gốc, chưa chuẩn hoá)
ctx.columns          -> dict[str, dict[str, str]]    bảng -> {cột: nghĩa đã gán (tag)}
ctx.readme           -> str   README của shop
ctx.lines, ctx.orders-> DataFrame chuẩn hoá (có thể None). lines: order_id,d,dt,status,customer_id,
                        customer_name,product_code,product_name,product_group,qty,unit,qty_base,unit_price,
                        line_amount,discount,cost_unit,payment_method,channel; orders: order_id,d,status,
                        customer_id,customer_name,payment_method,revenue. status ∈ done|shipping|cancelled|returned
ctx.period(text)     -> Period(start, end, kind) đã giải theo rulebook; ctx.periods(text) -> list
num(v) / day(v, year=None) / qty(v) -> đọc tiền-số / ngày / (số lượng, đơn vị) từ ô bẩn
key(s)               -> khoá so khớp tên (bỏ dấu, chữ thường)
money(x)             -> làm tròn tiền về đồng (nửa lên);  r2(x) -> làm tròn 2 chữ số
status(v)            -> chuẩn hoá trạng thái đơn
'''


@dataclass
class Ctx:
    biz: Business
    can: Canonical
    today: date
    cal: LunarCalendar | None = None
    tables: dict[str, pd.DataFrame] = field(init=False)
    columns: dict[str, dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.tables = {t.name: t.raw.df for t in self.biz.tables}
        self.columns = {t.name: dict(t.tags) for t in self.biz.tables}

    @property
    def readme(self) -> str:
        return self.biz.readme.text

    @property
    def lines(self) -> pd.DataFrame | None:
        return self.can.lines

    @property
    def orders(self) -> pd.DataFrame | None:
        return self.can.orders

    def periods(self, text: str) -> list[Period]:
        return Resolver(self.today, self.cal).resolve(text).periods

    def period(self, text: str) -> Period:
        ps = self.periods(text)
        if not ps:
            raise ValueError(f"không giải được thời gian: {text!r}")
        return ps[0]


SANDBOX_GLOBALS = {
    "pd": pd, "date": date, "num": parse_number, "day": parse_date, "qty": parse_qty, "key": fold,
    "money": to_int_money, "r2": round2, "status": norm_status,
}
