"""Thư viện metric tất định trên bảng chuẩn. Mỗi hàm trả `Result` có giá trị + dấu vết (để ghi log).

Luật lấy từ rulebook mặc định; chỗ nào phụ thuộc rulebook của split được ghi rõ bằng tham chiếu §.
Hàm nào không đủ dữ liệu thì trả `Result.unanswerable(...)` (§7) hoặc ném `Unsupported` để tầng
trên chuyển sang đường khác — KHÔNG trả 0 khi thực ra là không có dữ liệu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from socai.io.cells import round2, to_int_money
from socai.rules.timeexpr import Period
from socai.semantic.canonical import VALID, Canonical


class Unsupported(Exception):
    """Metric không áp được cho dữ liệu/câu hỏi này → thử đường khác."""


@dataclass
class Result:
    type: str
    value: Any = None
    trace: dict = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def unanswerable(cls, reason: str, **trace: Any) -> Result:
        return cls("unanswerable", None, trace, reason)


def coverage(can: Canonical, p: Period) -> str:
    """§7: kỳ nằm hoàn toàn ngoài khoảng ngày có dữ liệu → không trả lời được; một phần → tính phần có."""
    if can.data_start is None:
        return "unknown"
    end = can.data_end or can.orders["d"].max()
    if p.end < can.data_start or p.start > end:
        return "outside"
    return "partial" if (p.start < can.data_start or p.end > end) else "full"


def _orders_in(can: Canonical, p: Period, valid_only: bool = True) -> pd.DataFrame:
    if not can.ok:
        raise Unsupported("chưa dựng được bảng đơn")
    o = can.orders
    o = o[(o["d"] >= p.start) & (o["d"] <= p.end)]
    return o[o["status"].isin(VALID)] if valid_only else o


def revenue(can: Canonical, p: Period) -> Result:
    """§1: tổng thành tiền dòng (đã trừ giảm dòng) của đơn không huỷ, không trả; theo ngày đặt."""
    o = _orders_in(can, p)
    if o["revenue"].isna().all() and len(o):
        raise Unsupported("không có cột thành tiền")
    if "n_bad_amount" in o and o["n_bad_amount"].sum() > 0:
        # không bao giờ cộng lặng lẽ khi có ô tiền không đọc được: thà bỏ trống còn hơn sai
        raise Unsupported(f"{int(o['n_bad_amount'].sum())} ô thành tiền không đọc được trong kỳ")
    v = to_int_money(float(o["revenue"].fillna(0).sum()))
    return Result("number", v, {"fn": "revenue", "period": [str(p.start), str(p.end)], "orders": len(o)})


def order_count(can: Canonical, p: Period) -> Result:
    if not can.has_order_id and can.lines is not None:
        raise Unsupported("sổ không ghi mã đơn")
    o = _orders_in(can, p)
    return Result("number", int(len(o)), {"fn": "order_count", "period": [str(p.start), str(p.end)]})


def cancelled_count(can: Canonical, p: Period) -> Result:
    if not can.has_order_id and can.lines is not None:
        raise Unsupported("sổ không ghi mã đơn")
    o = _orders_in(can, p, valid_only=False)
    if not (can.orders["status"] == "cancelled").any():
        return Result.unanswerable("shop không ghi đơn huỷ")
    return Result("number", int((o["status"] == "cancelled").sum()), {"fn": "cancelled_count"})


def avg_order_value(can: Canonical, p: Period) -> Result:
    """§3: giá trị trung bình một đơn, làm tròn về đồng, nửa đồng lên."""
    r, n = revenue(can, p), order_count(can, p)
    if n.value == 0:
        return Result.unanswerable("kỳ không có đơn hợp lệ", period=[str(p.start), str(p.end)])
    return Result("number", to_int_money(r.value / n.value),
                  {"fn": "avg_order_value", "revenue": r.value, "orders": n.value})


def cancel_rate(can: Canonical, p: Period) -> Result:
    """§1: % đơn huỷ trong mọi đơn đặt trong kỳ (mọi trạng thái), 2 chữ số thập phân."""
    if not can.has_order_id:
        raise Unsupported("sổ không ghi mã đơn")
    o = _orders_in(can, p, valid_only=False)
    if len(o) == 0:
        return Result.unanswerable("kỳ không có đơn")
    if not (o["status"] == "cancelled").any() and not (can.orders["status"] == "cancelled").any():
        return Result.unanswerable("shop không ghi đơn huỷ")
    v = round2(100.0 * (o["status"] == "cancelled").sum() / len(o))
    return Result("percent", v, {"fn": "cancel_rate", "orders": len(o)})


def pct_change(cur: Result, prev: Result) -> Result:
    """§1: so với kỳ trước, giảm ra số âm, 2 chữ số thập phân."""
    if prev.value in (0, None):
        return Result.unanswerable("kỳ so sánh bằng 0 hoặc không có dữ liệu")
    v = round2(100.0 * (cur.value - prev.value) / prev.value)
    return Result("percent", v, {"fn": "pct_change", "cur": cur.trace, "prev": prev.trace})


def diff(cur: Result, prev: Result) -> Result:
    """§1: kỳ hỏi − kỳ so, giữ đơn vị."""
    return Result(cur.type, cur.value - prev.value, {"fn": "diff", "cur": cur.trace, "prev": prev.trace})


def best_day(can: Canonical, p: Period, by: str = "revenue") -> Result:
    """Ngày bán nhiều tiền / nhiều đơn nhất; bằng nhau lấy ngày sớm hơn (không có mã → theo tên)."""
    o = _orders_in(can, p)
    if o.empty:
        return Result.unanswerable("kỳ không có đơn")
    s = o.groupby("d")["revenue"].sum() if by == "revenue" else o.groupby("d").size()
    top = s[s == s.max()].index.min()
    return Result("date", top.isoformat(), {"fn": "best_day", "by": by, "value": float(s.max())})


METRICS = {"revenue": revenue, "order_count": order_count, "cancelled_count": cancelled_count,
           "avg_order_value": avg_order_value,
           "cancel_rate": cancel_rate, "best_day_revenue": best_day}
