"""Lệnh cơ bản (q.where · q.group · q.top · q.share · q.compare · series) chạy TẤT ĐỊNH trên bảng chuẩn.

LLM chỉ điền `QuerySpec` (đo gì, lọc gì, nhóm theo gì, top mấy, dạng đáp án); mọi luật rulebook nằm ở đây:
- §1 đơn tính doanh thu: không huỷ, không trả (trả một phần cũng bỏ cả đơn), theo ngày đặt;
- §3 "bán chạy" = số lượng quy về đơn vị cơ bản; bằng nhau xếp theo mã tăng dần (§9);
- §7 kỳ nằm hoàn toàn ngoài dữ liệu → unanswerable;
- tiền làm tròn về đồng, % 2 chữ số.
Trường hợp chưa chắc đúng luật (ví dụ giá trị TB đơn có lọc theo món) → `Unsupported`, không đoán.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from socai.exec.metrics import Result, Unsupported, coverage
from socai.io.cells import round2, to_int_money
from socai.rules.timeexpr import Period
from socai.semantic.canonical import VALID, Canonical

MEASURES = ("revenue", "qty", "orders", "aov", "customers")
GROUPS = ("product", "product_group", "customer", "day", "month", "payment_method", "channel", "staff", "hour")
SHAPES = ("value", "top", "best", "share", "series", "compare", "split")
LINE_FIELDS = {"product_name", "product_code", "product_group"}
ORDER_FIELDS = {"customer_name", "customer_id", "customer_group", "payment_method", "channel", "staff_id"}


@dataclass
class QuerySpec:
    measure: str = "revenue"
    shape: str = "value"
    periods: list[Period] = field(default_factory=list)
    compare_to: Period | None = None
    compare: str | None = None  # pct | diff
    filters: dict[str, list[str]] = field(default_factory=dict)  # field → giá trị NGUYÊN VĂN trong dữ liệu
    excludes: dict[str, list[str]] = field(default_factory=dict)
    size: str | None = None  # M | L (tên món kết thúc bằng (M)/(L), rulebook ngành cà phê)
    group_by: str | None = None
    top_k: int | None = None
    ascending: bool = False
    split: list[dict[str, list[str]]] = field(default_factory=list)  # mỗi phần tử: một bộ lọc → một số
    trend: str | None = None  # up | down: câu đúng/sai "có tăng liên tục không"

    def describe(self) -> dict:
        return {"measure": self.measure, "shape": self.shape, "periods": [[str(p.start), str(p.end)] for p in self.periods],
                "compare_to": [str(self.compare_to.start), str(self.compare_to.end)] if self.compare_to else None,
                "compare": self.compare, "filters": self.filters, "excludes": self.excludes, "size": self.size,
                "group_by": self.group_by, "top_k": self.top_k, "ascending": self.ascending, "split": self.split,
                "trend": self.trend}


def _valid_lines(can: Canonical) -> pd.DataFrame:
    """Dòng hàng của đơn hợp lệ (§1) + trường cấp đơn. Tính một lần mỗi shop rồi giữ lại."""
    cached = getattr(can, "_valid_cache", None)
    if cached is not None:
        return cached
    if can.lines is None or can.orders is None:
        raise Unsupported("cần bảng dòng hàng có mã đơn")
    O = can.orders.set_index("order_id")
    L = can.lines.copy()
    L["ostatus"] = L["order_id"].map(O["status"])
    for c in ("customer_group", "channel", "staff_id", "payment_method", "customer_id", "customer_name"):
        if c in O:  # trường cấp đơn lấy từ bảng đơn (đã điền theo nhóm §8)
            L[c] = L["order_id"].map(O[c]).fillna(L[c] if c in L else "")
    L = L[L["ostatus"].isin(VALID)]
    can._valid_cache = L  # type: ignore[attr-defined]
    return L


def _apply_filters(L: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
    for f, vals in spec.filters.items():
        if f not in L:
            raise Unsupported(f"không có trường {f}")
        L = L[L[f].isin(vals)]
    for f, vals in spec.excludes.items():
        if f in L:
            L = L[~L[f].isin(vals)]
    if spec.size:
        L = L[L["product_name"].str.strip().str.upper().str.endswith(f"({spec.size})")]
    return L


def _in(L: pd.DataFrame, p: Period) -> pd.DataFrame:
    return L[(L["d"] >= p.start) & (L["d"] <= p.end)]


def _measure(L: pd.DataFrame, m: str) -> float | int:
    if m == "revenue":
        if L["line_amount"].isna().any():
            raise Unsupported("ô thành tiền không đọc được trong kỳ")
        return to_int_money(float(L["line_amount"].sum()))
    if m == "qty":
        q = L["qty_base"].where(L["qty_base"].notna(), L["qty"])
        if q.isna().any():
            raise Unsupported("số lượng không đọc được")
        v = float(q.sum())
        return int(v) if v.is_integer() else round2(v)
    if m == "orders":
        return int(L["order_id"].nunique())
    if m == "customers":
        key = L["customer_id"].where(L["customer_id"].ne(""), L["customer_name"])
        return int(key[key.ne("") & ~key.str.lower().isin(["khách lẻ", "khach le"])].nunique())
    raise Unsupported(f"đo {m}")


GROUP_COL = {"product": ("product_name", "product_code"), "product_group": ("product_group", "product_group"),
             "customer": ("customer_name", "customer_id"), "payment_method": ("payment_method", "payment_method"),
             "channel": ("channel", "channel"), "staff": ("staff_id", "staff_id")}


def _ranked(L: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
    """Bảng (tên, mã, giá trị) đã xếp: giá trị giảm dần, bằng nhau theo mã tăng dần (§9)."""
    if spec.group_by in ("day", "month", "hour"):
        if spec.group_by == "hour":
            if L["dt"].isna().all():
                raise Unsupported("không có giờ")
            key = L["dt"].map(lambda t: t.hour if t is not None else None)
        else:
            key = L["d"].map(lambda d: d.isoformat() if spec.group_by == "day" else d.strftime("%Y-%m"))
        g = L.assign(_k=key).groupby("_k")
        vals = {k: _measure(part, spec.measure) for k, part in g}
        df = pd.DataFrame({"name": list(vals), "code": list(vals), "v": list(vals.values())})
    else:
        name_col, code_col = GROUP_COL[spec.group_by]
        if L[name_col].eq("").all():
            raise Unsupported(f"không có cột {name_col}")
        LL = L[L[name_col].ne("")]
        if spec.group_by == "customer":
            LL = LL[~LL[name_col].str.lower().isin(["khách lẻ", "khach le"])]
        code = LL[code_col].where(LL[code_col].ne(""), LL[name_col])
        rows = []
        for (n, c), part in LL.assign(_c=code).groupby([name_col, "_c"]):
            rows.append((n, c, _measure(part, spec.measure)))
        df = pd.DataFrame(rows, columns=["name", "code", "v"])
    if df.empty:
        return df
    return df.sort_values(["v", "code"], ascending=[spec.ascending, True], kind="mergesort")


def execute(can: Canonical, spec: QuerySpec) -> Result:
    if not spec.periods:
        raise Unsupported("thiếu kỳ")
    for p in spec.periods + ([spec.compare_to] if spec.compare_to else []):
        if coverage(can, p) == "outside":
            return Result.unanswerable("kỳ nằm ngoài khoảng dữ liệu (§7)", period=[str(p.start), str(p.end)])
    if spec.measure == "aov" and (spec.filters.keys() & LINE_FIELDS or spec.size):
        raise Unsupported("giá trị TB đơn có lọc theo món: chưa chắc luật")
    if spec.measure in ("orders", "aov") and not can.has_order_id:
        raise Unsupported("sổ không ghi mã đơn: không đếm được số đơn")
    base = _valid_lines(can)
    L = _apply_filters(base, spec)
    if (spec.filters or spec.size) and L.empty and spec.shape in ("top", "best"):
        raise Unsupported("bộ lọc không khớp dòng nào")
    trace = {"fn": "query", "spec": spec.describe()}

    def value(LL: pd.DataFrame, p: Period) -> float | int:
        X = _in(LL, p)
        if spec.measure == "aov":
            n = X["order_id"].nunique()
            if n == 0:
                raise Unsupported("kỳ không có đơn")
            return to_int_money(_measure(X, "revenue") / n)
        return _measure(X, spec.measure)

    p0 = spec.periods[0]
    if spec.shape == "value":
        return Result("number", value(L, p0), trace)
    if spec.shape == "split":
        out = []
        for f in spec.split:
            sub = QuerySpec(filters={**spec.filters, **f}, excludes=spec.excludes, size=spec.size)
            out.append(value(_apply_filters(base, sub), p0))
        return Result("list", out, trace)
    if spec.shape == "series":
        vals = [value(L, p) for p in spec.periods]
        if spec.trend:
            if len(vals) < 2:
                raise Unsupported("cần ≥ 2 kỳ để xét xu hướng")
            pairs = list(zip(vals, vals[1:]))
            ok = all(b > a for a, b in pairs) if spec.trend == "up" else all(b < a for a, b in pairs)
            return Result("boolean", ok, {**trace, "series": vals})
        return Result("list", vals, trace)
    if spec.shape == "compare":
        if spec.compare_to is None or spec.compare not in ("pct", "diff"):
            raise Unsupported("thiếu kỳ so sánh")
        cur, prev = value(L, p0), value(L, spec.compare_to)
        if spec.compare == "diff":
            return Result("number", cur - prev, trace)
        if not prev:
            return Result.unanswerable("kỳ so sánh bằng 0")
        return Result("percent", round2(100.0 * (cur - prev) / prev), trace)
    if spec.shape == "share":
        if spec.measure not in ("revenue", "qty", "orders"):
            raise Unsupported("tỉ trọng cho thước đo này")
        part, whole = value(L, p0), value(base if spec.measure == "orders" else _apply_filters(base, QuerySpec()), p0)
        if not whole:
            return Result.unanswerable("tổng bằng 0")
        return Result("percent", round2(100.0 * part / whole), trace)
    if spec.shape in ("top", "best"):
        if not spec.group_by:
            raise Unsupported("thiếu nhóm theo")
        df = _ranked(_in(L, p0), spec)
        if df.empty:
            return Result.unanswerable("kỳ không có dữ liệu")
        k = spec.top_k or 1
        if spec.shape == "best" and spec.group_by in ("day", "month"):
            return Result("date", str(df.iloc[0]["name"]), trace)
        if spec.shape == "best" and spec.group_by == "hour":
            return Result("number", int(df.iloc[0]["name"]), trace)
        names = [str(x) for x in df.head(k)["name"]]
        trace["ranking"] = df.head(k + 2).astype(str).values.tolist()
        return Result("string", names[0], trace) if (spec.top_k in (None, 1)) else Result("list", names, trace)
    raise Unsupported(f"dạng {spec.shape}")
