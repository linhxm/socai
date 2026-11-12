"""Bảng chuẩn (semantic layer) dựng từ các bảng đã gán nghĩa: `lines` (dòng hàng) và `orders` (đơn).

Đây là ĐƯỜNG NHANH cho các mẫu dữ liệu quen thuộc. Shop không dựng được bảng chuẩn (ngành mới,
SQLite chưa gán nghĩa, nguồn hỗn hợp) không bị bỏ: tầng trả lời chuyển sang đường sinh code trên
bảng thô. Mọi luật áp ở đây đều trích từ rulebook §1, §8 và có tham chiếu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time

import pandas as pd

from socai.io.cells import clean, fold, parse_date, parse_datetime, parse_number, parse_qty, parse_time
from socai.profile.business import Business, Table

STATUS_RULES: list[tuple[tuple[str, ...], str]] = [
    (("huy", "cancel", "khong giao duoc"), "cancelled"),
    (("tra hang", "doi tra", "tra lai", "hoan tra", "hoan hang", "hang tra ve", "return"), "returned"),
    (("dang giao", "dang van chuyen", "dang gui", "cho giao", "shipping", "dang xu ly", "dang giao dich"),
     "shipping"),
    (("hoan thanh", "xong", "completed", "da lay", "da giao", "thanh cong", "ok", "done"), "done"),
]
VALID = ("done", "shipping")  # §1: đơn tính doanh thu = không huỷ, không trả; đơn đang giao vẫn tính


def norm_status(v: object) -> str:
    s = fold(v)
    if not s:
        return ""
    for keys, st in STATUS_RULES:
        if any(k in s for k in keys):
            return st
    return "unknown"


LINE_FIELDS = ["order_id", "line_no", "order_date", "order_datetime", "order_time", "status", "note",
               "customer_id", "customer_name", "product_code", "product_name", "product_group", "qty",
               "unit", "qty_base", "unit_price", "line_amount", "discount", "cost_unit", "qty_returned_base",
               "payment_method", "channel", "staff_id"]
ORDER_LEVEL = ["order_date", "order_datetime", "status", "customer_id", "customer_name", "note",
               "payment_method", "channel", "staff_id"]


@dataclass
class Canonical:
    lines: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    data_start: date | None = None
    data_end: date | None = None
    has_order_id: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.orders is not None and not self.orders.empty


def _project(t: Table, fields: list[str]) -> pd.DataFrame:
    df = t.raw.df
    out = pd.DataFrame(index=df.index)
    for f in fields:
        col = t.col(f)
        if f == "product_name" and col is None and t.role == "products":
            col = t.col("name")
        out[f] = df[col] if col else None
    out["_sheet_year"] = df["_sheet_year"] if "_sheet_year" in df else None
    out["_src"] = t.name
    return out


_SUBTOTAL = ("tong", "cong", "total", "subtotal", "luy ke")


def _drop_subtotals(df: pd.DataFrame) -> pd.DataFrame:
    """Dòng tổng chèn giữa sổ ("TỔNG NGÀY 09/04", "Tổng tháng 4/2026", "Cộng"): không có mã đơn, không có
    số lượng, có một ô chữ bắt đầu bằng 'tổng/cộng'. Phải bỏ TRƯỚC khi điền ô trống theo nhóm, nếu không
    chúng bị gán vào đơn liền trước và doanh thu bị đếm lặp (lỗi thật đã gặp ở một shop train)."""
    if df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for c in ("product_name", "order_id", "customer_name", "note"):
        if c in df:
            mask |= df[c].map(fold).str.startswith(_SUBTOTAL)
    return df[~(mask & df["qty"].map(clean).eq(""))]


def _ffill_groups(df: pd.DataFrame) -> pd.DataFrame:
    """§8: ô trống ở các dòng sau của cùng một nhóm dòng = giống ô ở dòng đầu nhóm."""
    df = _drop_subtotals(df)
    blank = df["order_id"].map(clean).eq("")
    if df["order_id"].notna().any() and not blank.all():
        df["order_id"] = df["order_id"].where(~blank).ffill()
        for c in ORDER_LEVEL:
            if c in df:
                v = df[c].where(df[c].map(clean).ne(""))
                df[c] = v.groupby(df["order_id"], sort=False).ffill()
    return df


def build(biz: Business) -> Canonical:
    can = Canonical(data_end=biz.readme.data_end)
    line_tabs = biz.by_role("order_lines")
    order_tabs = biz.by_role("orders")
    if not line_tabs and not order_tabs:
        can.issues.append("không có bảng dòng hàng/đơn được nhận diện")
        return can

    if line_tabs:
        lines = pd.concat([_ffill_groups(_project(t, LINE_FIELDS)) for t in line_tabs], ignore_index=True)
        can.has_order_id = lines["order_id"].map(clean).ne("").any()
        if not can.has_order_id:
            # sổ không ghi mã đơn: mỗi dòng là một giao dịch. Doanh thu/số lượng/bán chạy vẫn tính được;
            # "số đơn" thì không (query.py và fastpath từ chối khi has_order_id = False)
            lines["order_id"] = [f"_L{i}" for i in range(len(lines))]
        if can.has_order_id:  # §8: hai dòng giống hệt nhau của cùng một đơn là lỗi xuất file
            key = lines.drop(columns=["_src"]).astype(str)
            lines = lines[~key.duplicated()]
        # bổ sung từ bảng đơn / danh mục
        for t in order_tabs:
            o = _project(t, ORDER_LEVEL + ["order_id"]).drop(columns=["_src", "_sheet_year"])
            o = o.drop_duplicates("order_id").set_index("order_id")
            for c in ORDER_LEVEL:
                if c in o and o[c].notna().any():
                    filled = lines["order_id"].map(o[c])
                    lines[c] = lines[c].where(lines[c].map(clean).ne(""), filled)
        for t in biz.by_role("products"):
            p = _project(t, ["product_code", "product_name", "product_group"])
            p = p[p["product_code"].map(clean).ne("")].drop_duplicates("product_code").set_index("product_code")
            for c in ("product_name", "product_group"):
                if p[c].notna().any():
                    lines[c] = lines[c].where(lines[c].map(clean).ne(""), lines["product_code"].map(p[c]))
        for t in biz.by_role("products"):  # dòng bán chỉ có tên (không mã): nối danh mục theo tên đã fold
            ncol, gcol = t.col("product_name") or t.col("name"), t.col("product_group")
            if ncol and gcol:
                gmap = {fold(n): clean(g) for n, g in zip(t.raw.df[ncol], t.raw.df[gcol]) if fold(n)}
                lines["product_group"] = lines["product_group"].where(
                    lines["product_group"].map(clean).ne(""), lines["product_name"].map(lambda v: gmap.get(fold(v))))
        lines["customer_group"] = None
        for t in biz.by_role("customers"):  # nhóm khách, tên khách từ danh sách khách hàng
            cid = t.col("customer_id")
            if not cid:
                continue
            cdf = t.raw.df.drop_duplicates(cid).set_index(cid)
            if (gcol := t.col("customer_group")):
                lines["customer_group"] = lines["customer_group"].where(
                    lines["customer_group"].map(clean).ne(""), lines["customer_id"].map(cdf[gcol]))
            if (ncol := t.col("name") or t.col("customer_name")):
                lines["customer_name"] = lines["customer_name"].where(
                    lines["customer_name"].map(clean).ne(""), lines["customer_id"].map(cdf[ncol]))
        can.lines = _typed_lines(lines)
        _fill_qty_base(can.lines, biz, can)
        can.orders = _orders_from_lines(can.lines)
    else:
        can.orders = _orders_only(order_tabs)
    if can.orders is not None and not can.orders.empty:
        can.data_start = can.orders["d"].min()
    elif can.lines is not None and not can.lines.empty:
        can.data_start = can.lines["d"].min()
    return can


def _typed_lines(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    yr = df["_sheet_year"]
    out["order_id"] = df["order_id"].map(clean)
    dt = [parse_datetime(a, y if pd.notna(y) else None) for a, y in zip(df["order_datetime"], yr)]
    d = [parse_date(a, y if pd.notna(y) else None) for a, y in zip(df["order_date"], yr)]
    out["d"] = [x or (t.date() if t else None) for x, t in zip(d, dt)]
    if df["order_time"].notna().any():  # giờ ghi ở cột riêng ("Giờ"): ghép với ngày đặt
        tm = [parse_time(v) for v in df["order_time"]]
        dt = [t if (t is not None and t.time() != time()) or x is None or h is None else datetime.combine(x, h)
              for t, x, h in zip(dt, out["d"], tm)]
    out["dt"] = dt
    out["status"] = df["status"].map(norm_status)
    if out["status"].eq("").all() and df["note"].notna().any():  # sổ không có cột trạng thái → Ghi chú
        out["status"] = df["note"].map(norm_status).replace({"": "done", "unknown": "done"})
    out["status"] = out["status"].replace({"": "done"})
    for c in ("customer_id", "customer_name", "customer_group", "product_code", "product_name", "product_group",
              "unit", "payment_method", "channel", "staff_id"):
        out[c] = df[c].map(clean) if c in df else ""
    out["qty"] = [parse_qty(v)[0] for v in df["qty"]]
    out["qty_base"] = df["qty_base"].map(parse_number)
    for c in ("unit_price", "line_amount", "discount", "cost_unit", "qty_returned_base"):
        out[c] = df[c].map(parse_number)
    out["_src"] = df["_src"]
    return out[out["d"].notna()]


def unit_factors(biz: Business) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """(tên hàng fold, đơn vị fold) → số đơn vị cơ bản; và tên hàng fold → đơn vị cơ bản.
    Nguồn: bảng quy đổi, danh mục hàng (đơn vị cơ bản), dòng "1 thùng = 24 lon" trong README (§10)."""
    fac: dict[tuple[str, str], float] = {}
    base: dict[str, str] = {}
    for u in biz.readme.units:
        fac[(fold(u.product), fold(u.big_unit))] = u.factor
        base.setdefault(fold(u.product), fold(u.base_unit))
    for t in biz.by_role("units"):
        n, bu, f, b = (t.col("product_name") or t.col("name"), t.col("big_unit"), t.col("factor"), t.col("base_unit"))
        if n and bu and f:
            for row in t.raw.df[[c for c in (n, bu, f, b) if c]].itertuples(index=False):
                x = parse_number(row[2])
                if x:
                    fac[(fold(row[0]), fold(row[1]))] = x
                    if b:
                        base.setdefault(fold(row[0]), fold(row[3]))
    for t in biz.by_role("products"):
        n, b = t.col("product_name") or t.col("name"), t.col("base_unit")
        if n and b:
            for nm, bu in zip(t.raw.df[n], t.raw.df[b]):
                base.setdefault(fold(nm), fold(bu))
    return fac, base


def _fill_qty_base(lines: pd.DataFrame, biz: Business, can: Canonical) -> None:
    """Thiếu cột 'số lượng quy đổi' → qty × hệ số. Đơn vị không tra được → để trống (metric sẽ từ chối)."""
    if lines.empty or lines["qty_base"].notna().mean() > 0.5:
        return
    fac, base = unit_factors(biz)
    if not fac and not base:
        return

    def conv(name: str, unit: str, q: float | None) -> float | None:
        if q is None or pd.isna(q):
            return None
        n, u = fold(name), fold(unit)
        if (n, u) in fac:
            return q * fac[(n, u)]
        if not u or base.get(n) == u or not any(k[0] == n for k in fac):
            return q  # đơn vị cơ bản, hoặc mặt hàng không có đơn vị lớn nào
        return None

    lines["qty_base"] = [conv(n, u, q) for n, u, q in zip(lines["product_name"], lines["unit"], lines["qty"])]
    can.issues.append("qty_base suy từ bảng quy đổi")


def _orders_from_lines(lines: pd.DataFrame) -> pd.DataFrame:
    g = lines.assign(_bad=lines["line_amount"].isna().astype(int)).groupby("order_id", sort=False)
    o = g.agg(d=("d", "first"), dt=("dt", "first"), status=("status", "first"), customer_id=("customer_id", "first"),
              customer_name=("customer_name", "first"), customer_group=("customer_group", "first"),
              payment_method=("payment_method", "first"), channel=("channel", "first"), staff_id=("staff_id", "first"),
              revenue=("line_amount", "sum"), n_lines=("line_amount", "size"), n_bad_amount=("_bad", "sum"))
    # §1: đơn có trả hàng (kể cả một phần) là đơn trả hàng
    flags = pd.DataFrame({
        "order_id": lines["order_id"],
        "ret": (lines["qty_returned_base"].fillna(0).astype(float) > 0) | (lines["status"] == "returned"),
        "cancel": lines["status"] == "cancelled",
    }).groupby("order_id", sort=False)[["ret", "cancel"]].any()
    o.loc[flags["ret"].to_numpy(dtype=bool), "status"] = "returned"
    o.loc[(flags["cancel"] & ~flags["ret"]).to_numpy(dtype=bool), "status"] = "cancelled"
    return o.reset_index()


def _orders_only(tabs: list[Table]) -> pd.DataFrame:
    """Shop chỉ có danh sách hoá đơn: doanh thu = tiền hàng (trước giảm giá đơn, chưa gồm phí giao)."""
    parts = []
    for t in tabs:
        df = t.raw.df
        yr = df["_sheet_year"] if "_sheet_year" in df else pd.Series([None] * len(df), index=df.index)
        o = pd.DataFrame({
            "order_id": df[t.col("order_id")].map(clean),
            "d": [parse_date(v, y if pd.notna(y) else None) for v, y in zip(df[t.col("order_date")], yr)]
            if t.col("order_date") else None,
            "status": df[t.col("status")].map(norm_status).replace({"": "done"}) if t.col("status") else "done",
            "customer_name": df[t.col("customer_name")].map(clean) if t.col("customer_name") else "",
            "revenue": df[t.col("goods_total")].map(parse_number) if t.col("goods_total") else None,
        })
        parts.append(o)
    o = pd.concat(parts, ignore_index=True)
    return o[o["d"].notna()]
