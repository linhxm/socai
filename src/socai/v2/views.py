"""Bảng mà não v2 nhìn thấy: tên cột tiếng Việt, kiểu cột, mô tả, giá trị mẫu. Luật rulebook nằm Ở ĐÂY.

- `ban`: dòng hàng (bảng chuẩn). Mặc định chỉ đơn hợp lệ §1; lọc `trang_thai` thì bỏ mặc định.
  `so_luong` đã quy về đơn vị cơ bản (§3), `khach` đã gộp cách viết (§3), dòng phí phục vụ đánh dấu.
- `don`: mỗi đơn một dòng, cùng mặc định §1.
- Mọi bảng khác của shop: đọc kiểu cột bằng dữ liệu (≥ 90% ô đọc được), mô tả lấy từ README; có cột
  trạng thái thì mặc định bỏ dòng huỷ (§5). Không viết luật riêng cho từng loại bảng.
- Danh mục nối được với mặt hàng (trùng mã/tên ≥ 60%) → thuộc tính của nó nối vào `ban` (kê đơn, hoạt chất…).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from socai.io.cells import clean, fold, parse_date, parse_number
from socai.profile.business import Business, Table
from socai.semantic.canonical import VALID, Canonical, norm_status, unit_factors

STATUS_VI = {"done": "hoàn thành", "shipping": "đang giao", "cancelled": "huỷ", "returned": "trả hàng"}
VALID_VI = tuple(STATUS_VI[s] for s in VALID)
WALKIN = {"", "khach le", "khach vang lai", "vang lai", "le", "kl", "khachle", "nan", "none"}
NON_ITEM = ("phi phuc vu", "phi dich vu", "service charge", "phi giao hang", "phi ship")
CANON_ROLES = {"order_lines", "orders"}


@dataclass
class Col:
    name: str
    kind: str  # money | qty | num | cat | date | hour | id
    desc: str = ""
    samples: list[str] = field(default_factory=list)


@dataclass
class View:
    name: str
    desc: str
    df: pd.DataFrame
    cols: dict[str, Col]
    date_col: str | None = None
    status_col: str | None = None
    default_status: tuple[str, ...] | None = None  # giá trị trạng thái giữ lại khi câu không lọc trạng thái
    default_desc: str = ""
    code_of: dict[str, str] = field(default_factory=dict)  # cột hiển thị → cột mã (xếp bằng nhau theo mã, §9)
    key_of: dict[str, str] = field(default_factory=dict)  # cột hiển thị → cột khoá nhóm (khách §3)
    asof: Callable[[date], pd.DataFrame] | None = None  # bảng trạng thái "tại hết ngày X" (công nợ)
    all_status: tuple[str, ...] = ()  # cột mà luật tính trên MỌI đơn (giá bán §5): không áp lọc trạng thái mặc định
    _index: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def span(self) -> tuple[date | None, date | None]:
        if not self.date_col or self.df.empty:
            return None, None
        s = self.df[self.date_col].dropna()
        return (s.min(), s.max()) if len(s) else (None, None)

    def values(self, col: str) -> dict[str, str]:
        """{giá trị đã fold → giá trị gốc} của một cột chữ (đáp án tên phải lấy đúng như trong file)."""
        if col not in self._index:
            vals: dict[str, str] = {}
            for v in pd.unique(self.df[col].dropna()):
                if (k := fold(v)) and k not in ("nan", "none"):
                    vals.setdefault(k, clean(v))
            self._index[col] = vals
        return self._index[col]

    def card(self, max_samples: int = 2, want: Callable[[Col], bool] | None = None) -> str:
        """Mô tả gọn cho prompt: một dòng mỗi cột. `want` lọc cột nối thêm (tên có dấu chấm) theo câu hỏi."""
        head = f"Bảng `{self.name}`: {self.desc}"
        if self.default_desc:
            head += f" Mặc định: {self.default_desc}"
        lines = [head]
        for c in self.cols.values():
            if want is not None and "." in c.name and not want(c):
                continue
            s = f"- {c.name} ({c.kind})"
            if c.desc:
                s += f": {c.desc[:70]}"
            if c.samples:
                s += " · vd: " + " | ".join(x[:24] for x in c.samples[:max_samples])
            lines.append(s)
        return "\n".join(lines)


def _samples(s: pd.Series, k: int = 4) -> list[str]:
    vc = s.map(clean)
    vc = vc[vc.ne("")].value_counts()
    return [str(x) for x in vc.index[:k]]


def _num_or_none(v: object) -> float | None:
    x = parse_number(v)
    return None if x is None or pd.isna(x) else float(x)


# ------------------------------------------------------------------ ban / don (bảng chuẩn)
def _customer_key(cid: pd.Series, name: pd.Series, has_ids: bool) -> pd.Series:
    """§3: có mã khách → mỗi mã một khách; không có → theo tên, khác cách viết vẫn là một khách."""
    key = cid.map(clean) if has_ids else name.map(fold)
    folded = name.map(fold)
    key = key.where(~(key.map(fold).isin(WALKIN) | (key.eq("") & folded.isin(WALKIN))), "")
    return key


def _display(key: pd.Series, name: pd.Series, cid: pd.Series) -> pd.Series:
    disp = name.map(clean).where(name.map(clean).ne(""), cid.map(clean))
    first = pd.Series(disp.values, index=key.values)
    first = first[~first.index.duplicated()]
    return key.map(first).where(key.ne(""), "")


def _hours(dt: pd.Series) -> pd.Series:
    """Giờ tạo đơn; sổ chỉ ghi ngày (mọi giờ đều 00:00) thì coi như không có giờ."""
    h = dt.map(lambda t: t.hour + t.minute / 60 if t is not None and not pd.isna(t) else None)
    if h.dropna().eq(0).all():
        return pd.Series([None] * len(dt), index=dt.index)
    return dt.map(lambda t: t.hour if t is not None and not pd.isna(t) else None)


def sales_views(can: Canonical, base_unit: dict[str, str] | None = None) -> list[View]:
    if can.orders is None or can.orders.empty:
        return []
    if can.lines is None or can.lines.empty:  # shop chỉ có danh sách hoá đơn
        has = can.orders["customer_id"].map(clean).ne("").mean() > 0.3 if "customer_id" in can.orders else False
        return [_orders_view(can, has)]
    O = can.orders.set_index("order_id")
    L = can.lines.copy()
    for c in ("customer_group", "channel", "staff_id", "payment_method", "customer_id", "customer_name"):
        if c in O:  # trường cấp đơn lấy từ bảng đơn (đã điền theo nhóm §8)
            L[c] = L["order_id"].map(O[c]).fillna(L[c] if c in L else "")
    ostatus = L["order_id"].map(O["status"]).fillna(L["status"])
    has_ids = L["customer_id"].map(clean).ne("").mean() > 0.3
    ckey = _customer_key(L["customer_id"], L["customer_name"], has_ids)
    qty_base = L["qty_base"].where(L["qty_base"].notna(), L["qty"])
    factor = (L["qty_base"] / L["qty"]).where(L["qty_base"].notna() & L["qty"].gt(0), 1.0)
    hours = _hours(L["dt"])
    bu = base_unit or {}
    base = L["product_name"].map(lambda n: bu.get(fold(n), ""))
    # không có bảng quy đổi: mặt hàng chỉ bán theo MỘT đơn vị thì đó là đơn vị cơ bản
    units = L.assign(_n=L["product_name"].map(fold), _u=L["unit"].map(fold)).groupby("_n")["_u"].agg(
        lambda u: u.iloc[0] if u[u.ne("")].nunique() == 1 else "")
    base = base.where(base.ne(""), L["product_name"].map(fold).map(units).fillna(""))
    same = L["qty_base"].notna() & (L["qty_base"] == L["qty"]) & L["unit"].map(fold).ne("")
    if same.any():  # mặt hàng bán nhiều đơn vị: đơn vị của dòng không quy đổi là đơn vị cơ bản
        by_unit = L[same].assign(_n=L["product_name"].map(fold), _u=L["unit"].map(fold)).groupby("_n")["_u"].agg(
            lambda u: u.value_counts().index[0])
        base = base.where(base.ne(""), L["product_name"].map(fold).map(by_unit).fillna(""))
    # §8: các tên cùng chỉ một thứ là một thứ; file có mã thì theo mã. Khoá mặt hàng = mã, không có thì tên bỏ dấu
    code = L["product_code"].map(clean)
    mh_key = code.where(code.ne(""), L["product_name"].map(fold))
    b = pd.DataFrame({
        "ngay": L["d"], "gio": hours, "ma_don": L["order_id"] if can.has_order_id else "",
        "trang_thai": ostatus.map(STATUS_VI).fillna("hoàn thành"),
        "khach": _display(ckey, L["customer_name"], L["customer_id"]), "_khach_key": ckey,
        "ma_khach": L["customer_id"].map(clean), "nhom_khach": L["customer_group"].map(clean),
        "mat_hang": L["product_name"].map(clean), "ma_hang": L["product_code"].map(clean),
        "nhom_hang": L["product_group"].map(clean), "don_vi": L["unit"].map(clean),
        "don_vi_co_ban": base, "so_luong": qty_base, "don_gia": L["unit_price"],
        "don_gia_co_ban": L["unit_price"] / factor, "thanh_tien": L["line_amount"], "giam_gia": L["discount"],
        "thanh_toan": L["payment_method"].map(clean), "kenh": L["channel"].map(clean),
        "nhan_vien": L["staff_id"].map(clean), "_ma_don": L["order_id"], "_mh_key": mh_key,
    }, index=L.index)
    b["_la_mat_hang"] = ~b["mat_hang"].map(fold).str.startswith(NON_ITEM)
    if L["cost_unit"].notna().any():
        b["gia_von"] = L["cost_unit"] * L["qty"]
        b["lai_gop"] = L["line_amount"] - b["gia_von"]
    cols = [
        Col("ngay", "date", "ngày đặt đơn (§1: đơn thuộc kỳ theo ngày đặt)"),
        Col("gio", "hour", "giờ tạo đơn 0–23; ca sáng: gio >= 6 và gio <= 13; ca chiều: gio >= 14 và gio <= 21"),
        Col("ma_don", "id", "mã đơn; số đơn = count_distinct ma_don"),
        Col("trang_thai", "cat", "trạng thái đơn", list(STATUS_VI.values())),
        Col("khach", "cat", "khách (đã gộp cách viết; khách lẻ bị bỏ khi nhóm/đếm)"),
        Col("ma_khach", "id", "mã khách"), Col("nhom_khach", "cat", "nhóm khách"),
        Col("mat_hang", "cat", "tên mặt hàng / món"), Col("ma_hang", "id", "mã hàng"),
        Col("nhom_hang", "cat", "nhóm hàng"), Col("don_vi", "cat", "đơn vị bán ghi trên dòng"),
        Col("don_vi_co_ban", "cat", "đơn vị cơ bản của mặt hàng (hỏi 'bán bao nhiêu kg/lon' thì lọc cột này)"),
        Col("so_luong", "qty", "số lượng quy về đơn vị cơ bản (bán chạy = sum so_luong)"),
        Col("don_gia", "money", "đơn giá ghi trên dòng"),
        Col("don_gia_co_ban", "money", "giá bán MỘT đơn vị cơ bản, trước chiết khấu (hỏi giá bán; tự tính cả đơn huỷ/trả)"),
        Col("thanh_tien", "money", "thành tiền dòng = doanh thu (§1)"), Col("giam_gia", "money", "giảm giá trên dòng"),
        Col("thanh_toan", "cat", "hình thức thanh toán ghi trên đơn"), Col("kenh", "cat", "kênh bán"),
        Col("nhan_vien", "cat", "nhân viên bán"),
    ]
    if "lai_gop" in b:
        cols += [Col("gia_von", "money", "giá vốn của dòng"), Col("lai_gop", "money", "lãi gộp = thanh_tien − gia_von")]
    cols = [c for c in cols if c.kind in ("money", "qty") or b[c.name].notna().any() and
            not b[c.name].map(clean).eq("").all()]
    for c in cols:
        if c.kind in ("cat",) and not c.samples:
            c.samples = _samples(b[c.name])
    ban = View("ban", "mỗi dòng là một mặt hàng trong một đơn bán.", b, {c.name: c for c in cols}, "ngay",
               "trang_thai", VALID_VI, "chỉ đơn hoàn thành/đang giao (bỏ huỷ, bỏ cả đơn có trả hàng); "
               "lọc trang_thai '*' để lấy mọi đơn.",
               code_of={"mat_hang": "ma_hang"}, key_of={"khach": "_khach_key", "mat_hang": "_mh_key"},
               all_status=("don_gia_co_ban",))
    return [ban, _orders_view(can, has_ids)]


def _orders_view(can: Canonical, has_ids: bool) -> View:
    Od = can.orders.copy()
    okey = _customer_key(Od["customer_id"] if "customer_id" in Od else pd.Series("", index=Od.index),
                         Od["customer_name"] if "customer_name" in Od else pd.Series("", index=Od.index), has_ids)
    d = pd.DataFrame({
        "ma_don": Od["order_id"] if (can.has_order_id or can.lines is None) else "", "ngay": Od["d"],
        "gio": _hours(Od["dt"]) if "dt" in Od else None,
        "trang_thai": Od["status"].map(STATUS_VI).fillna("hoàn thành"),
        "khach": _display(okey, Od.get("customer_name", pd.Series("", index=Od.index)),
                          Od.get("customer_id", pd.Series("", index=Od.index))),
        "_khach_key": okey,
        "nhom_khach": Od.get("customer_group", pd.Series("", index=Od.index)).map(clean),
        "thanh_toan": Od.get("payment_method", pd.Series("", index=Od.index)).map(clean),
        "kenh": Od.get("channel", pd.Series("", index=Od.index)).map(clean),
        "nhan_vien": Od.get("staff_id", pd.Series("", index=Od.index)).map(clean),
        "doanh_thu_don": Od["revenue"].where((Od["n_bad_amount"] if "n_bad_amount" in Od else pd.Series(0, index=Od.index)) == 0),
        "so_dong": Od.get("n_lines"), "_ma_don": Od["order_id"],
    }, index=Od.index)
    dcols = [Col("ma_don", "id", "mã đơn; số đơn = count ma_don"), Col("ngay", "date", "ngày đặt"),
             Col("gio", "hour", "giờ tạo đơn; ca sáng: gio >= 6 và gio <= 13; ca chiều: gio >= 14 và gio <= 21"),
             Col("trang_thai", "cat", "trạng thái đơn", list(STATUS_VI.values())),
             Col("khach", "cat", "khách (khách lẻ bị bỏ khi nhóm/đếm)"), Col("nhom_khach", "cat", "nhóm khách"),
             Col("thanh_toan", "cat", "hình thức thanh toán"), Col("kenh", "cat", "kênh bán"),
             Col("nhan_vien", "cat", "nhân viên"), Col("doanh_thu_don", "money", "doanh thu của đơn (§1)"),
             Col("so_dong", "num", "số dòng hàng của đơn")]
    dcols = [c for c in dcols if c.name in d and d[c.name].notna().any() and not d[c.name].map(clean).eq("").all()]
    for c in dcols:
        if c.kind == "cat" and not c.samples:
            c.samples = _samples(d[c.name])
    return View("don", "mỗi dòng là một đơn bán.", d, {c.name: c for c in dcols}, "ngay", "trang_thai", VALID_VI,
                "chỉ đơn hoàn thành/đang giao; lọc trang_thai '*' để lấy mọi đơn (vd tỉ lệ huỷ).",
                key_of={"khach": "_khach_key"})


# ------------------------------------------------------------------ bảng khác: tự đọc kiểu cột
_DATE_HINT = ("ngay", "date", "thoi gian", "time", "han")
_MONEY_HINT = ("tien", "gia", "tong", "thanh toan", "amount", "price", "total", "von", "chi phi", "no")
_STATUS_HINT = ("trang thai", "status", "tinh trang")


def _alias(name: str, used: set[str]) -> str:
    base = name.split("::")[-1]
    base = re.sub(r"\.(xlsx|xls|csv|sqlite|db|json|xml)$", "", base, flags=re.I)
    a = re.sub(r"[^a-z0-9]+", "_", fold(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base))).strip("_")[:28] or "bang"
    out, i = a, 2
    while out in used:
        out, i = f"{a}_{i}", i + 1
    used.add(out)
    return out


def _col_name(c: str, used: set[str]) -> str:
    a = re.sub(r"[^a-z0-9]+", "_", fold(c)).strip("_")[:30] or "cot"
    out, i = a, 2
    while out in used:
        out, i = f"{a}_{i}", i + 1
    used.add(out)
    return out


_BOOL = {"true": "có", "false": "không", "co": "có", "khong": "không", "x": "có", "yes": "có", "no": "không",
         "1": "có", "0": "không"}


def typed_view(t: Table, biz: Business, used: set[str]) -> View | None:
    df = t.raw.df
    if df.empty:
        return None
    yr = df["_sheet_year"] if "_sheet_year" in df else pd.Series([None] * len(df), index=df.index)
    out = pd.DataFrame(index=df.index)
    cols: dict[str, Col] = {}
    names: set[str] = set()
    date_col = status_col = None
    for c in t.raw.columns:
        s = df[c]
        nonblank = s.map(clean).ne("")
        if not nonblank.any():
            continue
        n = _col_name(c, names)
        desc = biz.readme.describe(t.name, c, t.raw.sheets) or biz.readme.describe(t.raw.file, c, t.raw.sheets) or ""
        fc = fold(c) + " " + fold(desc)
        folded = set(s[nonblank].map(fold))
        if folded <= set(_BOOL) and len(folded) <= 2 and not fold(c).startswith(("so ", "sl")):
            out[n] = s.map(lambda v: _BOOL.get(fold(v), ""))
            cols[n] = Col(n, "bool", desc or c, ["có", "không"])
            continue
        nums = s[nonblank].map(_num_or_none)
        dates = [parse_date(v, y if pd.notna(y) else None) for v, y in zip(s, yr)] if any(
            h in fc for h in _DATE_HINT) else None
        if dates is not None and sum(x is not None for x in dates) >= 0.9 * nonblank.sum():
            out[n] = dates
            cols[n] = Col(n, "date", desc or c)
            if date_col is None and "han" not in fold(c).split():  # hạn dùng/hạn trả không phải ngày phát sinh
                date_col = n
        elif nums.notna().mean() >= 0.9 and not fold(c).startswith(("ma", "so dien", "sdt", "id", "dien thoai")):
            out[n] = s.map(_num_or_none)
            cols[n] = Col(n, "money" if any(h in fc for h in _MONEY_HINT) else "num", desc or c)
        else:
            out[n] = s.map(clean)
            kind = "id" if fold(c).startswith(("ma ", "ma_", "id", "so phieu", "so hd")) or fold(c) in ("ma", "id") \
                else "cat"
            cols[n] = Col(n, kind, desc or c, _samples(s) if kind == "cat" else [])
            if status_col is None and any(h in fc for h in _STATUS_HINT):
                status_col = n
    if not cols:
        return None
    v = View(_alias(t.name, used), f"bảng `{t.name}` của shop" + (f" (vai trò: {t.role})" if t.role != "unknown"
                                                                   else ""), out, cols, date_col, status_col)
    _set_status_default(v)
    return v


def _set_status_default(v: View) -> None:
    """§5 và luật chung: bảng có cột trạng thái thì mặc định bỏ dòng huỷ."""
    if not v.status_col:
        return
    keep = tuple(x for x in v.values(v.status_col).values() if norm_status(x) != "cancelled")
    if len(keep) < len(v.values(v.status_col)):
        v.default_status, v.default_desc = keep, f"bỏ dòng huỷ (lọc {v.status_col} '*' để lấy cả dòng huỷ)."


def _key_cols(v: View, unique: bool) -> list[tuple[str, set[str], pd.Series]]:
    out = []
    for c, meta in v.cols.items():
        if meta.kind not in ("cat", "id"):
            continue
        col = v.df[c].map(fold)
        nb = col[col.ne("")]
        vals = set(nb)
        if len(vals) < 3 or (unique and len(vals) < 0.95 * len(nb)):
            continue
        out.append((c, vals, col))
    return out


def _join_parent(views: list[View]) -> None:
    """Bảng chi tiết không có ngày (chi tiết phiếu nhập) → lấy ngày, trạng thái từ bảng cha khớp mã (≥ 60%)."""
    for w in views:
        if w.date_col:
            continue
        hit = None
        for c, wvals, wcol in _key_cols(w, unique=False):
            for p in views:
                if p is w or not p.date_col:
                    continue
                for pc, pvals, pcol in _key_cols(p, unique=True):
                    if len(wvals & pvals) >= 0.6 * len(wvals):
                        hit = (c, wcol, p, pcol)
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            continue
        c, wcol, p, pcol = hit
        m = p.df.assign(_k=pcol).drop_duplicates("_k").set_index("_k")
        w.df["ngay"] = wcol.map(m[p.date_col])
        w.cols["ngay"] = Col("ngay", "date", f"{p.cols[p.date_col].desc} (lấy từ {p.name} theo {c})")
        w.date_col = "ngay"
        if p.status_col and "trang_thai" not in w.cols:
            w.df["trang_thai"] = wcol.map(m[p.status_col])
            w.cols["trang_thai"] = Col("trang_thai", "cat", f"trạng thái (lấy từ {p.name})", p.cols[p.status_col].samples)
            w.status_col = "trang_thai"
            _set_status_default(w)


def _fill_names_from_code(ban: View, views: list[View]) -> None:
    """Dòng bán chỉ ghi mã hàng, tên (và nhóm) để trống → điền theo danh mục khớp mã (≥ 60% mã). Không điền thì các
    dòng đó rơi khỏi mọi câu theo mặt hàng / nhóm hàng."""
    if "ma_hang" not in ban.df:
        return
    code = ban.df["ma_hang"].map(clean)
    miss = ban.df["mat_hang"].map(clean).eq("") & code.ne("")
    if not miss.any():
        return
    codes = set(code[miss].map(fold))
    for w in views:
        name = next((c for c in w.cols if w.cols[c].kind == "cat" and (c == "mat_hang" or re.match(r"^(ten|name)", c))), None)
        if name is None:
            continue
        for c, vals, col in _key_cols(w, unique=True):
            if c == name or len(vals & codes) < 0.6 * len(codes):
                continue
            first = w.df.assign(_k=col).drop_duplicates("_k").set_index("_k")
            k = code[miss].map(fold)
            ban.df.loc[miss, "mat_hang"] = k.map(first[name]).fillna("")
            grp = next((g for g in w.cols if g == "nhom_hang" or g.startswith("nhom_hang")), None)
            if grp and "nhom_hang" in ban.df:
                blank = miss & ban.df["nhom_hang"].map(clean).eq("")
                ban.df.loc[blank, "nhom_hang"] = code[blank].map(fold).map(first[grp]).fillna("")
            return


def _code_from_name(ban: View, views: list[View]) -> None:
    """Dòng bán chỉ ghi tên → điền ma_hang theo danh mục có cả mã và tên (tên ↔ mã một-một). Nhờ đó nối được các
    bảng chỉ ghi mã (danh mục thuốc, lô, chi tiết nhập) và xếp hoà theo mã (§9)."""
    if "ma_hang" not in ban.df or ban.df["ma_hang"].map(clean).ne("").mean() > 0.5:
        return
    names = ban.df["mat_hang"].map(fold)
    nameset = set(names) - {""}
    if not nameset:
        return
    for w in views:
        keys = _key_cols(w, unique=True)
        for nc, nvals, ncol in keys:
            if len(nvals & nameset) < 0.6 * len(nameset):
                continue
            code = next((c for c, _, _ in keys if c != nc and w.cols[c].kind == "id"), None)
            if code is None:
                continue
            pairs = {}
            for n, k in zip(ncol, w.df[code].map(clean)):
                if n and k:
                    pairs.setdefault(n, set()).add(k)
            n2c = {n: next(iter(ks)) for n, ks in pairs.items() if len(ks) == 1}
            have = ban.df["ma_hang"].map(clean).ne("")
            ban.df["ma_hang"] = ban.df["ma_hang"].where(have, names.map(n2c).fillna(""))
            ban.cols["ma_hang"] = Col("ma_hang", "id", f"mã hàng (dòng chỉ ghi tên: lấy theo {w.name}.{code})",
                                      _samples(ban.df["ma_hang"]))
            return


def _name_from_code(ban: View, views: list[View]) -> None:
    """Bảng chỉ ghi mã hàng (chi tiết nhập, lô) → thêm cột mat_hang theo danh mục bán (để lọc theo tên)."""
    if "ma_hang" not in ban.cols:
        return
    code2name = ban.df.assign(_k=ban.df["ma_hang"].map(fold)).drop_duplicates("_k").set_index("_k")["mat_hang"]
    codes = set(code2name.index) - {""}
    for w in views:
        if "mat_hang" in w.cols:
            continue
        for c, vals, col in _key_cols(w, unique=False):
            if len(vals & codes) >= 0.6 * len(vals):
                w.df["mat_hang"] = col.map(code2name)
                w.cols["mat_hang"] = Col("mat_hang", "cat", f"tên mặt hàng (theo {c})", _samples(w.df["mat_hang"]))
                break


def _join_attributes(ban: View, others: list[View], max_unique: int = 500) -> None:
    """Danh mục khớp mặt hàng (mã hoặc tên, ≥ 60% giá trị) → thêm thuộc tính của nó vào `ban` (kê đơn, hoạt chất).
    Chỉ danh mục (mỗi mặt hàng một dòng); bỏ cột gần như duy nhất từng dòng (tên, địa chỉ)."""
    keys = {"ma_hang": set(ban.df["ma_hang"].map(fold)) - {""}, "mat_hang": set(ban.df["mat_hang"].map(fold)) - {""}}
    for v in others:
        cands = _key_cols(v, unique=True)
        # mã trước tên: tên hàng có nhiều cách viết (§8), nối theo tên dễ rơi dòng
        hit = next(((c, col, bk) for bk, bset in keys.items() for c, vals, col in cands
                    if bset and len(vals & bset) >= 0.6 * len(vals)), None)
        if hit is None:
            continue
        c, col, bk = hit
        m = v.df.assign(_k=col).drop_duplicates("_k").set_index("_k")
        for a, am in v.cols.items():
            if a == c or am.kind in ("date", "id") or a == "mat_hang":
                continue
            if am.kind == "cat" and (m[a].nunique() > max_unique or m[a].nunique() >= 0.95 * len(m)):
                continue
            name = f"{v.name}.{a}"
            if name in ban.cols:
                continue
            ban.df[name] = ban.df[bk].map(fold).map(m[a])
            ban.cols[name] = Col(name, am.kind, f"{am.desc} (nối từ {v.name} theo {bk})", am.samples)


def build_views(biz: Business, can: Canonical) -> dict[str, View]:
    _, base_unit = unit_factors(biz)
    views = {v.name: v for v in sales_views(can, base_unit)}
    used = set(views)
    others = [v for t in biz.tables if t.role not in CANON_ROLES and (v := typed_view(t, biz, used)) is not None]
    _join_parent(others)
    from socai.v2.debt import debt_views
    for dv in debt_views(biz, views.get("don")):
        views[dv.name] = dv
    if "ban" in views:
        _fill_names_from_code(views["ban"], others)
        _code_from_name(views["ban"], others)
        _name_from_code(views["ban"], others)
        _join_attributes(views["ban"], others)
    views.update({v.name: v for v in others})
    return views
