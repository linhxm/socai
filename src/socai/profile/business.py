"""Hồ sơ một doanh nghiệp: các bảng đã đọc, nghĩa từng cột (tag), vai trò từng bảng.

Thứ tự gán nghĩa cho một cột: mô tả README → tên cột (lexicon train) → (sau) LLM.
Mọi quyết định được ghi lại trong `mapping_log` để đưa vào log nộp bài và để debug.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socai.io import docs  # noqa: F401  (đăng ký bộ đọc tài liệu mở rộng)
from socai.io.readers import RawTable, read_any
from socai.profile.readme import Readme, load_readme
from socai.semantic.lexicon import tag_from_description, tag_from_header

ROLE_RULES: list[tuple[str, set[str], set[str]]] = [
    # (vai trò, tag bắt buộc, tag không được có)
    ("order_lines", {"order_id", "line_amount"}, {"purchase_id"}),
    ("order_lines", {"product_name", "line_amount"}, {"purchase_id"}),
    ("order_lines", {"order_id", "product_code", "qty"}, {"purchase_id"}),
    ("payments", {"order_id", "receipt_amount"}, {"order_total", "status", "goods_total"}),  # bảng đơn có cột "khách đã trả" vẫn là bảng đơn
    ("orders", {"order_id", "status"}, {"product_name", "product_code"}),
    ("orders", {"order_id", "order_total"}, {"product_name", "product_code"}),
    ("purchase_lines", {"purchase_id", "product_code"}, set()),
    ("purchase_lines", {"purchase_id", "product_name"}, set()),
    ("purchases", {"purchase_id"}, set()),
    ("units", {"factor"}, set()),
    ("prices", {"price_new"}, set()),
    ("lot_sales", {"lot_no", "lot_qty_base"}, set()),
    ("lots", {"lot_no"}, set()),
    ("customers", {"customer_id", "name"}, set()),
    ("customers", {"customer_id", "phone"}, set()),
    ("products", {"product_code", "name"}, set()),
    ("products", {"product_code", "product_name"}, {"order_id"}),
    ("inventory", {"stock_base"}, set()),
    ("deliveries", {"delivery_round"}, set()),
]


@dataclass
class Table:
    raw: RawTable
    tags: dict[str, str] = field(default_factory=dict)  # cột → tag
    role: str = "unknown"

    @property
    def name(self) -> str:
        return self.raw.name

    def col(self, tag: str) -> str | None:
        hits = [c for c, t in self.tags.items() if t == tag]
        return hits[0] if hits else None


@dataclass
class Business:
    biz_id: str
    path: Path
    readme: Readme
    tables: list[Table]
    mapping_log: list[dict] = field(default_factory=list)

    def by_role(self, role: str) -> list[Table]:
        return [t for t in self.tables if t.role == role]

    def summary(self) -> dict:
        return {
            "biz_id": self.biz_id,
            "tables": [{"name": t.name, "role": t.role, "rows": len(t.raw.df),
                        "tags": dict(Counter(t.tags.values()))} for t in self.tables],
            "data_end": self.readme.data_end.isoformat() if self.readme.data_end else None,
            "untagged": sum(len(t.raw.columns) - len(t.tags) for t in self.tables),
        }


def assign_role(tags: set[str]) -> str:
    for role, need, forbid in ROLE_RULES:
        if need <= tags and not (forbid & tags):
            return role
    return "unknown"


def tag_table(raw: RawTable, rd: Readme, log: list[dict]) -> Table:
    t = Table(raw)
    for col in raw.columns:
        desc = rd.describe(raw.name, col, raw.sheets) or rd.describe(raw.file, col, raw.sheets)
        tag, src = tag_from_description(desc), "readme"
        if tag is None:
            tag, src = tag_from_header(col), "header"
        if tag and tag != "empty":
            t.tags[col] = tag
        log.append({"table": raw.name, "column": col, "desc": desc, "tag": tag, "via": src if tag else None})
    # tag "name" (README ghi "tên") giữ nguyên: nghĩa của nó suy theo vai trò bảng ở bước dựng bảng chuẩn
    t.role = assign_role(set(t.tags.values()))
    return t


DATE_TAGS = {"order_date", "order_datetime"}
_DATE_NAME = ("thoi gian", "ngay", "time", "date", "luc")


def infer_missing_dates(t: Table) -> str | None:
    """Bảng đơn / dòng hàng không có cột ngày đã gán: lấy cột chưa gán mà ≥ 90% ô đọc được thành ngày
    (ưu tiên tên chứa 'thời gian/ngày/time/date'; tránh 'ngày tạo khách/hạn'). Kiểm bằng dữ liệu, không đoán theo shop."""
    from socai.io.cells import fold, parse_datetime

    if t.role not in ("orders", "order_lines") or DATE_TAGS & set(t.tags.values()):
        return None
    best = None
    for c in t.raw.columns:
        if c in t.tags:
            continue
        name = fold(c)
        if any(x in name for x in ("han", "sinh", "tao khach", "cap nhat", "giao")):
            continue
        vals = [v for v in t.raw.df[c].head(400) if str(v).strip() not in ("", "None", "nan")]
        if len(vals) < 5:
            continue
        ok = sum(parse_datetime(v) is not None for v in vals) / len(vals)
        if ok >= 0.9:
            score = ok + (0.5 if any(x in name for x in _DATE_NAME) else 0) - (0.3 if "tao" in name else 0)
            if best is None or score > best[0]:
                best = (score, c)
    if best:
        t.tags[best[1]] = "order_datetime"
        return best[1]
    return None


_ID_RE = None


def link_order_ids(tables: list[Table], min_overlap: float = 0.6) -> list[tuple[str, str]]:
    """table.link: cột dạng mã (chữ + số) trùng giá trị với cột mã đơn đã biết, hoặc giữa bảng 'đầu đơn'
    (trạng thái/ngày/tổng tiền) và bảng 'dòng hàng' (mã hàng/thành tiền) → gán order_id. Kiểm bằng dữ liệu."""
    import re

    global _ID_RE
    _ID_RE = _ID_RE or re.compile(r"^[A-Za-z]{1,8}[-_./]?\d{3,}$")

    def id_values(t: Table, c: str) -> tuple[set[str], float] | None:
        vals = [str(v).strip() for v in t.raw.df[c].head(5000) if str(v).strip() not in ("", "None", "nan")]
        if len(vals) < 20 or sum(bool(_ID_RE.match(v)) for v in vals) / len(vals) < 0.9:
            return None
        return set(vals), len(set(vals)) / len(vals)

    from socai.io.cells import fold

    skip = {"phone", "customer_group", "address", "base_unit", "factor", "active_ingredient", "prescription",
            "stock_base", "price_new", "purchase_id", "lot_no", "purchase_qty", "purchase_qty_base", "purchase_price",
            "purchase_amount", "purchase_total", "purchase_date", "purchase_unit", "supplier_id", "supplier_name",
            "purchase_line_no", "purchase_status"}
    purchase_words = ("nhap", "ncc", "nha cung cap", "supplier", "purchase")
    header_tags = {"status", "order_date", "order_datetime", "order_total", "goods_total", "paid", "debt_remaining"}
    line_tags = {"product_code", "product_name", "line_amount", "qty", "unit_price"}
    known: list[set[str]] = [set(t.raw.df[t.col("order_id")].dropna().map(str).str.strip())
                             for t in tables if t.col("order_id") and t.role in ("orders", "order_lines", "payments")]
    heads, lines_ = [], []
    for t in tables:
        tags = set(t.tags.values())
        if t.col("order_id") or tags & skip or any(w in fold(t.name) for w in purchase_words):
            continue  # bảng nhập hàng không bao giờ là đơn bán
        is_head, is_line = bool(tags & header_tags) and not (tags & line_tags), bool(tags & line_tags)
        for c in t.raw.columns:
            if c in t.tags or any(w in fold(c) for w in purchase_words) or (iv := id_values(t, c)) is None:
                continue
            vals, uniq = iv
            if is_head and uniq >= 0.95:  # mã đơn ở bảng đầu đơn: mỗi dòng một mã (mã khách thì lặp)
                heads.append((t, c, vals))
            elif is_line and uniq < 0.95:  # mã đơn ở bảng dòng hàng: lặp theo số dòng của đơn
                lines_.append((t, c, vals))

    def ov(a: set[str], b: set[str]) -> float:
        return len(a & b) / max(min(len(a), len(b)), 1)

    out = []
    for group, others in ((heads, lines_), (lines_, heads)):
        for t, c, vals in group:
            refs = known + [v for (t2, _, v) in others if t2 is not t]
            if not t.col("order_id") and max((ov(vals, r) for r in refs), default=0) >= min_overlap:
                t.tags[c] = "order_id"
                t.role = assign_role(set(t.tags.values()))
                out.append((t.name, c))
    return out


def load_business(biz_dir: Path, cache_dir: Path | None = None) -> Business:
    rd = load_readme(biz_dir)
    known = rd.known_headers()
    log: list[dict] = []
    tables: list[Table] = []
    for p in sorted(biz_dir.iterdir()):
        if p.name.lower() == "readme.md" or not p.is_file():
            continue
        for raw in read_any(p, known, cache_dir):
            tables.append(tag_table(raw, rd, log))
    for tn, c in link_order_ids(tables):
        log.append({"table": tn, "column": c, "tag": "order_id", "via": "data"})
    for t in tables:
        if (c := infer_missing_dates(t)) is not None:
            log.append({"table": t.name, "column": c, "tag": "order_datetime", "via": "data"})
    return Business(biz_dir.name, biz_dir, rd, tables, log)
