"""Đọc một ô theo kiểu (v4-ham §1.5). Dùng lại `io/cells.py`, thêm chỗ còn thiếu và chữ ký hình dạng.

Mọi hàm tất định, trả None khi không đọc được, không đoán.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time

from socai.io.cells import (clean, fold, is_blank, match_key, parse_bool, parse_date, parse_number,  # noqa: F401
                            strip_accents)
from socai.io.cells import parse_datetime as _parse_datetime
from socai.io.cells import parse_qty as _parse_qty
from socai.io.cells import parse_time as _parse_time

KINDS = ("money", "number", "qty", "date", "datetime", "time", "bool", "pct", "code", "text")

_RE_HM = re.compile(r"\b(\d{1,2})\s*[hg:]\s*(\d{2})?\b", re.I)
_RE_AMPM = re.compile(r"\b(am|pm|sa|ch)\b", re.I)  # SA/CH: sáng/chiều trên máy tiếng Việt
_RE_XQTY = re.compile(r"^\s*(?:x\s*([\d.,]+)|([\d.,]+)\s*x)\s*$", re.I)


def parse_time(v: object) -> time | None:
    """'14:35', '14h35', '14g', '2:35 PM', '2:35 CH', phần giờ của serial Excel."""
    if isinstance(v, (datetime, time)) or (isinstance(v, float) and 0 <= v < 1):
        return _parse_time(v)
    s = clean(v)
    if not s:
        return None
    t = _parse_time(s)
    if t is None:
        m = _RE_HM.search(s)
        if not m or "/" in s[m.start():m.end()]:
            return None
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if h > 23 or mi > 59:
            return None
        t = time(h, mi)
    ap = _RE_AMPM.search(s)
    if ap and t.hour < 12 and ap.group(1).lower() in ("pm", "ch"):
        t = t.replace(hour=t.hour + 12)
    if ap and t.hour == 12 and ap.group(1).lower() in ("am", "sa"):
        t = t.replace(hour=0)
    return t


def parse_datetime(v: object, default_year: int | None = None) -> datetime | None:
    d = _parse_datetime(v, default_year)
    if d is None:
        return None
    t = parse_time(v) if not isinstance(v, (int, float)) else None
    return datetime.combine(d.date(), t) if t else d


def parse_qty(v: object) -> tuple[float | None, str]:
    """'2 ly' → (2, 'ly'); 'x2' / '2x' → (2, ''); '1 bao 25kg' → (1, 'bao 25kg')."""
    m = _RE_XQTY.match(clean(v)) if isinstance(v, str) else None
    if m:
        return parse_number(m.group(1) or m.group(2)), ""
    return _parse_qty(v)


def parse_pct(v: object, scale: str = "auto") -> float | None:
    """'5%' → 5; '0,05' với scale='fraction' → 5; 'auto': có % thì là số phần trăm, không thì để nguyên."""
    s = clean(v)
    if not s:
        return None
    x = parse_number(s.replace("%", ""))
    if x is None:
        return None
    return x * 100 if scale == "fraction" else x


def parse_code(v: object) -> str | None:
    s = clean(v)
    return s or None


def parse_cell(v: object, kind: str, year: int | None = None):
    """Đọc một ô theo `kind` (KINDS). qty trả số (đơn vị chữ lấy bằng `parse_qty`)."""
    if is_blank(v):
        return None
    if kind in ("money", "number"):
        return parse_number(v)
    if kind == "qty":
        return parse_qty(v)[0]
    if kind == "date":
        return parse_date(v, year)
    if kind == "datetime":
        return parse_datetime(v, year)
    if kind == "time":
        return parse_time(v)
    if kind == "bool":
        return parse_bool(v)
    if kind == "pct":
        return parse_pct(v)
    if kind == "code":
        return parse_code(v)
    if kind == "text":
        return clean(v)
    raise ValueError(f"kind lạ: {kind}")


def shape(v: object) -> str:
    """Chữ ký hình dạng của ô để gom mẫu: '2tr900' → '9tr9', '01/04/2025' → '9/9/9', 'Tống Văn A' → 'w w a'.
    Giữ chữ ngắn (k, tr, đ, thg, ly…) vì chúng mang nghĩa đơn vị; chữ dài thành 'w'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "∅"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "num" if float(v) != int(v) else ("int5" if 20000 <= v <= 80000 else "int")
    if isinstance(v, datetime):
        return "datetime"
    if isinstance(v, date):
        return "date"
    s = clean(v).lower()
    if not s:
        return "∅"
    s = re.sub(r"\d+", "9", s)
    s = re.sub(r"[^\W\d_]{4,}", "w", s)
    return s[:24]
