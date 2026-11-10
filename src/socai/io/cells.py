"""Chuẩn hoá giá trị ô: tiền, số lượng, ngày, giờ, chữ.

Dữ liệu thật ghi cùng một giá trị theo nhiều cách: `86k`, `53.500đ`, `2,760,000.0000`, `9.656.666đ`,
ngày dạng số serial Excel `45717`, `01/02/2026`, `1 thg 7 2025`, số lượng `5 chai`.
Mọi hàm ở đây tất định, không gọi LLM, trả None khi không đọc được (không đoán).
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime, time, timedelta

EXCEL_EPOCH = date(1899, 12, 30)


# ---------------------------------------------------------------- chữ
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def clean(v: object) -> str:
    """Chuỗi đã NFC, bỏ khoảng trắng thừa; None/NaN → ''."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return " ".join(nfc(str(v)).split())


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def fold(v: object) -> str:
    """Khoá so khớp lỏng: chữ thường, bỏ dấu, ký tự không phải chữ/số thành khoảng trắng."""
    s = strip_accents(clean(v).lower())
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s).split())


def match_key(v: object) -> str:
    """Khoá so khớp đúng luật chấm chuỗi của BTC: NFC, chữ thường, ký tự không phải chữ/số/khoảng
    trắng thành khoảng trắng, gộp khoảng trắng. GIỮ dấu tiếng Việt."""
    s = clean(v).lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


def is_blank(v: object) -> bool:
    return clean(v) == ""


# ---------------------------------------------------------------- số
_MULT = {"k": 1e3, "nghin": 1e3, "ngan": 1e3, "ng": 1e3, "tr": 1e6, "trieu": 1e6, "ty": 1e9, "ti": 1e9}


def _plain_number(t: str, currency: bool) -> float | None:
    """`t` chỉ gồm chữ số, '.', ',' (có thể có dấu âm)."""
    neg = t.startswith("-")
    t = t.lstrip("+-")
    if not t or not re.fullmatch(r"[\d.,]+", t) or not re.search(r"\d", t):
        return None
    dots, commas = t.count("."), t.count(",")
    if dots and commas:  # dấu xuất hiện sau cùng là dấu thập phân
        if t.rfind(".") > t.rfind(","):
            t = t.replace(",", "")
        else:
            t = t.replace(".", "").replace(",", ".")
    elif dots:
        groups = t.split(".")
        if dots > 1 or (currency and len(groups[-1]) == 3) or (
                len(groups[-1]) == 3 and len(groups[0]) <= 3 and groups[0] != "0"):
            # 9.656.666 · 53.500đ · 1.500 (kiểu Việt Nam: chấm phân cách nghìn)
            if all(len(g) == 3 for g in groups[1:]):
                t = t.replace(".", "")
    elif commas:
        groups = t.split(",")
        if all(len(g) == 3 for g in groups[1:]) and len(groups[0]) <= 3:
            t = t.replace(",", "")  # 22,000 · 2,760,000
        elif commas == 1:
            t = t.replace(",", ".")  # 1,5 (thập phân kiểu Việt Nam)
        else:
            return None
    try:
        x = float(t)
    except ValueError:
        return None
    return -x if neg else x


_RE_MONEY = re.compile(r"([-+]?[\d.,]+)(k|nghin|ngan|ng|tr|trieu|ty|ti)?(vnd|dong|d)?")


def parse_number(v: object) -> float | None:
    """Đọc số/tiền: 86k → 86000, 53.500đ → 53500, '2,760,000.0000' → 2760000, '(1,000)' → -1000,
    '1,5tr' → 1500000. Không đọc được → None."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = strip_accents(clean(v).lower()).replace(" ", "").replace("₫", "d")
    neg = s.startswith("(") and s.endswith(")")
    # '1tr590' = 1,590 triệu, '2ty3' = 2,3 tỷ, '1k5' = 1.500: phần sau hậu tố là phần thập phân
    if (mm := re.fullmatch(r"([-+]?\d+)(k|tr|trieu|ty|ti)(\d{1,3})(?:vnd|dong|d)?", s.strip("()"))):
        x = float(f"{mm.group(1)}.{mm.group(3)}") * _MULT[mm.group(2)]
        return float(round(-x if neg else x, 6))
    m = _RE_MONEY.fullmatch(s.strip("()"))
    if not m:
        return None
    num = m.group(1)
    if m.group(2) and num.count(",") + num.count(".") == 1:
        num = num.replace(",", ".")  # có hậu tố k/tr/tỷ: '1,156tr' = 1.156 triệu, dấu là thập phân
        x = float(num) if re.fullmatch(r"[-+]?\d*\.?\d+", num) else None
    else:
        x = _plain_number(num, currency=bool(m.group(3)))
    if x is None:
        return None
    if m.group(2):
        x *= _MULT[m.group(2)]
    return float(round(-x if neg else x, 6))


_RE_QTY = re.compile(r"^\s*([-+]?[\d.,]+)\s*(.*)$")


def parse_qty(v: object) -> tuple[float | None, str]:
    """'5 chai' → (5.0, 'chai'); 3 → (3.0, ''); 'hai' → (None, '')."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (None, "") if (isinstance(v, float) and math.isnan(v)) else (float(v), "")
    m = _RE_QTY.match(clean(v))
    if not m:
        return None, ""
    x = _plain_number(m.group(1).replace(" ", ""), currency=False)
    return x, m.group(2).strip()


def to_int_money(x: float | None) -> int | None:
    """Tiền về đồng, nửa đồng làm tròn lên (rulebook §9)."""
    if x is None:
        return None
    return int(math.floor(x + 0.5 + 1e-9))


def round2(x: float) -> float:
    """Làm tròn 2 chữ số thập phân kiểu nửa lên, tránh lỗi dấu phẩy động của round()."""
    return math.floor(abs(x) * 100 + 0.5 + 1e-9) / 100 * (1 if x >= 0 else -1)


# ---------------------------------------------------------------- ngày giờ
_VI_MONTH = r"(?:thg|thang|th)"


def excel_serial_to_date(n: float) -> date:
    return EXCEL_EPOCH + timedelta(days=int(n))


def parse_date(v: object, default_year: int | None = None) -> date | None:
    """Ngày ghi kiểu ngày trước tháng (Việt Nam). Không có năm → `default_year` (năm của sheet)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        return excel_serial_to_date(v) if 20000 <= v <= 80000 else None
    s = fold(v)
    if not s:
        return None
    if re.fullmatch(r"\d{5}(\s0)?", s):
        return excel_serial_to_date(int(s.split()[0]))
    m = re.fullmatch(r"(\d{4}) (\d{1,2}) (\d{1,2})(?: \d{1,2} \d{2}(?: \d{2})?)?", s)
    if m:
        return _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(rf"(?:ngay )?(\d{{1,2}}) (?:{_VI_MONTH} )?(\d{{1,2}})(?: (?:nam )?(\d{{2,4}}))?"
                     r"(?: \d{1,2} \d{2}(?: \d{2})?)?", s)
    if m:
        y = m.group(3)
        year = (2000 + int(y) if len(y) == 2 else int(y)) if y else default_year
        return _mk(year, int(m.group(2)), int(m.group(1))) if year else None
    return None


def _mk(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_datetime(v: object, default_year: int | None = None) -> datetime | None:
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime.combine(v, time())
    d = parse_date(v, default_year)
    if d is None:
        return None
    t = parse_time(v)
    return datetime.combine(d, t or time())


def parse_time(v: object) -> time | None:
    if isinstance(v, datetime):
        return v.time()
    if isinstance(v, time):
        return v
    if isinstance(v, float) and 0 <= v < 1:  # phần giờ của serial Excel
        secs = round(v * 86400)
        return time(secs // 3600 % 24, secs // 60 % 60, secs % 60)
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", clean(v))
    if not m:
        return None
    h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return time(h, mi, se) if h < 24 and mi < 60 and se < 60 else None


def parse_bool(v: object) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return bool(v)
    s = fold(v)
    if s in {"true", "1", "x", "co", "yes", "y", "dung", "ke don"}:
        return True
    if s in {"false", "0", "khong", "no", "n", "sai", ""}:
        return False if s else None
    return None
