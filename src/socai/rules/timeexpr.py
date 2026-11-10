"""Giải biểu thức thời gian trong câu hỏi thành khoảng ngày, tất định, theo rulebook mặc định.

LLM KHÔNG tự tính ngày: nó (nếu được dùng) chỉ chọn biểu thức; ngày cụ thể luôn do module này tính.
Quy ước mặc định (rulebook §2, có thể bị `RulePack` ghi đè): tuần bắt đầu thứ Hai; "tuần này",
"tháng này", "từ đầu năm" chạy đến hết hôm nay; "tuần rồi" là cả tuần liền trước; tháng/quý không
ghi năm là của năm nay nếu không sau hôm nay, sau hôm nay thì năm trước; "N ngày gần nhất" kể cả
hôm nay; "tháng này so với tháng trước" lấy cùng số ngày của tháng trước.
Hỗ trợ cả câu không dấu, viết tắt (T7/2025, th 6/26, 2/26, hnay) và vài cách nói tiếng Anh/Trung.
"""

from __future__ import annotations

import calendar as _cal
import re
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from socai.io.cells import strip_accents
from socai.rules.calendar import LunarCalendar


@dataclass(frozen=True)
class Period:
    start: date
    end: date  # gồm cả ngày này
    kind: str  # day | week | month | quarter | year | range | ytd | last_n | tet | until
    label: str = ""
    to_today: bool = False  # kỳ "đến hết hôm nay" (tháng này, tuần này...)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


@dataclass
class TimeSpec:
    periods: list[Period] = field(default_factory=list)
    compare_to: Period | None = None
    compare_kind: str | None = None  # prev | same_last_year | explicit
    ambiguous: list[str] = field(default_factory=list)

    @property
    def main(self) -> Period | None:
        return self.periods[0] if self.periods else None


@dataclass(frozen=True)
class TimeRules:
    week_start: int = 0  # 0 = thứ Hai


def tnorm(q: str) -> str:
    """Chữ thường, bỏ dấu, giữ '/', '-', ':' và chữ Hán; gộp khoảng trắng."""
    from socai.nlq.normalize import detect_lang, untelex  # gỡ Telex sót: "thangs", "hoom nay", "nawm"

    fix = untelex if detect_lang(q) not in ("en", "zh", "ja", "ko") else (lambda w: w)
    s = strip_accents(q.lower())
    s = re.sub(r"(\d{4}年\d{1,2}月)", r" \1 ", s)  # chữ Hán không có khoảng trắng giữa các từ
    s = re.sub(r"(?<!\d):|:(?!\d)", " ", s)  # giữ ':' của giờ (14:30), bỏ ':' cuối câu ("tháng 6/2026:")
    s = re.sub(r"[^\w/:\-一-鿿 ]+", " ", s)
    return " " + " ".join(fix(t) for t in s.split()) + " "


def month_bounds(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, _cal.monthrange(y, m)[1])


def shift_month(y: int, m: int, k: int) -> tuple[int, int]:
    i = y * 12 + (m - 1) + k
    return i // 12, i % 12 + 1


def _year_for(month_or_q_end: date, today: date) -> bool:
    return month_or_q_end.replace(day=1) <= today


def _month(y: int, m: int, label: str) -> Period | None:
    if not 1 <= m <= 12:
        return None
    a, b = month_bounds(y, m)
    return Period(a, b, "month", label)


def _month_no_year(m: int, today: date, label: str) -> Period | None:
    if not 1 <= m <= 12:
        return None
    y = today.year if date(today.year, m, 1) <= today else today.year - 1
    return _month(y, m, label)


def _quarter(y: int, q: int, label: str) -> Period | None:
    if not 1 <= q <= 4:
        return None
    a = date(y, 3 * q - 2, 1)
    return Period(a, month_bounds(y, 3 * q)[1], "quarter", label)


def _yy(s: str) -> int:
    return 2000 + int(s) if len(s) == 2 else int(s)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


class Resolver:
    def __init__(self, today: date, cal: LunarCalendar | None = None, rules: TimeRules = TimeRules()):
        self.today, self.cal, self.rules = today, cal, rules
        t = today
        ws = t - timedelta(days=(t.weekday() - rules.week_start) % 7)
        self.week_start = ws
        # (regex, hàm tạo Period từ match); thứ tự = ưu tiên khi chồng lấn
        D, M, Y = r"(\d{1,2})", r"(\d{1,2})", r"(\d{4})"
        self.patterns: list[tuple[str, object]] = [
            (rf" tu (?:ngay )?{D}/{M}/{Y} (?:den|toi|-) (?:het )?(?:ngay )?{D}/{M}/{Y} ", self._range),
            (r" tu dau thang (?:nay )?(?:den|toi) (?:het )?(?:hom nay|hnay|bua nay|nay) ",
             lambda m: Period(t.replace(day=1), t, "month", "tháng này", True)),
            (r" tu dau tuan (?:nay )?(?:den|toi) (?:het )?(?:hom nay|hnay|bua nay|nay) ",
             lambda m: Period(ws, t, "week", "tuần này", True)),
            (rf" (?:tinh )?tu (?:ngay )?{D}/{M} (?:den|toi) (?:het )?(?:hom nay|hnay|nay) ",
             lambda m: self._range_to_today(int(m[1]), int(m[2]))),
            (rf" (?:den het|tinh den(?: het)?|toi het|cuoi|het) (?:ngay )?{D}/{M}/{Y} ", self._until),
            (r" (?:tinh den|den) (?:het )?(?:hom nay|hnay|nay) ", lambda m: Period(t, t, "until", "đến hôm nay")),
            (rf" (?:ngay )?{D}/{M}/{Y} ", self._day),
            (rf" (?:ngay )?{D}-{M}-{Y} ", self._day),
            (rf" {Y}-{M}-{D} ", lambda m: self._mkday(int(m[1]), int(m[2]), int(m[3]))),
            (r" (?:hom nay|bua nay|hnay|today|今天) ", lambda m: Period(t, t, "day", "hôm nay")),
            (r" (?:hom qua|bua qua|hqua|yesterday|昨天) ",
             lambda m: Period(t - timedelta(1), t - timedelta(1), "day", "hôm qua")),
            (r" (?:tuan nay|this week|本周) ", lambda m: Period(ws, t, "week", "tuần này", True)),
            (r" (?:tuan truoc|tuan roi|tuan vua roi|tuan vua qua|last week|上周) ",
             lambda m: Period(ws - timedelta(7), ws - timedelta(1), "week", "tuần trước")),
            (r" (?:thang nay|this month|本月) ", lambda m: Period(t.replace(day=1), t, "month", "tháng này", True)),
            (r" (?:thang truoc|thang roi|last month|上个月) ", self._prev_month),
            (r" (?:tu dau nam(?: (?:toi|den) (?:nay|gio|hnay|hom nay))?|year to date|ytd|今年以来) ",
             lambda m: Period(date(t.year, 1, 1), t, "ytd", "từ đầu năm", True)),
            (r" (?:nam nay|this year|今年) ", lambda m: Period(date(t.year, 1, 1), t, "year", "năm nay", True)),
            (r" (?:nam ngoai|nam truoc|last year|去年) ",
             lambda m: Period(date(t.year - 1, 1, 1), date(t.year - 1, 12, 31), "year", "năm ngoái")),
            (r" (\d+) ngay (?:gan nhat|gan day|qua|vua qua|vua roi) ", self._last_n),
            (r" (?:last|past) (\d+) days ", self._last_n),
            (r" tet (?:nam )?(\d{4}) ", self._tet),
            (rf" quy (\d)(?:/| nam | |-){Y} ", lambda m: _quarter(int(m[2]), int(m[1]), f"quý {m[1]}/{m[2]}")),
            (rf" q(\d)(?:/| |-){Y} ", lambda m: _quarter(int(m[2]), int(m[1]), f"quý {m[1]}/{m[2]}")),
            (r" quy (\d) ", self._quarter_no_year),
            (rf" (?:thang|th|t) ?{M}(?:/| nam | |-){Y} ", lambda m: _month(int(m[2]), int(m[1]), f"{m[1]}/{m[2]}")),
            (rf" (?:thang|th|t) ?{M}/(\d{{2}}) ", lambda m: _month(_yy(m[2]), int(m[1]), f"{m[1]}/{m[2]}")),
            (rf" {M}/{Y} ", lambda m: _month(int(m[2]), int(m[1]), f"{m[1]}/{m[2]}")),
            (rf" {Y}年{M}月 ", lambda m: _month(int(m[1]), int(m[2]), f"{m[2]}/{m[1]}")),
            (r" (\d{1,2})/(\d{2}) ", self._m_yy),
            (r" (?:thang|th) (\d{1,2}) ", lambda m: _month_no_year(int(m[1]), t, f"tháng {m[1]}")),
            (rf" nam {Y} ", lambda m: Period(date(int(m[1]), 1, 1), date(int(m[1]), 12, 31), "year", m[1])),
            (rf" (?:in )?{Y} ", lambda m: Period(date(int(m[1]), 1, 1), date(int(m[1]), 12, 31), "year", m[1])),
        ]
        self.ambiguous: list[str] = []

    # ---- bộ tạo
    def _mkday(self, y: int, m: int, d: int) -> Period | None:
        x = _safe_date(y, m, d)
        return Period(x, x, "day", x.isoformat()) if x else None

    def _day(self, m: re.Match) -> Period | None:
        return self._mkday(int(m[3]), int(m[2]), int(m[1]))

    def _until(self, m: re.Match) -> Period | None:
        x = _safe_date(int(m[3]), int(m[2]), int(m[1]))
        return Period(x, x, "until", f"đến hết {x.isoformat()}") if x else None

    def _range(self, m: re.Match) -> Period | None:
        a = _safe_date(int(m[3]), int(m[2]), int(m[1]))
        b = _safe_date(int(m[6]), int(m[5]), int(m[4]))
        return Period(a, b, "range", f"{a}..{b}") if a and b and a <= b else None

    def _range_to_today(self, d: int, mo: int) -> Period | None:
        a = _safe_date(self.today.year, mo, d)
        return Period(a, self.today, "range", f"{a}..hôm nay", True) if a and a <= self.today else None

    def _prev_month(self, m: re.Match) -> Period:
        y, mo = shift_month(self.today.year, self.today.month, -1)
        a, b = month_bounds(y, mo)
        return Period(a, b, "month", "tháng trước")

    def _last_n(self, m: re.Match) -> Period | None:
        n = int(m[1])
        return Period(self.today - timedelta(n - 1), self.today, "last_n", f"{n} ngày gần nhất", True) if n else None

    def _tet(self, m: re.Match) -> Period | None:
        if self.cal is None:
            self.ambiguous.append("tet: thiếu calendar.csv")
            return None
        w = self.cal.tet_window(int(m[1]))
        return Period(w[0], w[1], "tet", f"Tết {m[1]}") if w else None

    def _quarter_no_year(self, m: re.Match) -> Period | None:
        q = int(m[1])
        if not 1 <= q <= 4:
            return None
        y = self.today.year if date(self.today.year, 3 * q - 2, 1) <= self.today else self.today.year - 1
        return _quarter(y, q, f"quý {q}")

    def _m_yy(self, m: re.Match) -> Period | None:
        a, b = int(m[1]), int(m[2])
        if 1 <= a <= 12 and b > 12:  # 2/26, 11/25: tháng/năm 2 chữ số
            return _month(2000 + b, a, f"{a}/{b}")
        if 1 <= a <= 31 and 1 <= b <= 12:  # 08/04: ngày/tháng không năm → năm nay (mơ hồ)
            self.ambiguous.append(f"'{m[0].strip()}' hiểu là ngày/tháng năm nay")
            d = _safe_date(self.today.year, b, a)
            return Period(d, d, "day", d.isoformat()) if d else None
        return None

    # ---- giải
    def resolve(self, question: str) -> TimeSpec:
        s = tnorm(question)
        found: list[tuple[int, int, Period]] = []
        taken = [False] * len(s)
        for rx, fn in self.patterns:
            # khoảng trắng biên → lookaround, để hai mốc liền nhau ("4/2026 5/2026") không tranh nhau
            rx = re.sub(r"^ ", "(?<= )", re.sub(r" $", "(?= )", rx))
            for m in re.finditer(rx, s):
                a, b = m.start(), m.end()
                if any(taken[a:b]):
                    continue
                p = fn(m)  # type: ignore[operator]
                if p is None:
                    continue
                for i in range(a, b):
                    taken[i] = True
                found.append((a, b, p))
        found.sort(key=lambda x: x[0])
        spec = TimeSpec([p for _, _, p in found], ambiguous=list(self.ambiguous))
        self._comparison(s, spec)
        return spec

    def _comparison(self, s: str, spec: TimeSpec) -> None:
        if not spec.periods:
            return
        same_ly = re.search(r" cung ky (?:nam (?:ngoai|truoc)|nam roi)|same period last year|同比 ", s)
        cmp_word = re.search(r" so (?:voi|vs|vs\.|vơi)| (?:tang|giam|thay doi|tang truong|chenh lech)"
                             r"| lien truoc| ky truoc| truoc do| cung ky| hon hay kem| compared| vs |环比", s)
        if same_ly:  # "cùng kỳ năm trước": cụm "năm trước" trong đó không phải một mốc riêng
            spec.periods = [p for p in spec.periods if p.label != "năm ngoái"] or spec.periods
            spec.compare_to, spec.compare_kind = self.same_last_year(spec.periods[0]), "same_last_year"
            return
        # "so với tháng trước (đó), doanh thu tháng 2/2026…": mốc tương đối là kỳ so sánh của mốc tường minh
        relative = {"tháng trước", "tuần trước", "hôm qua", "năm ngoái"}
        if cmp_word and len(spec.periods) >= 2 and any(p.label in relative for p in spec.periods):
            explicit = [p for p in spec.periods if p.label not in relative]
            if explicit:
                spec.periods = explicit
                spec.compare_to, spec.compare_kind = self.previous(explicit[0]), "prev"
                return
        if len(spec.periods) >= 2 and cmp_word:
            spec.compare_to, spec.compare_kind = spec.periods[1], "explicit"
            return
        if not (cmp_word or same_ly):
            return
        p = spec.periods[0]
        if same_ly:
            spec.compare_to, spec.compare_kind = self.same_last_year(p), "same_last_year"
        else:
            spec.compare_to, spec.compare_kind = self.previous(p), "prev"

    def previous(self, p: Period) -> Period:
        if p.kind == "month":
            y, mo = shift_month(p.start.year, p.start.month, -1)
            a, b = month_bounds(y, mo)
            if p.to_today:  # tháng này so với tháng trước: cùng số ngày, không quá cuối tháng trước
                b = min(b, a + timedelta(p.end.day - 1))
            return Period(a, b, "month", "kỳ trước")
        if p.kind == "quarter":
            y, mo = shift_month(p.start.year, p.start.month, -3)
            return _quarter(y, (mo - 1) // 3 + 1, "quý trước")  # type: ignore[return-value]
        if p.kind in ("year", "ytd"):
            a = date(p.start.year - 1, 1, 1)
            b = date(p.start.year - 1, 12, 31) if not p.to_today else _safe_date(
                p.end.year - 1, p.end.month, min(p.end.day, 28 if p.end.month == 2 else p.end.day))
            return Period(a, b or a, p.kind, "năm trước")
        return Period(p.start - timedelta(p.days), p.start - timedelta(1), p.kind, "kỳ trước")

    def same_last_year(self, p: Period) -> Period:
        def back(d: date) -> date:
            return _safe_date(d.year - 1, d.month, d.day) or date(d.year - 1, d.month, 28)

        if p.kind == "month" and not p.to_today:
            a, b = month_bounds(p.start.year - 1, p.start.month)
            return Period(a, b, "month", "cùng kỳ năm trước")
        return replace(p, start=back(p.start), end=back(p.end), label="cùng kỳ năm trước")


def resolve(question: str, as_of: date, cal: LunarCalendar | None = None) -> TimeSpec:
    return Resolver(as_of, cal).resolve(question)
