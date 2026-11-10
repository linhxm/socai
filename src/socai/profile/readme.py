"""Đọc README.md của một doanh nghiệp: mô tả cột, quy ước, ngày cuối dữ liệu, quy đổi đơn vị.

README có chất lượng không đều (đề bài nói rõ): có shop đủ từ điển dữ liệu, có shop gần như trống.
Parser chỉ lấy những gì có cấu trúc; phần văn bản tự do giữ nguyên để đưa cho LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from socai.io.cells import clean, fold, parse_date, parse_number


@dataclass
class UnitRule:
    product: str
    big_unit: str
    factor: float
    base_unit: str


@dataclass
class Readme:
    text: str
    intro: str = ""
    files: list[str] = field(default_factory=list)
    columns: dict[str, dict[str, str]] = field(default_factory=dict)  # file/bảng → {cột: mô tả}
    conventions: list[str] = field(default_factory=list)
    data_end: date | None = None
    units: list[UnitRule] = field(default_factory=list)

    def known_headers(self) -> set[str]:
        return {fold(c) for cols in self.columns.values() for c in cols}

    def describe(self, table: str, column: str, sheets: list[str] | None = None) -> str | None:
        """Mô tả của cột; khớp theo 'file::sheet' trước, rồi tên gốc, rồi phần sau '::'."""
        file = table.split("::")[0]
        keys = [table] + [f"{file}::{s}" for s in (sheets or [])] + [file, table.split("::")[-1]]
        for key in keys:
            cols = self.columns.get(key)
            if cols and column in cols:
                return cols[column]
        hits = {d for cols in self.columns.values() for c, d in cols.items() if c == column}
        return hits.pop() if len(hits) == 1 else None


_RE_SECTION = re.compile(r"^##\s+(.+)$", re.M)
_RE_COLS_HEAD = re.compile(r"Cột trong [`'\"]?([^`'\"]+)[`'\"]?", re.I)
_RE_COL = re.compile(r"^-\s+\*\*(.+?)\*\*\s*:\s*(.+)$")
_RE_UNIT = re.compile(r"^\s*-\s+(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$")
_RE_END = re.compile(r"đến hết\s+(\d{1,2}/\d{1,2}/\d{4})")


def parse_readme(text: str) -> Readme:
    rd = Readme(text=text)
    parts = _RE_SECTION.split(text)
    rd.intro = clean(re.sub(r"^#.*$", "", parts[0], flags=re.M))
    for head, body in zip(parts[1::2], parts[2::2]):
        lines = body.strip().splitlines()
        if fold(head).startswith("file"):
            rd.files = [m.group(1) for ln in lines if (m := re.search(r"`([^`]+)`", ln))]
        elif (m := _RE_COLS_HEAD.search(head)) is not None:
            # "Cột trong `file.xlsx`, sheet `Tên sheet`": mỗi sheet một bộ mô tả riêng (không ghi đè nhau)
            sh = re.search(r"sheet\s+[`'\"]([^`'\"]+)[`'\"]", head, re.I)
            key = f"{m.group(1).strip()}::{sh.group(1).strip()}" if sh else m.group(1).strip()
            cols = rd.columns.setdefault(key, {})
            for ln in lines:
                if (c := _RE_COL.match(ln.strip())) is not None:
                    cols[c.group(1).strip()] = c.group(2).strip()
        elif fold(head).startswith("quy uoc") or fold(head).startswith("bang"):
            rd.conventions.extend(ln.strip() for ln in lines if ln.strip())
    if (m := _RE_END.search(text)) is not None:
        rd.data_end = parse_date(m.group(1))
    for ln in text.splitlines():
        if (u := _RE_UNIT.match(ln)) is not None and (f := parse_number(u.group(3))):
            rd.units.append(UnitRule(u.group(1).strip(), u.group(2).strip(), f, u.group(4).strip()))
    return rd


def load_readme(biz_dir: Path) -> Readme:
    p = biz_dir / "README.md"
    return parse_readme(p.read_text(encoding="utf-8")) if p.exists() else Readme(text="")
