"""Đọc CSV / XLSX / SQLite thành `RawTable` (chưa gán nghĩa).

Xử lý những kiểu bẩn gặp trong dữ liệu train/public:
- dòng tiêu đề báo cáo phía trên header ("SỔ CHI TIẾT BÁN HÀNG", "Kỳ báo cáo: ...");
- header hai tầng (tầng nhóm "Hàng / Tiền / Đơn" phía trên tầng tên cột);
- mỗi tháng một sheet cùng cấu trúc → gộp một bảng, giữ `_sheet` và năm/tháng suy từ tên sheet;
- một workbook chứa nhiều danh sách khác cấu trúc → mỗi sheet một bảng.
Các bộ đọc tài liệu khác (PDF, ảnh chụp, XML hoá đơn điện tử...) đăng ký ở `socai.io.docs`.
"""

from __future__ import annotations

import csv
import hashlib
import pickle
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from socai.io.cells import clean, fold, parse_number

HEADER_SCAN_ROWS = 15


@dataclass
class RawTable:
    name: str  # "file.xlsx" hoặc "file.xlsx::sheet" hoặc "pos.sqlite::bang"
    file: str
    df: pd.DataFrame  # mọi cột kiểu object, giữ nguyên giá trị gốc
    sheets: list[str] = field(default_factory=list)
    source_kind: str = "tabular"  # tabular | ocr | pdf | xml ...
    confidence: float = 1.0  # < 1 với nguồn trích xuất (OCR...); dùng cho chính sách bỏ trống
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [c for c in self.df.columns if not c.startswith("_")]


# ---------------------------------------------------------------- header
def _header_score(row: Iterable[object], known: set[str]) -> float:
    cells = [c for c in row if clean(c)]
    if not cells:
        return -1.0
    texty = sum(1 for c in cells if isinstance(c, str) and parse_number(c) is None)
    bonus = sum(5 for c in cells if fold(c) in known)
    # dòng tiêu đề chỉ có 1–2 ô chữ; header thật có nhiều ô chữ, không có số
    return texty + bonus - 2 * (len(cells) - texty)


def detect_header(rows: list[tuple], known: set[str] | None = None) -> int:
    """Chỉ số dòng header trong vài dòng đầu; ưu tiên dòng khớp tên cột README."""
    known = known or set()
    best, best_i = -1e9, 0
    for i, r in enumerate(rows[:HEADER_SCAN_ROWS]):
        s = _header_score(r, known)
        if s > best + 1e-9:
            best, best_i = s, i
    return best_i


def _dedupe(names: list[str]) -> list[str]:
    out, seen = [], {}
    for i, n in enumerate(names):
        n = clean(n) or f"_col{i}"
        k = seen.get(n, 0)
        seen[n] = k + 1
        out.append(n if k == 0 else f"{n}.{k}")
    return out


def _frame(rows: list[tuple], known: set[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    h = detect_header(rows, known)
    width = max(len(r) for r in rows)
    header = list(rows[h]) + [None] * (width - len(rows[h]))
    body = [list(r) + [None] * (width - len(r)) for r in rows[h + 1:]]
    body = [r for r in body if any(clean(c) for c in r)]
    df = pd.DataFrame(body, columns=_dedupe(header), dtype=object)
    empty = [c for c in df.columns if c.startswith("_col") and df[c].map(clean).eq("").all()]
    return df.drop(columns=empty)


# ---------------------------------------------------------------- tên sheet → năm/tháng
_RE_SHEET = re.compile(r"(?:t|thang|th)?\s*(\d{1,2})\D+(\d{4})|(\d{4})\D+(\d{1,2})$|^(\d{4})$")


def sheet_period(name: str) -> tuple[int | None, int | None]:
    """'T02-2025' → (2025, 2); 'Thang 02 2026' → (2026, 2); '2025' → (2025, None)."""
    m = _RE_SHEET.search(fold(name))
    if not m:
        return None, None
    if m.group(1):
        return int(m.group(2)), int(m.group(1))
    if m.group(3):
        return int(m.group(3)), int(m.group(4))
    return int(m.group(5)), None


# ---------------------------------------------------------------- bộ đọc
def read_csv(path: Path, known: set[str]) -> list[RawTable]:
    for enc in ("utf-8-sig", "utf-16", "cp1258", "cp1252"):
        try:
            with path.open(encoding=enc, newline="") as f:
                rows = [tuple(r) for r in csv.reader(f)]
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"không đọc được bảng mã của {path}")
    return [RawTable(path.name, path.name, _frame(rows, known))]


def _xlsx_sheets(path: Path) -> list[tuple[str, list[tuple]]]:
    """calamine (Rust) nhanh hơn openpyxl ~17 lần trên file 100k dòng; openpyxl là dự phòng."""
    try:
        from python_calamine import CalamineWorkbook

        wb = CalamineWorkbook.from_path(str(path))
        return [(sn, [tuple(r) for r in wb.get_sheet_by_name(sn).to_python()]) for sn in wb.sheet_names]
    except ImportError:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            return [(sn, [tuple(r) for r in wb[sn].iter_rows(values_only=True)]) for sn in wb.sheetnames]
        finally:
            wb.close()


def read_xlsx(path: Path, known: set[str]) -> list[RawTable]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for sn, rows in _xlsx_sheets(path):
        df = _frame(rows, known)
        if df.empty:
            continue
        y, m = sheet_period(sn)
        df["_sheet"], df["_sheet_year"], df["_sheet_month"] = sn, y, m
        frames.append((sn, df))
    # gộp các sheet cùng tập cột (sổ chia theo tháng); sheet khác cấu trúc là danh sách riêng
    groups: dict[frozenset, list[tuple[str, pd.DataFrame]]] = {}
    for sn, df in frames:
        groups.setdefault(frozenset(c for c in df.columns if not c.startswith("_")), []).append((sn, df))
    out = []
    for parts in groups.values():
        names = [sn for sn, _ in parts]
        df = pd.concat([d for _, d in parts], ignore_index=True)
        name = path.name if len(groups) == 1 else f"{path.name}::{names[0]}"
        out.append(RawTable(name, path.name, df, names))
    return out


def read_sqlite(path: Path, known: set[str]) -> list[RawTable]:
    uri = f"file:{path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        names = [r[0] for r in con.execute(
            "select name from sqlite_master where type in ('table','view') and name not like 'sqlite_%'")]
        out = []
        for t in names:
            df = pd.read_sql_query(f'select * from "{t}"', con).astype(object)
            out.append(RawTable(f"{path.name}::{t}", path.name, df, [t]))
        return out
    finally:
        con.close()


READERS: dict[str, Callable[[Path, set[str]], list[RawTable]]] = {
    ".csv": read_csv, ".xlsx": read_xlsx, ".xlsm": read_xlsx,
    ".sqlite": read_sqlite, ".db": read_sqlite, ".sqlite3": read_sqlite,
}


def register_reader(ext: str, fn: Callable[[Path, set[str]], list[RawTable]]) -> None:
    READERS[ext.lower()] = fn


def read_any(path: Path, known: set[str] | None = None, cache_dir: Path | None = None) -> list[RawTable]:
    """Đọc một file theo đuôi; có cache pickle theo nội dung file (đọc xlsx lớn mất vài giây)."""
    fn = READERS.get(path.suffix.lower())
    if fn is None:
        return []
    key = None
    if cache_dir is not None:
        h = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        key = cache_dir / f"{h}-{fold(path.name).replace(' ', '_')}.pkl"
        if key.exists():
            return pickle.loads(key.read_bytes())
    tables = fn(path, known or set())
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_bytes(pickle.dumps(tables))
    return tables
