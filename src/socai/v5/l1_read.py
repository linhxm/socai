"""L1: đọc file của shop và làm sạch HÌNH THỨC (không gán nghĩa nghiệp vụ).

Vào: thư mục shop (CSV/XLSX/SQLite + README.md). Ra: danh sách `Table` với cột đã có kiểu
(number | date | datetime | time | text), dòng tổng và header lặp đã bỏ, ô trống của dòng tiếp nối đã điền.
Mọi quyết định ghi vào `Table.notes` để LLM và log đều thấy. Không có gì gắn theo id shop hay ngành.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from socai.io.cells import clean, fold, is_blank, parse_date, parse_datetime, parse_number, parse_qty, parse_time
from socai.io.readers import READERS, RawTable, read_any
from socai.profile.readme import Readme, load_readme

SAMPLE = 2000
RE_CODE = re.compile(r"\b(ma|id|so hd|so hoa don|so phieu|so don|sdt|so dien thoai|dien thoai|imei|serial|bien so|"
                     r"ma vach|barcode|code|phone)\b")
RE_DATE = re.compile(r"\b(ngay|date|han|thoi gian|thoi diem|time|luc)\b")
RE_TIME = re.compile(r"\b(gio|time|luc|thoi gian|thoi diem)\b")
RE_TOTAL = re.compile(r"^(tong|cong|tong cong|total|sum|subtotal)\b")


@dataclass
class Col:
    name: str  # tên gốc trong file
    kind: str  # number | date | datetime | time | text
    desc: str = ""  # mô tả từ README
    unparsed: int = 0  # số ô có giá trị nhưng không đọc được theo kiểu (đã thành NULL)
    filled: bool = False  # ô trống đã điền theo dòng đầu nhóm


@dataclass
class Table:
    name: str
    file: str
    df: pd.DataFrame
    cols: list[Col]
    notes: list[str] = field(default_factory=list)
    confidence: float = 1.0


def as_text(v: object) -> str | None:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = clean(v)
    return s or None


def _is_numeric_like(v: object) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) or bool(re.fullmatch(r"\d+(\.0)?", clean(v)))


def infer_kind(name: str, desc: str, values: list, years: list) -> str:
    """Kiểu cột suy từ dữ liệu; tên và mô tả chỉ dùng để phân xử chỗ mơ hồ (số nguyên là mã hay ngày serial)."""
    if not values:
        return "text"
    hint = fold(f"{name} {desc}")
    n = len(values)
    if all(isinstance(v, datetime) for v in values):
        return "datetime" if any(v.time() != time() for v in values) else "date"
    code = bool(RE_CODE.search(hint))
    dates = [parse_date(v, y) for v, y in zip(values, years)]
    if sum(d is not None for d in dates) / n >= 0.9:
        numeric = sum(_is_numeric_like(v) for v in values) / n
        if numeric < 0.5 or (RE_DATE.search(hint) and not code):
            with_time = sum(1 for v in values if not _is_numeric_like(v) and (t := parse_time(v)) and t != time()) / n
            return "datetime" if with_time >= 0.2 else "date"
    if code:
        return "text"
    if sum(parse_number(v) is not None for v in values) / n >= 0.95:
        return "number"
    if sum(parse_qty(v)[0] is not None for v in values) / n >= 0.95 and \
            sum(bool(parse_qty(v)[1]) for v in values) / n >= 0.5:
        return "qty"
    if RE_TIME.search(hint) and sum(parse_time(v) is not None for v in values) / n >= 0.9:
        return "time"
    return "text"


def _convert(series: pd.Series, kind: str, years: pd.Series | None) -> tuple[list, int, list | None]:
    """Đọc từng ô theo kiểu (nhớ kết quả theo giá trị để nhanh). Trả (giá trị, số ô không đọc được, đơn vị nếu qty)."""
    memo: dict = {}
    out, units, bad = [], ([] if kind == "qty" else None), 0
    ys = years.tolist() if years is not None else [None] * len(series)
    for v, y in zip(series.tolist(), ys):
        if is_blank(v):
            out.append(None)
            if units is not None:
                units.append(None)
            continue
        key = (type(v).__name__, v, y)
        try:
            hit = memo.get(key, memo)
        except TypeError:  # giá trị không băm được
            key, hit = None, memo
        if hit is memo:
            if kind == "number":
                hit = parse_number(v)
            elif kind == "qty":
                x, u = parse_qty(v)
                hit = (x, u or None)
            elif kind == "date":
                hit = parse_date(v, y)
            elif kind == "datetime":
                hit = parse_datetime(v, y)
            elif kind == "time":
                t = parse_time(v)
                hit = t.strftime("%H:%M:%S") if t else None
            else:
                hit = as_text(v)
            if key is not None:
                memo[key] = hit
        if kind == "qty":
            x, u = hit
            out.append(x)
            units.append(u)
            bad += x is None
        else:
            out.append(hit)
            bad += hit is None
    return out, bad, units


def _drop_header_repeats(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    keys = [fold(c) for c in cols]
    hit = pd.Series(0, index=df.index)
    for c, k in zip(cols, keys):
        hit += df[c].map(lambda v, k=k: fold(v) == k and bool(k)).astype(int)
    mask = hit >= max(2, 0.6 * len(cols))
    return df[~mask], int(mask.sum())


def _drop_totals(df: pd.DataFrame, text_cols: list[str], num_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Dòng tổng: một ô chữ bắt đầu bằng 'Tổng/Cộng/Total', các ô chữ khác đa số trống, và có ít nhất một ô số."""
    if not text_cols or not num_cols:
        return df, 0
    label = pd.Series(False, index=df.index)
    blanks = pd.Series(0, index=df.index)
    for c in text_cols:
        f = df[c].map(fold)
        label |= f.map(lambda s: bool(RE_TOTAL.match(s)))
        blanks += (f == "").astype(int)
    has_num = pd.Series(False, index=df.index)
    for c in num_cols:
        has_num |= df[c].notna()
    mask = label & has_num & (blanks >= (len(text_cols) - 1) * 0.6)
    return df[~mask], int(mask.sum())


def _fill_groups(df: pd.DataFrame, cols: list[Col]) -> list[str]:
    """Điền ô trống của dòng tiếp nối (dòng 2..n của một đơn chỉ ghi hàng, bỏ trống mã/ngày/khách).

    Chỉ điền cột chữ và ngày, không điền cột số (điền số sẽ làm cộng trùng). Hai mẫu hình:
    (a) có cột khoá gần như không trống: trong một dãy dòng liên tiếp cùng khoá, cột chỉ có một giá trị → điền theo dãy;
    (b) không có khoá: một nhóm ≥ 2 cột trống cùng lúc trên cùng các dòng → dòng đó tiếp nối dòng trên.
    """
    n = len(df)
    if n < 3:
        return []
    cand = [c for c in cols if c.kind in ("text", "date", "datetime", "time") and 0.02 <= df[c.name].isna().mean() <= 0.95
            and df[c.name].iloc[0] is not None and not pd.isna(df[c.name].iloc[0])]
    if not cand:
        return []
    filled: list[str] = []
    keys = [c for c in cols if c.kind == "text" and df[c.name].isna().mean() <= 0.005 and c not in cand
            and 1 < df[c.name].nunique() < n]
    for c in cand:
        blank = df[c.name].isna()
        for key in keys:
            same_prev = df[key.name] == df[key.name].shift()
            if (same_prev[blank]).mean() < 0.99:  # ô trống phải nằm ở dòng cùng khoá với dòng trên
                continue
            run = (~same_prev).cumsum()
            per = df.groupby(run)[c.name].agg(lambda s: s.dropna().nunique())
            if (per <= 1).mean() >= 0.99:
                df[c.name] = df.groupby(run)[c.name].transform(lambda s: s.ffill())
                c.filled = True
                filled.append(c.name)
                break
    rest = [c for c in cand if not c.filled]
    masks: dict[tuple, list[Col]] = {}
    for c in rest:
        masks.setdefault(tuple(df[c.name].isna().tolist()), []).append(c)
    for _mask, group in masks.items():
        if len(group) >= 2:
            for c in group:
                df[c.name] = df[c.name].ffill()
                c.filled = True
                filled.append(c.name)
    return filled


def clean_table(raw: RawTable, readme: Readme) -> Table | None:
    df = raw.df.copy()
    names = [c for c in df.columns if not str(c).startswith("_")]
    if df.empty or not names:
        return None
    notes = list(raw.notes)
    df, n_head = _drop_header_repeats(df, names)
    if n_head:
        notes.append(f"đã bỏ {n_head} dòng lặp lại tiêu đề cột")
    df = df.reset_index(drop=True)
    years = df["_sheet_year"] if "_sheet_year" in df.columns else None
    cols: list[Col] = []
    data: dict[str, list] = {}
    for name in names:
        desc = readme.describe(raw.name, name, raw.sheets) or ""
        nonblank = [(v, (years.iloc[i] if years is not None else None)) for i, v in enumerate(df[name].tolist())
                    if not is_blank(v)]
        step = max(1, len(nonblank) // SAMPLE)
        sample = nonblank[::step][:SAMPLE]
        kind = infer_kind(name, desc, [v for v, _ in sample], [y for _, y in sample])
        vals, bad, units = _convert(df[name], kind, years)
        col = Col(name, "number" if kind == "qty" else kind, desc, bad)
        cols.append(col)
        data[name] = vals
        if units is not None:
            uname = f"{name} (đơn vị)"
            cols.append(Col(uname, "text", f"đơn vị ghi kèm trong ô của cột '{name}'"))
            data[uname] = units
    if "_sheet" in df.columns and df["_sheet"].nunique() > 1:
        cols.append(Col("_sheet", "text", "tên sheet gốc của dòng"))
        data["_sheet"] = [as_text(v) for v in df["_sheet"].tolist()]
    out = pd.DataFrame(data, dtype=object)
    text_cols = [c.name for c in cols if c.kind == "text" and c.name != "_sheet"]
    num_cols = [c.name for c in cols if c.kind == "number"]
    out, n_tot = _drop_totals(out, text_cols, num_cols)
    if n_tot:
        notes.append(f"đã bỏ {n_tot} dòng tổng/cộng chèn trong sổ")
    out = out.reset_index(drop=True)
    filled = _fill_groups(out, cols)
    if filled:
        notes.append("ô trống ở dòng tiếp nối đã điền theo dòng đầu nhóm: " + ", ".join(filled))
    conf = raw.confidence
    for c in cols:
        if c.unparsed:
            share = c.unparsed / max(len(out), 1)
            notes.append(f"cột '{c.name}': {c.unparsed} ô không đọc được theo kiểu {c.kind} (để trống)")
            if share > 0.005:
                conf = min(conf, 0.8)
    dup = int(out.duplicated().sum())
    if dup:
        notes.append(f"có {dup} dòng trùng hệt dòng khác (chưa bỏ; xem luật trước khi quyết định)")
    return Table(raw.name, raw.file, out, cols, notes, conf)


def read_shop(biz_dir: Path, cache: Path | None = None) -> tuple[Readme, list[Table]]:
    readme = load_readme(biz_dir)
    known = readme.known_headers()
    tables: list[Table] = []
    for p in sorted(biz_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in READERS:
            continue
        for raw in read_any(p, known, cache):
            t = clean_table(raw, readme)
            if t is not None and len(t.df):
                tables.append(t)
    return readme, tables
