"""Bảng thô của L1 (v4-ham §1.1–1.2): nạp ô thô của mọi file/sheet, dò header, dựng bảng có nguồn từng dòng.

`Grid` = ô thô của một sheet / một file CSV / một bảng SQLite, CHƯA chọn header.
`Table` = Grid sau khi chọn header: DataFrame mọi cột kiểu object + cột phụ `_src`, `_row` (số dòng gốc, 0-based),
`_sheet`, `_y`, `_m` (năm/tháng suy từ tên sheet hoặc tên file).
"""

from __future__ import annotations

import csv
import io
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from socai.io.readers import _dedupe, _xlsx_sheets, sheet_period
from socai.v4.cells import clean, fold, parse_number

HEADER_SCAN = 15
_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1258", "cp1252", "latin-1")
_DELIMS = (",", ";", "\t", "|")


@dataclass
class Grid:
    id: str  # "file.xlsx::sheet", "file.csv", "pos.sqlite::bang"
    file: str
    sheet: str | None
    rows: list[list]  # ô thô; SQLite: dòng 0 là tên cột
    kind: str  # xlsx | csv | sqlite
    period: tuple[int | None, int | None] = (None, None)
    notes: list[str] = field(default_factory=list)


@dataclass
class Table:
    id: str
    grids: list[str]
    df: pd.DataFrame
    header_row: int
    levels: int = 1
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [c for c in self.df.columns if not c.startswith("_")]


# ---------------------------------------------------------------- nạp
def _read_csv_rows(path: Path) -> tuple[list[list], str]:
    raw = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # latin-1 không bao giờ lỗi, nhánh này chỉ để mypy yên
        text, enc = raw.decode("latin-1", "replace"), "latin-1"
    head = text.splitlines()[:200]
    best, best_d = -1.0, ","
    for d in _DELIMS:
        widths = [len(r) for r in csv.reader(head, delimiter=d) if r]
        if not widths or max(widths) < 2:
            continue
        mode = max(set(widths), key=widths.count)
        score = widths.count(mode) / len(widths) + 0.01 * mode  # ổn định trước, nhiều cột sau
        if score > best:
            best, best_d = score, d
    rows = [list(r) for r in csv.reader(io.StringIO(text, newline=""), delimiter=best_d)]
    return rows, f"encoding={enc} delimiter={best_d!r}"


def _file_period(name: str) -> tuple[int | None, int | None]:
    y, m = sheet_period(Path(name).stem)
    return (y, m) if y else (None, None)


def load_grids(shop_dir: Path) -> dict[str, Grid]:
    """Mọi file dữ liệu của shop (trừ README) → Grid. Đuôi lạ: bỏ qua, có ghi chú."""
    out: dict[str, Grid] = {}
    for p in sorted(shop_dir.iterdir()):
        ext = p.suffix.lower()
        if p.name.lower().startswith("readme") or p.is_dir():
            continue
        if ext in (".xlsx", ".xlsm"):
            for sn, rows in _xlsx_sheets(p):
                rows = [list(r) for r in rows]
                if not any(any(clean(c) for c in r) for r in rows):
                    continue
                per = sheet_period(sn)
                gid = f"{p.name}::{sn}"
                out[gid] = Grid(gid, p.name, sn, rows, "xlsx", per if per[0] else _file_period(p.name))
        elif ext in (".csv", ".tsv", ".txt"):
            rows, note = _read_csv_rows(p)
            out[p.name] = Grid(p.name, p.name, None, rows, "csv", _file_period(p.name), [note])
        elif ext in (".sqlite", ".db", ".sqlite3"):
            con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
            try:
                names = [r[0] for r in con.execute(
                    "select name from sqlite_master where type in ('table','view') and name not like 'sqlite_%'")]
                for t in names:
                    cur = con.execute(f'select * from "{t}"')
                    cols = [d[0] for d in cur.description]
                    rows = [cols] + [list(r) for r in cur.fetchall()]
                    gid = f"{p.name}::{t}"
                    out[gid] = Grid(gid, p.name, t, rows, "sqlite")
            finally:
                con.close()
    return out


# ---------------------------------------------------------------- header
def split_camel(name: str) -> str:
    """'KenhBanHang' → 'Kenh Ban Hang'; 'ma_khach' → 'ma khach' (để so với README)."""
    s = re.sub(r"(?<=[a-zà-ỹ])(?=[A-ZÀ-Ỹ])", " ", str(name))
    return " ".join(s.replace("_", " ").split())


def _row_score(row: list, known: set[str]) -> tuple[float, int]:
    cells = [c for c in row if clean(c)]
    if not cells:
        return -1.0, 0
    texty = sum(1 for c in cells if isinstance(c, str) and parse_number(c) is None)
    hits = sum(1 for c in cells if fold(c) in known)
    return texty + 5 * hits - 2 * (len(cells) - texty), hits


def detect_header(g: Grid, known: set[str] | None = None) -> dict:
    """Điểm từng dòng trong HEADER_SCAN dòng đầu. Header thật: nhiều ô chữ, không có số, trùng tên cột README,
    và dòng ngay dưới có nhiều ô (không phải tiêu đề một ô). SQLite: header là dòng 0."""
    if g.kind == "sqlite":
        return {"row": 0, "score": None, "alts": [], "readme_hits": None}
    known = known or set()
    scored = []
    for i, r in enumerate(g.rows[:HEADER_SCAN]):
        s, hits = _row_score(r, known)
        below = g.rows[i + 1] if i + 1 < len(g.rows) else []
        filled = sum(1 for c in below if clean(c))
        width = sum(1 for c in r if clean(c))
        if width >= 2 and filled >= max(2, width // 2):
            s += 1.0
        scored.append((s, i, hits))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0] if scored else (0, 0, 0)
    return {"row": best[1], "score": round(best[0], 2), "readme_hits": best[2],
            "alts": [{"row": i, "score": round(s, 2)} for s, i, _ in scored[1:3]]}


def make_table(grids: list[Grid], header_row: int, levels: int = 1, names: list[str] | None = None) -> Table:
    """Grid(s) cùng cấu trúc → Table. levels=2: ghép tên hai tầng 'trên › dưới', ô gộp tầng trên điền ngang.
    names: đặt tên cột khi không có header (header_row=-1)."""
    frames = []
    header: list[str] = []
    for g in grids:
        rows = g.rows
        width = max((len(r) for r in rows), default=0)
        pad = [list(r) + [None] * (width - len(r)) for r in rows]
        if names is not None:
            hdr, start = list(names) + [None] * (width - len(names)), header_row + 1
        elif levels == 2:
            top, bot = pad[header_row], pad[header_row + 1]
            last, hdr = "", []
            for a, b in zip(top, bot):
                last = clean(a) or last
                hdr.append(f"{last} › {clean(b)}" if last and clean(b) and clean(b) != last else (clean(b) or last))
            start = header_row + 2
        else:
            hdr, start = pad[header_row], header_row + 1
        cols = _dedupe(hdr)  # giữ tên gốc (README ghi tên gốc); split_camel chỉ dùng khi so khớp
        if not header:
            header = cols
        body = pad[start:]
        df = pd.DataFrame(body, columns=cols, dtype=object)
        df["_src"], df["_row"], df["_sheet"] = g.id, range(start, start + len(body)), g.sheet
        df["_y"], df["_m"] = g.period
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    empty = [c for c in df.columns if str(c).startswith("_col") and df[c].map(clean).eq("").all()]
    df = df.drop(columns=empty)
    tid = grids[0].id if len(grids) == 1 else f"{grids[0].file}::*" if len({g.file for g in grids}) == 1 \
        else re.sub(r"\d+", "#", grids[0].file)
    return Table(tid, [g.id for g in grids], df, header_row, levels)


def merge_candidates(grids: dict[str, Grid], known: set[str] | None = None) -> list[list[str]]:
    """Nhóm các Grid có cùng tập tên cột sau khi dò header (sổ chia sheet theo tháng, file theo tháng)."""
    groups: dict[tuple, list[str]] = {}
    for gid, g in grids.items():
        h = detect_header(g, known)["row"]
        cols = tuple(fold(c) for c in (g.rows[h] if h < len(g.rows) else []) if clean(c))
        groups.setdefault((g.kind, cols), []).append(gid)
    return list(groups.values())
