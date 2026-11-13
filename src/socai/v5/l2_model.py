"""L2: mô hình shop. Nạp bảng sạch (L1) vào DuckDB, mô tả từng cột bằng số liệu thật, lập chỉ mục giá trị.

Ra `Shop`: kết nối DuckDB, danh mục bảng/cột (tên SQL ↔ tên gốc, kiểu, mô tả README, thống kê), khoảng ngày của
dữ liệu (cho cổng thời gian ở L6), ngành của shop (để lọc luật ngành ở L3), chỉ mục giá trị (để L4 khớp tên
trong câu hỏi với giá trị thật). Không dùng LLM: mọi thứ tất định, chạy lại ra đúng như cũ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from socai.io.cells import fold
from socai.profile.readme import Readme
from socai.v5.l1_read import Table

MAX_ENUM = 24  # cột chữ có ít giá trị khác nhau: liệt kê hết cho LLM
MAX_INDEX = 5000  # cột chữ có nhiều giá trị hơn thế thì không lập chỉ mục (ghi chú tự do, mã đơn…)
SQL_TYPE = {"number": "DOUBLE", "date": "DATE", "datetime": "TIMESTAMP", "time": "VARCHAR", "text": "VARCHAR"}


@dataclass
class ColInfo:
    sql: str
    orig: str
    type: str
    desc: str = ""
    filled: bool = False
    distinct: int = 0
    nulls: float = 0.0
    values: list[tuple[str, int]] = field(default_factory=list)  # (giá trị, số dòng) cho cột ít giá trị
    samples: list[str] = field(default_factory=list)
    lo: str = ""
    hi: str = ""


@dataclass
class TableInfo:
    sql: str
    orig: str
    rows: int
    cols: list[ColInfo]
    notes: list[str]
    confidence: float = 1.0


@dataclass
class Shop:
    biz: str
    con: duckdb.DuckDBPyConnection
    tables: dict[str, TableInfo]
    intro: str = ""
    conventions: list[str] = field(default_factory=list)
    date_range: tuple[str, str] | None = None
    industry: str | None = None
    index: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)  # khoá bỏ dấu → [(bảng, cột, giá trị)]
    date_span: tuple[str, str] | None = None  # hợp mọi cột ngày: chỉ chặn câu hỏi có kỳ nằm ngoài toàn bộ
    extra: str = ""  # quy đổi đơn vị + ghi chú cột trong README


def slug(name: str, used: set[str], prefix: str = "c") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", fold(name)).strip("_") or prefix
    if s[0].isdigit():
        s = f"{prefix}_{s}"
    base, i = s[:48], 2
    s = base
    while s in used:
        s, i = f"{base}_{i}", i + 1
    used.add(s)
    return s


def _table_slug(t: Table, used: set[str]) -> str:
    stem = t.name.split("::")[-1] if "::" in t.name else re.sub(r"\.[A-Za-z0-9]+$", "", t.name)
    return slug(stem, used, "t")


def _load(con: duckdb.DuckDBPyConnection, t: Table, tname: str) -> list[ColInfo]:
    used: set[str] = set()
    frame, infos, selects = {}, [], []
    for i, c in enumerate(t.cols):
        vals = t.df[c.name].tolist()
        typ = SQL_TYPE[c.kind]
        if c.kind == "number":
            ser = pd.Series([float("nan") if v is None else float(v) for v in vals], dtype="float64")
            ok = ser.dropna()
            if len(ok) and (ok == ok.round()).all() and ok.abs().max() < 9e15:
                typ = "BIGINT"
        elif c.kind in ("date", "datetime"):
            ser = pd.to_datetime(pd.Series(vals, dtype=object), errors="coerce")
        else:
            ser = pd.Series([None if v is None else str(v) for v in vals], dtype=object)
            if ser.isna().all():
                ser = pd.Series([None] * len(vals), dtype="float64")  # cột rỗng: DuckDB không đoán được kiểu object
        frame[f"c{i}"] = ser
        name = slug("sheet_goc" if c.name == "_sheet" else c.name, used)
        infos.append(ColInfo(name, c.name, typ, c.desc, c.filled))
        selects.append(f'CAST("c{i}" AS {typ}) AS "{name}"')
    df = pd.DataFrame(frame)  # noqa: F841 — DuckDB đọc biến cục bộ này qua tên
    con.execute(f'CREATE TABLE "{tname}" AS SELECT {", ".join(selects)} FROM df')
    return infos


def _profile(con: duckdb.DuckDBPyConnection, tname: str, c: ColInfo, rows: int) -> None:
    q = f'"{c.sql}"'
    distinct, nn = con.execute(f'SELECT count(DISTINCT {q}), count({q}) FROM "{tname}"').fetchone()
    c.distinct, c.nulls = int(distinct), round(1 - nn / max(rows, 1), 3)
    if not nn:
        return
    if c.type == "VARCHAR":
        top = con.execute(f'SELECT {q}, count(*) n FROM "{tname}" WHERE {q} IS NOT NULL GROUP BY 1 '
                          f"ORDER BY n DESC, 1 LIMIT {MAX_ENUM}").fetchall()
        if distinct <= MAX_ENUM:
            c.values = [(str(v), int(n)) for v, n in top]
        else:
            c.samples = [str(v) for v, _ in top[:4]]
    else:
        lo, hi = con.execute(f'SELECT min({q}), max({q}) FROM "{tname}"').fetchone()
        c.lo, c.hi = str(lo), str(hi)


def _date_range(con, tables: dict[str, TableInfo], readme: Readme) -> tuple[str, str] | None:
    """Khoảng ngày của dữ liệu = cột ngày có nhiều dòng nhất (sổ giao dịch chính). Mốc cuối ưu tiên câu
    'dữ liệu đến hết …' trong README. Cột ngày khác (hạn dùng, ngày đăng ký) không được kéo dãn khoảng này."""
    best = None
    for t in tables.values():
        for c in t.cols:
            if c.type in ("DATE", "TIMESTAMP") and c.lo:
                n = t.rows * (1 - c.nulls)
                if best is None or n > best[0]:
                    best = (n, c.lo[:10], c.hi[:10])
    if best is None:
        return None
    return best[1], readme.data_end.isoformat() if readme.data_end else best[2]


def _date_span(tables: dict[str, TableInfo], readme: Readme) -> tuple[str, str] | None:
    los = [c.lo[:10] for t in tables.values() for c in t.cols if c.type in ("DATE", "TIMESTAMP") and c.lo]
    his = [c.hi[:10] for t in tables.values() for c in t.cols if c.type in ("DATE", "TIMESTAMP") and c.hi]
    if not los:
        return None
    if readme.data_end:
        his.append(readme.data_end.isoformat())
    return min(los), max(his)


def _industry(intro: str, industries: list[str]) -> str | None:
    """Ngành của shop: tên ngành trong rulebook có cụm đặc trưng xuất hiện trong phần giới thiệu của README."""
    generic = {"cua", "hang", "tiem", "quan", "shop", "nha", "va", "ban", "dai", "ly"}
    text = f" {fold(intro)} "
    best, score = None, 0
    for ind in industries:
        f = fold(ind)
        words = [w for w in f.split() if w not in generic]
        if f" {f} " in text:
            s = 10 + len(words)
        else:  # tên ngành trong rulebook và README hiếm khi trùng hẳn: cần ≥ một nửa số từ đặc trưng
            hit = sum(1 for w in words if f" {w} " in text)
            s = hit if words and hit * 2 >= len(words) else 0
        if s > score:
            best, score = ind, s
    return best


def build_shop(biz: str, readme: Readme, tables: list[Table], industries: list[str] | None = None) -> Shop:
    con = duckdb.connect(":memory:")
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '8GB'")  # hai shop chạy cùng lúc; join lỗi không được ăn hết RAM
    # khoa(x) = socai.io.cells.fold: chữ thường, bỏ dấu, ký tự không phải chữ/số thành một khoảng trắng.
    # Dùng để so tên không phân biệt cách viết (luật: cùng một thứ có thể ghi nhiều cách).
    con.execute("CREATE MACRO khoa(x) AS trim(regexp_replace(replace(lower(strip_accents(CAST(x AS VARCHAR))), "
                "'đ', 'd'), '[^0-9a-z]+', ' ', 'g'))")
    used: set[str] = set()
    infos: dict[str, TableInfo] = {}
    for t in tables:
        tname = _table_slug(t, used)
        cols = _load(con, t, tname)
        info = TableInfo(tname, t.name, len(t.df), cols, t.notes, t.confidence)
        for c in cols:
            _profile(con, tname, c, info.rows)
        infos[tname] = info
    shop = Shop(biz, con, infos, readme.intro, readme.conventions, _date_range(con, infos, readme),
                _industry(readme.intro, industries or []), date_span=_date_span(infos, readme),
                extra="\n".join([f"- 1 {u.big_unit} {u.product} = {u.factor:g} {u.base_unit}" for u in readme.units] +
                                [f"- {k}: {v}" for k, v in readme.notes.items()]))
    for t in infos.values():
        for c in t.cols:
            if c.type != "VARCHAR" or not 0 < c.distinct <= MAX_INDEX or c.sql == "sheet_goc":
                continue
            for (v,) in con.execute(f'SELECT DISTINCT "{c.sql}" FROM "{t.sql}" WHERE "{c.sql}" IS NOT NULL ORDER BY 1').fetchall():
                k = fold(v)
                if len(k) >= 2:
                    shop.index.setdefault(k, []).append((t.sql, c.sql, v))
    return shop


def render_table(t: TableInfo, brief: bool = False) -> str:
    head = f"BẢNG {t.sql} (nguồn: {t.orig}; {t.rows} dòng)"
    if t.notes:
        head += "\n  ghi chú: " + "; ".join(t.notes[:6])
    lines = [head]
    for c in t.cols:
        s = f'  {c.sql} {c.type} — "{c.orig}"'
        if c.desc:
            s += f": {c.desc[:140]}"
        if c.filled:
            s += " (ô trống đã điền theo dòng đầu nhóm)"
        if not brief:
            if c.values:
                s += " | giá trị: " + ", ".join(f"'{v[:40]}' ({n})" for v, n in c.values)
            elif c.samples:
                s += f" | {c.distinct} giá trị khác nhau, vd: " + ", ".join(f"'{v[:40]}'" for v in c.samples)
            elif c.lo:
                s += f" | {c.lo} → {c.hi}"
            if c.nulls >= 0.05:
                s += f" | trống {round(c.nulls * 100)}%"
        lines.append(s)
    return "\n".join(lines)
