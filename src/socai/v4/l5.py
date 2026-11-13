"""L5: chạy SQL trên DuckDB (chỉ đọc, có giới hạn thời gian) và định dạng đáp án theo rulebook §9 (do code, không
do LLM): tiền số nguyên làm tròn nửa lên, phần trăm 2 chữ số, ngày YYYY-MM-DD / YYYY-MM, list có thứ tự, table cột
đầu là tên.
"""

from __future__ import annotations

import math
import re
import threading
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

import duckdb

FORBID = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|copy|pragma|install|load|export|import)\b", re.I)


class SQLError(Exception):
    pass


def run_sql(con: duckdb.DuckDBPyConnection, sql: str, timeout: float = 20.0, max_rows: int = 200):
    s = (sql or "").strip().rstrip(";")
    if not s:
        raise SQLError("SQL rỗng")
    if not re.match(r"^\s*(select|with)\b", s, re.I) or FORBID.search(s) or ";" in s:
        raise SQLError("chỉ cho phép một câu SELECT/WITH")
    cur = con.cursor()
    timer = threading.Timer(timeout, cur.interrupt)
    timer.start()
    try:
        res = cur.execute(s)
        cols = [d[0] for d in res.description]
        rows = res.fetchmany(max_rows + 1)
    except duckdb.Error as e:
        raise SQLError(str(e).split("\n")[0][:400]) from e
    finally:
        timer.cancel()
    return cols, rows


def _half_up(x: float, nd: int = 0) -> float | int:
    q = Decimal(str(round(float(x), 9))).quantize(Decimal(1) if nd == 0 else Decimal(10) ** -nd, ROUND_HALF_UP)
    return int(q) if nd == 0 else float(q)


def _num(x, unit: str):
    if x is None or isinstance(x, bool):
        raise SQLError("kết quả rỗng")
    if isinstance(x, (date, datetime)):
        raise SQLError("kết quả là ngày, không phải số")
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        raise SQLError("kết quả không phải số")
    if unit in ("tien", "dem", "gio"):
        return _half_up(v, 0)
    r = _half_up(v, 2)
    return int(r) if float(r).is_integer() and unit != "phan_tram" else r


def _date(x, unit: str) -> str:
    if isinstance(x, datetime):
        x = x.date()
    if isinstance(x, date):
        return x.strftime("%Y-%m") if unit == "thang" else x.isoformat()
    s = str(x)
    m = re.match(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if not m:
        raise SQLError(f"không đọc được ngày: {s[:30]}")
    if unit == "thang" or not m.group(3):
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def _scalar(x, unit: str):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _num(x, unit)
    if isinstance(x, (date, datetime)):
        return _date(x, unit)
    return None if x is None else str(x)


def format_answer(kind: str, unit: str, cols: list[str], rows: list[tuple]) -> dict:
    if kind == "unanswerable":
        return {"type": "unanswerable", "answer": None}
    if not rows:
        raise SQLError("SQL không trả dòng nào")
    if kind in ("number", "percent"):
        if len(rows) != 1:
            raise SQLError(f"cần 1 dòng, SQL trả {len(rows)} dòng")
        return {"type": kind, "answer": _num(rows[0][0], "phan_tram" if kind == "percent" else unit)}
    if kind == "string":
        v = rows[0][0]
        if v is None:
            raise SQLError("kết quả rỗng")
        return {"type": "string", "answer": str(v)}
    if kind == "date":
        return {"type": "date", "answer": _date(rows[0][0], unit)}
    if kind == "boolean":
        v = rows[0][0]
        if isinstance(v, str):
            v = v.strip().lower() in ("true", "1", "yes", "co", "có", "dung", "đúng")
        return {"type": "boolean", "answer": bool(v)}
    if kind == "list":
        if len(rows) == 1 and len(cols) > 1 and all(isinstance(x, (int, float)) for x in rows[0]):
            vals = list(rows[0])  # nhiều số theo thứ tự câu hỏi
        else:
            vals = [r[0] for r in rows]
        out = [_scalar(v, unit) for v in vals]
        if any(v is None for v in out):
            raise SQLError("list có phần tử rỗng")
        return {"type": "list", "answer": out}
    if kind == "table":
        body = []
        for r in rows:
            body.append([str(r[0]) if r[0] is not None else None] + [_scalar(x, unit) for x in r[1:]])
        return {"type": "table", "answer": {"columns": list(cols), "rows": body}}
    raise SQLError(f"loại đáp án lạ: {kind}")
