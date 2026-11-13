"""L5: tính. Chạy SQL của LLM trên DuckDB (chỉ SELECT, có giới hạn thời gian) rồi ĐỊNH DẠNG đáp án bằng code.

LLM không tự viết con số cuối: nó chỉ khai `loai` (kiểu đáp án) và `don_vi` (cách làm tròn), code đọc kết quả
SQL và dựng đáp án đúng format nộp bài.
"""

from __future__ import annotations

import math
import re
import threading
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

import duckdb

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")
INT_UNITS = ("tien", "dem")  # tiền (đồng) và số đếm: số nguyên, làm tròn nửa lên


class SQLError(Exception):
    pass


def run_sql(con: duckdb.DuckDBPyConnection, sql: str, timeout: float = 30.0, max_rows: int = 500):
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        raise SQLError("SQL rỗng")
    bare = re.sub(r"'(?:[^']|'')*'", "''", s)  # bỏ chuỗi trong nháy trước khi kiểm (chuỗi có thể chứa ';')
    if not re.match(r"^(select|with)\b", bare, re.I) or ";" in bare:
        raise SQLError("chỉ cho phép đúng một câu SELECT/WITH")
    cur = con.cursor()
    timer = threading.Timer(timeout, cur.interrupt)
    timer.start()
    try:
        res = cur.execute(s)
        cols = [d[0] for d in res.description]
        rows = res.fetchmany(max_rows + 1)
    except duckdb.Error as e:
        raise SQLError(str(e).split("\n\n")[0][:500]) from e
    finally:
        timer.cancel()
        cur.close()
    if len(rows) > max_rows:
        raise SQLError(f"kết quả quá {max_rows} dòng; thiếu điều kiện lọc hoặc LIMIT")
    return cols, rows


def half_up(x: float, nd: int) -> int | float:
    q = Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), ROUND_HALF_UP)
    return int(q) if nd == 0 else float(q)


def _number(x: object, unit: str) -> int | float:
    if x is None:
        raise SQLError("kết quả rỗng (NULL): kiểm lại điều kiện lọc và kỳ")
    if isinstance(x, bool) or isinstance(x, (date, datetime, str)):
        raise SQLError(f"kết quả là {type(x).__name__}, không phải số")
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        raise SQLError("kết quả không phải số hữu hạn (chia cho 0?)")
    if unit in INT_UNITS:
        return half_up(v, 0)
    r = half_up(v, 2)
    return int(r) if r == int(r) and unit != "phan_tram" else r


def _cell(x: object, unit: str = "so"):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, datetime):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, (int, float, Decimal)):
        return _number(x, unit)
    return str(x).strip()


def _date(x: object) -> str:
    if isinstance(x, datetime):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()
    s = str(x).strip()
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T].*", s):
        return s[:10]
    raise SQLError(f"kết quả '{s[:30]}' không phải ngày dạng YYYY-MM-DD hoặc YYYY-MM")


def format_answer(kind: str, unit: str, cols: list[str], rows: list[tuple]) -> dict:
    """Dựng đáp án đúng format từ kết quả SQL. Sai dạng (nhiều dòng cho câu một số…) → SQLError để LLM sửa."""
    if kind not in TYPES:
        raise SQLError(f"loai phải là một trong {TYPES} hoặc unanswerable")
    if not rows:
        raise SQLError("SQL không trả dòng nào: kiểm lại điều kiện lọc, tên giá trị và kỳ")
    if kind in ("number", "percent", "string", "date", "boolean"):
        if len(rows) != 1 or len(rows[0]) != 1:
            raise SQLError(f"loai = {kind} cần đúng 1 dòng 1 cột, SQL trả {len(rows)} dòng × {len(rows[0])} cột")
        x = rows[0][0]
        if kind == "number":
            return {"type": kind, "answer": _number(x, unit)}
        if kind == "percent":
            return {"type": kind, "answer": float(_number(x, "phan_tram"))}
        if x is None:
            raise SQLError("kết quả rỗng (NULL)")
        if kind == "date":
            return {"type": kind, "answer": _date(x)}
        if kind == "boolean":
            if isinstance(x, bool):
                return {"type": kind, "answer": x}
            if isinstance(x, (int, float)) and x in (0, 1):
                return {"type": kind, "answer": bool(x)}
            raise SQLError("loai = boolean cần SQL trả true/false")
        s = _cell(x)
        if not isinstance(s, str) or not s:
            raise SQLError("loai = string cần SQL trả một chuỗi")
        return {"type": kind, "answer": s}
    if kind == "list":
        if len(rows[0]) != 1:
            raise SQLError("loai = list cần đúng 1 cột (mỗi dòng một phần tử, đã ORDER BY đúng thứ tự)")
        items = [_cell(r[0], unit) for r in rows]
        if any(i is None or i == "" for i in items):
            raise SQLError("list có phần tử rỗng")
        return {"type": kind, "answer": items}
    table = [[_cell(v, unit) for v in r] for r in rows]
    return {"type": "table", "answer": {"columns": list(cols), "rows": table}}
