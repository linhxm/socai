"""Đại số truy vấn v2: LLM ghép biểu thức, code chạy TẤT ĐỊNH và áp luật theo loại cột.

Truy vấn gốc Q: {from, where, time, group, agg, having, sort, limit, out}
Phép ghép E:    {op: pct_change|diff|pct|ratio|add|list|trend_up|trend_down, a, b | items}
                b có thể là {"like": "a", ...chỗ khác}: giống a, chỉ đổi những khoá được ghi.
Lỗi có gợi ý (`V2Error`) để não hỏi lại LLM một lần; dữ liệu không đủ → `Unsupported` (bỏ trống);
kỳ nằm ngoài dữ liệu → unanswerable (§7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any, Callable

import pandas as pd

from socai.exec.metrics import Result, Unsupported
from socai.io.cells import clean, fold, parse_date, parse_number, round2, to_int_money
from socai.rules.timeexpr import Period, TimeSpec
from socai.semantic.entities import ALIASES
from socai.v2.views import WALKIN, View

AGGS = ("sum", "count", "count_distinct", "mean", "wmean", "max", "min", "unique")
OUTS = ("value", "name", "names", "values", "table", "count", "key")
OPS = ("pct_change", "diff", "pct", "ratio", "add", "list", "trend_up", "trend_down", "except")
DERIVED_GROUPS = {"thang": "month", "nam": "year"}
CMP = ("=", "!=", "in", "not in", ">", ">=", "<", "<=", "~", "$", "*")


class V2Error(Exception):
    """Biểu thức sai (cột/bảng/giá trị không có): thông báo kèm gợi ý, đưa lại cho LLM sửa."""


class Unanswerable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Val:
    v: Any
    unit: str  # money | qty | num | pct | name | names | key | table | list | bool | count
    keys: list | None = None  # khoá nhóm của từng tên (phép tập hợp so theo khoá, không theo cách viết)


@dataclass
class Ctx:
    views: dict[str, View]
    tspec: TimeSpec
    resolve_time: Callable[[str], TimeSpec]
    series_periods: Callable[[TimeSpec], list[Period]]
    notes: list[str] = field(default_factory=list)
    in_b: bool = False  # đang tính vế được so sánh (b) của một phép ghép
    today: date | None = None


def resolve_col(view: View, name: object) -> str:
    """Tên cột đúng; hoặc tên viết thiếu tiền tố bảng nối ('thuoc_ke_don' → 'danh_sach_thuoc.thuoc_ke_don')."""
    if isinstance(name, str) and name in view.cols:
        return name
    if isinstance(name, str):
        n = name.split(".")[-1]
        hits = [c for c in view.cols if c.split(".")[-1] == n or c.endswith("." + n)]
        if len(hits) == 1:
            return hits[0]
        hits = [c for c in view.cols if n and n in c]
        if len(hits) == 1:
            return hits[0]
    raise V2Error(f"bảng {view.name} không có cột '{name}'; các cột: {list(view.cols)}")


# ------------------------------------------------------------------ giá trị trong câu → giá trị trong dữ liệu
def match_value(view: View, col: str, span: str) -> str | None:
    """Khớp đúng (bỏ dấu) → bí danh (tiền mặt ~ cash) → chứa nhau theo ranh giới từ (duy nhất) → gần đúng ≥ 0,88."""
    vals = view.values(col)
    q = fold(span)
    if not q:
        return None
    if q in vals:
        return vals[q]
    for k in ALIASES.get(q, ()):
        if k in vals:
            return vals[k]
    inside = [k for k in vals if f" {q} " in f" {k} " or f" {k} " in f" {q} "]
    if len(inside) == 1:
        return vals[inside[0]]
    scored = sorted(((SequenceMatcher(None, q, k).ratio(), k) for k in vals), reverse=True)
    if scored and scored[0][0] >= 0.88 and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.03):
        return vals[scored[0][1]]
    return None


def _near(view: View, col: str, span: str, k: int = 4) -> list[str]:
    q = fold(span)
    vals = view.values(col)
    return [vals[x] for _, x in sorted(((SequenceMatcher(None, q, x).ratio(), x) for x in vals), reverse=True)[:k]]


def _filter(view: View, df: pd.DataFrame, cond: list, ctx: Ctx) -> pd.DataFrame:
    if not isinstance(cond, (list, tuple)) or len(cond) not in (2, 3):
        raise V2Error(f"điều kiện lọc phải là [cột, phép, giá trị], nhận {cond!r}")
    col, op = cond[0], cond[1]
    val = cond[2] if len(cond) == 3 else None
    if op not in CMP:
        raise V2Error(f"phép lọc '{op}' không có; dùng một trong {list(CMP)}")
    col = resolve_col(view, col)
    if op == "*":
        return df
    kind = view.cols[col].kind
    s = df[col]
    if kind in ("money", "qty", "num", "hour"):
        x = val if isinstance(val, (int, float)) else parse_number(val)
        if x is None:
            raise V2Error(f"cột {col} là số, giá trị '{val}' không phải số")
        s = pd.to_numeric(s, errors="coerce")
        return df[{"=": s == x, "!=": s != x, ">": s > x, ">=": s >= x, "<": s < x, "<=": s <= x}.get(op, s == x)]
    if kind == "date":
        x = parse_date(val) if not isinstance(val, date) else val
        if x is None or op not in ("=", "!=", ">", ">=", "<", "<="):
            raise V2Error(f"lọc ngày cột {col} cần ngày dạng YYYY-MM-DD và phép so sánh, nhận {op} {val!r}")
        s = df[col]
        ok = s.notna()
        return df[ok & {"=": s == x, "!=": s != x, ">": s > x, ">=": s >= x, "<": s < x, "<=": s <= x}[op]]
    if kind == "bool":
        b = _BOOLQ.get(fold(val))
        if b is None:
            raise V2Error(f"cột {col} là có/không, nhận {val!r}")
        keep = df[col].eq(b)
        return df[~keep] if op in ("!=", "not in") else df[keep]
    vals = val if isinstance(val, list) else [val]
    if op in ("$", "~") and all(isinstance(v, str) and fold(v) in view.values(col) for v in vals):
        op = "="  # giá trị là một tên có thật trong cột: lọc đúng tên, không lọc theo đuôi/chứa
    if op in ("$", "~"):
        pat = [fold(v) for v in vals if isinstance(v, str) and fold(v)]
        if not pat:
            raise V2Error(f"phép {op} cần chuỗi chữ")
        test = (lambda x: x.endswith(tuple(pat))) if op == "$" else (lambda x: any(p in x for p in pat))
        if not view.df[col].map(fold).map(test).any():
            raise V2Error(f"không giá trị nào của cột {col} {'kết thúc bằng' if op == '$' else 'chứa'} {vals}; "
                          f"lọc theo tên thì dùng '=' với tên chép từ câu hỏi; gần nhất: {_near(view, col, str(vals[0]))}")
        return df[df[col].map(fold).map(test)]
    hits = []
    for v in vals:
        if not isinstance(v, str) or not v.strip():
            raise V2Error(f"giá trị lọc cột {col} phải là chữ chép từ câu hỏi, nhận {v!r}")
        hit = match_value(view, col, v)
        if hit is None:
            raise V2Error(f"'{v}' không có trong cột {col}; gần nhất: {_near(view, col, v)}")
        hits.append(hit)
    folded = {fold(h) for h in hits}
    keep = df[col].map(fold).isin(folded)
    if col in view.key_of:  # §8: mọi cách viết của cùng một mã là một thứ
        kc = view.key_of[col]
        keys = set(view.df.loc[view.df[col].map(fold).isin(folded), kc]) - {""}
        keep = keep | df[kc].isin(keys)
    return df[~keep] if op in ("!=", "not in") else df[keep]


_BOOLQ = {"co": "có", "true": "có", "1": "có", "yes": "có", "x": "có", "khong": "không", "false": "không",
          "0": "không", "no": "không", "hok": "không", "ko": "không", "k": "không"}


def _relocate(view: View, cond: list) -> list:
    """Tên không có ở cột LLM chọn nhưng khớp ĐÚNG MỘT cột chữ khác của bảng → chuyển cột (tự sửa, ghi log)."""
    if len(cond) != 3 or cond[1] not in ("=", "!=", "in", "not in"):
        return cond
    try:
        cond = [resolve_col(view, cond[0]), cond[1], cond[2]]
    except V2Error:
        return cond
    if view.cols[cond[0]].kind not in ("cat", "id"):
        return cond
    vals = cond[2] if isinstance(cond[2], list) else [cond[2]]
    if all(isinstance(v, str) and match_value(view, cond[0], v) for v in vals):
        return cond
    alt = [c for c, m in view.cols.items() if m.kind in ("cat", "id") and c != cond[0]
           and all(isinstance(v, str) and match_value(view, c, v) for v in vals)]
    return [alt[0], cond[1], cond[2]] if len(alt) == 1 else cond


# ------------------------------------------------------------------ thời gian
def _periods(view: View, t: Any, ctx: Ctx) -> list[Period] | None:
    """None = không lọc thời gian. Danh sách > 1 phần tử = chuỗi kỳ (kết quả là list)."""
    if t in (None, "", "none", "$tat_ca"):
        return None
    ts = ctx.tspec
    if isinstance(t, dict):  # §5 hiệu ứng tăng giá: N ngày kể từ D (gồm D) / N ngày ngay trước D
        n = t.get("so_ngay")
        d = parse_date(t.get("tu") or t.get("truoc"))
        n = 1 if n == 0 else n
        if not isinstance(n, int) or n <= 0 or d is None:
            raise V2Error('khoảng ngày cần {"tu"|"truoc": "YYYY-MM-DD", "so_ngay": N}')
        if t.get("tu"):
            return [Period(d, d + timedelta(days=n - 1), "range")]
        return [Period(d - timedelta(days=n), d - timedelta(days=1), "range")]
    if t == "$truoc_ky":  # từ đầu dữ liệu đến hết ngày ngay trước kỳ chính (vd "đã từng bán trước kỳ")
        start, _ = view.span
        if ts.main is None or start is None:
            raise V2Error("$truoc_ky cần kỳ chính")
        if ts.main.start <= start:
            raise Unanswerable("không có dữ liệu trước kỳ hỏi (§7)")
        return [Period(start, ts.main.start - timedelta(days=1), "range")]
    if t == "$den_het" and view.asof is None:  # bảng phát sinh: "đến hết hôm nay" của kỳ chính đã nằm trong $ky
        t = "$ky"
    if t == "$ky_so" and not ctx.in_b:
        ctx.notes.append("$ky_so ngoài vế so sánh → $ky")
        t = "$ky"
    if t == "$ky":
        if ts.main is None:
            raise V2Error("câu hỏi không có kỳ rõ ràng; ghi time là cụm thời gian trong câu, hoặc null")
        return [ts.main]
    if t == "$ky_so":
        if ts.compare_to is None:
            if len(ts.periods) >= 2:
                return [ts.periods[1]]
            raise V2Error("không tìm thấy kỳ so sánh trong câu; ghi time là cụm thời gian của kỳ so sánh")
        return [ts.compare_to]
    if t == "$tung_ky":
        ps = ctx.series_periods(ts)
        if not ps:
            raise V2Error("không giải được các kỳ")
        return ps
    if t == "$den_het":
        start, _ = view.span
        end = ts.main.end if ts.main else None
        if start is None or end is None:
            raise V2Error("không xác định được mốc 'đến hết'")
        return [Period(start, end, "until")]
    if isinstance(t, str):
        r = ctx.resolve_time(t)
        if not r.periods:
            raise V2Error(f"không hiểu cụm thời gian '{t}'; dùng $ky, $ky_so, $tung_ky, $den_het hoặc null")
        return [r.periods[0]]
    raise V2Error(f"time phải là chuỗi, nhận {t!r}")


def _check_coverage(view: View, p: Period) -> None:
    s, e = view.span
    if s is None:
        return
    if p.end < s or p.start > e:
        raise Unanswerable(f"kỳ {p.start}..{p.end} nằm ngoài dữ liệu {s}..{e} (§7)")


# ------------------------------------------------------------------ gộp
def _agg(view: View, df: pd.DataFrame, agg: list) -> Val:
    if not isinstance(agg, (list, tuple)) or not agg or agg[0] not in AGGS:
        raise V2Error(f"agg phải là [phép, cột] với phép trong {list(AGGS)}, nhận {agg!r}")
    fn, col = agg[0], (agg[1] if len(agg) > 1 else None)
    if col is None:
        if fn != "count":
            raise V2Error(f"{fn} cần tên cột")
        return Val(len(df), "count")
    col = resolve_col(view, col)
    kind = view.cols[col].kind
    s = df[col]
    if fn in ("count", "count_distinct"):
        key = df[view.key_of.get(col, col)]
        if kind in ("cat", "id"):
            if view.df[col].map(clean).eq("").all():
                raise Unsupported(f"cột {col} trống trong dữ liệu của shop")
            key = key.map(clean)
            key = key[key.ne("")]
            if col in view.key_of or col == "khach":
                key = key[~key.map(fold).isin(WALKIN)]
            if col in ("mat_hang", "ma_hang") and "_la_mat_hang" in df:
                key = key[df.loc[key.index, "_la_mat_hang"]]
        distinct = fn == "count_distinct" or kind == "id"
        return Val(int(key.nunique() if distinct else key.size), "count")
    if fn == "wmean":
        if len(agg) < 3:
            raise V2Error("wmean cần [wmean, cột giá trị, cột trọng số]")
        w = resolve_col(view, agg[2])
        x, wt = pd.to_numeric(s, errors="coerce"), pd.to_numeric(df[w], errors="coerce")
        ok = x.notna() & wt.notna()
        if not ok.any() or not wt[ok].sum():
            raise Unsupported("không có dòng nào để tính bình quân")
        if not ok.all():
            raise Unsupported(f"có ô {col}/{w} không đọc được")
        return Val(float((x * wt).sum() / wt.sum()), kind if kind in ("money", "qty") else "num")
    if fn == "unique":
        u = pd.unique(s.dropna())
        if len(u) != 1:
            raise Unsupported(f"cột {col} có {len(u)} giá trị khác nhau trong phạm vi lọc: {list(u)[:5]}")
        x = u[0]
        return Val(x, kind if kind in ("money", "qty", "num") else "name")
    if kind not in ("money", "qty", "num", "hour"):
        raise V2Error(f"{fn} cần cột số, '{col}' là {kind}")
    x = pd.to_numeric(s, errors="coerce")
    if fn == "sum" and x.isna().any() and s.notna().any():
        raise Unsupported(f"có ô {col} không đọc được trong phạm vi tính")
    if len(s) and x.isna().all():
        raise Unsupported(f"cột {col} không có số nào trong phạm vi tính")
    x = x.dropna()
    if fn in ("max", "min", "mean") and x.empty:
        raise Unsupported("không có dòng nào để tính")
    v = {"sum": x.sum(), "mean": x.mean() if len(x) else 0.0, "max": x.max(), "min": x.min()}[fn]
    return Val(float(v), kind if kind != "hour" else "num")


def _group_key(view: View, df: pd.DataFrame, g: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(khoá nhóm, tên hiển thị, mã để xếp bằng nhau)."""
    if g in ("ngay",) or (g in view.cols and view.cols[g].kind == "date"):
        k = df[g if g in df else view.date_col].map(lambda d: d.isoformat() if d is not None else "")
        return k, k, k
    if g in DERIVED_GROUPS:
        if not view.date_col:
            raise V2Error(f"bảng {view.name} không có cột ngày để nhóm theo {g}")
        fmt = "%Y-%m" if g == "thang" else "%Y"
        k = df[view.date_col].map(lambda d: d.strftime(fmt) if d is not None else "")
        return k, k, k
    try:
        g = resolve_col(view, g)
    except V2Error:
        raise V2Error(f"không nhóm được theo '{g}'; dùng một cột của {view.name}: {list(view.cols)} hoặc thang/nam")
    if view.cols[g].kind == "hour":
        k = pd.to_numeric(df[g], errors="coerce")
        if k.isna().all():
            raise Unsupported("dữ liệu không ghi giờ")
        k = k.map(lambda h: "" if pd.isna(h) else str(int(h)))
        return k, k, k.map(lambda h: h.zfill(2))
    key = df[view.key_of.get(g, g)].map(clean)
    disp = df[g].map(clean)
    code = df[view.code_of[g]].map(clean) if g in view.code_of else key
    code = code.where(code.ne(""), disp)
    return key, disp, code


def run_query(q: dict, ctx: Ctx) -> Val:
    if not isinstance(q, dict):
        raise V2Error(f"truy vấn phải là object, nhận {q!r}")
    vname = q.get("from") or "ban"
    if vname not in ctx.views:
        raise V2Error(f"không có bảng '{vname}'; các bảng: {list(ctx.views)}")
    view = ctx.views[vname]
    df = view.df
    if view.asof is not None:  # bảng trạng thái: time là mốc "tính đến hết ngày X"
        t = q.get("time", "$ky")
        t = "$ky" if t == "$den_het" else t
        ps = _periods(view, t, ctx) if t not in (None, "", "none") and (ctx.tspec.main or t != "$ky") else None
        x = ps[-1].end if ps else ctx.today
        if x is None:
            raise V2Error("cần mốc ngày cho bảng trạng thái")
        if ctx.today is not None and x > ctx.today:
            x = ctx.today
        start = view.df["ngay"].min() if "ngay" in view.df and len(view.df) else None
        if start is not None and x < start:
            raise Unanswerable(f"mốc {x} trước khi có dữ liệu (§7)")
        df = view.asof(x)
        ctx.notes.append(f"trạng thái tại hết ngày {x}")
        q = {**q, "time": None}
    where = q.get("where") or []
    if where and not isinstance(where[0], (list, tuple)):
        where = [where]
    where = [_relocate(view, list(c)) if isinstance(c, (list, tuple)) else c for c in where]
    agg_col = (q.get("agg") or [None, None])[1:2]
    all_status = bool(agg_col) and agg_col[0] in view.all_status
    if view.status_col and view.default_status and not all_status and not any(
            isinstance(c, (list, tuple)) and c and c[0] == view.status_col for c in where):
        df = df[df[view.status_col].isin(view.default_status)]
    for c in where:
        df = _filter(view, df, c, ctx)
    by_date = any(isinstance(c, (list, tuple)) and c and c[0] == view.date_col for c in where)
    gq, ps_all = q.get("group"), ctx.tspec.periods
    if q.get("time", "$ky") in ("$ky", "$tung_ky") and len(ps_all) > 1 and view.date_col and (
            gq in DERIVED_GROUPS or (gq in view.cols and view.cols[gq].kind == "date")):
        q = {**q, "time": None}
        span = Period(min(p.start for p in ps_all), max(p.end for p in ps_all), "range")
        _check_coverage(view, span)
        df = _in(df, view, span)
        by_date = True
    periods = None if by_date else _periods(view, q.get("time", "$ky" if ctx.tspec.main else None), ctx)
    if periods is not None and not view.date_col:
        ctx.notes.append(f"bảng {view.name} không có cột ngày: bỏ qua time")
        periods = None
    if periods is not None:
        for p in periods:
            _check_coverage(view, p)
    if periods is not None and len(periods) > 1:
        vals = [_one(view, _in(df, view, p), q, ctx) for p in periods]
        if any(x.unit not in ("money", "qty", "num", "count") for x in vals):
            raise V2Error("chuỗi kỳ ($tung_ky) chỉ dùng với out=value")
        return Val([_fmt(x)[1] for x in vals], "list")
    return _one(view, _in(df, view, periods[0]) if periods else df, q, ctx)


def _in(df: pd.DataFrame, view: View, p: Period) -> pd.DataFrame:
    d = df[view.date_col]
    return df[d.notna() & (d >= p.start) & (d <= p.end)]


def _one(view: View, df: pd.DataFrame, q: dict, ctx: Ctx) -> Val:
    agg = q.get("agg") or ["count"]
    g = q.get("group")
    out = q.get("out") or ("value" if not g else "name")
    if out not in OUTS:
        raise V2Error(f"out phải là một trong {list(OUTS)}")
    if g and out == "value":  # "mặt hàng nào bán chạy nhất": có nhóm mà hỏi một giá trị → đối tượng đứng đầu
        out = "name" if not isinstance(q.get("limit"), int) or q["limit"] <= 1 else "names"
    if g == "ma_hang" and "mat_hang" in view.cols:  # nhóm theo mã nhưng đáp án ghi tên (mã vẫn dùng để xếp)
        g = "mat_hang"
    if not g:
        if out not in ("value", "count"):
            raise V2Error(f"out={out} cần group")
        if q.get("having"):
            raise V2Error("having chỉ dùng cùng group; lọc theo giá trị một cột thì dùng where")
        if df.empty and any(isinstance(c, (list, tuple)) and c and c[0] != view.status_col
                            for c in (q.get("where") or [])):
            raise Unsupported("bộ lọc không khớp dòng nào trong kỳ: không có dữ liệu, không trả 0")
        return _agg(view, df, agg)
    key, disp, code = _group_key(view, df, g)
    g = g if g in view.cols or g in DERIVED_GROUPS else resolve_col(view, g)
    m = key.ne("")
    if g in view.key_of or g == "khach":
        m &= ~key.map(fold).isin(WALKIN)
    if g in ("mat_hang", "ma_hang") and "_la_mat_hang" in df:  # §3: phí phục vụ không phải mặt hàng
        m &= df["_la_mat_hang"]
    df, key, disp, code = df[m], key[m], disp[m], code[m]
    if df.empty:
        raise Unsupported("không có dòng nào sau khi lọc")
    rows = []
    for k, part in df.groupby(key, sort=False):
        x = _agg(view, part, agg)
        rows.append((disp[part.index[0]], code[part.index[0]], x.v, x.unit, k))
    unit = rows[0][3]
    tab = pd.DataFrame(rows, columns=["name", "code", "v", "u", "key"])
    if (h := q.get("having")):
        if not isinstance(h, (list, tuple)) or len(h) != 2 or h[0] not in (">", ">=", "<", "<=", "=", "!="):
            raise V2Error("having phải là [phép, số], ví dụ ['>=', 5]")
        x = h[1] if isinstance(h[1], (int, float)) else parse_number(h[1])
        if x is None:
            raise V2Error("having cần số")
        s = tab["v"]
        tab = tab[{">": s > x, ">=": s >= x, "<": s < x, "<=": s <= x, "=": s == x, "!=": s != x}[h[0]]]
    if out == "count":
        return Val(len(tab), "count")
    asc = q.get("sort") == "asc"
    tab = tab.sort_values(["v", "code"], ascending=[asc, True], kind="mergesort")
    if tab.empty:
        raise Unsupported("không nhóm nào thoả điều kiện")
    k = q.get("limit")
    k = k if isinstance(k, int) and k > 0 else None
    head = tab.head(k) if k else tab
    ctx.notes.append(f"xếp hạng: {tab.head((k or 1) + 2)[['name', 'v']].values.tolist()}")
    timeish = g in DERIVED_GROUPS or view.cols[g].kind in ("date", "hour")
    if out == "name" and timeish:  # "ngày/tháng/giờ nào cao nhất" → đáp án kiểu ngày, không phải tên
        out = "key"
    if out == "name":
        return Val(str(tab.iloc[0]["name"]), "name", [tab.iloc[0]["key"]])
    if out == "key":
        top = str(tab.iloc[0]["name"])
        return Val(int(top) if top.isdigit() and len(top) <= 2 else top, "key")
    if out == "names":
        return Val([str(x) for x in head["name"]], "names", list(head["key"]))
    if out == "values":
        return Val([_fmt(Val(x, unit))[1] for x in head["v"]], "list")
    if out == "table":
        return Val({"columns": [g, agg[1] if len(agg) > 1 else agg[0]],
                    "rows": [[str(n), _fmt(Val(x, unit))[1]] for n, x in zip(head["name"], head["v"])]}, "table")
    return Val([str(x) for x in head["name"]], "names")


# ------------------------------------------------------------------ ghép
def _merge_like(a: dict, b: dict) -> dict:
    out = {k: v for k, v in a.items()}
    for k, v in b.items():
        if k in ("like", "drop"):
            continue
        out[k] = v
    if (drop := b.get("drop")):
        drop = [drop] if isinstance(drop, str) else drop
        out["where"] = [c for c in (out.get("where") or []) if not (isinstance(c, (list, tuple)) and c and c[0] in drop)]
    return out


def evaluate(e: Any, ctx: Ctx) -> Val:
    if not isinstance(e, dict):
        raise V2Error(f"biểu thức phải là object JSON, nhận {e!r}")
    op = e.get("op")
    if op is None:
        return run_query(e, ctx)
    if op == "unanswerable":  # LLM: shop không ghi loại thông tin câu hỏi cần (§7)
        raise Unanswerable(str(e.get("ly_do") or e.get("nghi") or "dữ liệu của shop không ghi thông tin này"))
    if op == "none":
        raise Unsupported("LLM: câu ngoài khả năng của bảng hiện có")
    if op not in OPS:
        raise V2Error(f"op '{op}' không có; dùng một trong {list(OPS)}")
    if op == "list":
        items = e.get("items") or []
        if not items:
            raise V2Error("op list cần items")
        base = items[0] if isinstance(items[0], dict) else {}
        vals = [evaluate(_merge_like(base, it) if isinstance(it, dict) and it.get("like") else it, ctx) for it in items]
        return Val([_fmt(v)[1] for v in vals], "list")
    a = e.get("a")
    if op == "except":  # đối tượng có ở a mà không có ở b (vd hàng ế: từng bán trước kỳ, trong kỳ không bán)
        b = e.get("b")
        if isinstance(b, dict) and b.get("like") == "a" and isinstance(a, dict):
            b = _merge_like(a, b)
        x = evaluate(a, ctx)
        try:
            y = evaluate(b, ctx)
        except Unsupported:  # vế b không có dòng nào: không loại ai
            y = Val([], "names", [])
        if x.unit != "names" or y.unit != "names" or x.keys is None or y.keys is None:
            raise V2Error("except cần a và b là truy vấn có group và out=names")
        gone = set(y.keys)
        left = sorted((k, n) for k, n in zip(x.keys, x.v) if k not in gone)  # §9: chỉ lọc, không số đo → theo mã
        if e.get("out") == "count":
            return Val(len(left), "count")
        return Val([n for _, n in left], "names", [k for k, _ in left])
    if op in ("trend_up", "trend_down"):
        x = evaluate(a, ctx)
        if x.unit != "list" or len(x.v) < 2:
            raise V2Error("trend cần a là chuỗi kỳ (time $tung_ky) có ≥ 2 kỳ")
        pairs = list(zip(x.v, x.v[1:]))
        ok = all(q > p for p, q in pairs) if op == "trend_up" else all(q < p for p, q in pairs)
        return Val(ok, "bool")
    b = e.get("b")
    if isinstance(b, dict) and b.get("like") == "a" and isinstance(a, dict):
        b = _merge_like(a, b)
    x = evaluate(a, ctx)
    ctx.in_b = True
    try:
        y = evaluate(b, ctx)
    finally:
        ctx.in_b = False
    for z in (x, y):
        if z.unit not in ("money", "qty", "num", "count", "pct"):
            raise V2Error(f"phép {op} cần hai con số, nhận {z.unit}")
    if op == "diff":
        return Val(x.v - y.v, x.unit)
    if op == "add":
        return Val(x.v + y.v, x.unit)
    if not y.v:
        raise Unanswerable("mẫu số bằng 0")
    if op == "pct_change":
        return Val(100.0 * (x.v - y.v) / y.v, "pct")
    if op == "pct":
        return Val(100.0 * x.v / y.v, "pct")
    return Val(x.v / y.v, "money" if x.unit == "money" else "num")  # ratio: tiền / lượng = tiền (§9 làm tròn)


# ------------------------------------------------------------------ định dạng đáp án (§9)
def _fmt(x: Val) -> tuple[str, Any]:
    if x.unit == "money":
        return "number", to_int_money(float(x.v))
    if x.unit in ("qty", "num", "count"):
        v = float(x.v)
        return "number", int(v) if v.is_integer() else round2(v)
    if x.unit == "pct":
        return "percent", round2(float(x.v))
    if x.unit == "name":
        return "string", str(x.v)
    if x.unit == "key":
        if isinstance(x.v, int):
            return "number", x.v
        return ("date", x.v) if len(str(x.v)) in (7, 10) and str(x.v)[:4].isdigit() else ("string", str(x.v))
    if x.unit in ("names", "list"):
        return "list", x.v
    if x.unit == "table":
        return "table", x.v
    if x.unit == "bool":
        return "boolean", bool(x.v)
    raise V2Error(f"không định dạng được {x.unit}")


def execute(e: Any, ctx: Ctx) -> Result:
    try:
        x = evaluate(e, ctx)
    except Unanswerable as u:
        return Result.unanswerable(u.reason)
    t, v = _fmt(x)
    return Result(t, v, {"fn": "v2", "expr": e, "notes": ctx.notes})


def signature(e: Any) -> str:
    """Khoá hiệu chỉnh: dạng biểu thức (op + phép gộp + out), không phụ thuộc shop."""
    if not isinstance(e, dict):
        return "bad"
    if e.get("op") in ("unanswerable", "none"):
        return e["op"]
    if (op := e.get("op")):
        inner = e.get("a") if isinstance(e.get("a"), dict) else ((e.get("items") or [{}])[0])
        return f"{op}:{signature(inner)}" if isinstance(inner, dict) else op
    agg = e.get("agg") or ["count"]
    return f"{agg[0]}:{e.get('out') or ('name' if e.get('group') else 'value')}"
