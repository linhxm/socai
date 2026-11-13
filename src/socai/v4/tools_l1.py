"""Tool L1 (v4-ham §1): phiên làm việc trên một shop. Tool hỏi chỉ đọc; tool làm ghi thao tác vào `recipe`
để `apply_recipe` dựng lại y hệt từ file gốc.

Khi mở phiên, code tự làm phần chắc chắn: nạp Grid, gộp sheet/file cùng cấu trúc, dò header, phân loại dòng.
LLM chỉ xem tóm tắt (`summary`) và sửa chỗ không chắc.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pandas as pd

from socai.profile.readme import load_readme
from socai.v4 import rows as R
from socai.v4.cells import KINDS, clean, fold, is_blank, parse_cell, parse_qty, shape
from socai.v4.grid import Grid, Table, detect_header, load_grids, make_table, merge_candidates, split_camel
from socai.v4.registry import tool

GUESS_ORDER = ("date", "datetime", "time", "money", "qty", "bool", "code", "text")


def _cut(v: object, n: int = 40) -> str:
    s = clean(v)
    return s if len(s) <= n else s[:n - 1] + "…"


class L1Session:
    def __init__(self, shop_dir: Path):
        self.dir = Path(shop_dir)
        self.readme = load_readme(self.dir)
        self.known = {fold(c) for cols in self.readme.columns.values() for c in cols}
        self.grids: dict[str, Grid] = load_grids(self.dir)
        self.tables: dict[str, Table] = {}
        self.kinds: dict[str, pd.DataFrame] = {}
        self.col_kinds: dict[str, dict[str, str]] = {}
        self.groups: dict[str, pd.Series] = {}
        self.recipe: list[dict] = []
        for gids in merge_candidates(self.grids, self.known):
            gs = [self.grids[g] for g in gids]
            h = detect_header(gs[0], self.known)["row"]
            t = make_table(gs, h)
            self.tables[t.id] = t
            self.kinds[t.id] = R.classify_rows(t)
            self.col_kinds[t.id] = {}

    # ------------------------------------------------------------ tiện ích
    def t(self, src: str) -> Table:
        if src not in self.tables:
            raise KeyError(src)
        return self.tables[src]

    def data_mask(self, src: str) -> pd.Series:
        return self.kinds[src]["kind"].isin(["du_lieu", "tiep_noi", "dong_phu_don"])

    def log(self, op: str, **args) -> None:
        self.recipe.append({"op": op, **args})

    def readme_for(self, t: Table) -> dict[str, str]:
        files = {self.grids[g].file for g in t.grids}
        out: dict[str, str] = {}
        for key, cols in self.readme.columns.items():
            if any(key == f or key in f or f.startswith(key.split("::")[0]) for f in files):
                out.update(cols)
        return out


# ---------------------------------------------------------------- hồ sơ cột
def column_profile(t: Table, col: str, mask: pd.Series | None = None, k: int = 5) -> dict:
    s = t.df[col] if mask is None else t.df.loc[mask, col]
    present = s[~s.map(is_blank)]
    n, nb = len(s), len(present)
    shapes = collections.Counter()
    ex: dict[str, str] = {}
    for v in present.values[:20000]:
        sh = shape(v)
        shapes[sh] += 1
        ex.setdefault(sh, _cut(v, 30))
    years = t.df.loc[present.index, "_y"] if "_y" in t.df else None
    rates = {}
    sample = present.iloc[:: max(1, nb // 3000)] if nb else present
    for kd in ("date", "time", "money", "qty", "bool"):
        if not nb:
            break
        ok = sum(1 for i, v in sample.items()
                 if parse_cell(v, kd, int(years[i]) if years is not None and pd.notna(years[i]) else None) is not None)
        rates[kd] = round(ok / len(sample), 3)
    return {"col": col, "n": n, "blank_rate": round(1 - nb / n, 3) if n else None,
            "distinct": int(present.map(clean).nunique()),
            "shapes": [{"shape": sh, "n": c, "ex": ex[sh]} for sh, c in shapes.most_common(k)],
            "parse_rates": rates, "guess": guess_kind(rates, present)}


def guess_kind(rates: dict, present: pd.Series) -> str:
    if not len(present):
        return "text"
    if rates.get("date", 0) >= 0.95 and present.map(lambda v: isinstance(v, str) and ("/" in v or "-" in v or "thg"
                                                                                       in v.lower())).mean() > 0.5:
        return "date"
    if rates.get("date", 0) >= 0.95 and rates.get("money", 0) < 0.95:
        return "date"
    if rates.get("time", 0) >= 0.95 and rates.get("money", 0) < 0.5:
        return "time"
    if rates.get("money", 0) >= 0.95:
        return "money"
    if rates.get("qty", 0) >= 0.95:
        return "qty"
    if rates.get("bool", 0) >= 0.98 and present.map(clean).nunique() <= 3:
        return "bool"
    return "text"


# ---------------------------------------------------------------- tool hỏi
S = {"type": "string", "description": "id bảng (từ list_sources)"}
C = {"type": "string", "description": "tên cột đúng như list_sources/profile_table"}


@tool("L1", "Liệt kê các bảng của shop (đã gộp sheet/file cùng cấu trúc): id, nguồn, số dòng, cột, kỳ theo tên sheet, "
            "số dòng theo loại.")
def list_sources(s: L1Session) -> dict:
    out = []
    for tid, t in s.tables.items():
        k = s.kinds[tid]["kind"].value_counts().to_dict()
        per = sorted({(y, m) for y, m in zip(t.df["_y"], t.df["_m"]) if pd.notna(y)})
        out.append({"id": tid, "from": t.grids[:3] + (["…"] if len(t.grids) > 3 else []), "rows": len(t.df),
                    "header_row": t.header_row, "columns": t.columns[:40], "row_kinds": k,
                    "sheet_periods": [f"{int(y)}-{int(m):02d}" if pd.notna(m) else str(int(y)) for y, m in per][:6]})
    return {"tables": out}


@tool("L1", "Xem vài dòng thô của bảng, kèm loại dòng. at: head | tail | random | số dòng (index) để xem quanh đó.",
      src=S, at={"type": "string"}, n={"type": "integer"}, cols={"type": "array", "items": {"type": "string"}},
      kind={"type": "string", "description": "chỉ xem dòng loại này (vd tong, tiep_noi)"})
def peek(s: L1Session, src: str, at: str = "head", n: int = 8, cols: list | None = None, kind: str | None = None) -> dict:
    t = s.t(src)
    df, kd = t.df, s.kinds[src]
    idx = df.index if kind is None else df.index[kd["kind"] == kind]
    n = min(int(n), 20)
    if at == "tail":
        idx = idx[-n:]
    elif at == "random":
        idx = idx[:0] if not len(idx) else pd.Index(sorted(pd.Series(idx).sample(min(n, len(idx)), random_state=0)))
    elif str(at).lstrip("-").isdigit():
        p = df.index.get_loc(int(at)) if int(at) in df.index else 0
        idx = df.index[max(0, p - n // 2): p + n // 2 + 1]
    else:
        idx = idx[:n]
    cols = cols or t.columns
    return {"rows": [{"i": int(i), "kind": kd.at[i, "kind"], **{c: _cut(df.at[i, c]) for c in cols
                                                                   if not is_blank(df.at[i, c])}} for i in idx]}


@tool("L1", "Tóm tắt mọi cột của bảng (chỉ dòng dữ liệu): tỉ lệ trống, chữ ký hình dạng hay gặp, kiểu đoán, mô tả "
            "README.", src=S)
def profile_table(s: L1Session, src: str) -> dict:
    t = s.t(src)
    rd = s.readme_for(t)
    rdf = {fold(k): v for k, v in rd.items()}
    m = s.data_mask(src)
    cols = []
    for c in t.columns:
        p = column_profile(t, c, m, k=2)
        cols.append({"col": c, "blank": p["blank_rate"], "distinct": p["distinct"], "guess": p["guess"],
                     "shapes": [f"{x['shape']}×{x['n']} ({x['ex']})" for x in p["shapes"]],
                     "readme": (rdf.get(fold(c)) or rdf.get(fold(split_camel(c))) or "")[:90]})
    return {"id": src, "rows": int(m.sum()), "columns": cols}


@tool("L1", "Hồ sơ đầy đủ một cột: các chữ ký hình dạng (9tr9, 9/9/9, int5=serial Excel…), tỉ lệ đọc được theo từng "
            "kiểu, số giá trị khác nhau.", src=S, col=C)
def profile_column(s: L1Session, src: str, col: str) -> dict:
    t = s.t(src)
    if col not in t.df:
        raise KeyError(col)
    return column_profile(t, col, s.data_mask(src), k=10)


@tool("L1", "Các giá trị khác nhau của một cột và tần suất (chỉ dòng dữ liệu).", src=S, col=C,
      k={"type": "integer"}, kind={"type": "string", "description": "lọc theo loại dòng"})
def distinct(s: L1Session, src: str, col: str, k: int = 30, kind: str | None = None) -> dict:
    t = s.t(src)
    m = s.data_mask(src) if kind is None else s.kinds[src]["kind"] == kind
    vc = t.df.loc[m, col].map(clean).value_counts()
    return {"n_distinct": int(vc.size), "values": [{"v": _cut(v, 60), "n": int(c)} for v, c in vc.head(min(k, 60)).items()]}


@tool("L1", "Tìm dòng chứa chữ. accents=true (mặc định) so CÒN DẤU: 'Tổng' khác 'Tống'.", src=S,
      text={"type": "string"}, col={"type": "string"}, accents={"type": "boolean"})
def find_rows(s: L1Session, src: str, text: str, col: str | None = None, accents: bool = True) -> dict:
    t = s.t(src)
    cols = [col] if col else t.columns
    norm = (lambda v: clean(v).lower()) if accents else fold
    needle = norm(text)
    hit = pd.Series(False, index=t.df.index)
    for c in cols:
        hit |= t.df[c].map(lambda v, c=c: needle in norm(v) if isinstance(v, str) else False)
    idx = t.df.index[hit]
    return {"n": int(hit.sum()), "rows": [{"i": int(i), "kind": s.kinds[src].at[i, "kind"],
                                          **{c: _cut(t.df.at[i, c]) for c in t.columns if not is_blank(t.df.at[i, c])}}
                                         for i in idx[:10]]}


@tool("L1", "Đoạn README: giới thiệu, quy ước, mô tả cột của file chứa bảng (hoặc của cả shop nếu không có src).",
      src={"type": "string"})
def readme(s: L1Session, src: str | None = None) -> dict:
    rd = s.readme
    cols = s.readme_for(s.t(src)) if src else {}
    return {"intro": rd.intro[:1200], "conventions": rd.conventions[:20], "columns": cols,
            "data_end": str(rd.data_end) if rd.data_end else None}


# ---------------------------------------------------------------- header, gộp
@tool("L1", "Chấm điểm các dòng đầu làm header của từng nguồn trong bảng.", src=S)
def header_report(s: L1Session, src: str) -> dict:
    t = s.t(src)
    return {"current": t.header_row, "by_source": {g: detect_header(s.grids[g], s.known) for g in t.grids[:4]}}


@tool("L1", "Đặt lại header cho bảng (mọi nguồn trong bảng). levels=2: header hai tầng. Dựng lại bảng, phân loại "
            "lại dòng.", src=S, row={"type": "integer"}, levels={"type": "integer"},
      names={"type": "array", "items": {"type": "string"}, "description": "chỉ khi không có header (row=-1)"})
def set_header(s: L1Session, src: str, row: int, levels: int = 1, names: list | None = None) -> dict:
    t = s.t(src)
    nt = make_table([s.grids[g] for g in t.grids], int(row), int(levels), names)
    nt.id = src
    s.tables[src] = nt
    s.kinds[src] = R.classify_rows(nt)
    s.log("set_header", src=src, row=row, levels=levels, names=names)
    return {"columns": nt.columns, "rows": len(nt.df)}


# ---------------------------------------------------------------- loại dòng
@tool("L1", "Đếm dòng theo loại (du_lieu, trong, tong, ghi_chu_cuoi, header_lap, tiep_noi…) và liệt kê dòng KHÔNG "
            "CHẮC để xem lại.", src=S, key_cols={"type": "array", "items": {"type": "string"},
                                                    "description": "cột khoá (mã đơn, ngày, tên hàng): giúp nhận dòng "
                                                                   "tiếp nối và dòng tổng; có thì phân loại lại"})
def row_kinds(s: L1Session, src: str, key_cols: list | None = None) -> dict:
    t = s.t(src)
    if key_cols:
        s.kinds[src] = R.classify_rows(t, key_cols)
        s.log("row_kinds", src=src, key_cols=key_cols)
    kd = s.kinds[src]
    un = kd[kd["unsure"]]
    return {"counts": kd["kind"].value_counts().to_dict(),
            "unsure": [{"i": int(i), "kind": kd.at[i, "kind"], "why": kd.at[i, "why"],
                        "cells": {c: _cut(t.df.at[i, c]) for c in t.columns if not is_blank(t.df.at[i, c])}}
                       for i in un.index[:10]], "n_unsure": int(len(un)),
            "sure_examples": {k: kd.loc[kd["kind"] == k, "why"].head(2).tolist()
                              for k in kd["kind"].unique() if k != "du_lieu"}}


@tool("L1", "Gán loại cho các dòng (sửa phân loại). rows: danh sách index; hoặc match: chữ (còn dấu) xuất hiện ở cột col.",
      src=S, kind={"type": "string", "enum": list(R.ROW_KINDS)}, rows={"type": "array", "items": {"type": "integer"}},
      match={"type": "string"}, col={"type": "string"})
def set_row_kind(s: L1Session, src: str, kind: str, rows: list | None = None, match: str | None = None,
                 col: str | None = None) -> dict:
    t = s.t(src)
    kd = s.kinds[src]
    if rows is not None:
        idx = [int(i) for i in rows if int(i) in kd.index]
    elif match is not None:
        cols = [col] if col else t.columns
        hit = pd.Series(False, index=t.df.index)
        for c in cols:
            hit |= t.df[c].map(lambda v: isinstance(v, str) and clean(v).lower().startswith(match.lower()))
        idx = list(t.df.index[hit])
    else:
        raise ValueError("cần rows hoặc match")
    kd.loc[idx, ["kind", "why", "unsure"]] = [kind, "LLM gán", False]
    s.log("set_row_kind", src=src, kind=kind, rows=rows, match=match, col=col)
    return {"changed": len(idx), "counts": kd["kind"].value_counts().to_dict()}


@tool("L1", "Kiểm một dòng có phải dòng tổng: so số ở cột amount_col với tổng các dòng dữ liệu phía trên (scope=above).",
      src=S, row={"type": "integer"}, amount_col=C)
def check_total_row(s: L1Session, src: str, row: int, amount_col: str) -> dict:
    return R.check_total_row(s.t(src), int(row), amount_col, s.kinds[src]["kind"])


# ---------------------------------------------------------------- nhóm dòng
@tool("L1", "Định nhóm dòng (đơn). by: cột mã đơn; start_when_notna: cột chỉ ghi ở dòng đầu đơn (STT…); không có cả hai: "
            "mỗi dòng một nhóm (sổ không theo đơn).", src=S, by={"type": "string"},
      start_when_notna={"type": "string"})
def group_rows(s: L1Session, src: str, by: str | None = None, start_when_notna: str | None = None) -> dict:
    t = s.t(src)
    g, rep = R.group_rows(t, by, start_when_notna, s.kinds[src]["kind"])
    s.groups[src] = g
    s.log("group_rows", src=src, by=by, start_when_notna=start_when_notna)
    return rep


@tool("L1", "Điền ô trống của các dòng sau bằng dòng đầu nhóm (§8: khách, ngày, trạng thái của đơn ghi ở dòng đầu). "
            "Cần group_rows trước. Báo số nhóm mà dòng sau ghi khác dòng đầu.", src=S,
      cols={"type": "array", "items": {"type": "string"}})
def ffill_group(s: L1Session, src: str, cols: list) -> dict:
    t = s.t(src)
    if src not in s.groups:
        raise ValueError("chưa group_rows")
    t.df, rep = R.ffill_group(t.df, cols, s.groups[src])
    s.log("ffill_group", src=src, cols=cols)
    return {"by_col": rep}


@tool("L1", "Trong mỗi nhóm, cột col có mấy giá trị khác nhau (vd ngày khác nhau giữa các dòng của một đơn).",
      src=S, col=C)
def group_profile(s: L1Session, src: str, col: str) -> dict:
    if src not in s.groups:
        raise ValueError("chưa group_rows")
    return R.group_profile(s.t(src).df, s.groups[src], col)


@tool("L1", "Báo dòng trùng. scope: table (trùng hệt), group (trùng trong cùng nhóm/đơn), cross_source (trùng giữa các "
            "sheet/file), key (trùng khoá khác nội dung, cần key).", src=S,
      scope={"type": "string", "enum": ["table", "group", "cross_source", "key"]}, key={"type": "string"})
def dup_report(s: L1Session, src: str, scope: str = "table", key: str | None = None) -> dict:
    t = s.t(src)
    m = s.data_mask(src)
    df = t.df[m]
    g = s.groups.get(src)
    return R.dup_report(df, t.columns, scope, g[m] if g is not None else None, key)


# ---------------------------------------------------------------- ô
@tool("L1", "Đọc thử một cột theo kiểu và trả các ô KHÔNG đọc được. kind: money | number | qty | date | datetime | time "
            "| bool | pct | code | text. Ngày thiếu năm lấy năm theo tên sheet.", src=S, col=C,
      kind={"type": "string", "enum": list(KINDS)})
def parse_column(s: L1Session, src: str, col: str, kind: str) -> dict:
    t = s.t(src)
    m = s.data_mask(src) & ~t.df[col].map(is_blank)
    ys = t.df["_y"]
    vals = [(i, v, parse_cell(v, kind, int(ys[i]) if pd.notna(ys[i]) else None)) for i, v in t.df.loc[m, col].items()]
    bad = [(i, v) for i, v, x in vals if x is None]
    good = [x for _, _, x in vals if x is not None]
    stats: dict = {}
    if good and kind in ("money", "number", "qty", "pct"):
        stats = {"min": min(good), "max": max(good), "neg": sum(1 for x in good if x < 0)}
    elif good and kind in ("date", "datetime"):
        stats = {"min": str(min(good)), "max": str(max(good))}
    if kind == "qty":
        units = collections.Counter(parse_qty(v)[1] for _, v, _ in vals if parse_qty(v)[1])
        stats["unit_text"] = dict(units.most_common(10))
    seen, fails = set(), []
    for i, v in bad:
        if clean(v) not in seen:
            seen.add(clean(v))
            fails.append({"i": int(i), "v": _cut(v, 50)})
        if len(fails) >= 15:
            break
    return {"n": len(vals), "ok_rate": round(len(good) / len(vals), 4) if vals else None, "n_fail": len(bad),
            "failures": fails, "stats": stats}


@tool("L1", "Chốt kiểu của cột (dùng khi dựng bảng sạch).", src=S, col=C, kind={"type": "string", "enum": list(KINDS)})
def set_column_kind(s: L1Session, src: str, col: str, kind: str) -> dict:
    if col not in s.t(src).df:
        raise KeyError(col)
    s.col_kinds[src][col] = kind
    s.log("set_column_kind", src=src, col=col, kind=kind)
    return {"col_kinds": s.col_kinds[src]}


@tool("L1", "Tách ô chứa nhiều thông tin. sep: regex phân cách → list (IMEI 'a / b', 'việc 1, việc 2'); regex: nhóm có "
            "tên → cột mới (dòng xe + biển số); preset: name_phone (tên kèm sđt) | item_addons (món + topping (giá)). "
            "qty_col: kiểm số phần tử = SL.", src=S, col=C, sep={"type": "string"}, regex={"type": "string"},
      preset={"type": "string", "enum": ["name_phone", "item_addons"]}, qty_col={"type": "string"})
def split_cell(s: L1Session, src: str, col: str, sep: str | None = None, regex: str | None = None,
               preset: str | None = None, qty_col: str | None = None) -> dict:
    t = s.t(src)
    t.df, rep = R.split_cell(t.df, col, sep, regex, preset, qty_col)
    s.log("split_cell", src=src, col=col, sep=sep, regex=regex, preset=preset, qty_col=qty_col)
    return rep


# ---------------------------------------------------------------- tóm tắt tự động cho LLM
def summary(s: L1Session, src: str) -> dict:
    """Gói sẵn cho LLM ở đầu vòng L1: nguồn, header, cột, loại dòng, dòng không chắc, dòng trùng (≤ ~2K token)."""
    t = s.t(src)
    rk = row_kinds(s, src)
    return {"id": src, "from": t.grids[:4], "header_row": t.header_row, "profile": profile_table(s, src)["columns"],
            "row_kinds": rk["counts"], "unsure": rk["unsure"][:6],
            "dup_table": R.dup_report(t.df[s.data_mask(src)], t.columns, "table")["n_rows_extra"]}


