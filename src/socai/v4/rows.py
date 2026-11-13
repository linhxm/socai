"""Loại dòng, nhóm dòng, ô nhóm để trống, dòng trùng, tách ô (v4-ham §1.3, 1.4, 1.6, 1.7). Không gọi LLM.

Bài học từ quét 44 shop: nhận dòng tổng bằng nhãn ĐÃ BỎ DẤU bắt nhầm họ "Tống" và hàng "Công tắc" hàng nghìn dòng,
còn dòng tổng thật ghi ngay trong cột tên khách ("Tổng ngày 01/04"). Vì vậy: so nhãn trên chữ CÒN DẤU, xét cấu trúc
dòng, rồi cộng thử (`check_total_row`) để xác nhận.
"""

from __future__ import annotations

import re

import pandas as pd

from socai.v4.cells import clean, fold, is_blank, parse_number, parse_qty, strip_accents
from socai.v4.grid import Table

ROW_KINDS = ("du_lieu", "trong", "tong", "ghi_chu_cuoi", "header_lap", "tiep_noi", "dong_phu_don", "dong_huy")

# nhãn tổng còn dấu (chữ thường); bản không dấu chỉ nhận khi đi kèm từ chỉ kỳ/khoản, và luôn là "không chắc"
_RE_TOTAL = re.compile(r"^(tổng cộng|tổng|cộng|total|sum|subtotal)(\s|:|$)")
_RE_TOTAL_FOLDED = re.compile(r"^(tong cong|tong|cong)\s+(ngay|thang|nam|tuan|ca|ky|tien|so|hang|don|phieu)\b")
_RE_NOTE = re.compile(r"^(ghi chú|lưu ý|chú thích|chú ý|note|ghi chu|luu y)\s*[:：-]")


def _strs(row: pd.Series) -> list[tuple[str, str]]:
    return [(c, clean(v)) for c, v in row.items() if isinstance(v, str) and clean(v)]


def classify_rows(t: Table, key_cols: list[str] | None = None) -> pd.DataFrame:
    """Mỗi dòng một nhãn trong ROW_KINDS + lý do + cờ không chắc. key_cols: cột khoá (mã đơn, ngày, tên hàng);
    có thì dùng để nhận dòng tiếp nối và kiểm cấu trúc dòng tổng."""
    cols = t.columns
    df = t.df
    blank = df[cols].map(is_blank)
    nb = (~blank).sum(axis=1)
    kind = pd.Series("du_lieu", index=df.index, dtype=object)
    why = pd.Series("", index=df.index, dtype=object)
    unsure = pd.Series(False, index=df.index)

    kind[nb == 0], why[nb == 0] = "trong", "mọi ô trống"
    # lọc ứng viên theo cột (trên giá trị khác nhau) rồi mới xét từng dòng: bảng 170k dòng vẫn nhanh
    hdr = pd.Series(0, index=df.index)
    cand = pd.Series(False, index=df.index)
    for c in cols:
        u = df[c].dropna().unique()
        fmap = {v: fold(v) for v in u}
        hdr += df[c].map(fmap).eq(fold(c)).astype(int)
        hits = {v for v in u if isinstance(v, str) and (_RE_TOTAL.match(clean(v).lower()) or _RE_NOTE.match(
            clean(v).lower()) or (v == strip_accents(v) and _RE_TOTAL_FOLDED.match(fmap[v])))}
        if hits:
            cand |= df[c].isin(hits)
    m = (hdr >= 3) & (kind == "du_lieu")
    kind[m], why[m] = "header_lap", "≥3 ô trùng tên cột"

    typical = int(nb[kind == "du_lieu"].median()) if (kind == "du_lieu").any() else 0
    for i in df.index[cand & (kind == "du_lieu") & (nb <= max(2, typical))]:
        ss = _strs(df.loc[i, cols])
        if nb[i] <= 2 and any(_RE_NOTE.match(s.lower()) for _, s in ss):
            kind[i], why[i] = "ghi_chu_cuoi", "≤2 ô, nhãn ghi chú"
            continue
        lab = next(((c, s) for c, s in ss if _RE_TOTAL.match(s.lower())), None)
        folded = False
        if lab is None:
            lab = next(((c, s) for c, s in ss if s == strip_accents(s) and _RE_TOTAL_FOLDED.match(fold(s))), None)
            folded = lab is not None
        if lab is None:
            continue
        if key_cols:
            structural = all(is_blank(df.at[i, k]) for k in key_cols if k in cols and k != lab[0])
        else:
            structural = nb[i] <= max(3, typical // 4)  # dòng tổng chỉ có nhãn + vài số
        if not structural and folded:
            continue  # tên khách không dấu "Tong Van …" ở dòng đủ ô là dòng dữ liệu
        kind[i], why[i] = "tong", f"nhãn '{lab[1][:30]}' ở cột {lab[0]}"
        unsure[i] = folded or not structural

    if key_cols:
        kc = [k for k in key_cols if k in cols]
        if kc:
            m = (kind == "du_lieu") & blank[kc].all(axis=1) & (nb >= 2)
            kind[m], why[m] = "tiep_noi", f"ô khoá {kc} trống"

    # xác nhận dòng tổng bằng cộng thử. Khớp (kể cả sau khi bỏ dòng trùng): chắc. Gần khớp (≤5%) mà dòng có cấu trúc
    # dòng tổng (nhãn còn dấu, rất ít ô so với dòng dữ liệu): chắc, ghi độ lệch (thường do đơn huỷ/dòng trùng không
    # tính vào tổng ngày; L2 dùng độ lệch này để hiểu cách shop cộng). Còn lại: không chắc, LLM xem.
    sparse = nb <= max(3, typical // 4)
    for i in df.index[kind == "tong"]:
        chk = auto_check_total(t, i, kind)
        if chk and chk["how"] != "close":
            why[i] += f"; khớp tổng cột {chk['col']} ({chk['how']}, {chk['rows_used']} dòng)"
            unsure[i] = False
        elif chk and sparse[i] and not unsure[i]:
            why[i] += f"; gần khớp cột {chk['col']} (lệch {chk['rel']:.1%})"
        elif unsure[i] and nb[i] > max(3, typical // 4):
            kind[i], why[i], unsure[i] = "du_lieu", "nhãn giống tổng nhưng dòng đủ ô, cộng thử không khớp", False
        else:
            unsure[i] = True
    return pd.DataFrame({"kind": kind, "why": why, "unsure": unsure})


def _numeric_cols(t: Table, i: int) -> list[str]:
    return [c for c in t.columns if parse_number(t.df.at[i, c]) is not None]


def check_total_row(t: Table, row: int, amount_col: str, kinds: pd.Series | None = None,
                    scope: str = "above", date_col: str | None = None, label_date=None) -> dict:
    """So số ở `row` với tổng `amount_col` của các dòng dữ liệu:
    scope='above': từ sau dòng tổng trước đó (cùng nguồn) tới ngay trên `row` (tổng theo khối);
    scope='above_level': như trên nhưng chỉ dừng ở dòng tổng CÙNG CẤP (tổng tháng cộng qua các dòng tổng ngày);
    scope='date': mọi dòng có `date_col` = label_date (tổng theo ngày ghi trong nhãn)."""
    df = t.df
    v = parse_number(df.at[row, amount_col])
    if v is None:
        return {"ok": False, "error": "ô không phải số"}
    kinds = kinds if kinds is not None else pd.Series("du_lieu", index=df.index)
    pos = df.index.get_loc(row)
    if scope in ("above", "above_level"):
        rows = []
        src = df["_src"].iloc[pos] if "_src" in df else None
        lv = _level(df.iloc[pos])
        for j in range(pos - 1, -1, -1):
            k = kinds.iloc[j]
            if src is not None and df["_src"].iloc[j] != src:
                break
            if k == "tong" and (scope == "above" or _level(df.iloc[j]) == lv):
                break
            if k in ("du_lieu", "tiep_noi"):
                rows.append(df.index[j])
    elif scope == "date" and date_col is not None and label_date is not None:
        from socai.v4.cells import parse_date
        rows = [j for j in df.index if kinds[j] in ("du_lieu", "tiep_noi")
                and parse_date(df.at[j, date_col], df.at[j, "_y"] if "_y" in df else None) == label_date]
    else:
        return {"ok": False, "error": "scope không hợp lệ"}
    s = sum(x for x in (parse_number(df.at[j, amount_col]) for j in rows) if x is not None)
    cols = [c for c in t.columns]
    uniq = df.loc[rows, cols].map(clean).drop_duplicates().index if rows else []
    s_dd = sum(x for x in (parse_number(df.at[j, amount_col]) for j in uniq) if x is not None)
    best = min((abs(v - s), "all"), (abs(v - s_dd), "dedup"))
    return {"ok": True, "value": v, "sum": round(s, 2), "sum_dedup": round(s_dd, 2), "diff": round(v - s, 2),
            "match": best[0] <= 1, "how": best[1] if best[0] <= 1 else "close",
            "rel": round(best[0] / abs(v), 4) if v else None, "rows_used": len(rows)}


_LEVELS = ("ngay", "tuan", "thang", "quy", "nam", "ca", "ky")


def _level(row: pd.Series) -> str:
    """Cấp của dòng tổng theo nhãn: 'Tổng ngày 01/04' → ngay; 'Cộng tháng 05/2025' → thang; 'Tổng cộng' → ''."""
    for v in row.values:
        if isinstance(v, str):
            w = fold(v).split()
            if w and w[0] in ("tong", "cong") and len(w) > 1:
                return next((x for x in w[1:3] if x in _LEVELS), "")
    return ""


def auto_check_total(t: Table, i: int, kinds: pd.Series) -> dict | None:
    """Thử mọi cột số của dòng i theo khối phía trên: khớp đúng (cả bản bỏ dòng trùng) thì trả ngay; không thì trả
    cột gần khớp nhất nếu lệch ≤ 5%."""
    close = None
    for scope in ("above", "above_level"):
        for c in _numeric_cols(t, i):
            r = check_total_row(t, i, c, kinds, scope)
            if not r.get("ok") or r["rows_used"] < 1 or r["value"] == 0:
                continue
            if r["match"]:
                return {"col": c, "how": r["how"] + ("" if scope == "above" else ", cùng cấp"),
                        "rows_used": r["rows_used"], "rel": 0.0}
            if r["rel"] is not None and r["rel"] <= 0.05 and (close is None or r["rel"] < close["rel"]):
                close = {"col": c, "how": "close", "rows_used": r["rows_used"], "rel": r["rel"]}
    return close


# ---------------------------------------------------------------- nhóm dòng
def group_rows(t: Table, by: str | None = None, start_when_notna: str | None = None,
               kinds: pd.Series | None = None) -> tuple[pd.Series, dict]:
    """Định nhóm (đơn) cho dòng dữ liệu. by: theo giá trị cột (đã strip); start_when_notna: nhóm mới khi cột có
    giá trị; cả hai None: mỗi dòng một nhóm (sổ không ghi theo đơn)."""
    df = t.df
    data = (kinds.isin(["du_lieu", "tiep_noi", "dong_phu_don"]) if kinds is not None
            else pd.Series(True, index=df.index))
    g = pd.Series(pd.NA, index=df.index, dtype=object)
    if by is not None:
        key = df[by].map(lambda v: clean(v) or pd.NA)
        if key[data].isna().any() and start_when_notna is None:
            key = key.where(data).ffill()  # mã chỉ ghi dòng đầu
        g[data] = key[data]
    elif start_when_notna is not None:
        start = ~df[start_when_notna].map(is_blank) & data
        g[data] = start.cumsum()[data]
    else:
        g[data] = range(int(data.sum()))
    sizes = g[data].value_counts()
    odd = []
    if start_when_notna is not None and data.any():
        first = df.index[data][0]
        if is_blank(df.at[first, start_when_notna]):
            odd.append({"group": None, "why": "các dòng đầu bảng không có dòng mở nhóm"})
    hist = sizes.value_counts().sort_index()
    return g, {"n_groups": int(sizes.size), "size_hist": {int(k): int(v) for k, v in hist.head(8).items()},
               "max_size": int(sizes.max()) if sizes.size else 0, "odd": odd}


def ffill_group(df: pd.DataFrame, cols: list[str], group: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Điền ô trống của các dòng sau bằng giá trị dòng đầu nhóm (§8). Báo nhóm mà dòng sau ghi KHÁC dòng đầu."""
    out = df.copy()
    rep = {}
    ok = group.notna()
    for c in cols:
        v = out[c].where(~out[c].map(is_blank))
        filled = v[ok].groupby(group[ok]).ffill()
        n_fill = int((v[ok].isna() & filled.notna()).sum())
        nun = v[ok].groupby(group[ok]).nunique()
        out.loc[ok, c] = filled
        rep[c] = {"filled": n_fill, "groups_conflict": int((nun > 1).sum())}
    return out, rep


def group_profile(df: pd.DataFrame, group: pd.Series, col: str, k: int = 5) -> dict:
    ok = group.notna() & ~df[col].map(is_blank)
    nun = df.loc[ok, col].groupby(group[ok]).nunique()
    multi = nun[nun > 1]
    ex = [{"group": str(gid), "values": [clean(x) for x in df.loc[ok & (group == gid), col].unique()[:4]]}
          for gid in multi.index[:k]]
    return {"n_groups": int(nun.size), "n_groups_multi_value": int(multi.size), "examples": ex}


# ---------------------------------------------------------------- dòng trùng
def dup_report(df: pd.DataFrame, cols: list[str], scope: str = "table", group: pd.Series | None = None,
               key: str | None = None, k: int = 3) -> dict:
    """scope: 'table' (trùng hệt mọi cột), 'group' (trùng hệt trong cùng nhóm), 'cross_source' (trùng giữa các
    sheet/file khác nhau, bỏ qua cột nguồn), 'key' (trùng khoá nhưng khác nội dung)."""
    norm = df[cols].map(clean)
    if scope == "key" and key:
        x = norm.assign(_k=norm[key])
        nun = x.groupby("_k")[cols].nunique().gt(1).any(axis=1)
        bad = nun[nun].index
        return {"scope": scope, "n_keys_conflict": int(len(bad)), "examples": [str(b) for b in bad[:k]]}
    if scope == "group" and group is not None:
        norm = norm.assign(_g=group.astype(str))
    if scope == "cross_source" and "_src" in df:
        d = norm.duplicated(keep=False)
        src = df["_src"][d]
        multi = norm[d].assign(_s=src).groupby(cols, dropna=False)["_s"].nunique()
        n = int((multi > 1).sum())
        return {"scope": scope, "n_groups": n, "n_rows_extra": int(norm.duplicated().sum()) if n else 0}
    dup = norm.duplicated(keep="first")
    ex = [dict(zip(cols[:6], (str(x)[:20] for x in norm.loc[i, cols[:6]]))) for i in norm.index[dup][:k]]
    return {"scope": scope, "n_rows_extra": int(dup.sum()), "n_groups": int(norm[norm.duplicated(keep=False)]
                                                                           .drop_duplicates().shape[0]),
            "examples": ex}


# ---------------------------------------------------------------- tách ô
_RE_PHONE = re.compile(r"(?:\+?84|0)\d{9}\b")
_RE_ADDON = re.compile(r"^\s*(?:(\d+)\s*x?\s+)?(.+?)\s*\(([^)]*)\)\s*$")


def split_name_phone(v: object) -> tuple[str, str | None]:
    s = clean(v)
    m = _RE_PHONE.search(s)
    if not m:
        return s, None
    return " ".join((s[:m.start()] + s[m.end():]).replace("-", " ").split()), m.group(0)


def split_item_addons(v: object, sep: str = " + ") -> dict | None:
    """'Trà sữa (L) + 2 Topping trân châu (10k)' → {main: 'Trà sữa (L)', addons: [{name, qty, amount}]}.
    Phần sau dấu tách chỉ là topping khi ngoặc cuối đọc được thành tiền; không thì trả None (để LLM xem)."""
    s = clean(v)
    parts = s.split(sep)
    if len(parts) < 2:
        return {"main": s, "addons": []}
    addons = []
    for p in parts[1:]:
        m = _RE_ADDON.match(p)
        amt = parse_number(m.group(3)) if m else None
        if m is None or amt is None:
            return None
        addons.append({"name": m.group(2).strip(), "qty": int(m.group(1) or 1), "amount": amt})
    return {"main": parts[0].strip(), "addons": addons}


def split_cell(df: pd.DataFrame, col: str, sep: str | None = None, regex: str | None = None,
               preset: str | None = None, qty_col: str | None = None, k: int = 10) -> tuple[pd.DataFrame, dict]:
    """Tách ô nhiều giá trị. sep → cột `<col>__list` (list); regex có nhóm tên → mỗi nhóm một cột `<col>__<tên>`;
    preset 'name_phone' → `__ten`, `__sdt`; 'item_addons' → `__parts` (dict). qty_col: kiểm số phần tử = SL."""
    out = df.copy()
    s = df[col]
    present = ~s.map(is_blank)
    bad = pd.Series(False, index=df.index)
    new: list[str] = []
    if sep is not None:
        rx = re.compile(sep)
        lst = s.map(lambda v: [x.strip() for x in rx.split(clean(v)) if x.strip()] if not is_blank(v) else [])
        out[f"{col}__list"] = lst
        new.append(f"{col}__list")
        if qty_col:
            q = df[qty_col].map(lambda v: parse_qty(v)[0])
            bad = present & (lst.map(len) != q) & (lst.map(len) > 1)
    elif regex is not None:
        rx = re.compile(regex)
        ms = s.map(lambda v: rx.search(clean(v)) if not is_blank(v) else None)
        for name in rx.groupindex:
            out[f"{col}__{name}"] = ms.map(lambda m, n=name: m.group(n) if m else None)
            new.append(f"{col}__{name}")
        bad = present & ms.isna()
    elif preset == "name_phone":
        sp = s.map(split_name_phone)
        out[f"{col}__ten"], out[f"{col}__sdt"] = sp.map(lambda x: x[0]), sp.map(lambda x: x[1])
        new += [f"{col}__ten", f"{col}__sdt"]
    elif preset == "item_addons":
        parts = s.map(lambda v: split_item_addons(v) if not is_blank(v) else None)
        out[f"{col}__parts"] = parts
        new.append(f"{col}__parts")
        bad = present & parts.isna()
    else:
        raise ValueError("cần sep, regex hoặc preset")
    n, nbad = int(present.sum()), int(bad.sum())
    fails = [{"row": int(i), "value": clean(s[i])[:60]} for i in df.index[bad][:k]]
    return out, {"new_cols": new, "n": n, "ok_rate": round(1 - nbad / n, 4) if n else None, "n_fail": nbad,
                 "failures": fails}
