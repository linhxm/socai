"""L2: mô hình shop (v4.md §3). Mỗi bảng sạch của L1 → thẻ bảng (vai trò, nghĩa cột, khoá đơn) → bảng có kiểu trong
DuckDB, kèm cột chuẩn hoá do code dựng: trạng thái 4 loại, khoá khách (khách lẻ = NULL), tên hàng chuẩn (§8),
số lượng quy về đơn vị cơ bản, giờ.

LLM: một lần gọi cho thẻ bảng, một lần cho các giá trị trạng thái/khách lẻ/ngành. Không có LLM: mặc định bằng code.
Code luôn kiểm lại đề xuất của LLM bằng dữ liệu (kiểu cột đọc được, cột có thật).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from socai.llm.client import LLM, extract_json
from socai.v4.cells import KINDS, clean, fold, is_blank, parse_cell, parse_qty, parse_number
from socai.v4.l1 import CleanTable
from socai.v4.rows import split_name_phone

ROLES = ("ban_dong", "don_hang", "danh_muc", "quy_doi", "khach_hang", "nhan_vien", "thu_tien", "nhap_hang",
         "lo_hang", "ton_kho", "chi_phi", "lich_su_gia", "giao_hang", "so_no", "tra_gop", "phieu_dich_vu",
         "bao_hanh", "thu_may_cu", "ghi_tat", "khac")
ROLE_DESC = {
    "ban_dong": "mỗi dòng một mặt hàng trong một đơn bán (có thể kèm thông tin đơn)",
    "don_hang": "mỗi dòng một đơn bán (không có chi tiết mặt hàng)", "danh_muc": "danh mục mặt hàng",
    "quy_doi": "bảng quy đổi đơn vị", "khach_hang": "danh sách khách", "nhan_vien": "nhân viên / thợ",
    "thu_tien": "phiếu thu, các lần khách trả tiền", "nhap_hang": "phiếu / dòng nhập hàng",
    "lo_hang": "lô hàng, hạn dùng, dòng bán theo lô", "ton_kho": "tồn kho / thẻ kho", "chi_phi": "sổ chi phí",
    "lich_su_gia": "lịch sử giá bán", "giao_hang": "các đợt giao hàng", "so_no": "sổ nợ, các lần trả nợ",
    "tra_gop": "hợp đồng trả góp, các lần góp", "phieu_dich_vu": "mỗi dòng một phiếu dịch vụ/sửa chữa",
    "bao_hanh": "lượt bảo hành", "thu_may_cu": "thu máy cũ trừ vào đơn", "ghi_tat": "bảng cách ghi tắt tên",
    "khac": "khác"}
FIELDS = ("ma_don", "stt_dong", "ngay", "gio", "ngay_gio", "ma_khach", "ten_khach", "nhom_khach", "sdt",
          "ma_hang", "ten_hang", "ten_khac", "nhom_hang", "don_vi", "don_vi_co_ban", "he_so_quy_doi", "so_luong",
          "so_luong_co_ban", "don_gia", "thanh_tien", "giam_gia", "tong_don", "phi_ship", "da_tra", "con_no",
          "trang_thai", "ghi_chu", "thanh_toan", "kenh", "nhan_vien", "gia_von", "gia_nhap", "nha_cung_cap",
          "ma_phieu", "ma_lo", "han_dung", "ton", "hoat_chat", "ke_don", "ten_tat", "ten_day_du", "khac")
STATES = ("hoan_thanh", "huy", "tra_hang", "dang_giao")


@dataclass
class ColInfo:
    name: str  # tên cột trong DuckDB
    orig: str
    field: str
    kind: str
    desc: str
    samples: list[str]
    derived: bool = False  # cột chuẩn hoá do code dựng (ưu tiên dùng trong SQL)


@dataclass
class View:
    name: str
    table: str  # id bảng L1
    role: str
    desc: str
    per_order: bool
    cols: list[ColInfo]
    rows: int
    notes: list[str] = field(default_factory=list)


@dataclass
class ShopModel:
    shop: str
    con: duckdb.DuckDBPyConnection
    views: dict[str, View]
    industry: str
    facets: set[str]
    date_range: tuple[str, str] | None
    vocab: dict[str, list[tuple[str, str, str]]]  # fold(giá trị) → [(giá trị gốc, view, cột)]
    readme_intro: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- thẻ bảng
CARD_SCHEMA = {"type": "object", "properties": {
    "vai_tro": {"type": "string", "enum": list(ROLES)},
    "mo_ta": {"type": "string"},
    "theo_don": {"type": "boolean"},
    "cot_ma_don": {"type": "string"},
    "cot": {"type": "array", "items": {"type": "object", "properties": {
        "ten": {"type": "string"}, "nghia": {"type": "string", "enum": list(FIELDS)},
        "kieu": {"type": "string", "enum": list(KINDS)}}, "required": ["ten", "nghia", "kieu"]}}},
    "required": ["vai_tro", "mo_ta", "theo_don", "cot_ma_don", "cot"]}

CARD_PROMPT = """Bạn lập THẺ BẢNG cho một bảng dữ liệu cửa hàng (shop: {intro}).
Bảng "{tid}" (file {files}), {n} dòng dữ liệu. Các cột (mô tả README | kiểu code đoán | mẫu):
{cols}

Trả JSON:
- vai_tro: một trong {roles}
- mo_ta: một câu mô tả bảng
- theo_don: true nếu bảng ghi theo đơn (nhiều dòng thuộc cùng một đơn/hoá đơn/bill), false nếu mỗi dòng là một lần ghi
  riêng (sổ phiếu, danh mục…)
- cot_ma_don: tên cột mã đơn/số hoá đơn/STT đơn (đúng như trên), "" nếu không có
- cot: mỗi cột một mục {{ten, nghia, kieu}}; nghia trong {fields}; kieu trong {kinds}.
  Tiền là money, số lượng là qty, ngày là date, ngày+giờ là datetime, giờ là time, mã là code, còn lại text.
  Cột không rõ nghĩa: nghia = khac."""


def _col_lines(ct: CleanTable) -> str:
    out = []
    for p in ct.profile:
        ex = ", ".join(x["ex"] for x in p["shapes"][:3])
        out.append(f"- {p['col']} | {ct.readme.get(p['col'], '')[:80]} | {p['guess']} | {ex}")
    return "\n".join(out)


def _kw(ct: CleanTable, col: str) -> str:
    return fold(f"{col} {ct.readme.get(col, '')}")


# mặc định khi không có LLM: đoán theo tên cột + mô tả README (chỉ là dự phòng, không dùng khi có LLM)
_FALLBACK = [("ma_don", r"\b(ma don|so ct|bill|hoa don|so hd|ma hd|don so)\b"), ("ngay_gio", r"\b(gio tao|thoi gian)\b"),
             ("ngay", r"\bngay\b"), ("gio", r"\bgio\b"), ("ma_khach", r"\bma (khach|kh|nguoi mua|the)\b"),
             ("ten_khach", r"\b(khach|ten kh|doi tac|ben mua)\b"), ("ma_hang", r"\b(ma hang|ma sp|sku|ma vach|barcode)\b"),
             ("ten_hang", r"\b(ten hang|mat hang|san pham|hang hoa|ten sp|mon)\b"), ("nhom_hang", r"\b(nhom hang|phan loai|danh muc)\b"),
             ("so_luong_co_ban", r"\b(quy doi|co ban|sl le|so luong le)\b"), ("so_luong", r"\b(sl|so luong|slg|sluong)\b"),
             ("don_vi", r"\b(dvt|don vi)\b"), ("thanh_tien", r"\b(thanh tien|tri gia|cong)\b"), ("don_gia", r"\b(don gia|gia ban)\b"),
             ("trang_thai", r"\b(trang thai|tinh trang|tien do|xu ly)\b"), ("ghi_chu", r"\b(ghi chu|note)\b"),
             ("nhan_vien", r"\b(nhan vien|tho)\b"), ("kenh", r"\b(kenh|nguon)\b"), ("thanh_toan", r"\b(thanh toan|phuong thuc)\b")]


def fallback_card(ct: CleanTable) -> dict:
    cols = []
    used = set()
    for c in ct.columns:
        k = _kw(ct, c)
        f = next((f for f, rx in _FALLBACK if f not in used and re.search(rx, k)), "khac")
        used.add(f)
        cols.append({"ten": c, "nghia": f, "kieu": ct.guess.get(c, "text")})
    key = next((x["ten"] for x in cols if x["nghia"] == "ma_don"), "")
    has_item = any(x["nghia"] in ("ten_hang", "ma_hang") for x in cols)
    role = "ban_dong" if key and has_item else "khac"
    return {"vai_tro": role, "mo_ta": "", "theo_don": bool(key), "cot_ma_don": key, "cot": cols}


def make_card(ct: CleanTable, intro: str, llm: LLM | None, trace: list) -> dict:
    fb = fallback_card(ct)
    if llm is None:
        return fb
    prompt = CARD_PROMPT.format(intro=intro[:300], tid=ct.id, files=", ".join(ct.files), n=len(ct.df),
                                cols=_col_lines(ct), roles=list(ROLES), fields=list(FIELDS), kinds=list(KINDS))
    try:
        c = llm.chat([{"role": "user", "content": prompt}], max_tokens=1500, json_schema=CARD_SCHEMA)
        card = extract_json(c.content) or {}
        trace.append({"layer": "L2", "task": "card", "table": ct.id, "reasoning": c.reasoning, "output": card})
    except Exception as e:  # noqa: BLE001
        trace.append({"layer": "L2", "task": "card", "table": ct.id, "error": str(e)[:200]})
        return fb
    return validate_card(ct, card, fb)


def validate_card(ct: CleanTable, card: dict, fb: dict) -> dict:
    """Giữ đề xuất của LLM khi kiểm được bằng dữ liệu; chỗ sai thì lấy mặc định của code."""
    if card.get("vai_tro") not in ROLES:
        card["vai_tro"] = fb["vai_tro"]
    if card.get("cot_ma_don") not in ct.columns:
        card["cot_ma_don"] = ""
    by = {x.get("ten"): x for x in card.get("cot", []) if isinstance(x, dict)}
    cols = []
    for c in ct.columns:
        x = by.get(c) or next(y for y in fb["cot"] if y["ten"] == c)
        f = x.get("nghia") if x.get("nghia") in FIELDS else "khac"
        k = x.get("kieu") if x.get("kieu") in KINDS else ct.guess.get(c, "text")
        if k in ("money", "number", "qty", "date", "datetime", "time", "bool") and _rate(ct, c, k) < 0.9:
            k = ct.guess.get(c, "text") if _rate(ct, c, ct.guess.get(c, "text")) >= 0.9 else "text"
        f = _check_field(ct, c, f, k)
        cols.append({"ten": c, "nghia": f, "kieu": k})
    seen: set[str] = set()
    for x in cols:  # một nghĩa chuẩn chỉ gán cho một cột (cột sau giữ tên gốc)
        if x["nghia"] != "khac" and x["nghia"] in seen:
            x["nghia"] = "khac"
        seen.add(x["nghia"])
    card["cot"] = cols
    card["vai_tro"] = _check_role(card["vai_tro"], {x["nghia"] for x in cols}, len(ct.df))
    key = card.get("cot_ma_don") or ""
    if key and not _codeish(ct, key):  # khoá đơn phải trông như mã (vd cột số bàn "Bàn 3" không phải mã đơn)
        key = ""
    if not key:
        key = next((x["ten"] for x in cols if x["nghia"] == "ma_don" and _codeish(ct, x["ten"])), "") or \
            next((x["ten"] for x in fb["cot"] if x["nghia"] == "ma_don" and _codeish(ct, x["ten"])), "")
    card["cot_ma_don"] = key
    card["theo_don"] = bool(card.get("theo_don") or fb["theo_don"]) and bool(key) and \
        ct.df[key].map(clean).replace("", None).dropna().nunique() < 0.95 * len(ct.df)
    return card


def _codeish(ct: CleanTable, col: str) -> bool:
    s = ct.df[col]
    s = s[~s.map(is_blank)].head(2000).map(clean)
    return bool(len(s)) and s.map(lambda v: " " not in v and any(ch.isdigit() for ch in v)).mean() >= 0.7


_CODE_FIELDS = {"ma_don": "khac", "ma_khach": "ten_khach", "ma_hang": "ten_hang", "ma_lo": "khac", "ma_phieu": "khac"}
_NUM_FIELDS = {"so_luong", "so_luong_co_ban", "don_gia", "thanh_tien", "giam_gia", "tong_don", "phi_ship", "da_tra",
               "con_no", "gia_von", "gia_nhap", "ton"}


def _check_field(ct: CleanTable, col: str, f: str, kind: str) -> str:
    """Nghĩa do LLM gán phải hợp với dữ liệu: cột mã trông như mã (không khoảng trắng, phần lớn có chữ số); cột tiền/
    số lượng có kiểu số; cột ngày có kiểu ngày."""
    s = ct.df[col]
    s = s[~s.map(is_blank)].head(2000).map(clean)
    if not len(s):
        return f
    if f in _CODE_FIELDS:
        codeish = s.map(lambda v: " " not in v and any(ch.isdigit() for ch in v)).mean()
        if codeish < 0.7:
            return _CODE_FIELDS[f]
    if f in _NUM_FIELDS and kind not in ("money", "number", "qty"):
        return "khac"
    if f in ("ngay", "han_dung") and kind not in ("date", "datetime"):
        return "khac"
    return f


def _check_role(role: str, fields: set[str], rows: int) -> str:
    """Vai trò phải có đủ trường tối thiểu; bảng có ngày + mặt hàng + tiền/số lượng mà bị gán vai trò danh mục/quy đổi
    là bảng dòng bán."""
    item = bool(fields & {"ten_hang", "ma_hang"})
    money = bool(fields & {"thanh_tien", "so_luong", "don_gia"})
    dated = bool(fields & {"ngay", "ngay_gio"})
    if role in ("quy_doi", "danh_muc", "khac", "ton_kho") and item and money and dated and "ma_don" in fields:
        return "ban_dong"
    if role == "ban_dong" and not dated and "ma_don" not in fields:
        return "danh_muc" if item else "khac"
    if role == "quy_doi" and "he_so_quy_doi" not in fields and "don_vi_co_ban" not in fields:
        return "danh_muc" if item and not dated else "khac"
    return role


def _rate(ct: CleanTable, col: str, kind: str) -> float:
    s = ct.df[col]
    s = s[~s.map(is_blank)].head(3000)
    if not len(s) or kind in ("text", "code"):
        return 1.0
    ys = ct.df.loc[s.index, "_y"]
    ok = sum(parse_cell(v, kind, int(ys[i]) if pd.notna(ys[i]) else None) is not None for i, v in s.items())
    return ok / len(s)


# ---------------------------------------------------------------- giá trị: trạng thái, khách lẻ
VALUES_SCHEMA = {"type": "object", "properties": {
    "trang_thai": {"type": "array", "items": {"type": "object", "properties": {
        "gia_tri": {"type": "string"}, "loai": {"type": "string", "enum": list(STATES)}}, "required": ["gia_tri", "loai"]}},
    "khach_le": {"type": "array", "items": {"type": "string"}}}, "required": ["trang_thai", "khach_le"]}

VALUES_PROMPT = """Cửa hàng ghi trạng thái đơn và tên khách theo nhiều cách. Rulebook: mỗi đơn có đúng một trong bốn trạng
thái hoan_thanh, huy, tra_hang (kể cả trả một phần), dang_giao. Ghi chú không nhắc trạng thái thì là hoan_thanh.
1) Gán mỗi giá trị dưới đây vào một trạng thái (cột "{scol}"):
{svals}
2) Trong các giá trị cột khách "{kcol}" dưới đây, giá trị nào nghĩa là KHÁCH LẺ / khách vãng lai / không ghi khách
(chép nguyên văn, có thể rỗng):
{kvals}
Trả JSON {{"trang_thai": [{{"gia_tri", "loai"}}], "khach_le": [...]}}."""

_ST_KW = [("huy", r"\b(huy|cancel|bom|khach khong lay)"), ("tra_hang", r"\b(tra|hoan hang|return|doi tra)"),
          ("dang_giao", r"\b(dang giao|dang ship|van chuyen|shipping|dang xu ly|cho giao|dang giao dich)")]
_WALKIN = {"khach le", "kl", "le", "khach vang lai", "vang lai", "khach", "walk in", "guest"}


def _state_kw(v: str) -> str:
    f = fold(v)
    return next((s for s, rx in _ST_KW if re.search(rx, f)), "hoan_thanh")


def map_values(ct: CleanTable, scol: str | None, kcol: str | None, llm: LLM | None, trace: list,
               note_status: bool = False) -> tuple[dict[str, str], set[str]]:
    svals = ct.df[scol].map(clean).value_counts().head(60).index.tolist() if scol else []
    kvc = ct.df[kcol].map(clean).value_counts() if kcol else pd.Series(dtype=int)
    kvals = [v for v in kvc.head(40).index if v and len(v) <= 30]
    smap = {v: _state_kw(v) for v in svals}
    walk = {v for v in kvals if fold(v) in _WALKIN}
    if llm is None or (not svals and not kvals) or (note_status and len(svals) > 60):
        return smap, walk
    prompt = VALUES_PROMPT.format(scol=scol or "-", svals="\n".join(f"- {v}" for v in svals) or "(không có)",
                                  kcol=kcol or "-", kvals=", ".join(kvals) or "(không có)")
    try:
        c = llm.chat([{"role": "user", "content": prompt}], max_tokens=1200, json_schema=VALUES_SCHEMA)
        out = extract_json(c.content) or {}
        trace.append({"layer": "L2", "task": "values", "table": ct.id, "reasoning": c.reasoning, "output": out})
        for x in out.get("trang_thai", []):
            v, st = x.get("gia_tri"), x.get("loai")
            if v in smap and st in STATES:
                kw = _state_kw(v)
                # chữ ghi rõ huỷ/trả thì theo chữ; LLM chỉ được quyết khi chữ không nói rõ (vd "bom hàng", "khách bùng")
                smap[v] = kw if kw in ("huy", "tra_hang") else st
        n = max(len(ct.df), 1)
        # khách lẻ là giá trị ngắn, gặp nhiều; tên người thật (≥ 3 từ) hoặc hiếm thì không phải
        lw = {v for v in out.get("khach_le", []) if v in set(kvals)
              and (fold(v) in _WALKIN or (len(v.split()) <= 2 and kvc.get(v, 0) >= 0.01 * n))}
        walk = lw | walk
    except Exception as e:  # noqa: BLE001
        trace.append({"layer": "L2", "task": "values", "table": ct.id, "error": str(e)[:200]})
    return smap, walk


# ---------------------------------------------------------------- dựng bảng
def _slug(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", fold(s))).strip("_") or "cot"


def _groups_and_ffill(df: pd.DataFrame, cols: list[str], key: str) -> tuple[pd.DataFrame, list[str], pd.Series | None]:
    """Nhóm đơn + điền ô nhóm để trống (§8). Cột được điền là cột có mẫu ô trống trùng với mẫu ô trống của cột khoá
    (ô đơn chỉ ghi ở dòng đầu)."""
    if not key:
        return df, [], None
    kb = df[key].map(is_blank)
    if not kb.any():
        return df, [], df[key].map(clean)
    filled = []
    out = df.copy()
    for c in cols:
        if c == key:
            continue
        cb = out[c].map(is_blank)
        if cb.any() and (cb & ~kb).sum() <= 0.01 * len(df) and (cb & kb).sum() >= 0.9 * kb.sum():
            filled.append(c)
    grp = (~kb).cumsum()
    out[key] = out[key].where(~kb).ffill()
    for c in filled:
        out[c] = out[c].where(~out[c].map(is_blank)).groupby(grp).ffill()
    return out, [key] + filled, out[key].map(clean)


def _typed(s: pd.Series, kind: str, years: pd.Series) -> pd.Series:
    if kind in ("money", "number"):
        return pd.to_numeric(s.map(lambda v: parse_number(v) if not is_blank(v) else None), errors="coerce")
    if kind == "qty":
        return pd.to_numeric(s.map(lambda v: parse_qty(v)[0] if not is_blank(v) else None), errors="coerce")
    if kind in ("date", "datetime", "time", "bool", "pct"):
        vals = [parse_cell(v, kind, int(y) if pd.notna(y) else None) for v, y in zip(s, years)]
        if kind == "date":
            return pd.to_datetime(pd.Series(vals, index=s.index), errors="coerce")
        if kind == "datetime":
            return pd.to_datetime(pd.Series(vals, index=s.index), errors="coerce")
        if kind == "time":
            return pd.Series([v.strftime("%H:%M:%S") if v else None for v in vals], index=s.index, dtype=object)
        return pd.Series(vals, index=s.index, dtype=object)
    return s.map(lambda v: clean(v) or None)


DERIVED = ("ma_don", "ngay", "gio_so", "trang_thai_chuan", "khach_chuan", "ten_hang_chuan", "so_luong_co_ban")
_RE_TAIL = re.compile(r"^(.*?[^\W\d_].*?)\s+([0-9A-Za-z]*\d[0-9A-Za-z]*[-.][0-9A-Za-z.\-]*\d)$")


def _split_tail_code(s: pd.Series):
    """≥ 90% ô có dạng '<chữ> <mã có số và dấu - hoặc .>' (vd 'xe số 29X1-123.45') → (tên, mã)."""
    v = s.dropna()
    v = v[v.map(lambda x: isinstance(x, str))]
    if len(v) < 20:
        return None
    ok = v.map(lambda x: bool(_RE_TAIL.match(x.strip())))
    if ok.mean() < 0.9 or v.map(lambda x: x.split()[-1]).nunique() < 10:
        return None

    def part(x, g):
        m = _RE_TAIL.match(x.strip()) if isinstance(x, str) else None
        return m.group(g) if m else None
    return _cluster_names(s.map(lambda x: part(x, 1))), s.map(lambda x: part(x, 2))


def _cluster_names(s: pd.Series) -> pd.Series:
    """Gộp cách viết của tên ngắn dạng danh mục (≤ 200 giá trị): cùng chữ khi bỏ dấu/hoa thường/khoảng trắng là một;
    tên chỉ gồm chữ đầu các từ của đúng một tên dài hơn (ab ↔ air blade) là một. Hiển thị: cách viết hay gặp nhất."""
    vc = s.dropna().value_counts()
    if not len(vc) or len(vc) > 200:
        return s
    groups: dict[str, list] = {}
    for v in vc.index:
        groups.setdefault(fold(v).replace(" ", ""), []).append(v)
    for k in [k for k in groups if len(k) <= 3]:
        hits = [k2 for k2, g in groups.items() if len(k2) > 3 and len(fold(g[0]).split()) > 1
                and "".join(w[0] for w in fold(g[0]).split()) == k]
        if len(hits) == 1:
            groups[hits[0]] += groups.pop(k)
    disp = {}
    for g in groups.values():
        best = max(g, key=lambda x: vc[x])
        for x in g:
            disp[x] = best
    return s.map(lambda x: disp.get(x, x) if isinstance(x, str) else x)


def build_view(ct: CleanTable, card: dict, smap: dict, walk: set, name: str) -> tuple[pd.DataFrame, View]:
    cols = {x["ten"]: x for x in card["cot"]}
    key = card.get("cot_ma_don") or ""
    df, filled, grp = _groups_and_ffill(ct.df, ct.columns, key)
    notes = list(ct.notes)
    if filled[1:]:
        notes.append(f"điền ô nhóm để trống theo dòng đầu đơn: {filled[1:]}")
    # §8: trùng hệt trong cùng một đơn tính một lần (cả khi trùng do hai sheet/file chồng nhau); bảng không theo đơn giữ
    if card.get("theo_don") and key:
        before = len(df)
        df = df.loc[~df[ct.columns].map(clean).duplicated()]
        if len(df) < before:
            notes.append(f"bỏ {before - len(df)} dòng trùng hệt trong cùng đơn (§8)")
    out = pd.DataFrame(index=df.index)
    infos: list[ColInfo] = []
    used: set[str] = set(DERIVED)
    for c in ct.columns:
        x = cols[c]
        f, k = x["nghia"], x["kieu"]
        nm = _slug(c)
        while nm in used:
            nm += "_g"
        used.add(nm)
        out[nm] = _typed(df[c], k, df["_y"])
        smp = [clean(v)[:25] for v in df[c].dropna().unique()[:3] if clean(v)]
        infos.append(ColInfo(nm, c, f, k, ct.readme.get(c, ""), smp))
    by_field = {i.field: i.name for i in infos if i.field != "khac"}
    der: list[ColInfo] = []

    def add(nm: str, values, kind: str, desc: str, src: str = "", field: str = "khac", smp=None):
        out[nm] = values
        der.append(ColInfo(nm, src, field, kind, desc, smp or [], True))

    if key:
        add("ma_don", df[key].map(lambda v: clean(v) or None), "code",
            f"mã đơn (cột gốc '{key}'). Số đơn = COUNT(DISTINCT ma_don)", key, "ma_don")
    dsrc = by_field.get("ngay")
    if dsrc:
        add("ngay", out[dsrc], "date", f"ngày (cột gốc '{next(i.orig for i in infos if i.name == dsrc)}')", "", "ngay")
    elif "ngay_gio" in by_field:
        add("ngay", pd.to_datetime(out[by_field["ngay_gio"]]).dt.normalize(), "date", "ngày (từ cột ngày giờ)", "", "ngay")
    hsrc = by_field.get("ngay_gio") or by_field.get("gio")
    if hsrc:
        hv = out[hsrc]
        hours = pd.to_datetime(hv, errors="coerce").dt.hour if by_field.get("ngay_gio") == hsrc else \
            hv.map(lambda v: int(str(v)[:2]) if isinstance(v, str) and str(v)[:2].isdigit() else None)
        add("gio_so", hours, "number", "giờ trong ngày 0–23 (ca sáng 6–13, ca chiều 14–21; giờ cao điểm)", "", "gio")
    st = by_field.get("trang_thai")
    if st:
        src_col = next(i.orig for i in infos if i.name == st)
        add("trang_thai_chuan", df[src_col].map(lambda v: smap.get(clean(v), _state_kw(clean(v))) if clean(v)
                                                else "hoan_thanh"), "text",
            f"trạng thái đơn quy về hoan_thanh | huy | tra_hang | dang_giao (từ cột '{src_col}')", src_col,
            "trang_thai", list(STATES))
    elif "ghi_chu" in by_field and card["vai_tro"] in ("ban_dong", "don_hang"):
        gc = next(i.orig for i in infos if i.name == by_field["ghi_chu"])
        note = df[gc].map(clean)
        if grp is not None:
            note = note.groupby(grp.loc[df.index]).transform(lambda z: " ".join(x for x in z if x))
        add("trang_thai_chuan", note.map(lambda v: _state_kw(v) if v else "hoan_thanh"), "text",
            f"trạng thái đơn đọc từ ghi chú '{gc}' (§8): hoan_thanh | huy | tra_hang | dang_giao", gc, "trang_thai",
            list(STATES))
    kc = by_field.get("ma_khach") or by_field.get("ten_khach")
    if kc:
        src = df[next(i.orig for i in infos if i.name == kc)].map(clean)

        def ck(v):
            if not isinstance(v, str) or not v or v in walk or fold(v) in _WALKIN:
                return None
            return split_name_phone(v)[0] if kc == by_field.get("ten_khach") else v
        kv = src.map(ck)
        if kc == by_field.get("ten_khach"):  # không có mã: chỉ khác hoa thường/khoảng trắng là một khách
            nn = kv.dropna()
            disp = nn.groupby(nn.map(lambda v: " ".join(v.lower().split()))).agg(lambda z: z.value_counts().index[0])
            kv = kv.map(lambda v: disp.get(" ".join(v.lower().split())) if isinstance(v, str) else None)
        add("khach_chuan", kv, "text", "khách đã chuẩn hoá; NULL = khách lẻ (không đếm, không xếp hạng)", "",
            "ma_khach")
    for i in list(infos):  # ô "tên + mã ở cuối" (vd dòng xe + biển số): tách theo hình dạng
        if i.kind == "text" and i.field not in ("ten_khach", "ma_khach"):
            sp = _split_tail_code(out[i.name])
            if sp is not None:
                add(f"{i.name}_ten", sp[0], "text", f"phần tên của cột '{i.orig}' (đã gộp cách viết)", i.orig,
                    smp=[x for x in sp[0].dropna().unique()[:4]])
                add(f"{i.name}_ma", sp[1], "code", f"mã ở cuối ô của cột '{i.orig}' (vd biển số)", i.orig)
    hide = {i.name for i in infos if (i.field == "ngay" and dsrc == i.name) or (key and i.orig == key)}
    infos = der + [i for i in infos if i.name not in hide]  # cột gốc đã có bản chuẩn: không lặp trong danh mục
    view = View(name, ct.id, card["vai_tro"], card.get("mo_ta", ""), bool(card.get("theo_don")), infos, len(out), notes)
    return out.reset_index(drop=True), view


def _canon_items(views: dict[str, View], frames: dict[str, pd.DataFrame]) -> None:
    """Tên hàng chuẩn (§8): theo bảng ghi tắt, rồi theo tên danh mục (so bỏ hoa thường/khoảng trắng/dấu câu),
    còn lại cách viết hay gặp nhất của cùng khoá."""
    alias: dict[str, str] = {}
    for v in views.values():
        f = {c.field: c.name for c in v.cols}
        if v.role == "ghi_tat" and "ten_tat" in f and "ten_day_du" in f:
            for a, b in zip(frames[v.name][f["ten_tat"]], frames[v.name][f["ten_day_du"]]):
                if a and b:
                    alias[" ".join(str(a).lower().split())] = b
                    alias[" ".join(str(b).lower().split())] = b
        if v.role == "danh_muc" and "ten_hang" in f:
            for b in frames[v.name][f["ten_hang"]].dropna():
                alias.setdefault(" ".join(str(b).lower().split()), b)
            if "ten_khac" in f:
                for a, b in zip(frames[v.name][f["ten_khac"]], frames[v.name][f["ten_hang"]]):
                    for x in str(a or "").split(","):
                        if x.strip() and b:
                            alias.setdefault(" ".join(x.lower().split()), b)
    for v in views.values():
        f = {c.field: c.name for c in v.cols}
        if v.role not in ("ban_dong", "nhap_hang", "lo_hang", "ton_kho", "lich_su_gia") or "ten_hang" not in f:
            continue
        s = frames[v.name][f["ten_hang"]]
        k = s.map(lambda x: " ".join(str(x).lower().split()) if isinstance(x, str) and x else None)
        common = s.dropna().groupby(k.dropna()).agg(lambda z: z.value_counts().index[0])
        frames[v.name]["ten_hang_chuan"] = k.map(lambda x: (alias.get(x) or common.get(x)) if isinstance(x, str) else None)
        v.cols.insert(0, ColInfo("ten_hang_chuan", "", "ten_hang", "text",
                                 "tên mặt hàng đã gộp các cách viết (§8); dùng để nhóm/lọc/xếp hạng mặt hàng",
                                 [x for x in frames[v.name]["ten_hang_chuan"].dropna().unique()[:3]], True))


def _join_catalog(views: dict[str, View], frames: dict[str, pd.DataFrame], max_unique: int = 500) -> None:
    """Bảng bán không ghi nhóm hàng / thuộc tính: nối từ danh mục (theo mã trước, rồi theo tên đã chuẩn hoá).
    Thêm cột dm_<cột danh mục> cho thuộc tính ít giá trị (nhóm, hãng, kê đơn, hoạt chất, đơn vị cơ bản, giá vốn…)."""
    cats = [v for v in views.values() if v.role == "danh_muc"]
    for v in views.values():
        if v.role not in ("ban_dong", "nhap_hang", "lo_hang"):
            continue
        d = frames[v.name]
        f = {c.field: c.name for c in v.cols if not c.derived}
        for cat in cats:
            cd = frames[cat.name]
            cf = {c.field: c.name for c in cat.cols if not c.derived}
            pairs = []
            if "ma_hang" in f and "ma_hang" in cf:
                pairs.append((f["ma_hang"], cf["ma_hang"]))
            if "ten_hang" in cf:
                pairs.append(("ten_hang_chuan" if "ten_hang_chuan" in d else f.get("ten_hang"), cf["ten_hang"]))
            for a, b in pairs:
                if not a or a not in d:
                    continue
                key = lambda x: " ".join(str(x).lower().split()) if isinstance(x, str) and x else None  # noqa: E731
                ck = cd[b].map(key)
                if ck.duplicated().any():
                    continue
                lk = d[a].map(key)
                cover = lk.dropna().isin(set(ck.dropna())).mean() if lk.notna().any() else 0
                if cover < 0.6:
                    continue
                for c in cat.cols:
                    if c.derived or c.name == b or cd[c.name].nunique() > max_unique or cd[c.name].isna().all():
                        continue
                    nm = f"dm_{c.name}"
                    if nm in d or c.field in f and c.field in ("nhom_hang",) and False:
                        continue
                    m = dict(zip(ck, cd[c.name]))
                    d[nm] = lk.map(m)
                    v.cols.append(ColInfo(nm, c.orig, c.field, c.kind,
                                          f"[chuẩn hoá] nối từ danh mục '{cat.name}': {c.desc or c.orig}",
                                          [str(x) for x in cd[c.name].dropna().unique()[:3]], True))
                break


_HONOR = {"a", "anh", "chi", "c", "co", "chu", "em", "e", "bac", "ong", "ba", "tho", "thay"}


def _canon_people(views: dict[str, View], frames: dict[str, pd.DataFrame]) -> None:
    """Tên gọi của người (§8: 'A.Tư', 'a Tư', 'Tư' → họ tên đầy đủ trong bảng nhân viên/thợ): bỏ xưng hô, so tên
    cuối; chỉ nhận khi tên cuối ứng với đúng MỘT họ tên. Thêm cột <cột>_chuan cho cột có ≥ 60% giá trị khớp."""
    full: dict[str, list[str]] = {}
    for v in views.values():
        if v.role != "nhan_vien":
            continue
        for c in v.cols:
            if c.kind != "text":
                continue
            vals = [x for x in frames[v.name][c.name].dropna().unique() if isinstance(x, str)]
            if vals and sum(len(x.split()) >= 2 for x in vals) >= 0.8 * len(vals):
                for x in vals:
                    full.setdefault(fold(x).split()[-1], []).append(x)
    if not full:
        return

    def resolve(x):
        if not isinstance(x, str):
            return None
        ws = [w for w in fold(x.replace(".", " ")).split() if w not in _HONOR]
        if not ws:
            return None
        if len(ws) >= 2 and any(fold(f) == " ".join(ws) for g in full.values() for f in g):
            return next(f for g in full.values() for f in g if fold(f) == " ".join(ws))
        g = full.get(ws[-1])
        return g[0] if g and len(g) == 1 else None

    for v in views.values():
        if v.role == "nhan_vien":
            continue
        for c in list(v.cols):
            if c.kind != "text" or c.derived:
                continue
            vals = frames[v.name][c.name].dropna().unique()
            if not 1 <= len(vals) <= 300:
                continue
            hit = sum(resolve(x) is not None for x in vals) / len(vals)
            if hit >= 0.6:
                nm = f"{c.name}_chuan"
                frames[v.name][nm] = frames[v.name][c.name].map(resolve)
                v.cols.insert(0, ColInfo(nm, c.orig, c.field, "text",
                                         f"họ tên đầy đủ của người ở cột '{c.orig}' (đã quy tên gọi về danh sách nhân viên)",
                                         sorted({x for x in frames[v.name][nm].dropna().unique()})[:4], True))


def _units(views: dict[str, View], frames: dict[str, pd.DataFrame]) -> None:
    """so_luong_co_ban cho bảng bán khi chưa có: hệ số theo (hàng, đơn vị) từ bảng quy đổi / danh mục, nhiều cấp."""
    fac: dict[tuple[str, str], tuple[float, str]] = {}
    base: dict[str, str] = {}
    for v in views.values():
        f = {c.field: c.name for c in v.cols}
        d = frames[v.name]
        item = f.get("ten_hang") or f.get("ma_hang")
        if not item:
            continue
        if "don_vi_co_ban" in f:
            for a, b in zip(d[item], d[f["don_vi_co_ban"]]):
                if a and b:
                    base[fold(a)] = fold(b)
        if v.role == "danh_muc" and "don_vi" in f and "don_vi_co_ban" not in f:
            for a, b in zip(d[item], d[f["don_vi"]]):
                if a and b:
                    base.setdefault(fold(a), fold(b))
        if "he_so_quy_doi" in f and "don_vi" in f:
            for a, u, h in zip(d[item], d[f["don_vi"]], d[f["he_so_quy_doi"]]):
                m = re.match(r"\s*([\d.,]+)\s*(.*)$", str(h or ""))
                if a and u and m:
                    fac[(fold(a), fold(u))] = (parse_number(m.group(1)) or 0, fold(m.group(2)))

    def factor(item: str, unit: str, depth: int = 0) -> float | None:
        if not unit or unit == base.get(item) or depth > 4:
            return 1.0 if depth <= 4 else None
        x = fac.get((item, unit))
        if x is None:
            return 1.0 if item in base and (item, unit) not in fac and not any(k[0] == item for k in fac) else None
        n, sub = x
        rest = factor(item, sub, depth + 1) if sub else 1.0
        return n * rest if rest is not None else None

    for v in views.values():
        f = {c.field: c.name for c in v.cols}
        if v.role != "ban_dong" or "so_luong_co_ban" in f or "so_luong" not in f or not (fac or base):
            continue
        d = frames[v.name]
        item = "ten_hang_chuan" if "ten_hang_chuan" in d else f.get("ten_hang") or f.get("ma_hang")
        if not item:
            continue
        unit = d[f["don_vi"]] if "don_vi" in f else pd.Series([None] * len(d))
        q = d[f["so_luong"]]
        d["so_luong_co_ban"] = [qq * factor(fold(it), fold(u)) if qq is not None and pd.notna(qq) and it
                                and factor(fold(it), fold(u)) is not None else None
                                for qq, it, u in zip(q, d[item], unit)]
        v.cols.insert(0, ColInfo("so_luong_co_ban", "", "so_luong_co_ban", "qty",
                                 "số lượng quy về đơn vị cơ bản (§3: bán chạy, bán được bao nhiêu <đơn vị>)", [], True))


INDUSTRY_SCHEMA = {"type": "object", "properties": {"nganh": {"type": "string"}}, "required": ["nganh"]}


def detect_industry(intro: str, table_names: list[str], industries: list[str], llm: LLM | None, trace: list) -> str:
    if not industries:
        return ""
    text = f" {fold(intro + ' ' + ' '.join(table_names))} "
    generic = {"cua", "hang", "shop", "tiem", "nha", "va"}
    def key(x: str) -> str:  # cụm đặc trưng của tên ngành: "Cửa hàng điện thoại" → "dien thoai"
        return " ".join(w for w in fold(x).split() if w not in generic)
    hits = [x for x in industries if key(x) and f" {key(x)} " in text]
    best = hits[0] if len(hits) == 1 else ""
    if llm is None:
        return best
    sch = {**INDUSTRY_SCHEMA, "properties": {"nganh": {"type": "string", "enum": industries + ["khac"]}}}
    try:
        c = llm.chat([{"role": "user", "content": f"Cửa hàng: {intro[:600]}\nCác bảng: {table_names[:12]}\n"
                                                  f"Shop thuộc ngành nào trong {industries}? Không thuộc thì 'khac'. "
                                                  "Trả JSON {\"nganh\": ...}."}], max_tokens=100, json_schema=sch)
        out = (extract_json(c.content) or {}).get("nganh", "")
        trace.append({"layer": "L2", "task": "industry", "output": out})
        return out if out in industries else ""
    except Exception:  # noqa: BLE001
        return best


def build_model(shop: str, tables: list[CleanTable], intro: str, industries: list[str], llm: LLM | None,
                trace: list | None = None) -> ShopModel:
    trace = trace if trace is not None else []
    con = duckdb.connect()
    views: dict[str, View] = {}
    frames: dict[str, pd.DataFrame] = {}
    count: Counter = Counter()
    for ct in tables:
        card = make_card(ct, intro, llm, trace)
        f = {x["nghia"]: x["ten"] for x in card["cot"]}
        smap, walk = map_values(ct, f.get("trang_thai") or (f.get("ghi_chu") if card["vai_tro"] in ("ban_dong", "don_hang")
                                                             and "trang_thai" not in f else None),
                                f.get("ten_khach"), llm, trace, note_status="trang_thai" not in f)
        role = card["vai_tro"]
        count[role] += 1
        name = role if count[role] == 1 else f"{role}_{count[role]}"
        df, v = build_view(ct, card, smap, walk, name)
        views[name], frames[name] = v, df
    _canon_items(views, frames)
    _units(views, frames)
    _canon_people(views, frames)
    _join_catalog(views, frames)
    vocab: dict[str, list[tuple[str, str, str]]] = {}
    for name, df in frames.items():
        con.register(f"_df_{name}", df)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "_df_{name}"')
        con.unregister(f"_df_{name}")
        for c in views[name].cols:
            if c.kind in ("text", "code") and df[c.name].nunique() <= 3000:
                for val in df[c.name].dropna().unique():
                    sv = str(val)
                    if 2 <= len(sv) <= 80:
                        vocab.setdefault(fold(sv), []).append((sv, name, c.name))
    facets = {v.role for v in views.values()} | {c.field for v in views.values() for c in v.cols}
    dr = None
    for v in views.values():
        if v.role in ("ban_dong", "don_hang", "phieu_dich_vu") and any(c.name == "ngay" for c in v.cols):
            r = con.execute(f'SELECT min(ngay), max(ngay) FROM "{v.name}"').fetchone()
            if r and r[0] is not None:
                lo, hi = str(r[0])[:10], str(r[1])[:10]
                dr = (min(dr[0], lo), max(dr[1], hi)) if dr else (lo, hi)
    industry = detect_industry(intro, [ct.id for ct in tables], industries, llm, trace)
    return ShopModel(shop, con, views, industry, facets, dr, vocab, intro)


def catalog(model: ShopModel, names: list[str] | None = None, max_cols: int = 28) -> str:
    """Mô tả bảng cho prompt L4: tên bảng, vai trò, cột (kiểu, nghĩa, mô tả, mẫu)."""
    out = []
    for n in names or list(model.views):
        v = model.views[n]
        out.append(f'Bảng "{n}" ({ROLE_DESC.get(v.role, v.role)}; {v.rows} dòng; '
                   f'{"ghi theo đơn" if v.per_order else "mỗi dòng một bản ghi"}). {v.desc}')
        for c in v.cols[:max_cols]:
            typ = {"money": "số tiền", "number": "số", "qty": "số", "date": "DATE", "datetime": "TIMESTAMP",
                   "time": "giờ HH:MM:SS", "bool": "bool"}.get(c.kind, "chữ")
            d = ("[chuẩn hoá] " if c.derived else "") + (c.desc or (c.orig if c.orig and c.orig != c.name else ""))
            smp = f" vd {c.samples}" if c.samples and c.kind in ("text", "code") else ""
            out.append(f"  - {c.name} [{typ}] {d}{smp}"[:200])
        for nt in v.notes[:3]:
            out.append(f"  * {nt}")
    return "\n".join(out)
