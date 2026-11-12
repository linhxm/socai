"""Bảng `cong_no` (§4): trạng thái nợ TẠI hết ngày X, dựng lại theo X mỗi lần hỏi.

Nguồn: tổng đơn (số khách phải trả cả đơn, đã trừ giảm giá đơn, gồm phí giao) từ bảng đơn; các khoản trả
từ bảng phiếu thu (mã đơn + tiền + thời gian). Không có đủ hai nguồn này thì shop không có bảng cong_no.
Luật §4 nằm trong bảng: bỏ đơn huỷ (đơn trả hàng vẫn tính), bỏ COD và đơn bán qua sàn TMĐT, chỉ giữ đơn
còn nợ > 0; tuổi nợ tính từ ngày đặt; khách lẻ có trong tổng nhưng không thành dòng khi nhóm theo khách.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from socai.io.cells import clean, fold, parse_date, parse_number
from socai.profile.business import Business
from socai.v2.views import Col, View

MARKETPLACE = ("shopee", "lazada", "tiktok", "tiki", "sendo", "san tmdt", "san")
COD_RE = r"\bcod\b|thu ho|giao hang thu tien"
TOTAL_NAMES = {"tong sau giam gia", "khach phai tra", "tong tien", "tong bill", "tong don", "khach can tra", "can thanh toan", "tong cong", "tong thanh toan"}


def _payments(biz: Business, order_ids: set[str]) -> pd.DataFrame | None:
    """Bảng phiếu thu: cột mã khớp mã đơn (≥ 60%), cột tiền, cột ngày (đọc được ≥ 90%)."""
    best = None
    for t in biz.tables:
        df = t.raw.df
        if df.empty or t.role in ("order_lines", "orders"):
            continue
        oid = t.col("order_id")
        if not oid:
            for c in t.raw.columns:
                v = set(df[c].map(clean)) - {""}
                if v and len(v & order_ids) >= 0.6 * len(v):
                    oid = c
                    break
        when = t.col("receipt_time") or t.col("receipt_date")
        if not when:
            for c in t.raw.columns:
                if any(h in fold(c) for h in ("thoi gian", "ngay", "time", "date")):
                    when = c
                    break
        amount = t.col("receipt_amount")
        if not amount and oid and when:  # chưa gán nghĩa: cột số duy nhất còn lại của sổ thu (≥ 90% đọc được số)
            nums = [c for c in t.raw.columns if c not in (oid, when) and not fold(c).startswith(("ma", "stt", "so phieu"))
                    and df[c].map(parse_number).notna().mean() >= 0.9]
            amount = nums[0] if len(nums) == 1 else None
        if not (oid and when and amount):
            continue
        p = pd.DataFrame({"oid": df[oid].map(clean), "amt": df[amount].map(parse_number),
                          "d": df[when].map(parse_date)})
        p = p[p["oid"].ne("") & p["d"].notna()]
        if best is None or len(p) > len(best):
            best = p
    return best


DEBT_RE = r"\bno\b|debt|mua chiu|tra sau"


def debt_views(biz: Business, don: View | None) -> list[View]:
    """[cong_no (trạng thái tại ngày X), no_them, tra_no (phát sinh trong kỳ)] hoặc [] nếu shop không ghi."""
    if don is None:
        return []
    totals = None
    for t in biz.by_role("orders"):
        oid, tot = t.col("order_id"), t.col("order_total")
        if not tot:  # README không ghi: nhận theo tên cột "tổng tiền/tổng bill/khách cần trả"
            tot = next((c for c in t.raw.columns if fold(c) in TOTAL_NAMES), None)
        if oid and tot:
            totals = pd.Series(t.raw.df[tot].map(parse_number).values, index=t.raw.df[oid].map(clean).values)
            totals = totals[~totals.index.duplicated()]
            break
    if totals is None:
        return []
    pay = _payments(biz, set(totals.index))
    if pay is None or pay.empty:
        return []
    base = don.df.copy()
    base["_oid"] = base["_ma_don"].map(clean)
    base["tong_don"] = base["_oid"].map(totals)
    ch = base["kenh"].map(fold) if "kenh" in base else pd.Series("", index=base.index)
    pm = base["thanh_toan"].map(fold) if "thanh_toan" in base else pd.Series("", index=base.index)
    keep = base["trang_thai"].ne("huỷ") & ~ch.str.startswith(MARKETPLACE) & ~pm.str.contains(COD_RE) & \
        ~ch.str.contains(COD_RE) & base["tong_don"].notna()
    base = base[keep]
    pay = pay.copy()

    def at(x: date) -> pd.DataFrame:
        df = base[base["ngay"] <= x].copy()
        paid = pay[pay["d"] <= x].groupby("oid")["amt"].sum()
        df["da_tra"] = df["_oid"].map(paid).fillna(0.0)
        df["con_no"] = df["tong_don"] - df["da_tra"]
        df = df[df["con_no"] > 0.5]
        df["tuoi_no"] = df["ngay"].map(lambda d: (x - d).days)
        return df

    cols = {c.name: c for c in [
        Col("ma_don", "id", "mã đơn còn nợ"), Col("ngay", "date", "ngày đặt đơn"),
        Col("khach", "cat", "khách nợ (khách lẻ vẫn tính vào tổng nợ, không thành dòng khi nhóm theo khách)",
            don.cols["khach"].samples if "khach" in don.cols else []),
        Col("tong_don", "money", "số khách phải trả cả đơn"), Col("da_tra", "money", "đã trả đến hết ngày X"),
        Col("con_no", "money", "còn nợ của đơn tại hết ngày X (> 0)"),
        Col("tuoi_no", "num", "tuổi nợ (ngày) = X − ngày đặt; đơn đặt hôm X có tuổi 0"),
    ] if c.name in base.columns or c.name in ("da_tra", "con_no", "tuoi_no")}
    v = View("cong_no", "các đơn còn nợ TẠI hết ngày X (X = ngày cuối của time; mặc định hôm nay). "
             "time ở bảng này là mốc 'tính đến hết ngày', không phải kỳ phát sinh. Chỉ dùng cho câu hỏi CÒN NỢ; "
             "hỏi nợ thêm / trả nợ / thu nợ trong kỳ thì dùng bảng no_them / tra_no. Số khách nợ = count_distinct khach.", base, cols,
             key_of={"khach": "_khach_key"})
    v.asof = at  # type: ignore[attr-defined]

    # §4 phát sinh trong kỳ: chỉ đơn GHI NỢ, không huỷ (đơn trả hàng vẫn tính)
    alive = don.df[don.df["trang_thai"].ne("huỷ")].copy()
    alive["_oid"] = alive["_ma_don"].map(clean)
    alive["tong_don"] = alive["_oid"].map(totals)
    apm = alive["thanh_toan"].map(fold) if "thanh_toan" in alive else pd.Series("", index=alive.index)
    odate = pd.Series(alive["ngay"].values, index=alive["_oid"].values)
    odate = odate[~odate.index.duplicated()]
    p2 = pay.assign(od=pay["oid"].map(odate))
    p2 = p2[p2["od"].notna()]
    same_day = p2[p2["d"] == p2["od"]].groupby("oid")["amt"].sum()
    if apm.str.contains(DEBT_RE).any():
        debt = alive[apm.str.contains(DEBT_RE)]
    else:  # sổ không ghi hình thức "ghi nợ": đơn chưa trả đủ ngay trong ngày mua
        debt = alive[(alive["tong_don"] - alive["_oid"].map(same_day).fillna(0.0)) > 0.5]
    debt = debt[debt["tong_don"].notna()]
    more = debt.copy()
    more["so_tien"] = more["tong_don"] - more["_oid"].map(same_day).fillna(0.0)
    keep_cols = [c for c in ("ma_don", "ngay", "khach", "_khach_key", "so_tien") if c in more]
    kcol = [Col("ma_don", "id", "mã đơn ghi nợ"), Col("ngay", "date", "ngày đặt đơn"), Col("khach", "cat", "khách"),
            Col("so_tien", "money", "phần tổng đơn chưa trả ngay trong ngày mua")]
    no_them = View("no_them", "nợ thêm trong kỳ: mỗi dòng là một đơn ghi nợ đặt trong kỳ (§4).", more[keep_cols],
                   {c.name: c for c in kcol if c.name in keep_cols}, "ngay", key_of={"khach": "_khach_key"})
    late = p2[(p2["d"] > p2["od"]) & p2["oid"].isin(set(debt["_oid"]))]
    who = debt.drop_duplicates("_oid").set_index("_oid")
    tr = pd.DataFrame({"ma_don": late["oid"].values, "ngay": late["d"].values,
                       "khach": late["oid"].map(who["khach"]).fillna("").values if "khach" in who else "",
                       "_khach_key": late["oid"].map(who["_khach_key"]).fillna("").values if "_khach_key" in who else "",
                       "so_tien": late["amt"].values})
    tcol = [Col("ma_don", "id", "mã đơn ghi nợ được trả"), Col("ngay", "date", "ngày trả"), Col("khach", "cat", "khách"),
            Col("so_tien", "money", "số tiền trả lần đó (trả sau ngày mua)")]
    tra_no = View("tra_no", "trả nợ / thu nợ trong kỳ: mỗi dòng là một khoản trả SAU ngày mua của đơn ghi nợ (§4).",
                  tr, {c.name: c for c in tcol}, "ngay", key_of={"khach": "_khach_key"})
    return [v, no_them, tra_no]
