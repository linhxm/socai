"""Đường nhanh bằng luật cho các ý định phổ biến nhất (doanh thu, số đơn, giá trị TB đơn, % thay đổi,
tỉ lệ huỷ, ngày bán nhiều tiền nhất). Nguyên tắc: CHÍNH XÁC hơn PHỦ RỘNG.

Với điểm PA (+1 / 0 / −1), một luật trả lời sai là tệ hơn không trả lời. Vì vậy câu nào có bất kỳ
bộ lọc nào chưa hỗ trợ (mặt hàng, nhóm, khách, hình thức thanh toán, ca, kênh, nhân viên…) đều bị
từ chối ở đây và chuyển cho LLM. Hỗ trợ cả vài cách hỏi tiếng Anh / Trung để minh hoạ IR đa ngữ.
"""

from __future__ import annotations

import re

from socai.io.cells import fold
from socai.nlq.ir import QueryIR
from socai.nlq.normalize import canon

BLOCKERS = re.compile(
    r"\b(nhom|mat hang|san pham|mon|hang nao|khach|thanh vien|tien mat|chuyen khoan|cod|vi dien tu|"
    r"quet the|bang the|the ngan hang|the tin dung|"
    r"ca sang|ca chieu|ca toi|nhan vien|kenh|online|shopee|lazada|tiktok|tai quay|giao hang|ship|size|"
    r"ly|chai|lon nuoc|goi|hop|thung|kg|bao gao|bao xi mang|vien|vi thuoc|bich|phong lanh|ban so|"
    r"gio cao diem|khung gio|thu [2-7]|chu nhat|cuoi tuan|"
    r"no|tra no|ton|nhap|von|lai gop|lai rong|loi|chi phi|voucher|giam gia|thue|vat|phi|tet|"
    r"thuoc|ke don|hoat chat|lo|han dung|xe|tho|bao hanh|tra gop|may cu|chi nhanh|"
    r"tien tuoi|ban qua|don qua|qua kenh|zalo|facebook|dien thoai|website|app|"
    r"customer|product|item|cash|branch|staff|category)\b")

R_REVENUE = re.compile(r"\b(doanh thu|doanh so|ban duoc bao nhieu tien|thu (?:ve|duoc) bao nhieu tien|"
                       r"tien hang|tien ban|revenue|sales)\b|营业额|销售额|收入")
R_AOV = re.compile(r"\b(trung binh|binh quan)\b.*\b(moi|1|mot) don\b|\bmoi don\b.*\b(trung binh|binh quan)\b|"
                   r"\baverage order value\b|\baov\b|客单价")
R_ORDERS = re.compile(r"\b(bao nhieu|may|so|so luong) don(?: hang)?\b(?! vi)|\bhow many orders\b|"
                      r"\bnumber of orders\b|订单数|多少(?:个)?订单")
R_CANCEL = re.compile(r"\bt[iy] le (?:don )?(?:bi )?huy\b|\b(?:bao nhieu|may) (?:%|phan tram) don (?:bi |da )?huy"
                      r"|\bcancel(?:lation)? rate\b|取消率")
R_CANCEL_COUNT = re.compile(r"\b(?:so|bao nhieu|may|dem) (?:luong )?don\b.*\b(?:bi huy|da huy|huy)\b"
                            r"|\bhuy (?:mat )?(?:bao nhieu|may) don\b"
                            r"|\bhow many (?:orders )?(?:were )?cancel")
R_BEST_DAY = re.compile(r"\bngay nao\b")
R_BEST_DAY_MEASURE = re.compile(r"\b(doanh thu|nhieu tien|ban dat|ban duoc nhieu)\b")
R_BEST_DAY_TOP = re.compile(r"\b(cao nhat|nhieu nhat|lon nhat|dat nhat)\b|\bwhich day\b.*\b(highest|most)\b")
# câu có ngưỡng, đếm ngày, so nhiều tháng: chưa hỗ trợ ở đường nhanh
R_UNSUPPORTED = re.compile(r"\b(tro len|tro xuong|it nhat|toi thieu|vuot|dat tu|tren|duoi|hon|thap nhat|"
                           r"co may ngay|bao nhieu ngay|so ngay|nhung ngay|thang nao|tuan nao|quy nao|cac thang|"
                           r"lan luot|tung thang|moi thang|ke ra|liet ke|top)\b|[≥≤<>]")
R_PCT = re.compile(r"%|\bphan tram\b|\btang truong\b|\bpercent|\bgrowth\b|增长率|百分")
R_DIFF = re.compile(r"\b(chenh lech|hon kem|tang bao nhieu tien|giam bao nhieu tien|difference)\b")
R_CMP = re.compile(r"\b(so voi|tang|giam|thay doi|tang truong|lien truoc|ky truoc|cung ky|compared|vs)\b"
                   r"|环比|同比")


def mentions_entity(question: str, names: set[str]) -> bool:
    """Câu có nhắc tên một mặt hàng/khách trong dữ liệu của shop (tên đã fold, ≥ 2 từ)?"""
    f = f" {fold(question)} "
    return any(f" {n} " in f for n in names)


def parse(question: str, entity_names: set[str] | None = None) -> QueryIR | None:
    f = canon(question)
    if BLOCKERS.search(f) or R_UNSUPPORTED.search(f):
        return None
    if entity_names and mentions_entity(question, entity_names):
        return None
    cmp = bool(R_CMP.search(f))
    if R_CANCEL.search(f):
        return QueryIR("cancel_rate", question, answer_type="percent") if not cmp else None
    if R_CANCEL_COUNT.search(f):
        return QueryIR("cancelled_count", question) if not cmp else None
    if R_BEST_DAY.search(f) and R_BEST_DAY_MEASURE.search(f) and R_BEST_DAY_TOP.search(f):
        return QueryIR("best_day_revenue", question, answer_type="date") if not cmp else None
    if R_BEST_DAY.search(f):
        return None
    if R_AOV.search(f):
        return QueryIR("avg_order_value", question) if not cmp else None
    if R_REVENUE.search(f):
        if cmp:
            if R_PCT.search(f):
                return QueryIR("revenue", question, compare="pct_change", answer_type="percent")
            if R_DIFF.search(f):
                return QueryIR("revenue", question, compare="diff")
            return None
        return QueryIR("revenue", question)
    if R_ORDERS.search(f) and not cmp:
        return QueryIR("order_count", question)
    return None
