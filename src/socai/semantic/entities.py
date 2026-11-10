"""find.value — neo cụm từ trong câu hỏi vào GIÁ TRỊ THẬT trong dữ liệu của shop.

Đáp án dạng tên phải ghi đúng như trong file (đủ dấu), nên agent không bao giờ tự chép tên từ câu hỏi:
nó lấy giá trị mà chỉ mục này trả về. Tra theo khoá bỏ dấu, rồi chứa nhau theo ranh giới từ, rồi gần
đúng (difflib). Nhiều ứng viên gần điểm nhau → báo mơ hồ, để tầng trên bỏ trống thay vì đoán.
Bảng đồng nghĩa nhỏ cho hình thức thanh toán / kênh bán: câu hỏi nói "tiền mặt", dữ liệu ghi "cash".
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd

from socai.io.cells import clean, fold
from socai.semantic.canonical import Canonical

FIELDS = ("product_name", "product_code", "product_group", "customer_name", "customer_id", "customer_group",
          "payment_method", "channel", "staff_id", "unit")

# cách nói trong câu hỏi → các giá trị (đã fold) có thể gặp trong dữ liệu
ALIASES: dict[str, tuple[str, ...]] = {
    "tien mat": ("cash", "tien mat", "tm"), "tien tuoi": ("cash", "tien mat"),
    "chuyen khoan": ("transfer", "chuyen khoan", "ck", "bank", "banking"),
    "vi dien tu": ("ewallet", "e wallet", "vi dien tu", "momo", "zalopay", "vnpay"),
    "cod": ("cod", "giao hang thu tien"), "ghi no": ("debt", "ghi no", "no", "cong no"),
    "tai quay": ("store", "tai quay", "tai cho", "pos", "offline"), "tai cho": ("store", "tai cho"),
    "dien thoai": ("phone", "dien thoai", "goi dien"), "zalo": ("zalo oa", "zalo"),
    "online": ("online", "web", "website"), "shopee": ("shopee",), "lazada": ("lazada",),
    "tiktok": ("tiktok", "tiktok shop"), "facebook": ("facebook", "fb"), "grab": ("grabfood", "grab"),
    "mang ve": ("takeaway", "mang ve", "mang di"),
}


@dataclass(frozen=True)
class Match:
    field: str
    value: str  # giá trị NGUYÊN VĂN trong dữ liệu
    score: float


class EntityIndex:
    def __init__(self, can: Canonical):
        self.values: dict[str, dict[str, str]] = {}  # field → {khoá fold → giá trị gốc}
        src = can.lines if can.lines is not None else can.orders
        if src is None:
            return
        for f in FIELDS:
            if f in src:
                vals = {}
                for v in pd.unique(src[f].dropna()):
                    k = fold(v)
                    if k and k not in ("khach le", "nan", "none"):
                        vals.setdefault(k, clean(v))
                if vals:
                    self.values[f] = vals

    def find(self, span: str, fields: tuple[str, ...] | None = None) -> list[Match]:
        q = fold(span)
        if not q:
            return []
        keys = {q, *ALIASES.get(q, ())}
        out: list[Match] = []
        for f, vals in self.values.items():
            if fields and f not in fields:
                continue
            for k, v in vals.items():
                if k in keys:
                    s = 1.0
                elif f" {q} " in f" {k} " or f" {k} " in f" {q} ":
                    s = 0.9 * min(len(q), len(k)) / max(len(q), len(k)) + 0.05
                else:
                    s = SequenceMatcher(None, q, k).ratio()
                    if s < 0.88:
                        continue
                    s *= 0.9
                out.append(Match(f, v, round(s, 4)))
        return sorted(out, key=lambda m: (-m.score, m.field, m.value))

    def resolve(self, span: str, fields: tuple[str, ...] | None = None, margin: float = 0.05
                ) -> tuple[Match | None, list[Match]]:
        """(khớp tốt nhất nếu KHÔNG mơ hồ, danh sách ứng viên). Mơ hồ = ≥ 2 giá trị khác nhau trong `margin`."""
        cands = self.find(span, fields)
        if not cands:
            return None, []
        top = cands[0]
        rivals = [c for c in cands[1:] if top.score - c.score <= margin and (c.field, c.value) != (top.field, top.value)
                  and not (c.field.endswith("_code") or c.field.endswith("_id"))]
        if top.score < 0.6 or rivals:
            return None, cands[:5]
        return top, cands[:5]
