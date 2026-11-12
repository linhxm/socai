"""Cổng "trả lời được không" theo loại thông tin (§7: shop không ghi loại thông tin câu hỏi cần → unanswerable).

Tách khỏi việc ghép biểu thức (research: một ngưỡng cho cả "đúng sai" lẫn "trả lời được" kém hơn hai cổng riêng).
Câu hỏi cần loại thông tin nào: từ khoá trên câu đã bỏ dấu + gỡ Telex. Shop có loại đó không: tên view/cột đã dựng.
Chỉ kết luận `unanswerable` khi chắc: câu nhắc RÕ loại thông tin, và shop KHÔNG có view/cột nào mang loại đó.
Một số loại không bao giờ có trong sổ bán hàng (đánh giá nhân viên, giới tính khách) nhưng vẫn kiểm theo cột, để
shop nào có ghi thì vẫn trả lời được.
"""

from __future__ import annotations

import re

from socai.io.cells import fold
from socai.nlq.normalize import untelex

# loại thông tin → mẫu câu hỏi (đã fold + gỡ Telex). Mẫu chặt: thà bỏ sót còn hơn chặn nhầm.
NEED: list[tuple[str, str]] = [
    ("ton_kho", r"\bton kho\b|\btrong kho\b|\bkho con\b|\bcon ton\b|\bcon lai may\b|\bhang con trong\b"),
    ("nhap_hang", r"\bnhap hang\b|\bnhap vao\b|\bgia nhap\b|\btien nhap\b|\bphieu nhap\b|\bcua hang nhap\b"),
    ("han_dung", r"\bhan dung\b|\bhet han\b|\bhan su dung\b|\bhsd\b"),
    ("cong_no", r"\bcon no\b|\bghi no\b|\btra no\b|\bcong no\b|\bmua chiu\b|\bno them\b|\bcon thieu\b|\bkhach no\b|\bno nhieu\b"),
    ("gioi_tinh", r"\bkhach nu\b|\bkhach nam\b|\bgioi tinh\b"),
    ("dia_chi", r"\bo duong\b|\bkhach duong\b|\bdia chi\b|\bo quan\b|\bo phuong\b"),
    ("danh_gia_nv", r"\blanh nghe\b|\btay nghe\b|\ble phep\b|\bchu dao\b|\bquy men\b|\bkhach khen\b|\bkhen nhan vien\b|\bthan thien\b"),
    ("ly_do_tra", r"\btra hang vi\b|\btra vi\b|\bly do tra\b|\bnguyen nhan tra\b"),
    ("gia_von", r"\blai gop\b|\bloi nhuan\b|\bgia von\b|\btien lai\b|\blai nhat\b|\bloi nhat\b"),
    ("khach", r"\bkhach nao\b|\b(bao nhieu|bn|bnhiu|may|so|dem so) khach\b|\bkhach (da )?mua (it nhat|tu|nhieu)|"
              r"\bai la khach\b|\bkhach (mua|chi) nhieu\b"),
    ("nhom_khach", r"\bnhom khach\b|\bkhach nhom\b|\bkhach than thiet\b|\bkhach vip\b|\bkhach thuoc nhom\b"),
    ("thanh_toan", r"\bcod\b|\bchuyen khoan\b|\bvi dien tu\b|\btra bang\b|\bthanh toan bang\b"),
]

# loại thông tin → mẫu tên view / cột mang thông tin đó. So trên tên đã BỎ dấu gạch dưới: shop SQLite đặt tên dính
# liền ("HanSuDung" → hansudung, "LoaiKhachHang" → loaikhachhang)
HAVE: dict[str, str] = {
    "ton_kho": r"tonkho|toncuoi|tonthuc|thekho|stock|^ton",
    "nhap_hang": r"nhap",
    "han_dung": r"handung|hansudung|hsd|hethan",
    "cong_no": r"^congno$",
    "gioi_tinh": r"gioitinh|gender|^phai$",
    "dia_chi": r"diachi|duong|address",
    "danh_gia_nv": r"danhgia|taynghe|xeploai|rating",
    "ly_do_tra": r"lydo|nguyennhan",
    "gia_von": r"giavon|laigop",
    "khach": r"^khach$|^makhach$",
    "nhom_khach": r"nhomkhach|loaikhach|hangkhach",
    "thanh_toan": r"thanhtoan|phuongthuc|hinhthuctt|pttt",
}


def needs(question: str) -> list[str]:
    f = " ".join(untelex(w) for w in fold(question).split())
    return [k for k, pat in NEED if re.search(pat, f)]


def has(views: dict, facet: str) -> bool:
    pat = re.compile(HAVE[facet])
    for name, v in views.items():
        if pat.search(name.replace("_", "")):
            return True
        for c in v.cols:
            if pat.search(c.split(".")[-1].replace("_", "")) and v.df[c].notna().any():
                return True
    return False


def missing(question: str, views: dict) -> str | None:
    """Lý do `unanswerable` nếu câu cần loại thông tin mà shop không ghi; None nếu không kết luận được."""
    if not views:
        return None
    for facet in needs(question):
        if not has(views, facet):
            return f"shop không ghi {facet.replace('_', ' ')} (§7)"
    return None
