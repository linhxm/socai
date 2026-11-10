"""Từ vựng gán nghĩa cột: mô tả README → trường chuẩn (tag), và tên cột → tag khi README thiếu.

Nguồn: CHỈ README của tập train (Rule 6 cấm dùng dữ liệu test để huấn luyện; lexicon là một dạng
dữ liệu huấn luyện). Cột không gán được thì để LLM gán ở bước sau, không đoán ở đây.
Thứ tự quan trọng: cụm cụ thể đứng trước cụm tổng quát ("tên hàng" trước "tên").
"""

from __future__ import annotations

from socai.io.cells import fold

# (tiền tố mô tả đã fold, tag)
DESC_RULES: list[tuple[str, str]] = [
    ("cot trong", "empty"),
    ("so thu tu dong trong phieu nhap", "purchase_line_no"),
    ("so thu tu dong trong don", "line_no"),
    ("so luong khach tra lai", "qty_returned_base"),
    ("so luong ban lay tu lo", "lot_qty_base"),
    ("so luong nhap don vi co ban", "purchase_qty_base"),
    ("so luong nhap", "purchase_qty"),
    ("so luong da doi ra don vi co ban", "qty_base"),
    ("so luong", "qty"),
    ("so khach con no cua don", "debt_remaining"),
    ("so khach phai tra cho ca don", "order_total"),
    ("so khach cua don", "guests"),
    ("so ngay duoc no", "credit_days"),
    ("so dien thoai", "phone"),
    ("so lo", "lot_no"),
    ("so tien", "receipt_amount"),
    ("ma don", "order_id"),
    ("ma hang", "product_code"),
    ("ma khach", "customer_id"),
    ("ma nha cung cap", "supplier_id"),
    ("ma nhan vien", "staff_id"),
    ("ma phieu nhap", "purchase_id"),
    ("ma phieu thu", "receipt_id"),
    ("ngay ban", "order_date"),
    ("thoi diem dat don", "order_datetime"),
    ("gio dat don", "order_time"),
    ("ngay nhap", "purchase_date"),
    ("ngay giao", "delivery_date"),
    ("ngay bat dau ap dung gia", "price_date"),
    ("ngay tao khach", "customer_created"),
    ("ngay tao mat hang", "product_created"),
    ("thoi diem thu", "receipt_time"),
    ("han thanh toan", "due_date"),
    ("han su dung", "expiry"),
    ("ten hang", "product_name"),
    ("ten khach", "customer_name"),
    ("ten", "name"),
    ("thanh tien sau giam", "line_amount"),
    ("tien nhap cua ca dong", "purchase_amount"),
    ("tien hang cua don", "goods_total"),
    ("tong tien hang cua don", "goods_total"),
    ("tong tien phieu nhap", "purchase_total"),
    ("tong tien don", "order_total"),
    ("da thanh toan", "paid"),
    ("con no", "debt_remaining"),
    ("giam gia", "discount"),
    ("gia von", "cost_unit"),
    ("gia nhap mot don vi", "purchase_price"),
    ("gia ban ap dung", "price_new"),
    ("gia truoc lan doi", "price_old"),
    ("gia ban hien tai", "list_price"),
    ("don gia", "unit_price"),
    ("don vi co ban", "base_unit"),
    ("don vi lon", "big_unit"),
    ("don vi nhap", "purchase_unit"),
    ("don vi tinh", "unit"),
    ("he so", "factor"),
    ("trang thai phieu nhap", "purchase_status"),
    ("trang thai don", "status"),
    ("hinh thuc thanh toan cua don", "payment_method"),
    ("hinh thuc thanh toan", "receipt_method"),
    ("kenh ban", "channel"),
    ("nhom hang", "product_group"),
    ("nhom khach", "customer_group"),
    ("nha cung cap", "supplier_name"),
    ("ton kho", "stock_base"),
    ("hoat chat", "active_ingredient"),
    ("thuoc ke don", "prescription"),
    ("bien so xe", "plate"),
    ("tai xe", "driver"),
    ("dot giao", "delivery_round"),
    ("khu vuc cua ban", "table_area"),
    ("ban", "table_no"),
    ("dia chi", "address"),
    ("ghi chu", "note"),
]

# Tên cột thường gặp khi README không có mục "Cột trong ..." (sqlite, nguồn hỗn hợp).
# Khoá là tên cột đã fold; chỉ những tên không mơ hồ.
HEADER_RULES: dict[str, str] = {
    "ma don hang": "order_id", "ma hoa don": "order_id", "ma don": "order_id", "so hd": "order_id",
    "ma chung tu": "order_id", "madon": "order_id", "madonhang": "order_id",
    "ma hang": "product_code", "ma sp": "product_code", "masp": "product_code", "sku": "product_code",
    "ten hang": "product_name", "ten hang hoa": "product_name", "ten san pham": "product_name",
    "tensp": "product_name",
    "so luong": "qty", "sl": "qty", "soluong": "qty",
    "so luong quy doi": "qty_base", "slquydoi": "qty_base", "sl quy doi": "qty_base",
    "thanh tien": "line_amount", "trigia": "line_amount",
    "don gia": "unit_price", "dongia": "unit_price",
    "ngay ban": "order_date", "ngay": "order_date", "ngayban": "order_date",
    "trang thai": "status", "trangthai": "status",
    "khach hang": "customer_name", "ten khach hang": "customer_name",
    "ma khach hang": "customer_id", "makh": "customer_id",
    "nhom hang": "product_group", "nhom khach hang": "customer_group",
    "don vi tinh": "unit", "dvt": "unit",
    "gia von luc ban": "cost_unit", "giavonlucban": "cost_unit",
    "tongthanhtoan": "order_total", "tong thanh toan": "order_total",
    "sotienthu": "receipt_amount", "so tien thu": "receipt_amount", "tien thu": "receipt_amount",
    "ghi chu": "note",
}


def tag_from_description(desc: str | None) -> str | None:
    if not desc:
        return None
    d = fold(desc)
    for prefix, tag in DESC_RULES:
        if d == prefix or d.startswith(prefix + " "):
            return tag
    return None


def tag_from_header(col: str) -> str | None:
    return HEADER_RULES.get(fold(col)) or HEADER_RULES.get(fold(col).replace(" ", ""))
