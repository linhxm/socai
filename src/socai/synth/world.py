"""Thế giới dữ liệu tổng hợp: shop giả có ĐÁP ÁN CHÍNH XÁC, kết xuất ra nhiều định dạng bẩn.

Dùng cho: (1) test hồi quy không cần dữ liệu BTC; (2) test metamorphic — cùng câu hỏi bằng tiếng Việt
có dấu / không dấu / teencode / tiếng Anh / tiếng Trung phải ra cùng đáp án; (3) Track B — kết xuất sổ
thành ẢNH (chụp/scan giả) để đo OCR end-to-end khi đã biết sự thật; (4) dữ liệu tự sinh để fine-tune
(được phép theo Rule 6).

Đáp án vàng tính bằng vòng lặp Python đơn giản trên bản ghi gốc, KHÔNG gọi `socai.exec.metrics`:
nếu dùng chung code thì lỗi của hệ thống sẽ được "chấm đúng" (đánh giá vòng tròn).
"""

from __future__ import annotations

import csv
import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

CATALOG = {
    "grocery": [("Sting dâu 330ml", "nước ngọt", "chai", 10000, ("thùng", 24)),
                ("Mì Hảo Hảo tôm chua cay", "mì gói", "gói", 4500, ("thùng", 30)),
                ("Sữa tươi Vinamilk 180ml", "sữa", "hộp", 8000, ("lốc", 4)),
                ("Gạo ST25", "gạo", "kg", 32000, ("bao", 5)),
                ("Nước mắm Nam Ngư 750ml", "gia vị", "chai", 38000, None),
                ("Dầu ăn Simply 2L", "gia vị", "chai", 115000, None)],
    "cafe": [("Cà phê sữa (M)", "cà phê", "ly", 29000, None), ("Bạc xỉu (L)", "cà phê", "ly", 39000, None),
             ("Trà đào cam sả (M)", "trà", "ly", 35000, None), ("Trà sữa trân châu (L)", "trà sữa", "ly", 45000, None)],
}
NAMES = ["Nguyễn Văn An", "Trần Thị Bình", "Lê Hoàng Cường", "Phạm Thu Dung", "Hoàng Minh Đức", "Vũ Lan Hương"]
STATUS_TEXT = {"done": ["Đã hoàn thành", "xong", "ok", "Hoàn thành"], "cancelled": ["Đã huỷ", "huỷ đơn"],
               "returned": ["Trả hàng", "đổi trả"], "shipping": ["Đang giao"]}


@dataclass
class Line:
    product: str
    group: str
    unit: str
    qty: int
    factor: int
    price: int  # giá của đơn vị dòng
    discount: int

    @property
    def amount(self) -> int:
        return self.qty * self.price - self.discount


@dataclass
class Order:
    oid: str
    at: datetime
    status: str
    customer: str
    lines: list[Line] = field(default_factory=list)

    @property
    def revenue(self) -> int:
        return sum(ln.amount for ln in self.lines)


def make_orders(kind: str, start: date, months: int, rng: random.Random, per_day: tuple[int, int] = (3, 9)
                ) -> list[Order]:
    orders, n = [], 0
    d = start
    end = date(start.year + (start.month - 1 + months) // 12, (start.month - 1 + months) % 12 + 1, 1)
    while d < end:
        for _ in range(rng.randint(*per_day)):
            n += 1
            st = rng.choices(["done", "cancelled", "returned", "shipping"], [90, 5, 3, 2])[0]
            o = Order(f"HD{n:06d}", datetime.combine(d, datetime.min.time()) + timedelta(minutes=rng.randint(360, 1300)),
                      st, rng.choice(NAMES + ["Khách lẻ"] * 3))
            for name, group, base, price, big in rng.sample(CATALOG[kind], rng.randint(1, 3)):
                use_big = big is not None and rng.random() < 0.2
                unit, factor = (big[0], big[1]) if use_big else (base, 1)
                q = rng.randint(1, 3)
                disc = rng.choice([0, 0, 0, 1000 * q])
                o.lines.append(Line(name, group, unit, q, factor, price * factor, disc))
            orders.append(o)
        d += timedelta(1)
    return orders


# ------------------------------------------------------------------ kết xuất bẩn
def messy_money(x: int, rng: random.Random) -> object:
    style = rng.randint(0, 5)
    if style == 0:
        return x
    if style == 1:
        return f"{x:,}"  # 22,000
    if style == 2:
        return f"{x:,}".replace(",", ".") + "đ"  # 22.000đ
    if style == 3 and x % 1000 == 0:
        return f"{x // 1000}k"  # 22k
    if style == 4:
        return f"{x:,}.0000"
    return f"{x:,}".replace(",", ".")


def messy_date(d: date, rng: random.Random) -> object:
    style = rng.randint(0, 3)
    if style == 0:
        return (d - date(1899, 12, 30)).days  # serial Excel
    if style == 1:
        return d.strftime("%d/%m/%Y")
    if style == 2:
        return f"{d.day} thg {d.month} {d.year}"
    return d.isoformat()


README_FLAT = """# Dữ liệu {biz}

Một file phẳng, mỗi dòng một bản ghi (một dòng hàng trong đơn).

## File
- `{file}` ({ext})

## Cột trong `{file}`
- **Mã HĐ**: mã đơn
- **Ngày**: ngày bán
- **Khách**: tên khách (Khách lẻ nếu không lưu)
- **Tình trạng**: trạng thái đơn
- **Hàng**: tên hàng
- **Nhóm**: nhóm hàng
- **ĐVT**: đơn vị tính
- **SL**: số lượng
- **SL lẻ**: số lượng đã đổi ra đơn vị cơ bản
- **Giá**: đơn giá
- **Thành tiền**: thành tiền sau giảm

## Quy ước
- Dữ liệu có đến hết {end}.
"""


def _rows(orders: list[Order], rng: random.Random, messy: bool) -> list[list]:
    rows = []
    for o in orders:
        st = rng.choice(STATUS_TEXT[o.status])
        for i, ln in enumerate(o.lines):
            first = i == 0 or not messy or rng.random() < 0.5  # §8: ô trống ở dòng sau của cùng đơn
            rows.append([o.oid, messy_date(o.at.date(), rng) if messy else o.at.date().isoformat(),
                         o.customer if first else "", st if first else "", ln.product, ln.group, ln.unit,
                         ln.qty, ln.qty * ln.factor, messy_money(ln.price, rng) if messy else ln.price,
                         messy_money(ln.amount, rng) if messy else ln.amount])
    return rows


HEAD = ["Mã HĐ", "Ngày", "Khách", "Tình trạng", "Hàng", "Nhóm", "ĐVT", "SL", "SL lẻ", "Giá", "Thành tiền"]


def write_business(root: Path, biz: str, fmt: str, orders: list[Order], rng: random.Random) -> None:
    d = root / "data" / biz
    d.mkdir(parents=True, exist_ok=True)
    end = max(o.at.date() for o in orders)
    if fmt == "csv":
        f = "ban_hang.csv"
        with (d / f).open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEAD)
            w.writerows(_rows(orders, rng, messy=True))
    elif fmt == "xlsx":
        import openpyxl

        f = "SoBanHang.xlsx"
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        by_month: dict[tuple[int, int], list[Order]] = {}
        for o in orders:
            by_month.setdefault((o.at.year, o.at.month), []).append(o)
        for (y, m), os_ in sorted(by_month.items()):
            ws = wb.create_sheet(f"T{m:02d}-{y}")
            ws.append(["SỔ BÁN HÀNG"])  # dòng tiêu đề báo cáo phía trên header
            ws.append([f"Tháng {m}/{y}"])
            ws.append(HEAD)
            for r in _rows(os_, rng, messy=True):
                ws.append(r)
            ws.append(["", "", "", "", f"Tổng tháng {m}/{y}", "", "", "", "", "",
                       sum(o.revenue for o in os_)])  # dòng tổng chèn trong sổ
        wb.save(d / f)
    elif fmt == "sqlite":
        f = "pos.sqlite"
        con = sqlite3.connect(d / f)
        con.execute(f"create table ban_hang ({', '.join(chr(34) + h + chr(34) for h in HEAD)})")
        con.executemany(f"insert into ban_hang values ({','.join('?' * len(HEAD))})", _rows(orders, rng, messy=False))
        con.commit()
        con.close()
        f = "pos.sqlite"
    else:
        raise ValueError(fmt)
    text = README_FLAT.format(biz=biz, file=f if fmt != "sqlite" else "ban_hang", ext=fmt, end=end.strftime("%d/%m/%Y"))
    if fmt == "sqlite":
        text = text.replace("## File\n- `ban_hang` (sqlite)", "## File\n- `pos.sqlite` (sqlite)")
    (d / "README.md").write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ câu hỏi đa ngữ + đáp án vàng
TEMPLATES = {
    "revenue": {"vi": "Doanh thu tháng {m}/{y} là bao nhiêu?", "vi_teen": "dthu thang {m}/{y} bn z",
                "en": "What was the revenue in {m}/{y}?", "zh": "{y}年{m}月的营业额是多少?"},
    "order_count": {"vi": "Tháng {m}/{y} có bao nhiêu đơn hàng?", "vi_teen": "th {m}/{y} co may don",
                    "en": "How many orders in {m}/{y}?", "zh": "{y}年{m}月有多少个订单?"},
}


def _gold(orders: list[Order], metric: str, y: int, m: int) -> int:
    valid = [o for o in orders if o.at.year == y and o.at.month == m and o.status in ("done", "shipping")]
    return sum(o.revenue for o in valid) if metric == "revenue" else len(valid)


def write_questions(root: Path, plan: dict[str, list[Order]], rng: random.Random) -> int:
    qs, ans = [], []
    for biz, orders in plan.items():
        months = sorted({(o.at.year, o.at.month) for o in orders})
        as_of = max(o.at.date() for o in orders)
        for y, m in months[:-1]:
            for metric, langs in TEMPLATES.items():
                gold = _gold(orders, metric, y, m)
                for lang, tpl in langs.items():
                    qid = f"s_{biz}_{metric}_{y}{m:02d}_{lang}"
                    qs.append({"id": qid, "business_id": biz, "question": tpl.format(y=y, m=m),
                               "as_of_date": as_of.isoformat(), "lang": lang,
                               "metamorphic_group": f"{biz}:{metric}:{y}{m:02d}"})
                    ans.append({"id": qid, "type": "number", "answer": gold})
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")
    (root / "answers.jsonl").write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in ans), encoding="utf-8")
    return len(qs)


RULEBOOK_MIN = """# Rulebook (tổng hợp)

## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn không huỷ và không trả hàng; đơn đang giao vẫn tính |
| Doanh thu một đơn | tổng thành tiền các dòng hàng (đã trừ giảm giá trên từng dòng) |
"""


def render_ledger_image(orders: list[Order], path: Path, rng: random.Random) -> list[list[str]]:
    """Vẽ một trang sổ bán hàng thành ảnh (xoay nhẹ, nhiễu) + trả lưới sự thật để chấm OCR."""
    from PIL import Image, ImageDraw, ImageFilter

    grid = [["Ngày", "Hàng", "SL", "Thành tiền"]] + [
        [o.at.strftime("%d/%m"), ln.product, str(ln.qty), f"{ln.amount:,}".replace(",", ".")]
        for o in orders[:25] for ln in o.lines[:1]]
    img = Image.new("L", (1240, 60 + 44 * len(grid)), 245)
    dr = ImageDraw.Draw(img)
    xs = [40, 200, 760, 880]
    for i, row in enumerate(grid):
        for x, cell in zip(xs, row):
            dr.text((x + rng.randint(-3, 3), 40 + 44 * i + rng.randint(-2, 2)), cell, fill=20)
    img = img.rotate(rng.uniform(-1.5, 1.5), fillcolor=245, expand=True).filter(ImageFilter.GaussianBlur(0.6))
    img.save(path)
    return grid


def generate_split(out: Path, n_biz: int = 3, seed: int = 0, render_images: bool = False) -> dict:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    fmts = ["csv", "xlsx", "sqlite"]
    plan: dict[str, list[Order]] = {}
    for i in range(n_biz):
        biz = f"syn_{i + 1:03d}"
        orders = make_orders(rng.choice(list(CATALOG)), date(2025, 1 + i % 6, 1), 3, rng)
        write_business(out, biz, fmts[i % len(fmts)], orders, rng)
        plan[biz] = orders
        if render_images:
            try:
                img_dir = out / "images" / biz
                img_dir.mkdir(parents=True, exist_ok=True)
                truth = render_ledger_image(orders, img_dir / "so_ban_hang_trang1.png", rng)
                (img_dir / "truth.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
            except ImportError:
                pass
    n_q = write_questions(out, plan, rng)
    (out / "rulebook.md").write_text(RULEBOOK_MIN, encoding="utf-8")
    return {"out": str(out), "businesses": list(plan), "questions": n_q}
