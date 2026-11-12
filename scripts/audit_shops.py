"""Kiểm tầng dữ liệu của từng shop, KHÔNG cần đáp án (chạy được trên mọi split, kể cả ngành mới).

Bắt lỗi gán nghĩa/đọc file trước khi chúng làm sai hàng loạt câu: bảng không nhận được vai trò, cột chưa gán
nghĩa, dòng rơi vì không đọc được ngày, trạng thái không quy về 4 loại (§8), ô tiền/số lượng hỏng, tổng dòng lệch
tổng đơn, và shop có những loại thông tin nào rulebook nhắc tới (để biết câu nào `unanswerable`).

    PYTHONPATH=src python scripts/audit_shops.py --split ../train --cache /tmp/cache --out audit.json [--shops biz_101,...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socai.io.cells import clean, parse_number  # noqa: E402
from socai.profile.business import load_business  # noqa: E402
from socai.semantic.canonical import build, norm_status  # noqa: E402
from socai.v2.views import build_views  # noqa: E402

# Loại thông tin rulebook nhắc tới → dấu hiệu có trong shop (vai trò bảng hoặc tag cột)
FACETS = {
    "khách": ({"customers"}, {"customer_id", "customer_name"}),
    "nhóm khách": (set(), {"customer_group"}),
    "phiếu thu / công nợ": ({"payments"}, {"receipt_amount"}),
    "nhập hàng": ({"purchases", "purchase_lines"}, {"purchase_id"}),
    "lô / hạn dùng": ({"lots", "lot_sales"}, {"lot_no", "expiry"}),
    "tồn kho": ({"inventory"}, {"stock_base"}),
    "quy đổi đơn vị": ({"units"}, {"factor"}),
    "lịch sử giá": ({"prices"}, {"price_new"}),
    "giá vốn": (set(), {"cost_unit"}),
    "nhân viên": (set(), {"staff_id"}),
    "kênh bán": (set(), {"channel"}),
    "thanh toán": (set(), {"payment_method"}),
    "giao hàng nhiều đợt": ({"deliveries"}, {"delivery_round"}),
}


def audit(biz_dir: Path, cache: Path | None) -> dict:
    t0 = time.time()
    biz = load_business(biz_dir, cache)
    can = build(biz)
    flags: list[str] = []
    tables = []
    roles, tags = Counter(), set()
    for t in biz.tables:
        untagged = [c for c in t.raw.columns if c not in t.tags]
        roles[t.role] += 1
        tags |= set(t.tags.values())
        tables.append({"name": t.name, "role": t.role, "rows": len(t.raw.df), "cols": len(t.raw.columns),
                       "untagged": untagged[:12]})
        if t.role == "unknown":
            flags.append(f"WARN bảng không nhận được vai trò: {t.name} (cột: {list(t.raw.columns)[:8]})")
    out: dict = {"biz": biz.biz_id, "tables": tables, "issues": list(can.issues)}
    if not can.ok:
        flags.append("FAIL không dựng được bảng đơn chuẩn")
    raw_line_rows = sum(len(t.raw.df) for t in biz.by_role("order_lines"))
    if can.lines is not None:
        ln = can.lines
        n = max(1, len(ln))
        out["lines"] = len(ln)
        out["orders"] = 0 if can.orders is None else len(can.orders)
        out["range"] = [str(can.data_start), str(ln["d"].max())]
        out["has_order_id"] = can.has_order_id
        dropped = raw_line_rows - len(ln)
        out["dropped_rows"] = dropped
        if raw_line_rows and dropped / raw_line_rows > 0.01:
            flags.append(f"WARN {dropped}/{raw_line_rows} dòng hàng bị bỏ (không đọc được ngày hoặc dòng tổng)")
        st = Counter(ln["status"])
        out["status"] = dict(st)
        if st.get("unknown"):
            raws = Counter()
            for t in biz.by_role("order_lines") + biz.by_role("orders"):
                if (c := t.col("status")):
                    raws.update(v for v in t.raw.df[c].map(clean) if v and norm_status(v) == "unknown")
            flags.append(f"WARN {st['unknown']} dòng trạng thái không quy được: {raws.most_common(6)}")
        bad_amt = int(ln["line_amount"].isna().sum())
        if bad_amt / n > 0.005:
            flags.append(f"WARN {bad_amt}/{n} dòng không đọc được thành tiền")
        if "qty_base" in ln and ln["qty_base"].isna().mean() > 0.05:
            flags.append(f"WARN {int(ln['qty_base'].isna().sum())}/{n} dòng chưa quy được về đơn vị cơ bản")
        has_hour = ln["dt"].map(lambda x: x is not None and (x.hour, x.minute) != (0, 0)).any()
        out["has_hour"] = bool(has_hour)
        # Bất biến: tổng thành tiền các dòng của đơn ≈ tổng đơn ghi ở bảng đơn (khi có)
        for t in biz.by_role("orders"):
            oid, tot = t.col("order_id"), t.col("order_total") or t.col("goods_total")
            if not (oid and tot) or not can.has_order_id:
                continue
            ref = {clean(a): parse_number(b) for a, b in zip(t.raw.df[oid], t.raw.df[tot])}
            s = ln.groupby("order_id")["line_amount"].sum()
            pairs = [(s[k], v) for k, v in ref.items() if k in s.index and v is not None]
            if pairs:
                eq = sum(abs(a - b) < 1 for a, b in pairs) / len(pairs)
                out.setdefault("line_vs_order_total", {})[t.name] = round(eq, 3)
                if eq < 0.9:
                    flags.append(f"INFO tổng dòng khớp tổng đơn ({t.name}.{tot}) ở {eq:.0%} đơn "
                                 "(tổng đơn có thể đã trừ giảm giá đơn / cộng phí giao)")
    try:
        views = build_views(biz, can)
        out["views"] = {k: sorted(c for c in v.df.columns if not c.startswith("_"))[:40] for k, v in views.items()}
    except Exception as e:  # noqa: BLE001
        flags.append(f"FAIL dựng view lỗi: {e!r}")
    out["facets"] = {k: bool(r & set(roles) or tg & tags) for k, (r, tg) in FACETS.items()}
    out["flags"] = flags
    out["sec"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shops", default="")
    a = ap.parse_args()
    shops = a.shops.split(",") if a.shops else sorted(p.name for p in (a.split / "data").iterdir() if p.is_dir())
    res = []
    for b in shops:
        try:
            r = audit(a.split / "data" / b, a.cache)
        except Exception as e:  # noqa: BLE001
            r = {"biz": b, "flags": [f"FAIL đọc shop lỗi: {e!r}"]}
        res.append(r)
        fl = r.get("flags", [])
        print(f"{b}: {r.get('orders', '-')} đơn, {r.get('lines', '-')} dòng, {len(fl)} cờ, "
              f"có: {[k for k, v in (r.get('facets') or {}).items() if v]}", flush=True)
        for f in fl:
            print("   ", f[:220], flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
