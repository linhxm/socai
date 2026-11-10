"""Bộ não con "Bảng": LLM gán nghĩa cho cột mà README/lexicon chưa gán được, rồi KIỂM BẰNG DỮ LIỆU.

Cần cho: SQLite / nguồn hỗn hợp mà README không có mục "Cột trong …", và ngành lạ. Không viết luật theo
từng shop. Tag do LLM đề xuất chỉ được giữ khi giá trị trong cột thật sự giống loại đó (cột tiền phải đọc
được thành số, cột ngày phải đọc được thành ngày…). Tag lấy từ README luôn được ưu tiên, LLM không ghi đè.
"""

from __future__ import annotations

import json

from socai.io.cells import clean, fold, parse_date, parse_datetime, parse_number
from socai.llm.client import LLM, extract_json
from socai.profile.business import Business, assign_role, infer_missing_dates, link_order_ids
from socai.semantic.lexicon import DESC_RULES

TAGS = sorted({t for _, t in DESC_RULES if t != "empty"} | {"order_datetime", "receipt_amount", "name"})
NUMERIC = {"qty", "qty_base", "line_amount", "unit_price", "discount", "cost_unit", "order_total", "goods_total",
           "paid", "debt_remaining", "receipt_amount", "purchase_qty", "purchase_qty_base", "purchase_price",
           "purchase_amount", "purchase_total", "factor", "stock_base", "list_price", "price_new", "price_old",
           "qty_returned_base", "lot_qty_base", "guests", "credit_days"}
DATES = {"order_date", "order_datetime", "purchase_date", "delivery_date", "price_date", "receipt_time", "due_date",
         "expiry", "customer_created", "product_created"}

SYSTEM = """Bạn gán NGHĨA cho cột của một bảng dữ liệu bán hàng. Chỉ trả JSON {"tên cột": "tag"}.
Dùng đúng một tag trong danh sách, hoặc "none" nếu không chắc. Không bịa tag mới.
Gợi ý: mã đơn/mã hoá đơn/số chứng từ → order_id; thời điểm tạo đơn → order_datetime; tên hàng → product_name;
thành tiền dòng (đã trừ giảm giá dòng) → line_amount; số khách phải trả cả đơn → order_total;
số tiền một lần thu → receipt_amount; số lượng → qty; số lượng quy về đơn vị cơ bản → qty_base.
Danh sách tag: """ + ", ".join(TAGS)


def _looks_like(tag: str, values: list) -> bool:
    vals = [v for v in values if clean(v)]
    if not vals:
        return False
    if tag in NUMERIC:
        return sum(parse_number(v) is not None for v in vals) / len(vals) >= 0.9
    if tag in DATES:
        f = parse_datetime if tag in ("order_datetime", "receipt_time") else parse_date
        return sum(f(v) is not None for v in vals) / len(vals) >= 0.9
    return True


def tag_with_llm(biz: Business, llm: LLM, max_tables: int = 20, sample: int = 4) -> list[dict]:
    """Gán thêm tag cho các bảng có cột chưa gán; trả nhật ký các quyết định (cho log nộp bài)."""
    log: list[dict] = []
    readme = biz.readme.text[:2500]
    for t in biz.tables[:max_tables]:
        untagged = [c for c in t.raw.columns if c not in t.tags]
        if not untagged or t.raw.df.empty:
            continue
        df = t.raw.df
        cols = {c: [clean(v)[:40] for v in df[c].dropna().head(200) if clean(v)][:sample] for c in t.raw.columns}
        user = (f"README của shop (trích):\n{readme}\n\nBảng: {t.name} ({len(df)} dòng)\n"
                f"Cột đã biết: {json.dumps(t.tags, ensure_ascii=False)}\n"
                f"Cột cần gán (kèm giá trị mẫu):\n" +
                "\n".join(f"- {c}: {json.dumps(cols[c], ensure_ascii=False)}" for c in untagged))
        call = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                        max_tokens=min(900, 60 + 32 * len(untagged)), thinking=False)
        got = {fold(k): v for k, v in (extract_json(call.content) or {}).items() if isinstance(k, str)}
        for c in untagged:
            tag = got.get(fold(c))  # mô hình hay viết thường / bỏ dấu tên cột → so theo khoá fold
            if tag not in TAGS:
                continue
            ok = _looks_like(tag, list(df[c].head(500)))
            log.append({"table": t.name, "column": c, "tag": tag, "kept": ok})
            if ok and tag not in t.tags.values():  # không tạo hai cột cùng nghĩa trong một bảng
                t.tags[c] = tag
        t.role = assign_role(set(t.tags.values()))
    for tn, c in link_order_ids(biz.tables):  # khoá nối suy từ dữ liệu sau khi LLM gán thêm nghĩa
        log.append({"table": tn, "column": c, "tag": "order_id", "kept": True, "via": "data"})
    for t in biz.tables:
        if (c := infer_missing_dates(t)) is not None:
            log.append({"table": t.name, "column": c, "tag": "order_datetime", "kept": True, "via": "data"})
    biz.mapping_log.extend({"via": "llm", **x} for x in log)
    return log
