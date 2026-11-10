"""Bộ não con "Tách câu": LLM nhỏ điền KẾ HOẠCH JSON; code chuyển thành `QuerySpec` có thể chạy.

Chia việc theo số đo trên laptop (Qwen3-4B): điền mẫu JSON có ràng buộc ~17/20 đúng loại chỉ số,
còn tự viết code 0/23. Vì vậy LLM chỉ: chọn thước đo/dạng đáp án/nhóm theo, chép NGUYÊN VĂN các cụm tên
được nhắc, và nói câu có nằm trong phạm vi hỗ trợ không. Thời gian (timeexpr), tên (find.value), luật
(query.py) đều do code. Tên không neo được vào dữ liệu → từ chối, không đoán.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from socai.exec.query import QuerySpec
from socai.io.cells import fold
from socai.llm.client import LLM, extract_json
from socai.rules.timeexpr import Period, TimeSpec, month_bounds, shift_month
from socai.semantic.entities import EntityIndex

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "measure": {"enum": ["revenue", "qty", "orders", "aov", "customers", "other"]},
        "shape": {"enum": ["value", "top", "best", "share", "series", "compare"]},
        "group_by": {"enum": ["none", "product", "product_group", "customer", "day", "month", "payment_method",
                              "channel", "staff", "hour"]},
        "top_k": {"type": "integer"},
        "lowest": {"type": "boolean"},
        "mentions": {"type": "array", "items": {"type": "string"}},
        "exclude_mentions": {"type": "array", "items": {"type": "string"}},
        "size": {"enum": ["none", "M", "L"]},
        "compare": {"enum": ["none", "pct", "diff"]},
    },
    "required": ["supported", "measure", "shape", "group_by", "top_k", "lowest", "mentions", "exclude_mentions",
                 "size", "compare"],
}

SYSTEM = """Bạn tách câu hỏi của chủ cửa hàng thành KẾ HOẠCH JSON. KHÔNG trả lời câu hỏi, KHÔNG tính toán.
Câu có thể viết tắt, không dấu, teencode, trộn tiếng Anh: hãy hiểu nghĩa.

measure: revenue = doanh thu/tiền bán/bán được bao nhiêu tiền · qty = số lượng bán (ly, chai, kg, viên, bán chạy)
  · orders = số đơn · aov = giá trị trung bình một đơn · customers = số khách · other = khác
shape: value = một con số · top = xếp hạng/đứng đầu/top k/nào nhất · best = ngày/tháng/giờ nào cao nhất
  · share = chiếm bao nhiêu % · series = liệt kê từng kỳ (lần lượt, mỗi tháng) · compare = tăng/giảm so với kỳ khác
group_by: thứ được xếp hạng hoặc liệt kê (product = mặt hàng/món, product_group = nhóm hàng, customer = khách,
  day/month/hour, payment_method, channel = kênh bán, staff = nhân viên); none nếu không xếp hạng
top_k: số trong "top 3", "3 mặt hàng"; 0 nếu không nêu số
lowest: true nếu hỏi ít nhất/chậm nhất/thấp nhất
mentions: các cụm tên dùng để LỌC, chép nguyên văn từ câu (tên món, nhóm hàng, khách, nhóm khách, hình thức
  thanh toán, kênh bán). Không đưa từ chung chung như "mặt hàng", "khách", "cửa hàng", không đưa thời gian.
exclude_mentions: cụm bị loại trừ ("không tính đơn COD" → ["COD"])
size: M hoặc L nếu hỏi size; none nếu không
compare: pct nếu hỏi % tăng/giảm, diff nếu hỏi chênh lệch bao nhiêu (số), none nếu không so sánh
supported = false nếu câu hỏi về: công nợ/nợ/trả nợ, nhập hàng/giá nhập, tồn kho/hết hàng/hạn dùng, lãi/giá vốn/chi phí,
  giá bán/đổi giá, ngưỡng điều kiện ("từ 5 đơn trở lên", "trên 500 nghìn"), mặt hàng ế/không bán được,
  thuốc kê đơn/hoạt chất, giao hàng/đợt giao, bảo hành, trả góp, ca làm, hoặc thông tin không có trong sổ bán hàng."""

EXAMPLES = [
    ("dthu tuần rồi bn z", {"supported": True, "measure": "revenue", "shape": "value", "group_by": "none", "top_k": 0,
                             "lowest": False, "mentions": [], "exclude_mentions": [], "size": "none", "compare": "none"}),
    ("cho toi 3 mon ban chay nhat thang nay", {"supported": True, "measure": "qty", "shape": "top", "group_by": "product",
     "top_k": 3, "lowest": False, "mentions": [], "exclude_mentions": [], "size": "none", "compare": "none"}),
    ("Tháng 5/2025 nhóm đồ uống chiếm bao nhiêu % doanh thu?", {"supported": True, "measure": "revenue", "shape": "share",
     "group_by": "none", "top_k": 0, "lowest": False, "mentions": ["nhóm đồ uống"], "exclude_mentions": [], "size": "none",
     "compare": "none"}),
    ("ngày nào trong tháng 3 bán được nhiều tiền nhất", {"supported": True, "measure": "revenue", "shape": "best",
     "group_by": "day", "top_k": 0, "lowest": False, "mentions": [], "exclude_mentions": [], "size": "none", "compare": "none"}),
    ("quán bán được mấy ly size L tháng 6/2025", {"supported": True, "measure": "qty", "shape": "value", "group_by": "none",
     "top_k": 0, "lowest": False, "mentions": [], "exclude_mentions": [], "size": "L", "compare": "none"}),
    ("doanh thu các đơn trả bằng tiền mặt tháng này", {"supported": True, "measure": "revenue", "shape": "value",
     "group_by": "none", "top_k": 0, "lowest": False, "mentions": ["tiền mặt"], "exclude_mentions": [], "size": "none",
     "compare": "none"}),
    ("khách nào mua nhiều tiền nhất quý 2", {"supported": True, "measure": "revenue", "shape": "top", "group_by": "customer",
     "top_k": 1, "lowest": False, "mentions": [], "exclude_mentions": [], "size": "none", "compare": "none"}),
    ("tính đến hôm nay còn bao nhiêu khách nợ", {"supported": False, "measure": "other", "shape": "value", "group_by": "none",
     "top_k": 0, "lowest": False, "mentions": [], "exclude_mentions": [], "size": "none", "compare": "none"}),
    ("thang 4/2026 so voi thang truoc doanh thu tang may %", {"supported": True, "measure": "revenue", "shape": "compare",
     "group_by": "none", "top_k": 0, "lowest": False, "mentions": [], "exclude_mentions": [], "size": "none", "compare": "pct"}),
]

GENERIC = {"mat hang", "san pham", "hang", "mon", "khach", "khach hang", "cua hang", "shop", "quan", "tiem", "don",
           "don hang", "nhom", "nhom hang", "hang hoa", "do", "ly", "doanh thu", "tien"}
FIELD_HINT = [("nhom khach", ("customer_group",)), ("nhom", ("product_group", "customer_group")),
              ("khach hang", ("customer_name", "customer_id")), ("khach", ("customer_name", "customer_id", "customer_group")),
              ("mon", ("product_name",)), ("mat hang", ("product_name",)), ("san pham", ("product_name",)),
              ("hang", ("product_name", "product_group")),
              ("kenh", ("channel",)), ("ban qua", ("channel",)), ("nhan vien", ("staff_id",))]
FILTER_FIELDS = ("product_name", "product_group", "customer_name", "customer_id", "customer_group", "payment_method",
                 "channel", "staff_id")
FIELD_CANON: dict[str, str] = {}  # tên trường khớp → tên cột lọc (hiện trùng nhau)


@dataclass
class Plan:
    raw: dict
    spec: QuerySpec | None = None
    answer_hint: str = ""
    problems: list[str] = field(default_factory=list)
    resolved: list[dict] = field(default_factory=list)
    reasoning: str | None = None


def validate(raw: dict | None) -> dict | None:
    """Kiểm từng trường theo PLAN_SCHEMA (thay cho ràng buộc ngữ pháp lúc sinh). Sai dạng → None (bỏ trống)."""
    if not isinstance(raw, dict):
        return None
    props = PLAN_SCHEMA["properties"]
    out: dict = {}
    for k in PLAN_SCHEMA["required"]:
        v, p = raw.get(k), props[k]
        if "enum" in p:
            if v not in p["enum"]:
                return None
        elif p.get("type") == "boolean":
            if not isinstance(v, bool):
                return None
        elif p.get("type") == "integer":
            v = v if isinstance(v, int) and not isinstance(v, bool) else 0
        elif p.get("type") == "array":
            v = [x for x in v if isinstance(x, str)] if isinstance(v, list) else []
        out[k] = v
    return out


def ask(llm: LLM, question: str, max_tokens: int = 220, constrained: bool = False) -> tuple[dict | None, str | None]:
    """`constrained=False` mặc định: ràng buộc ngữ pháp JSON của llama.cpp làm sinh chậm 5–10 lần trên laptop
    (đo: 60 → 5–12 token/giây); 9 ví dụ mẫu + `validate()` đủ giữ đúng dạng. Dùng CÙNG cách ở laptop và image."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in EXAMPLES:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": question})
    call = llm.chat(msgs, max_tokens=max_tokens, json_schema=PLAN_SCHEMA if constrained else None, thinking=False)
    return validate(extract_json(call.content)), call.reasoning


def _resolve_mention(idx: EntityIndex, m: str) -> tuple[str, str] | None:
    f = fold(m)
    fields: tuple[str, ...] = FILTER_FIELDS
    for prefix, hint in FIELD_HINT:
        if f.startswith(prefix + " "):
            body = m.split(" ", len(prefix.split()))[-1]
            hit, _ = idx.resolve(body, hint)
            if hit is None:
                hit, _ = idx.resolve(m, hint)
            return (FIELD_CANON.get(hit.field, hit.field), hit.value) if hit else None
    hit, _ = idx.resolve(m, fields)
    return (FIELD_CANON.get(hit.field, hit.field), hit.value) if hit else None


def _series_periods(spec_t: TimeSpec) -> list[Period]:
    ps = spec_t.periods
    if len(ps) == 1 and ps[0].kind in ("range", "quarter", "year", "ytd"):  # "từng tháng trong quý 2"
        p, out = ps[0], []
        y, m = p.start.year, p.start.month
        while (y, m) <= (p.end.year, p.end.month):
            a, b = month_bounds(y, m)
            out.append(Period(max(a, p.start), min(b, p.end), "month"))
            y, m = shift_month(y, m, 1)
        return out
    return ps


SCAN_FIELDS = ("product_name", "product_group", "customer_name", "customer_group")
R_YESNO = re.compile(r"\b(co|phai|dung)\b.*\bkhong\b\s*$|\bkhong\s*$")
R_UP, R_DOWN = re.compile(r"\btang (lien tuc|deu|dan)\b"), re.compile(r"\bgiam (lien tuc|deu|dan)\b")
R_SPLIT = re.compile(r"\b(moi ben|moi loai|moi nhom|tung loai|tung nhom|lan luot|rieng tung|so .* voi)\b")


def scan_question(idx: EntityIndex, question: str) -> list[tuple[str, str]]:
    """Tên trong danh mục xuất hiện NGUYÊN trong câu (bỏ dấu), dài nhất trước, không chồng lấn.
    Bổ sung cho LLM khi nó chép thiếu tên ('thép hộp 40x80' thay vì cả tên)."""
    f = f" {fold(question)} "
    hits = []
    for field in SCAN_FIELDS:
        for k, v in idx.values.get(field, {}).items():
            if len(k) >= 4 and f" {k} " in f:
                hits.append((len(k), field, v, k))
    out, used = [], []
    for _, field, v, k in sorted(hits, reverse=True):
        if any(k in u for u in used):
            continue
        used.append(k)
        out.append((field, v))
    return out


def _is_time_or_size(m: str, tspec_resolver) -> bool:
    fm = fold(m)
    return bool(re.fullmatch(r"(size )?[ml]", fm)) or bool(tspec_resolver(m).periods)


def to_spec(raw: dict, question: str, tspec: TimeSpec, idx: EntityIndex, today) -> Plan:
    plan = Plan(raw)
    if not raw or not raw.get("supported", False) or raw.get("measure") == "other":
        plan.problems.append("ngoài phạm vi hỗ trợ")
        return plan
    fq = fold(question)
    s = QuerySpec(measure=raw["measure"], shape=raw["shape"])
    s.group_by = None if raw.get("group_by") in (None, "none") else raw["group_by"]
    s.top_k = raw.get("top_k") or None
    s.ascending = bool(raw.get("lowest"))
    s.size = None if raw.get("size") in (None, "none") else raw["size"]
    if s.size is None and (m := re.search(r"\bsize\s*([ml])\b", fq)):
        s.size = m.group(1).upper()
    # "bán được bao nhiêu đĩa/ly/kg" là SỐ LƯỢNG dù LLM chọn doanh thu (đơn vị lấy từ chính dữ liệu shop)
    units = {fold(u) for u in idx.values.get("unit", {}).values()} | {"ly", "chai", "lon", "kg", "dia", "hop", "goi",
                                                                      "vien", "thung", "bao", "cai", "phan", "to"}
    if s.measure == "revenue" and "tien" not in fq.split() and re.search(
            r"\b(bao nhieu|bn|bnhiu|may|tong cong may|bao nhieu cai) (" + "|".join(map(re.escape, units)) + r")\b", fq):
        s.measure = "qty"
        plan.resolved.append({"override": "measure=qty (câu hỏi theo đơn vị)"})
    # liệt kê theo đối tượng là XẾP HẠNG, không phải chuỗi kỳ
    if s.shape == "series" and s.group_by in ("product", "product_group", "customer", "payment_method", "channel", "staff"):
        s.shape = "top"
    # "ngày/tháng/giờ nào cao nhất" khi LLM quên nhóm theo
    if s.shape in ("best", "top") and not s.group_by:
        for pat, g in ((r"\bthang nao\b", "month"), (r"\bngay nao\b", "day"), (r"\b(gio nao|khung gio|gio cao diem)\b",
                                                                             "hour")):
            if re.search(pat, fq):
                s.group_by, s.shape = g, "best"
                break
    trend = None
    if R_YESNO.search(fq) and (R_UP.search(fq) or R_DOWN.search(fq)):
        trend = "up" if R_UP.search(fq) else "down"
        s.shape = "series"
    if s.shape == "series":
        s.periods = _series_periods(tspec)
    elif s.shape in ("top", "best") and len(tspec.periods) > 1:  # "từ tháng 4 tới tháng 6, tháng nào…"
        ps = tspec.periods
        s.periods = [Period(min(p.start for p in ps), max(p.end for p in ps), "range")]
    else:
        s.periods = tspec.periods[:1]
    if s.shape == "compare" or raw.get("compare") in ("pct", "diff"):
        s.shape, s.compare, s.compare_to = "compare", (raw.get("compare") if raw.get("compare") != "none" else "pct"), \
            tspec.compare_to
    scanned = scan_question(idx, question)
    from socai.rules.timeexpr import Resolver
    resolve_t = Resolver(today).resolve
    split_values: list[tuple[str, str]] = []
    for key, target in (("mentions", s.filters), ("exclude_mentions", s.excludes)):
        for m in raw.get(key) or []:
            if fold(m) in GENERIC or not fold(m) or _is_time_or_size(m, resolve_t):
                continue
            hit = _resolve_mention(idx, m)
            if hit is None:  # LLM chép thiếu: lấy tên đầy đủ trong câu chứa cụm này
                cand = [h for h in scanned if fold(m) in fold(h[1]) or fold(h[1]) in fold(m)]
                hit = cand[0] if len(cand) == 1 else None
            plan.resolved.append({"mention": m, "hit": hit})
            if hit is None:
                plan.problems.append(f"không neo được '{m}' vào dữ liệu")
                continue
            target.setdefault(hit[0], [])
            if hit[1] not in target[hit[0]]:
                target[hit[0]].append(hit[1])
            if key == "mentions":
                split_values.append(hit)
    # "doanh thu tiền mặt và ví điện tử, mỗi bên bao nhiêu" → danh sách theo thứ tự câu hỏi
    if s.shape == "value" and len(split_values) >= 2 and len({f for f, _ in split_values}) == 1 and R_SPLIT.search(fq):
        s.shape, s.split = "split", [{f: [v]} for f, v in split_values]
        s.filters = {k: v for k, v in s.filters.items() if k != split_values[0][0]}
    s.trend = trend
    if s.shape in ("top", "best") and not s.group_by:
        plan.problems.append("xếp hạng nhưng thiếu nhóm theo")
    if not s.periods:
        plan.problems.append("không giải được thời gian")
    plan.spec = None if plan.problems else s
    return plan
