"""L4: hiểu câu hỏi. Code làm phần tất định (thời gian, khớp tên với giá trị thật, chọn bảng, gợi ý kiểu đáp án,
truy hồi luật và ví dụ) rồi dựng prompt; LLM chỉ viết một JSON {nghi, loai, don_vi, sql}.

LLM KHÔNG tự tính ngày: kỳ đã được `rules/timeexpr.py` giải thành ngày cụ thể và ghi sẵn trong prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from socai.io.cells import fold
from socai.rules.calendar import LunarCalendar
from socai.rules.timeexpr import TimeSpec, resolve
from socai.v5.bank import Bank
from socai.v5.l2_model import Shop, render_table
from socai.v5.l3_rules import STOP, Rules, tokens

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "nghi": {"type": "string"},
        "loai": {"type": "string", "enum": ["number", "percent", "string", "date", "boolean", "list", "table",
                                              "unanswerable"]},
        "don_vi": {"type": "string", "enum": ["tien", "dem", "so", "phan_tram"]},
        "sql": {"type": "string"},
    },
    "required": ["nghi", "loai", "don_vi", "sql"],
}

SYSTEM = """Bạn là chuyên viên phân tích dữ liệu của một cửa hàng nhỏ ở Việt Nam. Bạn nhận một câu hỏi tiếng Việt đời thường (có thể không dấu, viết tắt, teencode) và viết MỘT câu SQL DuckDB để trả lời trên các bảng của shop.

Trả về đúng một JSON: {"nghi": "...", "loai": "...", "don_vi": "...", "sql": "..."}
- nghi: 1-3 câu ngắn: câu hỏi cần chỉ số gì, bảng/cột nào, luật nào áp dụng.
- loai: kiểu đáp án.
  number = một con số; percent = một tỉ lệ phần trăm (trả 12.5 cho 12,5%); string = một tên/chuỗi; date = một ngày (DATE) hoặc một tháng (chuỗi 'YYYY-MM'); boolean = true/false; list = nhiều phần tử có thứ tự; table = bảng nhiều cột; unanswerable = dữ liệu của shop không ghi loại thông tin mà câu hỏi cần.
- don_vi: tien = số tiền (đồng, số nguyên); dem = số đếm (đơn, khách, món, ngày…); phan_tram = phần trăm; so = số khác (trung bình, tỉ số, số lượng lẻ…).
- sql: một câu SELECT hoặc WITH … SELECT. Với unanswerable thì để sql rỗng.

Quy tắc viết SQL:
1. Chỉ dùng bảng và cột có trong LƯỢC ĐỒ, đúng tên SQL (chữ thường, không dấu). Không bịa cột.
2. Kỳ thời gian đã được giải sẵn trong phần KỲ: dùng đúng các ngày đó (BETWEEN DATE '…' AND DATE '…'), không tự tính lại.
3. Áp dụng các LUẬT được cung cấp (đơn nào được tính, cách tính doanh thu, xếp hạng, làm tròn…). Luật là của bộ dữ liệu này; nếu luật khác thói quen thông thường thì theo luật.
4. Lọc theo tên: dùng đúng giá trị trong mục THỰC THỂ hoặc trong danh sách giá trị của cột. Một thứ có thể được ghi nhiều cách (hoa thường, dấu, khoảng trắng): so bằng hàm khoa(cột) = 'khoá bỏ dấu' như gợi ý trong THỰC THỂ.
5. Kết quả phải khớp loai: number/percent/string/date/boolean → đúng 1 dòng 1 cột; list → 1 cột, đã ORDER BY đúng thứ tự câu hỏi cần và LIMIT đúng số phần tử; table → cột đầu là tên, các cột sau theo thứ tự câu hỏi.
6. Không làm tròn trong SQL (code sẽ làm tròn). Phép chia: dùng phép chia thực và NULLIF để tránh chia cho 0.
7. Khi xếp hạng mà bằng nhau, thêm tiêu chí phụ ổn định theo luật (nếu luật không nói thì theo mã/tên tăng dần).
8. Chỉ chọn unanswerable khi không bảng/cột nào ghi loại thông tin câu hỏi cần, hoặc câu hỏi dùng khái niệm chủ quan mà luật không định nghĩa. Không chọn unanswerable chỉ vì câu khó."""

HINTS = [
    ("percent", r"\b(bao nhieu (%|phan tram)|ti le|ty le|ti trong|ty trong|chiem bao nhieu|tang bao nhieu %|"
                r"giam bao nhieu %|may %|may phan tram)\b|%"),
    ("boolean", r"\bco\b.*\b(khong|ko|k|hok|hong)\b\s*\??$|^(co phai|lieu)\b"),
    ("date", r"\b(ngay nao|thang nao|hom nao|khi nao|luc nao|bao gio)\b"),
    ("list", r"\b(top \d+|\d+ .{0,30}(nhieu nhat|cao nhat|thap nhat|it nhat)|liet ke|nhung .{0,20} nao|cac .{0,20} nao|"
             r"ke ten)\b"),
]


@dataclass
class Ask:
    question: str
    today: date
    time: TimeSpec
    entities: list[dict] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    hint: str = ""
    rules: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)


def answer_hint(question: str) -> str:
    f = fold(question)
    for kind, pat in HINTS:
        if re.search(pat, f):
            return kind
    return ""


def link_entities(question: str, shop: Shop, limit: int = 12) -> list[dict]:
    """Khớp các cụm từ trong câu với GIÁ TRỊ THẬT trong dữ liệu (so trên dạng bỏ dấu). Cụm dài được ưu tiên;
    cụm một từ chỉ nhận khi là giá trị nguyên vẹn của một cột. Không chọn thay LLM: chỉ đưa ứng viên."""
    words = re.findall(r"[a-z0-9]+", fold(question))
    found: list[dict] = []
    taken: set[int] = set()
    for n in range(min(8, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            span = range(i, i + n)
            if taken & set(span):
                continue
            gram = " ".join(words[i:i + n])
            if len(gram) < 3 or gram.isdigit() or all(w in STOP for w in words[i:i + n]):
                continue
            hits = shop.index.get(gram)
            if not hits and n >= 2:  # cụm ≥ 2 từ là phần đầu/cuối/giữa của đúng vài giá trị
                part = [(k, v) for k, v in shop.index.items() if f" {gram} " in f" {k} "]
                if 0 < len(part) <= 3:
                    hits = [h for _, v in part for h in v]
            if hits:
                taken |= set(span)
                by_col: dict[tuple[str, str], list[str]] = {}
                for t, c, v in hits:
                    by_col.setdefault((t, c), []).append(v)
                for (t, c), vals in by_col.items():
                    found.append({"cum": gram, "bang": t, "cot": c, "gia_tri": sorted(set(vals))[:6],
                                  "khoa": sorted({fold(v) for v in vals})[:6]})
    return found[:limit]


def pick_tables(question: str, shop: Shop, entities: list[dict], budget: int) -> tuple[list[str], str]:
    """Chọn bảng đưa vào prompt trong ngân sách ký tự. Bảng có thực thể khớp hoặc trùng từ với câu được ưu tiên;
    hết ngân sách thì bảng còn lại chỉ hiện tên cột (không thống kê) để LLM vẫn biết là có."""
    q = set(tokens(question))
    hit_tables = {e["bang"] for e in entities}
    scored = []
    for i, t in enumerate(shop.tables.values()):
        words = set(tokens(" ".join([t.orig, t.sql] + [f"{c.orig} {c.desc}" for c in t.cols])))
        s = len(q & words) + (5 if t.sql in hit_tables else 0) + min(t.rows, 100000) / 1e6
        scored.append((-s, i, t))
    parts, names, used = [], [], 0
    for _, _, t in sorted(scored, key=lambda x: x[:2]):
        full = render_table(t)
        if used + len(full) <= budget or not parts:
            parts.append(full)
            used += len(full)
        else:
            brief = render_table(t, brief=True)
            if used + len(brief) > budget * 1.3:
                parts.append(f"BẢNG {t.sql} ({t.rows} dòng): " + ", ".join(c.sql for c in t.cols)[:300])
                used += 200
            else:
                parts.append(brief)
                used += len(brief)
        names.append(t.sql)
    return names, "\n\n".join(parts)


def render_time(spec: TimeSpec, shop: Shop) -> str:
    lines = [f"- kỳ {i + 1}: {p.start} → {p.end}" + (f" ({p.label})" if p.label else "")
             for i, p in enumerate(spec.periods)]
    if spec.compare_to:
        lines.append(f"- kỳ so sánh: {spec.compare_to.start} → {spec.compare_to.end}")
    if not lines:
        lines.append("- câu hỏi không nêu kỳ cụ thể (nếu luật không nói khác thì tính trên toàn bộ dữ liệu)")
    if shop.date_range:
        lines.append(f"- dữ liệu của shop: {shop.date_range[0]} → {shop.date_range[1]}")
    return "\n".join(lines)


def analyse(question: str, today: date, shop: Shop, rules: Rules, cal: LunarCalendar | None,
            bank: Bank | None, exclude_biz: str | None = None, k_rules: int = 12, k_examples: int = 4) -> Ask:
    spec = resolve(question, today, cal)
    ents = link_entities(question, shop)
    query = question + " " + " ".join(e["cot"].replace("_", " ") for e in ents)
    cards = rules.search(query, shop.industry, k_rules)
    return Ask(question, today, spec, ents, [], answer_hint(question),
               [f"[{c.path}] {c.text}" for c in cards],
               bank.search(question, k_examples, exclude_biz) if bank else [])


def build_prompt(ask: Ask, shop: Shop, schema_budget: int = 9000) -> list[dict]:
    ask.tables, schema = pick_tables(ask.question, shop, ask.entities, schema_budget)
    parts = [f"CÂU HỎI: {ask.question}", f"HÔM NAY: {ask.today.isoformat()}", "KỲ:\n" + render_time(ask.time, shop)]
    if shop.intro or shop.industry:
        parts.append("SHOP: " + (shop.intro[:400] or "") + (f" (ngành theo luật: {shop.industry})" if shop.industry else ""))
    if shop.conventions:
        parts.append("QUY ƯỚC GHI SỔ CỦA SHOP (README):\n" + "\n".join(shop.conventions)[:1500])
    if ask.entities:
        parts.append("THỰC THỂ (cụm trong câu khớp với giá trị thật trong dữ liệu):\n" + "\n".join(
            f"- \"{e['cum']}\" → {e['bang']}.{e['cot']}: " + ", ".join(f"'{v}'" for v in e["gia_tri"]) +
            f" → lọc: khoa({e['cot']}) IN (" + ", ".join(f"'{k}'" for k in e["khoa"]) + ")"
            for e in ask.entities))
    parts.append("LƯỢC ĐỒ:\n" + schema)
    if ask.rules:
        parts.append("LUẬT LIÊN QUAN (trích rulebook của bộ dữ liệu này):\n" + "\n".join(f"- {r[:500]}" for r in ask.rules))
    if ask.examples:
        parts.append("VÍ DỤ ĐÃ KIỂM CHỨNG (của shop khác: tên bảng/cột khác, chỉ tham khảo cách tính):\n" + "\n".join(
            f"- Hỏi: {e['q']}\n  loai={e['loai']}, don_vi={e.get('don_vi', 'so')}\n  SQL: {e['sql'][:700]}"
            for e in ask.examples))
    if ask.hint:
        parts.append(f"GỢI Ý: dạng câu hỏi cho thấy đáp án nhiều khả năng là kiểu {ask.hint}.")
    parts.append("Trả về JSON.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
