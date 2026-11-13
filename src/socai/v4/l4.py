"""L4: hiểu câu hỏi (v4.md §5). Chuẩn hoá → khớp viết tắt theo CÁCH viết tắt (không theo danh sách từ) → thời gian
(bộ giải tất định) → khớp thực thể với từ vựng của shop → truy hồi luật → 14B viết spec gồm SQL DuckDB.

Khớp viết tắt: token lạ t khớp cụm p (1–3 từ, lấy từ rulebook + README + tên cột) khi t ghép được từ các dãy con, mỗi
từ của p góp một dãy con BẮT ĐẦU bằng chữ đầu của từ đó: dthu → doanh thu, bnhiu → bao nhiêu, ncc → nhà cung cấp,
trc → trước, hnay → hôm nay, đh → đơn hàng. Cụm hay gặp trong rulebook được ưu tiên.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from socai.rules.timeexpr import Resolver, TimeSpec
from socai.v4.cells import fold
from socai.v4.l2 import ROLE_DESC, ShopModel, catalog
from socai.v4.l3 import Rules

# phép thế chữ của teencode (áp trên dạng bỏ dấu), và biến thể phương ngữ: tập đóng, nhỏ; dùng khi token chưa có
# trong từ vựng, kết quả phải nằm trong từ vựng mới nhận
TEEN = [(r"ck", "ch"), (r"k(?![hiey])", "c"), (r"j", "gi"), (r"j", "d"), (r"w", "qu"), (r"w", "u"), (r"z", "d"),
        (r"z", "v"), (r"f", "ph"), (r"dz", "d"), (r"^ko$", "khong"), (r"^hok$", "khong"), (r"^hem$", "khong")]
# chỉ thay chuỗi KHÔNG thể có trong chữ tiếng Việt: k đứng trước h/i/e/y là hợp lệ ("khai vị" giữ nguyên)
DIALECT = {"ni": "nay", "mo": "nao", "rua": "vay", "nhut": "nhat", "dzay": "vay", "sop": "shop",
           "bua ni": "hom nay", "hom ni": "hom nay", "thang ni": "thang nay", "tuan ni": "tuan nay", "nam ni": "nam nay",
           "hum qua": "hom qua", "hum nay": "hom nay"}  # một từ: chỉ thay khi từ đó không có trong từ vựng shop/rulebook
CORE = ("bao nhieu doanh thu don hang khach hang mat hang thang tuan nam ngay hom nay hom qua truoc nay nao nhat "
        "so luong tien phan tram ti le tang giam so voi cung ky nhap hang ban duoc cua hang nhieu it trung binh "
        "moi gio ca sang ca chieu top liet ke tong cong no tra no con no quy tu dau den het tinh den lan luot "
        "chiem nhan vien tho phieu don vi gom hang roi khong co duoc gi vay the")


def _strip_symbols(q: str) -> str:
    q = unicodedata.normalize("NFC", q)
    return "".join(ch for ch in q if unicodedata.category(ch)[0] not in ("S",) or ch in "%/+-:")


_RE_SYL = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?"
                     r"(uye|uya|ieu|yeu|uoi|uou|oai|oay|oeo|uay|uou|ai|ao|au|ay|eo|eu|ia|ie|iu|oa|oe|oi|oo|ua|ue|ui|uo|"
                     r"uy|ye|a|e|i|o|u|y)"
                     r"(ch|nh|ng|c|m|n|p|t)?$")


def is_syllable(tok: str) -> bool:
    """Hình dạng một âm tiết tiếng Việt (đã bỏ dấu): 'nhom', 'khai', 'nhieu' → True; 'dthu', 'bnhiu', 'ncc' → False."""
    return bool(_RE_SYL.match(tok))


def _subseq_anchor(tok: str, words: list[str]) -> bool:
    """tok ghép được từ dãy con của từng từ (mỗi phần bắt đầu bằng chữ đầu của từ), theo thứ tự, phủ hết tok."""
    def rec(i: int, j: int) -> bool:
        if j == len(words):
            return i == len(tok)
        w = words[j]
        if i >= len(tok) or tok[i] != w[0]:
            return False
        # chọn độ dài phần của từ j: dãy con của w bắt đầu ở w[0]
        k, p = i + 1, 1
        ends = [k]
        while k < len(tok) and p < len(w):
            if tok[k] == w[p]:
                k += 1
                ends.append(k)
            p += 1
        return any(rec(e, j + 1) for e in reversed(ends))
    return rec(0, 0)


@dataclass
class Vocab:
    words: Counter
    phrases: dict[int, Counter]  # số từ → Counter cụm (bỏ dấu)

    @classmethod
    def build(cls, texts: list[str], shop_texts: list[str] = ()) -> "Vocab":
        """texts: rulebook; shop_texts: README, tên/mô tả cột của shop (đếm gấp 3: từ của chính shop ưu tiên hơn)."""
        w: Counter = Counter()
        ph = {1: Counter(), 2: Counter(), 3: Counter()}
        for t, wt in [(x, 1) for x in texts] + [(x, 3) for x in shop_texts] + [(CORE, 3)]:
            ws = fold(t).split()
            for x in ws:
                w[x] += wt
            for n in (1, 2, 3):
                for i in range(len(ws) - n + 1):
                    ph[n][" ".join(ws[i:i + n])] += wt
        return cls(w, ph)

    def candidates(self, tok: str) -> list[str]:
        """Các cách mở token lạ, tốt nhất trước. Rỗng nếu token đã là từ có thật hoặc không mở được."""
        if tok in self.words or tok.isdigit() or len(tok) < 2:
            return []
        for a, b in TEEN:  # teencode: thế rồi phải ra từ có thật
            if re.search(a, tok):
                c = re.sub(a, b, tok)
                if c in self.words and self.words[c] >= 2:
                    return [c]
        if is_syllable(tok):
            return []  # âm tiết hợp lệ (dù từ vựng chưa có): không phải chữ viết tắt
        scored: dict[str, float] = {}
        for n in (2, 3, 1):
            for p, cnt in self.phrases[n].most_common(5000):
                ws = p.split()
                if len(ws) != n or p[0] != tok[0] or len(tok) > len(p.replace(" ", "")):
                    continue
                if n == 1 and len(tok) < 3:
                    continue
                if _subseq_anchor(tok, ws):
                    scored[p] = cnt * (len(tok) / len(p.replace(" ", ""))) * (1.5 if n > 1 else 1.0)
        top = sorted(scored.items(), key=lambda x: -x[1])
        return [p for p, sc in top[:3] if sc >= 2]

    def expand(self, tok: str) -> str | None:
        c = self.candidates(tok)
        return c[0] if c else None


@dataclass
class Normal:
    orig: str
    text: str  # bỏ dấu, đã thay viết tắt / phương ngữ
    changes: list[tuple[str, str]] = field(default_factory=list)


_RE_MY = re.compile(r"(?<![\d/])(\d{1,2})-(\d{4})(?![\d/-])")
_RE_TH = re.compile(r"\b(?:th|t|thg)(\d{1,2})\b")
_RE_UNIT = re.compile(r"\b(th|thg|t|tg|q|w|tuan|ng)\s+(nay|ni|truoc|trc|roi|sau|toi|qua|vua roi)\b")
_UNIT = {"th": "thang", "thg": "thang", "t": "thang", "q": "quy", "w": "tuan", "tuan": "tuan", "ng": "ngay",
         "tg": "thang"}


def normalize(q: str, vocab: Vocab) -> Normal:
    s = fold(_strip_symbols(q).replace("%", " phan tram ").replace("/", " / "))
    s = re.sub(r"\s*/\s*", "/", s)
    ch: list[tuple[str, str]] = []
    for k in sorted(DIALECT, key=lambda x: -len(x)):
        if (" " in k or k not in vocab.words) and re.search(rf"\b{re.escape(k)}\b", s):
            s = re.sub(rf"\b{re.escape(k)}\b", DIALECT[k], s)
            ch.append((k, DIALECT[k]))
    s = _RE_MY.sub(lambda m: f"thang {int(m.group(1))}/{m.group(2)}" if 1 <= int(m.group(1)) <= 12 else m.group(0), s)
    s = _RE_TH.sub(lambda m: f"thang {m.group(1)}", s)
    s = _RE_UNIT.sub(lambda m: f"{_UNIT[m.group(1)]} {m.group(2)}", s)  # "th này", "t trước", "q này"
    toks = []
    for t in s.split():
        cs = vocab.candidates(t) if not re.search(r"\d", t) else []
        if len(cs) == 1 or (len(cs) > 1 and vocab.phrases[len(cs[0].split())][cs[0]] >= 3 * vocab.phrases[len(cs[1].split())][cs[1]]):
            ch.append((t, cs[0]))
            toks.append(cs[0])
        elif cs:  # mơ hồ: giữ token, ghi các nghĩa có thể để LLM chọn theo ngữ cảnh
            ch.append((t, " | ".join(cs[:2])))
            toks.append(f"{t}[{' | '.join(cs[:2])}]")
        else:
            toks.append(t)
    return Normal(q, " ".join(toks), ch)


_RE_CMP_PREV = re.compile(r"\bso (?:voi )?(?:thang|ky|quy|nam) (?:truoc|lien truoc|truoc do|roi)\b")
_RE_CMP_YOY = re.compile(r"\bcung ky (?:nam )?(?:truoc|roi|ngoai|nam ngoai)\b")


def _shift(p, months: int):
    from socai.rules.timeexpr import Period, month_bounds, shift_month
    if p.kind == "month":
        y, m = shift_month(p.start.year, p.start.month, months)
        a, b = month_bounds(y, m)
        return Period(a, b, "month", f"{m}/{y}")
    if p.kind == "quarter":
        y, m = shift_month(p.start.year, p.start.month, months * 3 if abs(months) < 12 else months)
        a, _ = month_bounds(y, m)
        y2, m2 = shift_month(y, m, 2)
        return Period(a, month_bounds(y2, m2)[1], "quarter", f"Q/{y}")
    return None


def resolve_time(q: str, n: Normal, today: date, cal) -> TimeSpec:
    r = Resolver(today, cal)
    a = r.resolve(q)
    if not a.periods:
        a = r.resolve(n.text)
    main = a.main
    # "tháng M so với tháng trước", "cùng kỳ năm trước": kỳ so sánh tính theo kỳ chính, không theo hôm nay
    if main is not None and main.kind in ("month", "quarter") and not main.to_today:
        months = 12 if _RE_CMP_YOY.search(n.text) else (1 if _RE_CMP_PREV.search(n.text) else 0)
        if months:
            cmp_p = _shift(main, -months if main.kind == "month" or months == 12 else -1)
            if cmp_p is not None:
                a.periods = [main]
                a.compare_to = cmp_p
    return a


def answer_hint(t: str) -> str:
    if re.search(r"\b(phan tram|ti le|ty le|ti trong|ty trong|chiem)\b", t):
        return "percent"
    if re.search(r"\b(bang|moi nguoi|moi tho|moi khach|tung nguoi|tung tho)\b", t) and re.search(r"\b(bao nhieu|tien)\b", t):
        return "table"
    if re.search(r"\b(top \d+|\d+ (mat hang|mon|san pham|khach|nhan vien|tho|dong may)|liet ke|nhung (mat hang|khach)|"
                 r"lan luot|moi ben|moi kenh|tung thang)\b", t):
        return "list"
    if re.search(r"\b(ngay nao|hom nao|thang nao|khi nao|han dung)\b", t):
        return "date"
    if re.search(r"\b(co (tang|giam|dung|phai)|dung khong|phai khong|hay khong|lien tuc)\b", t) and \
            re.search(r"\bkhong\b", t):
        return "boolean"
    if re.search(r"\b(ai|nao|gi)\b", t) and re.search(r"\bnhat\b", t):
        return "string"
    return "number"


def link_entities(n: Normal, model: ShopModel, k: int = 12) -> list[dict]:
    words = n.text.split()
    qset = set(words)
    found: dict[str, dict] = {}
    for L in range(min(10, len(words)), 0, -1):
        for i in range(len(words) - L + 1):
            g = " ".join(words[i:i + L])
            if g in model.vocab and (L >= 2 or len(g) >= 4):
                for val, view, col in model.vocab[g][:3]:
                    found.setdefault(f"{view}.{col}={val}", {"value": val, "table": view, "col": col, "span": g, "score": L})
    if len(found) < k:  # khớp lỏng: mọi từ của giá trị (≥ 2 từ) đều có trong câu
        for key, lst in model.vocab.items():
            ws = key.split()
            if len(ws) >= 2 and set(ws) <= qset:
                for val, view, col in lst[:2]:
                    found.setdefault(f"{view}.{col}={val}", {"value": val, "table": view, "col": col, "span": key,
                                                             "score": len(ws) - 0.5})
    return sorted(found.values(), key=lambda x: -x["score"])[:k]


def pick_tables(n: Normal, model: ShopModel, ents: list[dict], k: int = 5) -> list[str]:
    q = set(n.text.split())
    scored = []
    for name, v in model.views.items():
        bag = set(fold(" ".join([name, v.role, ROLE_DESC.get(v.role, ""), v.desc] +
                                [f"{c.name} {c.orig} {c.desc}" for c in v.cols])).split())
        s = len(q & bag) + (3 if v.role in ("ban_dong", "don_hang", "phieu_dich_vu") else 0) \
            + 4 * sum(1 for e in ents if e["table"] == name)
        scored.append((s, name))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:k]]


SPEC_SCHEMA = {"type": "object", "properties": {
    "nghi": {"type": "string"},
    "loai": {"type": "string", "enum": ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]},
    "don_vi": {"type": "string", "enum": ["tien", "dem", "so", "phan_tram", "ngay", "thang", "chu", "gio"]},
    "sql": {"type": "string"},
    "ly_do": {"type": "string"}}, "required": ["nghi", "loai", "don_vi", "sql", "ly_do"]}

SYSTEM = """Bạn trả lời câu hỏi số liệu của một cửa hàng bằng MỘT câu SQL (DuckDB) chạy trên các bảng đã làm sạch.
KHÔNG tự tính, KHÔNG bịa số: con số do SQL tính. Luật tính (rulebook) đi kèm là bắt buộc.
Quy tắc SQL:
- Chỉ SELECT/WITH. Tên bảng, tên cột chép đúng như danh mục (đặt trong "ngoặc kép" nếu cần).
- Kỳ thời gian đã được tính sẵn: dùng đúng ngày bắt đầu/kết thúc cho (ngay BETWEEN DATE 'a' AND DATE 'b').
- Giá trị chữ (tên hàng, khách, nhóm…) chép NGUYÊN VĂN từ danh sách thực thể đã khớp.
- Có cột trang_thai_chuan thì doanh thu/số đơn chỉ tính trang_thai_chuan NOT IN ('huy','tra_hang') (đơn trả một
  phần cũng là tra_hang: loại CẢ ĐƠN). Có ten_hang_chuan / khach_chuan / so_luong_co_ban thì dùng các cột đó.
  khach_chuan NULL là khách lẻ: không đếm, không xếp hạng.
- Bảng "ghi theo đơn" có nhiều dòng cho một đơn: số đơn = COUNT(DISTINCT ma_don), số đơn của khách =
  COUNT(DISTINCT ma_don) theo khach_chuan; giá trị một đơn = SUM(thanh_tien) theo ma_don. Không dùng COUNT(*) để đếm đơn.
- Nhiều điều kiện OR trên ngày phải đặt trong ngoặc. So sánh nhiều kỳ: GROUP BY kỳ (vd strftime(ngay, '%Y-%m')).
- "tháng nào" → loai date, don_vi thang (trả 'YYYY-MM'); "ngày nào" → loai date, don_vi ngay.
- Kết quả: number/percent/string/date/boolean → 1 dòng 1 cột; list → mỗi phần tử một dòng ở cột đầu (đúng thứ tự,
  nhiều số theo thứ tự câu hỏi thì 1 dòng nhiều cột cũng được); table → cột đầu là tên đối tượng, các cột sau là số đo
  theo thứ tự câu hỏi. Xếp hạng: ORDER BY số đo DESC rồi theo mã/tên tăng dần; LIMIT k.
- percent: trả số phần trăm (12.5 nghĩa là 12,5%), không làm tròn trong SQL. Tiền không làm tròn trong SQL.
- Câu hỏi cần thông tin mà KHÔNG bảng nào ghi (ví dụ đánh giá chủ quan như nhiệt tình, uy tín, ngon… khi rulebook
  không định nghĩa), hoặc kỳ hỏi nằm hoàn toàn ngoài dữ liệu: loai = unanswerable, sql = "", ly_do = vì sao.
Trả JSON: {"nghi": một câu viết lại câu hỏi cho rõ, "loai", "don_vi": tien|dem|so|phan_tram|ngay|thang|chu|gio,
"sql", "ly_do"}."""


def _periods(spec: TimeSpec) -> str:
    if not spec.periods:
        return "(câu không nêu kỳ cụ thể)"
    out = [f"kỳ {i + 1}: {p.label or p.kind} = {p.start} → {p.end}" for i, p in enumerate(spec.periods)]
    if spec.compare_to:
        out.append(f"kỳ so sánh: {spec.compare_to.start} → {spec.compare_to.end}")
    return "; ".join(out)


def build_prompt(q: str, today: date, n: Normal, spec: TimeSpec, ents: list[dict], tables: list[str],
                 model: ShopModel, rules: Rules, hint: str, error: str | None = None) -> list[dict]:
    cards = rules.retrieve(f"{q} {n.text}", 7, model.industry)
    rule_txt = "\n".join(f"- [{c.path}] {c.text[:380]}" for c in cards)
    ent_txt = "\n".join(f"- \"{e['value']}\" (bảng {e['table']}, cột {e['col']})" for e in ents) or "(không có)"
    rng = f"{model.date_range[0]} → {model.date_range[1]}" if model.date_range else "?"
    user = (f"Hôm nay: {today}. Dữ liệu bán có từ {rng}. Ngành: {model.industry or 'chung'}.\n"
            f"Câu hỏi: {q}\n(đã chuẩn hoá: {n.text})\n"
            f"Kỳ thời gian (đã tính): {_periods(spec)}\n"
            f"Thực thể khớp trong dữ liệu:\n{ent_txt}\n"
            f"Dạng đáp án gợi ý: {hint}\n\n"
            f"Luật liên quan:\n{rules.core[:1500]}\n{rule_txt}\n\n"
            f"Danh mục bảng:\n{catalog(model, tables)}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if error:
        msgs.append({"role": "user", "content": f"Lần trước lỗi: {error}\nSửa lại, trả JSON như yêu cầu."})
    return msgs
