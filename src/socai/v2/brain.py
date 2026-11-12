"""Não v2: chọn bảng liên quan → LLM ghép biểu thức → code chạy → lỗi thì đưa lỗi + gợi ý cho LLM sửa một lần.

LLM chỉ chọn bảng, cột, phép và chép nguyên văn tên/cụm thời gian; con số luôn do `algebra.execute` tính.
Không có luật theo loại câu hỏi ở đây: luật nằm ở định nghĩa bảng (`views.py`) và cách định dạng (`algebra.py`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from socai.exec.metrics import Result, Unsupported
from socai.io.cells import fold
from socai.llm.client import LLM, extract_json
from socai.nlq.normalize import untelex
from socai.v2.algebra import Ctx, V2Error, execute, signature
from socai.v2.bank import Bank
from socai.v2.views import View

SYSTEM = """Bạn viết BIỂU THỨC JSON để code tính câu trả lời trên dữ liệu cửa hàng. KHÔNG tự tính, KHÔNG giải thích.
Câu hỏi có thể viết tắt, không dấu, teencode, trộn tiếng Anh: hiểu nghĩa rồi chọn bảng và cột.

Truy vấn: {"from": bảng, "where": [[cột, phép, giá trị]], "time": ..., "group": cột, "agg": [phép_gộp, cột],
 "having": [phép, số], "sort": "desc"|"asc", "limit": k, "out": ...}
- phép lọc: = != in not in > >= < <= ~ (chứa chữ) $ (kết thúc bằng) * (mọi giá trị). Giá trị chữ CHÉP NGUYÊN VĂN từ câu hỏi.
  "quá/trên/hơn N" là >, "từ N trở lên" là >=, "dưới N" là <.
- time: "$ky" kỳ chính của câu · "$ky_so" kỳ được so sánh · "$tung_ky" từng kỳ (ra danh sách) ·
  {"tu": "YYYY-MM-DD", "so_ngay": N} N ngày kể từ ngày đó · {"truoc": "YYYY-MM-DD", "so_ngay": N} N ngày ngay trước ·
  "$den_het" từ đầu dữ liệu đến hết kỳ chính (số còn lại, tính đến hết ngày) · "$truoc_ky" trước kỳ chính ·
  một cụm thời gian chép từ câu (vd "tháng 2/2026") · null không lọc thời gian.
- agg: sum | count | count_distinct | mean | max | min | unique (giá trị duy nhất, vd giá bán)
  | ["wmean", cột giá trị, cột trọng số] (bình quân theo số lượng).
- group: cột, hoặc "thang"/"nam". Không group thì out = "value".
- out: value (một số) · name (đối tượng đứng đầu) · names (danh sách đối tượng) · values (danh sách số theo thứ tự xếp)
  · table (bảng tên + số) · count (số nhóm thoả having) · key (ngày/tháng/giờ đứng đầu).
Phép ghép: {"op": pct_change|diff|pct|ratio|trend_up|trend_down, "a": ..., "b": ...} hoặc {"op": "list", "items": [...]}.
- pct_change = % tăng giảm a so với b · diff = a − b · pct = a chiếm bao nhiêu % của b · ratio = a / b.
- b viết gọn {"like": "a", <chỗ khác>}; "drop": [cột] bỏ điều kiện lọc của cột đó.
- {"op": "except", "a": ..., "b": ..., "out": "count"|"names"}: đối tượng có ở a mà không có ở b (a, b có group, out names).
Bảng ban/don đã bỏ đơn huỷ, đơn trả; hỏi về đơn huỷ/trả thì lọc trang_thai.
Câu hỏi cần loại thông tin mà KHÔNG bảng nào của shop ghi (vd tay nghề nhân viên, địa chỉ khách) thì trả
{"op": "unanswerable", "nghi": lý do}. Có dữ liệu nhưng không biết cách ghép thì trả {"op": "none"}."""

EXAMPLES: list[tuple[str, dict]] = [
    ("cho toi 3 mon ban chay nhat thang nay",
     {"nghi": "xếp mặt hàng theo số lượng", "from": "ban", "time": "$ky", "group": "mat_hang",
      "agg": ["sum", "so_luong"], "limit": 3, "out": "names"}),
    ("Tháng 5/2025 nhóm đồ uống chiếm bao nhiêu % doanh thu?",
     {"nghi": "tỉ trọng: doanh thu nhóm / doanh thu cả shop", "op": "pct",
      "a": {"from": "ban", "where": [["nhom_hang", "=", "đồ uống"]], "time": "$ky", "agg": ["sum", "thanh_tien"],
            "out": "value"}, "b": {"like": "a", "drop": ["nhom_hang"]}}),
    ("thang nay so voi thang truoc so don tang bao nhieu %",
     {"nghi": "% thay đổi số đơn kỳ này so với kỳ trước", "op": "pct_change",
      "a": {"from": "don", "time": "$ky", "agg": ["count", "ma_don"], "out": "value"},
      "b": {"like": "a", "time": "$ky_so"}}),
    ("tỉ lệ đơn huỷ tháng 3",
     {"nghi": "đơn huỷ / mọi đơn", "op": "pct",
      "a": {"from": "don", "where": [["trang_thai", "=", "huỷ"]], "time": "$ky", "agg": ["count", "ma_don"],
            "out": "value"}, "b": {"like": "a", "where": [["trang_thai", "*"]]}}),
    ("ca chiều hôm qua bán được bao nhiêu tiền",
     {"nghi": "doanh thu, lọc giờ ca chiều", "from": "ban", "where": [["gio", ">=", 14], ["gio", "<=", 21]],
      "time": "$ky", "agg": ["sum", "thanh_tien"], "out": "value"}),
    ("tháng 6 có mấy khách mua từ 3 đơn trở lên",
     {"nghi": "đếm khách có số đơn >= 3", "from": "don", "time": "$ky", "group": "khach", "agg": ["count", "ma_don"],
      "having": [">=", 3], "out": "count"}),
    ("doanh thu tiền mặt và chuyển khoản tháng 4, mỗi bên bao nhiêu",
     {"nghi": "hai con số theo thứ tự câu hỏi", "op": "list",
      "items": [{"from": "ban", "where": [["thanh_toan", "=", "tiền mặt"]], "time": "$ky",
                 "agg": ["sum", "thanh_tien"], "out": "value"},
                {"like": "a", "where": [["thanh_toan", "=", "chuyển khoản"]]}]}),
    ("giá bán 1 hộp Sữa Vinamilk ngày 5/3 là bao nhiêu",
     {"nghi": "giá một đơn vị cơ bản ghi trên dòng bán", "from": "ban", "where": [["mat_hang", "=", "Sữa Vinamilk"]],
      "time": "$ky", "agg": ["unique", "don_gia_co_ban"], "out": "value"}),
    ("tháng 7 bán được bao nhiêu kg Gạo ST25",
     {"nghi": "số lượng quy về đơn vị cơ bản kg", "from": "ban",
      "where": [["mat_hang", "=", "Gạo ST25"], ["don_vi_co_ban", "=", "kg"]], "time": "$ky",
      "agg": ["sum", "so_luong"], "out": "value"}),
    ("tháng 4 quán bán được bao nhiêu ly size L",
     {"nghi": "tên món kết thúc bằng (L)", "from": "ban", "where": [["mat_hang", "$", "(L)"]], "time": "$ky",
      "agg": ["sum", "so_luong"], "out": "value"}),
    ("Ngày 05/02/2026 Latte (L) tăng giá lên 53.500đ. So số ly bán 7 ngày kể từ ngày đó với 7 ngày trước đó thay đổi mấy %",
     {"nghi": "% thay đổi số lượng sau so với trước ngày tăng giá", "op": "pct_change",
      "a": {"from": "ban", "where": [["mat_hang", "=", "Latte (L)"]], "time": {"tu": "2026-02-05", "so_ngay": 7},
            "agg": ["sum", "so_luong"], "out": "value"},
      "b": {"like": "a", "time": {"truoc": "2026-02-05", "so_ngay": 7}}}),
    ("doanh thu từng tháng trong quý 2 có tăng liên tục không",
     {"nghi": "chuỗi doanh thu từng tháng, xét tăng", "op": "trend_up",
      "a": {"from": "ban", "time": "$tung_ky", "agg": ["sum", "thanh_tien"], "out": "value"}}),
    ("đơn trên 500k tháng này có bao nhiêu đơn",
     {"nghi": "đếm đơn có doanh thu > 500000", "from": "don", "where": [["doanh_thu_don", ">", "500k"]],
      "time": "$ky", "agg": ["count", "ma_don"], "out": "value"}),
    ("doanh thu thuốc không kê đơn tháng 5",
     {"nghi": "doanh thu, lọc cột kê đơn = không", "from": "ban", "where": [["ke_don", "=", "không"]],
      "time": "$ky", "agg": ["sum", "thanh_tien"], "out": "value"}),
    ("giá nhập bình quân 1 thùng Bia Hà Nội tháng 3",
     {"nghi": "bình quân giá nhập theo số lượng nhập, bảng nhập hàng của shop",
      "from": "so_nhap_hang", "where": [["mat_hang", "=", "Bia Hà Nội"]], "time": "$ky",
      "agg": ["wmean", "gia_nhap", "so_luong"], "out": "value"}),
    ("tính đến hết ngày 20/9 khách còn nợ tổng cộng bao nhiêu",
     {"nghi": "tổng còn nợ tại mốc", "from": "cong_no", "time": "$ky", "agg": ["sum", "con_no"], "out": "value"}),
    ("đến hôm nay 3 khách nào nợ nhiều nhất, mỗi người bao nhiêu",
     {"nghi": "xếp khách theo còn nợ, kèm số", "from": "cong_no", "time": "$ky", "group": "khach",
      "agg": ["sum", "con_no"], "limit": 3, "out": "table"}),
    ("từ 20/10/2025 đến hết 26/10/2025 có mấy mặt hàng ế, không bán được cái nào",
     {"nghi": "hàng từng bán trước kỳ nhưng trong kỳ không bán", "op": "except", "out": "count",
      "a": {"from": "ban", "time": "$truoc_ky", "group": "mat_hang", "agg": ["sum", "so_luong"], "out": "names"},
      "b": {"like": "a", "time": "$ky"}}),
    ("tháng 5 doanh thu ca chiều hơn hay kém ca sáng bao nhiêu %",
     {"nghi": "% chênh ca chiều so với ca sáng, cùng kỳ", "op": "pct_change",
      "a": {"from": "ban", "where": [["gio", ">=", 14], ["gio", "<=", 21]], "time": "$ky",
            "agg": ["sum", "thanh_tien"], "out": "value"},
      "b": {"like": "a", "where": [["gio", ">=", 6], ["gio", "<=", 13]]}}),
    ("trong ba tháng 2/2025, 3/2025, 4/2025 tháng nào doanh thu cao nhất",
     {"nghi": "tháng có doanh thu cao nhất", "from": "ban", "time": "$ky", "group": "thang",
      "agg": ["sum", "thanh_tien"], "out": "key"}),
    ("doanh thu 3 tháng 2/2026, 3/2026, 4/2026 lần lượt là bao nhiêu",
     {"nghi": "doanh thu từng kỳ theo thứ tự", "from": "ban", "time": "$tung_ky", "agg": ["sum", "thanh_tien"],
      "out": "value"}),
    ("trung bình mỗi đơn tuần trước mang về bao nhiêu doanh thu",
     {"nghi": "trung bình doanh thu một đơn", "from": "don", "time": "$ky", "agg": ["mean", "doanh_thu_don"],
      "out": "value"}),
    ("tháng 7/2025 khách Võ Thảo Khánh ghi nợ thêm bao nhiêu và trả nợ được bao nhiêu",
     {"nghi": "hai số: nợ thêm trong kỳ, trả nợ trong kỳ của khách", "op": "list",
      "items": [{"from": "no_them", "where": [["khach", "=", "Võ Thảo Khánh"]], "time": "$ky",
                 "agg": ["sum", "so_tien"], "out": "value"},
                {"from": "tra_no", "where": [["khach", "=", "Võ Thảo Khánh"]], "time": "$ky",
                 "agg": ["sum", "so_tien"], "out": "value"}]}),
    ("tính đến hết ngày 22/04/2026 khách còn nợ bao nhiêu ở nhóm tuổi nợ 0–7 ngày, 8–15 ngày và trên 15 ngày",
     {"nghi": "ba số còn nợ theo khoảng tuổi nợ", "op": "list",
      "items": [{"from": "cong_no", "where": [["tuoi_no", "<=", 7]], "time": "$ky", "agg": ["sum", "con_no"], "out": "value"},
                {"like": "a", "where": [["tuoi_no", ">=", 8], ["tuoi_no", "<=", 15]]},
                {"like": "a", "where": [["tuoi_no", ">", 15]]}]}),
    ("trừ hết chi phí thì tháng 1/2025 lãi ròng bao nhiêu",
     {"nghi": "lãi gộp trừ tổng chi phí trong sổ chi phí", "op": "diff",
      "a": {"from": "ban", "time": "$ky", "agg": ["sum", "lai_gop"], "out": "value"},
      "b": {"from": "so_chi_phi", "time": "$ky", "agg": ["sum", "so_tien"], "out": "value"}}),
    ("nhân viên nào có tay nghề cao nhất tiệm", {"nghi": "sổ không ghi tay nghề nhân viên", "op": "unanswerable"}),
]

# §2: "quá/trên/hơn N" không gồm N, "từ N trở lên / ít nhất N" gồm N; LLM nhỏ hay nhầm dấu → soát theo câu
_GE = __import__("re").compile(r"\b(tro len|tro di|it nhat|toi thieu|tu \S+ (dong |don |cai |lan )?tro len)\b")
_GT = __import__("re").compile(r"\b(qua|hon|tren|vuot)\b")
# câu có hỏi CON SỐ kèm theo đối tượng không (§9: kiểu đáp án theo câu hỏi) → table chỉ khi có
_ASK_NUM = __import__("re").compile(r"\b(bao nhieu|bn|bnhiu|may|so tien|kem|moi (nguoi|mon|khach|mat hang|loai|ben))\b")
_UNTIL = __import__("re").compile(r"\b(den het|tinh den|tu dau|con lai|con no|con ton)\b")
_MEAN = __import__("re").compile(r"\b(trung binh|tb)\b")
_SPAN = __import__("re").compile(r"\b\d+ ngay (ke tu|ngay truoc|lien truoc|truoc do|truoc ngay|sau)|ke tu ngay|ngay truoc do\b")
_BEFORE = __import__("re").compile(r"\b(e|khong ban|ko ban|k ban|chua ban|truoc do|truoc ky|tung ban)\b")
_ONE = __import__("re").compile(r"\bnhat\b")
_MANY = __import__("re").compile(r"\b(top|liet ke|nhung|cac|(?<![\d/])\d{1,2} (mat hang|mon|khach|san pham|hang))\b")


def lint(expr: object, question: str, notes: list[str]) -> object:
    f = " ".join(untelex(w) for w in fold(question).split())
    ge, gt = bool(_GE.search(f)), bool(_GT.search(f))

    def fix(d: dict) -> dict:
        d = dict(d)
        t = d.get("time")
        if isinstance(t, dict) and not _SPAN.search(f):
            d["time"] = "$ky"
            notes.append("khoảng ngày tự đặt nhưng câu không nói 'N ngày kể từ/trước' → $ky")
        if t == "$truoc_ky" and not _BEFORE.search(f):
            d["time"] = "$ky"
            notes.append("$truoc_ky nhưng câu không hỏi hàng ế/trước kỳ → $ky")
        agg = d.get("agg")
        if isinstance(agg, list) and len(agg) > 1 and agg[0] in ("count", "count_distinct") and \
                str(agg[1]).startswith("don_vi"):  # đếm cột đơn vị vô nghĩa: hỏi "mấy cây/bao nhiêu kg" là số lượng
            d["agg"] = ["sum", "so_luong"]
            notes.append("đếm cột đơn vị → sum so_luong")
        a, b = d.get("a"), d.get("b")
        if d.get("op") == "pct" and isinstance(a, dict) and isinstance(b, dict):
            aw = {c[0] for c in a.get("where") or [] if isinstance(c, list) and c}
            bw = b.get("where")
            if isinstance(bw, list) and any(isinstance(c, list) and c and c[0] == "trang_thai" for c in bw) \
                    and "trang_thai" not in aw:  # §1: mẫu số tỉ trọng là doanh thu cả shop, cùng luật đơn hợp lệ
                d["b"] = {**b, "where": [c for c in bw if not (isinstance(c, list) and c and c[0] == "trang_thai")]}
                notes.append("tỉ trọng: bỏ lọc trạng thái chỉ có ở mẫu số")
        if d.get("out") == "names" and not isinstance(d.get("limit"), int) and _ONE.search(f) \
                and not _MANY.search(f):  # "hàng nào bán chạy nhất" → một tên
            d["out"] = "name"
            notes.append("câu hỏi một đối tượng: names → name")
        if d.get("out") == "names" and d.get("limit") == 1 and not _MANY.search(f):
            d["out"] = "name"
        if d.get("out") == "table" and not _ASK_NUM.search(f):
            d["out"] = "names" if isinstance(d.get("limit"), int) and d["limit"] > 1 else "name"
            notes.append(f"câu không hỏi số kèm theo: table → {d['out']}")
        if d.get("time") == "$den_het" and d.get("from") != "cong_no" and not _UNTIL.search(f):
            d["time"] = "$ky"
            notes.append("$den_het ở bảng phát sinh, câu không hỏi số còn lại → $ky")
        if d.get("from") == "don" and _MEAN.search(f) and (d.get("agg") or [None])[0] == "sum" and not d.get("group"):
            d["agg"] = ["mean", *d["agg"][1:]]
            notes.append("câu hỏi trung bình một đơn: sum → mean")
        return d

    def walk(e: object) -> object:
        if isinstance(e, dict):
            return {k: walk(v) for k, v in fix(e).items()}
        if isinstance(e, list):
            if len(e) in (2, 3) and e and isinstance(e[0] if len(e) == 2 else e[1], str):
                op_i = 0 if len(e) == 2 else 1
                op = e[op_i]
                new = ">=" if (op == ">" and ge and not gt) else (">" if op == ">=" and gt and not ge else op)
                if new != op:
                    notes.append(f"soát dấu theo câu hỏi: {op} → {new}")
                    e = list(e)
                    e[op_i] = new
                    return e
            return [walk(x) for x in e]
        return e

    return walk(expr)


_STOP = {"bao", "nhieu", "thang", "ngay", "nay", "hom", "cua", "hang", "cho", "toi", "minh", "la", "co", "khong", "duoc",
         "nhat", "tong", "cac", "tinh", "den", "het", "trong", "voi", "so", "va", "mot", "nhung", "bang", "danh", "sach"}


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", fold(s)) if len(w) >= 2 and w not in _STOP}


def pick_views(views: dict[str, View], question: str, k: int = 2) -> list[View]:
    """Luôn có ban/don (nếu có); thêm ≤ k bảng khác có nhiều từ chung với câu hỏi (tên bảng, tên cột, mô tả)."""
    core = [views[n] for n in ("ban", "don") if n in views]
    q = _tokens(question)
    scored = []
    for v in views.values():
        if v.name in ("ban", "don"):
            continue
        text = v.name.replace("_", " ") + " " + v.desc + " " + " ".join(f"{c.name} {c.desc}" for c in v.cols.values())
        hit = q & _tokens(text)
        if hit:
            scored.append((len(hit), v.name, v))
    return core + [v for _, _, v in sorted(scored, key=lambda x: (-x[0], x[1]))[:k]]


@dataclass
class Thought:
    expr: dict | None = None
    result: Result | None = None
    attempts: list[dict] = field(default_factory=list)
    reasoning: list[str | None] = field(default_factory=list)
    problem: str | None = None
    shots: list[str] = field(default_factory=list)

    @property
    def sig(self) -> str:
        return signature(self.expr)


def messages(views: list[View], question: str, shots: list | None = None) -> list[dict]:
    """Luật + bảng của shop đứng trước (cố định theo shop → llama.cpp/vLLM dùng lại bộ đệm tiền tố);
    ví dụ đứng sau: truy hồi từ ngân hàng nếu có, không thì dùng danh sách cố định. Giữ prompt < ~3K token
    (GPU 4 GB tràn ở ~3.800 token)."""
    pairs = [(s.q, s.expr) for s in reversed(shots)] if shots else EXAMPLES  # giống nhất đứng sát câu hỏi
    text = "\n".join(f"{q} => {json.dumps(a, ensure_ascii=False, separators=(',', ':'))}" for q, a in pairs)
    q = _tokens(question)

    def want(c) -> bool:  # cột nối từ danh mục: chỉ khi câu hỏi nhắc tới
        return bool(q & _tokens(c.name.replace("_", " ").split(".")[-1] + " " + c.desc.split("(nối từ")[0]))

    cards = "\n\n".join(v.card(want=want) for v in views)
    note = " (lấy từ shop khác: tên bảng, cột, mặt hàng phải đổi theo CÁC BẢNG CỦA SHOP NÀY)" if shots else ""
    return [{"role": "system", "content": f"{SYSTEM}\n\nCÁC BẢNG CỦA SHOP NÀY:\n{cards}\n\n"
                                          f"VÍ DỤ{note} (câu => biểu thức):\n{text}"},
            {"role": "user", "content": question}]


_STATIC: Bank | None = None


def _static_bank() -> Bank:
    global _STATIC
    if _STATIC is None:
        _STATIC = Bank()
    return _STATIC


def think(llm: LLM, views: dict[str, View], question: str, ctx: Ctx, max_repairs: int = 1,
          max_tokens: int = 320, bank: Bank | None = None, exclude_biz: str | None = None, k: int = 8,
          skip: int = 0) -> Thought:
    """`skip` > 0: bỏ qua `skip` ví dụ giống nhất (dùng để lấy ý kiến thứ hai từ bộ ví dụ khác)."""
    th = Thought()
    if bank is None:  # không có ngân hàng: vẫn truy hồi, nhưng chỉ trong ví dụ viết tay (giữ prompt trong ngữ cảnh)
        bank = _static_bank()
    if not bank.seeded:  # ví dụ viết tay luôn có mặt để truy hồi (ngân hàng mới còn mỏng)
        for q0, e0 in EXAMPLES:
            bank.add(q0, e0, "_static", "static", reindex=False)
        bank.reindex()
        bank.seeded = True
    shots = bank.nearest(question, k + skip, exclude_biz)[skip:] if bank is not None else None
    th.shots = [s.id for s in shots] if shots else []
    msgs = messages(pick_views(views, question), question, shots)
    for _ in range(1 + max_repairs):
        call = llm.chat(msgs, max_tokens=max_tokens, thinking=False)
        th.reasoning.append(call.reasoning)
        expr = extract_json(call.content)
        th.attempts.append({"raw": call.content[:600], "expr": expr})
        if expr is None:
            err = "Không đọc được JSON. Chỉ trả về MỘT object JSON."
        else:
            ctx.notes.clear()
            expr = lint(expr, question, ctx.notes)
            th.expr = expr
            try:
                th.result = execute(expr, ctx)
                return th
            except V2Error as e:
                err = str(e)
            except Unsupported as e:
                th.problem = str(e)
                return th
        th.attempts[-1]["error"] = err
        msgs = msgs + [{"role": "assistant", "content": call.content},
                       {"role": "user", "content": f"Lỗi khi chạy: {err}\nSửa lại biểu thức. Chỉ trả JSON."}]
    th.problem = th.attempts[-1].get("error")
    th.result = None
    return th
