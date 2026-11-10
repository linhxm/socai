"""Chuẩn hoá câu hỏi: teencode / viết tắt / không dấu → dạng fold chuẩn; nhận diện ngôn ngữ.

Mục tiêu không phải phục hồi dấu đầy đủ (việc đó để LLM), mà là đưa các cách viết hay gặp
("dthu", "bnhiu", "ko dc", "hnay", "tíh", "nhứt") về một khoá ổn định cho đường nhanh và cho
truy hồi ví dụ. Bảng từ chỉ dựng từ câu hỏi train.
"""

from __future__ import annotations

import re

from socai.io.cells import fold

TEEN: dict[str, str] = {
    "dthu": "doanh thu", "dt": "doanh thu", "doanhthu": "doanh thu", "ds": "doanh so",
    "bn": "bao nhieu", "bnhiu": "bao nhieu", "bnhieu": "bao nhieu", "bnh": "bao nhieu", "bao nhiu": "bao nhieu",
    "baonhieu": "bao nhieu", "bnhiêu": "bao nhieu", "nhiu": "nhieu", "nhieu": "nhieu",
    "ko": "khong", "k": "khong", "hok": "khong", "hong": "khong", "kh": "khong", "kg": "khong",
    "dc": "duoc", "dk": "duoc", "duoc": "duoc",
    "sp": "san pham", "sl": "so luong", "hd": "hoa don", "kh": "khach hang", "ncc": "nha cung cap",
    "hnay": "hom nay", "hqua": "hom qua", "bua nay": "hom nay", "bua qua": "hom qua",
    # KHÔNG thay "quan" (quán / bình quân) và "ch" (cửa hàng / chưa): mơ hồ, để LLM quyết
    "tiem": "cua hang", "shop": "cua hang",
    "nhut": "nhat", "nhat": "nhat", "kon": "con", "dag": "dang", "tih": "tinh", "mik": "minh",
    "kua": "cua", "ngui": "nguoi", "vs": "voi", "trc": "truoc", "ng": "nguoi",
    "tb": "trung binh", "bq": "binh quan", "bill": "don", "hoa don": "don",
}
FILLERS = {"z", "v", "vay", "ta", "nha", "nhe", "nghen", "a", "ah", "oi", "e", "em", "anh", "chi", "ơi",
           "giup", "cho", "toi", "minh", "hoi", "nhanh", "xem", "luon", "do", "ne", "hen", "ha"}

_RE_CJK = re.compile(r"[一-鿿]")
_RE_HANGUL = re.compile(r"[가-힯]")
_RE_KANA = re.compile(r"[぀-ヿ]")
_RE_VI_MARK = re.compile(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
_EN_WORDS = {"what", "how", "many", "much", "revenue", "orders", "sales", "last", "this", "month", "week",
             "which", "top", "best", "selling", "customer", "total", "average", "the", "in", "of"}


def detect_lang(q: str) -> str:
    """vi | vi_nodiacritics | en | zh | ja | ko | mixed."""
    if _RE_HANGUL.search(q):
        return "ko"
    if _RE_KANA.search(q):
        return "ja"
    if _RE_CJK.search(q):
        return "zh"
    words = set(re.findall(r"[a-z]+", q.lower()))
    en = len(words & _EN_WORDS)
    if _RE_VI_MARK.search(q):
        return "mixed" if en >= 3 else "vi"
    return "en" if en >= 2 and en >= len(words) / 4 else "vi_nodiacritics"


_TELEX_PAIRS = (("aw", "a"), ("aa", "a"), ("ee", "e"), ("oo", "o"), ("ow", "o"), ("uw", "u"), ("dd", "d"))


def untelex(tok: str) -> str:
    """Gỡ dấu Telex gõ sót: âm tiết tiếng Việt không kết thúc bằng s/f/r/x/j/w và không có aa/ee/oo...
    'nhoms'→'nhom', 'trowr'→'tro', 'leen'→'len', 'tawng'→'tang', 'luwowtj'→'luot', 'naof'→'nao'."""
    if not tok.isalpha() or len(tok) < 3:
        return tok
    t = tok
    if t[-1] in "sfrxj" and re.search(r"[aeiouy]", t[:-1]):
        t = t[:-1]
    for a, b in _TELEX_PAIRS:
        t = t.replace(a, b)
    if t.endswith("w"):
        t = t[:-1]
    return t


def canon(q: str) -> str:
    """Khoá fold đã thay teencode, gỡ Telex sót; giữ chữ số và '/', bỏ từ đệm.
    Câu chữ Hán/Hàn/Nhật: giữ nguyên ký tự (fold sẽ xoá chúng), chỉ chuẩn hoá khoảng trắng."""
    if detect_lang(q) in ("zh", "ja", "ko"):
        return " ".join(q.lower().split())
    s = fold(q.replace("/", " / ").replace("%", " phan tram "))
    fix = untelex if detect_lang(q) not in ("en", "zh", "ja", "ko") else (lambda t: t)
    s = " " + " ".join(fix(t) for t in s.split()) + " "
    for k in sorted(TEEN, key=len, reverse=True):
        s = re.sub(rf"(?<= ){re.escape(k)}(?= )", TEEN[k], s)
    toks = [t for t in s.split() if t not in FILLERS]
    return " ".join(toks).replace(" / ", "/")
