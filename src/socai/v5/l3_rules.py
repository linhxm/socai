"""L3: luật từ `rulebook.md` của split ĐANG CHẠY. Cắt theo cấu trúc (mỗi dòng bảng, mỗi gạch đầu dòng, mỗi đoạn
là một thẻ, giữ đường dẫn mục), truy hồi bằng BM25 trên chữ bỏ dấu. Thẻ thuộc mục ngành chỉ dùng cho shop
đúng ngành đó. Không viết cứng luật nào trong code: rulebook đổi thì prompt đổi theo.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socai.io.cells import fold

STOP = set("la cua va cac nhung co khong thi ma cho voi trong ngoai tren duoi mot hai bao nhieu nao gi the "
           "duoc bi da se dang o tai tu den hay hoac neu khi de nay do ay minh shop a ah nhe nha".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9%]+", fold(text)) if w not in STOP and (len(w) > 1 or w.isdigit())]


@dataclass
class Card:
    path: str  # "6. Theo ngành › Nhà thuốc"
    text: str
    industry: str | None = None
    toks: Counter = field(default_factory=Counter)


@dataclass
class Rules:
    cards: list[Card]
    industries: list[str]
    idf: dict[str, float] = field(default_factory=dict)
    avg_len: float = 1.0

    def search(self, query: str, industry: str | None, k: int = 12) -> list[Card]:
        """Thẻ liên quan nhất tới câu hỏi. Thẻ của ngành khác bị loại; thẻ đúng ngành của shop được cộng điểm."""
        q = Counter(tokens(query))
        scored = []
        for i, c in enumerate(self.cards):
            if c.industry and industry and c.industry != industry:  # không rõ ngành: giữ cả, để BM25 chọn
                continue
            n = sum(c.toks.values()) or 1
            s = sum(self.idf.get(w, 0.0) * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * n / self.avg_len))
                    for w in q if (f := c.toks.get(w, 0)))
            if s > 0:
                scored.append((s * (1.3 if c.industry and industry else 1.0), -i, c))
        return [c for _, _, c in sorted(scored, key=lambda x: (-x[0], -x[1]))[:k]]


def parse_rules(text: str) -> Rules:
    cards: list[Card] = []
    h2 = h3 = ""
    header: list[str] = []
    para: list[str] = []

    def add(body: str) -> None:
        body = re.sub(r"\*\*|`", "", body).strip()
        if len(body) < 8:
            return
        ind = h3 if h3 and "nganh" in fold(h2) else None
        path = f"{h2} › {h3}" if h3 else h2
        cards.append(Card(path, body, ind))

    def flush() -> None:
        if para:
            add(" ".join(para))
            para.clear()

    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            title = s.lstrip("#").strip()
            if level <= 2:
                h2, h3 = title, ""
            else:
                h3 = title
            header = []
        elif s.startswith("|"):
            flush()
            if re.fullmatch(r"\|[\s:|-]+\|?", s):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not header:
                header = cells  # dòng đầu của bảng là tiêu đề cột
                continue
            add(" — ".join(c for c in cells if c))
        elif re.match(r"^([-*+]|\d+[.)])\s+", s):
            flush()
            header = []
            add(re.sub(r"^([-*+]|\d+[.)])\s+", "", s))
        elif not s:
            flush()
            header = []
        else:
            para.append(s)
    flush()
    for c in cards:
        c.toks = Counter(tokens(f"{c.path} {c.text}"))
    n = len(cards) or 1
    df = Counter(w for c in cards for w in c.toks)
    idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    industries = sorted({c.industry for c in cards if c.industry})
    return Rules(cards, industries, idf, sum(sum(c.toks.values()) for c in cards) / n)


def load_rules(path: Path) -> Rules:
    return parse_rules(path.read_text(encoding="utf-8")) if path.exists() else Rules([], [])
