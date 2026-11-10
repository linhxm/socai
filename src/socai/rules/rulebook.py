"""Rulebook của split: tách thành thẻ luật (RuleCard) có mã mục, và so khác biệt giữa hai split.

Rulebook ĐỔI theo split (train vs public đã khác: public thêm ngành Mẹ và bé, Sửa xe, Điện thoại,
thêm "Khách sộp nhất", "Giờ cao điểm"… và bỏ vài luật train). Hệ thống không được mã hoá cứng luật
train: toàn văn rulebook luôn nằm trong tiền tố prompt (CAG — dùng chung cho mọi câu của split,
vLLM prefix caching), còn thẻ luật được truy hồi để nhấn mạnh luật liên quan tới câu hỏi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from socai.io.cells import fold


@dataclass(frozen=True)
class RuleCard:
    section: str  # "1. Đơn hàng và doanh thu" / "6. Theo ngành › Nhà thuốc"
    key: str  # cột đầu của bảng luật
    text: str  # cả dòng luật

    @property
    def rid(self) -> str:
        return f"{fold(self.section)[:24]}::{fold(self.key)[:40]}"


@dataclass
class Rulebook:
    text: str
    cards: list[RuleCard]

    @classmethod
    def load(cls, path: Path) -> Rulebook:
        return cls.parse(path.read_text(encoding="utf-8"))

    @classmethod
    def parse(cls, text: str) -> Rulebook:
        cards: list[RuleCard] = []
        h2 = h3 = ""
        for ln in text.splitlines():
            if ln.startswith("## "):
                h2, h3 = ln[3:].strip(), ""
            elif ln.startswith("### "):
                h3 = ln[4:].strip()
            elif ln.startswith("|") and not re.match(r"^\|\s*[-:|\s]+\|?\s*$", ln):
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cells) >= 2 and fold(cells[0]) not in ("luat", "cach noi", "type", "cach ghi", "tu"):
                    sec = f"{h2} › {h3}" if h3 else h2
                    cards.append(RuleCard(sec, re.sub(r"\*\*", "", cells[0]), " | ".join(cells)))
        return cls(text, cards)

    def relevant(self, question: str, k: int = 8) -> list[RuleCard]:
        """Truy hồi thẻ luật theo trùng từ (BM25 rút gọn, không cần mô hình)."""
        q = set(fold(question).split())
        scored = []
        for c in self.cards:
            words = set(fold(c.text).split())
            s = len(q & words) / (1 + len(words) ** 0.5)
            if s > 0:
                scored.append((s, c))
        return [c for _, c in sorted(scored, key=lambda x: -x[0])[:k]]

    def industries(self) -> list[str]:
        return sorted({c.section.split("›")[-1].strip() for c in self.cards if "›" in c.section})


def diff(a: Rulebook, b: Rulebook) -> dict[str, list[str]]:
    """Luật chỉ có ở a, chỉ có ở b, và luật cùng khoá nhưng khác nội dung."""
    ka = {c.rid: c for c in a.cards}
    kb = {c.rid: c for c in b.cards}
    return {
        "only_a": [ka[k].text for k in ka.keys() - kb.keys()],
        "only_b": [kb[k].text for k in kb.keys() - ka.keys()],
        "changed": [f"{ka[k].text}  ⟶  {kb[k].text}" for k in ka.keys() & kb.keys() if ka[k].text != kb[k].text],
    }
