"""L3: luật từ rulebook của split đang chạy (v4.md §4). Code cắt theo cấu trúc markdown (mỗi dòng bảng / gạch đầu
dòng là một thẻ, giữ đường dẫn mục); chỉ mục BM25 trên chữ bỏ dấu (từ đơn + cặp từ). 14B (nếu có) làm giàu thẻ bằng
các cách nói khác (viết tắt, không dấu, teencode) để L4 truy hồi được câu hỏi viết lạ. Cache theo băm rulebook.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from socai.llm.client import LLM, extract_json
from socai.v4.cells import fold

ALWAYS = ("1.", "9.")  # mục luôn đưa vào prompt ở dạng rút gọn: doanh thu/đơn hợp lệ, định dạng đáp án


@dataclass
class Card:
    id: str
    path: str  # "6 > Nhà thuốc > Hàng hết hạn trong N ngày tới"
    section: str  # "6"
    industry: str  # "" = luật chung
    text: str
    alt: list[str] = field(default_factory=list)  # cách nói khác do LLM thêm


def chunk_rulebook(md: str) -> list[Card]:
    cards: list[Card] = []
    h2 = h3 = ""
    para: list[str] = []

    def flush_para():
        if para:
            txt = " ".join(para).strip()
            if len(txt) > 20:
                cards.append(Card(f"p{len(cards)}", " > ".join(x for x in (h2, h3) if x), _sec(h2), _ind(h2, h3), txt))
            para.clear()

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            flush_para()
            h2, h3 = s[3:].strip(), ""
        elif s.startswith("### "):
            flush_para()
            h3 = s[4:].strip()
        elif s.startswith("|"):
            flush_para()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not cells or set("".join(cells)) <= set("-: ") or fold(cells[0]) in ("luat", "cach noi", "tu", "type",
                                                                                      "cach ghi"):
                continue
            head = cells[0]
            cards.append(Card(f"c{len(cards)}", " > ".join(x for x in (h2, h3, head) if x), _sec(h2), _ind(h2, h3),
                              f"{head}: " + " | ".join(cells[1:])))
        elif s.startswith(("- ", "* ")):
            flush_para()
            cards.append(Card(f"b{len(cards)}", " > ".join(x for x in (h2, h3) if x), _sec(h2), _ind(h2, h3), s[2:]))
        elif s and not s.startswith("#"):
            para.append(s)
        else:
            flush_para()
    flush_para()
    return cards


def _sec(h2: str) -> str:
    m = re.match(r"(\d+)\.", h2)
    return m.group(1) if m else ""


def _ind(h2: str, h3: str) -> str:
    return h3 if h3 and re.search(r"theo ngành", h2.lower()) else ""


def industries(cards: list[Card]) -> list[str]:
    return sorted({c.industry for c in cards if c.industry})


def _toks(s: str) -> list[str]:
    w = fold(s).split()
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


class Index:
    def __init__(self, cards: list[Card], k1: float = 1.4, b: float = 0.7):
        self.cards = cards
        self.docs = [Counter(_toks(f"{c.path} {c.text} {' '.join(c.alt)}")) for c in cards]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(cards)
        self.idf = {t: math.log(1 + (n - x + 0.5) / (x + 0.5)) for t, x in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 6, industry: str = "") -> list[Card]:
        q = set(_toks(query))
        scored = []
        for i, (d, ln) in enumerate(zip(self.docs, self.len)):
            c = self.cards[i]
            if c.industry and c.industry != industry:
                continue
            s = sum(self.idf.get(t, 0) * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    for t in q if t in d)
            if c.industry and c.industry == industry:
                s *= 1.3  # luật ngành ghi đè luật chung (§6)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        return [self.cards[i] for _, i in scored[:k]]


ENRICH_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "alt": {"type": "array", "items": {"type": "string"}}}, "required": ["id", "alt"]}}},
    "required": ["items"]}

ENRICH_PROMPT = """Dưới đây là các thẻ luật của một rulebook tính số liệu cửa hàng. Với MỖI thẻ, viết 3–6 cách người
bán hàng có thể hỏi về khái niệm đó: không dấu, viết tắt (dthu, sl, đh, ncc…), teencode, phương ngữ. Ngắn, chỉ cụm từ.
{cards}
Trả JSON {{"items": [{{"id", "alt": [...]}}]}}."""


def enrich(cards: list[Card], llm: LLM, batch: int = 10) -> None:
    by = {c.id: c for c in cards}
    for i in range(0, len(cards), batch):
        part = cards[i:i + batch]
        txt = "\n".join(f"- id={c.id}: {c.path}: {c.text[:200]}" for c in part)
        try:
            out = extract_json(llm.chat([{"role": "user", "content": ENRICH_PROMPT.format(cards=txt)}],
                                        max_tokens=1500, json_schema=ENRICH_SCHEMA).content) or {}
        except Exception:  # noqa: BLE001
            continue
        for it in out.get("items", []):
            if it.get("id") in by:
                by[it["id"]].alt = [str(a)[:60] for a in it.get("alt", [])][:8]


@dataclass
class Rules:
    cards: list[Card]
    index: Index
    industries: list[str]
    core: str  # tóm tắt luật luôn đưa vào prompt

    def retrieve(self, query: str, k: int = 6, industry: str = "") -> list[Card]:
        return self.index.search(query, k, industry)


def load_rules(rulebook: Path, cache_dir: Path | None = None, llm: LLM | None = None) -> Rules:
    md = rulebook.read_text(encoding="utf-8")
    h = hashlib.sha1(md.encode()).hexdigest()[:12]
    cf = cache_dir / f"rules_{h}{'_llm' if llm else ''}.json" if cache_dir else None
    if cf and cf.exists():
        cards = [Card(**x) for x in json.loads(cf.read_text(encoding="utf-8"))]
    else:
        cards = chunk_rulebook(md)
        if llm is not None:
            enrich(cards, llm)
        if cf:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps([asdict(c) for c in cards], ensure_ascii=False), encoding="utf-8")
    core = "\n".join(f"- {c.text}" for c in cards if c.section in ("1",) and len(c.text) < 400)
    return Rules(cards, Index(cards), industries(cards), core)
