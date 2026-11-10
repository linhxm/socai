"""Bộ chấm cục bộ mô phỏng luật chấm DataMate (đề bài + rulebook §9). Penalized Accuracy:
đúng +1, abstain/thiếu 0, sai −1; trả lời câu unanswerable bằng giá trị là sai.

Luật so khớp:
- number: khớp tuyệt đối (chỉ bỏ qua sai số dấu phẩy động); percent: khớp sau làm tròn 2 chữ số.
- string: NFC, chữ thường, ký tự không phải chữ/số/khoảng trắng → khoảng trắng, gộp khoảng trắng,
  GIỮ dấu; nhận đáp án chuẩn hoặc bất kỳ `alias`.
- date / boolean: khớp tuyệt đối.
- list: có thứ tự, so từng vị trí theo luật string/number.
- table: cột theo tên chuẩn hoá, không khớp thì theo vị trí; dòng so như tập hợp.
Đây là MÔ PHỎNG; số liệu chính thức chỉ lấy từ leaderboard của BTC.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from socai.io.cells import match_key

EPS = 1e-6


def _num_eq(a: Any, b: Any, pct: bool = False) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    if pct:
        return round(a + 0.0, 2) == round(b + 0.0, 2) or abs(a - b) < EPS
    return abs(a - b) < EPS


def _str_eq(a: Any, ref: Any, aliases: list[str]) -> bool:
    if not isinstance(a, str):
        return False
    k = match_key(a)
    return any(k == match_key(x) for x in [ref, *aliases] if isinstance(x, str))


def _elem_eq(a: Any, b: Any, aliases: list[str]) -> bool:
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return _num_eq(a, b)
    return _str_eq(a, b, aliases)


def _table_eq(pred: Any, ref: dict, aliases: list[str]) -> bool:
    if not isinstance(pred, dict) or not isinstance(pred.get("columns"), list) \
            or not isinstance(pred.get("rows"), list):
        return False
    rc, pc = ref["columns"], pred["columns"]
    if len(rc) != len(pc) or any(not isinstance(r, list) or len(r) != len(pc) for r in pred["rows"]):
        return False
    rk, pk = [match_key(c) for c in rc], [match_key(c) for c in pc]
    order = [pk.index(k) for k in rk] if sorted(rk) == sorted(pk) and len(set(pk)) == len(pk) \
        else list(range(len(rc)))
    rows = [[r[i] for i in order] for r in pred["rows"]]
    if len(rows) != len(ref["rows"]):
        return False
    left = list(ref["rows"])
    for r in rows:
        hit = next((j for j, g in enumerate(left)
                    if all(_elem_eq(a, b, aliases) for a, b in zip(r, g))), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def judge(pred: dict | None, ref: dict) -> str:
    """'correct' | 'wrong' | 'abstain'."""
    if pred is None or pred.get("type") == "abstain":
        return "abstain"
    pt, rt = pred.get("type"), ref.get("type")
    if pt != rt:
        return "wrong"
    if rt == "unanswerable":
        return "correct"
    a, b, al = pred.get("answer"), ref.get("answer"), ref.get("alias") or []
    ok = False
    if rt == "number":
        ok = _num_eq(a, b)
    elif rt == "percent":
        ok = _num_eq(a, b, pct=True)
    elif rt == "string":
        ok = _str_eq(a, b, al)
    elif rt in ("date", "boolean"):
        ok = a == b and type(a) is type(b)
    elif rt == "list":
        ok = isinstance(a, list) and len(a) == len(b) and all(_elem_eq(x, y, al) for x, y in zip(a, b))
    elif rt == "table":
        ok = _table_eq(a, b, al)
    return "correct" if ok else "wrong"


def score(preds: dict[str, dict], refs: dict[str, dict]) -> dict:
    verdicts = {qid: judge(preds.get(qid), ref) for qid, ref in refs.items()}
    c = Counter(verdicts.values())
    n = len(refs) or 1
    return {"PA": (c["correct"] - c["wrong"]) / n, "n": len(refs), "correct": c["correct"],
            "wrong": c["wrong"], "abstain": c["abstain"],
            "precision_when_answering": c["correct"] / max(c["correct"] + c["wrong"], 1),
            "verdicts": verdicts}


def load_jsonl(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}
