"""Đánh giá trên split có đáp án (train) + hiệu chỉnh xác suất đúng cho chính sách trả lời.

Hiệu chỉnh theo LEAVE-BUSINESS-OUT: P(đúng | nguồn, metric) cho shop X chỉ ước lượng từ các shop khác,
vì private test có shop và cả NGÀNH chưa thấy. Báo cáo cả PA "nếu áp chính sách đó" để không tự lừa
mình bằng con số đo trên chính dữ liệu đã dùng để chỉnh ngưỡng.
Không bao giờ chạy hàm này trên public/private test để huấn luyện/chỉnh (Rule 6) — chúng không có đáp án.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from socai.eval.scorer import judge, load_jsonl


def _key(log: dict) -> str | None:
    d = log.get("decision") or {}
    cands = d.get("candidates") or []
    if not cands:
        return None
    src = d.get("source") or cands[0]["source"]
    code = log.get("code")
    metric = code.split("::")[-1] if isinstance(code, str) and "::" in code else (log["answer"].get("type") or "other")
    return f"{src}:{metric}"


def evaluate(answers: Path, logs: Path, min_support: int = 20, shrink: float = 10.0) -> dict:
    refs = load_jsonl(answers)
    recs = [json.loads(x) for x in logs.read_text(encoding="utf-8").splitlines() if x.strip()]
    # đáp án ứng viên tốt nhất (kể cả khi chính sách đã bỏ trống) để đo độ chính xác của từng nguồn
    rows = []
    for r in recs:
        if r["id"] not in refs:
            continue
        d = r.get("decision") or {}
        cand = next((c["answer"] for c in d.get("candidates", []) if c["source"] == d.get("source")), None)
        rows.append({"id": r["id"], "biz": r.get("business_id"), "key": _key(r),
                     "final": judge(r["answer"] if r["answer"]["type"] != "abstain" else None, refs[r["id"]]),
                     "cand": judge(cand, refs[r["id"]]) if cand else "abstain",
                     "type": refs[r["id"]]["type"]})
    n = len(refs)
    fin = Counter(x["final"] for x in rows)
    report = {"n": n, "PA_policy": (fin["correct"] - fin["wrong"]) / max(n, 1), "final": dict(fin)}

    # độ chính xác theo nguồn:metric, tổng thể và leave-business-out
    by_key: dict[str, Counter] = defaultdict(Counter)
    by_key_biz: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for x in rows:
        if x["key"] and x["cand"] != "abstain":
            by_key[x["key"]][x["cand"]] += 1
            by_key_biz[(x["key"], x["biz"])][x["cand"]] += 1
    base = sum(c["correct"] for c in by_key.values()) / max(sum(sum(c.values()) for c in by_key.values()), 1)

    def prec(c: Counter, prior: float) -> float:  # co về trung bình chung khi ít mẫu
        return (c["correct"] + shrink * prior) / (c["correct"] + c["wrong"] + shrink)

    priors = {k: round(prec(c, base), 4) for k, c in by_key.items() if sum(c.values()) >= min_support}
    # PA mô phỏng nếu áp ngưỡng trên xác suất LBO
    th = 0.6
    lbo = Counter()
    for x in rows:
        if not x["key"] or x["cand"] == "abstain":
            continue
        rest = by_key[x["key"]].copy()
        rest.subtract(by_key_biz[(x["key"], x["biz"])])
        p = prec(rest, base)
        lbo[x["cand"] if p > th else "abstain"] += 1
    report["PA_lbo_simulated"] = (lbo["correct"] - lbo["wrong"]) / max(n, 1)
    report["lbo"] = dict(lbo)
    report["per_key"] = {k: dict(c) for k, c in sorted(by_key.items(), key=lambda kv: -sum(kv[1].values()))}
    report["per_type"] = {t: dict(Counter(x["final"] for x in rows if x["type"] == t))
                          for t in sorted({x["type"] for x in rows})}
    report["calibration"] = {"threshold": th, "priors": priors, "base": round(base, 4)}
    return report
