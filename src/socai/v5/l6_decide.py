"""L6: kiểm và quyết định TRẢ LỜI hay BỎ TRỐNG.

Điểm: đúng +1, sai −1, bỏ trống 0 → chỉ trả lời khi P(đúng) > ngưỡng. P(đúng) lấy theo "nhóm tin cậy" của câu
(mức đồng thuận giữa các mẫu SQL), mỗi nhóm một xác suất đo trên train (`calibration.json`, sinh bằng
`python -m socai.v5 calibrate`). Chưa đo thì dùng giá trị mặc định thận trọng dưới đây.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socai.io.cells import match_key
from socai.rules.timeexpr import TimeSpec

DEFAULT_P = {
    "gate_time": 0.9,  # kỳ hỏi nằm hẳn ngoài dữ liệu (code kiểm)
    "all": 0.85,  # ≥ 3 mẫu, tất cả cùng đáp án
    "most": 0.65,  # quá nửa số mẫu cùng đáp án
    "two": 0.7,  # 2 mẫu, cùng đáp án
    "single": 0.55,  # chỉ 1 mẫu (hết giờ hoặc chạy thử)
    "split": 0.3,  # không đáp án nào quá nửa
    "unans_llm": 0.45,  # LLM tự bảo "không trả lời được": dễ sai, mặc định bỏ trống cho tới khi đo
    "low_conf": 0.4,  # bảng dùng tới có nhiều ô không đọc được
}


@dataclass
class Policy:
    p: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_P))
    threshold: float = 0.5

    @classmethod
    def load(cls, path: Path | None) -> Policy:
        pol = cls()
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            pol.p.update(data.get("p", {}))
            pol.threshold = data.get("threshold", pol.threshold)
        return pol


@dataclass
class Decision:
    answer: dict  # đáp án nộp (có thể là abstain)
    bucket: str
    agree: float
    n: int
    candidate: dict | None  # đáp án tốt nhất trước ngưỡng (để hiệu chỉnh)
    p: float


def gate_time(spec: TimeSpec, date_range: tuple[str, str] | None, as_of=None) -> str | None:
    """Kỳ hỏi (hoặc kỳ so sánh) nằm hoàn toàn ngoài khoảng dữ liệu → không trả lời được.

    date_range = hợp mọi cột ngày của shop. Kỳ sau hôm nay (as_of) không bị chặn: thường là lỗi đọc ngày,
    để LLM quyết (sai ở đây mất 1 điểm, bỏ trống thì không)."""
    if not date_range:
        return None
    lo, hi = date_range
    for p in spec.periods + ([spec.compare_to] if spec.compare_to else []):
        if p.kind == "until" or (as_of and p.start > as_of):
            continue
        if p.end.isoformat() < lo or p.start.isoformat() > hi:
            return f"kỳ {p.start} → {p.end} nằm ngoài dữ liệu của shop ({lo} → {hi})"
    return None


def _norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return match_key(v)


def answer_key(ans: dict) -> str:
    """Hai đáp án cùng khoá khi luật chấm coi chúng là một (số làm tròn 2 chữ số, chuỗi chuẩn hoá, bảng so dòng như tập)."""
    t, a = ans["type"], ans.get("answer")
    if t == "list":
        a = [_norm(x) for x in a]
    elif t == "table":
        a = sorted(json.dumps([_norm(x) for x in r], ensure_ascii=False) for r in a["rows"])
    else:
        a = _norm(a)
    return json.dumps([t, a], ensure_ascii=False, sort_keys=True)


def decide(samples: list[dict], policy: Policy, confidence: float = 1.0) -> Decision:
    """samples: mỗi mẫu {'answer': dict} hoặc {'error': str}. Bỏ phiếu theo đáp án đã định dạng."""
    n = len(samples)
    ok = [s["answer"] for s in samples if s.get("answer")]
    if not ok:
        return Decision({"type": "abstain"}, "no_result", 0.0, n, None, 0.0)
    keys = [answer_key(a) for a in ok]
    best, cnt = Counter(keys).most_common(1)[0]
    cand = ok[keys.index(best)]  # mẫu sớm nhất của nhóm thắng (mẫu 0 là greedy)
    agree = cnt / n
    if cand["type"] == "unanswerable":
        bucket = "unans_llm" if agree > 0.5 else "split"
    elif n == 1:
        bucket = "single"
    elif n == 2:
        bucket = "two" if cnt == 2 else "split"
    elif cnt == n:
        bucket = "all"
    elif agree > 0.5:
        bucket = "most"
    else:
        bucket = "split"
    p = policy.p.get(bucket, 0.0)
    if confidence < 1.0:
        p = min(p, policy.p.get("low_conf", 0.4)) if bucket not in ("all",) else p * confidence
    final = cand if p > policy.threshold else {"type": "abstain"}
    return Decision(final, bucket, round(agree, 3), n, cand, round(p, 3))


def calibrate(log_path: Path, answers: Path, out: Path, min_n: int = 20) -> dict:
    """Đo tỉ lệ đúng của từng nhóm tin cậy trên một lần chạy TRAIN, ghi `calibration.json`."""
    from socai.eval.scorer import judge, load_jsonl

    refs = load_jsonl(answers)
    stat: dict[str, list[int]] = {}
    for ln in log_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        d, ref = rec.get("decision") or {}, refs.get(rec["id"])
        if not ref or not d.get("candidate"):
            continue
        s = stat.setdefault(d["bucket"], [0, 0])
        s[0] += judge(d["candidate"], ref) == "correct"
        s[1] += 1
    p = dict(DEFAULT_P)
    report = {}
    for b, (c, n) in sorted(stat.items()):
        report[b] = {"correct": c, "n": n, "precision": round(c / n, 4)}
        if n >= min_n:
            p[b] = round((c + 1) / (n + 2), 4)  # làm trơn Laplace
    out.write_text(json.dumps({"threshold": 0.5, "p": p, "measured": report}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return report
