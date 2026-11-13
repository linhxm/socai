"""L1: đọc file, làm sạch hình thức (v4.md §2). Code làm phần chắc chắn; LLM chỉ được hỏi về dòng KHÔNG CHẮC.

Đầu ra mỗi bảng là `CleanTable`: chỉ còn dòng dữ liệu (đã bỏ dòng tổng, trống, ghi chú, header lặp), giữ giá trị gốc,
kèm kiểu đoán từng cột, mô tả README, và biên bản.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from socai.llm.client import LLM, extract_json
from socai.v4 import rows as R
from socai.v4.tools_l1 import L1Session, _cut, column_profile
from socai.v4.cells import is_blank

KEEP = ("du_lieu", "tiep_noi", "dong_phu_don")


@dataclass
class CleanTable:
    id: str
    files: list[str]
    df: pd.DataFrame  # dòng dữ liệu, giá trị gốc + cột phụ _src/_row/_sheet/_y/_m
    guess: dict[str, str]  # cột → kind đoán
    readme: dict[str, str]  # cột → mô tả README
    profile: list[dict]
    row_counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [c for c in self.df.columns if not str(c).startswith("_")]


LABEL_SCHEMA = {"type": "object", "properties": {"labels": {"type": "array", "items": {
    "type": "object", "properties": {"i": {"type": "integer"}, "kind": {"type": "string", "enum": list(R.ROW_KINDS)}},
    "required": ["i", "kind"]}}}, "required": ["labels"]}

LABEL_PROMPT = """Bảng "{tid}" của một sổ bán hàng. Các dòng dưới đây code CHƯA CHẮC thuộc loại nào.
Loại: du_lieu (dòng ghi giao dịch/bản ghi thật) · tong (dòng cộng tổng ngày/tháng/khối) · ghi_chu_cuoi (dòng chú thích)
· header_lap (dòng lặp tên cột) · trong · dong_phu_don (dòng phụ của đơn: voucher, phí ship, giảm giá).
Dòng dữ liệu thường có đủ ô như các dòng bên cạnh; dòng tổng thường chỉ có nhãn và vài số.
Cột: {cols}
Dòng mẫu bình thường: {normal}
Dòng cần gán:
{rows}
Trả JSON {{"labels": [{{"i": số dòng, "kind": loại}}]}}."""


def _label_unsure(s: L1Session, tid: str, llm: LLM, trace: list) -> None:
    kd = s.kinds[tid]
    un = kd.index[kd["unsure"]][:12]
    if not len(un):
        return
    t = s.tables[tid]
    normal = kd.index[kd["kind"] == "du_lieu"][:1]
    show = lambda i: {c: _cut(t.df.at[i, c], 30) for c in t.columns if not is_blank(t.df.at[i, c])}  # noqa: E731
    prompt = LABEL_PROMPT.format(tid=tid, cols=t.columns[:30], normal=[show(i) for i in normal],
                                 rows="\n".join(f"- i={i}: {show(i)}" for i in un))
    c = llm.chat([{"role": "user", "content": prompt}], max_tokens=600, json_schema=LABEL_SCHEMA)
    out = extract_json(c.content) or {}
    trace.append({"layer": "L1", "task": "label_unsure", "table": tid, "reasoning": c.reasoning,
                  "output": out})
    for lab in out.get("labels", []):
        i, k = lab.get("i"), lab.get("kind")
        if i in kd.index and k in R.ROW_KINDS:
            kd.loc[i, ["kind", "why", "unsure"]] = [k, "LLM", False]


def run_l1(shop_dir: Path, llm: LLM | None = None, trace: list | None = None) -> tuple[L1Session, list[CleanTable]]:
    trace = trace if trace is not None else []
    s = L1Session(shop_dir)
    out = []
    for tid, t in s.tables.items():
        kd = s.kinds[tid]
        if llm is not None and kd["unsure"].any():
            try:
                _label_unsure(s, tid, llm, trace)
            except Exception as e:  # noqa: BLE001 — LLM lỗi thì giữ nhãn của code
                trace.append({"layer": "L1", "table": tid, "error": str(e)[:200]})
        keep = kd["kind"].isin(KEEP)
        df = t.df[keep].copy()
        if df.empty:
            continue
        ct_tmp = type(t)(t.id, t.grids, df, t.header_row)
        prof = [column_profile(ct_tmp, c, k=3) for c in t.columns]
        notes = []
        for k, n in kd["kind"].value_counts().items():
            if k not in KEEP:
                notes.append(f"bỏ {n} dòng {k}")
        if kd["unsure"].any():
            notes.append(f"{int(kd['unsure'].sum())} dòng vẫn không chắc (giữ theo nhãn code)")
        out.append(CleanTable(tid, sorted({s.grids[g].file for g in t.grids}), df,
                              {p["col"]: p["guess"] for p in prof}, s.readme_for(t), prof,
                              {k: int(v) for k, v in kd["kind"].value_counts().items()}, notes))
    return s, out
