"""Bộ đọc tài liệu mở rộng (Track B): PDF, ảnh chụp/scan (OCR), DOCX, XML hoá đơn điện tử, JSON.

Không bộ nào trong đây cần cho bài thi (dữ liệu thi chỉ có CSV/XLSX/SQLite), nên mọi phụ thuộc
nặng là tuỳ chọn: thiếu thư viện thì bộ đọc báo `NotAvailable` và file bị bỏ qua có ghi log,
không làm hỏng pipeline. Đầu ra luôn là `RawTable` + `confidence` để tầng trả lời biết nguồn nào
là trích xuất (có thể sai) và hạ độ tin cậy của đáp án tương ứng.

Phần thuần Python, test được không cần mô hình: `boxes_to_table` dựng bảng từ các hộp chữ OCR
(gom dòng theo toạ độ y, tách cột theo khoảng trống x).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from socai.io.readers import RawTable, register_reader


class NotAvailable(RuntimeError):
    """Thiếu thư viện / mô hình cho loại tài liệu này."""


# ---------------------------------------------------------------- OCR
@dataclass
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


class OcrEngine(Protocol):
    name: str

    def read(self, image_path: Path) -> list[Box]: ...


_ENGINES: list[OcrEngine] = []


def register_ocr(engine: OcrEngine) -> None:
    _ENGINES.append(engine)


def boxes_to_table(boxes: Sequence[Box], row_tol: float = 0.6, col_gap: float = 1.5
                   ) -> tuple[list[list[str]], float]:
    """Gom hộp chữ thành lưới: cùng dòng nếu tâm y lệch < row_tol × chiều cao chữ; ranh giới cột là
    các khoảng trống x xuất hiện ở đa số dòng. Trả (lưới, độ tin cậy trung bình)."""
    if not boxes:
        return [], 0.0
    h = sorted(b.y1 - b.y0 for b in boxes)[len(boxes) // 2] or 1.0
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.yc):
        if rows and abs(rows[-1][-1].yc - b.yc) < row_tol * h:
            rows[-1].append(b)
        else:
            rows.append([b])
    # 1) trong một dòng, gộp các từ sát nhau (khoảng trống < khoảng cách giữa từ) thành một ô
    merged: list[list[Box]] = []
    for r in rows:
        cells_r: list[Box] = []
        for b in sorted(r, key=lambda b: b.x0):
            if cells_r and b.x0 - cells_r[-1].x1 < 0.5 * col_gap * h:
                p = cells_r[-1]
                cells_r[-1] = Box(f"{p.text} {b.text}", p.x0, min(p.y0, b.y0), b.x1, max(p.y1, b.y1),
                                  min(p.conf, b.conf))
            else:
                cells_r.append(b)
        merged.append(cells_r)
    # 2) ranh giới cột: điểm bắt đầu các ô gom cụm theo khoảng cách > col_gap × chiều cao chữ
    starts = sorted(b.x0 for r in merged for b in r)
    edges = [starts[0]]
    for x in starts[1:]:
        if x - edges[-1] > col_gap * h:
            edges.append(x)
    grid = []
    for r in merged:
        cells = [""] * len(edges)
        for b in r:
            j = max(i for i, e in enumerate(edges) if e <= b.x0 + 0.5 * h)
            cells[j] = f"{cells[j]} {b.text}".strip()
        grid.append(cells)
    conf = sum(b.conf for b in boxes) / len(boxes)
    return grid, conf


def read_image(path: Path, known: set[str]) -> list[RawTable]:
    if not _ENGINES:
        raise NotAvailable("chưa đăng ký OCR engine (PaddleOCR / docTR / VLM ≤ ngân sách tham số)")
    boxes = _ENGINES[0].read(path)
    grid, conf = boxes_to_table(boxes)
    if not grid:
        return []
    from socai.io.readers import _frame

    df = _frame([tuple(r) for r in grid], known)
    return [RawTable(path.name, path.name, df, source_kind="ocr", confidence=conf,
                     notes=[f"ocr:{_ENGINES[0].name}"])]


# ---------------------------------------------------------------- PDF (có lớp chữ)
def read_pdf(path: Path, known: set[str]) -> list[RawTable]:
    try:
        import pdfplumber
    except ImportError as e:
        raise NotAvailable("cần pdfplumber cho PDF có lớp chữ; PDF scan → render ảnh rồi OCR") from e
    from socai.io.readers import _frame

    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            for j, tb in enumerate(page.extract_tables() or [], 1):
                df = _frame([tuple(r) for r in tb], known)
                if not df.empty:
                    out.append(RawTable(f"{path.name}::p{i}t{j}", path.name, df, source_kind="pdf",
                                        confidence=0.95))
    return out


# ---------------------------------------------------------------- DOCX
def read_docx(path: Path, known: set[str]) -> list[RawTable]:
    try:
        import docx
    except ImportError as e:
        raise NotAvailable("cần python-docx") from e
    from socai.io.readers import _frame

    d = docx.Document(str(path))
    out = []
    for j, tb in enumerate(d.tables, 1):
        rows = [tuple(c.text for c in r.cells) for r in tb.rows]
        out.append(RawTable(f"{path.name}::t{j}", path.name, _frame(rows, known), source_kind="docx"))
    return out


# ---------------------------------------------------------------- XML (hoá đơn điện tử, xuất phần mềm)
def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_xml(path: Path, known: set[str]) -> list[RawTable]:
    """Làm phẳng XML: phần tử lặp lại nhiều nhất (≥ 2 lần, có con lá) là dòng; phần tử lá ngoài vòng
    lặp là thông tin đầu phiếu, lặp lại trên mọi dòng. Đủ cho hoá đơn điện tử dạng
    <HDon><DLHDon><TTChung>…</TTChung><NDHDon><DSHHDVu><HHDVu>…</HHDVu>…
    [Cần kiểm chứng: đối chiếu tên thẻ với XSD hoá đơn điện tử hiện hành trước khi dùng thật]."""
    root = ET.parse(path).getroot()
    counts: dict[str, int] = {}
    for el in root.iter():
        if len(el) and all(len(c) == 0 for c in el):
            counts[_strip_ns(el.tag)] = counts.get(_strip_ns(el.tag), 0) + 1
    repeated = [t for t, n in counts.items() if n >= 2]
    if not repeated:
        rec = {_strip_ns(e.tag): e.text for e in root.iter() if len(e) == 0}
        return [RawTable(path.name, path.name, pd.DataFrame([rec], dtype=object), source_kind="xml")]
    row_tag = max(repeated, key=lambda t: counts[t])
    header = {_strip_ns(e.tag): e.text for e in root.iter()
              if len(e) == 0 and _strip_ns(e.tag) not in {_strip_ns(c.tag) for r in root.iter()
                                                          if _strip_ns(r.tag) == row_tag for c in r}}
    rows = [{**header, **{_strip_ns(c.tag): c.text for c in el}}
            for el in root.iter() if _strip_ns(el.tag) == row_tag]
    return [RawTable(path.name, path.name, pd.DataFrame(rows, dtype=object), source_kind="xml")]


def read_json(path: Path, known: set[str]) -> list[RawTable]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lists = {k: v for k, v in data.items() if isinstance(v, list) and v and isinstance(v[0], dict)}
        return [RawTable(f"{path.name}::{k}", path.name, pd.json_normalize(v).astype(object), source_kind="json")
                for k, v in lists.items()]
    return [RawTable(path.name, path.name, pd.json_normalize(data).astype(object), source_kind="json")]


def _safe(fn):
    def wrapped(path: Path, known: set[str]) -> list[RawTable]:
        try:
            return fn(path, known)
        except NotAvailable as e:
            return [RawTable(path.name, path.name, pd.DataFrame(), source_kind="skipped", notes=[str(e)])]
    return wrapped


for _ext, _fn in {".pdf": read_pdf, ".docx": read_docx, ".xml": read_xml, ".json": read_json,
                  ".png": read_image, ".jpg": read_image, ".jpeg": read_image, ".webp": read_image,
                  ".tif": read_image, ".tiff": read_image}.items():
    register_reader(_ext, _safe(_fn))
