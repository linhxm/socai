"""data_manifest.json: đường dẫn, kích thước, SHA-256, số dòng của mọi file dữ liệu (mục 8 hướng dẫn BTC).

python training_code/data_generation/make_manifest.py <thư mục> [<thư mục> ...] > data_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def entry(p: Path) -> dict:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    e = {"path": p.as_posix(), "bytes": p.stat().st_size, "sha256": h.hexdigest()}
    if p.suffix == ".jsonl":
        e["records"] = sum(1 for ln in p.open(encoding="utf-8") if ln.strip())
    return e


def main(dirs: list[str]) -> None:
    files = sorted(f for d in dirs for f in Path(d).rglob("*") if f.is_file())
    print(json.dumps({"files": [entry(f) for f in files], "count": len(files)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
