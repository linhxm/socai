"""Chạy code do LLM sinh trong một tiến trình worker riêng, có timeout.

Worker nạp dữ liệu của MỘT shop một lần (từ cache) rồi nhận nhiều đoạn code; quá thời gian thì bị
giết và khởi động lại. Container chạy `--network none`, /data chỉ đọc (Rule 5), nên rủi ro chính là
treo/ngốn RAM, không phải rò rỉ. Mỗi lần chạy trả nguyên văn code + output để ghi log (bắt buộc).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class ExecResult:
    ok: bool
    value: object = None
    error: str = ""
    code: str = ""


def _worker(conn, biz_dir: str, cache_dir: str | None, calendar: str | None) -> None:
    from socai.exec.api import SANDBOX_GLOBALS, Ctx
    from socai.profile.business import load_business
    from socai.rules.calendar import LunarCalendar
    from socai.semantic.canonical import build

    biz = load_business(Path(biz_dir), Path(cache_dir) if cache_dir else None)
    can = build(biz)
    cal = LunarCalendar.load(Path(calendar)) if calendar else None
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        code, today = msg
        try:
            g = dict(SANDBOX_GLOBALS)
            exec(compile(code, "<generated>", "exec"), g)  # noqa: S102 — có chủ đích, trong sandbox
            out = g["answer"](Ctx(biz, can, date.fromisoformat(today), cal))
            conn.send(("ok", out))
        except Exception:  # noqa: BLE001
            conn.send(("err", traceback.format_exc(limit=3)[-1500:]))


class CodeWorker:
    def __init__(self, biz_dir: Path, cache_dir: Path | None, calendar: Path | None,
                 load_timeout: float = 300) -> None:
        self.args = (str(biz_dir), str(cache_dir) if cache_dir else None, str(calendar) if calendar else None)
        self.load_timeout = load_timeout
        self._start()

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, *self.args), daemon=True)
        self.proc.start()
        if not self.conn.poll(self.load_timeout):
            raise TimeoutError("worker không nạp xong dữ liệu")
        self.conn.recv()

    def run(self, code: str, today: date, timeout: float = 60) -> ExecResult:
        self.conn.send((code, today.isoformat()))
        if not self.conn.poll(timeout):
            self.proc.kill()
            self._start()
            return ExecResult(False, error=f"timeout {timeout}s", code=code)
        kind, val = self.conn.recv()
        return ExecResult(kind == "ok", val if kind == "ok" else None, "" if kind == "ok" else val, code)

    def close(self) -> None:
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.kill()
