"""Đăng ký tool cho LLM: tên, mô tả, schema tham số (JSON Schema kiểu OpenAI function calling).

Mỗi tool là hàm `fn(session, **args) -> dict`. Lỗi tham số không làm hỏng vòng: trả {"ok": False, "error", "hint"}
để LLM tự sửa lời gọi (v4-ham §0).
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

MAX_CHARS = 6000  # ~1.500 token: tool trả quá dài thì cắt, kèm cờ


@dataclass
class Tool:
    name: str
    desc: str
    params: dict  # {tên: {"type":..., "description":..., "enum"?:...}}
    required: list[str]
    fn: Callable[..., dict]
    layer: str

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.desc,
            "parameters": {"type": "object", "properties": self.params, "required": self.required}}}


TOOLS: dict[str, Tool] = {}


def tool(layer: str, desc: str, **params: dict):
    """@tool("L1", "mô tả", src={"type": "string", ...}, ...). Tham số không có giá trị mặc định là bắt buộc."""
    def deco(fn: Callable[..., dict]) -> Callable[..., dict]:
        sig = inspect.signature(fn)
        req = [n for n, p in list(sig.parameters.items())[1:] if p.default is inspect.Parameter.empty]
        TOOLS[fn.__name__] = Tool(fn.__name__, desc, params, req, fn, layer)
        return fn
    return deco


def schemas(layer: str | None = None) -> list[dict]:
    return [t.schema() for t in TOOLS.values() if layer is None or t.layer == layer]


def call(session: Any, name: str, args: dict | str) -> dict:
    """Gọi tool theo tên; mọi lỗi thành kết quả có `hint`, không ném ra ngoài vòng agent."""
    t = TOOLS.get(name)
    if t is None:
        return {"ok": False, "error": f"không có tool {name}", "hint": f"tool có: {sorted(TOOLS)}"}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"tham số không phải JSON: {e}"}
    miss = [r for r in t.required if r not in args]
    extra = [a for a in args if a not in t.params]
    if miss or extra:
        return {"ok": False, "error": f"thiếu {miss}" if miss else f"tham số lạ {extra}",
                "hint": f"tham số của {name}: {list(t.params)}; bắt buộc: {t.required}"}
    try:
        out = t.fn(session, **args)
    except (KeyError, ValueError, IndexError, TypeError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "hint": hint_for(session, e)}
    out = {"ok": True, **out} if "ok" not in out else out
    text = json.dumps(out, ensure_ascii=False, default=str)
    if len(text) > MAX_CHARS:
        out = {"ok": out.get("ok", True), "truncated": True, "text": text[:MAX_CHARS]}
    return out


def hint_for(session: Any, e: Exception) -> str:
    """Gợi ý sửa lời gọi: tên bảng/cột có thật gần nhất."""
    msg = str(e).strip("'\"")
    names = []
    if hasattr(session, "tables"):
        names += list(session.tables)
        for t in session.tables.values():
            names += t.columns
    from difflib import get_close_matches
    near = get_close_matches(msg, names, n=3, cutoff=0.5)
    return f"có phải: {near}" if near else "xem list_sources / profile_table để lấy tên đúng"
