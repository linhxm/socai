"""Client LLM chạy OFFLINE: server vLLM (OpenAI-compatible) khởi động trong container bởi predict.sh.

Rule 3: không gọi API thương mại, không mạng. Client chỉ nói chuyện với 127.0.0.1.
Tái lập (Rule 8, |ΔPA| ≤ 0.03): temperature 0, seed cố định, thứ tự câu hỏi cố định, mặc định
không gửi song song (batching trên GPU có thể làm lệch đầu ra kể cả khi greedy — hướng dẫn Docker
của BTC nói rõ). Log bắt buộc ghi `thinking` NGUYÊN VĂN mỗi lần gọi → `Call.reasoning`.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Call:
    content: str
    reasoning: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLM(Protocol):
    model: str

    def chat(self, messages: list[dict], *, max_tokens: int = 2048, json_schema: dict | None = None,
             thinking: bool = False) -> Call: ...


@dataclass
class VllmClient:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local"
    seed: int = 0
    timeout: float = 300.0
    api_key: str | None = None  # chỉ dùng khi phát triển với server có khoá; bài thi chạy offline không cần

    def chat(self, messages: list[dict], *, max_tokens: int = 2048, json_schema: dict | None = None,
             thinking: bool = False) -> Call:
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": 0.0, "top_p": 1.0, "seed": self.seed,
                      # Qwen3: bật/tắt chế độ suy nghĩ qua chat template
                      "chat_template_kwargs": {"enable_thinking": thinking}}
        if json_schema is not None:
            # Chuẩn OpenAI `response_format` — llama.cpp server và vLLM đều nhận
            # [Cần kiểm chứng với đúng bản vLLM 0.10.1.1 trong image BTC]
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "plan", "schema": json_schema}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        msg = data["choices"][0]["message"]
        u = data.get("usage") or {}
        return Call(msg.get("content") or "", msg.get("reasoning_content"), u.get("prompt_tokens", 0),
                    u.get("completion_tokens", 0), time.perf_counter() - t0)

    def ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                return r.status == 200
        except OSError:
            return False


@dataclass
class FakeLLM:
    """LLM giả cho test/CI: `responder(messages) -> str`."""

    responder: Callable[[list[dict]], str] | None = None
    model: str = "fake"
    calls: list[list[dict]] = field(default_factory=list)

    def chat(self, messages: list[dict], *, max_tokens: int = 2048, json_schema: dict | None = None,
             thinking: bool = False) -> Call:
        self.calls.append(messages)
        return Call(self.responder(messages) if self.responder else "", None)


def extract_json(text: str) -> dict | None:
    """Lấy object JSON đầu tiên trong văn bản (khi không dùng guided decoding)."""
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[s:i + 1])
                except json.JSONDecodeError:
                    break
        s = text.find("{", s + 1)
    return None


def extract_code(text: str) -> str | None:
    if "```" not in text:
        return None
    parts = text.split("```")
    for p in parts[1::2]:
        p = p.split("\n", 1)[1] if p.startswith(("python", "py")) else p
        if "def answer" in p or "result" in p:
            return p.strip()
    return None
