"""Khởi động llama-server (OpenAI-compatible) cho một hồ sơ mô hình trong `configs/models.json`.

Dùng chung cho laptop, Kaggle và image: `VllmClient` nói chuyện với llama.cpp hay vLLM như nhau.
    python -m socai.llm.serve --model qwen3-14b --bin ./llama-server --models-dir ./models --port 8088 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "models.json"


def profile(name: str, config: Path = CONFIG) -> dict:
    data = json.loads(Path(config).read_text(encoding="utf-8"))
    if name not in data:
        raise KeyError(f"không có hồ sơ '{name}'; có: {[k for k in data if not k.startswith('_')]}")
    return data[name]


def command(p: dict, binary: str, gguf: Path, port: int, ctx: int | None = None) -> list[str]:
    return [binary, "-m", str(gguf), "-ngl", "99", "-fa", "on", "-c", str(ctx or p["ctx"]), *p.get("server_args", []),
            "--jinja", "--host", "127.0.0.1", "--port", str(port), "-np", "1", "--seed", "0"]


def wait_ready(base_url: str, timeout_s: float = 900, proc: subprocess.Popen | None = None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"llama-server thoát với mã {proc.returncode}")
        try:
            with urllib.request.urlopen(base_url.rsplit("/v1", 1)[0] + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 — server chưa lên
            pass
        time.sleep(3)
    raise TimeoutError(f"server {base_url} chưa sẵn sàng sau {timeout_s:.0f}s")


def start(name: str, binary: str, gguf: Path, port: int, gpu: int | None = None, log: Path | None = None,
          ctx: int | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = open(log, "w") if log else subprocess.DEVNULL  # noqa: SIM115 — giữ mở suốt đời tiến trình
    return subprocess.Popen(command(profile(name), binary, gguf, port, ctx), stdout=out, stderr=subprocess.STDOUT,
                            env=env)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SOCAI_MODEL", "qwen3-4b"))
    ap.add_argument("--bin", default="llama-server")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=None)
    a = ap.parse_args()
    p = profile(a.model)
    proc = start(a.model, a.bin, a.models_dir / p["gguf"], a.port, a.gpu, ctx=a.ctx)
    wait_ready(f"http://127.0.0.1:{a.port}/v1", proc=proc)
    print(f"{a.model} sẵn sàng ở http://127.0.0.1:{a.port}/v1", flush=True)
    proc.wait()
