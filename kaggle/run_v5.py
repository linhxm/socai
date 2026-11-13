"""Kaggle (GPU T4 x2): chạy SOCAI v5 với Qwen3-14B, mỗi GPU một server, v5 tự chia câu hỏi cho hai server.

Biến môi trường:
- V5_SHOPS: "holdout" (mặc định, 6 shop trong configs/holdout.json), "dev" (các shop train còn lại), hoặc danh sách shop.
- V5_LIMIT: số câu tối đa mỗi shop (mặc định 20; "0" = tất cả).
- V5_ARGS: tham số thêm cho `python -m socai.v5 predict` (vd "--samples 6", "--thinking").
- V5_BUILD=1: sau khi chạy, dựng bank.json + calibration.json từ log (chỉ nên bật khi V5_SHOPS=dev).
Chỉ chạy trên TRAIN (có đáp án). Kết quả ở /kaggle/working/v5_out: submission.jsonl, logs/, eval.txt,
và bank.json + calibration.json khi V5_BUILD=1 (tải về, đặt vào src/socai/v5/ trước khi build Docker).
Quy trình: (1) V5_SHOPS=dev V5_BUILD=1 → (2) V5_SHOPS=holdout trong cùng phiên để đo có ngân hàng ví dụ.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_14b import W, build_llama, download, log  # noqa: E402

MODEL = "qwen3-14b"
PORTS = (8088, 8089)


def find_train() -> Path:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        with zipfile.ZipFile(z) as f:
            f.extractall(W / "inputs")
    for root in ("/kaggle/input", str(W / "inputs")):
        for p in glob.glob(f"{root}/**/answers.jsonl", recursive=True):
            d = Path(p).parent
            if (d / "questions.jsonl").exists() and (d / "data").exists():
                return d
    raise SystemExit("không thấy dữ liệu train (questions.jsonl + answers.jsonl + data/) trong Input")


def main() -> None:
    code = Path(__file__).resolve().parents[1]
    data = find_train()
    log(f"mã: {code} · dữ liệu: {data}")
    sys.path.insert(0, str(code / "src"))
    from socai.llm.serve import profile, start, wait_ready
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb", "python-calamine"], check=True)
    prof = profile(MODEL, code / "configs" / "models.json")
    got: dict = {}
    th = threading.Thread(target=lambda: got.setdefault("gguf", download(prof)))
    th.start()
    binary = build_llama()
    th.join()
    servers = [start(MODEL, str(binary), got["gguf"], port, gpu=g, log=W / f"server{g}.log")
               for g, port in enumerate(PORTS)]
    for port, p in zip(PORTS, servers):
        wait_ready(f"http://127.0.0.1:{port}/v1", timeout_s=900, proc=p)

    holdout = json.loads((code / "configs" / "holdout.json").read_text(encoding="utf-8"))["holdout"]
    which = os.environ.get("V5_SHOPS", "holdout")
    if which == "holdout":
        shops = holdout
    elif which == "dev":
        shops = sorted(p.name for p in (data / "data").iterdir() if p.name not in holdout)
    else:
        shops = which.split(",")
    out = W / "v5_out"
    env = dict(os.environ, PYTHONPATH=str(code / "src"), PYTHONIOENCODING="utf-8")
    v5 = [sys.executable, "-m", "socai.v5"]
    cmd = [*v5, "predict", "--data", str(data), "--out", str(out), "--cache", str(W / "cache"),
           "--llm", ",".join(f"http://127.0.0.1:{p}/v1" for p in PORTS), "--shops", ",".join(shops),
           "--bank-exclude-own", "--budget-min", "660", "--schema-budget", "5000",  # Kaggle: phiên 12 giờ, ngữ cảnh 8K
           *os.environ.get("V5_ARGS", "").split()]
    limit = os.environ.get("V5_LIMIT", "20")
    if limit != "0":
        cmd += ["--limit", limit]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, env=env)
    r = subprocess.run([*v5, "eval", "--pred", str(out / "submission.jsonl"), "--ref", str(data / "answers.jsonl"),
                        "--questions", str(data / "questions.jsonl")], env=env, capture_output=True, text=True)
    (out / "eval.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    print(r.stdout, r.stderr, flush=True)
    if os.environ.get("V5_BUILD") == "1":
        for name in ("bank", "calibrate"):
            subprocess.run([*v5, name, "--log", str(out / "logs" / "log.jsonl"), "--data", str(data)], env=env)
        for f in ("bank.json", "calibration.json"):  # ghi mặc định vào src/socai/v5/; chép ra để tải về
            src = code / "src" / "socai" / "v5" / f
            if src.exists():
                shutil.copy(src, out / f)
    for p in servers:
        p.terminate()
    log("XONG")


if __name__ == "__main__":
    main()
