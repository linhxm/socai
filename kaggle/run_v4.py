"""Kaggle (GPU T4 x2): chạy SOCAI v4 (L1→L6) với Qwen3-14B. Hai server, mỗi GPU một nửa số shop.

Biến môi trường:
- V4_SPLIT: "train" (mặc định, đo trên 6 shop hold-out, có đáp án) hoặc "public" (cần public_test/ trong dataset).
- V4_SHOPS: danh sách shop, phẩy ngăn cách (mặc định: hold-out trong configs/holdout.json khi split=train; mọi shop khi public).
- V4_LIMIT: số câu tối đa mỗi shop (mặc định 20).
- V4_ARGS: tham số thêm cho `python -m socai.v4 predict` (vd "--samples 3", "--thinking", "--enrich-rules").
Kết quả ở /kaggle/working/v4_out: submission.jsonl, logs/ (log từng câu, log dựng shop), eval.txt (khi có đáp án).
"""

import glob
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_14b import W, build_llama, download, log  # noqa: E402

MODEL = "qwen3-14b"
SPLIT = os.environ.get("V4_SPLIT", "train")
LIMIT = os.environ.get("V4_LIMIT", "20")
ARGS = os.environ.get("V4_ARGS", "")


def find_data() -> tuple[Path, Path]:
    """Mã = repo chứa script này (git clone vào /kaggle/working/socai). Dữ liệu theo V4_SPLIT, tìm trong mọi Input
    (zip thì giải nén vào /kaggle/working/inputs):
    - train: thư mục có questions.jsonl + answers.jsonl (dataset socai-data);
    - public: thư mục có questions.jsonl + rulebook.md, không có answers.jsonl (dataset public riêng).
    v4 không cần bank.json."""
    code = Path(__file__).resolve().parents[1]
    if not (code / "src" / "socai" / "v4").exists():
        raise SystemExit(f"mã chưa có v4: {code} (git pull / clone lại)")
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        with zipfile.ZipFile(z) as f:
            f.extractall(W / "inputs")
    cands = [Path(p).parent for r in ("/kaggle/input", str(W / "inputs"))
             for p in glob.glob(f"{r}/**/questions.jsonl", recursive=True)]
    if SPLIT == "train":
        data = next((d for d in cands if (d / "answers.jsonl").exists() and (d / "data").exists()), None)
    else:
        data = next((d for d in cands if (d / "rulebook.md").exists() and (d / "data").exists()
                     and not (d / "answers.jsonl").exists()), None)
    if data is None:
        found = [str(d) for d in cands]
        raise SystemExit(f"không thấy dữ liệu split={SPLIT} trong Input. questions.jsonl tìm thấy ở: {found or 'không có'}"
                         " — gắn dataset (Add Input) rồi chạy lại")
    log(f"mã: {code} · dữ liệu ({SPLIT}): {data}")
    return code, data


def main() -> None:
    code, data = find_data()
    sys.path.insert(0, str(code / "src"))
    from socai.llm.serve import profile, start, wait_ready
    prof = profile(MODEL, code / "configs" / "models.json")
    got: dict = {}
    th = threading.Thread(target=lambda: got.setdefault("gguf", download(prof)))
    th.start()
    binary = build_llama()
    th.join()
    servers = [start(MODEL, str(binary), got["gguf"], 8088 + g, gpu=g, log=W / f"server{g}.log") for g in (0, 1)]
    for g, p in enumerate(servers):
        wait_ready(f"http://127.0.0.1:{8088 + g}/v1", timeout_s=900, proc=p)
    log("hai server 14B sẵn sàng")

    if os.environ.get("V4_SHOPS"):
        shops = os.environ["V4_SHOPS"].split(",")
    elif SPLIT == "train":
        shops = json.loads((code / "configs" / "holdout.json").read_text(encoding="utf-8"))["holdout"]
    else:
        shops = sorted(p.name for p in (data / "data").iterdir())
    halves = [shops[0::2], shops[1::2]]
    out = W / "v4_out"
    env = dict(os.environ, PYTHONPATH=str(code / "src"), PYTHONIOENCODING="utf-8")
    procs = []
    for g, part in enumerate(halves):
        if not part:
            continue
        cmd = [sys.executable, "-m", "socai.v4", "predict", "--data", str(data), "--out", str(out / f"gpu{g}"),
               "--llm", f"http://127.0.0.1:{8088 + g}/v1", "--shops", ",".join(part), "--limit", LIMIT,
               "--cache", str(W / "cache"), *ARGS.split()]
        log("$ " + " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(W / f"v4_gpu{g}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    # gộp hai nửa
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for name in ("submission.jsonl", "logs/log.jsonl", "logs/shops.jsonl"):
        with open(out / name, "w", encoding="utf-8") as f:
            for g in (0, 1):
                part = out / f"gpu{g}" / name
                if part.exists():
                    f.write(part.read_text(encoding="utf-8"))
    if (data / "answers.jsonl").exists():
        r = subprocess.run([sys.executable, "-m", "socai.v4", "eval", "--pred", str(out / "submission.jsonl"),
                            "--ref", str(data / "answers.jsonl"), "--questions", str(data / "questions.jsonl")],
                           env=env, capture_output=True, text=True)
        (out / "eval.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        print(r.stdout, r.stderr, flush=True)
    for p in servers:
        p.terminate()
    log("XONG")


if __name__ == "__main__":
    main()
