"""Kaggle (GPU T4 x2): chạy SOCAI v5 với Qwen3-14B, mỗi GPU một server, v5 tự chia câu hỏi cho hai server.

Hai chế độ (V5_SPLIT):
- "train" (mặc định): đo trên train như cũ. V5_SHOPS = "holdout" (mặc định) | "dev" | danh sách shop.
  V5_BUILD=1 dựng bank.json + calibration.json từ log (chỉ nên bật khi V5_SHOPS=dev).
- "public": đo trên public test, ba bước trong một phiên, cùng một bộ câu (V5_LIMIT câu đầu mỗi shop):
  (1) nếu có train trong Input và V5_BUILD khác "0": chạy shop dev train (V5_DEV_LIMIT câu/shop) → bank + calibration;
  (2) v2 trên public (mốc so sánh, cần bank.json của v2 trong Input; tắt bằng V5_V2=0);
  (3) v5 trên public với --base v2 --verify: v5 chạy mọi câu, nên từ một log tính được v5 riêng, v2+v5, kiểm chéo.
  Public không có đáp án trên Kaggle: tải v5_out/ về rồi chấm trên máy.

Biến chung: V5_LIMIT (câu tối đa mỗi shop, mặc định 20; "0" = tất cả), V5_ARGS (tham số thêm cho
`python -m socai.v5 predict`, dùng cho cả bước dựng ngân hàng, vd "--v2-tables --samples 4").
Kết quả ở /kaggle/working/v5_out.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_14b import W, build_llama, download, log  # noqa: E402

MODEL = "qwen3-14b"
PORTS = (8088, 8089)
SPLIT = os.environ.get("V5_SPLIT", "train")
T0 = time.time()


def find_inputs() -> tuple[Path | None, Path | None]:
    """(train, public) trong mọi Input; zip thì giải nén vào /kaggle/working/inputs.
    train = có answers.jsonl + data/; public = có rulebook.md + data/, không có answers.jsonl."""
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        with zipfile.ZipFile(z) as f:
            f.extractall(W / "inputs")
    cands = [Path(p).parent for r in ("/kaggle/input", str(W / "inputs"))
             for p in glob.glob(f"{r}/**/questions.jsonl", recursive=True)]
    train = next((d for d in cands if (d / "answers.jsonl").exists() and (d / "data").exists()), None)
    public = next((d for d in cands if (d / "rulebook.md").exists() and (d / "data").exists()
                   and not (d / "answers.jsonl").exists()), None)
    log(f"train: {train} · public: {public} · questions.jsonl thấy ở: {[str(d) for d in cands]}")
    return train, public


def subset(data: Path, limit: int, dst: Path, shops: list[str] | None = None) -> Path:
    """Thư mục split chỉ giữ `limit` câu đầu mỗi shop (cùng cách chọn với `--limit` của v5); data/ và luật trỏ về bản gốc."""
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for name in ("data", "rulebook.md", "calendar.csv"):
        if (data / name).exists():
            (dst / name).symlink_to(data / name)
    seen: dict[str, int] = {}
    keep = []
    for ln in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        b = json.loads(ln)["business_id"]
        if shops is not None and b not in shops:
            continue
        seen[b] = seen.get(b, 0) + 1
        if not limit or seen[b] <= limit:
            keep.append(ln)
    (dst / "questions.jsonl").write_text("\n".join(keep) + "\n", encoding="utf-8")
    log(f"{dst.name}: {len(keep)} câu, {len(seen)} shop")
    return dst


def run_v2(code: Path, data: Path, out: Path, env: dict) -> Path | None:
    """v2 (`python -m socai predict`, một server mỗi lượt): chia shop làm hai nửa chạy song song, gộp theo thứ tự câu."""
    bank = code / "src" / "socai" / "v2" / "bank.json"
    if not bank.exists():
        src = next((Path(p) for r in ("/kaggle/input", str(W / "inputs"))
                    for p in sorted(glob.glob(f"{r}/**/bank.json", recursive=True)) if "/socai/" not in p), None)
        if src is None:
            log("không có bank.json của v2 trong Input → bỏ bước v2")
            return None
        shutil.copy2(src, bank)
    qs = [ln for ln in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    shops = sorted({json.loads(ln)["business_id"] for ln in qs})
    procs = []
    for i, port in enumerate(PORTS):
        half = subset(data, 0, out / f"split{i}", shops[i::2])
        e = dict(env, SOCAI_LLM_URL=f"http://127.0.0.1:{port}/v1")
        cmd = [sys.executable, "-m", "socai", "predict", "--data", str(half), "--out", str(out / f"run{i}"),
               "--cache", str(W / "cache_v2")]
        log("$ " + " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=e))
    for p in procs:
        p.wait()
    got = {}
    for i in range(len(PORTS)):
        f = out / f"run{i}" / "submission.jsonl"
        if f.exists():
            got.update({r["id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines()) if r})
    sub = out / "submission.jsonl"
    with sub.open("w", encoding="utf-8") as f:
        for ln in qs:
            qid = json.loads(ln)["id"]
            f.write(json.dumps(got.get(qid, {"id": qid, "type": "abstain"}), ensure_ascii=False) + "\n")
    log(f"v2 xong: {len(got)}/{len(qs)} câu có kết quả · {time.time() - T0:.0f}s")
    return sub


def main() -> None:
    code = Path(__file__).resolve().parents[1]
    train, public = find_inputs()
    if SPLIT == "public" and public is None:
        raise SystemExit("V5_SPLIT=public nhưng không thấy public_test (questions.jsonl + rulebook.md + data/) trong Input")
    if SPLIT != "public" and train is None:
        raise SystemExit("không thấy dữ liệu train (questions.jsonl + answers.jsonl + data/) trong Input")
    log(f"mã: {code}")
    sys.path.insert(0, str(code / "src"))
    from socai.llm.serve import profile, start, wait_ready
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb", "python-calamine", "openpyxl"], check=True)
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
    log(f"hai server 14B sẵn sàng · {time.time() - T0:.0f}s")

    holdout = json.loads((code / "configs" / "holdout.json").read_text(encoding="utf-8"))["holdout"]
    out = W / "v5_out"
    out.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(code / "src"), PYTHONIOENCODING="utf-8")
    v5 = [sys.executable, "-m", "socai.v5"]
    extra = os.environ.get("V5_ARGS", "").split()
    llms = ",".join(f"http://127.0.0.1:{p}/v1" for p in PORTS)
    common = ["--llm", llms, "--cache", str(W / "cache"), "--schema-budget", "5000", *extra]  # ngữ cảnh 8K

    def predict(data: Path, dst: Path, budget_min: float, *more: str) -> None:
        cmd = [*v5, "predict", "--data", str(data), "--out", str(dst), "--budget-min", f"{budget_min:.0f}", *common, *more]
        log("$ " + " ".join(cmd))
        subprocess.run(cmd, env=env)

    def build_bank(log_path: Path) -> None:
        for name in ("bank", "calibrate"):  # ghi vào src/socai/v5/ (predict sau đó tự dùng); chép ra để tải về
            subprocess.run([*v5, name, "--log", str(log_path), "--data", str(train)], env=env)
        for f in ("bank.json", "calibration.json"):
            src = code / "src" / "socai" / "v5" / f
            if src.exists():
                shutil.copy(src, out / f)

    def evaluate(dst: Path, data: Path) -> None:
        r = subprocess.run([*v5, "eval", "--pred", str(dst / "submission.jsonl"), "--ref", str(data / "answers.jsonl"),
                            "--questions", str(data / "questions.jsonl"), "--log", str(dst / "logs" / "log.jsonl")],
                           env=env, capture_output=True, text=True)
        (dst / "eval.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        print(r.stdout, r.stderr, flush=True)

    if SPLIT == "public":
        limit = int(os.environ.get("V5_LIMIT", "20"))
        if train is not None and os.environ.get("V5_BUILD", "1") != "0":
            dev = sorted(p.name for p in (train / "data").iterdir() if p.name not in holdout)
            sub = subset(train, int(os.environ.get("V5_DEV_LIMIT", "20")), W / "train_dev", dev)
            predict(sub, out / "dev", float(os.environ.get("V5_DEV_MIN", "240")), "--bank-exclude-own")
            evaluate(out / "dev", train)
            build_bank(out / "dev" / "logs" / "log.jsonl")
        else:
            log("bỏ bước dựng ngân hàng v5 (không có train hoặc V5_BUILD=0): v5 chạy không ví dụ, ngưỡng mặc định")
        pub = subset(public, limit, W / "public_sub")
        shutil.copy(pub / "questions.jsonl", out / "public_questions.jsonl")
        base = run_v2(code, pub, out / "v2", env) if os.environ.get("V5_V2", "1") != "0" else None
        left = 700 - (time.time() - T0) / 60  # phiên Kaggle 12 giờ, chừa 20 phút
        if base is not None:
            predict(pub, out / "public", left, "--base", str(base), "--verify")
        else:
            predict(pub, out / "public", left)
    else:
        which = os.environ.get("V5_SHOPS", "holdout")
        shops = holdout if which == "holdout" else (
            sorted(p.name for p in (train / "data").iterdir() if p.name not in holdout) if which == "dev"
            else which.split(","))
        limit = os.environ.get("V5_LIMIT", "20")
        predict(train, out, 660, "--shops", ",".join(shops), "--bank-exclude-own",
                *(["--limit", limit] if limit != "0" else []))
        evaluate(out, train)
        if os.environ.get("V5_BUILD") == "1":
            build_bank(out / "logs" / "log.jsonl")
    for p in servers:
        p.terminate()
    # Output Kaggle = cả /kaggle/working: bỏ GGUF (~9 GB), dữ liệu giải nén, cache và thư mục chứa symlink để tải về nhanh
    for d in (W / "models", W / "inputs", W / "cache", W / "cache_v2", W / "train_dev", W / "public_sub",
              out / "v2" / "split0", out / "v2" / "split1"):
        shutil.rmtree(d, ignore_errors=True)
    log(f"XONG · {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
