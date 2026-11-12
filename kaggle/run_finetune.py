"""Kaggle (GPU T4 x2): một lượt fine-tune LoRA Qwen3-4B cho v2 rồi đo trên hold-out, so với bản gốc.

GPU 0: fine-tune (dữ liệu `training_code/data/finetune`, chỉ từ train) → gộp adapter → GGUF f16 → đo.
GPU 1: trong lúc GPU 0 train, đo bản gốc chuyển GGUF f16 theo đúng quy trình (so sánh công bằng, không lẫn lượng tử hoá).
Đo: 120 câu hold-out (20 câu × 6 shop, đúng bộ câu của các lần đo 4B/14B trước), có và không có ngân hàng ví dụ.
Biến môi trường: EPOCHS (2), TRAIN_MIN (300, trần giờ train), N_PER_SHOP (20), KEEP_GGUF (0).
Mã: git clone repo vào /kaggle/working/socai; dữ liệu: dataset `socai-data` (train/ + bank.json).
Kết quả ở /kaggle/working: summary.json, train.log, lora/ (adapter + log_history), ft_*.json, base_*.json.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_14b import T0, W, build_llama, find_inputs, log, sh, tail  # noqa: E402

BASE = "Qwen/Qwen3-4B"
MODEL = "qwen3-4b"  # hồ sơ server trong configs/models.json
EPOCHS = os.environ.get("EPOCHS", "2")
TRAIN_MIN = float(os.environ.get("TRAIN_MIN", "300"))
N_PER_SHOP = int(os.environ.get("N_PER_SHOP", "20"))
KEEP_GGUF = os.environ.get("KEEP_GGUF", "0") == "1"


def download_base() -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download
    rev = HfApi().model_info(BASE).sha
    p = Path(snapshot_download(BASE, revision=rev, local_dir=str(W / "hf-base"),
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "vocab*", "merges*"]))
    log(f"tải xong {BASE}@{rev}")
    return p, rev


def llama_source() -> Path:
    src = W / "llama.cpp"
    if not (src / "convert_hf_to_gguf.py").exists():
        shutil.rmtree(src, ignore_errors=True)
        try:
            sh(f"git clone -q --depth 1 --branch b12246 https://github.com/ggml-org/llama.cpp {src}")
        except subprocess.CalledProcessError:
            sh(f"git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {src}")
    return src


def to_gguf(hf_dir: Path, out: Path) -> Path:
    """HF safetensors → GGUF f16 bằng script của llama.cpp; thiếu module nào thì cài module đó rồi thử lại."""
    src = llama_source()
    env = dict(os.environ, PYTHONPATH=str(src / "gguf-py"), CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, str(src / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "f16", "--outfile", str(out)]
    for _ in range(4):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode == 0:
            log(f"GGUF xong: {out.name} ({out.stat().st_size / 1e9:.2f} GB)")
            return out
        m = re.search(r"No module named '([\w\.]+)'", r.stderr)
        if not m:
            print(r.stderr[-3000:], flush=True)
            raise RuntimeError("chuyển GGUF lỗi")
        mod = m.group(1).split(".")[0].replace("_", "-")
        sh(f"pip install -q {mod}")
    raise RuntimeError("chuyển GGUF lỗi sau khi cài thêm module")


def evaluate(code: Path, train: Path, url: str, out: Path, nobank: bool) -> dict | None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(code / "src"))
    cmd = (f"{sys.executable} {code}/scripts/eval_holdout.py --train {train} --cache {W}/cache "
           f"--n-per-shop {N_PER_SHOP} --llm {url} --out {out}" + (" --no-bank" if nobank else ""))
    with open(out.with_suffix(".log"), "w") as f:
        subprocess.run(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)
    s = json.loads(out.read_text(encoding="utf-8"))["summary"] if out.exists() else None
    log(f"{out.stem}: {json.dumps(s and {k: s[k] for k in ('n', 'final', 'raw', 'sec_per_q')}, ensure_ascii=False)}")
    return s


def serve(code: Path, binary: Path, gguf: Path, gpu: int, port: int):
    from socai.llm.serve import start, wait_ready
    p = start(MODEL, str(binary), gguf, port, gpu=gpu, log=W / f"server_{gguf.stem}_{gpu}.log")
    wait_ready(f"http://127.0.0.1:{port}/v1", timeout_s=900, proc=p)
    return p


def main() -> None:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    code, train = find_inputs()
    log(f"mã: {code} | dữ liệu: {train}")
    sh("pip install -q peft 'pandas==3.1.1' 'python-calamine==0.9.1' 'openpyxl==3.1.5' huggingface_hub "
       "|| pip install -q peft python-calamine openpyxl huggingface_hub")
    sh(f"{sys.executable} -c \"import torch, transformers, peft; print('torch', torch.__version__, 'transformers', "
       f"transformers.__version__, 'peft', peft.__version__)\"")
    sys.path.insert(0, str(code / "src"))

    # Dữ liệu fine-tune sinh tại chỗ từ train + ngân hàng ví dụ (không đưa lên GitHub), song song với tải mô hình.
    data = code / "training_code" / "data" / "finetune"
    export = subprocess.Popen(
        f"{sys.executable} {code}/training_code/export_finetune.py --train {train} --bank {code}/src/socai/v2/bank.json "
        f"--out {data} --cache {W}/cache --val-biz biz_109,biz_124",
        shell=True, env=dict(os.environ, PYTHONPATH=str(code / "src"), PYTHONIOENCODING="utf-8"),
        stdout=open(W / "export.log", "w"), stderr=subprocess.STDOUT)
    hf, rev = download_base()
    if export.wait() or not (data / "train.jsonl").exists():
        tail(W / "export.log")
        raise SystemExit("sinh dữ liệu fine-tune lỗi: xem export.log")
    log(f"dữ liệu fine-tune: {json.loads((data / 'manifest.json').read_text(encoding='utf-8'))['counts']}")
    lora = W / "lora"
    train_cmd = (f"{sys.executable} {code}/training_code/finetune_lora.py --data {data} --out {lora} --base {hf} "
                 f"--epochs {EPOCHS} --max-minutes {TRAIN_MIN}")
    trainer = subprocess.Popen(train_cmd, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                               stdout=open(W / "train.log", "w"), stderr=subprocess.STDOUT)
    log(f"đang fine-tune trên GPU 0 (EPOCHS={EPOCHS}, trần {TRAIN_MIN:.0f} phút); xem train.log")

    res: dict = {}

    def base_side() -> None:  # GPU 1: đo bản gốc trong lúc GPU 0 train
        try:
            res["binary"] = build_llama()
            g = to_gguf(hf, W / "base-f16.gguf")
            p = serve(code, res["binary"], g, 1, 8089)
            for k, nb in (("base_bank", False), ("base_nobank", True)):
                res[k] = evaluate(code, train, "http://127.0.0.1:8089/v1", W / f"{k}.json", nb)
            p.terminate()
        except Exception as e:  # noqa: BLE001
            res["base_error"] = repr(e)
            log(f"phía bản gốc lỗi: {e!r}")

    side = threading.Thread(target=base_side)
    side.start()
    while trainer.poll() is None:
        time.sleep(120)
        last = (W / "train.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
        log(f"train: {last[0][:150] if last else '...'}")
    tail(W / "train.log", 15)
    if trainer.returncode or not (lora / "adapter").exists():
        side.join()
        raise SystemExit("fine-tune lỗi: xem train.log")

    merged = W / "merged"
    sh(f"CUDA_VISIBLE_DEVICES= {sys.executable} {code}/training_code/finetune_lora.py --merge {lora} --out {merged} "
       f"--base {hf} > {W}/merge.log 2>&1")
    ft = to_gguf(merged, W / "ft-f16.gguf")
    side.join()
    if "binary" not in res:
        raise SystemExit(f"không có llama-server: {res.get('base_error')}")
    servers = [serve(code, res["binary"], ft, g, 8088 + g) for g in (0, 1)]
    threads = [threading.Thread(target=lambda k=k, g=g, nb=nb: res.__setitem__(
        k, evaluate(code, train, f"http://127.0.0.1:{8088 + g}/v1", W / f"{k}.json", nb)))
        for k, g, nb in (("ft_bank", 0, False), ("ft_nobank", 1, True))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for p in servers:
        p.terminate()

    hist = json.loads((lora / "log_history.json").read_text()) if (lora / "log_history.json").exists() else []
    out = {"base": f"{BASE}@{rev}", "epochs": EPOCHS, "n_per_shop": N_PER_SHOP,
           "minutes": round((time.time() - T0) / 60, 1), "val_loss": [h for h in hist if "val_loss" in h],
           **{k: res.get(k) for k in ("base_bank", "base_nobank", "ft_bank", "ft_nobank", "base_error")}}
    (W / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(data / "manifest.json", W / "finetune_manifest.json")
    shutil.rmtree(merged, ignore_errors=True)
    shutil.rmtree(W / "hf-base", ignore_errors=True)
    if not KEEP_GGUF:
        for g in glob.glob(f"{W}/*.gguf"):
            os.remove(g)
    log("TỔNG KẾT " + json.dumps({k: out[k] and {"final": out[k]["final"], "raw": out[k]["raw"]}
                                  for k in ("base_bank", "base_nobank", "ft_bank", "ft_nobank")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
