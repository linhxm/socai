"""Kaggle (GPU T4 x2): đo Qwen3-14B với hệ thống hiện tại trên hold-out. Giới hạn giờ: biến BUDGET_MIN (mặc định không).

GPU 0: cấu hình A, có ngân hàng ví dụ + ý kiến thứ hai (giống bản chạy laptop 12/11).
GPU 1: cấu hình B, KHÔNG ngân hàng (chỉ ví dụ viết tay): đo câu kiểu mới, rủi ro chính ở vòng private.
Dữ liệu: dataset riêng tư chứa socai/ (mã + bank.json) và train/. Kết quả ở /kaggle/working.
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

T0 = time.time()
BUDGET_MIN = float(os.environ.get("BUDGET_MIN", "0"))  # 0 = không giới hạn (Kaggle tự cắt ở 12 giờ)
N_PER_SHOP = int(os.environ.get("N_PER_SHOP", "20"))
W = Path("/kaggle/working")
MODEL = "qwen3-14b"


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:5.1f} phút] {msg}", flush=True)


def sh(cmd: str, **kw) -> None:
    log("$ " + cmd[:160])
    subprocess.run(cmd, shell=True, check=True, **kw)


def find_inputs() -> tuple[Path, Path]:
    """Mã = repo chứa script này (git clone từ GitHub vào /kaggle/working/socai). Dữ liệu = dataset `socai-data`
    (train/ + bank.json, không có trên GitHub); giải nén nếu Kaggle để nguyên file zip."""
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        with zipfile.ZipFile(z) as f:
            f.extractall(W / "inputs")
    roots = [Path("/kaggle/input"), W / "inputs"]
    code = Path(__file__).resolve().parents[1]
    train = next((Path(p).parent for r in roots for p in glob.glob(f"{r}/**/questions.jsonl", recursive=True)
                  if Path(p).parent.joinpath("answers.jsonl").exists()), None)
    if not code.joinpath("src", "socai").exists() or not train:
        raise SystemExit(f"không tìm thấy mã hoặc dữ liệu: code={code} train={train}")
    dst = W / "socai"  # bản ghi được (eval ghi cache/kết quả); clone sẵn ở đây thì dùng luôn
    if dst.resolve() != code:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(code, dst, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    bank = dst / "src" / "socai" / "v2" / "bank.json"
    if not bank.exists():
        src = next((Path(p) for r in roots for p in sorted(glob.glob(f"{r}/**/bank.json", recursive=True))
                    if "/socai/" not in p), None)
        if src is None:
            raise SystemExit("thiếu bank.json (ngân hàng ví dụ) trong dataset socai-data")
        shutil.copy2(src, bank)
        log(f"ngân hàng ví dụ: {src}")
    return dst, train


def tail(p: Path, n: int = 40) -> None:
    if p.exists():
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"----- {p.name} (cuối) -----\n" + "\n".join(lines[-n:]), flush=True)


def find_nvcc() -> str | None:
    for c in [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", *sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"))]:
        if c and Path(c).exists():
            return c
    return None


def build_from_source(nvcc: str) -> Path:
    src = W / "llama.cpp"
    if not src.exists():
        try:
            sh(f"git clone --depth 1 --branch b12246 https://github.com/ggml-org/llama.cpp {src}")
        except subprocess.CalledProcessError:
            sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {src}")
    cuda = Path(nvcc).parent.parent
    os.environ["PATH"] = f"{cuda / 'bin'}:{os.environ['PATH']}"
    os.environ["CUDACXX"] = nvcc
    stubs = cuda / "lib64" / "stubs"  # libcuda.so giả để link lúc dựng; driver thật có sẵn lúc chạy
    os.environ["LIBRARY_PATH"] = f"{stubs}:{cuda / 'lib64'}:{os.environ.get('LIBRARY_PATH', '')}"
    # Ảnh Kaggle (CUDA 13) không để libcuda.so ở chỗ FindCUDAToolkit tìm → target CUDA::cuda_driver không có.
    # Chỉ thẳng file libcuda (stub của toolkit hoặc driver thật của phiên GPU) + tắt VMM cho chắc.
    libcuda = next((p for pat in (f"{cuda}/lib64/stubs/libcuda.so", f"{cuda}/targets/*/lib/stubs/libcuda.so",
                                  "/usr/local/cuda*/**/stubs/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so*",
                                  "/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib64-nvidia/libcuda.so*",
                                  "/usr/lib/wsl/lib/libcuda.so*")
                    for p in sorted(glob.glob(pat, recursive=True))), None)
    log(f"libcuda: {libcuda}")
    flags = (f"-DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER={nvcc} "
             f"-DCUDAToolkit_ROOT={cuda} -DCMAKE_LIBRARY_PATH={stubs} -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF "
             "-DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release"
             + (f" -DCUDA_cuda_driver_LIBRARY={libcuda}" if libcuda else ""))
    try:
        sh(f"rm -rf {src}/build && cmake -S {src} -B {src}/build {flags} > {W}/cmake.log 2>&1")
        cmd = f"cmake --build {src}/build --target llama-server -j {os.cpu_count()} > {W}/build.log 2>&1"
        log("$ " + cmd[:160] + "  (CUDA kernel biên dịch 30–40 phút trên 4 nhân; in tiến độ mỗi 2 phút)")
        p = subprocess.Popen(cmd, shell=True)
        while p.poll() is None:
            time.sleep(120)
            lines = (W / "build.log").read_text(encoding="utf-8", errors="replace").splitlines() \
                if (W / "build.log").exists() else []
            pct = next((ln.split("]")[0] + "]" for ln in reversed(lines) if ln.startswith("[")), "?")
            log(f"đang biên dịch {pct} ({len(lines)} dòng log)")
        if p.returncode:
            raise subprocess.CalledProcessError(p.returncode, cmd)
    except subprocess.CalledProcessError:
        txt = (W / "cmake.log").read_text(encoding="utf-8", errors="replace") if (W / "cmake.log").exists() else ""
        print("----- dòng lỗi cmake -----\n" + "\n".join(
            ln for ln in txt.splitlines() if "Error" in ln or "Target" in ln or "CUDA::" in ln or "not found" in ln),
            flush=True)
        tail(W / "build.log")
        raise
    return next(Path(p) for p in glob.glob(f"{src}/build/**/llama-server", recursive=True))


def prebuilt() -> Path | None:
    """Bản dựng sẵn trên trang phát hành llama.cpp: ưu tiên Linux + CUDA, sau đó Linux + Vulkan."""
    import tarfile
    import urllib.request
    req = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30",
                                 headers={"User-Agent": "socai"})
    rels = json.loads(urllib.request.urlopen(req, timeout=60).read())
    assets = [(a["name"], a["browser_download_url"]) for r in rels for a in r.get("assets", [])]
    log(f"{len(rels)} bản phát hành, gói Linux: {[n for n, _ in assets if 'ubuntu' in n or 'linux' in n][:8]}")
    for want in (("cuda",), ("vulkan",)):
        for name, url in assets:
            n = name.lower()
            if ("ubuntu" in n or "linux" in n) and all(w in n for w in want) and "arm" not in n:
                dst = W / "prebuilt"
                dst.mkdir(exist_ok=True)
                f = dst / name
                urllib.request.urlretrieve(url, f)
                if name.endswith(".zip"):
                    zipfile.ZipFile(f).extractall(dst)
                else:
                    tarfile.open(f).extractall(dst)
                b = next((Path(p) for p in glob.glob(f"{dst}/**/llama-server", recursive=True)), None)
                if b:
                    b.chmod(0o755)
                    os.environ["LD_LIBRARY_PATH"] = f"{b.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                    log(f"dùng bản dựng sẵn {name}")
                    return b
    return None


def cached_binary() -> Path | None:
    """llama-server đã dựng ở lần chạy trước (gắn output của notebook trước làm Input) → bỏ qua 30–40 phút dựng."""
    found = sorted(glob.glob("/kaggle/input/**/llama-bin/llama-server", recursive=True)) + \
        sorted(glob.glob("/kaggle/input/**/build/bin/llama-server", recursive=True))  # output của bản script cũ
    for p in found:
        dst = W / "llama-bin"
        if not dst.exists():
            shutil.copytree(Path(p).parent, dst)
        b = dst / "llama-server"
        b.chmod(0o755)
        os.environ["LD_LIBRARY_PATH"] = f"{dst}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        if subprocess.run(f"{b} --version", shell=True, capture_output=True).returncode == 0:
            log(f"dùng llama-server dựng sẵn từ lần trước: {p}")
            return b
    return None


def keep_binary(b: Path) -> Path:
    """Chép llama-server + thư viện .so vào /kaggle/working/llama-bin để lần sau gắn lại, khỏi dựng."""
    dst = W / "llama-bin"
    dst.mkdir(exist_ok=True)
    shutil.copy2(b, dst / "llama-server")
    for so in glob.glob(f"{W}/llama.cpp/build/**/*.so*", recursive=True):
        shutil.copy2(so, dst / Path(so).name)
    os.environ["LD_LIBRARY_PATH"] = f"{dst}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    log(f"đã lưu bản dựng vào {dst} (gắn output notebook này làm Input cho lần sau)")
    return dst / "llama-server"


def build_llama() -> Path:
    if (b := cached_binary()) is not None:
        return b
    nvcc = find_nvcc()
    log(f"nvcc: {nvcc}")
    if nvcc:
        try:
            b = build_from_source(nvcc)
            log(f"build xong: {b}")
            return keep_binary(b)
        except Exception as e:  # noqa: BLE001
            log(f"dựng từ mã nguồn lỗi ({e}); thử bản dựng sẵn")
    b = prebuilt()
    if b is None:
        raise SystemExit("không có llama-server chạy GPU: xem log cmake ở trên")
    return b


def download(profile: dict) -> Path:
    from huggingface_hub import HfApi, hf_hub_download
    repos = [profile["hf_repo"], "unsloth/Qwen3-14B-GGUF", "bartowski/Qwen_Qwen3-14B-GGUF"]
    for repo in repos:
        try:
            files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf") and "Q4_K_M" in f]
            name = profile["gguf"] if profile["gguf"] in files else (files[0] if files else None)
            if name:
                p = Path(hf_hub_download(repo, name, local_dir=str(W / "models")))
                log(f"tải xong {repo}/{name} ({p.stat().st_size / 1e9:.2f} GB)")
                return p
        except Exception as e:  # noqa: BLE001
            log(f"không tải được từ {repo}: {e}")
    raise SystemExit("không tải được GGUF 14B")


def main() -> None:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    code, train = find_inputs()
    log(f"mã: {code} | dữ liệu: {train}")
    sh("pip install -q 'pandas==3.1.1' 'python-calamine==0.9.1' 'openpyxl==3.1.5' huggingface_hub "
       "|| pip install -q python-calamine openpyxl huggingface_hub")
    sys.path.insert(0, str(code / "src"))
    from socai.llm.serve import profile, start, wait_ready

    prof = profile(MODEL, code / "configs" / "models.json")
    got: dict = {}
    th = threading.Thread(target=lambda: got.setdefault("gguf", download(prof)))
    th.start()
    binary = build_llama()
    th.join()
    gguf = got["gguf"]

    servers = [start(MODEL, str(binary), gguf, 8088 + g, gpu=g, log=W / f"server{g}.log") for g in (0, 1)]
    for g, p in enumerate(servers):
        wait_ready(f"http://127.0.0.1:{8088 + g}/v1", timeout_s=900, proc=p)
    log("hai server 14B sẵn sàng")

    left = BUDGET_MIN - (time.time() - T0) / 60 if BUDGET_MIN else 0.0
    if BUDGET_MIN and left < 10:
        raise SystemExit(f"chỉ còn {left:.0f} phút sau khi dựng: dừng để không vượt BUDGET_MIN={BUDGET_MIN:.0f}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(code / "src"))
    common = (f"{sys.executable} {code}/scripts/eval_holdout.py --train {train} --cache {W}/cache "
              f"--n-per-shop {N_PER_SHOP}" + (f" --deadline-min {left - 3:.1f}" if BUDGET_MIN else ""))
    runs = {"bank": f"{common} --llm http://127.0.0.1:8088/v1 --out {W}/h14_bank.json",
            "nobank": f"{common} --llm http://127.0.0.1:8089/v1 --out {W}/h14_nobank.json --no-bank"}
    procs = {k: subprocess.Popen(v, shell=True, env=env, stdout=open(W / f"eval_{k}.log", "w"),
                                 stderr=subprocess.STDOUT) for k, v in runs.items()}
    log(f"đang đo, {'hạn %.0f phút' % (left - 3) if BUDGET_MIN else 'không giới hạn giờ'}, "
        f"{N_PER_SHOP or 'mọi'} câu/shop mỗi cấu hình")
    while any(p.poll() is None for p in procs.values()):
        time.sleep(60)
        for k in procs:
            tail = (W / f"eval_{k}.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:]
            log(f"{k}: {tail[0][:120] if tail else '...'}")
    out = {}
    for k in procs:
        f = W / f"h14_{k}.json"
        out[k] = json.loads(f.read_text(encoding="utf-8"))["summary"] if f.exists() else None
    (W / "summary.json").write_text(json.dumps({"model": MODEL, "gguf": gguf.name, "n_per_shop": N_PER_SHOP,
                                                "minutes": round((time.time() - T0) / 60, 1), **out},
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    log("TỔNG KẾT " + json.dumps(out, ensure_ascii=False))
    for p in servers:
        p.terminate()


if __name__ == "__main__":
    main()
