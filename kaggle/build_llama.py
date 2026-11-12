"""Kaggle: chỉ dựng llama-server (CUDA, T4 = sm75) rồi lưu vào /kaggle/working/llama-bin để dành.

Các notebook khác gắn output của notebook này làm Input thì `run_14b.py` / `run_finetune.py` tự dùng lại
(tìm `**/llama-bin/llama-server`), khỏi dựng 30–40 phút mỗi lần. Chỉ cần chạy lại khi Kaggle đổi bản CUDA/driver.
Không cần dataset. Chọn GPU (T4) để ảnh có sẵn CUDA toolkit và để kiểm thử bản dựng.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_14b import W, build_from_source, find_nvcc, keep_binary, log, prebuilt, sh  # noqa: E402


def main() -> None:
    subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv", shell=True)
    nvcc = find_nvcc()
    log(f"nvcc: {nvcc}")
    if nvcc:
        b = keep_binary(build_from_source(nvcc))
    else:
        log("ảnh không có nvcc: lấy bản dựng sẵn trên trang phát hành llama.cpp")
        p = prebuilt()
        if p is None:
            raise SystemExit("không có nvcc và không tải được bản dựng sẵn")
        dst = W / "llama-bin"
        shutil.copytree(p.parent, dst, dirs_exist_ok=True)
        b = dst / "llama-server"
    ver = subprocess.run(f"{b} --version", shell=True, capture_output=True, text=True)
    info = {"binary": str(b), "ok": ver.returncode == 0, "version": (ver.stdout + ver.stderr).strip()[-400:],
            "files": sorted(x.name for x in b.parent.iterdir())}
    (b.parent / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(info, ensure_ascii=False))
    for d in ("llama.cpp", "prebuilt", "socai"):  # bỏ mã nguồn, thư mục dựng (vài GB), bản clone: chỉ giữ llama-bin
        shutil.rmtree(W / d, ignore_errors=True)
    sh(f"du -sh {b.parent}")
    if not info["ok"]:
        raise SystemExit("bản dựng không chạy được: xem build_info.json")
    log("XONG: gắn output notebook này làm Input cho notebook đo / fine-tune")


if __name__ == "__main__":
    main()
