"""Khởi động trên Kaggle: clone mã mới nhất từ GitHub rồi chạy một script trong kaggle/.

Dữ liệu (train/ + bank.json) lấy từ dataset riêng `socai-data`, không có trên GitHub.
Chọn script bằng biến SCRIPT: run_finetune.py (mặc định) hoặc run_14b.py. REF: nhánh/commit (mặc định main).
"""

import os
import subprocess
import sys

REPO = "https://github.com/linhxm/socai"
DST = "/kaggle/working/socai"
script = os.environ.get("SCRIPT", "run_finetune.py")
ref = os.environ.get("REF", "main")

subprocess.run(f"rm -rf {DST} && git clone -q {REPO} {DST} && git -C {DST} checkout -q {ref}", shell=True, check=True)
commit = subprocess.run(f"git -C {DST} log -1 --format='%h %s'", shell=True, capture_output=True, text=True).stdout
print(f"mã: {REPO}@{commit.strip()}", flush=True)
sys.exit(subprocess.run([sys.executable, f"{DST}/kaggle/{script}"]).returncode)
