# Chạy trên Kaggle (GPU T4 x2)

**Mã** lấy thẳng từ GitHub (`linhxm/socai`, public) mỗi lần chạy, nên sửa mã xong chỉ cần push rồi chạy lại notebook.
**Dữ liệu** nằm trong dataset riêng tư `socai-data`, tải lên **một lần**: `train/` + `bank.json` (ngân hàng ví dụ).
Hai thứ này là dữ liệu cuộc thi nên không đưa lên GitHub. Dữ liệu fine-tune được sinh lại ngay trên Kaggle từ hai thứ này.

| Script (`SCRIPT=`) | Làm gì | Thời gian |
|---|---|---|
| `run_finetune.py` (mặc định) | Fine-tune LoRA Qwen3-4B rồi đo, so với bản gốc ([mục cuối](#fine-tune-lora-run_finetunepy)) | ~4–5 giờ |
| `run_14b.py` | Đo Qwen3-14B trên 120 câu hold-out | ~1 giờ |
| `build_llama.py` | Chỉ dựng `llama-server` để dành (chạy một lần) | ~35 phút |

## Chuẩn bị (một lần)
1. **Dataset `socai-data`:** kaggle.com → *Datasets* → *New Dataset* → kéo `socai_data.zip` (129 MB: `train/` + `bank.json`) vào
   → tên `socai-data` → **Private** → *Create*. Chỉ tải lại (*New Version*) khi ngân hàng ví dụ đổi.
2. **Notebook `socai-llama-build`** (dựng llama.cpp để dành, chỉ làm lại khi Kaggle đổi bản CUDA):
   *New Notebook* → đặt tên `socai-llama-build` → *Accelerator* **GPU T4** (x1 là đủ), *Internet* **On**, không cần Input →
   một ô code:
   ```python
   !rm -rf /kaggle/working/socai && git clone -q https://github.com/linhxm/socai /kaggle/working/socai
   !python /kaggle/working/socai/kaggle/build_llama.py
   ```
   → *Save & Run All (Commit)*. Xong khi log in `XONG` và output có `llama-bin/` (llama-server + các file `.so`,
   `build_info.json`). Mã nguồn llama.cpp bị xoá khỏi output cho nhẹ.
3. Tài khoản Kaggle cần xác minh số điện thoại để bật Internet và GPU.

## Chạy (giao diện web)
1. *Code* → *New Notebook*. Cột phải: *Accelerator* **GPU T4 x2**, *Internet* **On**.
2. *Add Input*: dataset `socai-data` + **output của notebook `socai-llama-build`** (*Your Work* → notebook đó) để dùng lại
   `llama-bin` (khỏi dựng 30 phút; không gắn thì script tự dựng).
3. Một ô code:
   ```python
   import os
   os.environ["EPOCHS"] = "2"
   !rm -rf /kaggle/working/socai && git clone -q https://github.com/linhxm/socai /kaggle/working/socai
   !git -C /kaggle/working/socai log -1 --oneline
   !python /kaggle/working/socai/kaggle/run_finetune.py
   ```
   Dòng thứ hai in commit đang chạy. Đo 14B thì thay `run_finetune.py` bằng `run_14b.py`.
4. *Save Version* → *Save & Run All (Commit)*, tắt trình duyệt được. Kết quả ở tab *Output* của phiên bản đó.

## Chạy (CLI Kaggle trên máy)
```bash
kaggle datasets create -p <thư mục chứa socai_data.zip và dataset-metadata.json>   # một lần
kaggle kernels push -p socai/kaggle --accelerator NvidiaTeslaT4                    # kernel-metadata.json → launcher.py
kaggle kernels status linhxm/socai-finetune
kaggle kernels output linhxm/socai-finetune -p <thư mục nhận kết quả>
```

## Đo Qwen3-14B (`run_14b.py`)
Đo hệ thống hiện tại với Qwen3-14B trên 120 câu hold-out (20 câu × 6 shop), hai cấu hình chạy song song:

| GPU | Cấu hình | Đo cái gì |
|---|---|---|
| 0 | có ngân hàng ví dụ + ý kiến thứ hai | giống bản laptop 12/11 (4B được 0,715 trên toàn hold-out) |
| 1 | không ngân hàng, chỉ ví dụ viết tay | câu kiểu mới, rủi ro chính ở vòng private |

Dựng `llama-server` và tải mô hình mất khoảng 15 phút.

**Kết quả (trong `/kaggle/working`):**
| File | Nội dung |
|---|---|
| `summary.json` | tóm tắt hai cấu hình: số đúng/sai/bỏ trống, PA, giây/câu |
| `h14_bank.json`, `h14_nobank.json` | từng câu: đáp án đúng/sai, nguồn, biểu thức LLM viết |
| `eval_bank.log`, `eval_nobank.log` | tiến độ đo |
| `build.log`, `server0.log`, `server1.log` | dựng llama.cpp, log server |

Trong `summary.json` có hai cách tính:
- `final` là đáp án sau ngưỡng. Ngưỡng hiện được hiệu chỉnh cho 4B, nên cấu hình không ngân hàng sẽ bỏ trống gần hết.
- `raw` là đáp án trước ngưỡng, dùng để so mô hình.

**Gặp lỗi:**

| Triệu chứng | Xem | Hướng xử lý |
|---|---|---|
| Dừng ở bước build | `build.log` | Kaggle đổi bản CUDA: sửa `-DCMAKE_CUDA_ARCHITECTURES=75` hoặc bỏ `--branch b12246` |
| `không tải được GGUF 14B` | log ô chạy | Hugging Face chậm/lỗi mạng: chạy lại |
| Server thoát, `out of memory` | `server0.log` | Giảm ngữ cảnh: sửa `"ctx": 8192` → `6144` trong `configs/models.json` |
| Chỉ có 1 GPU | `nvidia-smi` đầu log | Chưa chọn **T4 x2** |
| Hết giờ mà ít câu | `eval_*.log` | 14B chậm hơn dự kiến: đặt biến `N_PER_SHOP=10` |

## Fine-tune LoRA (`run_finetune.py`)
Một lượt thử: sinh dữ liệu fine-tune từ train + ngân hàng ví dụ (`export.log`, ~10 phút, song song với tải mô hình),
Qwen3-4B + LoRA (thiết kế ở `training_code/README.md`), rồi đo trên đúng 120 câu hold-out của các lần trước.

| GPU | Việc |
|---|---|
| 0 | Fine-tune (~2–3 giờ, 2 epoch) → gộp adapter → GGUF f16 → đo **có** ngân hàng |
| 1 | Trong lúc GPU 0 train: đo **bản gốc** đổi sang GGUF f16 cùng cách (so sánh công bằng) cả hai cấu hình; sau đó đo bản fine-tune **không** ngân hàng |

Biến môi trường (đặt bằng `os.environ[...]` trong ô code): `EPOCHS` (2), `TRAIN_MIN` (300, trần giờ train), `N_PER_SHOP` (20), `KEEP_GGUF` (0; đặt 1 để giữ file GGUF 8 GB trong output).

**Kết quả:**

| File | Nội dung |
|---|---|
| `summary.json` | PA của 4 cấu hình: `base_bank`, `base_nobank`, `ft_bank`, `ft_nobank` (`final` sau ngưỡng, `raw` trước ngưỡng); val loss từng epoch |
| `train.log`, `lora/log_history.json` | loss theo bước, số bước fp16 tràn (`skipped`) |
| `lora/adapter/` | adapter LoRA (~130 MB) để gộp lại khi đóng gói |
| `base_*.json`, `ft_*.json` | từng câu |

**Đọc kết quả:** chỉ số chính là `ft_nobank.raw.PA` so với `base_nobank.raw.PA` (mốc trước: 4B 0,525, 14B 0,642). Ngưỡng `final` hiệu chỉnh cho bản gốc, chưa đúng cho bản fine-tune.

| Triệu chứng | Xem | Hướng xử lý |
|---|---|---|
| `skipped` tăng liên tục, loss `nan` | `train.log` | fp16 tràn: giảm `lr` (sửa `--lr 5e-5` trong script) |
| `CUDA out of memory` khi train | `train.log` | thêm `--max-len 3072` vào lệnh train |
| `sinh dữ liệu fine-tune lỗi` | `export.log` | thiếu `bank.json` hoặc `train/` trong `socai-data` |
| `chuyển GGUF lỗi` | log ô chạy | thiếu thư viện của `convert_hf_to_gguf.py`: dòng lỗi cuối cho biết tên |
| Hết 12 giờ Kaggle | — | đặt `EPOCHS=1` |
