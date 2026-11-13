# training_code — SOCAI

Theo mục "README and training code" của hướng dẫn đóng gói.

## Trạng thái hiện tại (cập nhật trước mỗi lần nộp)
- **Không fine-tune mô hình** trong bản hiện tại. Mô hình dùng nguyên bản công khai (xem `/code/README.md`).
- **Có sinh dữ liệu / tham số từ dữ liệu train**, cần khai báo:
  1. `src/socai/semantic/lexicon.py`: từ vựng "mô tả cột → trường chuẩn", dựng thủ công từ README của
     **tập train** (không dùng README public/private — Rule 6).
  2. `src/socai/calibration.json`: xác suất đúng theo (nguồn, metric) đo trên **train** theo
     leave-business-out, sinh bằng lệnh dưới.
  3. Dữ liệu tổng hợp `socai synth` (seed cố định) — dùng cho test; nếu sau này dùng để fine-tune thì
     bản dữ liệu đã dùng phải nằm trong `data/` kèm checksum.

  4. `src/socai/v2/bank.json`: ngân hàng ví dụ (câu hỏi train → biểu thức) dùng để truy hồi lúc suy luận.
     Sinh bằng `build_bank.py`: Qwen3-4B (cùng mô hình lúc suy luận) viết biểu thức cho câu **train**, chỉ giữ
     biểu thức chạy ra đúng đáp án chuẩn của train. Không dùng dữ liệu public/private. 6 shop hold-out
     (`configs/holdout.json`) không có trong ngân hàng dùng để đo; trước khi nộp có thể dựng lại trên cả 30 shop.

## Lệnh tái lập
```bash
pip install -r training_code/requirements.txt
# (1) hiệu chỉnh chính sách trả lời trên train (CPU, ~4 phút)
python -m socai eval --split /path/train --out runs/train --no-llm --calibrate src/socai/calibration.json
# (2) dữ liệu tổng hợp (seed cố định)
python -m socai synth --out training_code/data/synth --n-biz 6 --seed 0
# (3) manifest
python training_code/data_generation/make_manifest.py training_code/data > training_code/data_manifest.json
```

## Dựng ngân hàng ví dụ (cần server LLM, ~2 giờ vòng đầu trên laptop 4 GB)
```bash
PYTHONPATH=src python training_code/build_bank.py --train /path/train --out src/socai/v2/bank.json     --exclude-biz biz_105,biz_111,biz_113,biz_118,biz_120,biz_123      # chạy lại 2–3 lần: vòng sau nhặt thêm câu
```

## Fine-tune LoRA (lượt thử đầu: `kaggle/run_finetune.py`)
Mục tiêu: Qwen3-4B viết biểu thức đúng cả khi ngân hàng không có ví dụ cùng kiểu (câu kiểu mới ở private).

**Dữ liệu** (`export_finetune.py`, chỉ từ train; 6 shop hold-out không có trong ngân hàng nên không bao giờ vào đây):
- mỗi mẫu = hội thoại đúng như lúc suy luận → đích là biểu thức đã chạy ra đúng đáp án chuẩn;
- tối đa 20 mẫu mỗi khung biểu thức (ngân hàng lệch: khung phổ biến nhất 203/2.090 mẫu), xoay vòng theo shop;
- ba kiểu prompt gán cố định theo id: `bank` 40% (ví dụ shop khác, như lúc chạy), `novel` 30% (bỏ mọi ví dụ cùng
  khung với đáp án), `static` 30% (chỉ ví dụ viết tay);
- đích chuẩn hoá bằng `lint` + `$ky_so` ngoài vế so sánh → `$ky`; val = 2 shop (biz_109, biz_124).

**Huấn luyện** (`finetune_lora.py`): base fp16 + LoRA r16 (mọi lớp tuyến tính) fp32, GradScaler (T4 không có bf16),
loss chỉ trên token biểu thức, 2 epoch, lr 1e-4 cosine, tích luỹ 16, seed 0.
```bash
PYTHONPATH=src python training_code/export_finetune.py --train /path/train --bank src/socai/v2/bank.json     --out training_code/data/finetune --val-biz biz_109,biz_124
python training_code/finetune_lora.py --data training_code/data/finetune --out runs/lora --revision <REV>   # GPU ≥ 15 GB
python training_code/finetune_lora.py --merge runs/lora --out models/llm --revision <REV>                    # gộp adapter
```
Khi chạy thật phải lưu: revision mô hình gốc, `run_config.json`, `log_history.json`, checksum dữ liệu
(`data/finetune/manifest.json`). Chỉ thay mô hình trong image khi PA hold-out hơn bản gốc.
