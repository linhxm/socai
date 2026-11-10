# training_code — SổCái

Theo mục 8 "README and training code" của hướng dẫn Docker DataMate.

## Trạng thái hiện tại (cập nhật trước mỗi lần nộp)
- **Không fine-tune mô hình** trong bản hiện tại. Mô hình dùng nguyên bản công khai (xem `/code/README.md`).
- **Có sinh dữ liệu / tham số từ dữ liệu train**, cần khai báo:
  1. `src/socai/semantic/lexicon.py`: từ vựng "mô tả cột → trường chuẩn", dựng thủ công từ README của
     **tập train** (không dùng README public/private — Rule 6).
  2. `src/socai/calibration.json`: xác suất đúng theo (nguồn, metric) đo trên **train** theo
     leave-business-out, sinh bằng lệnh dưới.
  3. Dữ liệu tổng hợp `socai synth` (seed cố định) — dùng cho test; nếu sau này dùng để fine-tune thì
     bản dữ liệu đã dùng phải nằm trong `data/` kèm checksum.

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

## Nếu fine-tune (kế hoạch, chưa làm)
Fine-tune LoRA cho bước sinh code từ (câu hỏi, hồ sơ shop) → `answer(ctx)` trên **train + dữ liệu tổng
hợp**; đánh giá leave-business-out; gộp adapter vào trọng số gốc trước khi đóng gói (mục 4 hướng dẫn).
Phải lưu: config, seed, log, checkpoint xác minh, prompt/response nếu dữ liệu được sinh bằng mô hình.
