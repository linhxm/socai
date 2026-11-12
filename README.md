# SổCái: agent hỏi đáp sổ sách tiếng Việt

Bài giải cho **Zalo AI Challenge 2026 · DataMate**: trả lời câu hỏi tiếng Việt đời thường (có cả không dấu, teencode, gõ Telex sót) trên dữ liệu CSV/XLSX/SQLite bẩn của cửa hàng nhỏ.

Điểm chấm là Penalized Accuracy: đúng +1, sai −1, bỏ trống 0. Vì vậy hệ thống **chỉ trả lời khi đủ chắc**.

## Hai phiên bản
| | v1 (10–11/11) | **v2 (11/11 → nay, mặc định)** |
|---|---|---|
| Hiểu câu hỏi | Regex 8 ý định + Qwen3-4B điền kế hoạch JSON từ danh sách loại chỉ số | Qwen3-4B/14B ghép **biểu thức đại số** (bảng → lọc → kỳ → nhóm → phép gộp, ghép bằng pct/diff/ratio/except…) |
| Luật rulebook | Trong từng metric viết tay | Trong **định nghĩa bảng** (bộ lọc mặc định, khoá mặt hàng/khách, công nợ theo ngày) |
| Ví dụ cho LLM | Cố định | Truy hồi 8 ví dụ **đã kiểm chứng** từ train + ý kiến thứ hai |
| PA hold-out | 0,349 | **0,715** |
| Tài liệu | [docs/v1.md](docs/v1.md) | [docs/v2.md](docs/v2.md) |

Phần dùng chung cho cả hai phiên bản nằm ở [docs/chung.md](docs/chung.md): đọc file bẩn, thời gian, bảng chuẩn, chính sách trả lời, hạ tầng, nộp bài. Ở cả hai phiên bản, LLM **không tự tính toán**.

## Kết quả đo được
Đo trên **hold-out 6 shop train** (631 câu), là các shop không dùng để phát triển phần mới ([configs/holdout.json](configs/holdout.json)):

| Cấu hình | Đúng | Sai | Bỏ trống | PA |
|---|---|---|---|---|
| Chỉ đường luật | 142 | 2 | 487 | 0,222 |
| Luật + kế hoạch Qwen3-4B (bản 10/11) | 239 | 66 | 326 | 0,274 |
| Luật + kế hoạch + bảng chuẩn mới + ngưỡng hiệu chỉnh (bản 11/11) | 289 | 69 | 273 | 0,349 |
| v2: luật + đại số truy vấn | 380 | 82 | 169 | 0,472 |
| **v2 + ví dụ đã kiểm chứng truy hồi từ train + ý kiến thứ hai** (12/11) | 513 | 62 | 56 | **0,715** |

Qwen3-14B (Kaggle T4, 120 câu hold-out): PA 0,808 so với 0,775 của 4B khi có ngân hàng ví dụ; 0,642 so với 0,525 khi không có ([docs/v2.md](docs/v2.md) §4).

Giới hạn của bảng trên:
- Ngưỡng hiệu chỉnh tính trên 24 shop dev, không dùng hold-out. Nhưng đợt sửa sổ không mã đơn và SQLite được làm sau khi thấy biz_113, biz_123 bị 0 điểm ở lần đo 10/11, nên hold-out không còn "sạch" hoàn toàn.
- Đường luật được chỉnh khi đã nhìn lỗi cả 30 shop train, nên riêng số của nó vẫn lạc quan.

## Chạy
```bash
pip install -e ".[dev]"
pytest -q                                   # 79 test; test dữ liệu thật tự bỏ qua nếu không có ../train

# chỉ đường luật (không cần GPU)
python -m socai eval --split ../train --out runs/train --no-llm

# có LLM: khởi động server OpenAI-compatible (llama.cpp hoặc vLLM) chạy Qwen3-4B rồi trỏ tới nó
# ví dụ llama.cpp, GPU 4 GB:  llama-server -m Qwen3-4B-Q4_K_M.gguf -ngl 99 -fa on -c 4096 -b 512 -ub 256 -ctk q8_0 -ctv q8_0 --jinja --port 8088 -np 1 --seed 0
SOCAI_LLM_URL=http://127.0.0.1:8088/v1 python -m socai predict --data ../public_test --out runs/public

# tiện ích
python -m socai profile --biz ../train/data/biz_101
python -m socai rulebook-diff ../train/rulebook.md ../public_test/rulebook.md
python -m socai synth --out /tmp/synth --n-biz 3 --render-images
```
Nộp bài: Docker theo hướng dẫn của BTC, xem `docker/Dockerfile`, `predict.sh` và [docs/chung.md](docs/chung.md) §6.

## Tài liệu
- [docs/chung.md](docs/chung.md): đề bài, ràng buộc, dữ liệu, phần tất định dùng chung, mô hình và hạ tầng, nộp bài, Track B
- [docs/v1.md](docs/v1.md): đường luật + kế hoạch 4B, số đo và giới hạn dẫn tới v2
- [docs/v2.md](docs/v2.md): đại số truy vấn, ngân hàng ví dụ, kết quả 4B/14B, việc tiếp theo
- [kaggle/README.md](kaggle/README.md): chạy 14B trên Kaggle · [training_code/README.md](training_code/README.md): ngân hàng ví dụ, fine-tune

## Không có trong repo
- Dữ liệu cuộc thi (train/public/private), vì quy định của BTC.
- Trọng số mô hình: tải Qwen3-4B từ nguồn chính thức và ghi revision trong README nộp bài.

## Khai báo mô hình (điền trước khi nộp)
| Mô hình | Nguồn | Revision | Ngày phát hành | License | Tham số |
|---|---|---|---|---|---|
| Qwen3-4B (dự kiến) | huggingface.co/Qwen | _cần ghi_ | _cần kiểm_ | _cần kiểm_ | ~4,0B |
