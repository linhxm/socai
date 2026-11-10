# SổCái: agent hỏi đáp sổ sách tiếng Việt

Bài giải cho **Zalo AI Challenge 2026 · DataMate**: trả lời câu hỏi tiếng Việt đời thường (có cả không dấu, teencode, gõ Telex sót) trên dữ liệu CSV/XLSX/SQLite bẩn của cửa hàng nhỏ.

Điểm chấm là Penalized Accuracy: đúng +1, sai −1, bỏ trống 0. Vì vậy hệ thống **chỉ trả lời khi đủ chắc**.

## Cách làm, tóm tắt
| Tầng | Ai làm | Ở đâu |
|---|---|---|
| Đọc file bẩn (tiền `86k`/`1tr590`, ngày serial Excel, header nhiều tầng, dòng tổng, sheet theo tháng) | Code tất định | `io/` |
| Gán nghĩa cột (README → tên cột → **suy từ dữ liệu** → **Qwen3-4B**, có kiểm lại bằng dữ liệu) | Code + LLM nhỏ | `profile/`, `semantic/lexicon.py` |
| Bảng chuẩn dòng hàng / đơn theo rulebook §1, §8 | Code | `semantic/canonical.py` |
| Thời gian ("tuần rồi", "2/26", Tết âm lịch, "cùng kỳ năm trước") | Code | `rules/timeexpr.py` |
| Hiểu câu hỏi: đường luật nhanh + **Qwen3-4B tách câu thành kế hoạch JSON** | Luật + LLM nhỏ | `nlq/fastpath.py`, `nlq/plan.py` |
| Neo tên trong câu vào giá trị thật trong dữ liệu | Code | `semantic/entities.py` |
| Lệnh cơ bản: lọc, nhóm, top, tỉ trọng, so kỳ, chuỗi kỳ | Code | `exec/query.py`, `exec/metrics.py` |
| Quyết định trả lời / bỏ trống | Code | `answer/decide.py` |

LLM **không tự tính toán**. Trên laptop, Qwen3-4B tự viết code pandas đạt 0/23 câu, nhưng tách câu thành kế hoạch thì tốt; chi tiết ở [docs/thiet-ke-gon.md](docs/thiet-ke-gon.md).

## Kết quả đo được
Đo trên **hold-out 6 shop train** (631 câu), là các shop không dùng để phát triển phần mới ([configs/holdout.json](configs/holdout.json)):

| Cấu hình | Đúng | Sai | Bỏ trống | PA |
|---|---|---|---|---|
| Chỉ đường luật | 142 | 2 | 487 | 0,222 |
| Luật + kế hoạch Qwen3-4B (bản 10/11) | 239 | 66 | 326 | 0,274 |

Hai giới hạn của bảng trên:
- Số này đo **trước** đợt sửa dựng bảng chuẩn cho sổ không mã đơn và SQLite/nguồn hỗn hợp; sẽ đo lại.
- Đường luật được chỉnh khi đã nhìn lỗi cả 30 shop train, nên riêng số của nó vẫn lạc quan.

## Chạy
```bash
pip install -e ".[dev]"
pytest -q                                   # 70 test; test dữ liệu thật tự bỏ qua nếu không có ../train

# chỉ đường luật (không cần GPU)
python -m socai eval --split ../train --out runs/train --no-llm

# có LLM: khởi động server OpenAI-compatible (llama.cpp hoặc vLLM) chạy Qwen3-4B rồi trỏ tới nó
# ví dụ llama.cpp, GPU 4 GB:  llama-server -m Qwen3-4B-Q4_K_M.gguf -ngl 99 -fa on -c 4096 -ctk q8_0 -ctv q8_0 --jinja --port 8088 -np 1 --seed 0
SOCAI_LLM_URL=http://127.0.0.1:8088/v1 python -m socai predict --data ../public_test --out runs/public

# tiện ích
python -m socai profile --biz ../train/data/biz_101
python -m socai rulebook-diff ../train/rulebook.md ../public_test/rulebook.md
python -m socai synth --out /tmp/synth --n-biz 3 --render-images
```
Nộp bài: Docker theo hướng dẫn của BTC, xem `docker/Dockerfile`, `predict.sh` và [docs/ky-thuat.md](docs/ky-thuat.md).

## Tài liệu
- [docs/idea.md](docs/idea.md): phân tích đề, dữ liệu, kiến trúc ban đầu, số đo đường luật
- [docs/thiet-ke-gon.md](docs/thiet-ke-gon.md): bản gọn dùng Qwen3-4B
- [docs/ke-hoach-bo-nao.md](docs/ke-hoach-bo-nao.md) + [docs/ban-do-bo-nao.html](docs/ban-do-bo-nao.html): kế hoạch bộ não + nhà công cụ, bản đồ tương tác
- [docs/ky-thuat.md](docs/ky-thuat.md): mô hình, thời gian, tái lập, log, đóng gói

## Không có trong repo
- Dữ liệu cuộc thi (train/public/private), vì quy định của BTC.
- Trọng số mô hình: tải Qwen3-4B từ nguồn chính thức và ghi revision trong README nộp bài.

## Khai báo mô hình (điền trước khi nộp)
| Mô hình | Nguồn | Revision | Ngày phát hành | License | Tham số |
|---|---|---|---|---|---|
| Qwen3-4B (dự kiến) | huggingface.co/Qwen | _cần ghi_ | _cần kiểm_ | _cần kiểm_ | ~4,0B |
