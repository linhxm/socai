# SổCái — Kỹ thuật, hạ tầng, đóng gói

> Đi kèm [idea.md](idea.md). Ở đây là các con số và cấu hình để build, chạy và nộp; mỗi mục ghi rõ chỗ nào đã kiểm, chỗ nào chưa.
>
> **Cập nhật 10/11:** hướng hiện tại dùng **Qwen3-4B làm bộ tách câu + gán nghĩa cột**, tính toán bằng code tất định ([thiet-ke-gon.md](thiet-ke-gon.md)). Phần chọn mô hình 14B dưới đây giữ lại để so sánh. Bài học vận hành đo được: trên Windows, VRAM tràn sang RAM không báo lỗi mà làm chậm 5–10 lần; với 4 GB VRAM dùng ngữ cảnh 4K + KV cache 8-bit.

## 1. Chọn mô hình trong ngân sách 15B

Tổng tham số của **mọi** mô hình dùng khi suy luận phải ≤ 15B, tính cả MoE, embedding, reranker và OCR. Quantization không làm giảm số tham số được tính.

| Phương án | Tham số | Ưu | Nhược |
|---|---|---|---|
| **A. Một LLM dense ~14B** (ví dụ họ Qwen3-14B; BTC đã thử khởi tạo mất khoảng 2 phút trên máy chấm) | ~14,8B | Mạnh nhất cho viết code; BTC đã chứng minh chạy được trên 2×RTX 5090 | Gần như hết ngân sách: không còn chỗ cho embedding |
| B. LLM ~8B + mô hình phụ | ~8B + ≤ 7B | Có chỗ cho reranker hoặc VLM nhỏ | Viết code kém hơn **[SL]** |
| C. LLM coder ~14B | ~14B | Chuyên code | Hiểu tiếng Việt đời thường kém hơn **[SL]** |

**Khuyến nghị: A.**
- Truy hồi trong hệ thống này (thẻ luật, ví dụ) dùng trùng từ, nên không cần embedding.
- Chọn dứt khoát giữa A và C bằng precision `llm_code` trên hold-out ngày 11/11.

**[CKC] cho bất kỳ mô hình nào trước khi khai báo trong README:**
- ngày phát hành gốc trước 26/10/2026;
- license cho phép sử dụng;
- số tham số chính xác;
- revision cố định.

## 2. Thời gian

| Thành phần | Ước lượng | Nguồn |
|---|---|---|
| Nạp mô hình | ~2 phút | Hướng dẫn BTC (Qwen3-14B: 128 giây) |
| Đọc và chuẩn hoá dữ liệu | 0,1–14 giây/shop | **Đo được** trên 44 shop, máy dev, có calamine |
| Đường nhanh | ~5–10 ms/câu | **Đo được** (log `elapsed_ms`) |
| Đường LLM | 1–2 lần gọi × (prefill nhờ prefix cache + 300–800 token sinh) | **[CB]**, cần đo trên RTX 5090 |

**Ngân sách:**
- Đặt 5,5 giờ cho phần chạy, chừa 30 phút.
- Với khoảng 500 câu private (công thức kiểm tra của BTC: thời gian nạp + TB/câu × 500), ngân sách là ≈ **39 giây/câu**. Dư dả nếu mỗi câu gọi LLM ≤ 2 lần **[SL]**.
- Còn dưới `llm_reserve_s` (15 phút) thì tự tắt LLM, chỉ chạy đường nhanh.

## 3. Tái lập (Rule 8)

- **vLLM:**
  - `--seed 0`, `temperature 0`;
  - `--max-num-seqs 1`: không batching, vì BTC cảnh báo batching GPU có thể làm lệch đầu ra kể cả khi greedy;
  - `--enable-prefix-caching`.
- **Thứ tự cố định:** theo shop, rồi theo thứ tự trong `questions.jsonl`. `PYTHONHASHSEED=0`.
- **Kiểm trước khi nộp:** chạy image 2 lần trên train, so `submission.jsonl` và PA. Đường luật đã có test "hai lần chạy giống hệt từng byte" (`tests/test_end_to_end.py`); đường LLM **[CKC]**.

## 4. Log (mục 3.3 hướng dẫn)

Một dòng JSON mỗi câu trong `/result/logs/log.jsonl`:

| Trường | Cách điền |
|---|---|
| `id`, `answer`, `elapsed_ms` | Bắt buộc. `answer` là object `{type, answer}` |
| `thinking` | `reasoning_content` nguyên văn của vLLM; nhiều lần gọi thì là list theo thứ tự; không gọi LLM thì `null` |
| `code` | Đường nhanh: `"socai/exec/metrics.py::revenue"` + `inputs`. Đường LLM: code đã chạy; nhiều lần sửa thì là list |
| `output` | Giá trị trả về thật |
| Thêm | `time_spec` (kỳ đã giải), `decision` (các ứng viên, p̂, ngưỡng), `notes` (lý do từ chối). Dùng để giải trình khi BTC hỏi |

Ngoài ra có `run_summary.json` và `vllm.log`.

## 5. Bố cục image

```
/code/
├── predict.sh            # khởi động vLLM (nếu có GPU + mô hình) → python -m socai predict
├── README.md             # cách làm, khai báo mô hình (tên, nguồn, revision, ngày phát hành, license, tham số)
├── requirements.txt
├── src/socai/            # mã suy luận + calibration.json
├── models/llm/           # trọng số + tokenizer + config (đầy đủ, offline)
└── training_code/        # README, requirements, data_generation/, data/, logs/, data_manifest.json
```

**Build và kiểm** theo mục 9–11 của hướng dẫn: `docker/Dockerfile`, lệnh chạy thử giống hệt lệnh của BTC (`--network none --shm-size=16g`, `/data:ro`).

**Dung lượng:**
- Image mẫu 14B của BTC khoảng 50,6 GB khi chưa nén.
- Giới hạn là 100 GB.
- Cần chừa thêm khoảng 50 GB đĩa lúc export.

## 6. Cấu trúc mã nguồn

```
src/socai/
├── io/        cells.py (đọc ô bẩn) · readers.py (CSV/XLSX/SQLite) · docs.py (PDF/DOCX/XML/JSON/ảnh-OCR)
├── profile/   readme.py · business.py (gán nghĩa cột, vai trò bảng)
├── semantic/  lexicon.py (chỉ từ train) · canonical.py (lines/orders theo §1, §8)
├── rules/     timeexpr.py · calendar.py · rulebook.py (thẻ luật, diff)
├── nlq/       normalize.py · ir.py · fastpath.py · prompts.py
├── exec/      metrics.py · api.py (ctx cho code sinh ra) · sandbox.py (worker có timeout)
├── answer/    decide.py (format + chính sách p̂)
├── llm/       client.py (vLLM OpenAI-compatible, FakeLLM)
├── eval/      scorer.py (mô phỏng luật chấm) · evaluate.py (LBO + hiệu chỉnh)
├── synth/     world.py (shop tổng hợp, đa định dạng, đa ngữ, ảnh sổ)
├── agent.py · runner.py · __main__.py
```

## 7. Cách mở rộng

| Muốn thêm | Làm ở đâu | Bất biến phải giữ |
|---|---|---|
| Định dạng tài liệu mới | `register_reader(".ext", fn)` trả `list[RawTable]` có `source_kind`, `confidence` | Không ném lỗi làm hỏng cả shop; thiếu thư viện thì `NotAvailable` |
| OCR engine | Lớp có `name`, `read(path) -> list[Box]`; `register_ocr(engine)` | `conf` theo từng hộp chữ để chính sách hạ p̂ |
| Metric mới | Hàm trong `exec/metrics.py` trả `Result`, docstring trích § rulebook; thêm ý định vào `fastpath.py` | Ô không đọc được ⇒ `Unsupported`; kỳ ngoài dữ liệu ⇒ `unanswerable` |
| Ngôn ngữ mới | `normalize.detect_lang`, từ khoá trong `fastpath.py`, mẫu trong `timeexpr.py`, mẫu câu trong `synth/world.py` | Test metamorphic: cùng đáp án với câu tiếng Việt |
| Luật ngành mới | Không mã hoá cứng: rulebook đi qua prompt; chỉ thêm metric khi luật xuất hiện ở ≥ 2 split | Ghi § trong docstring |
