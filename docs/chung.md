# SổCái: phần chung cho mọi phiên bản

> Đề bài, dữ liệu, ràng buộc, hạ tầng và nộp bài. Cách giải từng phiên bản: [v1.md](v1.md), [v2.md](v2.md).
> Cập nhật 12/11/2026. Nhãn: **[ĐCBC]** đã có bằng chứng · **[SL]** suy luận · **[CB]** chưa biết · **[CKC]** cần kiểm chứng.

## 1. Đề bài
**Zalo AI Challenge 2026 · DataMate.** Mỗi lượt agent nhận:
- gói dữ liệu của một shop (CSV/XLSX/SQLite bẩn) kèm README;
- `rulebook.md` và `calendar.csv` (lịch âm) của split đang chạy;
- một câu hỏi tiếng Việt đời thường, có thể không dấu, teencode hoặc gõ Telex sót.

Đầu ra là đáp án đúng format, `unanswerable`, hoặc bỏ trống.

**Chấm điểm (Penalized Accuracy):** đúng +1, sai −1, bỏ trống 0. Vì vậy chỉ trả lời khi P(đúng) > 0,5. Đây là bài toán **trả lời có chọn lọc**, không chỉ là text-to-SQL.

## 2. Ràng buộc và hệ quả thiết kế
| Ràng buộc | Hệ quả |
|---|---|
| Sai −1, bỏ trống 0 | Cần P(đúng) đã hiệu chỉnh theo nguồn đáp án, ngưỡng mặc định 0,6 |
| `unanswerable` chỉ đúng khi đáp án chuẩn là `unanswerable` | Kỳ ngoài dữ liệu (§7) do code kiểm, không để LLM quyết |
| Rulebook riêng từng split | Rulebook là dữ liệu đầu vào; `socai rulebook-diff` để biết luật nào đã đổi |
| Private có ngành chưa xuất hiện | Không chỉ dựa vào lexicon train; LLM phải hiểu được bảng lạ |
| ≤ 15B tham số tổng mọi mô hình (kể cả embedding, OCR) | Một LLM, truy hồi bằng trùng từ, không embedding |
| Offline `--network none`, `/data` chỉ đọc, 6 giờ kể cả nạp mô hình | Ghi từng câu ngay; ngân sách giờ toàn cục; cache ở `/tmp` |
| \|PA nộp − PA BTC chạy lại\| ≤ 0,03 | Greedy, seed 0, không batching, thứ tự cố định, log `thinking` nguyên văn |
| *"No manual intervention, hard-coded answers or logic keyed on question/business IDs"*; cấm huấn luyện trên public/private | Lexicon, ngưỡng, ngân hàng ví dụ, fine-tune **chỉ từ train**. Public chỉ dùng để chạy và nộp lên bảng xếp hạng (5 lượt/ngày) |

**Mốc:** Docker cuối cùng 16/11 23:59 (tối đa 2 lần nộp) · private phát 20/11 09:00 · hạn kết quả 23/11 23:59.

## 3. Dữ liệu [ĐCBC]
- **Train:** 30 shop, 3.000 câu có đáp án. **Public:** 14 shop, 1.000 câu, không có đáp án. Mỗi shop có 12–30 tháng dữ liệu, từ vài trăm đến ~170.000 dòng.
- **Kiểu đáp án (train):** number 1.884 · percent 578 · list 193 · date 145 · string 88 · unanswerable 70 · boolean 23 · table 19.
- **Chia train:** 24 shop dev (phát triển, hiệu chỉnh, ngân hàng ví dụ) và 6 shop hold-out `biz_105, 111, 113, 118, 120, 123` chỉ để đo ([configs/holdout.json](../configs/holdout.json)).

**Các kiểu dữ liệu bẩn:**
| Kiểu | Ví dụ thật |
|---|---|
| Tiền | `86k`, `53.500đ`, `2,760,000.0000`, `1,156tr`, `1tr590` |
| Ngày | serial Excel `45717`, `1 thg 7 2025`, `28/4` (năm lấy theo tên sheet) |
| Bố cục | Dòng tiêu đề trên header, header hai tầng, mỗi tháng một sheet, **dòng tổng chèn giữa sổ**, ô trống theo nhóm đơn (§8) |
| Trạng thái | 30+ cách viết ("đã lấy", "ok", "khách huỷ", "hàng trả về"…) |
| Tên cột | 578 tên khác nhau; SQLite viết dính (`KenhBanHang`, `TinhTrang`) |

**Tín hiệu README:** mô tả cột là từ vựng có kiểm soát (193 cụm), nên ánh xạ theo mô tả bền hơn ánh xạ theo tên cột.

**Câu hỏi:**
- Cách viết: teencode ("dthu", "bnhiu"), Telex sót ("trowr leen", "tawng").
- Tham chiếu tương đối: "tuần rồi", "th 6/26", "cùng kỳ năm trước", Tết.

**Rulebook train → public:** 70 → 92 thẻ luật (thêm 25, bỏ 3, sửa ~3), có 3 ngành mới (điện thoại, sửa xe, mẹ và bé). Luật bị bỏ ở public gồm "giá bán X ngày D", "tồn theo đơn vị lớn", "giá nhập bình quân". Áp lặng lẽ luật train cho câu public sẽ ra đáp án sai.

## 4. Nguyên tắc dùng cho mọi phiên bản
1. **Trả lời theo kỳ vọng điểm:** p của mỗi ứng viên là precision đo được của nguồn đó trên train (leave-business-out). Hai đường cùng đáp án thì p tăng.
2. **LLM không tự tính ngày, không tự đọc tiền:** `rules/timeexpr.py` và `io/cells.py` là code tất định.
3. **Không bỏ qua lặng lẽ:** ô tiền không đọc được trong kỳ hỏi thì từ chối (`Unsupported`), không cộng thiếu. Ba lỗi gốc lớn nhất đều thuộc dạng này: `1,156tr` bị đọc thành tỷ, `1tr590` thành NaN, dòng "TỔNG NGÀY" bị gán vào đơn. Cả ba có test hồi quy.

## 5. Phần tất định dùng chung (L0–L3)
| Tầng | File |
|---|---|
| Đọc CSV/XLSX/SQLite, phát hiện header, gộp sheet theo tháng, đọc ô bẩn | `io/readers.py`, `io/cells.py` |
| README → mô tả cột; gán nghĩa cột (mô tả → tên → suy từ dữ liệu → LLM); vai trò bảng | `profile/`, `semantic/lexicon.py` |
| Bảng chuẩn dòng hàng/đơn: bỏ dòng tổng, điền ô nhóm, khử trùng, trạng thái §1 | `semantic/canonical.py` |
| Thời gian, lịch âm, thẻ luật và `rulebook-diff` | `rules/` |
| Chính sách trả lời, hiệu chỉnh | `answer/decide.py`, `eval/`, `calibration.json` |
| Runner, log, ngân sách giờ | `runner.py`, `agent.py`, `predict.sh` |

## 6. Mô hình và hạ tầng
| Mô hình | Tham số | Chạy ở đâu | Tốc độ đo được |
|---|---|---|---|
| Qwen3-4B Q4_K_M | 4,02B | Laptop RTX 3050 Ti 4 GB: ctx 4096, `-b 512 -ub 256`, KV q8_0. Tràn VRAM ở prompt ~3.800 token nên giữ prompt ≤ 3K | 2,5–4 giây/câu |
| Qwen3-14B Q4_K_M | 14,77B | Kaggle T4 16 GB, ctx 8192 | 10,5 giây/câu (2 lần gọi) |

- Hồ sơ mô hình nằm trong [configs/models.json](../configs/models.json). Đổi mô hình bằng `SOCAI_MODEL=<tên>`. Server khởi động bằng `python -m socai.llm.serve`.
- Trên Windows, VRAM tràn sang RAM không báo lỗi mà làm chậm 5–10 lần.
- Kaggle (hướng dẫn: [kaggle/README.md](../kaggle/README.md)):
  - phải dựng llama.cpp từ nguồn với `GGML_CUDA_NO_VMM=ON` và libcuda stubs, mất ~30 phút;
  - bản dựng được lưu trong output để lần sau dùng lại.
- **Trước khi khai báo mô hình trong README nộp bài [CKC]:** ngày phát hành gốc trước 26/10/2026, license, số tham số chính xác, revision cố định.

**Thời gian:** ngân sách chạy 5,5 giờ. Với ~500 câu private thì được ≈ 39 giây/câu. Khi còn dưới 15 phút, hệ thống tắt LLM và chỉ chạy đường luật.

**Tái lập:**
- vLLM với `--seed 0`, temperature 0, `--max-num-seqs 1`, `--enable-prefix-caching`; `PYTHONHASHSEED=0`.
- Đường luật đã có test "hai lần chạy giống từng byte".
- Đường LLM trên GPU thật **[CKC]**.

**Log** (`/result/logs/log.jsonl`, một dòng mỗi câu):
- trường bắt buộc: `id`, `answer`, `elapsed_ms`, `thinking` (nguyên văn, nhiều lần gọi thì là list), `code`, `inputs`, `output`;
- thêm `time_spec`, `decision` (ứng viên, p̂, ngưỡng), `notes` (lý do từ chối).

**Image** (`docker/Dockerfile`):
- Bố cục: `/code/{predict.sh, README.md, requirements.txt, src/socai/, models/llm/, training_code/}`.
- Kiểm bằng đúng lệnh của BTC (`--network none --shm-size=16g`, `/data:ro`).
- Image 14B ~50 GB, giới hạn 100 GB.
- **Chưa build và chạy thử trên Linux có GPU [CKC].**

## 7. Sau cuộc thi (Track B, không ghi điểm thi)
Kế hoạch "bộ não + nhà công cụ" cho sản phẩm doanh nghiệp, làm sau 23/11:
- Nạp thêm loại nguồn: PDF/DOCX/XML hoá đơn điện tử/JSON (`io/docs.py` đã có khung), OCR ảnh và sổ ghi tay, VLM cho hoá đơn bán lẻ, ASR cho câu hỏi bằng giọng.
- Đa ngữ: Anh, Trung, Hàn, câu trộn ngôn ngữ.
- Hỏi lại người dùng thay vì bỏ trống; mức quyền (tự làm / báo / chờ duyệt / cấm); trí nhớ dài hạn có duyệt và có phiên bản.
- Kiểm chứng không vòng tròn:
  - shop tổng hợp có sự thật (`socai synth`);
  - test metamorphic (cùng câu ở nhiều cách viết phải ra cùng đáp án);
  - nhiễu loạn rulebook.

Không đưa thành phần nào ở đây vào image thi: chúng tốn ngân sách tham số mà không ghi điểm. Bản kế hoạch đầy đủ và bản đồ tương tác nằm ở commit `ae8ac6a` (`docs/ke-hoach-bo-nao.md`, `docs/ban-do-bo-nao.html`).
