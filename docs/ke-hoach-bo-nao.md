# Kế hoạch: Bộ não SổCái và Nhà công cụ

> **Bản nháp để góp ý**, 10/11/2026. Đi kèm bản đồ tương tác: mở [ban-do-bo-nao.html](ban-do-bo-nao.html) bằng trình duyệt.
>
> Nhãn:
> - **[ĐCBC]** đã có bằng chứng
> - **[SL]** suy luận
> - **[CB]** chưa biết
> - **[CKC]** cần kiểm chứng
>
> Ký hiệu trạng thái button:
> - ✅ có code + test
> - 🟡 có một phần / khung
> - ⬇️ cần tải công cụ ngoài về thử
> - ⬜ phải viết mới

---

## 0. Một ràng buộc phải đặt lên đầu: hai chế độ chạy

Cùng một bộ não, cùng một nhà công cụ, nhưng **mỗi chế độ bật/tắt các button khác nhau**:

| | **Chế độ Thi** (Zalo AI DataMate) | **Chế độ Nền tảng** (sản phẩm cho doanh nghiệp) |
|---|---|---|
| Mạng | **Cấm** (chạy `--network none`) | Được, có kiểm soát |
| Tìm trên mạng, GitHub | Tắt | Bật, **cần người duyệt** |
| Cập nhật trí nhớ | **Đóng băng**: chỉ học trước từ train, không học từ test (Rule 6) | Học liên tục, mục mới phải qua kiểm chứng/duyệt |
| Con người can thiệp | **Cấm** (Rule 4) | Có: duyệt, ra lệnh, sửa |
| Mô hình | Tổng ≤ 15B tham số, **tính cả OCR, ASR, embedding** | Không giới hạn cứng |
| Loại dữ liệu | Chỉ CSV/XLSX/SQLite + câu hỏi chữ | Mọi loại: ảnh, PDF scan, âm thanh, web… |
| Hạn | Docker **16/11**, private 20–23/11 | Liên tục |

**Hệ quả quan trọng:** OCR, âm thanh, web, GitHub **không ghi điểm thi**. Đưa chúng vào image thi chỉ tốn ngân sách tham số và thêm rủi ro. Vì vậy kế hoạch chia hai luồng song song: **Luồng A** (bộ não tối thiểu cho thi) và **Luồng B** (nhà công cụ đầy đủ). Mọi button đều mang nhãn "có ghi điểm thi không".

Đề thi **không yêu cầu GitHub**: bài nộp là Docker image cộng link tải và md5. GitHub chỉ là button tuỳ chọn ở chế độ nền tảng (lưu code, theo dõi việc).

---

## 1. Cây vấn đề: từ bao quát đến từng nhánh

```
Chủ shop hỏi → hệ thống trả lời đúng, hoặc biết im lặng/hỏi lại
├── A. Hiểu NGƯỜI HỎI nói gì
│   ├── A1 Ngôn ngữ: Việt có dấu / không dấu · Anh · Hàn · Trung · Nhật · TRỘN trong một câu
│   ├── A2 Cách viết: viết tắt (dt, sl, ck) · teencode (bnhiu, khum, j) · Telex sót (tawng, nhoms)
│   │        · từ mượn (order, ship, sale) · tiếng lóng theo thời đại/ngành · từ mới mỗi năm
│   ├── A3 Thời gian: tương đối (tuần rồi, 2/26, bữa qua) · âm lịch (Tết) · ca · kỳ kế toán
│   ├── A4 Ý định: chỉ số gì · lọc gì · nhóm theo gì · top mấy · so với kỳ nào
│   └── A5 Mơ hồ / thiếu thông tin → hỏi lại (nền tảng) hoặc bỏ trống (thi)
├── B. Hiểu DỮ LIỆU nói gì
│   ├── B1 Loại nguồn (mục 3.1)
│   ├── B2 Loại bảng (mục 3.2)
│   ├── B3 Ô bẩn: tiền, ngày, số lượng + đơn vị, trạng thái, tên không thống nhất
│   ├── B4 Nghĩa cột · khoá nối giữa bảng · độ tin cậy nguồn (OCR, ASR)
│   └── B5 Ngành lạ chưa từng thấy (sửa xe, điện thoại, mẹ và bé…)
├── C. Suy luận và tính
│   ├── C1 Luật nghiệp vụ (rulebook đổi theo split)
│   ├── C2 Lập kế hoạch truy vấn: select · where · group · order/top · join · so sánh kỳ · tỉ trọng
│   ├── C3 Thực thi: SQL (DuckDB) · Python sandbox · metric đã kiểm
│   └── C4 Kiểm chứng: hai đường độc lập · bất biến (tổng dòng = tổng đơn) · hợp lý (đơn vị, độ lớn)
├── D. Quyết định và trình bày: đúng format · unanswerable · bỏ trống · giải trình nguồn gốc
├── E. Tri thức và trí nhớ
│   ├── E1 Ngắn hạn (trong phiên) · E2 Dài hạn (từ điển, ánh xạ cột, kỹ năng truy vấn, lỗi đã gặp)
│   ├── E3 Tri thức ngoài: rulebook, tài liệu, web
│   └── E4 Tối ưu bộ nhớ: CAG cho phần cố định, RAG cho phần lớn, nén, hết hạn
└── F. Điều hành: quyền (người duyệt gì) · đa nhiệm · ngân sách thời gian · tái lập · nhật ký
```

---

## 2. Kiến trúc: Bộ não + Nhà công cụ

### 2.1 Bộ não
- **LLM lập kế hoạch:** Qwen3-4B trên laptop, có thể đổi sang 14B.
- **Bộ định tuyến:** đọc **thẻ công cụ** để biết button nào làm gì.
- **Trí nhớ làm việc:** sổ nháp trong phiên.
- **Bộ phản biện:** kiểm trước khi trả lời.
- **Vòng lặp:** Hiểu → Lập kế hoạch → Bấm button → Xem kết quả → Kiểm chứng → Trả lời / Bỏ trống / Hỏi người.

**"Hiểu nghĩa để gọi đúng công cụ":** mỗi button có **thẻ công cụ** gồm:
- tên;
- nó giải vấn đề nào (bằng tiếng Việt thường);
- đầu vào/đầu ra;
- chi phí;
- bật ở chế độ nào;
- mức quyền;
- ví dụ khi nào nên dùng.

Bộ não **tìm thẻ phù hợp bằng truy hồi** (RAG trên thẻ công cụ), không nhồi hết mọi thẻ vào prompt. Nhờ vậy nhà công cụ lớn lên mà prompt không phình.

### 2.2 Nhà công cụ (các phòng và button)

| Phòng | Button | Việc | Trạng thái | Ghi điểm thi |
|---|---|---|---|---|
| **Ngôn ngữ** | `lang.detect` | Nhận ngôn ngữ **từng đoạn** trong câu trộn Việt–Anh–Hàn | 🟡 (theo bảng chữ) + ⬇️ fastText lid.176 hoặc lingua | Có (ít) |
| | `lang.normalize` | Viết tắt/teencode/Telex → dạng chuẩn, **dùng từ điển trong trí nhớ** | 🟡 | **Có** |
| | `lang.learn` | **Tự viết luật từ vựng** (mục 4) | ⬜ | Có (học trước từ train) |
| | `lang.translate` | Dịch đoạn Anh/Hàn/Trung sang tiếng Việt chuẩn | ⬜ (LLM) | Ít |
| | `lang.diacritics` | Phục hồi dấu câu không dấu | ⬇️ thử mô hình nhỏ hoặc dùng LLM | Có |
| **Thời gian** | `time.resolve` | "tuần rồi", "2/26", "cùng kỳ năm trước" → khoảng ngày | ✅ | **Có** |
| | `time.lunar` | Tết, ngày âm | ✅ | Có |
| | `time.business` | Ca sáng/chiều, kỳ kế toán, ngày làm việc | 🟡 | Có |
| **Nạp tài liệu** | `ingest.tabular` | CSV, XLSX nhiều sheet, SQLite | ✅ | **Có** |
| | `ingest.docs` | PDF có chữ, DOCX, XML hoá đơn điện tử, JSON | 🟡 | Không |
| | `ingest.ocr` | Ảnh chụp, PDF scan, sổ ghi tay, hoá đơn | ⬇️ | Không |
| | `ingest.vlm` | Ảnh không phải bảng (hoá đơn bán lẻ, biên nhận) → trích trường | ⬇️ | Không |
| | `ingest.asr` | Ghi âm, câu hỏi bằng giọng nói | ⬇️ | Không |
| | `ingest.web` | Bảng trên web, API | ⬜ | Không |
| **Hiểu bảng** | `table.layout` | Header nhiều tầng, dòng tổng, ô gộp, **bảng ngang** (tháng làm cột) | 🟡 | **Có** |
| | `table.clean` | Ô bẩn → kiểu sạch | ✅ | **Có** |
| | `table.autotype` | **Tự ép kiểu cột** cho bảng ngành lạ | ⬜ | **Có** |
| | `table.classify` | Bảng thuộc loại nào (mục 3.2) | 🟡 | Có |
| | `table.schema_map` | Nghĩa từng cột (README → tên cột → LLM) | 🟡 | **Có** |
| | `table.link` | Tìm khoá nối giữa các bảng | ⬜ | Có |
| **Tra cứu** | `find.value` | Tên trong câu → **giá trị thật trong dữ liệu** (đúng dấu) | ⬜ | **Có** |
| | `find.table` | Câu hỏi → bảng/cột liên quan | ⬜ | **Có** |
| | `find.rule` | Câu hỏi → luật rulebook liên quan | ✅ | Có |
| | `find.example` | Câu hỏi → bài đã giải tương tự (kỹ năng) | ⬜ | Có |
| **Truy vấn** | `q.metric` | Chỉ số đã kiểm (doanh thu, số đơn…) | ✅ | **Có** |
| | `q.select` · `q.where` · `q.group` · `q.top` · `q.join` · `q.compare` · `q.share` | **Lệnh cơ bản viết sẵn**; LLM chỉ điền tham số | ⬜ (DuckDB) | **Có** |
| | `q.sql` | LLM viết SQL tự do (sau khi đã tra cứu) | ⬜ | Có |
| | `q.python` | Sandbox Python cho câu lạ | ✅ | Có |
| **Tri thức và trí nhớ** | `mem.cag` | Phần cố định nạp sẵn đầu prompt (rulebook, thẻ công cụ chính) | ✅ | Có |
| | `mem.rag` | Truy hồi trên tài liệu lớn / ví dụ / thẻ công cụ | 🟡 (BM25) | Có |
| | `mem.store` | Kho dài hạn có phiên bản: từ điển, ánh xạ cột, kỹ năng, lỗi | ⬜ | Có (đóng băng) |
| | `mem.compact` | Nén, gộp trùng, hết hạn | ⬜ | Không |
| | `web.search` | Tìm trên mạng | ⬜ (nền tảng) | **Không** (cấm) |
| | `github` | Lưu code, issue, PR | ⬜ (nền tảng) | Không |
| **Kiểm chứng và quyết định** | `check.second_path` | Tính lại bằng đường khác, so kết quả | 🟡 | **Có** |
| | `check.invariants` | Bất biến dữ liệu, độ hợp lý | 🟡 | Có |
| | `decide.policy` | Trả lời khi P(đúng) > ngưỡng | ✅ | **Có** |
| | `answer.format` | Đúng format BTC | ✅ | **Có** |
| | `explain.lineage` | Số này đến từ file/sheet/dòng nào | 🟡 | Không |
| **Điều hành** | `gate.permission` | Tự làm / báo / **chờ duyệt** / cấm | ⬜ | Không (thi: cấm hết) |
| | `budget.time` | Ngân sách 6 giờ, tắt bớt khi gần hết | ✅ | **Có** |
| | `log.audit` | Nhật ký đủ trường BTC yêu cầu | ✅ | **Có** |
| | `human.ask` | Hỏi lại người khi mơ hồ | ⬜ | Không (thi cấm) |
| | `task.queue` | Nhận nhiều nhiệm vụ theo lệnh, chạy lần lượt | ⬜ | Không |

### 2.3 Thang leo khi gặp khó (đúng thứ tự bạn đề xuất)

| Bậc | Cách | Khi nào lên bậc trên |
|---|---|---|
| **B0 Trí nhớ** | Đã gặp mẫu câu này và đã kiểm đúng → dùng lại kỹ năng | Chưa gặp |
| **B1 Luật / metric đã kiểm** | `q.metric` | Không có metric phù hợp |
| **B2 Lệnh cơ bản** | LLM chọn `q.select/where/group/top/compare` và điền tham số | Không ghép được |
| **B3 Truy vấn tự do** | LLM viết SQL **sau khi** `describe` + `find.value` | SQL lỗi hoặc kết quả vô lý |
| **B4 Python** | Sandbox | Vẫn lỗi |
| **B5 RAG** | Tìm ví dụ đã giải + luật liên quan → thử lại B2–B4 | Vẫn không chắc |
| **B6 CAG** | Nạp **toàn bộ** rulebook + hồ sơ shop vào ngữ cảnh dài (cần mô hình lớn hơn hoặc chế độ suy nghĩ) → thử lại | Vẫn không chắc |
| **B7 Dừng** | Thi: **bỏ trống**. Nền tảng: **hỏi người** | — |

Mỗi bậc tốn thêm thời gian, nên `budget.time` quyết định được leo tới đâu. Mọi bậc đều đi qua `check` và `decide` trước khi trả lời.

---

## 3. Danh mục dữ liệu cần hiểu

### 3.1 Loại nguồn
| Nhóm | Loại | Đặc điểm khó | Button | Thi? |
|---|---|---|---|---|
| Bảng số | CSV | Bảng mã, dấu phân cách | `ingest.tabular` | Có |
| | XLSX | Nhiều sheet, header nhiều tầng, ô gộp, dòng tổng, công thức | `ingest.tabular` + `table.layout` | Có |
| | SQLite / DB | Nhiều bảng, khoá nối, cột không mô tả | `ingest.tabular` + `table.link` | Có |
| | JSON / API | Lồng nhau | `ingest.docs` | Không |
| | XML hoá đơn điện tử | Đầu phiếu + dòng hàng | `ingest.docs` | Không |
| Văn bản | PDF có chữ | Bảng trong trang, tách cột | `ingest.docs` | Không |
| | DOCX | Bảng trong văn bản | `ingest.docs` | Không |
| | Email, chat (Zalo) | Đơn hàng viết tự do | LLM trích xuất | Không |
| Hình ảnh | PDF scan, ảnh chụp sổ | Nghiêng, mờ, chữ viết tay | `ingest.ocr` | Không |
| | Ảnh hoá đơn bán lẻ, biên nhận | Không phải dạng bảng | `ingest.vlm` | Không |
| Âm thanh | Ghi âm đặt hàng, câu hỏi bằng giọng | Giọng địa phương, ồn, trộn tiếng | `ingest.asr` | Không |
| Web | Trang có bảng, bảng giá | Thay đổi theo thời gian | `ingest.web` + `web.search` | Không |

### 3.2 Loại bảng (`table.classify`)
| Loại | Ví dụ | Đặc điểm |
|---|---|---|
| Giao dịch theo dòng hàng | Sổ chi tiết bán hàng | Một dòng = một mặt hàng trong một đơn |
| Hoá đơn (đầu phiếu) | Danh sách hoá đơn | Một dòng = một đơn, có tổng tiền, trạng thái |
| Danh mục | Hàng hoá, khách hàng, NCC, thợ | Khoá + thuộc tính |
| Quy đổi đơn vị | 1 thùng = 24 lon | Hệ số theo mặt hàng |
| Bảng giá / lịch sử giá | Giá mới từ ngày X | Hiệu lực theo thời gian |
| Phiếu nhập | Nhập hàng, NCC | Đơn vị nhập ≠ đơn vị bán |
| Lô / hạn dùng | Nhà thuốc | Một dòng bán lấy từ nhiều lô |
| Tồn kho | Ảnh chụp cuối kỳ hoặc thẻ kho | Trạng thái tại một thời điểm |
| Thu chi / sổ quỹ / báo cáo ca | Tiền mặt theo ca | Theo giờ thu, không theo ngày đặt |
| Công nợ / trả góp | Sổ nợ, kỳ góp | Nhiều lần trả cho một đơn |
| Giao hàng nhiều đợt | Vật liệu xây dựng | Một đơn nhiều đợt |
| Dịch vụ | Sổ sửa xe, bảo hành | Một ô ghi nhiều việc |
| Báo cáo tổng hợp | Pivot, đã cộng sẵn | **Không được cộng lại lần nữa** |
| Bảng ngang | Tháng làm cột | Phải xoay về dạng dọc |
| Sổ tay tự do | Ghi chép chủ shop | Ít cấu trúc |

---

## 4. Bộ não tự viết luật

Không chỉ từ điển teencode. Có **bốn loại luật** bộ não tự đề xuất, và tất cả đi chung một quy trình **Đề xuất → Kiểm chứng → Lên chức → Có phiên bản → Gỡ được**.

| Loại luật | Ví dụ | Kiểm chứng bằng gì |
|---|---|---|
| **Từ vựng** | `khum → không` (teencode, gen Z), `ck → chuyển khoản` (viết tắt, ngành bán lẻ), `아메리카노 → Americano` (Hàn, tên món) | Thay vào câu train → hệ thống trả lời đúng hơn hay không; nền tảng: người duyệt |
| **Nghĩa cột** | Cột "T.Tiền" của shop X = thành tiền sau giảm | Bất biến: tổng dòng ≈ tổng đơn |
| **Bố cục bảng** | Sheet có 2 dòng tiêu đề, dòng "TỔNG NGÀY" cần bỏ | Số dòng/tổng khớp |
| **Kỹ năng truy vấn** | "tỉ trọng nhóm X trong doanh thu tháng Y" = mẫu SQL Z | Đúng trên các câu train cùng mẫu |

**Thẻ từ vựng** (trường lưu trong `mem.store`):
- dạng viết, nghĩa chuẩn;
- loại: viết tắt / teencode / Telex sót / từ mượn / trộn ngôn ngữ;
- ngôn ngữ gốc (Việt / Anh / Hàn…);
- **thời đại/nhóm dùng** (8x–9x, gen Z, gen Alpha, ngành nghề);
- lĩnh vực;
- **mức mơ hồ** (một nghĩa / nhiều nghĩa theo ngữ cảnh);
- độ tin cậy, nguồn (train / bộ dữ liệu công khai / web / người dạy), ngày thêm, người duyệt.

Nhãn thời đại chủ yếu để phân tích và làm thông tin phụ khi gặp từ mơ hồ. Nó gần như không đổi điểm thi **[SL]**, nên không nên tốn nhiều công.

**Chặn rủi ro:**
- Ở chế độ thi, luật **chỉ được học trên train** và đóng băng trước khi nộp. Học từ câu public/private là vi phạm Rule 6.
- Luật mơ hồ (`dt`: doanh thu hay điện thoại?) **không thay thế mù**. Luật đó chỉ đưa gợi ý cho LLM quyết theo ngữ cảnh.
- Luật mới phải qua bộ **test hồi quy** rồi mới lên chức, để luật mới không phá cái cũ.

---

## 5. Trí nhớ và tối ưu bộ nhớ

| Tầng | Chứa gì | Cơ chế | Ghi chú |
|---|---|---|---|
| **Nóng (CAG)** | Rulebook của split, thẻ công cụ chính, hướng dẫn trả lời | Đặt đầu prompt, tái dùng bộ đệm KV | Ổn định, dùng ở mọi câu |
| **Ấm (RAG)** | Ví dụ đã giải, luật chi tiết, tài liệu, thẻ công cụ phụ, hồ sơ bảng | Truy hồi theo câu hỏi (BM25 trước; embedding nếu cần, tính vào 15B) | Lớn, chỉ lấy phần liên quan |
| **Lạnh** | Nhật ký, lỗi cũ, luật đã gỡ | Lưu trữ, không đưa vào prompt | Dùng để học lại, kiểm toán |
| **Nén** | Gộp trùng, tóm tắt, hết hạn (từ lóng không còn dùng) | `mem.compact` định kỳ | Chế độ nền tảng |

---

## 6. Điều hành: con người ra lệnh và cho phép

| Mức quyền | Ví dụ button | Hành vi |
|---|---|---|
| **Tự làm** | Đọc dữ liệu, giải thời gian, truy vấn, tính | Không hỏi |
| **Làm rồi báo** | Thêm luật từ vựng độ tin cậy cao vào bản nháp | Báo trong nhật ký |
| **Chờ duyệt** | Tìm trên mạng, lên chức luật mới, ghi vào kho dài hạn, gửi ra ngoài, GitHub | Dừng, chờ người bấm duyệt |
| **Cấm** | Xoá dữ liệu gốc, mọi thứ cần mạng ở chế độ thi | Không bao giờ |

Ngoài ra:
- **Đa nhiệm:** `task.queue` nhận nhiều lệnh (ví dụ "lập báo cáo cuối tháng cho 3 shop"), chạy lần lượt trong ngân sách, mỗi bước ghi nhật ký.
- **Chế độ thi:** toàn bộ "chờ duyệt" chuyển thành "cấm", vì không có người.

---

## 7. Khi một bộ não không đủ: tách thành các bộ não con chuyên trách

Số đo đã có cho thấy 4B **giỏi việc hẹp** (phân loại câu, 17/20) nhưng **kém việc rộng** (tự viết code, 0/23) **[ĐCBC, mẫu nhỏ]**. Vì vậy hướng hợp lý là chia việc cho các bộ não con. Mỗi bộ làm một việc hẹp, đo riêng được, fine-tune riêng được.

| Bộ não con | Việc | Dạng mô hình đề xuất | Fine-tune bằng dữ liệu gì |
|---|---|---|---|
| **Điều phối** | Chọn chuỗi button, quyết leo thang | LLM nền 4B (+ LoRA "planner" nếu cần) | Chuỗi button đã chạy đúng trên train |
| **Ngôn ngữ** | Câu lộn xộn → câu chuẩn; tra/đề xuất từ điển | LLM nền + LoRA, hoặc mô hình seq2seq nhỏ tiếng Việt **[CKC]** | **Tự sinh cặp dữ liệu**: lấy câu chuẩn → tự bỏ dấu, teencode hoá, gõ Telex sót, chèn tiếng Anh → có nhãn chính xác, không tốn người gán |
| **Tách câu** | Gắn nhãn từng từ: CHỈ SỐ · THỜI GIAN · TÊN · LỌC · NHÓM · TOP · SO SÁNH · **THỪA** | Mô hình gắn nhãn chuỗi nhỏ (họ PhoBERT ~135M **[CKC]**): nhanh, tất định | Câu tổng hợp có nhãn sẵn + nhãn yếu trên câu train (regex + LLM) |
| **Bảng** | Cột → nghĩa; loại bảng | Phân loại nhỏ trên (tên cột, mô tả, giá trị mẫu), hoặc LLM + LoRA | ~500 cột có mô tả trong README train + bảng tổng hợp |
| **Truy vấn** | Câu đã tách → lệnh cơ bản / SQL | LLM nền + LoRA "sql" | **Lọc theo kết quả thực thi**: sinh nhiều SQL cho mỗi câu train, chỉ giữ SQL ra **đúng đáp án vàng**. Không cần người viết SQL mẫu |
| **Phản biện** | Ước lượng P(đúng) để trả lời hay bỏ trống | Mô hình rất nhỏ (cây quyết định) trên đặc trưng: đường nào, đồng thuận, loại câu, chất lượng nguồn | Kết quả đúng/sai trên train (leave-business-out) |

**Ngân sách tham số (chế độ thi):** một LLM nền 4B dùng chung, mỗi bộ não con là một **adapter LoRA** (vài chục MB; vLLM nạp được nhiều LoRA cùng lúc **[CKC với bản vLLM của BTC]**), cộng mô hình tách câu ~0,135B, cộng phản biện không đáng kể → **khoảng 4,2B**, còn dư nhiều so với 15B.

**Fine-tune ở đâu:**
- Mô hình tách câu và phản biện fine-tune được **trên laptop 4 GB** **[SL]**.
- LoRA cho LLM 4B trên 4 GB VRAM thì **rất khó** **[SL]**. Cần GPU 16 GB trở lên, ví dụ Kaggle/Colab miễn phí, chỉ upload **dữ liệu train** dưới dạng private.
- Hướng dẫn BTC yêu cầu đóng gói **toàn bộ** code, dữ liệu, log fine-tune vào `training_code/`, và cấm dùng dữ liệu test để huấn luyện.

**Rủi ro của chia nhỏ:**
- Lỗi **dồn qua nhiều tầng**: mỗi tầng đúng 90% thì 5 tầng còn khoảng 59% **[SL]**. Vì vậy phải đo **đầu cuối** chứ không chỉ đo từng bộ.
- Nhiều lần gọi mô hình hơn → chậm hơn (vẫn trong 6 giờ, cần đo).
- Mỗi bộ cần dữ liệu và test riêng.

## 8. Tách câu truy vấn: chỉ giữ từ cần thiết

Ví dụ: **"ê cho hỏi nhanh, th5/2026 dthu đơn bán tại quầy chiếm bn % tổng dthu kua shop mik luôn z"**

| Đoạn | Nhãn | Xử lý |
|---|---|---|
| ê · cho hỏi nhanh · luôn · z · mik | **THỪA** | Bỏ |
| th5/2026 | THỜI GIAN | `time.resolve` → 01/05–31/05/2026 |
| dthu | CHỈ SỐ | doanh thu |
| đơn bán tại quầy | LỌC (kênh bán) | `find.value` → giá trị kênh trong dữ liệu |
| chiếm bn % tổng dthu | CHỈ SỐ PHỤ: tỉ trọng | doanh thu lọc / doanh thu toàn shop × 100 |
| kua shop | THỪA (ngữ cảnh) | Bỏ |

→ Bản rút gọn: `tỉ trọng(doanh thu | kênh = tại quầy ; toàn shop) · 05/2026`, rồi chuyển cho bộ Truy vấn.

**Các trường hợp tách khó:**
- **Câu nhiều ý:** "nợ thêm bao nhiêu **và** trả được bao nhiêu" → 2 truy vấn con → đáp án dạng list theo đúng thứ tự câu hỏi.
- **Phủ định, loại trừ:** "không tính đơn COD", "trừ phí ship".
- **Ngưỡng:** "từ 5 đơn trở lên" (gồm 5) khác "trên 5 đơn" (không gồm 5), theo §2 rulebook.
- **Số bằng chữ:** "hai chục", "1 chục", "nửa tháng", "mấy hôm nay".
- **Câu rỗng ý** (chào hỏi, ngoài phạm vi dữ liệu) → `unanswerable` hoặc bỏ trống.

## 9. Danh sách vấn đề mở rộng (tôi tự rà thêm)

| # | Vấn đề | Vì sao nguy hiểm | Hướng giải | Thi? |
|---|---|---|---|---|
| 1 | Đơn vị và quy đổi (thùng ↔ lon, bao ↔ kg) | Đáp số lệch đúng bằng hệ số | Bảng quy đổi từ README/bảng, `q.*` mặc định quy về đơn vị cơ bản | Có |
| 2 | Tên trùng hoặc na ná (2 khách cùng tên khác mã; Coca 330ml vs 1,5L) | Cộng nhầm | `find.value` trả **nhiều ứng viên** → mơ hồ thì bỏ trống/hỏi | Có |
| 3 | Bằng nhau khi xếp hạng | Sai thứ tự | Luật §9: bằng nhau xếp theo mã tăng dần, mã hoá trong `q.top` | Có |
| 4 | Dữ liệu ghi **sau** ngày hôm nay | Đếm cả tương lai | Lọc mặc định theo `as_of` (rulebook đầu trang) | Có |
| 5 | Báo cáo đã cộng sẵn bị cộng lần nữa | Gấp đôi | `table.classify` nhận "báo cáo tổng hợp", chặn cộng | Có |
| 6 | Hai nguồn mâu thuẫn (sổ tay vs phần mềm) | Chọn bừa | Báo mâu thuẫn; thi → bỏ trống | Có |
| 7 | Không có dữ liệu ≠ bằng 0 (§7) | Trả 0 là sai | Kiểm phủ dữ liệu tất định | Có |
| 8 | Rulebook đổi giữa các split | Áp luật cũ | Luật trong CAG + `rulebook-diff` khi chạy | Có |
| 9 | Từ lóng mới, ngành mới | Không hiểu | `lang.learn` + LLM; ngành mới → `table.autotype` | Có |
| 10 | Lỗi dồn qua nhiều bộ não con | Đúng từng phần, sai tổng | Đo đầu cuối; đồng thuận 2 đường | Có |
| 11 | Fine-tune học thuộc mẫu train | Kém trên private | Giữ hold-out theo shop; dữ liệu tổng hợp đa dạng | Có |
| 12 | Không tái lập (GPU, batching) | Bị loại | Greedy, seed, không batching, test chạy 2 lần | Có |
| 13 | Shop lớn ~300k dòng, nhiều lần gọi mô hình | Quá 6 giờ | Cache bảng, DuckDB, giới hạn bậc leo theo `budget.time` | Có |
| 14 | **Tiêm lệnh qua dữ liệu** (ô ghi chú chứa "bỏ qua hướng dẫn…") | Bộ não làm theo dữ liệu | Dữ liệu luôn là dữ liệu, không bao giờ là lệnh; cách ly trong prompt | Nền tảng |
| 15 | Dữ liệu cá nhân (SĐT, tên khách) đưa ra web/API | Lộ thông tin | Che trước khi gửi ra ngoài; `gate.permission` | Nền tảng |
| 16 | Trí nhớ bị đầu độc bởi web sai | Sai lan rộng | Duyệt, ghi nguồn, hết hạn, gỡ được | Nền tảng |
| 17 | Âm thanh: giọng địa phương, ồn, trộn tiếng | Chép sai tên món | ASR + `find.value` sửa theo danh mục; hỏi lại khi độ tin thấp | Nền tảng |
| 18 | OCR: chữ viết tay, số 1/7, dấu chấm/phẩy | Sai tiền | Độ tin từng ô; kiểm tổng dòng = tổng in trên phiếu | Nền tảng |

---

## 10. Kế hoạch test: tải về, thử, giữ nếu ổn

Mỗi công cụ ngoài đều qua cùng một cổng:
- chạy được trên laptop của bạn (hoặc ghi rõ cần máy lớn);
- **đo trên dữ liệu có sự thật**;
- kích thước, tốc độ, license;
- quyết định **giữ / bỏ / để sau**, ghi vào sổ đăng ký công cụ.

Theo yêu cầu, công cụ đạt thì **giữ lại** (đặt trong một thư mục riêng ngoài repo, để không làm phình repo), không xoá ngay.

| Đợt | Thử gì | Ứng viên (cần kiểm license, kích thước) | Dữ liệu test có sự thật | Ghi điểm thi |
|---|---|---|---|---|
| **1** | `lang.learn` trên câu train; `find.value`; `table.autotype`; lệnh cơ bản `q.*` + DuckDB; agent 4B dùng công cụ | Qwen3-4B (đã thử), DuckDB | 3.000 câu train (hold-out 6 shop), test metamorphic có dấu / không dấu / teencode / Anh / Hàn | **Có** |
| **1b** | Bộ tách câu (mục 8) + bộ phản biện (mục 7): sinh dữ liệu có nhãn, fine-tune trên laptop | Mô hình gắn nhãn họ PhoBERT, cây quyết định | Câu tổng hợp có nhãn + hold-out train | **Có** |
| **2** | Nhận ngôn ngữ, phục hồi dấu | fastText lid.176 hoặc lingua-py; mô hình phục hồi dấu nhỏ, hoặc dùng LLM | Câu train bỏ dấu tự động → so với bản có dấu | Một phần |
| **3** | OCR ảnh/PDF scan → bảng | PaddleOCR (có tiếng Việt), VietOCR, Tesseract + gói `vie` | **Ảnh sổ tổng hợp** do `socai synth --render-images` sinh, biết sự thật từng ô; thêm vài ảnh bạn tự chụp | Không |
| **4** | Ảnh hoá đơn không theo bảng | Mô hình đọc ảnh nhỏ (họ Qwen-VL ~3B, chạy GGUF) | Hoá đơn tổng hợp | Không |
| **5** | Âm thanh → chữ | faster-whisper (small), PhoWhisper (bản tiếng Việt) | Bạn tự đọc khoảng 10 câu hỏi; so bản chép | Không |
| **6** | Nền tảng: `web.search`, `mem.store` có duyệt, `gate.permission`, `task.queue` | Sau 23/11 | Kịch bản thao tác | Không |

**Thứ tự đề xuất:**
- Đợt 1 **trước 16/11**, vì nó quyết định điểm thi.
- Đợt 2–3 có thể chạy xen, trên CPU, không ảnh hưởng đợt 1.
- Đợt 4–6 sau vòng private.

---

## 11. Rủi ro và điều chưa biết

| | Nhãn | Ghi chú |
|---|---|---|
| 4B chưa chắc **tự chọn đúng button** khi nhà công cụ lớn | [CB] | Đo ở đợt 1. Nếu kém: giảm số button lộ ra cho mỗi câu (truy hồi thẻ), thêm ví dụ |
| Phạm vi quá lớn so với 6 ngày tới hạn thi | [ĐCBC] (lịch) | Luồng A chỉ gồm các button có cột "Ghi điểm thi = Có" |
| Luật tự học có thể sai và lan rộng | [SL] | Kiểm chứng + test hồi quy + phiên bản + gỡ được |
| OCR/ASR tiếng Việt trên laptop 4 GB chậm hoặc kém | [CKC] | Đo đợt 3, 5; CPU chạy được nhưng chậm |
| Một số công cụ có license hạn chế dùng thương mại | [CKC] | Kiểm từng công cụ trước khi giữ |
| Web search ở nền tảng đưa thông tin sai vào trí nhớ | [SL] | Luôn qua duyệt; ghi nguồn; có hết hạn |

---

## 12. Cần bạn góp ý
1. Đồng ý tách **Luồng A** (thi, đến 16/11) và **Luồng B** (nhà công cụ đầy đủ) như trên không?
2. Danh sách button và phòng: thiếu gì, thừa gì?
3. **Mức quyền** ở mục 6 có hợp ý bạn không (cái gì phải chờ duyệt)?
4. **Bộ não con** (mục 7): bắt đầu với **tách câu + phản biện** (fine-tune được trên laptop) trước, để LoRA cho LLM sau khi có GPU 16 GB+ — đồng ý không?
5. Cho phép bắt đầu **Đợt 1** (cần tải lại Qwen3-4B 2,5 GB) và **Đợt 3** (OCR, tải PaddleOCR/VietOCR khoảng 0,5–1 GB) ngay không?
