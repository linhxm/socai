# SổCái bản gọn: Qwen3-4B làm "người đọc câu hỏi", code làm "người tính"

> **Bản nháp để góp ý**, 10/11/2026. Bổ sung cho [idea.md](idea.md), thay phần "LLM viết code" bằng hướng ít tài nguyên hơn. Mọi số đo dưới đây đo trên **laptop phát triển**: RTX 3050 Ti Laptop 4 GB, RAM 8 GB.
>
> Nhãn:
> - **[ĐCBC]** đã có bằng chứng
> - **[SL]** suy luận
> - **[CB]** chưa biết
> - **[CKC]** cần kiểm chứng

## 1. Câu hỏi thiết kế
Có thể đạt kết quả tốt với **một mô hình nhỏ (4B) chạy được trên laptop** không? Điều kiện:
- những gì thử trên máy bạn cũng chính là thứ nộp đi;
- không cần thuê GPU.

## 2. Đo được gì trên máy bạn [ĐCBC]

### 2.1 Tốc độ, độ vừa bộ nhớ (Qwen3-4B Q4_K_M, llama.cpp b12246, CUDA)
| Chỉ số | Kết quả |
|---|---|
| Kích thước mô hình | 2,32 GiB, 4,02 tỷ tham số |
| Đọc prompt | **~1.940–2.250 token/giây** (512–4.096 token) |
| Sinh chữ | **60,5 token/giây** |
| Ngữ cảnh 16K + bộ đệm 8-bit | Nạp được, nhưng VRAM còn **144 MB**, rất sát trần |
| Prompt kiểu cũ (toàn văn rulebook + hồ sơ shop) | **8.000–10.500 token** (rulebook 4.745; phần hệ thống 5.294; hồ sơ shop 2.200–5.000) |

### 2.2 Thử nghiệm 1: 4B tự viết code pandas (kiểu thiết kế cũ)
| | Kết quả |
|---|---|
| Số câu thử | 23 câu train mà đường luật bỏ qua (2 shop) |
| Có đáp án chạy được | **0/23**, kể cả sau một lần tự sửa lỗi |
| Nguyên nhân chính | **Bịa tên cột** (`order_date`, `order_total`, `ctx.products`), dù prompt đã liệt kê đúng tên `d`, `revenue`. Logic cũng sai: lọc theo `order_id` khi muốn lọc món |
| Thời gian | 19–70 giây/câu |

→ **Kết luận [ĐCBC, mẫu nhỏ]:** với 4B, hướng "LLM tự viết code" **không dùng được**. Giữ nó trong hệ thống chỉ tốn giờ chạy. Khi đã có chính sách bỏ trống thì nó không gây mất điểm, nhưng cũng không đóng góp gì.

### 2.3 Thử nghiệm 2: 4B chỉ điền một mẫu JSON có ràng buộc
Mô hình chỉ chọn **loại chỉ số** trong một danh sách 15 loại có mô tả, và chép **tên được nhắc tới** trong câu. Cùng 20 câu, hai phiên bản schema:

| | Schema v1 (ép mọi trường) | Schema v2 (trường tuỳ chọn, có mô tả) |
|---|---|---|
| Loại chỉ số đúng | ~11/20 | **17/20** |
| Tên mặt hàng/nhóm/khách được chép | Đúng nhưng nhân đôi vào nhiều trường | Đúng ở mọi câu có tên thật; có cụm rác ("mặt hàng", "khách") |
| Trường so sánh kỳ trước | — | Hỏng: gần như luôn `true` |
| Thời gian/câu | 2–3 giây | **0,6–1,2 giây** |

Giới hạn của số đo này:
- chỉ 20 câu;
- tôi tự chấm bằng mắt;
- một lần chạy;
- chạy GGUF qua llama.cpp chứ không phải vLLM.

Đủ để chọn hướng, **chưa đủ để dự báo điểm**.

## 3. Đánh giá

| | |
|---|---|
| **Điểm mạnh nhất** | 4B làm rất tốt đúng việc mà regex làm dở: **hiểu câu lộn xộn để chọn loại chỉ số**, và chép tên riêng. Chạy nhanh, nhẹ, vừa laptop |
| **Điểm yếu nhất** | 4B không viết được code trên dữ liệu thật. Mọi phép tính phải nằm trong **thư viện metric cố định**. Câu nào rơi ra ngoài thư viện thì bỏ trống |
| **Hệ quả** | Điểm số bị giới hạn bởi **độ phủ của thư viện metric**, không phải bởi mô hình. Việc chính chuyển từ "chỉnh prompt" sang "**viết thêm metric đúng rulebook**" |
| **Rủi ro lớn nhất** | Private có **ngành mới**: chỉ số riêng của ngành (sửa xe, điện thoại, mẹ và bé…) sẽ không có trong thư viện → bỏ trống. Rulebook public đã cho thấy luật riêng cho 3 ngành mới; private có thể có ngành khác nữa **[CB]** |
| **Phản biện mạnh nhất** | Một mô hình 14B viết code có thể phủ được ngành mới mà thư viện metric không phủ. Chưa đo được 14B nên **không biết** khoảng cách này lớn đến đâu **[CB]** |

## 4. Thiết kế bản gọn

```
câu hỏi ──► chuẩn hoá (teencode, Telex) ──► [A] luật regex (đường nhanh, đã có)
        │                                ──► [B] Qwen3-4B → JSON {metric, tên được nhắc, top_k}
        └─► thời gian: code tất định (đã có) ──────────────────────────┐
                                                                         ▼
                 [C] tra tên trong danh mục của shop (món / nhóm / khách / thanh toán / size)
                                                                         ▼
                 [D] thư viện metric cố định (mở rộng) ──► đáp án + điều kiện hợp lệ
                                                                         ▼
                 [E] chính sách: trả lời chỉ khi qua mọi kiểm tra và A/B đồng thuận
```

| Bước | Ai làm | Ghi chú |
|---|---|---|
| A | Regex (đã có, precision 94,6% trên train) | Giữ nguyên |
| B | **Qwen3-4B**, prompt ngắn (~600–1.500 token, **không đưa rulebook**), JSON có ràng buộc, kèm vài ví dụ lấy từ train | Không cần 16K ngữ cảnh → VRAM dư dả |
| C | **Code**: so khớp tên đã bỏ dấu/fuzzy với danh mục thật của shop | Tên khớp **đúng một** mục mới dùng; khớp 0 hoặc nhiều mục → bỏ trống. Code tự biết đó là món hay nhóm, LLM không phải đoán |
| D | **Code**: mỗi metric ghi rõ § rulebook | Ô dữ liệu không đọc được → từ chối (nguyên tắc cũ) |
| E | **Code** | Xem bảng dưới |

**Quy tắc trả lời (E):**

| Tình huống | Quyết định |
|---|---|
| A và B cùng chỉ số, metric chạy được | Trả lời |
| Chỉ B có, tên tra được duy nhất, metric chạy được | Trả lời nếu precision đo trên hold-out của nhóm này > ngưỡng; ngược lại bỏ trống |
| A và B mâu thuẫn | Bỏ trống |
| B chọn `other`, hoặc metric chưa có | Bỏ trống |
| Kỳ hỏi nằm ngoài dữ liệu (§7) | `unanswerable` (code quyết, không để LLM quyết) |

**Bỏ khỏi bản gọn:**
- LLM viết code tự do (đo được 0/23);
- rulebook trong prompt (4B không dùng tốt mà tốn 5.000 token).

Luật được áp **trong code metric**. Vì vậy mỗi lần có rulebook mới phải chạy `rulebook-diff` và rà các metric bị ảnh hưởng.

## 5. Tài nguyên

| | Bản cũ (14B viết code) | **Bản gọn (4B đọc câu hỏi)** |
|---|---|---|
| Tham số | ~14,8B | **~4B** (còn ~11B nếu sau này muốn thêm mô hình phụ) |
| Chạy trên laptop của bạn | Không | **Có** |
| Prompt/câu | 8.000–10.500 token | **600–1.500 token** |
| Thời gian LLM/câu trên laptop | (không chạy được) | **~1 giây** |
| Đo hold-out 600 câu trên laptop | — | **~10–15 phút** cho phần LLM (thêm thời gian nạp dữ liệu) |
| Image Docker | ~50 GB | **~15–20 GB** (4B bf16 ~8 GB) |
| Cần thuê GPU | Có | **Không bắt buộc**. Vẫn nên thử image một lần trên Linux có GPU nếu được **[CKC]** |

## 6. Kế hoạch theo thứ tự giá trị

1. **Tách hold-out 6 shop train** (đủ loại file, đủ ngành). Mọi con số từ đây đo trên hold-out.
2. **Kiểm kê loại câu hỏi trong train:** cho 4B phân loại 3.000 câu train (~1 giờ trên laptop), đếm tần suất từng loại chỉ số. Cái này cho biết nên viết metric nào trước để phủ nhiều điểm nhất.
3. **Bộ tra tên** (bước C) + **mở rộng thư viện metric** theo thứ tự tần suất. Dự kiến các loại lớn: doanh thu/số lượng theo món, nhóm, khách, hình thức thanh toán; tỉ trọng; top bán chạy; ngày/tháng cao nhất; công nợ; nhập hàng **[SL, chờ bước 2]**.
4. **Prompt bước B có ví dụ từ train** (vài câu cho mỗi loại), đo lại độ đúng phân loại trên hold-out.
5. **Hiệu chỉnh ngưỡng** theo leave-business-out, đo PA hold-out.
6. **Đóng gói:** Qwen3-4B trong image vLLM. Bước B dùng guided JSON của vLLM (tham số thật cần kiểm với vLLM 0.10.1.1).

## 7. Cần bạn góp ý
1. **Chấp nhận bỏ hẳn "LLM viết code"** với 4B? Đổi lại là mất khả năng phủ ngành lạ.
2. **Ưu tiên giữa độ phủ và an toàn:** muốn mạnh tay trả lời (ngưỡng thấp) hay chỉ trả lời câu thật chắc?
3. **Có muốn giữ cửa cho 14B** (thiết kế để đổi mô hình chỉ bằng một dòng cấu hình) nếu sau này có máy mạnh, hay chốt luôn 4B?
4. **Bước 2 (phân loại 3.000 câu train):** chạy trên laptop của bạn khoảng 1 giờ, máy sẽ nóng và chậm trong lúc chạy. Có ổn không?
