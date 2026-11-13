# SOCAI v4: danh mục trường hợp và hàm (13/11/2026)

> Đi kèm [v4.md](v4.md) (kiến trúc 6 lớp). File này trả lời ba câu cho **từng trường hợp**: nhận ra bằng gì, sửa bằng hàm nào, kiểm bằng gì.
> Số liệu "train x / public y" là **số shop có trường hợp đó**, đếm bằng script quét toàn bộ 30 shop train và 14 shop public (chỉ đọc, ngày 13/11).
> Không chứa dữ liệu cuộc thi; ví dụ là mẫu hình, tên riêng là tên giả.

## 0. Quy ước chung cho mọi hàm

- **Hai loại hàm:**
  - **tool**: LLM gọi được. Nhận JSON, trả JSON ngắn (≤ 1.500 token): số đếm, tỉ lệ, tối đa 20 mẫu.
  - **hàm trong**: chỉ code gọi.
- **Tool không làm hỏng dữ liệu.**
  - Tool hỏi (`peek`, `profile_*`, `check_*`) chỉ đọc.
  - Tool làm (`set_*`, `apply_*`) ghi vào **hồ sơ shop** (`ShopProfile`) dưới dạng thao tác có tên. Bảng chuẩn luôn dựng lại được bằng cách chạy lại các thao tác từ file gốc (tái lập, cache theo băm file).
- **Kiểu dùng chung:**
  - `SrcId = "file.xlsx::sheet"`, hoặc `"file.xlsx"` cho bảng đã gộp sheet.
  - `Col` là tên cột gốc.
  - `RowSel = {"rows": [...]} | {"kind": "..."} | {"where": [col, op, value]}`.
- **Mọi tool trả `{"ok": bool, ...}`.** Lỗi thì có `"error"` và `"hint"`, để LLM tự sửa lời gọi.
- **Thư mục mã:** `src/socai/v4/`.
  - `cells.py`: đọc ô.
  - `grid.py`: bảng thô + loại dòng.
  - `tools_l1.py`, `tools_l2.py`: tool.
  - `recipe.py`: chạy thao tác.
  - `profile.py`: hồ sơ shop.
  - `registry.py`: schema tool cho LLM.
  - `loop.py`: vòng gọi tool.
  - `l3_rules.py`, `l4_ask.py`, `l5_compute.py`, `l6_decide.py`.

---

## 1. L1: đọc file và làm sạch hình thức

### 1.1. Vỏ file

| # | Trường hợp | Có ở | Nhận ra | Sửa bằng | Kiểm |
|---|---|---|---|---|---|
| F1 | XLSX một hoặc nhiều sheet | train 22 / public 10 | đuôi file | `load_grids(shop_dir)` → mỗi sheet một `Grid` (ô thô, **chưa** chọn header) | đủ sheet, đủ dòng |
| F2 | CSV | train 8 / public 4 | đuôi file | `load_grids`: dò mã hoá (`utf-8-sig`, `utf-8`, `cp1258`, `latin-1`) và dấu phân cách (`,` `;` `\t` `\|`), chọn cách cho **số cột ổn định nhất** qua 200 dòng đầu | số cột mỗi dòng bằng nhau ≥ 98% |
| F3 | SQLite | train 1 / public 1 | đuôi file | `load_grids`: mỗi bảng một `Grid` đã có header; tên cột dính liền được tách thành bản đọc được (`KenhBanHang` → `kenh ban hang`) để so với README | — |
| F4 | Nhiều sheet cùng cấu trúc (theo tháng/năm) | train 11 / public 4 | cùng tập header sau `detect_header` | `merge_grids(src_ids)` → một bảng, thêm `_sheet`, `_ky_sheet` (năm, tháng từ `sheet_period`) | tổng dòng = tổng các sheet |
| F5 | Nhiều file cùng cấu trúc (`…_T01.xlsx`, `…_1.xlsx`) | train + public | cùng header, tên file chỉ khác số | `merge_grids` như F4, thêm `_file` | như F4 |
| F6 | Sheet phụ trong cùng file (ghi tắt, danh mục, hướng dẫn) | public 2 | header khác bảng chính | Giữ bảng riêng; L2 quyết vai trò | — |
| F7 | `.xls` cũ, `.json`, `.tsv`, sheet ẩn, ô công thức không có giá trị lưu | chưa gặp | đuôi file / cờ openpyxl | `load_grids` có chỗ cắm reader; ô công thức thiếu giá trị ghi vào biên bản | ghi rõ ô thiếu |

### 1.2. Header

| # | Trường hợp | Có ở | Nhận ra | Sửa bằng | Kiểm |
|---|---|---|---|---|---|
| H1 | Dòng tiêu đề, dòng trống phía trên header | train + public (header ở dòng 2 và dòng 4) | `detect_header(src)`: chấm 15 dòng đầu theo số ô chữ không phải số, trùng tên cột README, các dòng ngay dưới có kiểu ổn định | **tool** `detect_header(src) → {row, score, alts:[{row, score}], readme_hits}`; **tool** `set_header(src, row, levels=1)` | các dòng dưới header có kiểu cột ổn định |
| H2 | Header hai tầng | chưa gặp rõ | hai dòng liền nhau đều nhiều chữ, dòng trên có ô gộp | `set_header(src, row, levels=2)`: ghép `trên › dưới`, điền ngang ô gộp | tên cột không trùng |
| H3 | Header lặp giữa bảng (dán nối nhiều lần) | public 2 (54 dòng) | dòng có ≥ 3 ô trùng đúng tên cột của chính nó | `row_kinds` → `header_lap` | dòng loại này bị bỏ, có đếm |
| H4 | Tên cột trùng nhau | train | — | `_dedupe` (đã có): thêm hậu tố `.2` | — |
| H5 | Không có header | chưa gặp | không dòng nào đạt điểm | LLM đặt tên cột từ README + mẫu: `set_header(src, row=-1, names=[...])` | số tên = số cột |

### 1.3. Loại dòng

Một hàm phân loại: `classify_rows(src) → Series[kind]`. Một tool tóm tắt: `row_kinds(src) → {counts, uncertain:[{row, kind?, why, cells}]}`. Một tool sửa nhãn: `set_row_kind(src, sel, kind)`.

| # | Loại | Có ở | Nhận ra (code) | Cái dễ nhầm | Kiểm |
|---|---|---|---|---|---|
| R1 | `du_lieu` | — | mặc định | — | — |
| R2 | `trong` | nhiều | mọi ô trống / khoảng trắng | — | — |
| R3 | `tong` | train + public (`Tổng ngày 01/04`, `Cộng ngày 01/05/2025`, ghi **ngay trong cột tên khách**) | (a) ô chữ khớp nhãn tổng **trên chữ còn dấu** (`Tổng`, `Cộng`, `Total`, `TỔNG`), **và** (b) dòng thiếu các ô khoá như mã đơn hay tên hàng, **và** (c) `check_total_row` khớp | So nhãn sau khi bỏ dấu sẽ bắt nhầm họ **Tống** (hàng nghìn dòng khách ở nhiều shop) và hàng **Công tắc…**. Không bao giờ nhận dòng tổng chỉ bằng nhãn bỏ dấu | **tool** `check_total_row(src, row, amount_col, scope="day"\|"group"\|"above") → {value, sum, diff, rows_used}`: khớp ±1 đồng thì chắc chắn; không khớp thì để `uncertain` cho LLM |
| R4 | `ghi_chu_cuoi` | train 2 / public 4 (`Ghi chú: tiền tính bằng đồng`) | ≤ 2 ô có chữ, khớp `Ghi chú` / `Lưu ý` / `Note`, nằm sau dòng dữ liệu cuối hoặc trước header | Cột "Ghi chú" của dòng dữ liệu (dòng đó còn nhiều ô khác) | nằm cuối khối |
| R5 | `header_lap` | public 2 | như H3 | — | — |
| R6 | `tiep_noi` (ô nhóm để trống, dòng đầu đại diện) | train 14 / public 9 | ô khoá (mã đơn, ngày) trống, ô hàng có giá trị, dòng trên cùng khối có ô khoá | Dòng **thiếu** thật (lỗi ghi) cũng trông giống | `group_rows` báo nhóm lạ: nhóm không có dòng đầu |
| R7 | `dong_phu_don` (voucher, ship, điểm, bớt, phụ thu trong đơn) | public 1 (dòng không có SL, có tiền) | trong nhóm đơn, không có SL nhưng có tiền; tên lặp ở nhiều đơn | Dòng hàng quên ghi SL | LLM xác nhận qua `distinct(src, col, where=no_qty)` |
| R8 | `dong_huy` (gạch, ghi "xoá", "nhập nhầm") | chưa gặp | ghi chú / định dạng gạch ngang (openpyxl `font.strike`) | — | LLM xác nhận |

### 1.4. Nhóm dòng (đơn) và ô để trống

| # | Trường hợp | Có ở | Hàm |
|---|---|---|---|
| G1 | Có mã đơn ở mọi dòng | nhiều | `group_rows(src, by=col)` |
| G2 | Mã đơn / STT chỉ ở dòng đầu, các dòng sau trống | train + public | `group_rows(src, start_when_notna=col)`: nhóm mới khi cột có giá trị |
| G3 | Các ô của đơn (ngày, khách, trạng thái, ghi chú) chỉ ở dòng đầu | train + public (README 4 / 6 shop nói rõ) | **tool** `ffill_group(src, cols, group)`; trả về số ô đã điền và **số nhóm mà dòng sau ghi khác dòng đầu** (mâu thuẫn) |
| G4 | Các dòng trong đơn ghi ngày khác nhau | public 2 | `group_profile(src, group, col) → {n_groups_multi_value, examples}`; chọn cách lấy (`first_with_qty` / `min` / `first`) do L2 quyết qua `set_group_value(src, group, col, how)` |
| G5 | Không có khoá nhóm, mỗi dòng là một lần ghi (sổ phiếu) | public 1 | `group_rows(src, by=None)` → mỗi dòng một nhóm; cờ `theo_don=false` cho L2 |

**Tool** `group_rows(...) → {n_groups, size_hist, odd:[{group, why}]}`. Nhóm lạ là nhóm rỗng, nhóm chỉ có dòng phụ, hoặc nhóm không có dòng đầu.

### 1.5. Ô dữ liệu

Hàm trong ở `v4/cells.py` dùng lại `io/cells.py`. Tool `parse_column(src, col, kind, **opt) → {ok_rate, n, failures:[{row, value}], stats}`. Tool `set_column_kind(src, col, kind, **opt)`.

| # | `kind` | Mẫu hình đã gặp | Có ở | Ghi chú / còn thiếu |
|---|---|---|---|---|
| C1 | `money` | `41k`, `2tr900`, `1tr550`, `2tr`, `25tr`, `800đ`, `1.500đ`, `53.500đ`, `2,760,000.0000`, `1,14tr`, `1,156tr`, `-200k`, `(200.000)`, `1 triệu 2`, `100.000 VND` | tiền dạng chữ: train 6 / public 10 | `parse_number` đã đọc đúng cả 19 mẫu (đã thử). Thêm: tham số `decimal=","\|"."\|"auto"` cho số kiểu Âu/Mỹ (chưa gặp); `auto` chọn theo đa số trong cột |
| C2 | `qty` | `2`, `2 ly`, `3 chai`, `1 hộp`, `0,5 kg`, `1 bao 25kg`, `2 xe 4m3` | train 10 / public 5 | `parse_qty` → `(số, chữ đơn vị)`. Thêm: `x2`, `2x` (đang trả None); đơn vị ghép `bao 25kg` giữ nguyên chữ để L2 tra bảng quy đổi |
| C3 | `date` | serial Excel `45717`; `dd/mm/yyyy`; ISO; `1 thg 7 2025`; `28/4` (năm theo sheet); `1-7-2025` | serial: train 22 / public 13 · dmy: 28 / 12 · ISO: 14 / 11 · thg: 5 / 5 · thiếu năm: 1 / 2 | **Đọc từng ô**, không đọc cả cột với `dayfirst=True` (đã từng đọc ngược ngày ở một cột trộn ISO với dd/mm). Ô thiếu năm lấy `_ky_sheet`; không có thì lấy năm của dòng trên. `m/d/yyyy` kiểu Mỹ chỉ nhận khi cả cột có ô ngày > 12 ở vị trí thứ hai |
| C4 | `datetime` | `2025-07-01 14:35:00`, serial có phần thập phân | nhiều | `parse_datetime` |
| C5 | `time` | `14:35`, `2:35 PM`, `14h35`, `14g35` | nhiều | Sửa: `14h35` đang trả None; `2:35 PM` đang trả 02:35, phải là 14:35 |
| C6 | `bool` | `true/false`, `1/0`, `x`, `có/không` | train 6 / public 2 | `parse_bool` |
| C7 | `code` | `" HD0012"`, `"SP0006 "` | public 2 (hơn 22.000 ô) | strip + giữ hoa thường gốc; khoá so khớp = `match_key` |
| C8 | `pct` | `5%`, `5`, `0,05` | train | `parse_pct(v, scale="auto")` |
| C9 | `text` | — | — | NFC, gộp khoảng trắng |
| C10 | Ô "không có" | `-`, `—`, `nan`, `không`, `0` ở cột ngày | train | `is_blank` + danh sách giá trị rỗng **theo cột** (LLM xác nhận qua `distinct`) |

### 1.6. Ô chứa nhiều thông tin

Tool `split_cell(src, col, how) → {ok_rate, new_cols, failures}`. Tham số `how` có ba dạng:
- `{"sep": " / ", "into": "list"}`: tách thành list giữ trong một ô;
- `{"regex": "(?P<a>…)…(?P<b>…)"}`: tách thành nhiều cột;
- `{"preset": "<tên>"}`: dùng mẫu có sẵn.

| # | Trường hợp | Có ở | Cách tách | Kiểm |
|---|---|---|---|---|
| M1 | Hai serial/IMEI trong một ô cho dòng SL = 2 | public 1 | `sep " / "` thành list | số phần tử = SL (`check_split_vs(col, qty_col)`) |
| M2 | Dòng xe + biển số chung một ô | public 1 | regex biển số ở cuối ô | 100% dòng có biển số |
| M3 | Nhiều việc trong một ô (`a, b + c; d`) | public 1 | `sep "[,+;]"` thành list | — |
| M4 | Món + topping (`Món + 2 Topping X (10k)`) | public 2 | `preset "item_plus_addons"` → các dòng con: (tên, SL, tiền trong ngoặc); tiền món = tiền dòng − tiền topping | tiền con cộng lại = tiền dòng |
| M5 | Tên khách kèm số điện thoại | train 14 / public 9 | `preset "name_phone"` → tên, sđt | sđt 10 số đầu 0 |
| M6 | Ngày + giờ chung ô | nhiều | `kind datetime`, tách `_ngay`, `_gio` | — |
| M7 | Hoạt chất phối hợp `A + B` | train 6 / public 5 | **Không tách**. Luật ngành nói phối hợp là một loại riêng | L1 chỉ báo ô có dấu tách; L2/L3 quyết |
| M8 | Danh sách tên khác trong danh mục (`tên 1, tên 2`) | public 1 | `sep ","` → alias (L2) | — |
| M9 | Ghi chú tự do chứa số tiền, trạng thái (`còn thiếu 200k`, `huỷ`, `trả đủ`) | train 2 / public 3 | Không tách ở L1; L2 dùng `note_templates` | — |

### 1.7. Dòng trùng

| # | Trường hợp | Có ở | Hàm | Ai quyết bỏ |
|---|---|---|---|---|
| D1 | Trùng hệt trong cùng một đơn | train 14 / public 13 | **tool** `dup_report(src, scope="group", group=…)` → `{n_groups, n_rows_extra, examples}` | L2, theo L3 (§8): bảng theo đơn thì bỏ |
| D2 | Trùng hệt ở bảng không theo đơn | public 1 | `dup_report(scope="table")` | L2: giữ cả hai |
| D3 | Trùng do hai sheet/file chồng kỳ | public 1 | `dup_report(scope="cross_source", ignore=["_sheet", "_file"])` | L2: bỏ bản lặp |
| D4 | Trùng mã đơn nhưng khác nội dung | train | `dup_report(scope="key", key=col)` | L2: báo, không tự gộp |

### 1.8. Recipe L1 và bất biến

LLM chốt mỗi bảng bằng **một recipe**. Recipe là danh sách các lời gọi `set_*` / `split_cell` / `ffill_group` đã thử. `apply_recipe(src, recipe)` chạy lại từ đầu; kết quả giống nhau mọi lần.

`l1_invariants(src) → [{name, ok, detail}]`:
1. Mỗi dòng có đúng một `kind`. Mọi dòng `tong` đã qua `check_total_row`, hoặc LLM đã xác nhận.
2. Mỗi cột đã gán `money/date/qty/time` đọc được ≥ 99,5%; ô không đọc được liệt kê trong biên bản.
3. Ngày nằm trong khoảng README ghi; ngày của sheet theo tháng nằm đúng tháng đó (cho phép lệch ≤ 1 ngày ở mép).
4. Số dòng vào = số dòng ra + số dòng bỏ theo từng loại.
5. Cột số không còn ô dạng chữ.

**Vòng L1 cho một bảng** (`l1_run(src)`):
1. Code tự gọi `detect_header`, `profile_table`, `row_kinds`, `dup_report` và gom thành **tóm tắt** (≤ 2.000 token).
2. LLM nhận tóm tắt cùng đoạn README của file, rồi gọi tool cho tới khi chốt recipe. Trần 15 lần gọi mỗi bảng.
3. Code chạy `l1_invariants`. Trượt thì đưa lỗi lại cho LLM, tối đa 2 vòng. Vẫn trượt thì bảng mang cờ `chua_sach` cùng lý do.

**Tool hỏi chung cho L1:**

| Tool | Trả về |
|---|---|
| `list_sources()` | Danh sách file/sheet, số dòng, số cột, kỳ suy từ tên |
| `peek(src, at="head"\|"tail"\|"random"\|row, n=10, cols=None)` | ≤ 20 dòng, ô cắt 40 ký tự |
| `profile_table(src)` | Mỗi cột: tỉ lệ trống, 3 chữ ký hình dạng hay gặp, kiểu đoán, 3 mẫu |
| `profile_column(src, col)` | Chữ ký hình dạng đầy đủ (`9tr9`, `99/99/9999`, `99999`, `w`, `trống`) + tỉ lệ đọc được theo từng `kind` |
| `distinct(src, col, k=30, where=None)` | Giá trị khác nhau + tần suất |
| `find_rows(src, text, col=None, accents=True)` | Dòng chứa chữ. Mặc định **giữ dấu** (bài học `Tống`/`Tổng`) |
| `readme(file=None, col=None)` | Đoạn README của file / của cột |

---

## 2. L2: mô hình shop

**Đầu ra:** `ShopProfile` gồm các phần sau:
- `tables`: vai trò, cột → trường chuẩn;
- `aliases`: theo loại thực thể;
- `units`: đồ thị quy đổi;
- `status_map`;
- `note_rules`;
- `joins`;
- `facets`;
- `industry`;
- `flags`.

Từ hồ sơ này, `build_canonical(profile)` dựng các bảng chuẩn: `orders`, `lines`, `items`, `customers`, `staff`, `payments`, `imports`, `lots`, `stock`, `costs`, `prices`, `deliveries`, `debts`, `installments`, `tickets`, `warranty`, `tradeins`, `other_*`.

### 2.1. Vai trò bảng và nghĩa cột

| # | Trường hợp | Có ở | Hàm | Kiểm |
|---|---|---|---|---|
| S1 | README mô tả cột (từ vựng có kiểm soát) | mọi shop | `readme_columns(file) → {col: mô tả}` (code đọc markdown) | mọi cột trong README có mặt trong bảng |
| S2 | Vai trò bảng | mọi shop | **tool** `role_candidates(src) → [{role, score, why}]` (từ README + tên cột + kiểu); LLM chọn `set_role(src, role, theo_don: bool)` | bảng `lines` nối được `orders` hoặc tự chứa ngày |
| S3 | Nghĩa cột: tên tự đặt, viết tắt, SQLite dính liền | train 578 tên cột khác nhau | **tool** `field_candidates(src, field) → [{col, score, why}]`; LLM gán `map_fields(src, {field: col})` | kiểu khớp (`money` cho tiền…); `check_field(src, field)` |
| S4 | Hai cột cùng ứng viên (SL gốc / SL quy đổi; giá vốn dòng / danh mục) | train + public | Gán cả hai với hai trường khác nhau (`qty`, `qty_base`; `cost_line`, `cost_catalog`) | `qty_base / qty` là hệ số > 0 nhất quán theo (hàng, đơn vị) |
| S5 | Bảng vai trò chưa có trong danh sách (ngành mới) | chưa gặp | `set_role(src, "khac", desc=…)`: vẫn vào DuckDB với mô tả cột từ README | — |
| S6 | Ngành shop | — | `industry_candidates()` từ README + tên bảng; L3 lọc thẻ ngành theo đây | — |

### 2.2. Nối bảng

| # | Trường hợp | Có ở | Hàm |
|---|---|---|---|
| J1 | Dòng ↔ đơn theo mã | nhiều | **tool** `join_coverage(a, acol, b, bcol) → {raw, stripped, folded}` (tỉ lệ khoá tìm thấy theo ba cách so) |
| J2 | Dòng ↔ danh mục: có mã thì theo mã, chỉ có tên thì theo tên | train + public | `join_coverage` thử mọi cặp; **ưu tiên mã** |
| J3 | Dòng ghi tên, danh mục phụ ghi mã (tên → mã → thuộc tính) | train | `join_chain(lines.name → items.name → items.code → attrs.code)` |
| J4 | Mã có khoảng trắng / khác hoa thường | public 2 | `join_coverage` với cách `stripped` |
| J5 | Đơn trong bảng đơn không có dòng, hoặc ngược lại | public | `orphan_report(a, b)`: đếm và giữ, không tự bịa dòng |
| J6 | Thu tiền / máy cũ / bảo hành trỏ về đơn qua ghi chú (`Trừ vào STT 12`) | public | `extract_ref(src, col, regex)` → cột khoá |

### 2.3. Gộp tên

**tool** `alias_clusters(entity, src, col) → {clusters:[{members, signals, score}], singletons}`
**tool** `set_alias(entity, members, canonical)` · `reject_alias(entity, members)`

Tín hiệu do code tính: `exact_ci` (khác hoa thường, khoảng trắng, dấu câu), `folded` (khác dấu), `initials` (viết tắt chữ đầu), `token_subset`, `edit≤2`, `same_code`, `alias_table`.

| # | Trường hợp | Có ở | Quyết |
|---|---|---|---|
| A1 | Tên hàng chỉ khác hoa thường, khoảng trắng | train + public | code gộp |
| A2 | Có bảng/sheet tên gọi tắt | public 2 | code nạp thẳng (`load_alias_table(src, short_col, full_col, type_col?)`) |
| A3 | Danh mục có cột tên khác | public 1 | code (`split_cell sep ","` → `set_alias`) |
| A4 | Tên người gọi tắt (`a. Tư` → họ tên đầy đủ trong danh sách nhân sự) | public 1 | code đề xuất theo token cuối sau khi bỏ xưng hô (`a.`, `anh`, `chị`, `c.`, `cô`, `chú`…); trùng token cuối thì LLM xem ngữ cảnh |
| A5 | Tên dòng sản phẩm viết tắt (`AB`, `Ex`, `wave α`) | public 1 | `alias_clusters` với `initials`; LLM xác nhận cụm mơ hồ |
| A6 | Khách lẻ nhiều cách ghi | train + public | **tool** `walkin_values(src, col)`: giá trị hay gặp nhất, ngắn, không phải tên người; LLM xác nhận → `set_walkin(values)` |
| A7 | Khách không có mã: chỉ khác cách viết là một người; khác chữ, khác số là hai | train + public | code chỉ gộp `exact_ci`; khác dấu thanh → LLM; không chắc thì không gộp |
| A8 | Hai mã khác nhau trùng tên | train | **không gộp** (mã là khoá) |
| A9 | Tên kèm số điện thoại | train 14 / public 9 | khoá = tên đã bỏ sđt; giữ bản gốc để hiển thị |

Tên hiển thị của một cụm: tên danh mục nếu có, không thì cách viết hay gặp nhất.

### 2.4. Trạng thái và ghi chú

| # | Trường hợp | Có ở | Hàm |
|---|---|---|---|
| T1 | Cột trạng thái có nhiều cách ghi (tổng 374 giá trị khác nhau trên 38 shop) | train 27 / public 11 | **tool** `status_values(src, col) → [{value, n}]`; LLM gán **mỗi giá trị** vào `hoan_thanh / huy / tra_hang / dang_giao` (guided JSON): `set_status_map(src, col, {value: state})` |
| T2 | Không có cột trạng thái, trạng thái ghi trong ghi chú | train 2 / public 3 | **tool** `note_templates(src, col) → [{template, n, examples}]` (che số, tiền, tên → mẫu câu); LLM gán từng mẫu → `set_note_rules(src, col, [{template, status?, debt?}])` |
| T3 | Trả hàng ghi ở bảng khác (lô trả > 0), trả một phần | public 1 | `set_status_source(src, key, rule="any_return_qty>0" → tra_hang)`; đơn có bất kỳ dòng trả nào là trả hàng cả đơn |
| T4 | Trạng thái tiếng Anh (`cancelled`, `returned`) | public 2 | như T1 |
| T5 | Một đơn có nhiều trạng thái khác nhau giữa các dòng | — | `group_profile(..., col=status)`: báo; ưu tiên `tra_hang > huy > dang_giao > hoan_thanh`, cần LLM xác nhận |

Kiểm: mỗi đơn đúng một trạng thái; mọi giá trị trong cột đã có ánh xạ.

### 2.5. Đơn vị

| # | Trường hợp | Có ở | Hàm |
|---|---|---|---|
| U1 | Bảng quy đổi (hàng, đơn vị lớn, hệ số) | train 29 / public 10 (README nói về quy đổi) | `unit_graph(items, conv_tables)`: đồ thị (hàng, đơn vị) → (đơn vị nhỏ hơn, hệ số) |
| U2 | Quy đổi nhiều cấp (`hộp = 10 vỉ`, `vỉ = 10 viên`) | public | `unit_factor(item, unit) → (factor, path)`: nhân theo đường đi tới đơn vị cơ bản |
| U3 | Quy đổi ghi dạng chuỗi trong danh mục (`1 thùng = 24`) | public | `parse_conversion_text(text) → [(unit, n, sub_unit)]` |
| U4 | Quy đổi chỉ ghi trong README (`bó = 10 cây`) | public | LLM trích từ README → `add_unit_rule(scope, unit, factor, base)`; code kiểm U6 |
| U5 | Cột SL đã quy đổi sẵn | train + public | S4: `qty_base` lấy thẳng; đối chiếu với U2 nếu có cả hai |
| U6 | Kiểm: mọi cặp (hàng, đơn vị) có ở dòng bán đều có hệ số | — | **tool** `unit_coverage() → [{item, unit, n_rows}]` các cặp còn thiếu |
| U7 | Đơn vị ghép trong ô SL (`1 bao 25kg`, `2 xe 4m3`) | train | C2 giữ chữ; `unit_factor` thử cả `bao 25kg` và `bao` |
| U8 | Đơn vị không có dòng quy đổi | — | Là đơn vị cơ bản, hệ số 1 (rulebook §10) |

### 2.6. Thực thể đặc thù (ngành)

Mỗi thực thể có một hàm dựng. LLM chỉ gán cột; công thức nằm trong code.

| # | Thực thể | Có ở | Hàm dựng |
|---|---|---|---|
| E1 | Topping / dòng con tách từ M4 | public 2 | `explode_addons(lines)` |
| E2 | Dòng phí phục vụ, dòng phụ phí không phải mặt hàng | train + public | `flag_non_items(lines, values)` (LLM chọn giá trị từ `distinct`) |
| E3 | Công nợ từ phiếu thu | train + public | `debt_ledger(orders, payments)` (dùng lại `v2/debt.py`) |
| E4 | Công nợ từ ghi chú | public 1 | `debt_from_notes(orders, note_rules)` |
| E5 | Trả góp: lịch hạn theo ngày trong tháng, tháng ngắn về cuối tháng | public 1 | `installment_schedule(contracts) → dues`; `installment_state(x) → {còn nợ, quá hạn, số kỳ đến hạn}` |
| E6 | Lô, hạn dùng, giá nhập lô, bán theo lô, trả theo lô | train + public | `lot_ledger(imports, line_lots) → tồn từng lô theo ngày` |
| E7 | Tồn kho theo ngày | train + public | `stock_at(item, day)` (từ thẻ kho, hoặc nhập − bán) |
| E8 | Phiếu dịch vụ (sửa xe): mỗi dòng một phiếu, phiếu bảo hành tiền 0 | public 1 | `tickets(src)`; `theo_don=false` |
| E9 | Bảo hành, thu máy cũ | public 1 | `warranty(src)`, `tradeins(src)` nối qua J6 |
| E10 | Giao nhiều đợt | public | `deliveries(src)` |
| E11 | Lịch sử giá, ngày tăng giá | train + public | `price_changes(prices)` |
| E12 | Thành viên, ngày đăng ký, điểm | public | `customers(src)` |

### 2.7. Bất biến L2 và `facets`

`l2_invariants(profile) → [{name, ok, detail}]`:

| Bất biến | Điều kiện |
|---|---|
| `order_one_date_one_state` | mỗi đơn đúng 1 ngày, 1 trạng thái |
| `line_has_item` | dòng chưa nối mặt hàng = 0 (hoặc đã giải thích) |
| `unit_factor_all` | U6 rỗng |
| `line_sum_vs_order_total` | lệch nằm trong giảm cả đơn + ship |
| `payments_le_total` | |
| `stock_nonneg` | |
| `dates_in_range` | |
| `monthly_revenue_smooth` | cảnh báo khi một tháng gấp hoặc bằng 1/10 tháng liền kề: bắt lỗi đọc triệu thành tỷ |

`facets(profile) → {has: [...], missing: [...]}`. Danh sách loại thông tin gồm: tồn kho, nhập hàng, lô/hạn dùng, công nợ, nhóm khách, giới tính, địa chỉ, giờ, nhân viên, chi phí, giá vốn, kênh, thanh toán, đánh giá… Đây là đầu vào của cổng `unanswerable` ở L6.

---

## 3. L3: rulebook

| Hàm | Làm gì | Ai |
|---|---|---|
| `chunk_rulebook(md) → [Card]` | Cắt theo heading → bảng → dòng bảng / gạch đầu dòng; giữ đường dẫn mục | code |
| `enrich_card(card) → Card` | Gắn `loai`, `pham_vi.nganh`, `khai_niem`, `cach_noi_khac` (có/không dấu, viết tắt), `tham_so`, `ghi_de`, `can_du_lieu`, `lop` | 14B, guided JSON, mỗi thẻ một lần gọi |
| `extract_params(cards) → Params` | Điền schema tham số cố định (ca, tuần, tháng này, Tết, làm tròn, hoà, ngưỡng khách sộp…) | 14B + code kiểm kiểu |
| `diff_params(params, defaults_train)` | Khác biệt so với rulebook train → `diff.json` | code |
| `build_index(cards)` | BM25 trên `khai_niem + cach_noi_khac + text` (có dấu và bỏ dấu) | code |
| `retrieve(query, k=6, industry) → [Card]` | Truy hồi, lọc theo ngành, thẻ ngành ghi đè thẻ chung | code |
| `glossary(cards) → {term: [card_id]}` | Bảng thuật ngữ cho L4 | code |

**Cache:** `rules/<sha rulebook>/` gồm `cards.jsonl`, `params.json`, `glossary.json`, `diff.json`.

**Test:**
- `extract_params` trên rulebook train và public, so với bảng tham số viết tay;
- nhiễu loạn có chủ đích (đổi số trong luật) thì tham số phải đổi theo;
- recall@6 của thẻ đúng trên câu public.

---

## 4. L4: hiểu câu

| Hàm | Làm gì | Bằng chứng cần |
|---|---|---|
| `normalize(q) → {orig, nfc, lower, folded, tokens}` | NFC, bỏ emoji, giữ bản gốc | — |
| `abbrev_score(tok, phrase) → (score, kind)` | So một token với một cụm theo các kiểu: `initials` (`dh`, `ncc`), `skeleton` (`dthu`, `bnhiu`), `prefix` (`th`), `edit` sau phép thế teencode (`k↔c`, `j↔gi/d`, `w↔qu`, `z↔d/v`, `ck↔ch`, `f↔ph`), `dialect` (`nhứt`, `mô`, `ni`, `rứa`) | 391 từ ở public không có ở train |
| `link_entities(q, vocab) → [{span, entity, id, score, kind}]` | n-gram 1–8 token × từ vựng shop (tên hàng + alias, khách, thợ, nhóm, kênh, trạng thái, đơn vị, size, dòng máy, biển số, hoạt chất) | — |
| `match_concepts(q, glossary, columns) → [{span, concept, card_ids, score}]` | Như trên nhưng với thuật ngữ L3 và mô tả cột L2 | — |
| `resolve_time(q, as_of, params, calendar) → TimeSpec` | Họ mẫu thời gian (bảng dưới); đi qua `abbrev_score` trước khi khớp từ khoá thời gian | Bộ giải hiện tại: train bỏ sót 28/3.000 câu, **public bỏ sót 93/1.000** (`hom ni`, `th này`, `6-2025`, `12-2025`, `tuần trc`, `tháng ni`) |
| `rewrite_time(q) → str` | Cụm thời gian không giải được: 14B viết lại thành cách nói chuẩn, rồi gọi lại `resolve_time` | — |
| `retrieve_rules(q, concepts) → [Card]` | L3 `retrieve` | — |
| `answer_form(q) → {type, k?, order?}` | Dạng đáp án từ cấu trúc câu (bao nhiêu, %, có … không, top k, liệt kê, bảng, ngày nào, tháng nào), cũng qua `abbrev_score` | — |
| `compose_spec(ctx) → [Spec]` | 14B, guided JSON, k mẫu | — |
| `validate_spec(spec, ctx) → [error]` | Lọc phải là thực thể đã khớp; chỉ số có thẻ hoặc cột; dạng khớp `answer_form`; kỳ trong dữ liệu | — |

**Họ mẫu thời gian.** Mỗi họ là một regex có tham số, không phải danh sách chuỗi:

| Họ | Mẫu (sau `normalize` + `abbrev_score`) | Ví dụ |
|---|---|---|
| tháng | `(t\|th\|thg\|thang)\s*M([/-]Y\|\s+nam\s+Y)?` | `th3`, `t5/2026`, `tháng 2 năm 2026` |
| tháng không chữ | `M[/-]YYYY` khi không phải ngày | `6-2025`, `12/2025` |
| quý | `(q\|quy)\s*Q([/-]Y)?` | `quý 2/2026`, `q1` |
| ngày | `D[/-]M([/-]Y)?`, `D thg M Y`, `ngày D` | `22/04/2026` |
| tương đối, ngày | hôm nay / qua / kia + biến thể (`hom ni`, `bữa ni`, `hum qua`) | |
| tương đối, tuần/tháng/năm | (tuần\|tháng\|năm\|quý) + (này\|ni\|nay\|trước\|trc\|rồi\|qua) | `th này`, `tuần trc`, `tháng ni` |
| N ngày | N ngày (gần nhất\|qua\|vừa rồi) | |
| từ đầu | từ đầu (năm\|tháng\|quý) (tới\|đến) (nay\|hôm nay) | |
| cùng kỳ | cùng kỳ (năm trước\|năm rồi\|năm ngoái) | phải test riêng: từng bị giải thành tháng trước |
| khoảng | từ X đến (hết)? Y; đến hết ngày X; tính đến X | |
| dịp | Tết Y, dịp Tết, trước Tết | từ `calendar.csv` |
| nhiều kỳ | X, Y, Z (lần lượt\|từng) | |
| tương đối với sự kiện | N ngày kể từ / ngay trước D | |

---

## 5. L5: primitive

Mỗi primitive nhận bảng chuẩn, kỳ và bộ lọc, trả số hoặc danh sách thô. Định dạng đáp án nằm ở `format_answer(value, type)`.

| Nhóm | Hàm |
|---|---|
| Đơn | `revenue`, `n_orders`, `aov`, `n_days_ge`, `n_orders_ge`, `top_day`, `peak_hour`, `n_cancelled`, `cancel_rate` |
| Mặt hàng | `top_items(by="qty_base"\|"amount"\|"n_orders", k)`, `qty_base(item\|unit\|group)`, `dead_items`, `price_effect(item, d, n)` |
| Khách | `n_customers`, `n_customers_ge(n)`, `share_customers_ge(n)`, `top_customers(by, k)`, `loyal`, `big_spender(min_orders)`, `fickle` |
| Nhân viên / thợ | `staff_revenue`, `staff_tickets`, `staff_workdays`, `commission_table` |
| Công nợ | `debt_at(x, by=None)`, `debt_added(period)`, `debt_paid(period)`, `debt_aging(x, bins)`, `installment_due/overdue/remaining` |
| Kho | `stock_at`, `import_minus_sales`, `stock_value`, `expiring_lots(x, n, value="lot_price")`, `nearest_expiry_lot` |
| Lãi | `gross_profit`, `net_profit`, `most_profitable` |
| Ghép | `pct_change(a, b)`, `diff`, `share`, `ratio`, `as_list`, `trend_up/down`, `per_weekday_avg` |
| Định dạng | `format_answer`: `Decimal`, nửa lên; tiền số nguyên; % 2 chữ số; `YYYY-MM` khi hỏi theo tháng; table cột đầu là tên |
| Dự phòng | `sql_on_views(sql)`: DuckDB trên bảng chuẩn, chỉ đọc, có timeout |

**Bài học khi làm tay:**
- Nợ dương tính bằng `> 0,5` đồng, không so với 0.
- Kỳ có một phần nằm trong dữ liệu thì tính trên phần có.
- `tie_check(series, k)` báo hoà ở ranh giới.

---

## 6. L6: kiểm và quyết định

| Hàm | Làm gì |
|---|---|
| `vote(results) → (value, agree)` | Bỏ phiếu theo giá trị đã định dạng |
| `sanity(spec, value) → [warn]` | Đếm âm, % ngoài miền, 0 trong kỳ có dữ liệu, hoà ở ranh giới top-k |
| `gate_unanswerable(spec, facets, cards) → reason?` | Kỳ ngoài dữ liệu; `facets.missing` chứa loại thông tin cần; khái niệm chủ quan **không có thẻ định nghĩa** |
| `decide(cands, calib) → answer \| abstain` | Ngưỡng theo nguồn × mức đồng thuận, hiệu chỉnh trên train |

---

## 7. Thứ tự code và test 14B

| Bước | Việc | Test không cần LLM | Test với LLM |
|---|---|---|---|
| 1 | `v4/cells.py` (sửa C2, C5), `grid.py`, `classify_rows`, `check_total_row`, `group_rows`, `ffill_group`, `split_cell`, `dup_report`, `profile_*` | Unit test theo mẫu hình mục 1; chạy trên 44 shop: in số dòng mỗi loại, so với số đã biết (dòng tổng, header lặp) | — |
| 2 | `registry.py` (schema tool), `loop.py` (vòng gọi tool, trần, log) | LLM giả phát lại kịch bản gọi tool | — |
| 3 | `recipe.py`, `l1_run` | Recipe viết tay cho vài bảng khó → `l1_invariants` qua | **14B trên Kaggle**: `l1_run` cho 44 shop, đo % bảng qua bất biến, số lần gọi tool, thời gian |
| 4 | L2: `role_candidates`, `field_candidates`, `join_coverage`, `alias_clusters`, `status_values`, `note_templates`, `unit_*`, `build_canonical` | So bảng chuẩn với dữ liệu làm sạch tay của 14 shop public: doanh thu, số đơn theo ngày | 14B: `l2_run` |
| 5 | L3 | `chunk_rulebook`, bảng tham số viết tay | 14B: `enrich_card`, `extract_params` |
| 6 | L4, L5, L6 | Spec chuẩn → primitive → đáp án 1,000 | 14B: `compose_spec` |
