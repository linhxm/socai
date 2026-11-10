"""Bộ đọc mở rộng (Track B) và phát hiện header — test được không cần OCR engine."""

from pathlib import Path

from socai.io.docs import Box, boxes_to_table, read_xml
from socai.io.readers import detect_header, sheet_period
from socai.profile.readme import parse_readme


def test_boxes_to_table_rebuilds_grid():
    boxes = [Box("Ngày", 10, 10, 60, 30), Box("Hàng", 200, 10, 260, 30), Box("Tiền", 500, 11, 560, 31),
             Box("01/07", 10, 50, 60, 70, 0.9), Box("Sting", 200, 52, 250, 72, 0.8), Box("dâu", 255, 52, 290, 72),
             Box("22.000", 500, 50, 570, 70, 0.95)]
    grid, conf = boxes_to_table(boxes)
    assert grid == [["Ngày", "Hàng", "Tiền"], ["01/07", "Sting dâu", "22.000"]]
    assert 0.8 < conf <= 1


def test_xml_einvoice_flatten(tmp_path: Path):
    p = tmp_path / "hd.xml"
    p.write_text("""<HDon><DLHDon><TTChung><SHDon>123</SHDon><NLap>2026-08-15</NLap></TTChung>
      <NDHDon><DSHHDVu>
        <HHDVu><THHDVu>Xi măng PCB40</THHDVu><SLuong>10</SLuong><ThTien>950000</ThTien></HHDVu>
        <HHDVu><THHDVu>Cát xây</THHDVu><SLuong>2</SLuong><ThTien>600000</ThTien></HHDVu>
      </DSHHDVu></NDHDon></DLHDon></HDon>""", encoding="utf-8")
    t = read_xml(p, set())[0]
    assert len(t.df) == 2
    assert set(t.df["THHDVu"]) == {"Xi măng PCB40", "Cát xây"}
    assert (t.df["SHDon"] == "123").all()  # thông tin đầu phiếu lặp trên mọi dòng


def test_detect_header_skips_title_and_group_rows():
    rows = [("SỔ CHI TIẾT BÁN HÀNG",), ("Kỳ báo cáo: toàn bộ", "Kênh bán: Tất cả"), (),
            ("Ngày", "Mã đơn hàng", "Tên hàng hóa", "Số lượng", "Thành tiền"),
            ("01/02/2025", "ORD000001", "Ống PVC", 2, "130,000.0000")]
    assert detect_header(rows) == 3
    two_level = [("Hàng", None, None, "Tiền", None), ("Hàng", "so_luong", "Unit", "Trị giá", "Status"),
                 ("Cappuccino (L)", -1, "ly", "-53.500đ", "Đổi trả")]
    assert detect_header(two_level) == 1


def test_sheet_period():
    assert sheet_period("T02-2025") == (2025, 2)
    assert sheet_period("Thang 02 2026") == (2026, 2)
    assert sheet_period("Tháng 01-2025") == (2025, 1)
    assert sheet_period("Sheet1") == (None, None)


def test_readme_parser():
    rd = parse_readme("""# Dữ liệu x

## File
- `ban_hang.csv` (csv)

## Cột trong `ban_hang.csv`
- **Thanh tien**: thành tiền sau giảm
- **Mã hóa đơn**: mã đơn

## Quy ước
- Dữ liệu có đến hết 24/08/2026.
- **Quy đổi đơn vị**: số lượng ghi theo đơn vị của dòng:
  - Coca Cola 330ml: 1 thùng = 24 lon
""")
    assert rd.files == ["ban_hang.csv"]
    assert rd.columns["ban_hang.csv"]["Thanh tien"] == "thành tiền sau giảm"
    assert str(rd.data_end) == "2026-08-24"
    assert (rd.units[0].product, rd.units[0].big_unit, rd.units[0].factor) == ("Coca Cola 330ml", "thùng", 24)
