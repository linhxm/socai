"""Hồi quy trên dữ liệu train thật của BTC (bỏ qua nếu không có thư mục train/ cạnh socai/).
Các con số ở đây là những lỗi thật đã gặp và đã sửa — giữ để không tái phát."""

from datetime import date
from pathlib import Path

import pytest

from socai.exec import metrics as M
from socai.profile.business import load_business
from socai.rules.timeexpr import Period
from socai.semantic.canonical import build

TRAIN = Path(__file__).resolve().parents[2] / "train" / "data"
pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="cần dữ liệu train của BTC")


def month(y, m, last):
    return Period(date(y, m, 1), date(y, m, last), "month")


def test_subtotal_rows_not_double_counted():
    """biz_130 có dòng 'TỔNG NGÀY…'/'Tổng tháng…' chèn giữa sổ; trước khi sửa doanh thu bị ×3."""
    can = build(load_business(TRAIN / "biz_130"))
    assert M.revenue(can, month(2026, 4, 30)).value == 1380833500


def test_tr_suffix_money_formats():
    """biz_129 ghi '1,156tr'; biz_127 ghi '1tr590' — trước khi sửa lệch ×40 và thiếu 8%."""
    assert M.revenue(build(load_business(TRAIN / "biz_129")), month(2026, 3, 31)).value == 1016912500
    assert M.revenue(build(load_business(TRAIN / "biz_127")), Period(date(2026, 4, 1), date(2026, 6, 30),
                                                                      "quarter")).value == 3267042000
