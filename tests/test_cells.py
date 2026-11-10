from datetime import date

import pytest

from socai.io.cells import match_key, parse_date, parse_number, parse_qty, round2, to_int_money


@pytest.mark.parametrize("raw,want", [
    ("86k", 86000), ("53.500đ", 53500), ("-53.500đ", -53500), ("22,000", 22000), ("2,760,000.0000", 2760000),
    ("9.656.666đ", 9656666), ("(1,000)", -1000), ("1,5", 1.5), ("1.5", 1.5), ("1.500", 1500),
    # các định dạng gây lỗi thật trên train (đã sửa): '1,156tr' từng bị đọc thành 1,156 tỷ; '1tr590' → NaN
    ("1,156tr", 1_156_000), ("2,89tr", 2_890_000), ("1tr590", 1_590_000), ("2ty3", 2_300_000_000),
    ("15000.0", 15000), (15000.0, 15000), ("", None), ("abc", None),
])
def test_parse_number(raw, want):
    assert parse_number(raw) == want


def test_parse_qty():
    assert parse_qty("5 chai") == (5.0, "chai")
    assert parse_qty(3) == (3.0, "")
    assert parse_qty("hai") == (None, "")


@pytest.mark.parametrize("raw,year,want", [
    (45658, None, date(2025, 1, 1)), (45717, None, date(2025, 3, 1)), ("45717", None, date(2025, 3, 1)),
    ("01/02/2026", None, date(2026, 2, 1)),
    ("1 thg 7 2025", None, date(2025, 7, 1)), ("2026-08-15", None, date(2026, 8, 15)),
    ("28/4", 2025, date(2025, 4, 28)), ("28/4", None, None), ("31/02/2026", None, None),
])
def test_parse_date(raw, year, want):
    assert parse_date(raw, year) == want


def test_rounding_half_up():
    assert to_int_money(1234.5) == 1235
    assert round2(1.005) == 1.01
    assert round2(-12.345) == -12.35


def test_match_key_keeps_diacritics():
    assert match_key("Sting dâu 330ml!") == "sting dâu 330ml"
    assert match_key("Sting dâu 330ml") != match_key("Sting dầu 330ml")
