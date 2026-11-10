from datetime import date
from pathlib import Path

import pytest

from socai.rules.calendar import LunarCalendar
from socai.rules.timeexpr import resolve

TODAY = date(2026, 3, 4)  # thứ Tư


def span(q, today=TODAY, cal=None, i=0):
    p = resolve(q, today, cal).periods[i]
    return str(p.start), str(p.end)


def test_week_monday_start():
    assert span("tuần này bán bao nhiêu") == ("2026-03-02", "2026-03-04")
    assert span("sp bán dc nhiều nhất tuần rồi là gì") == ("2026-02-23", "2026-03-01")


def test_month_forms():
    assert span("tháng này") == ("2026-03-01", "2026-03-04")
    assert span("T7/2025 doanh thu") == ("2025-07-01", "2025-07-31")
    assert span("th 6/26 tăng trưởng") == ("2026-06-01", "2026-06-30")
    assert span("2/26 ngày nào dthu cao nhất z?") == ("2026-02-01", "2026-02-28")
    assert span("thang 10 nam 2025") == ("2025-10-01", "2025-10-31")
    assert span("tháng 5 bán được bao nhiêu") == ("2025-05-01", "2025-05-31")  # sau hôm nay → năm trước


def test_quarter_ytd_lastn_yesterday():
    assert span("quý 2/2026") == ("2026-04-01", "2026-06-30")
    assert span("từ đầu năm tới nay") == ("2026-01-01", "2026-03-04")
    assert span("7 ngày gần nhất") == ("2026-02-26", "2026-03-04")
    assert span("dthu bua qua") == ("2026-03-03", "2026-03-03")


def test_adjacent_mentions_all_found():
    ps = resolve("3 thang 4/2026, 5/2026, 6/2026 tung thang doanh thu", TODAY).periods
    assert [p.start.month for p in ps] == [4, 5, 6]


def test_month_to_date_compare_same_days():
    s = resolve("tháng này so với tháng trước doanh thu tăng bao nhiêu %", date(2026, 3, 31))
    assert (str(s.compare_to.start), str(s.compare_to.end)) == ("2026-02-01", "2026-02-28")


def test_relative_mention_becomes_comparison():
    s = resolve("So với tháng trước đó, doanh thu tháng 2/2026 tăng hay giảm mấy phần trăm?", date(2026, 6, 2))
    assert str(s.main.start) == "2026-02-01"
    assert str(s.compare_to.start) == "2026-01-01"


def test_same_period_last_year():
    s = resolve("doanh thu tháng 5/2026 so cùng kỳ năm trước", TODAY)
    assert str(s.compare_to.start) == "2025-05-01"


def test_chinese_and_english_months():
    assert span("2025年3月的营业额是多少?") == ("2025-03-01", "2025-03-31")
    assert span("How many orders in 3/2025?") == ("2025-03-01", "2025-03-31")


CAL = Path(__file__).resolve().parents[2] / "train" / "calendar.csv"


@pytest.mark.skipif(not CAL.exists(), reason="cần train/calendar.csv")
def test_tet_window_from_lunar_calendar():
    cal = LunarCalendar.load(CAL)
    a, b = cal.tet_window(2025)
    assert cal.rows[a][:3] == (23, 12, 2024)
    assert cal.rows[b][:3] == (10, 1, 2025)
    assert span("doanh thu tết 2025", date(2025, 6, 1), cal) == (str(a), str(b))
