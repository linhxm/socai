from socai.nlq.fastpath import parse
from socai.nlq.normalize import canon, detect_lang, untelex


def test_untelex():
    assert [untelex(t) for t in ["nhoms", "trowr", "leen", "tawng", "luwowtj", "naof", "thang", "khach"]] == \
           ["nhom", "tro", "len", "tang", "luot", "nao", "thang", "khach"]


def test_canon_teencode():
    assert "doanh thu" in canon("dthu bua qua cua cua hang la bao nhieu")
    assert "bao nhieu" in canon("tuần rồi bnhiu đơn z")
    assert "phan tram" in canon("bao nhiêu % đơn bị huỷ")


def test_lang():
    assert detect_lang("Doanh thu tháng này bao nhiêu?") == "vi"
    assert detect_lang("dthu thang nay bn z") == "vi_nodiacritics"
    assert detect_lang("What was the revenue last month?") == "en"
    assert detect_lang("上个月的营业额是多少?") == "zh"


def test_fastpath_intents():
    assert parse("Cho mình biết số đơn hôm qua.").metric == "order_count"
    assert parse("thang 9/2025 cua hang co bnhiu % don bi huy").metric == "cancel_rate"
    assert parse("thang 10/2025 bi huy may don").metric == "cancelled_count"
    assert parse("Doanh thu ngày nào cao nhất trong tháng 7 năm 2026?").metric == "best_day_revenue"
    ir = parse("So với tháng trước đó, doanh thu tháng 2/2026 tăng hay giảm mấy phần trăm?")
    assert (ir.metric, ir.compare) == ("revenue", "pct_change")
    assert parse("What was the revenue in 3/2025?").metric == "revenue"
    assert parse("2025年3月有多少个订单?").metric == "order_count"


def test_fastpath_refuses_filters_thresholds_and_entities():
    assert parse("doanh thu nhóm hàng lẩu tháng 4 năm 2025") is None
    assert parse("tháng 1/2025 có mấy đơn có dthu từ 560.000 đồng trowr leen") is None
    assert parse("dthu các đơn trả bằng tiền tươi tháng này") is None
    assert parse("doanh thu tháng 3/2026 của Sting dâu 330ml", {"sting dau 330ml"}) is None
    assert parse("xét 3 tháng 3/2025, 4/2025 và 5/2025, doanh thu tháng nào lớn nhất") is None
