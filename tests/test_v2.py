"""v2: bảng (views) + đại số + não, trên shop tổng hợp có sự thật. Không cần LLM thật: biểu thức viết tay,
hoặc FakeLLM trả biểu thức (kể cả biểu thức sai để thử vòng tự sửa)."""

import json
import random
from collections import defaultdict
from datetime import date

import pytest

from socai.io.cells import round2
from socai.llm.client import FakeLLM
from socai.nlq.plan import _series_periods
from socai.profile.business import load_business
from socai.rules.timeexpr import Period, Resolver, TimeSpec
from socai.semantic.canonical import build
from socai.synth.world import make_orders, write_business
from socai.v2.algebra import Ctx, execute, signature
from socai.v2.brain import lint, pick_views, think
from socai.v2.views import build_views

MAR = Period(date(2025, 3, 1), date(2025, 3, 31), "month")
FEB = Period(date(2025, 2, 1), date(2025, 2, 28), "month")
TODAY = date(2025, 4, 15)


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2")
    rng = random.Random(11)
    orders = make_orders("grocery", date(2025, 2, 1), 3, rng)
    write_business(root, "syn_v2", "csv", orders, rng)
    biz = load_business(root / "data" / "syn_v2")
    return orders, build_views(biz, build(biz))


def ctx(views, periods=(MAR,), compare_to=None):
    return Ctx(views, TimeSpec(list(periods), compare_to), Resolver(TODAY).resolve, _series_periods)


def _valid(orders, p=MAR):
    return [o for o in orders if o.status in ("done", "shipping") and p.start <= o.at.date() <= p.end]


def q(**kw):
    return {"from": "ban", "time": "$ky", "out": "value", **kw}


def test_revenue_default_excludes_cancelled_and_returned(shop):
    orders, views = shop
    gold = sum(ln.amount for o in _valid(orders) for ln in o.lines)
    r = execute(q(agg=["sum", "thanh_tien"]), ctx(views))
    assert (r.type, r.value) == ("number", gold)


def test_filter_and_top_by_base_quantity(shop):
    orders, views = shop
    gold = sum(ln.amount for o in _valid(orders) for ln in o.lines if ln.product == "Sting dâu 330ml")
    r = execute(q(where=[["mat_hang", "=", "sting dau 330ml"]], agg=["sum", "thanh_tien"]), ctx(views))
    assert r.value == gold  # tên trong câu không dấu vẫn neo đúng giá trị trong file
    qty = defaultdict(int)
    for o in _valid(orders):
        for ln in o.lines:
            qty[ln.product] += ln.qty * ln.factor  # §3: quy về đơn vị cơ bản
    r = execute(q(group="mat_hang", agg=["sum", "so_luong"], limit=3, out="names"), ctx(views))
    assert r.type == "list" and r.value[0] == sorted(qty, key=lambda k: (-qty[k], k))[0]


def test_share_compare_and_cancel_rate_are_compositions(shop):
    orders, views = shop
    tot = sum(o.revenue for o in _valid(orders))
    part = sum(ln.amount for o in _valid(orders) for ln in o.lines if ln.group == "sữa")
    share = {"op": "pct", "a": q(where=[["nhom_hang", "=", "sữa"]], agg=["sum", "thanh_tien"]),
             "b": {"like": "a", "drop": ["nhom_hang"]}}
    assert execute(share, ctx(views)).value == round2(100 * part / tot)
    prev = sum(o.revenue for o in _valid(orders, FEB))
    cmp_ = {"op": "pct_change", "a": q(agg=["sum", "thanh_tien"]), "b": {"like": "a", "time": "$ky_so"}}
    r = execute(cmp_, ctx(views, compare_to=FEB))
    assert (r.type, r.value) == ("percent", round2(100 * (tot - prev) / prev))
    allm = [o for o in orders if MAR.start <= o.at.date() <= MAR.end]
    rate = {"op": "pct", "a": {"from": "don", "where": [["trang_thai", "=", "huỷ"]], "time": "$ky",
                               "agg": ["count", "ma_don"]}, "b": {"like": "a", "where": [["trang_thai", "*"]]}}
    gold = round2(100 * sum(o.status == "cancelled" for o in allm) / len(allm))
    assert execute(rate, ctx(views)).value == gold


def test_having_count_and_date_where_and_out_of_range(shop):
    orders, views = shop
    per = defaultdict(int)
    for o in _valid(orders):
        if o.customer:
            per[o.customer] += 1
    r = execute({"from": "don", "time": "$ky", "group": "khach", "agg": ["count", "ma_don"], "having": [">=", 2],
                 "out": "count"}, ctx(views))
    assert r.value == sum(v >= 2 for k, v in per.items() if k.lower() not in ("khách lẻ", ""))
    d = date(2025, 3, 10)
    gold = sum(o.revenue for o in _valid(orders) if o.at.date() == d)
    r = execute(q(where=[["ngay", "=", "2025-03-10"]], agg=["sum", "thanh_tien"]), ctx(views))
    assert r.value == gold  # lọc ngày bằng where thay cho time
    far = Period(date(2030, 1, 1), date(2030, 1, 31), "month")
    assert execute(q(agg=["sum", "thanh_tien"]), ctx(views, periods=(far,))).type == "unanswerable"  # §7


def test_empty_filter_abstains_not_zero(shop):
    _, views = shop
    from socai.exec.metrics import Unsupported
    from socai.v2.algebra import V2Error
    with pytest.raises(V2Error):  # mẫu không khớp giá trị nào của cột: báo lỗi có gợi ý để LLM sửa
        execute(q(where=[["mat_hang", "~", "khong ton tai xyz"]], agg=["sum", "thanh_tien"]), ctx(views))
    one_day = Period(date(2025, 3, 10), date(2025, 3, 10), "day")
    with pytest.raises(Unsupported):  # lọc đúng nhưng kỳ không có dòng nào: bỏ trống, không trả 0 (§7)
        execute(q(where=[["mat_hang", "=", "Sting dâu 330ml"], ["thanh_tien", "<", 0]], agg=["sum", "thanh_tien"]),
                ctx(views, periods=(one_day,)))


def test_name_variants_share_one_key_and_except(shop):
    orders, views = shop
    ban = views["ban"]
    assert "mat_hang" in ban.key_of  # §8: lọc/nhóm theo khoá mặt hàng, không theo một cách viết
    r = execute(q(where=[["mat_hang", "$", "Sting dâu 330ml"]], agg=["sum", "thanh_tien"]), ctx(views))
    gold = sum(ln.amount for o in _valid(orders) for ln in o.lines if ln.product == "Sting dâu 330ml")
    assert r.value == gold  # '$' trên một tên có thật được hiểu là '='
    sold_before = {ln.product for o in _valid(orders, FEB) for ln in o.lines}
    sold_in = {ln.product for o in _valid(orders) for ln in o.lines}
    unsold = {"op": "except", "out": "count",
              "a": q(time="$truoc_ky", group="mat_hang", agg=["sum", "so_luong"], out="names"),
              "b": {"like": "a", "time": "$ky"}}
    assert execute(unsold, ctx(views)).value == len(sold_before - sold_in)
    best = execute(q(group="ngay", agg=["sum", "thanh_tien"], out="name"), ctx(views))
    assert best.type == "date"  # "ngày nào cao nhất" trả kiểu ngày dù LLM ghi out=name


def test_lint_inclusive_bounds_from_question():
    notes = []
    assert lint({"where": [["x", ">", 5]]}, "đơn từ 5 trowr leen", notes)["where"][0][1] == ">="
    assert lint({"having": [">=", 5]}, "khách mua quá 5 đơn", notes)["having"][0] == ">"
    assert signature({"op": "pct", "a": q(agg=["sum", "thanh_tien"])}) == "pct:sum:value"


def test_brain_repairs_once_with_error_feedback(shop):
    orders, views = shop
    gold = sum(ln.amount for o in _valid(orders) for ln in o.lines)
    answers = iter([json.dumps(q(agg=["sum", "doanh_thu"])), json.dumps(q(agg=["sum", "thanh_tien"]))])
    llm = FakeLLM(lambda msgs: next(answers))
    th = think(llm, views, "tháng 3 bán được bao nhiêu tiền", ctx(views))
    assert th.result is not None and th.result.value == gold and len(th.attempts) == 2
    assert "không có cột 'doanh_thu'" in llm.calls[1][-1]["content"]  # lỗi + danh sách cột đưa lại cho LLM
    assert [v.name for v in pick_views(views, "doanh thu")][:1] == ["ban"]


def test_camelcase_headers_reuse_train_lexicon():
    from socai.semantic.lexicon import tag_from_header
    assert [tag_from_header(c) for c in ("KenhBanHang", "PhuongThucTT", "TinhTrang", "MaKhachHang")] == \
        ["channel", "payment_method", "status", "customer_id"]
    assert tag_from_header("IDCuaHang") is None  # không đoán khi từ điển không có
