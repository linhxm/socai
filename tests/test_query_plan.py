"""Bộ truy vấn (lệnh cơ bản), find.value và chuyển kế hoạch → QuerySpec, trên shop tổng hợp có sự thật.
Không cần LLM: kế hoạch JSON được viết tay, đúng dạng mô hình trả về."""

import random
from collections import defaultdict
from datetime import date

import pytest

from socai.exec.query import QuerySpec, execute
from socai.io.cells import round2
from socai.nlq.plan import scan_question, to_spec
from socai.profile.business import load_business
from socai.rules.timeexpr import Period, resolve
from socai.semantic.canonical import build
from socai.semantic.entities import EntityIndex
from socai.synth.world import make_orders, write_business

MAR = Period(date(2025, 3, 1), date(2025, 3, 31), "month")


@pytest.fixture(scope="module")
def shop(tmp_path_factory):
    root = tmp_path_factory.mktemp("q")
    rng = random.Random(11)
    orders = make_orders("grocery", date(2025, 2, 1), 3, rng)
    write_business(root, "syn_q", "csv", orders, rng)
    can = build(load_business(root / "data" / "syn_q"))
    return orders, can, EntityIndex(can)


def _valid(orders, p=MAR):
    return [o for o in orders if o.status in ("done", "shipping") and p.start <= o.at.date() <= p.end]


def test_revenue_filtered_by_product(shop):
    orders, can, _ = shop
    gold = sum(ln.amount for o in _valid(orders) for ln in o.lines if ln.product == "Sting dâu 330ml")
    r = execute(can, QuerySpec(measure="revenue", periods=[MAR], filters={"product_name": ["Sting dâu 330ml"]}))
    assert r.value == gold


def test_top_by_base_quantity_with_tie_rule(shop):
    orders, can, _ = shop
    q = defaultdict(int)
    for o in _valid(orders):
        for ln in o.lines:
            q[ln.product] += ln.qty * ln.factor  # §3: quy về đơn vị cơ bản
    gold = sorted(q, key=lambda k: (-q[k], k))[:3]
    r = execute(can, QuerySpec(measure="qty", shape="top", periods=[MAR], group_by="product", top_k=3))
    assert r.type == "list" and r.value[0] == gold[0] and set(r.value) <= set(q)


def test_share_and_split_and_trend(shop):
    orders, can, _ = shop
    tot = sum(o.revenue for o in _valid(orders))
    part = sum(ln.amount for o in _valid(orders) for ln in o.lines if ln.group == "sữa")
    r = execute(can, QuerySpec(measure="revenue", shape="share", periods=[MAR], filters={"product_group": ["sữa"]}))
    assert r.value == round2(100 * part / tot)
    sp = execute(can, QuerySpec(measure="revenue", shape="split", periods=[MAR],
                                split=[{"product_group": ["sữa"]}, {"product_group": ["gạo"]}]))
    assert sp.type == "list" and sp.value[0] == part
    months = [Period(date(2025, m, 1), date(2025, m, 28), "month") for m in (2, 3, 4)]
    vals = [sum(o.revenue for o in _valid(orders, p)) for p in months]
    tr = execute(can, QuerySpec(measure="revenue", shape="series", periods=months, trend="up"))
    assert tr.type == "boolean" and tr.value == (vals[0] < vals[1] < vals[2])


def test_outside_data_is_unanswerable(shop):
    _, can, _ = shop
    r = execute(can, QuerySpec(measure="revenue", periods=[Period(date(2023, 1, 1), date(2023, 1, 31), "month")]))
    assert r.type == "unanswerable"


def test_find_value_returns_exact_data_spelling(shop):
    _, _, idx = shop
    hit, _ = idx.resolve("sting dau 330ml")
    assert hit.value == "Sting dâu 330ml"
    hit, cands = idx.resolve("sữa")  # chỉ là nhóm hàng "sữa" (khớp đúng) chứ không phải tên món chứa "sữa"
    assert hit is not None and hit.field == "product_group"


def test_plan_to_spec_teencode_and_truncated_name(shop):
    _, can, idx = shop
    q = "th3/2025 ban dc bn tien mi hao hao tom chua cay z"
    raw = {"supported": True, "measure": "revenue", "shape": "value", "group_by": "none", "top_k": 0, "lowest": False,
           "mentions": ["mi hao hao"], "exclude_mentions": [], "size": "none", "compare": "none"}
    plan = to_spec(raw, q, resolve(q, date(2025, 4, 30)), idx, date(2025, 4, 30))
    assert plan.spec is not None, plan.problems
    assert plan.spec.filters == {"product_name": ["Mì Hảo Hảo tôm chua cay"]}  # tên đủ, lấy từ quét câu hỏi
    assert [(str(p.start), str(p.end)) for p in plan.spec.periods] == [("2025-03-01", "2025-03-31")]
    assert ("product_name", "Mì Hảo Hảo tôm chua cay") in scan_question(idx, q)


def test_plan_overrides_measure_for_unit_questions_and_listing(shop):
    _, _, idx = shop
    q = "tháng 3/2025 bán được bao nhiêu chai Sting dâu 330ml"
    raw = {"supported": True, "measure": "revenue", "shape": "value", "group_by": "none", "top_k": 0, "lowest": False,
           "mentions": ["Sting dâu 330ml"], "exclude_mentions": [], "size": "none", "compare": "none"}
    assert to_spec(raw, q, resolve(q, date(2025, 4, 30)), idx, date(2025, 4, 30)).spec.measure == "qty"
    q2 = "Liệt kê 3 khách chi tiêu nhiều nhất tháng 3/2025"
    raw2 = {**raw, "shape": "series", "group_by": "customer", "top_k": 3, "mentions": []}
    assert to_spec(raw2, q2, resolve(q2, date(2025, 4, 30)), idx, date(2025, 4, 30)).spec.shape == "top"


def test_plan_rejects_unresolvable_names(shop):
    _, _, idx = shop
    q = "doanh thu món bánh mì pate tháng 3/2025"
    raw = {"supported": True, "measure": "revenue", "shape": "value", "group_by": "none", "top_k": 0, "lowest": False,
           "mentions": ["bánh mì pate"], "exclude_mentions": [], "size": "none", "compare": "none"}
    plan = to_spec(raw, q, resolve(q, date(2025, 4, 30)), idx, date(2025, 4, 30))
    assert plan.spec is None and any("không neo được" in p for p in plan.problems)
