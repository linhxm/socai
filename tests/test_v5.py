"""v5 trên dữ liệu tổng hợp (không cần dữ liệu thi, không cần GPU): từng tầng và đầu cuối với LLM giả."""

import json
import re
from datetime import date

import pytest

from socai.eval.scorer import load_jsonl, score
from socai.llm.client import FakeLLM
from socai.rules.timeexpr import Period, TimeSpec
from socai.synth.world import generate_split
from socai.v5 import l4_ask as L4
from socai.v5.bank import Bank, build
from socai.v5.l1_read import read_shop
from socai.v5.l2_model import build_shop
from socai.v5.l3_rules import parse_rules
from socai.v5.l5_compute import SQLError, format_answer, half_up, run_sql
from socai.v5.l6_decide import Policy, answer_key, calibrate, decide, gate_time
from socai.v5.pipeline import Config, run

VALID = "khoa(tinh_trang) NOT IN ('da huy', 'huy don', 'tra hang', 'doi tra')"


def fake_sql(messages: list[dict]) -> str:
    """LLM giả: đọc tên bảng và kỳ từ prompt, viết đúng SQL cho hai dạng câu của bộ tổng hợp."""
    user = messages[1]["content"]
    table = re.search(r"BẢNG (\w+)", user).group(1)
    m = re.search(r"kỳ 1: (\S+) → (\S+)", user)
    if not m:
        return json.dumps({"nghi": "không rõ kỳ", "loai": "number", "don_vi": "so", "sql": "SELECT"})
    where = f"ngay BETWEEN DATE '{m.group(1)}' AND DATE '{m.group(2)}' AND {VALID}"
    q = user.split("\n")[0].lower()
    if any(w in q for w in ("đơn", "don", "orders", "订单")):
        return json.dumps({"nghi": "đếm đơn", "loai": "number", "don_vi": "dem",
                           "sql": f"SELECT count(DISTINCT ma_hd) FROM {table} WHERE {where}"})
    return json.dumps({"nghi": "doanh thu", "loai": "number", "don_vi": "tien",
                       "sql": f"SELECT sum(thanh_tien) FROM {table} WHERE {where}"})


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    root = tmp_path_factory.mktemp("v5")
    generate_split(root / "split", n_biz=3, seed=7)
    return root


@pytest.fixture(scope="module")
def shops(split):
    out = {}
    for d in sorted((split / "split" / "data").iterdir()):
        readme, tables = read_shop(d)
        out[d.name] = (tables, build_shop(d.name, readme, tables))
    return out


def test_l1_types_totals_and_group_fill(shops):
    for biz, (tables, shop) in shops.items():
        assert len(tables) == 1, biz
        t = tables[0]
        kinds = {c.name: c.kind for c in t.cols}
        assert kinds["Ngày"] == "date" and kinds["Thành tiền"] == "number" and kinds["Mã HĐ"] == "text", (biz, kinds)
        assert not t.df["Hàng"].map(lambda v: str(v).startswith("Tổng")).any(), "còn dòng tổng"
        assert t.df["Khách"].notna().all() and t.df["Tình trạng"].notna().all(), "chưa điền dòng tiếp nối"
        assert all(c.unparsed == 0 for c in t.cols), [(c.name, c.unparsed) for c in t.cols if c.unparsed]
        one_status = shop.con.execute(
            f"SELECT max(n) FROM (SELECT count(DISTINCT tinh_trang) n FROM {next(iter(shop.tables))} GROUP BY ma_hd)"
        ).fetchone()[0]
        assert one_status == 1, "một đơn phải có đúng một trạng thái sau khi điền"


def test_l2_catalog_index_and_date_range(shops):
    _, shop = shops["syn_001"]
    t = next(iter(shop.tables.values()))
    cols = {c.sql: c for c in t.cols}
    assert cols["ngay"].type == "DATE" and cols["thanh_tien"].type == "BIGINT"
    assert cols["tinh_trang"].values and not cols["ma_hd"].values
    assert shop.date_range == ("2025-01-01", "2025-03-31")
    assert shop.con.execute("SELECT khoa('  Đã   HUỶ ')").fetchone()[0] == "da huy"
    ents = L4.link_entities("doanh thu tra dao cam sa thang 2 la bao nhieu", shop)
    assert any(e["cot"] == "hang" and "Trà đào cam sả (M)" in e["gia_tri"] for e in ents), ents


def test_l3_cards_and_industry_filter():
    rules = parse_rules("""# Rulebook
## 1. Đơn hàng
| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn không huỷ và không trả hàng |
- Bán chạy tính theo số lượng đơn vị cơ bản.

## 6. Theo ngành
### Nhà thuốc
| Luật | Mặc định |
|---|---|
| Hàng hết hạn | tính doanh thu theo hạn dùng của lô |
### Sửa xe
| Luật | Mặc định |
|---|---|
| Lượt xe | mỗi phiếu là một lượt, tính doanh thu theo phiếu |
""")
    assert rules.industries == ["Nhà thuốc", "Sửa xe"]
    assert len(rules.cards) == 4
    got = rules.search("doanh thu tháng này", "Nhà thuốc")
    assert any("không huỷ" in c.text for c in got)
    assert all(c.industry in (None, "Nhà thuốc") for c in got)
    assert any(c.industry for c in rules.search("doanh thu tháng này", None))  # không rõ ngành: giữ thẻ ngành


def test_l4_hint_and_prompt(shops):
    assert L4.answer_hint("Tỉ lệ đơn huỷ tháng 3 là bao nhiêu?") == "percent"
    assert L4.answer_hint("thang 2 co ban duoc mi goi khong") == "boolean"
    assert L4.answer_hint("Ngày nào doanh thu cao nhất?") == "date"
    assert L4.answer_hint("Doanh thu tháng 3/2025 là bao nhiêu?") == ""
    _, shop = shops["syn_002"]
    ask = L4.analyse("Doanh thu tháng 3/2025 là bao nhiêu?", date(2025, 4, 30), shop, parse_rules(""), None, None)
    user = L4.build_prompt(ask, shop)[1]["content"]
    assert "kỳ 1: 2025-03-01 → 2025-03-31" in user and "BẢNG sobanhang" in user and "thanh_tien BIGINT" in user


def test_l5_sql_guard_and_format(shops):
    _, shop = shops["syn_001"]
    for bad in ("DROP TABLE ban_hang", "SELECT 1; SELECT 2", ""):
        with pytest.raises(SQLError):
            run_sql(shop.con, bad)
    assert run_sql(shop.con, "SELECT 'a;b' AS x")[1] == [("a;b",)]
    assert half_up(2.5, 0) == 3 and half_up(0.125, 2) == 0.13 and half_up(1e6 + 0.5, 0) == 1000001
    assert format_answer("number", "tien", ["x"], [(1234.5,)]) == {"type": "number", "answer": 1235}
    assert format_answer("number", "so", ["x"], [(2.0,)]) == {"type": "number", "answer": 2}
    assert format_answer("percent", "phan_tram", ["x"], [(12.345,)]) == {"type": "percent", "answer": 12.35}
    assert format_answer("date", "so", ["x"], [(date(2025, 3, 1),)])["answer"] == "2025-03-01"
    assert format_answer("date", "so", ["x"], [("2025-03",)])["answer"] == "2025-03"
    assert format_answer("list", "so", ["x"], [("A",), ("B",)])["answer"] == ["A", "B"]
    assert format_answer("boolean", "so", ["x"], [(True,)])["answer"] is True
    for kind, rows in (("number", [(1,), (2,)]), ("number", [(None,)]), ("number", []), ("list", [("a", 1)])):
        with pytest.raises(SQLError):
            format_answer(kind, "so", ["x"], rows)


def test_l6_vote_threshold_and_time_gate():
    pol = Policy()
    a, b = {"type": "number", "answer": 10}, {"type": "number", "answer": 11}
    assert decide([{"answer": a}] * 4, pol).bucket == "all" and decide([{"answer": a}] * 4, pol).answer == a
    d = decide([{"answer": a}, {"answer": a}, {"answer": a}, {"answer": b}], pol)
    assert d.bucket == "most" and d.answer == a
    d = decide([{"answer": a}, {"answer": b}, {"error": "x"}, {"error": "y"}], pol)
    assert d.bucket == "split" and d.answer == {"type": "abstain"} and d.candidate == a
    assert decide([{"error": "x"}], pol).answer == {"type": "abstain"}
    assert decide([{"answer": {"type": "unanswerable"}}] * 4, pol).answer == {"type": "abstain"}  # mặc định thận trọng
    assert decide([{"answer": a}] * 4, pol, confidence=0.8).answer == a
    assert decide([{"answer": a}, {"answer": a}, {"answer": b}], pol, confidence=0.8).answer == {"type": "abstain"}
    assert answer_key({"type": "string", "answer": "Trà  Đào!"}) == answer_key({"type": "string", "answer": "trà đào"})
    assert answer_key({"type": "number", "answer": 3}) == answer_key({"type": "number", "answer": 3.001})
    spec = TimeSpec([Period(date(2024, 1, 1), date(2024, 1, 31), "month")])
    assert gate_time(spec, ("2025-01-01", "2025-03-31"))
    assert gate_time(TimeSpec([Period(date(2025, 3, 20), date(2025, 4, 10), "range")]), ("2025-01-01", "2025-03-31")) is None
    assert gate_time(TimeSpec(), ("2025-01-01", "2025-03-31")) is None
    # kỳ sau hôm nay không bị chặn (thường là lỗi đọc ngày; để LLM quyết)
    assert gate_time(TimeSpec([Period(date(2026, 12, 1), date(2026, 12, 31), "month")]), ("2025-01-01", "2025-03-31"),
                     date(2026, 1, 15)) is None


def test_review_fixes(shops):
    import time

    import pandas as pd

    from socai.v5.l1_read import Col, _fill_groups
    from socai.v5.pipeline import _Budget

    shop = next(iter(shops.values()))[1]
    # số 4 chữ số không phải năm ("mã sp 1203", "2000 nghìn") không tạo kỳ giả
    for q in ("đơn có mã sp 1203 bán bao nhiêu", "bao nhiêu đơn trên 2000 nghìn"):
        assert L4.analyse(q, date(2025, 3, 31), shop, parse_rules(""), None, None).time.periods == []
    # cột tuỳ chọn trống cùng nhau (ngày trả + lý do trả) không bị điền xuôi
    df = pd.DataFrame({"ma": [f"A{i}" for i in range(10)], "ngay_tra": ["2025-01-01"] + [None] * 9,
                       "ly_do": ["sai size"] + [None] * 9})
    cols = [Col("ma", "text"), Col("ngay_tra", "text"), Col("ly_do", "text")]
    assert _fill_groups(df, cols) == [] and df["ly_do"].isna().sum() == 9
    # percent dạng 0..1 bị trả lại cho LLM sửa; cột tháng trả 'YYYY-MM'
    with pytest.raises(SQLError):
        format_answer("percent", "phan_tram", ["ti_le"], [(0.125,)], percent_scaled=False)
    assert format_answer("date", "so", ["thang"], [(date(2025, 3, 1),)])["answer"] == "2025-03"
    # ngân sách tính cả số luồng song song: 2 luồng, đủ giờ cho 4 mẫu/câu
    b = _Budget(100, time.time() + 100 * 8 * 4 / 2 + 60, 4, n_workers=2)
    assert b.take() == 4


def test_base_floor_keeps_v2_answers(split, tmp_path):
    qs = [json.loads(x) for x in (split / "split" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    base = {qs[0]["id"]: {"id": qs[0]["id"], "type": "number", "answer": 123456789}}
    run(split / "split", tmp_path, [FakeLLM(fake_sql)], Config(n_samples=1, base=base), split / "cache")
    sub = load_jsonl(tmp_path / "submission.jsonl")
    assert sub[qs[0]["id"]]["answer"] == 123456789 and len(sub) == len(qs)
    assert json.loads((tmp_path / "logs" / "run_summary.json").read_text())["kept_from_base"] == 1


@pytest.fixture(scope="module")
def e2e(split):
    llms = [FakeLLM(fake_sql), FakeLLM(fake_sql)]  # hai "server": shop chia cho hai luồng
    stats = run(split / "split", split / "out", llms, Config(n_samples=3), split / "cache")
    return split, stats


def test_e2e_every_question_in_order_with_log_fields(e2e):
    root, stats = e2e
    qs = [json.loads(x) for x in (root / "split" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    sub = [json.loads(x) for x in (root / "out" / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [s["id"] for s in sub] == [q["id"] for q in qs]
    assert stats["n"] == len(qs) == stats["answered"] + stats["abstain"] + stats["unanswerable"]
    logs = [json.loads(x) for x in (root / "out" / "logs" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["id"] for r in logs} == {q["id"] for q in qs}
    for r in logs:
        assert {"id", "answer", "elapsed_ms", "thinking", "code", "output", "decision"} <= r.keys()


def test_e2e_correct_on_csv_xlsx_sqlite_and_never_wrong(e2e):
    root, _ = e2e
    s = score(load_jsonl(root / "out" / "submission.jsonl"), load_jsonl(root / "split" / "answers.jsonl"))
    assert s["wrong"] == 0, s
    assert s["correct"] == s["n"], s


def test_bank_and_calibration_from_train_log(e2e, tmp_path):
    root, _ = e2e
    data, log = root / "split", root / "out" / "logs" / "log.jsonl"
    n = build(log, data / "questions.jsonl", data / "answers.jsonl", tmp_path / "bank.json")
    assert n > 0
    bank = Bank.load(tmp_path / "bank.json")
    hit = bank.search("Doanh thu tháng 5/2025 là bao nhiêu?", k=2)
    assert hit and "sum(thanh_tien)" in hit[0]["sql"]
    assert all(h["biz"] != "syn_001" for h in bank.search("doanh thu thang 1", 4, exclude_biz="syn_001"))
    rep = calibrate(log, data / "answers.jsonl", tmp_path / "cal.json")
    assert rep["all"]["precision"] == 1.0
    assert Policy.load(tmp_path / "cal.json").p["all"] > 0.9


def test_time_budget_exhausted_abstains(split, tmp_path):
    stats = run(split / "split", tmp_path / "out", [FakeLLM(fake_sql)], Config(n_samples=3, time_budget_s=0.0))
    assert stats["answered"] == 0 and stats["abstain"] + stats["unanswerable"] == stats["n"]
