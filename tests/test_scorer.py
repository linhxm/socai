from socai.answer.decide import Candidate, Policy, normalize
from socai.eval.scorer import judge, score


def test_types_and_values():
    assert judge({"type": "number", "answer": 48250000}, {"type": "number", "answer": 48250000}) == "correct"
    assert judge({"type": "number", "answer": "48250000"}, {"type": "number", "answer": 48250000}) == "wrong"
    assert judge({"type": "percent", "answer": -12.45}, {"type": "percent", "answer": -12.45}) == "correct"
    assert judge({"type": "percent", "answer": 12.45}, {"type": "percent", "answer": -12.45}) == "wrong"
    assert judge({"type": "abstain"}, {"type": "number", "answer": 1}) == "abstain"
    assert judge(None, {"type": "number", "answer": 1}) == "abstain"


def test_string_alias_and_diacritics():
    ref = {"type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]}
    assert judge({"type": "string", "answer": "xi măng hà tiên pcb40 bao 50kg"}, ref) == "correct"
    assert judge({"type": "string", "answer": "SP0001"}, ref) == "correct"
    assert judge({"type": "string", "answer": "Xi mang Ha Tien PCB40 bao 50kg"}, ref) == "wrong"


def test_unanswerable_rules():
    ref = {"type": "unanswerable", "answer": None}
    assert judge({"type": "unanswerable"}, ref) == "correct"
    assert judge({"type": "number", "answer": 0}, ref) == "wrong"


def test_list_ordered_table_unordered():
    assert judge({"type": "list", "answer": ["A", "B"]}, {"type": "list", "answer": ["B", "A"]}) == "wrong"
    ref = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000],
                                                                                       ["Anh Tú", 1500000]]}}
    pred = {"type": "table", "answer": {"columns": ["Khách hàng", "Số tiền nợ"], "rows": [["Anh Tú", 1500000],
                                                                                        ["Chị Lan", 3200000]]}}
    assert judge(pred, ref) == "correct"


def test_penalized_accuracy():
    refs = {"a": {"type": "number", "answer": 1}, "b": {"type": "number", "answer": 2}, "c": {"type": "number", "answer": 3}}
    preds = {"a": {"type": "number", "answer": 1}, "b": {"type": "number", "answer": 5}}
    s = score(preds, refs)
    assert (s["correct"], s["wrong"], s["abstain"]) == (1, 1, 1)
    assert s["PA"] == 0


def test_normalize_rejects_bad_format_and_policy_threshold():
    assert normalize({"type": "number", "answer": "4,82 triệu"}) is None
    assert normalize({"type": "percent", "answer": 12.4449}) == {"type": "percent", "answer": 12.44}
    assert normalize({"type": "number", "answer": 18.0}) == {"type": "number", "answer": 18}
    pol = Policy(threshold=0.6, priors={"llm_code": 0.55, "fastpath": 0.9})
    ans = {"type": "number", "answer": 1}
    assert pol.decide([Candidate(ans, "llm_code")])[0] == {"type": "abstain"}
    assert pol.decide([Candidate(ans, "fastpath")])[0] == ans
    # hai đường độc lập đồng thuận → vượt ngưỡng
    assert pol.decide([Candidate(ans, "llm_code"), Candidate(dict(ans), "llm_ir")])[0] == ans
