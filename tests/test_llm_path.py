"""Đường LLM → code → sandbox, với LLM giả (không cần GPU): kiểm prompt, trích code, worker, sửa lỗi, log."""

import json

from socai.agent import AgentConfig
from socai.answer.decide import Policy
from socai.eval.scorer import load_jsonl
from socai.llm.client import FakeLLM
from socai.runner import RunConfig, run
from socai.synth.world import generate_split

GOOD = '''```python
def answer(ctx):
    o = ctx.orders
    p = ctx.period("tháng 2/2025")
    v = o[(o.d >= p.start) & (o.d <= p.end) & o.status.isin(["done", "shipping"])].revenue.sum()
    return {"type": "number", "answer": money(v)}
```'''
BROKEN = "```python\ndef answer(ctx):\n    return 1/0\n```"


def test_llm_code_path_with_repair(tmp_path):
    generate_split(tmp_path / "split", n_biz=1, seed=3)
    qs = [json.loads(x) for x in (tmp_path / "split" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    # chỉ giữ một câu mà đường nhanh từ chối (có tên nhóm hàng) để buộc đi đường LLM
    q = {**qs[0], "id": "llm_q", "question": "doanh thu nhóm hàng tháng 2/2025 tất cả các nhóm cộng lại"}
    (tmp_path / "split" / "questions.jsonl").write_text(json.dumps(q, ensure_ascii=False), encoding="utf-8")
    replies = iter([BROKEN, GOOD])
    llm = FakeLLM(lambda msgs: next(replies))
    pol_path = tmp_path / "pol.json"
    pol_path.write_text(json.dumps({"threshold": 0.5, "priors": {"llm_code": 0.7}}), encoding="utf-8")
    run(tmp_path / "split", tmp_path / "out", llm, AgentConfig(use_llm=True, use_plan=False, use_codegen=True),
        RunConfig(), tmp_path / "cache", pol_path)
    sub = load_jsonl(tmp_path / "out" / "submission.jsonl")["llm_q"]
    log = json.loads((tmp_path / "out" / "logs" / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sub["type"] == "number" and sub["answer"] > 0
    assert len(llm.calls) == 2  # lần 1 lỗi ZeroDivision → nhận traceback → sửa
    assert "ZeroDivisionError" in llm.calls[1][-1]["content"]
    assert "RULEBOOK" in llm.calls[0][0]["content"]  # rulebook nằm trong tiền tố hệ thống (CAG)
    assert isinstance(log["code"], list) and "ctx.period" in log["code"][-1]
    assert log["output"]["answer"] == sub["answer"]
    assert Policy().threshold == 0.6
