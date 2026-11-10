"""Đầu cuối trên dữ liệu tổng hợp (không cần dữ liệu BTC, không cần GPU) + test metamorphic đa ngữ."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from socai.agent import AgentConfig
from socai.eval.scorer import load_jsonl, score
from socai.runner import RunConfig, run
from socai.synth.world import generate_split


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate_split(root / "split", n_biz=3, seed=7)
    stats = run(root / "split", root / "out", None, AgentConfig(use_llm=False), RunConfig(), root / "cache")
    return root, stats


def test_runner_outputs_every_question_and_required_log_fields(synth_run):
    root, stats = synth_run
    qs = load_jsonl(root / "split" / "questions.jsonl")
    sub = load_jsonl(root / "out" / "submission.jsonl")
    assert set(sub) == set(qs)
    for ln in (root / "out" / "logs" / "log.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        assert {"id", "answer", "elapsed_ms", "thinking"} <= rec.keys()
        assert isinstance(rec["elapsed_ms"], int) and rec["elapsed_ms"] >= 0


def test_no_wrong_answers_on_synthetic_csv_xlsx_sqlite(synth_run):
    root, _ = synth_run
    s = score(load_jsonl(root / "out" / "submission.jsonl"), load_jsonl(root / "split" / "answers.jsonl"))
    assert s["wrong"] == 0, s
    assert s["correct"] >= 0.8 * s["n"], s


def test_metamorphic_same_answer_across_languages(synth_run):
    """vi có dấu / teencode / en / zh của cùng một câu phải cho CÙNG đáp án."""
    root, _ = synth_run
    qs = load_jsonl(root / "split" / "questions.jsonl")
    sub = load_jsonl(root / "out" / "submission.jsonl")
    groups = defaultdict(set)
    for qid, q in qs.items():
        a = sub[qid]
        if a["type"] != "abstain":
            groups[q["metamorphic_group"]].add(json.dumps(a.get("answer")))
    assert groups and all(len(v) == 1 for v in groups.values())


def test_rerun_is_deterministic(synth_run, tmp_path):
    root, _ = synth_run
    run(root / "split", tmp_path / "out2", None, AgentConfig(use_llm=False), RunConfig(), root / "cache")
    assert (tmp_path / "out2" / "submission.jsonl").read_text(encoding="utf-8") == \
           (root / "out" / "submission.jsonl").read_text(encoding="utf-8")
