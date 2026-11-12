"""Chạy cả một split: /data → /result/submission.jsonl + /result/logs/log.jsonl.

- Ghi đáp án NGAY sau mỗi câu (flush + fsync): hết 6 giờ thì câu chưa ghi bị tính abstain.
- Ngân sách thời gian toàn cục; còn ít thời gian thì chỉ chạy đường nhanh, hết hẳn thì ghi abstain.
- Thứ tự cố định (theo shop, rồi theo thứ tự trong questions.jsonl) để hai lần chạy cho cùng kết quả.
- Không có logic theo id câu hỏi / id shop (Rule 4): id chỉ dùng để tra file và ghi kết quả.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from socai.v2.bank import Bank
from socai.agent import Agent, AgentConfig
from socai.answer.decide import Policy
from socai.llm.client import LLM
from socai.rules.calendar import LunarCalendar
from socai.rules.rulebook import Rulebook


@dataclass
class RunConfig:
    time_budget_s: float = 5.5 * 3600  # chừa 30 phút cho nạp mô hình và dự phòng
    llm_reserve_s: float = 900  # còn ít hơn → tắt LLM, chỉ đường nhanh
    limit: int | None = None


def _append(f, rec: dict) -> None:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def run(data: Path, out: Path, llm: LLM | None, agent_cfg: AgentConfig, run_cfg: RunConfig,
        cache_dir: Path | None = None, policy_path: Path | None = None, t_start: float | None = None) -> dict:
    t_start = t_start or time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    questions = [json.loads(x) for x in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if run_cfg.limit:
        questions = questions[:run_cfg.limit]
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for q in questions:
        groups.setdefault(q["business_id"], []).append(q)

    cal_path = data / "calendar.csv"
    agent = Agent(Rulebook.load(data / "rulebook.md"),
                  LunarCalendar.load(cal_path) if cal_path.exists() else None,
                  Policy.load(policy_path), llm, agent_cfg, cache_dir, cal_path if cal_path.exists() else None,
                  bank=Bank.load(Path(__file__).parent / "v2" / "bank.json") if agent_cfg.use_v2 else None)
    stats = {"n": len(questions), "answered": 0, "abstain": 0, "unanswerable": 0, "errors": 0}
    done: set[str] = set()
    with (out / "submission.jsonl").open("w", encoding="utf-8") as fs, \
            (out / "logs" / "log.jsonl").open("w", encoding="utf-8") as fl:
        for biz_id, qs in groups.items():
            remaining = run_cfg.time_budget_s - (time.time() - t_start)
            if remaining <= 0:
                break
            session = None
            try:
                session = agent.open(data / "data" / biz_id)
            except Exception as e:  # noqa: BLE001
                stats["errors"] += 1
                for q in qs:
                    _append(fs, {"id": q["id"], "type": "abstain"})
                    _append(fl, {"id": q["id"], "answer": {"type": "abstain"}, "elapsed_ms": 0, "thinking": None,
                                 "code": None, "output": None, "notes": [f"không nạp được shop: {e}"[:300]]})
                    done.add(q["id"])
                continue
            for q in qs:
                remaining = run_cfg.time_budget_s - (time.time() - t_start)
                if remaining <= 0:
                    break
                agent.cfg.use_llm = agent_cfg.use_llm and llm is not None and remaining > run_cfg.llm_reserve_s
                sub, log = agent.answer(session, q["id"], q["question"], q["as_of_date"])
                _append(fs, sub)
                _append(fl, log)
                done.add(q["id"])
                key = sub["type"] if sub["type"] in ("abstain", "unanswerable") else "answered"
                stats[key] += 1
            agent.close(session)
        for q in questions:  # hết giờ: ghi abstain tường minh cho câu chưa làm
            if q["id"] not in done:
                _append(fs, {"id": q["id"], "type": "abstain"})
                stats["abstain"] += 1
    stats["seconds"] = round(time.time() - t_start, 1)
    (out / "logs" / "run_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats
