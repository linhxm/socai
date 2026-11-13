"""Chạy cả split: L3 một lần → mỗi shop L1 + L2 → mỗi câu L4 → L5 → L6. Ghi đáp án ngay sau mỗi câu.

- Nhiều server LLM (mỗi GPU một server): shop được chia đều, mỗi server một luồng. File nộp cuối cùng được
  ghi lại theo đúng thứ tự `questions.jsonl`.
- Ngân sách giờ: số mẫu mỗi câu tự giảm khi thời gian còn lại không đủ; hết giờ thì câu còn lại bỏ trống.
- Log mỗi câu: id, answer, elapsed_ms, thinking (nguyên văn, theo thứ tự gọi), code (các SQL đã chạy), output,
  inputs (kỳ, thực thể, bảng, luật), decision (nhóm tin cậy, mức đồng thuận, đáp án trước ngưỡng).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from socai.llm.client import LLM, extract_json
from socai.rules.calendar import LunarCalendar
from socai.v5 import l4_ask as L4
from socai.v5.bank import Bank
from socai.v5.l1_read import read_shop, read_shop_v2
from socai.v5.l2_model import Shop, build_shop
from socai.v5.l3_rules import Rules, load_rules
from socai.v5.l5_compute import SQLError, format_answer, run_sql
from socai.v5.l6_decide import Policy, answer_key, decide, gate_time


@dataclass
class Config:
    n_samples: int = 4
    temperature: float = 0.7
    thinking: bool = False
    max_tokens: int = 1200
    repairs: int = 1
    schema_budget: int = 9000  # ký tự lược đồ trong prompt (ngữ cảnh 8K token)
    time_budget_s: float = 5.4 * 3600
    bank_exclude_own: bool = False  # đo trên train: không lấy ví dụ của chính shop đang hỏi
    base: dict[str, dict] | None = None  # đáp án của hệ khác (v2): câu nó đã trả lời thì giữ nguyên, v5 chỉ làm câu nó bỏ trống
    verify: bool = False  # có base: v5 chạy cả câu base đã trả lời; cả k mẫu đồng thuận ở đáp án KHÁC base → bỏ trống
    v2_tables: bool = False  # viết SQL trên bảng đã làm sạch của v2 khi v2 nhận ra sổ bán hàng; không thì bảng thô
    shops: list[str] | None = None
    limit: int | None = None  # số câu tối đa mỗi shop


def one_sample(llm: LLM, msgs: list[dict], shop: Shop, cfg: Config, seed: int, temp: float, thinking: list,
               tokens: list | None = None) -> dict:
    """Một mẫu: LLM viết JSON → chạy SQL → định dạng. Lỗi chạy/định dạng thì trả lỗi cho LLM sửa `cfg.repairs` lần."""
    cur, err, spec = list(msgs), None, {}
    tokens = [] if tokens is None else tokens
    for _ in range(cfg.repairs + 1):
        c = llm.chat(cur, max_tokens=cfg.max_tokens, json_schema=L4.SPEC_SCHEMA, thinking=cfg.thinking,
                     temperature=temp, seed=seed)
        thinking.append(c.reasoning or "")
        tokens.append(c.prompt_tokens)
        try:
            spec = json.loads(c.content)
        except ValueError:
            spec = extract_json(c.content) or {}
        spec = spec if isinstance(spec, dict) else {}
        kind, unit, sql = spec.get("loai"), spec.get("don_vi") or "so", (spec.get("sql") or "").strip()
        if kind == "unanswerable":
            return {"answer": {"type": "unanswerable"}, "sql": "", "don_vi": unit, "nghi": spec.get("nghi", "")}
        try:
            cols, rows = run_sql(shop.con, sql)
            ans = format_answer(kind, unit, cols, rows, percent_scaled="100" in sql)
            return {"answer": ans, "sql": sql, "don_vi": unit, "nghi": spec.get("nghi", ""),
                    "rows": [[str(v) for v in r] for r in rows[:5]]}
        except SQLError as e:
            err = str(e)
        cur = cur + [{"role": "assistant", "content": c.content[:3000]},
                     {"role": "user", "content": f"Lỗi: {err}\nSửa lại và trả về JSON đầy đủ."}]
    return {"error": err, "sql": spec.get("sql", ""), "nghi": spec.get("nghi", "")}


def answer_question(q: dict, shop: Shop, rules: Rules, cal: LunarCalendar | None, llm: LLM | None, bank: Bank | None,
                    policy: Policy, cfg: Config, n_samples: int) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    ask = L4.analyse(q["question"], date.fromisoformat(q["as_of_date"]), shop, rules, cal, bank,
                     q["business_id"] if cfg.bank_exclude_own else None)
    log: dict = {"id": q["id"], "business_id": q["business_id"]}
    thinking: list = []
    samples: list[dict] = []
    why = gate_time(ask.time, shop.date_span, ask.today)
    if why:
        final = {"type": "unanswerable"}
        p = policy.p.get("gate_time", 0.9)
        decision = {"bucket": "gate_time", "agree": 1.0, "n": 0, "candidate": final, "p": p, "why": why}
        if p <= policy.threshold:
            final = {"type": "abstain"}
        code, output = "socai/v5/l6_decide.py::gate_time", why
    elif llm is None or n_samples <= 0:
        final, code, output = {"type": "abstain"}, None, None
        decision = {"bucket": "no_llm", "agree": 0.0, "n": 0, "candidate": None, "p": 0.0}
    else:
        msgs = L4.build_prompt(ask, shop, cfg.schema_budget)
        tokens: list[int] = []
        for i in range(n_samples):
            try:  # mẫu 0 greedy; các mẫu sau có nhiệt độ với seed cố định (chạy lại ra như cũ)
                samples.append(one_sample(llm, msgs, shop, cfg, seed=i, temp=0.0 if i == 0 else cfg.temperature,
                                          thinking=thinking, tokens=tokens))
            except Exception as e:  # noqa: BLE001 — một mẫu hỏng không làm hỏng cả câu
                samples.append({"error": f"{type(e).__name__}: {e}"[:300]})
        sqls = " ".join(s.get("sql") or "" for s in samples if "answer" in s)
        used = [t for t in shop.tables if re.search(rf'\b{re.escape(t)}\b', sqls)]  # chỉ bảng SQL thực sự dùng
        conf = min([shop.tables[t].confidence for t in used] or [1.0])
        d = decide(samples, policy, conf)
        final = d.answer
        decision = {"bucket": d.bucket, "agree": d.agree, "n": d.n, "candidate": d.candidate, "p": d.p,
                    "prompt_tokens": max(tokens or [0])}
        code = [s.get("sql") for s in samples]
        output = [s.get("answer") or s.get("error") for s in samples]
    log.update({
        "answer": final, "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "thinking": thinking if len(thinking) > 1 else (thinking[0] if thinking else None),
        "code": code, "output": output, "decision": decision, "samples": samples,
        "inputs": {"periods": [[str(p.start), str(p.end), p.kind] for p in ask.time.periods],
                   "compare_to": [str(ask.time.compare_to.start), str(ask.time.compare_to.end)]
                   if ask.time.compare_to else None,
                   "entities": ask.entities, "tables": ask.tables, "hint": ask.hint, "rules": ask.rules,
                   "examples": [e["q"] for e in ask.examples]}})
    sub = {"id": q["id"], "type": final["type"]}
    if "answer" in final:
        sub["answer"] = final["answer"]
    return sub, log


class _Budget:
    """Chia thời gian còn lại cho số câu còn lại; số mẫu mỗi câu giảm dần khi không đủ giờ."""

    def __init__(self, total_q: int, deadline: float, n_samples: int, n_workers: int = 1):
        self.left, self.deadline, self.n, self.workers = total_q, deadline, n_samples, n_workers
        self.per_sample = 8.0  # giây mỗi mẫu, cập nhật theo số đo thật
        self.lock = threading.Lock()

    def take(self) -> int:
        with self.lock:
            remain = self.deadline - time.time()
            if remain <= 5:
                return 0
            per_q = remain * self.workers / max(self.left, 1)  # các luồng chạy song song
            self.left -= 1
            return max(1, min(self.n, int(per_q / self.per_sample)))

    def skip(self, k: int) -> None:
        with self.lock:
            self.left -= k

    def observe(self, seconds: float, k: int) -> None:
        if k > 0:
            with self.lock:
                self.per_sample = 0.8 * self.per_sample + 0.2 * (seconds / k)


def run(data: Path, out: Path, llms: list[LLM], cfg: Config, cache: Path | None = None,
        policy_path: Path | None = None, bank_path: Path | None = None, t_start: float | None = None) -> dict:
    t_start = t_start or time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    qs = [json.loads(x) for x in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if cfg.shops:
        qs = [q for q in qs if q["business_id"] in cfg.shops]
    groups: dict[str, list[dict]] = {}
    for q in qs:
        g = groups.setdefault(q["business_id"], [])
        if not cfg.limit or len(g) < cfg.limit:
            g.append(q)
    qs = [q for q in qs if q in groups[q["business_id"]]]
    cal = LunarCalendar.load(data / "calendar.csv") if (data / "calendar.csv").exists() else None
    rules = load_rules(data / "rulebook.md")
    policy = Policy.load(policy_path)
    bank = Bank.load(bank_path)
    base = cfg.base or {}
    answered = {q["id"] for q in qs if base.get(q["id"], {}).get("type", "abstain") != "abstain"}
    kept = set() if cfg.verify else answered  # kept: không chạy v5
    n_workers = max(1, len(llms))
    budget = _Budget(len(qs) - len(kept), t_start + cfg.time_budget_s, cfg.n_samples, n_workers)
    print(f"[v5] {len(qs)} câu · giữ {len(kept)} đáp án của hệ nền · ngân hàng {len(bank.items) if bank else 0} ví dụ · "
          f"hiệu chỉnh: {policy_path if policy_path and policy_path.exists() else 'mặc định (chưa đo)'}", flush=True)
    subs: dict[str, dict] = {}
    wlock = threading.Lock()
    f_sub = (out / "submission.jsonl").open("w", encoding="utf-8")
    f_log = (out / "logs" / "log.jsonl").open("w", encoding="utf-8")
    f_shop = (out / "logs" / "shops.jsonl").open("w", encoding="utf-8")

    def write(f, rec: dict) -> None:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())

    def emit(sub: dict, log: dict | None) -> None:
        with wlock:
            subs[sub["id"]] = sub
            write(f_sub, sub)
            if log:
                write(f_log, log)

    def work(shops: list[str], llm: LLM | None) -> None:
        for biz in shops:
            for q in groups[biz]:
                if q["id"] in kept:
                    emit(base[q["id"]], {"id": q["id"], "answer": base[q["id"]], "elapsed_ms": 0, "thinking": None,
                                         "code": "hệ nền v2: xem /result/v2/logs", "output": None, "decision": {"bucket": "base"}})
            items = [q for q in groups[biz] if q["id"] not in kept]
            if not items:
                continue
            t0 = time.time()
            try:
                got = None
                if cfg.v2_tables:
                    try:
                        got = read_shop_v2(data / "data" / biz, cache)
                    except Exception:  # noqa: BLE001 — tầng dữ liệu v2 lỗi với shop lạ: dùng bảng thô
                        got = None
                readme, tables = got or read_shop(data / "data" / biz, cache)
                shop = build_shop(biz, readme, tables, rules.industries)
            except Exception as e:  # noqa: BLE001 — shop không đọc được: bỏ trống cả shop, không dừng cả lượt
                with wlock:
                    write(f_shop, {"shop": biz, "error": f"{type(e).__name__}: {e}"[:500]})
                budget.skip(len(items))
                for q in items:
                    emit({"id": q["id"], "type": "abstain"},
                         {"id": q["id"], "answer": {"type": "abstain"}, "elapsed_ms": 0, "thinking": None,
                          "code": None, "output": None, "decision": {"bucket": "shop_error"}})
                continue
            with wlock:
                write(f_shop, {"shop": biz, "seconds": round(time.time() - t0, 1), "industry": shop.industry,
                               "v2_tables": got is not None,
                               "date_range": shop.date_range,
                               "tables": {t.sql: {"source": t.orig, "rows": t.rows, "notes": t.notes,
                                                  "columns": {c.sql: [c.orig, c.type] for c in t.cols}}
                                          for t in shop.tables.values()}})
            print(f"[{time.time() - t_start:7.0f}s] {biz}: {len(shop.tables)} bảng, ngành={shop.industry or '-'}, "
                  f"dữ liệu={shop.date_range}, {len(items)} câu", flush=True)
            for q in items:
                k = budget.take()
                t1 = time.time()
                try:
                    sub, log = answer_question(q, shop, rules, cal, llm, bank, policy, cfg, k)
                except Exception as e:  # noqa: BLE001
                    sub = {"id": q["id"], "type": "abstain"}
                    log = {"id": q["id"], "answer": {"type": "abstain"}, "elapsed_ms": int((time.time() - t1) * 1000),
                           "thinking": None, "code": None, "output": f"{type(e).__name__}: {e}"[:300],
                           "decision": {"bucket": "error"}}
                if llm is not None and log.get("samples"):
                    budget.observe(time.time() - t1, len(log["samples"]))
                if q["id"] in answered:  # kiểm chéo: giữ đáp án base trừ khi v5 đồng thuận tuyệt đối ở đáp án khác
                    d, b = log.get("decision", {}), base[q["id"]]
                    veto = bool(d.get("bucket") == "all" and d.get("candidate")
                                and answer_key(d["candidate"]) != answer_key(b))
                    d.update(base=b, verify="veto" if veto else "keep")
                    sub = {"id": q["id"], "type": "abstain"} if veto else b
                    log["answer"] = {k: v for k, v in sub.items() if k != "id"}
                emit(sub, log)
                print(f"[{time.time() - t_start:7.0f}s] {biz} {len(subs)}/{len(qs)} {sub['type']:12s} "
                      f"k={k} {log.get('decision', {}).get('bucket', ''):9s} {q['question'][:60]}", flush=True)
            shop.con.close()

    order = list(groups)
    threads = [threading.Thread(target=work, args=(order[i::n_workers], llms[i] if llms else None))
               for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for f in (f_sub, f_log, f_shop):
        f.close()
    with (out / "submission.jsonl").open("w", encoding="utf-8") as f:  # ghi lại đúng thứ tự câu hỏi, đủ mọi câu
        for q in qs:
            f.write(json.dumps(subs.get(q["id"], {"id": q["id"], "type": "abstain"}), ensure_ascii=False) + "\n")
    stats = {"n": len(qs), "answered": 0, "abstain": 0, "unanswerable": 0}
    for q in qs:
        t = subs.get(q["id"], {"type": "abstain"})["type"]
        stats[t if t in ("abstain", "unanswerable") else "answered"] += 1
    stats["seconds"] = round(time.time() - t_start, 1)
    stats["kept_from_base"] = len(kept)
    stats["base_vetoed"] = sum(1 for i in answered if subs.get(i, {}).get("type") == "abstain")
    stats["bank_examples"] = len(bank.items) if bank else 0
    stats["calibrated"] = bool(policy_path and policy_path.exists())
    (out / "logs" / "run_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats
