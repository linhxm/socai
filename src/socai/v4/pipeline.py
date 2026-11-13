"""Chạy cả split bằng v4: L3 một lần → mỗi shop L1 + L2 → mỗi câu L4 → L5 → L6. Ghi đáp án ngay sau mỗi câu.

Log mỗi câu theo yêu cầu BTC: id, answer, elapsed_ms, thinking (list nguyên văn theo thứ tự gọi), code (SQL đã chạy),
inputs (kỳ, thực thể, bảng), output, kèm decision. Log dựng shop (thẻ bảng, ánh xạ giá trị) ghi riêng theo shop.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from socai.llm.client import LLM, extract_json
from socai.rules.calendar import LunarCalendar
from socai.v4 import l4 as L4
from socai.v4.l1 import run_l1
from socai.v4.l2 import ShopModel, build_model, catalog
from socai.v4.l3 import Rules, load_rules
from socai.v4.l5 import SQLError, format_answer, run_sql
from socai.v4.l6 import decide, gate_time


@dataclass
class Config:
    n_samples: int = 1
    temperature: float = 0.7
    thinking: bool = False
    max_tokens: int = 1500
    repairs: int = 1
    min_agree: float = 0.5
    enrich_rules: bool = False
    time_budget_s: float = 5.5 * 3600


def _write(f, rec: dict) -> None:
    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    f.flush()
    os.fsync(f.fileno())


def shop_vocab(model: ShopModel, rules: Rules) -> L4.Vocab:
    shop = [model.readme_intro]
    for v in model.views.values():
        shop += [c.desc for c in v.cols] + [c.orig for c in v.cols] + [v.desc]
    return L4.Vocab.build([c.text + " " + c.path for c in rules.cards], shop)


STRONG_HINTS = {"date", "percent", "boolean"}


def one_sample(llm: LLM, msgs: list[dict], model: ShopModel, cfg: Config, seed: int, temp: float,
               thinking_log: list, hint: str = "") -> dict:
    """Một mẫu: LLM viết spec → chạy SQL → định dạng; lỗi thì đưa lỗi cho LLM sửa (cfg.repairs lần)."""
    err = None
    cur = list(msgs)
    for attempt in range(cfg.repairs + 1):
        c = llm.chat(cur, max_tokens=cfg.max_tokens, json_schema=L4.SPEC_SCHEMA, thinking=cfg.thinking,
                     temperature=temp, seed=seed)
        thinking_log.append(c.reasoning or c.content[:2000])
        spec = extract_json(c.content) or {}
        kind, unit, sql = spec.get("loai"), spec.get("don_vi", "so"), spec.get("sql", "")
        if kind == "unanswerable":
            return {"answer": {"type": "unanswerable", "reason": (spec.get("ly_do") or "không đủ dữ liệu")[:300]},
                    "spec": spec}
        try:
            if hint in STRONG_HINTS and kind != hint and attempt < cfg.repairs:
                raise SQLError(f"câu hỏi có dạng {hint} (vd 'tháng nào/ngày nào' → date, '%' → percent, "
                               f"'có … không' → boolean) nhưng loai = {kind}")
            cols, rows = run_sql(model.con, sql)
            if kind in ("table", "list") and len(rows) == 1 and hint in ("number", "percent") and \
                    isinstance(rows[0][-1], (int, float)):
                kind, cols, rows = hint, cols[-1:], [rows[0][-1:]]  # câu một con số mà LLM ghi table/list
            ans = format_answer(kind, unit, cols, rows)
            return {"answer": ans, "spec": spec, "rows": [list(map(str, r)) for r in rows[:5]], "cols": cols}
        except SQLError as e:
            err = str(e)
        cur = cur + [{"role": "assistant", "content": c.content[:3000]},
                     {"role": "user", "content": f"Lỗi khi chạy: {err}\nSửa SQL (hoặc loai/don_vi) và trả lại JSON."}]
    return {"error": err, "spec": spec}


def answer_question(q: dict, model: ShopModel, rules: Rules, vocab: L4.Vocab, cal, llm: LLM | None,
                    cfg: Config) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    today = date.fromisoformat(q["as_of_date"])
    n = L4.normalize(q["question"], vocab)
    spec = L4.resolve_time(q["question"], n, today, cal)
    ents = L4.link_entities(n, model)
    tables = L4.pick_tables(n, model, ents)
    hint = L4.answer_hint(n.text)
    log: dict = {"id": q["id"], "business_id": q["business_id"],
                 "inputs": {"normalized": n.text, "changes": n.changes,
                            "periods": [[str(p.start), str(p.end), p.kind] for p in spec.periods],
                            "compare_to": [str(spec.compare_to.start), str(spec.compare_to.end)] if spec.compare_to else None,
                            "entities": ents[:8], "tables": tables, "hint": hint}}
    thinking: list = []
    why = gate_time(spec, model.date_range)
    if why:
        final = {"type": "unanswerable", "reason": why}
        log.update({"answer": final, "decision": {"source": "gate_time"}, "code": "socai/v4/l6.py::gate_time",
                    "output": why})
    elif llm is None:
        final = {"type": "abstain"}
        log.update({"answer": final, "decision": {"source": "no_llm"}, "code": None, "output": None})
    else:
        msgs = L4.build_prompt(q["question"], today, n, spec, ents, tables, model, rules, hint)
        results = []
        for i in range(cfg.n_samples):
            try:
                results.append(one_sample(llm, msgs, model, cfg, seed=i, temp=0.0 if i == 0 else cfg.temperature,
                                          thinking_log=thinking, hint=hint))
            except Exception as e:  # noqa: BLE001 — một mẫu lỗi không làm hỏng câu
                results.append({"error": f"{type(e).__name__}: {e}"[:300]})
        d = decide(results, cfg.min_agree)
        final = d.answer
        log.update({"answer": final, "decision": {"source": d.source, "agree": d.agree, "n": d.n},
                    "code": [r.get("spec", {}).get("sql") for r in results],
                    "output": [r.get("answer") or r.get("error") for r in results],
                    "specs": [{k: v for k, v in (r.get("spec") or {}).items() if k != "sql"} for r in results]})
    log["thinking"] = thinking if len(thinking) > 1 else (thinking[0] if thinking else None)
    log["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    sub = {"id": q["id"], **{k: v for k, v in final.items() if k in ("type", "answer", "reason")}}
    if sub["type"] == "unanswerable":
        sub.pop("answer", None)
    return sub, log


def run(data: Path, out: Path, llm: LLM | None, cfg: Config, shops: list[str] | None = None,
        limit: int | None = None, cache: Path | None = None) -> dict:
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    qs = [json.loads(x) for x in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if shops:
        qs = [q for q in qs if q["business_id"] in shops]
    if limit:
        by: dict[str, list] = {}
        for q in qs:
            by.setdefault(q["business_id"], []).append(q)
        qs = [q for v in by.values() for q in v[:limit]]
    cal = LunarCalendar.load(data / "calendar.csv") if (data / "calendar.csv").exists() else None
    rules = load_rules(data / "rulebook.md", cache, llm if cfg.enrich_rules else None)
    groups: dict[str, list] = {}
    for q in qs:
        groups.setdefault(q["business_id"], []).append(q)
    stats = {"n": len(qs), "answered": 0, "abstain": 0, "unanswerable": 0}
    done = set()
    with (out / "submission.jsonl").open("w", encoding="utf-8") as fs, \
            (out / "logs" / "log.jsonl").open("w", encoding="utf-8") as fl, \
            (out / "logs" / "shops.jsonl").open("w", encoding="utf-8") as fsh:
        for biz, items in groups.items():
            if time.time() - t_start > cfg.time_budget_s:
                break
            trace: list = []
            t0 = time.time()
            try:
                print(f"[{time.time() - t_start:7.0f}s] {biz}: dựng L1+L2 ({len(items)} câu)", flush=True)
                sess, tables = run_l1(data / "data" / biz, llm, trace)
                model = build_model(biz, tables, sess.readme.intro + " " + " ".join(sess.readme.conventions),
                                    rules.industries, llm, trace)
            except Exception as e:  # noqa: BLE001
                _write(fsh, {"shop": biz, "error": f"{type(e).__name__}: {e}"[:500]})
                for q in items:
                    _write(fs, {"id": q["id"], "type": "abstain"})
                    done.add(q["id"])
                continue
            print(f"[{time.time() - t_start:7.0f}s] {biz}: xong L1+L2 sau {time.time() - t0:.0f}s, "
                  f"{len(model.views)} bảng, ngành={model.industry or '-'}", flush=True)
            _write(fsh, {"shop": biz, "seconds": round(time.time() - t0, 1), "industry": model.industry,
                         "date_range": model.date_range, "views": {k: {"role": v.role, "rows": v.rows,
                                                                       "per_order": v.per_order, "notes": v.notes}
                                                                   for k, v in model.views.items()},
                         "catalog": catalog(model), "trace": trace})
            vocab = shop_vocab(model, rules)
            for q in items:
                if time.time() - t_start > cfg.time_budget_s:
                    break
                sub, log = answer_question(q, model, rules, vocab, cal, llm, cfg)
                _write(fs, sub)
                _write(fl, log)
                done.add(q["id"])
                k = sub["type"] if sub["type"] in ("abstain", "unanswerable") else "answered"
                stats[k] += 1
                print(f"[{time.time() - t_start:7.0f}s] {biz} {len(done)}/{len(qs)} {sub['type']:12s} "
                      f"{log['elapsed_ms'] / 1000:5.1f}s {q['question'][:60]}", flush=True)
        for q in qs:
            if q["id"] not in done:
                _write(fs, {"id": q["id"], "type": "abstain"})
                stats["abstain"] += 1
    stats["seconds"] = round(time.time() - t_start, 1)
    (out / "logs" / "run_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats
