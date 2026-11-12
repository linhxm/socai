"""Agent một shop: nhận câu hỏi → các đường trả lời → chính sách quyết định → (đáp án, bản ghi log).

Đường 1 (fastpath): luật + metric tất định trên bảng chuẩn. Nhanh, chính xác, phủ hẹp.
Đường 2 (llm_code): LLM viết `answer(ctx)` trên bảng thô/chuẩn, chạy trong sandbox, có 1 lần sửa lỗi.
Đường 2 chạy khi đường 1 không có ứng viên, hoặc khi bật `verify_fastpath` (để đo đồng thuận).
Log theo hướng dẫn Docker của BTC: id, answer, elapsed_ms, thinking (nguyên văn, list theo thứ tự gọi),
code (code sinh ra đã chạy, hoặc 'file::hàm' cho hàm đóng gói), inputs, output.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from socai.answer.decide import Candidate, Policy, normalize
from socai.exec import metrics as M
from socai.exec.metrics import Unsupported
from socai.exec.query import execute
from socai.exec.sandbox import CodeWorker
from socai.nlq import plan as plan_mod
from socai.semantic.entities import EntityIndex
from socai.io.cells import fold
from socai.llm.client import LLM, extract_code
from socai.nlq import prompts
from socai.nlq.fastpath import parse as fast_parse
from socai.profile.business import Business, load_business
from socai.profile.llm_tagger import tag_with_llm
from socai.rules.calendar import LunarCalendar
from socai.rules.rulebook import Rulebook
from socai.rules.timeexpr import Resolver, TimeSpec
from socai.semantic.canonical import Canonical, build
from socai.v2 import answerable
from socai.v2 import brain as v2_brain
from socai.v2.algebra import Ctx
from socai.v2.bank import Bank
from socai.v2.views import View, build_views


@dataclass
class AgentConfig:
    use_llm: bool = True
    use_plan: bool = True  # LLM tách câu → kế hoạch → lệnh cơ bản tất định (v1)
    use_v2: bool = False  # LLM ghép biểu thức đại số (v2); bật thì thay cho use_plan
    bank_lbo: bool = False  # khi đo trên train: không lấy ví dụ của chính shop đang hỏi
    second_opinion: bool = False  # hỏi lại với bộ ví dụ khác; hai lần khác đáp án thì hạ độ tin
    use_codegen: bool = False  # LLM tự viết code: tắt với 4B (đo được 0/23 trên laptop)
    llm_tagging: bool = True  # LLM gán nghĩa cột còn thiếu (kiểm lại bằng dữ liệu)
    verify_fastpath: bool = False
    thinking: bool = False
    max_tokens: int = 2048
    max_repairs: int = 1
    code_timeout_s: float = 60.0
    use_gates: bool = True  # cổng §7: kỳ ngoài dữ liệu / shop không ghi loại thông tin → unanswerable
    n_samples: int = 1  # > 1: lấy nhiều mẫu biểu thức, chạy hết, bỏ phiếu theo kết quả (thay cho second_opinion)
    v2_thinking: bool = False  # Qwen3 chế độ suy nghĩ khi ghép biểu thức (14B trên máy chấm)
    sample_temperature: float = 0.7


@dataclass
class Session:
    biz: Business
    can: Canonical
    names: set[str]
    index: EntityIndex | None = None
    worker: CodeWorker | None = None
    views: dict[str, View] = field(default_factory=dict)


@dataclass
class Agent:
    rulebook: Rulebook
    calendar: LunarCalendar | None
    policy: Policy
    llm: LLM | None = None
    cfg: AgentConfig = field(default_factory=AgentConfig)
    cache_dir: Path | None = None
    calendar_path: Path | None = None
    bank: Bank | None = None  # ví dụ đã kiểm chứng từ train (v2/bank.json)

    def open(self, biz_dir: Path) -> Session:
        biz = load_business(biz_dir, self.cache_dir)
        if self.llm is not None and self.cfg.use_llm and self.cfg.llm_tagging:
            try:  # bộ não con "Bảng": gán nghĩa cột còn thiếu, kiểm bằng dữ liệu
                tag_with_llm(biz, self.llm)
            except Exception as e:  # noqa: BLE001 — gán nghĩa lỗi thì vẫn chạy với tag sẵn có
                biz.mapping_log.append({"via": "llm", "error": str(e)[:200]})
        can = build(biz)
        names: set[str] = set()
        if can.lines is not None:
            for c in ("product_name", "customer_name"):
                names |= {f for x in can.lines[c].unique() if len((f := fold(x)).split()) >= 2}
        return Session(biz, can, names, EntityIndex(can), views=build_views(biz, can) if self.cfg.use_v2 else {})

    def close(self, s: Session) -> None:
        if s.worker is not None:
            s.worker.close()

    # ------------------------------------------------------------------ cổng §7 (trước mọi đường)
    def _gate(self, s: Session, q: str, spec: TimeSpec, log: dict) -> Candidate | None:
        """Không trả lời được, kết luận bằng dữ liệu (không cần LLM): kỳ hỏi hoặc kỳ so sánh nằm hoàn toàn ngoài
        khoảng dữ liệu, hoặc câu cần loại thông tin mà shop không ghi. Đo trên dev: không chặn nhầm câu nào."""
        why, src = None, None
        if spec.main is not None and M.coverage(s.can, spec.main) == "outside":
            why, src = "kỳ hỏi nằm ngoài khoảng dữ liệu (§7)", "gate_time"
        elif spec.compare_to is not None and M.coverage(s.can, spec.compare_to) == "outside":
            why, src = "kỳ so sánh nằm ngoài khoảng dữ liệu (§7)", "gate_time"
        elif (r := answerable.missing(q, s.views)) is not None:
            why, src = r, "gate_facet"
        if why is None:
            return None
        log["code"], log["inputs"], log["output"] = "socai/agent.py::_gate", {"source": src}, why
        return Candidate(normalize({"type": "unanswerable", "answer": None, "reason": why}), src, "unanswerable")

    # ------------------------------------------------------------------ đường 1
    def _fastpath(self, s: Session, q: str, spec: TimeSpec, log: dict) -> Candidate | None:
        ir = fast_parse(q, s.names)
        if ir is None or spec.main is None or ir.metric not in M.METRICS:
            return None
        fn = M.METRICS[ir.metric]
        try:
            if M.coverage(s.can, spec.main) == "outside":
                r = M.Result.unanswerable("kỳ hỏi nằm ngoài khoảng dữ liệu (§7)")
            elif ir.compare:
                if spec.compare_to is None:
                    return None
                if M.coverage(s.can, spec.compare_to) == "outside":
                    r = M.Result.unanswerable("kỳ so sánh nằm ngoài khoảng dữ liệu (§7)")
                else:
                    cur, prev = fn(s.can, spec.main), fn(s.can, spec.compare_to)
                    r = M.pct_change(cur, prev) if ir.compare == "pct_change" else M.diff(cur, prev)
            else:
                r = fn(s.can, spec.main)
        except M.Unsupported as e:
            log.setdefault("notes", []).append(f"fastpath: {e}")
            return None
        ans = normalize({"type": r.type, "answer": r.value, "reason": r.reason})
        if ans is None:
            return None
        log["code"] = f"socai/exec/metrics.py::{ir.metric}" + (f"+{ir.compare}" if ir.compare else "")
        log["inputs"] = {"data_dir": str(s.biz.path), "periods": [[str(p.start), str(p.end)] for p in spec.periods],
                         "compare_to": [str(spec.compare_to.start), str(spec.compare_to.end)]
                         if spec.compare_to else None, "ir": ir.to_dict()}
        log["output"] = r.value
        return Candidate(ans, "fastpath", ir.metric + (f"+{ir.compare}" if ir.compare else ""), trace=r.trace)

    # ------------------------------------------------------------------ đường 2: tách câu → lệnh cơ bản
    def _llm_plan(self, s: Session, q: str, today: date, spec: TimeSpec, log: dict) -> Candidate | None:
        if self.llm is None or s.index is None or not s.can.ok:
            return None
        raw, reasoning = plan_mod.ask(self.llm, q)
        log.setdefault("thinking_list", []).append(reasoning)
        plan = plan_mod.to_spec(raw or {}, q, spec, s.index, today)
        log["plan"] = {"raw": raw, "resolved": plan.resolved, "problems": plan.problems}
        if plan.spec is None:
            return None
        try:
            r = execute(s.can, plan.spec)
        except Unsupported as e:
            log.setdefault("notes", []).append(f"plan: {e}")
            return None
        ans = normalize({"type": r.type, "answer": r.value, "reason": r.reason})
        if ans is None:
            return None
        if "code" not in log or log["code"] is None:
            log["code"], log["inputs"], log["output"] = "socai/exec/query.py::execute", plan.spec.describe(), r.value
        key = f"{plan.spec.measure}:{plan.spec.shape}"
        return Candidate(ans, "llm_plan", key, trace=r.trace)

    # ------------------------------------------------------------------ đường 2 (v2): ghép biểu thức đại số
    def _v2(self, s: Session, q: str, today: date, spec: TimeSpec, log: dict) -> Candidate | None:
        if self.llm is None or not s.views:
            return None
        resolver = Resolver(today, self.calendar)
        ctx = Ctx(s.views, spec, resolver.resolve, plan_mod._series_periods, today=today)
        ex = s.biz.biz_id if self.cfg.bank_lbo else None
        if self.cfg.n_samples > 1:
            return self._v2_vote(s, q, spec, today, resolver, ex, log)
        th = v2_brain.think(self.llm, s.views, q, ctx, max_repairs=self.cfg.max_repairs, bank=self.bank,
                            exclude_biz=ex)
        log.setdefault("thinking_list", []).extend(th.reasoning)
        log["v2"] = {"expr": th.expr, "attempts": th.attempts, "problem": th.problem, "sig": th.sig,
                     "shots": th.shots}
        if th.result is None:
            return None
        r = th.result
        ans = normalize({"type": r.type, "answer": r.value, "reason": r.reason})
        if ans is None:
            return None
        if self.cfg.second_opinion and self.bank is not None:
            ctx2 = Ctx(s.views, spec, resolver.resolve, plan_mod._series_periods, today=today)
            th2 = v2_brain.think(self.llm, s.views, q, ctx2, max_repairs=0, bank=self.bank, exclude_biz=ex, skip=4)
            log.setdefault("thinking_list", []).extend(th2.reasoning)
            a2 = normalize({"type": th2.result.type, "answer": th2.result.value, "reason": th2.result.reason}) \
                if th2.result is not None else None
            log["v2"]["second"] = {"expr": th2.expr, "problem": th2.problem, "agree": a2 == ans}
            agree = "agree" if a2 == ans else ("alone" if a2 is None else "differ")
            if log.get("code") is None:
                log["code"], log["inputs"], log["output"] = "socai/v2/algebra.py::execute", th.expr, r.value
            return Candidate(ans, f"v2_{agree}", th.sig, trace=r.trace)
        if log.get("code") is None:
            log["code"], log["inputs"], log["output"] = "socai/v2/algebra.py::execute", th.expr, r.value
        return Candidate(ans, "v2", th.sig, trace=r.trace)

    def _v2_vote(self, s: Session, q: str, spec: TimeSpec, today: date, resolver: Resolver, ex: str | None,
                 log: dict) -> Candidate | None:
        """k mẫu: greedy, greedy với bộ ví dụ khác, rồi các mẫu có nhiệt độ (seed cố định → tái lập). Chạy hết,
        bỏ phiếu theo KẾT QUẢ chạy (research: C3, MBR-exec). Mức đồng thuận → nhãn đã hiệu chỉnh của v2:
        tất cả cùng đáp án → v2_agree; quá nửa → v2_differ; còn lại → v2_alone."""
        k = self.cfg.n_samples
        plan = [(0, 0.0, None), (4, 0.0, None)] + [(0, self.cfg.sample_temperature, i) for i in range(1, k - 1)]
        votes: dict[str, list] = {}
        samples = []
        for i, (skip, temp, seed) in enumerate(plan[:k]):
            ctx = Ctx(s.views, spec, resolver.resolve, plan_mod._series_periods, today=today)
            th = v2_brain.think(self.llm, s.views, q, ctx, max_repairs=self.cfg.max_repairs if i == 0 else 0,
                                bank=self.bank, exclude_biz=ex, skip=skip, thinking=self.cfg.v2_thinking,
                                temperature=temp, seed=seed)
            log.setdefault("thinking_list", []).extend(th.reasoning)
            r = th.result
            ans = normalize({"type": r.type, "answer": r.value, "reason": r.reason}) if r is not None else None
            samples.append({"expr": th.expr, "problem": th.problem, "answer": ans})
            if ans is not None:
                votes.setdefault(json.dumps(ans, sort_keys=True, ensure_ascii=False), []).append((i, th, ans))
        log["v2"] = {"samples": samples, "votes": {key: len(v) for key, v in votes.items()}}
        if not votes:
            return None
        # nhiều phiếu nhất; hoà thì theo mẫu đến trước (mẫu 0 là greedy)
        key, group = max(votes.items(), key=lambda kv: (len(kv[1]), -kv[1][0][0]))
        m = len(group)
        label = "agree" if m == k else ("differ" if m > k / 2 else "alone")
        _, th, ans = group[0]
        log["v2"].update({"expr": th.expr, "sig": th.sig, "agree": f"{m}/{k}"})
        log["code"], log["inputs"], log["output"] = "socai/v2/algebra.py::execute", th.expr, th.result.value
        return Candidate(ans, f"v2_{label}", th.sig, trace=th.result.trace)

    # ------------------------------------------------------------------ đường 3: LLM viết code (tắt mặc định)
    def _llm_code(self, s: Session, q: str, today: date, spec: TimeSpec, log: dict) -> Candidate | None:
        if self.llm is None:
            return None
        if s.worker is None:
            s.worker = CodeWorker(s.biz.path, self.cache_dir, self.calendar_path)
        msgs = prompts.messages(self.rulebook, s.biz, s.can.ok, q, today.isoformat(), spec)
        thinking, codes, outputs = log.setdefault("thinking_list", []), [], []
        for _attempt in range(1 + self.cfg.max_repairs):
            call = self.llm.chat(msgs, max_tokens=self.cfg.max_tokens, thinking=self.cfg.thinking)
            thinking.append(call.reasoning)
            code = extract_code(call.content)
            if code is None:
                msgs = msgs + [{"role": "assistant", "content": call.content},
                               {"role": "user", "content": "Thiếu khối ```python``` có hàm answer(ctx). Viết lại."}]
                continue
            res = s.worker.run(code, today, self.cfg.code_timeout_s)
            codes.append(code)
            outputs.append(res.value if res.ok else {"error": res.error})
            if res.ok and (ans := normalize(res.value)) is not None:
                log["code"], log["output"] = (codes if len(codes) > 1 else code), res.value
                return Candidate(ans, "llm_code", ans["type"])
            msgs = msgs + [{"role": "assistant", "content": call.content},
                           {"role": "user", "content": f"Code lỗi hoặc sai format:\n{res.error or res.value}\nSửa lại."}]
        log["code"], log["output"] = codes or None, outputs or None
        return None

    # ------------------------------------------------------------------ một câu
    def answer(self, s: Session, qid: str, question: str, as_of: str) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        today = date.fromisoformat(as_of)
        log: dict = {"id": qid, "business_id": s.biz.biz_id}
        spec = Resolver(today, self.calendar).resolve(question)
        log["time_spec"] = {"periods": [[str(p.start), str(p.end), p.kind] for p in spec.periods],
                            "compare_to": [str(spec.compare_to.start), str(spec.compare_to.end)]
                            if spec.compare_to else None, "ambiguous": spec.ambiguous}
        cands = []
        try:
            if self.cfg.use_gates and (g := self._gate(s, question, spec, log)) is not None:
                cands.append(g)
            elif (c := self._fastpath(s, question, spec, log)) is not None:
                cands.append(c)
            if self.cfg.use_llm and self.cfg.use_v2 and (not cands or self.cfg.verify_fastpath):
                if (c := self._v2(s, question, today, spec, log)) is not None:
                    cands.append(c)
            elif self.cfg.use_llm and self.cfg.use_plan and (not cands or self.cfg.verify_fastpath):
                if (c := self._llm_plan(s, question, today, spec, log)) is not None:
                    cands.append(c)
            if self.cfg.use_llm and self.cfg.use_codegen and not cands:
                if (c := self._llm_code(s, question, today, spec, log)) is not None:
                    cands.append(c)
        except Exception as e:  # noqa: BLE001 — một câu lỗi không được làm hỏng cả lượt chạy
            log.setdefault("notes", []).append(f"lỗi: {type(e).__name__}: {e}"[:500])
        final, why = self.policy.decide(cands)
        tl = log.pop("thinking_list", None)
        log.update({"answer": final, "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "thinking": (tl if tl and len(tl) > 1 else (tl[0] if tl else None)), "decision": why})
        log.setdefault("code", None)
        log.setdefault("output", None)
        return {"id": qid, **final}, log
