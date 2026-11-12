"""Fine-tune LoRA cho Qwen3 làm "bộ não ghép biểu thức" của v2, trên dữ liệu `export_finetune.py` (chỉ từ TRAIN).

Chạy trên một GPU ≥ 15 GB. Kaggle T4 không có bf16 nên dùng base fp16 + LoRA fp32 + GradScaler; không cần bitsandbytes.
    python training_code/finetune_lora.py --data training_code/data/finetune --out runs/lora --revision <REV>
    python training_code/finetune_lora.py --merge runs/lora --out models/llm --revision <REV>   # gộp adapter

- Prompt dựng bằng chat template với `enable_thinking=False`, đúng như lúc suy luận (`VllmClient`).
- Loss chỉ trên token của biểu thức + `<|im_end|>`; chỉ tính logits cho các vị trí đó (`logits_to_keep`), không tính
  logits 151K từ vựng cho cả prompt ~3K token.
- Mẫu dài hơn `--max-len` bị bỏ (không cắt: cắt sẽ mất phần luật đầu prompt). `--max-minutes` dừng sớm và vẫn lưu.
- Ghi `run_config.json`, `log_history.json`, `adapter/` để tái lập (mục 8 hướng dẫn Docker).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

BASE = "Qwen/Qwen3-4B"


def fp16(torch) -> dict:
    """transformers >= 4.56 đổi `torch_dtype` → `dtype`; bản cũ bỏ qua tên lạ và nạp fp32 (tràn T4)."""
    import transformers
    ver = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    return {"dtype": torch.float16} if ver >= (4, 56) else {"torch_dtype": torch.float16}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def encode(tok, row: dict) -> tuple[list[int], int]:
    msgs = row["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    ans = tok(msgs[-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + ans, len(ans)


def train(a: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    data = {}
    for split in ("train", "val"):
        f = a.data / f"{split}.jsonl"
        rows = load_rows(f) if f.exists() else []
        enc = [encode(tok, r) for r in rows]
        data[split] = [(ids, n) for ids, n in enc if len(ids) <= a.max_len]
        lens = sorted(len(ids) for ids, _ in enc) or [0]
        print(f"{split}: {len(data[split])}/{len(rows)} mẫu (bỏ {len(rows) - len(data[split])} mẫu > {a.max_len} token); "
              f"token p50={lens[len(lens) // 2]} p95={lens[int(len(lens) * .95)]} max={lens[-1]}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.base, revision=a.revision, attn_implementation="sdpa",
                                                 **fp16(torch)).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # LoRA fp32, base fp16: GradScaler cần tham số học được ở fp32
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(data["train"]) / a.accum)
    total = max(1, int(steps_per_epoch * a.epochs))
    warm = max(1, int(total * 0.05))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")

    def loss_of(ids: list[int], n: int):
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            try:
                logits = model(input_ids=x, logits_to_keep=n + 1).logits[:, :-1]
            except TypeError:  # transformers cũ
                logits = model(input_ids=x).logits[:, -n - 1:-1]
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, -n:].reshape(-1))

    def evaluate() -> float | None:
        if not data["val"]:
            return None
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for ids, n in data["val"]:
                tot += loss_of(ids, n).item() * n
                cnt += n
        model.train()
        return tot / max(1, cnt)

    hist = [{"step": 0, "val_loss": evaluate(), "min": round((time.time() - t0) / 60, 1)}]
    print(hist[-1], flush=True)
    model.train()
    step, run, skipped, stop = 0, [], 0, False
    order = list(range(len(data["train"])))
    epoch = 0
    while step < total and not stop:
        random.Random(f"{a.seed}:{epoch}").shuffle(order)
        for j, i in enumerate(order):
            ids, n = data["train"][i]
            loss = loss_of(ids, n)
            scaler.scale(loss / a.accum).backward()
            run.append(loss.item())
            if (j + 1) % a.accum and j + 1 != len(order):
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += scaler.get_scale() < scale  # fp16 tràn → bước bị bỏ
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % a.log_every == 0 or step == total:
                hist.append({"step": step, "epoch": round(step / steps_per_epoch, 2),
                             "loss": round(sum(run) / len(run), 4), "lr": sched.get_last_lr()[0],
                             "skipped": int(skipped), "min": round((time.time() - t0) / 60, 1)})
                run = []
                print(hist[-1], flush=True)
            if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
                print(f"HẾT GIỜ ({a.max_minutes} phút): dừng ở bước {step}/{total}", flush=True)
                stop = True
            if step >= total or stop:
                break
        epoch += 1
        hist.append({"step": step, "epoch": epoch, "val_loss": evaluate(), "min": round((time.time() - t0) / 60, 1)})
        print(hist[-1], flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(a.out / "adapter"))
    tok.save_pretrained(str(a.out / "adapter"))
    (a.out / "run_config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()},
                                                       "steps": step, "total_steps": total,
                                                       "n_train": len(data["train"]), "n_val": len(data["val"])},
                                                      indent=1), encoding="utf-8")
    (a.out / "log_history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    print(f"xong sau {(time.time() - t0) / 60:.1f} phút, {step} bước, {skipped} bước fp16 tràn", flush=True)


def merge(a: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(a.base, revision=a.revision, **fp16(torch))
    merged = PeftModel.from_pretrained(base, str(Path(a.merge) / "adapter")).merge_and_unload()
    merged.save_pretrained(str(a.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base, revision=a.revision).save_pretrained(str(a.out))
    print("đã gộp vào", a.out, "| tham số:", sum(p.numel() for p in merged.parameters()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merge", default=None, help="thư mục lượt train có adapter/ → gộp vào trọng số gốc")
    ap.add_argument("--base", default=BASE, help="tên HF hoặc thư mục đã tải")
    ap.add_argument("--revision", default=None, help="GHI revision cố định khi chạy thật")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args()
    merge(ns) if ns.merge else train(ns)
