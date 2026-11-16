# SOCAI: a question-answering agent for Vietnamese small-shop ledgers

SOCAI answers everyday Vietnamese questions (with or without diacritics, teencode, abbreviations, Telex typos) over the messy CSV / XLSX / SQLite files of small shops: revenue, best sellers, debts, returns, period comparisons, and so on. It runs fully offline with one open model of at most 15B parameters.

Answers are scored with **Penalized Accuracy (PA)**: correct +1, wrong −1, abstain 0. So every version is built around one rule: **the LLM never computes a number itself**. Code reads the data and executes the plan, and the system **answers only when it is confident enough**.

## Versions at a glance

| | Period | Idea | Status |
|---|---|---|---|
| **v1** | 10–11 Nov | Regex intents + hand-written metrics; Qwen3-4B fills a JSON plan from a metric catalog | Superseded; still used as v2's rule path |
| **v2** | 11–13 Nov | Qwen3-4B/14B composes a **query-algebra expression** over canonical tables; verified examples retrieved from train; calibrated abstention | **Submitted** (with Qwen3-4B) |
| **v3** | 12 Nov | Research and error analysis of v2; plan for a layered rewrite | Docs only |
| **v4** | 13 Nov | Six-layer rewrite (L1 read → L6 decide); the LLM makes narrow JSON decisions and writes SQL on DuckDB | Measured negative on public; stopped |
| **v5** | 13–14 Nov | Lean rewrite: the LLM writes one SQL query per sample on DuckDB, k samples vote, calibrated threshold; can cross-check v2 | Measured; best as a verifier for v2 |

### v1: rule path + plan filling ([docs/v1.md](docs/v1.md))
Deterministic readers turn each shop into canonical tables (orders, lines, customers, debts). A fast path matches about 8 question intents with regular expressions and hand-written metrics. For other questions, Qwen3-4B fills a JSON plan by choosing from a fixed catalog of metric types, and code executes it. Rulebook logic lives inside each hand-written metric, so coverage stops at what was written by hand.

### v2: query algebra ([docs/v2.md](docs/v2.md))
- The LLM composes an algebraic expression: `table → filters → period → group → aggregate`, combined with `pct`, `diff`, `ratio`, `except`, … Code type-checks and executes it.
- Rulebook rules move into **table definitions** (default filters, item/customer keys, debt as of a date), so every expression gets them for free.
- For each question, **8 verified examples** are retrieved from an example bank built on the training split (only expressions that reproduce the training answers are kept), plus a second opinion.
- Several samples vote, and a threshold calibrated on training shops decides between answering and abstaining. Periods outside the data are answered `unanswerable` by code.

### v3: research and plan ([docs/v3.md](docs/v3.md))
Error analysis of v2 per answer type and per shop, a review of table-QA and text-to-SQL literature, and the plan that led to v4: split the work into layers and let code verify every LLM decision.

### v4: six layers ([docs/v4.md](docs/v4.md), functions in [docs/v4-ham.md](docs/v4-ham.md))
- **L1 read:** header detection, row kinds, total rows verified by summing (block- and level-aware).
- **L2 shop model:** an LLM table card validated against the data; derived columns (order id, date, normalized status/customer/item, base quantity); loaded into DuckDB.
- **L3 rules:** the rulebook cut into cards, BM25 retrieval, industry filter.
- **L4 understand:** abbreviation and teencode normalization without hard-coding, entity linking, time families, SQL prompt with guided JSON.
- **L5 compute:** read-only SQL with a timeout; answers formatted by code.
- **L6 decide:** time gate, k-sample vote.

### v5: SQL + voting ([docs/v5.md](docs/v5.md))
A rewrite that keeps only the shared base modules. L1/L2 load typed tables into DuckDB with column profiles and an accent-free value index. L3 retrieves rule cards from the split's own rulebook. L4 resolves time periods in code and links entities. The LLM returns `{reasoning, answer type, unit, sql}` under a closed JSON schema. L5 runs the SELECT and formats the answer, with one repair round on SQL errors. L6 votes over k samples and maps the agreement level to a calibrated probability of being correct.

`predict.sh` modes (fixed with `ENV` in the Dockerfile):
- `v2` (default);
- `v5`;
- `v2+v5`: v5 answers only where v2 abstains;
- `SOCAI_VERIFY=1`: v5 re-checks v2's answers and abstains when all v5 samples agree on a different answer;
- `SOCAI_V2_TABLES=1`: v5 writes SQL on v2's cleaned tables.

## Results

**Hold-out training shops** (6 shops, 631 questions, [configs/holdout.json](configs/holdout.json); Qwen3-4B unless noted):

| Configuration | Correct | Wrong | Abstain | PA |
|---|---|---|---|---|
| v1 rule path only | 142 | 2 | 487 | 0.222 |
| v1 rules + Qwen3-4B plan | 239 | 66 | 326 | 0.274 |
| v1 + new canonical tables + calibrated threshold | 289 | 69 | 273 | 0.349 |
| v2 query algebra | 380 | 82 | 169 | 0.472 |
| **v2 + verified examples + second opinion** | 513 | 62 | 56 | **0.715** |

On 120 hold-out questions, Qwen3-14B reaches 0.808 (4B: 0.775) with the example bank and 0.642 (4B: 0.525) without it.

**Public split** (Qwen3-14B Q4_K_M on Kaggle 2×T4; measurement only, never used to tune code):

| Configuration | Questions | Correct | Wrong | Abstain | PA |
|---|---|---|---|---|---|
| v4 | 280 | 118 | 136 | 26 | −0.064 |
| v2 | 126 | 70 | 40 | 16 | 0.238 |
| v5 alone | 126 | 56 | 49 | 21 | 0.056 |
| v2, v5 fills v2's abstentions | 126 | 72 | 49 | 5 | 0.183 |
| **v2, v5 cross-check** | 126 | 65 | 28 | 33 | **0.294** |

What these numbers say:
- **v2 generalizes poorly to new shops and industries:** 0.808 on hold-out versus 0.238 on public, with 40 wrong answers out of 110. Its abstention threshold is overconfident outside the training distribution.
- **v5 is weaker than v2 on its own, but useful as a verifier.** Of the 28 v2 answers it vetoed, 21 were wrong and 7 were right.
- 126 questions give a PA standard error of about ±0.08, so the cross-check gain needs confirmation on more data.
- The hold-out is not perfectly clean: some v1/v2 reader fixes were made after seeing hold-out failures.

## Running

```bash
pip install -e ".[dev]"
pytest -q                       # real-data tests are skipped when ../train is absent

# v1/v2 rule path only (no GPU)
python -m socai eval --split ../train --out runs/train --no-llm

# with an LLM: any OpenAI-compatible server (llama.cpp or vLLM) serving Qwen3
# e.g. llama.cpp on a 4 GB GPU:
#   llama-server -m Qwen3-4B-Q4_K_M.gguf -ngl 99 -fa on -c 4096 --jinja --port 8088 -np 1 --seed 0
SOCAI_LLM_URL=http://127.0.0.1:8088/v1 python -m socai predict --data ../public_test --out runs/v2   # v2

python -m socai.v4 predict --data ../public_test --out runs/v4 --llm http://127.0.0.1:8088/v1        # v4
python -m socai.v5 predict --data ../public_test --out runs/v5 --llm http://127.0.0.1:8088/v1        # v5
python -m socai.v5 shop --data ../train --shop <shop_id>        # inspect how v5 reads one shop (no LLM)

# utilities
python -m socai profile --biz ../train/data/<shop_id>
python -m socai rulebook-diff ../train/rulebook.md ../public_test/rulebook.md
python -m socai synth --out /tmp/synth --n-biz 3 --render-images
```

Kaggle runners for Qwen3-14B (v2 hold-out, v4, v5 with example-bank building) are in [kaggle/README.md](kaggle/README.md).

## Docker image
`docker/Dockerfile` builds an offline image on `vllm/vllm-openai:v0.10.1.1`. `predict.sh` starts vLLM with the weights in `models/llm` (tensor-parallel over all GPUs, fixed seed, one request at a time), runs the selected pipeline on `/data`, and writes `/result/submission.jsonl` plus one log record per question in `/result/logs/`: id, answer, elapsed_ms, thinking, code, output. If vLLM fails to start, the pipeline still writes every answer without the LLM.

Before building:
- put the model weights in `models/llm`;
- put the example bank in `src/socai/v2/bank.json`, plus `src/socai/v5/bank.json` for v5 modes.

Both banks are generated from training data and are not in the repository. The build stops if they are missing.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile -t socai:final .
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v /path/to/split:/data:ro -v /path/to/out:/result socai:final /code/predict.sh
```

## Model

| Model | Source | Revision | Released | License | Parameters |
|---|---|---|---|---|---|
| Qwen3-4B (submitted image) | https://huggingface.co/Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c | 2025-04-27 | Apache-2.0 | 4,022,468,096 |
| Qwen3-14B (Kaggle experiments) | https://huggingface.co/Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 | 2025-04-27 | Apache-2.0 | 14,768,307,200 |

Neither model is fine-tuned. A LoRA fine-tuning pipeline exists in `training_code/`, but it was never measured or shipped.

## Repository layout

```
src/socai/
  io/ profile/ rules/ llm/ eval/      shared: file readers, shop README, time expressions, lunar calendar, LLM client, scorer
  nlq/ exec/ answer/ semantic/        v1 rule path, canonical tables, answer policy (also used by v2)
  agent.py runner.py __main__.py      v1/v2 entry point: python -m socai
  v2/                                 query algebra, views, example bank retrieval
  v4/                                 six-layer pipeline: python -m socai.v4
  v5/                                 SQL + voting pipeline: python -m socai.v5
  synth/                              synthetic shops for tests
docs/          design notes per version (Vietnamese); chung.md covers the shared parts
kaggle/        Qwen3-14B runners for Kaggle 2×T4
scripts/       hold-out evaluation, example-bank replay, shop audit
training_code/ example-bank building and the (unused) LoRA fine-tuning pipeline
tests/         unit and end-to-end tests (fake LLM, synthetic and real data)
```

## Not in this repository
- **Data:** the training, public, and private splits are not published.
- **Generated example banks** (`bank.json`) and fine-tuning data: they contain training questions.
- **Model weights:** download them from the official sources at the revisions above.
