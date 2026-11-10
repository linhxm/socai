#!/usr/bin/env bash
# Entry point theo hướng dẫn Docker DataMate: bash /code/predict.sh (không tham số).
# /data (chỉ đọc) → /result/submission.jsonl + /result/logs/. Mọi tham số cố định ở đây (Rule 5.3).
set -uo pipefail
T0=$(date +%s)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=0
export SOCAI_CACHE=/tmp/socai_cache PYTHONPATH=/code/src
MODEL_DIR=${SOCAI_MODEL_DIR:-/code/models/llm}
mkdir -p /result/logs "$SOCAI_CACHE"

NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
LLM_FLAG=""
if [ "$NGPU" -ge 1 ] && [ -d "$MODEL_DIR" ]; then
  # Greedy + seed cố định; max-num-seqs=1 vì batching GPU có thể làm lệch đầu ra (Rule 8: |ΔPA| ≤ 0.03).
  python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --served-model-name local \
    --host 127.0.0.1 --port 8000 --tensor-parallel-size "$NGPU" --seed 0 --max-num-seqs 1 \
    --enable-prefix-caching --max-model-len 32768 --gpu-memory-utilization 0.85 \
    > /result/logs/vllm.log 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 600); do  # chờ tối đa 10 phút
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    kill -0 $VLLM_PID 2>/dev/null || break
    sleep 1
  done
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || { echo "[predict] vLLM không lên → chỉ đường nhanh"; LLM_FLAG="--no-llm"; }
else
  echo "[predict] không có GPU/mô hình → chỉ đường nhanh"
  LLM_FLAG="--no-llm"
fi
echo "[predict] load_model_s=$(( $(date +%s) - T0 ))"

python3 -m socai predict --data /data --out /result $LLM_FLAG
STATUS=$?
[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null
echo "[predict] total_s=$(( $(date +%s) - T0 )) status=$STATUS"
exit $STATUS
