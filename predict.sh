#!/usr/bin/env bash
# Entry point: bash /code/predict.sh (không tham số).
# /data (chỉ đọc) → /result/submission.jsonl + /result/logs/. Mọi tham số cố định ở đây (Rule 5.3).
set -uo pipefail
T0=$(date +%s)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 PYTHONHASHSEED=0
export SOCAI_CACHE=/tmp/socai_cache PYTHONPATH=/code/src
MODEL_DIR=${SOCAI_MODEL_DIR:-/code/models/llm}
mkdir -p /result/logs "$SOCAI_CACHE"

NGPU=$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
LLM_FLAG=""
PIPE=${SOCAI_PIPELINE:-v2}   # v2 (mặc định) hoặc v5
vllm() {  # $1 = cổng, phần còn lại = tham số thêm. Seed cố định; max-num-seqs=1 vì batching GPU có thể làm lệch đầu ra (Rule 8: |ΔPA| ≤ 0.03).
  local port=$1; shift
  python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --served-model-name local \
    --host 127.0.0.1 --port "$port" --seed 0 --max-num-seqs 1 \
    --enable-prefix-caching --max-model-len 32768 --gpu-memory-utilization 0.85 "$@" \
    > "/result/logs/vllm_$port.log" 2>&1 &
  PIDS="$PIDS $!"
}
PIDS=""; URLS=""
if [ "$NGPU" -ge 1 ] && [ -d "$MODEL_DIR" ]; then
  if [ "$PIPE" = v5 ]; then
    # v5: mỗi GPU một server, pipeline chia câu hỏi cho từng server
    for g in $(seq 0 $((NGPU - 1))); do
      CUDA_VISIBLE_DEVICES=$g vllm $((8000 + g))
      URLS="${URLS:+$URLS,}http://127.0.0.1:$((8000 + g))/v1"
    done
  else
    vllm 8000 --tensor-parallel-size "$NGPU"
    URLS="http://127.0.0.1:8000/v1"
  fi
  for _ in $(seq 1 600); do  # chờ tối đa 10 phút
    UP=1
    for u in ${URLS//,/ }; do curl -sf "$u/models" >/dev/null 2>&1 || UP=0; done
    [ "$UP" = 1 ] && break
    for pid in $PIDS; do kill -0 "$pid" 2>/dev/null || break 2; done
    sleep 1
  done
  [ "${UP:-0}" = 1 ] || { echo "[predict] vLLM không lên → không dùng LLM"; LLM_FLAG="--no-llm"; URLS=""; }
else
  echo "[predict] không có GPU/mô hình → không dùng LLM"
  LLM_FLAG="--no-llm"
fi
echo "[predict] load_model_s=$(( $(date +%s) - T0 ))"

if [ "$PIPE" = v5 ]; then
  # Không có LLM thì v5 bỏ trống mọi câu (vẫn ghi đủ submission.jsonl). --budget-min tính cả thời gian nạp mô hình.
  python3 -m socai.v5 predict --data /data --out /result --cache "$SOCAI_CACHE" ${URLS:+--llm "$URLS"} \
    --samples "${SOCAI_SAMPLES:-4}" --budget-min $(( 324 - ($(date +%s) - T0) / 60 ))
else
  python3 -m socai predict --data /data --out /result $LLM_FLAG
fi
STATUS=$?
for pid in $PIDS; do kill "$pid" 2>/dev/null; done
echo "[predict] total_s=$(( $(date +%s) - T0 )) status=$STATUS"
exit $STATUS
