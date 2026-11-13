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
PIPE=${SOCAI_PIPELINE:-v2}   # v2 | v5 | v2+v5 (v2 trả lời trước, v5 chỉ làm câu v2 bỏ trống). Đặt cố định bằng ENV trong Dockerfile.
PIDS=""; URL=""
if [ "$NGPU" -ge 1 ] && [ -d "$MODEL_DIR" ]; then
  # Một server chia mô hình lên mọi GPU (14B bf16 ~28 GB không vừa một GPU 32 GB với 0.85 bộ nhớ).
  # Seed cố định; max-num-seqs=1 vì batching GPU có thể làm lệch đầu ra (Rule 8: |ΔPA| ≤ 0.03).
  python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --served-model-name local \
    --host 127.0.0.1 --port 8000 --tensor-parallel-size "$NGPU" --seed 0 --max-num-seqs 1 \
    --enable-prefix-caching --max-model-len 32768 --gpu-memory-utilization 0.85 \
    > /result/logs/vllm.log 2>&1 &
  PIDS=$!
  for _ in $(seq 1 600); do  # chờ tối đa 10 phút
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    kill -0 "$PIDS" 2>/dev/null || break
    sleep 1
  done
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then URL=http://127.0.0.1:8000/v1
  else echo "[predict] vLLM không lên → không dùng LLM"; LLM_FLAG="--no-llm"; fi
else
  echo "[predict] không có GPU/mô hình → không dùng LLM"
  LLM_FLAG="--no-llm"
fi
echo "[predict] load_model_s=$(( $(date +%s) - T0 ))"

v5() {  # "$@" = tham số thêm. Không có LLM thì v5 bỏ trống mọi câu nhưng vẫn ghi đủ submission.jsonl.
  python3 -m socai.v5 predict --data /data --out /result --cache "$SOCAI_CACHE" ${URL:+--llm "$URL"} \
    --samples "${SOCAI_SAMPLES:-4}" --budget-min $(( 324 - ($(date +%s) - T0) / 60 )) ${SOCAI_V2_TABLES:+--v2-tables} "$@"
}
case "$PIPE" in
  v5) v5 ;;
  v2+v5)
    python3 -m socai predict --data /data --out /result/v2 $LLM_FLAG
    echo "[predict] v2 xong sau $(( $(date +%s) - T0 ))s"
    # SOCAI_VERIFY=1: v5 chạy cả câu v2 đã trả lời và bỏ trống câu mà cả k mẫu đồng thuận ở đáp án khác v2
    if [ -s /result/v2/submission.jsonl ]; then v5 --base /result/v2/submission.jsonl ${SOCAI_VERIFY:+--verify}; else v5; fi ;;
  *) python3 -m socai predict --data /data --out /result $LLM_FLAG ;;
esac
STATUS=$?
for pid in $PIDS; do kill "$pid" 2>/dev/null; done
echo "[predict] total_s=$(( $(date +%s) - T0 )) status=$STATUS"
exit $STATUS
