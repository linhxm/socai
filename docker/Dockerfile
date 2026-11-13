# Image chạy offline cho SOCAI.
# Build: docker build --platform linux/amd64 -f docker/Dockerfile -t socai:final .
# Thử offline: docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
#   -v /path/public_test:/data:ro -v /path/out:/result socai:final /code/predict.sh
FROM vllm/vllm-openai:v0.10.1.1
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1 PYTHONHASHSEED=0
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
# Mẫu tham chiếu sửa treo khởi tạo 2 GPU bằng NCCL 2.27.7 — cần tự kiểm lại với stack của mình
RUN pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.27.7
# Tách COPY để giữ cache layer trọng số khi chỉ sửa code
COPY models/ /code/models/
COPY training_code/ /code/training_code/
COPY predict.sh README.md /code/
COPY src/ /code/src/
RUN chmod +x /code/predict.sh
# Chọn đường chạy (BTC chạy predict.sh không truyền biến môi trường): v2 | v5 | v2+v5. Chỉ đổi sau khi đo trên Kaggle.
ENV SOCAI_PIPELINE=v2 SOCAI_SAMPLES=4
# Tuỳ chọn của v5, bật sau khi đo (docs/v5.md §5): ENV SOCAI_VERIFY=1 · ENV SOCAI_V2_TABLES=1
# v5 cần ngân hàng ví dụ (bị gitignore): thiếu thì dừng build thay vì nộp bản mất ~0,15 PA
RUN case "$SOCAI_PIPELINE" in *v5*) test -s /code/src/socai/v5/bank.json || { echo "thiếu src/socai/v5/bank.json"; exit 1; } ;; esac
CMD ["bash", "/code/predict.sh"]
