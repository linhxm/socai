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
CMD ["bash", "/code/predict.sh"]
