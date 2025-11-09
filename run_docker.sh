#!/bin/bash
# IndexTTS Docker 多GPU服务启动脚本
# 用法: ./run_docker.sh [GPU_IDS] [WORKERS] [PORT] [CONTAINER_NAME]
# 示例: ./run_docker.sh 0,1 4 8020 voicetts

GPU_IDS=${1:-"0,1"}
WORKERS=${2:-4}
PORT=${3:-8020}
CONTAINER_NAME=${4:-"voicetts"}

echo "========================================"
echo "IndexTTS Docker Service"
echo "Container: $CONTAINER_NAME"
echo "GPU: $GPU_IDS | Workers: $WORKERS | Port: $PORT"
echo "========================================"

# 停止并删除已存在的同名容器
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping and removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# 启动新容器
docker run -itd --gpus all --name $CONTAINER_NAME \
-p $PORT:$PORT \
-w /app \
-v $PWD:/app \
-e HF_ENDPOINT="https://hf-mirror.com" \
-e HF_HUB_CACHE=/app/models/hf_cache \
-e CUDA_VISIBLE_DEVICES=$GPU_IDS \
-e TZ=Asia/Shanghai \
--network host \
--restart unless-stopped \
voicetts:dev \
python server.py --host 0.0.0.0 --port $PORT --workers $WORKERS

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "✓ Container started successfully!"
    echo "  Logs: docker logs -f $CONTAINER_NAME"
    echo "  Stop: docker stop $CONTAINER_NAME"
    echo "  API:  http://localhost:$PORT"
    echo "  Test: curl http://localhost:$PORT/debug/worker-info"
    echo "========================================"
else
    echo "✗ Failed to start container!"
    exit 1
fi
