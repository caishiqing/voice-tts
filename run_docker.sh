#!/bin/bash
# IndexTTS Docker 多GPU服务启动脚本
# 用法: ./run_docker.sh [GPU_IDS] [WORKERS] [PORT] [CONTAINER_NAME]
# 示例: ./run_docker.sh 0,1 4 8020 voicetts
#
# 环境变量配置（可选）:
#   REDIS_HOST: Redis 服务器地址，默认 localhost
#   REDIS_PORT: Redis 端口，默认 6379
#   REDIS_AUDIO_EXPIRE: 音频缓存过期时间（秒），默认 3600（1小时）
#   REDIS_ENABLED: 是否启用缓存，默认 true

GPU_IDS=${1:-"0"}
WORKERS=${2:-1}
PORT=${3:-8020}
CONTAINER_NAME=${4:-"voicetts"}

# Redis 配置（从环境变量获取或使用默认值）
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}
REDIS_AUDIO_EXPIRE=${REDIS_AUDIO_EXPIRE:-3600}
REDIS_ENABLED=${REDIS_ENABLED:-"true"}

echo "========================================"
echo "IndexTTS Docker Service"
echo "Container: $CONTAINER_NAME"
echo "GPU: $GPU_IDS | Workers: $WORKERS | Port: $PORT"
echo "Redis: $REDIS_HOST:$REDIS_PORT (enabled=$REDIS_ENABLED, expire=${REDIS_AUDIO_EXPIRE}s)"
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
-e REDIS_HOST=$REDIS_HOST \
-e REDIS_PORT=$REDIS_PORT \
-e REDIS_AUDIO_EXPIRE=$REDIS_AUDIO_EXPIRE \
-e REDIS_ENABLED=$REDIS_ENABLED \
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
