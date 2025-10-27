#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p models/
hf download IndexTeam/IndexTTS-2 --local-dir=models/IndexTTS

# 快速启动脚本：使用 docker run 运行容器
docker run -itd --gpus all --name indextts \
-p 8000:8000 \
-w /app \
-v $(pwd)/data:/app/data \
-v $(pwd)/models:/app/models \
-e TZ=Asia/Shanghai \
--network host \
--restart unless-stopped \
index-tts:latest \
python server.py --host 0.0.0.0 --port 8000 --workers 1
