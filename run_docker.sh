#!/bin/bash

docker run -itd --gpus all --name voicetts \
-p 8020:8020 \
-w /app \
-v $PWD/indextts:/app/indextts \
-e HF_ENDPOINT="https://hf-mirror.com" \
-e HF_HUB_CACHE=/app/models/hf_cache \
-e CUDA_VISIBLE_DEVICES=6 \
-e TZ=Asia/Shanghai \
--network host \
--restart unless-stopped \
voicetts:dev \
python server.py --host 0.0.0.0 --port 8020 --workers 2
