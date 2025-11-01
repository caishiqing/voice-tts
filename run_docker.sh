#!/bin/bash

docker run -itd --gpus '"device=7"' --name voicetts \
-p 8020:8020 \
-w /app \
-e HF_ENDPOINT="https://hf-mirror.com" \
-e TZ=Asia/Shanghai \
--network host \
--restart unless-stopped \
voice-tts:latest \
python server.py --host 0.0.0.0 --port 8020 --workers 3
