# ============================================
# IndexTTS2 API Server - Official Python Image
# ============================================

FROM docker.m.daocloud.io/python:3.10-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ============================================
# 安装系统依赖
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 音频处理
    ffmpeg \
    libsndfile1 \
    # 编译工具（DeepSpeed 需要）
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 复制项目文件
# ============================================
COPY . .

# ============================================
# 安装 Python 依赖
# ============================================
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

# ============================================
# 创建必要目录
# ============================================
RUN mkdir -p models/

# 暴露端口
EXPOSE 8020

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8020/')"

# 默认启动命令
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8020"]

