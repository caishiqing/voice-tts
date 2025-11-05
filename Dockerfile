# ============================================
# IndexTTS2 API Server - NVIDIA CUDA Image
# ============================================

FROM docker.m.daocloud.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ============================================
# 安装 Python 3.10 和系统依赖
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.10
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-distutils \
    # 音频处理
    ffmpeg \
    libsndfile1 \
    # 编译工具（DeepSpeed 需要）
    build-essential \
    git \
    # 其他必要工具
    wget \
    ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 下载模型
RUN mkdir -p models/
ENV HF_ENDPOINT="https://hf-mirror.com"
RUN hf download IndexTeam/IndexTTS-2 --local-dir=models/IndexTTS

# 暴露端口
EXPOSE 8020

# 默认启动命令
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8020"]

