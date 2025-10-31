# ============================================
# 单阶段构建 Dockerfile - Ubuntu 22.04
# 仅包含后端环境
# ============================================

FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ============================================
# 安装所有系统依赖
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 音频处理相关
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    # Python 相关
    software-properties-common \
    # 编译工具
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    # 清理
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 安装 Python 3.11
# ============================================
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python3.11 -m ensurepip --upgrade \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# ============================================
# 复制项目所有文件
# ============================================
COPY . .

# ============================================
# 安装 Python 依赖（使用 pyproject.toml）
# ============================================
RUN pip install -e .

# ============================================
# 创建数据目录
# ============================================
RUN mkdir -p data/db data/voices models/

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8020/')"

# 默认启动命令
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8020"]

