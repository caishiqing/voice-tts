# IndexTTS2 API 服务

基于 IndexTTS2 模型的高性能语音合成 REST API 服务，支持无状态推理和多种音频输入格式。

> **IndexTTS2**: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech

## ✨ 核心特性

### 🚀 无状态推理
- 每次请求独立处理，无需预先上传音频
- 支持多种音频输入格式（URL、hex 编码）
- 自动格式识别，开箱即用

### 🎯 高性能
- 支持 DeepSpeed 加速（自动检测 CUDA）
- FP16 混合精度推理
- 实时因子（RTF）监控
- 多进程部署支持

### 🎵 灵活的音频输入
- **URL 模式**: 直接传入音频文件 URL
- **Hex 编码**: 传入 hex 编码的音频数据
- **自动识别**: 无需指定格式，自动判断

### 🎭 情绪控制
- 独立的说话人音频和情绪音频
- 可调节情绪强度（emo_alpha: 0.0-1.0）
- 支持多种情绪表达

## 📋 系统要求

- **Python**: 3.10+
- **CUDA**: 11.8+ (可选，用于 GPU 加速)
- **PyTorch**: 2.8+
- **操作系统**: Linux / Windows / macOS

## 🚀 快速开始

### 方式 1: 使用 uv (推荐)

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 启动服务
uv run python server.py
```

### 方式 2: 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .

# 启动服务
python server.py
```

### 方式 3: 使用 Docker

```bash
# 构建镜像
docker build -t indextts-api .

# 运行容器（GPU）
docker run -d \
  --name indextts-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  indextts-api

# 运行容器（CPU）
docker run -d \
  --name indextts-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  indextts-api
```

## 📁 项目结构

```
voice-tts/
├── indextts/             # IndexTTS2 推理模块
│   ├── infer_v2.py      # 推理引擎
│   ├── gpt/             # GPT 模型
│   ├── BigVGAN/         # 声码器
│   └── utils/           # 工具函数
├── models/              # 模型文件目录
│   └── IndexTTS/
│       ├── config.yaml
│       └── ...
├── server.py            # FastAPI 服务器
├── Dockerfile           # Docker 配置
├── run_docker.sh        # Docker 快速启动脚本
└── pyproject.toml       # 项目配置
```

## 🎮 启动选项

```bash
# 默认启动（端口 8000）
python server.py

# 指定端口和主机
python server.py --host 0.0.0.0 --port 8213

# 多进程模式（提高并发）
python server.py --workers 4

# 开发模式（自动重载）
python server.py --reload

# 调试模式（详细日志）
python server.py --log-level debug
```

### 可用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --host | string | 0.0.0.0 | 绑定的主机地址 |
| --port | int | 8000 | 绑定的端口 |
| --workers | int | 1 | Worker 进程数 |
| --reload | flag | False | 开发模式（自动重载） |
| --log-level | string | info | 日志级别（debug/info/warning/error） |

## 📚 API 文档

服务启动后访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 核心端点

#### 1. 健康检查

```bash
GET /health
```

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "deepspeed_enabled": true
}
```

#### 2. TTS 推理

```bash
POST /tts
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | ✅ | 要合成的文本 |
| spk_audio | string | ✅ | 说话人参考音频（URL 或 hex） |
| emo_audio | string | ❌ | 情绪参考音频（URL 或 hex） |
| emo_alpha | float | ❌ | 情绪强度（0.0-1.0），默认 1.0 |

**方式 1: 使用 URL**

```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个语音合成测试",
    "spk_audio": "https://example.com/speaker.wav",
    "emo_audio": "https://example.com/emotion.wav",
    "emo_alpha": 1.0
  }'
```

**方式 2: 使用 Hex 编码**

```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个语音合成测试",
    "spk_audio": "52494646...",
    "emo_alpha": 0.8
  }'
```

**方式 3: 混合使用**

```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个语音合成测试",
    "spk_audio": "https://example.com/speaker.wav",
    "emo_audio": "52494646...",
    "emo_alpha": 0.5
  }'
```

**响应示例：**

```json
{
  "audio_hex": "52494646...",
  "audio_length": 2.5,
  "inference_time": 0.35,
  "rtf": 0.14,
  "text": "你好，这是一个语音合成测试"
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| audio_hex | string | hex 编码的 WAV 音频数据 |
| audio_length | float | 音频时长（秒） |
| inference_time | float | 推理耗时（秒） |
| rtf | float | 实时因子（越小越好） |
| text | string | 输入的文本 |

## 💻 客户端示例

### Python 客户端

```python
import requests

def tts_request(text, spk_audio, emo_audio=None, emo_alpha=1.0):
    """调用 TTS API"""
    url = "http://localhost:8000/tts"
    
    payload = {
        "text": text,
        "spk_audio": spk_audio,
        "emo_alpha": emo_alpha
    }
    
    if emo_audio:
        payload["emo_audio"] = emo_audio
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()

def save_audio(audio_hex, output_path):
    """保存 hex 编码的音频"""
    audio_bytes = bytes.fromhex(audio_hex)
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)

# 使用示例
result = tts_request(
    text="你好，世界！",
    spk_audio="https://example.com/speaker.wav",
    emo_alpha=0.8
)

print(f"音频时长: {result['audio_length']:.2f}秒")
print(f"推理耗时: {result['inference_time']:.2f}秒")
print(f"RTF: {result['rtf']:.4f}")

# 保存音频
save_audio(result['audio_hex'], "output.wav")
```

### JavaScript/TypeScript 客户端

```typescript
interface TTSRequest {
  text: string;
  spk_audio: string;
  emo_audio?: string;
  emo_alpha?: number;
}

interface TTSResponse {
  audio_hex: string;
  audio_length: number;
  inference_time: number;
  rtf: number;
  text: string;
}

async function textToSpeech(request: TTSRequest): Promise<TTSResponse> {
  const response = await fetch('http://localhost:8000/tts', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`TTS 请求失败: ${response.statusText}`);
  }
  
  return await response.json();
}

// 将 hex 转换为音频 Blob
function hexToBlob(hex: string): Blob {
  const bytes = new Uint8Array(
    hex.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16))
  );
  return new Blob([bytes], { type: 'audio/wav' });
}

// 使用示例
const result = await textToSpeech({
  text: '你好，世界！',
  spk_audio: 'https://example.com/speaker.wav',
  emo_alpha: 0.8,
});

// 播放音频
const audioBlob = hexToBlob(result.audio_hex);
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

## 🎵 音频格式处理

### 支持的输入格式

API 自动识别以下格式：

1. **URL 格式**: 以 `http://`、`https://` 或 `ftp://` 开头
2. **Hex 编码**: 纯 16 进制字符串（长度为偶数，大于 100 字符）

### 转换为 Hex 编码

**Python:**
```python
def audio_to_hex(audio_path):
    """将音频文件转换为 hex 编码"""
    with open(audio_path, 'rb') as f:
        return f.read().hex()

hex_audio = audio_to_hex("speaker.wav")
```

**JavaScript:**
```javascript
async function audioToHex(file) {
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

const hexAudio = await audioToHex(audioFile);
```

**命令行:**
```bash
# Linux/Mac
xxd -p audio.wav | tr -d '\n' > audio.hex

# 或使用 Python
python -c "print(open('audio.wav','rb').read().hex())" > audio.hex
```

## ⚡ 性能优化

### DeepSpeed 加速

服务会自动检测并启用 DeepSpeed（需要 CUDA）：

```bash
# 确认 DeepSpeed 已安装
pip show deepspeed

# 启动服务（自动启用）
python server.py
```

**日志输出：**
```
✅ DeepSpeed is available (version: 0.18.1), acceleration will be enabled
```

### 性能指标

**RTF (Real-Time Factor)** = 推理时间 / 音频时长

- **RTF < 1.0**: ✅ 实时生成
- **RTF < 0.5**: ⚡ 高性能
- **RTF < 0.2**: 🚀 超高性能
- **RTF > 1.0**: ⚠️ 需要优化

### 多进程部署

提高并发处理能力：

```bash
# 4 个 worker 进程
python server.py --workers 4

# 结合 nginx 做负载均衡
# 每个 worker 监听不同端口
```

## 🐳 Docker 部署

### 使用预构建脚本

```bash
# 快速启动
chmod +x run_docker.sh
./run_docker.sh
```

### 手动构建和运行

```bash
# 构建镜像
docker build -t indextts-api .

# GPU 运行
docker run -d \
  --name indextts-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  indextts-api

# CPU 运行
docker run -d \
  --name indextts-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  indextts-api
```

### Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  tts-api:
    build: .
    container_name: indextts-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
```

启动：
```bash
docker-compose up -d
```

## 🔧 故障排查

### 常见错误

| 状态码 | 错误 | 原因 | 解决方案 |
|--------|------|------|----------|
| 400 | Invalid audio input format | 音频格式无法识别 | 检查 URL 或 hex 编码格式 |
| 408 | Download timeout | URL 下载超时 | 检查网络或增加超时时间 |
| 422 | Unprocessable Entity | 参数验证失败 | 检查请求参数是否正确 |
| 503 | Model not loaded | 模型未加载 | 等待模型加载完成 |
| 500 | TTS inference failed | 推理失败 | 查看服务器日志 |

### 调试技巧

**1. 查看详细日志**
```bash
python server.py --log-level debug
```

**2. 测试健康状态**
```bash
curl http://localhost:8000/health
```

**3. 验证音频格式**
```python
import re

def is_valid_hex(s):
    return bool(re.match(r'^[0-9a-fA-F]+$', s)) and len(s) % 2 == 0

def is_valid_url(s):
    return s.startswith(('http://', 'https://'))
```

**4. 检查模型文件**
```bash
ls -la models/IndexTTS/
# 应包含: config.yaml, checkpoint files
```

### 模型加载失败

```bash
# 检查模型目录
ls -la models/IndexTTS/

# 确保包含必要文件
# - config.yaml
# - *.pt 或 *.pth (模型权重)
```

### CUDA 内存不足

```bash
# 监控 GPU 内存
nvidia-smi -l 1

# 如果内存不足，服务会自动回退到 CPU
```

### DeepSpeed 安装问题

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# 重新安装
pip uninstall deepspeed
pip install deepspeed --no-cache-dir
```

## 📋 最佳实践

### 音频质量建议

- **采样率**: 16kHz 或 22.05kHz
- **格式**: WAV（PCM 编码）
- **时长**: 3-10 秒参考音频效果最佳
- **质量**: 清晰无噪音的录音

### API 调用建议

1. **缓存音频**: 重复使用的音频可以缓存 hex 编码
2. **超时设置**: 设置合理的请求超时（30-60 秒）
3. **错误重试**: 实现指数退避重试机制
4. **并发控制**: 根据服务器性能控制并发数

### 生产环境建议

1. **反向代理**: 使用 Nginx 做负载均衡
2. **认证**: 添加 API Key 或 JWT 认证
3. **限流**: 使用 rate limiting 防止滥用
4. **监控**: 集成 Prometheus + Grafana
5. **日志**: 集中化日志管理（ELK Stack）
6. **HTTPS**: 生产环境必须使用 HTTPS

## 📖 相关资源

- **IndexTTS2 项目**: https://github.com/index-tts/index-tts
- **FastAPI 文档**: https://fastapi.tiangolo.com/
- **DeepSpeed 文档**: https://www.deepspeed.ai/

## 📄 许可证

本项目基于 MIT License 开源。

IndexTTS2 模型遵循 Bilibili IndexTTS License，详见 `INDEX_MODEL_LICENSE`。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📧 联系方式

如有问题或建议，欢迎提交 Issue。
