from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger
import uvicorn
import os
import tempfile
import requests
import wave
import re
from enum import Enum

# 延迟导入 torch 和模型相关的模块，避免在主进程中初始化 CUDA 上下文
# torch 和 IndexTTS2 将在子进程的 lifespan 中导入

# 全局变量存储模型（在子进程中初始化）
tts_model = None
USE_DEEPSPEED = None  # 将在子进程中检测


class Emotion(str, Enum):
    """情绪枚举"""
    NORMAL = "normal"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期 - 在子进程中初始化所有 torch 和模型相关的内容"""
    global tts_model, USE_DEEPSPEED
    
    # 在子进程中导入 torch 和模型
    import torch
    from indextts.infer_v2 import IndexTTS2
    
    # 在子进程中检测 DeepSpeed 和 CUDA 是否可用
    def check_deepspeed_availability():
        """检测 DeepSpeed 是否安装并且 CUDA 是否可用"""
        if not torch.cuda.is_available():
            logger.info("CUDA is not available, DeepSpeed will be disabled")
            return False

        try:
            import deepspeed
            logger.success(f"DeepSpeed is available (version: {deepspeed.__version__}), acceleration will be enabled")
            return True
        except ImportError:
            logger.warning("DeepSpeed is not installed, falling back to normal inference. Install with: pip install deepspeed")
            return False
        except Exception as e:
            logger.warning(f"DeepSpeed is not available due to: {e}. Falling back to normal inference.")
            return False
    
    # Startup
    try:
        logger.info("Worker process started, initializing torch and models...")
        USE_DEEPSPEED = check_deepspeed_availability()
        
        logger.info("Loading IndexTTS2 model...")
        logger.info(f"DeepSpeed acceleration: {'enabled' if USE_DEEPSPEED else 'disabled'}")
        tts_model = IndexTTS2(
            cfg_path="models/IndexTTS/config.yaml",
            model_dir="models/IndexTTS",
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=USE_DEEPSPEED
        )
        logger.success("Model loaded successfully in worker process!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    yield

    # Shutdown
    logger.info("Worker process shutting down...")


app = FastAPI(title="IndexTTS API Server - Stateless", lifespan=lifespan)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_hex_string(s: str) -> bool:
    """判断字符串是否为有效的hex编码"""
    if not s:
        return False
    # hex字符串应该只包含0-9, a-f, A-F，且长度为偶数
    # 长度大于100字符，避免误判短字符串（因为音频数据通常很长）
    return bool(re.match(r'^[0-9a-fA-F]+$', s)) and len(s) % 2 == 0 and len(s) > 100


def is_url(s: str) -> bool:
    """判断字符串是否为URL"""
    return s.startswith(('http://', 'https://', 'ftp://'))


def download_audio_from_url(url: str, timeout: float = 30.0) -> bytes:
    """
    从URL下载音频文件并返回二进制数据

    Args:
        url: 音频文件的URL
        timeout: 下载超时时间（秒）

    Returns:
        音频文件的二进制数据

    Raises:
        HTTPException: 下载失败时抛出
    """
    try:
        logger.info(f"Downloading audio from URL: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # 检查Content-Type（可选）
        content_type = response.headers.get('content-type', '')
        if content_type and not any(t in content_type.lower() for t in ['audio', 'octet-stream', 'wav', 'mp3', 'mpeg']):
            logger.warning(f"URL返回的Content-Type可能不是音频: {content_type}")

        audio_data = response.content
        logger.success(f"Downloaded audio from URL: {url}, size={len(audio_data)} bytes")
        return audio_data

    except requests.Timeout:
        logger.error(f"Download timeout: {url}")
        raise HTTPException(status_code=408, detail=f"Download timeout: {url}")
    except requests.HTTPError as e:
        logger.error(f"Failed to download audio: HTTP {e.response.status_code}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Failed to download audio from URL: HTTP {e.response.status_code}"
        )
    except Exception as e:
        logger.error(f"Error downloading audio from URL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading audio from URL: {str(e)}"
        )


def get_audio_data(audio_input: str) -> bytes:
    """
    从不同类型的输入获取音频数据

    Args:
        audio_input: 音频输入，可以是URL或hex编码字符串

    Returns:
        音频文件的二进制数据

    Raises:
        HTTPException: 处理失败时抛出
    """
    if is_url(audio_input):
        # 如果是URL，下载音频
        return download_audio_from_url(audio_input)
    elif is_hex_string(audio_input):
        # 如果是hex编码，直接解码
        try:
            logger.info(f"Decoding hex audio data, size={len(audio_input)//2} bytes")
            return bytes.fromhex(audio_input)
        except ValueError as e:
            logger.error(f"Invalid hex string: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid hex encoded audio data: {str(e)}")
    else:
        # 无法识别的格式
        raise HTTPException(
            status_code=400,
            detail="Invalid audio input format. Must be URL (http://, https://) or hex encoded string"
        )


class TTSRequest(BaseModel):
    """TTS推理请求模型"""
    text: str = Field(..., description="要合成的文本")
    spk_audio: str = Field(..., description="说话人参考音频（支持URL或hex编码）")
    emo_audio: Optional[str] = Field(None, description="情绪参考音频（支持URL或hex编码，可选）")
    emo_alpha: float = Field(default=1.0, description="情绪强度（0.0-1.0），默认1.0")

    @validator('emo_alpha')
    def validate_emo_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('emo_alpha must be between 0.0 and 1.0')
        return v


class TTSResponse(BaseModel):
    """TTS推理响应模型"""
    audio_hex: str = Field(..., description="hex编码的音频数据")
    audio_length: float = Field(..., description="音频长度（秒）")
    inference_time: float = Field(..., description="推理时间（秒）")
    rtf: float = Field(..., description="Real-Time Factor")
    text: str = Field(..., description="输入的文本")


@app.get("/")
def root():
    """健康检查端点"""
    return {
        "status": "running",
        "model_loaded": tts_model is not None,
        "service": "IndexTTS API Server - Stateless",
        "version": "2.0"
    }


@app.get("/health")
def health_check():
    """健康检查端点"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "deepspeed_enabled": USE_DEEPSPEED
    }


@app.post("/tts", response_model=TTSResponse)
def text_to_speech(request: TTSRequest):
    """
    TTS文本转语音接口（无状态版本）

    通过URL或hex编码传入参考音频进行语音合成
    - spk_audio: 说话人参考音频（URL或hex编码，必需）
    - emo_audio: 情绪参考音频（URL或hex编码，可选）
    - emo_alpha: 情绪强度（0.0-1.0）

    Args:
        request: TTS请求，包含文本和音频数据

    Returns:
        包含hex编码音频数据的响应
    """
    if tts_model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    output_path = None

    try:
        # 获取说话人参考音频数据
        logger.info(f"Processing TTS request: text='{request.text[:50]}...'")
        spk_audio_data = get_audio_data(request.spk_audio)

        # 获取情绪参考音频数据（如果提供）
        emo_audio_data = None
        if request.emo_audio:
            emo_audio_data = get_audio_data(request.emo_audio)

        logger.info(
            f"Audio data ready: spk_size={len(spk_audio_data)} bytes, "
            f"emo_size={len(emo_audio_data) if emo_audio_data else 0} bytes, "
            f"emo_alpha={request.emo_alpha}"
        )

        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        # 记录推理开始时间
        import time
        start_time = time.time()

        # 执行推理 - 直接传入 bytes 数据
        result_path = tts_model.infer(
            spk_audio_prompt=spk_audio_data,
            text=request.text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_data if emo_audio_data else None,
            emo_alpha=request.emo_alpha,
            verbose=False
        )

        inference_time = time.time() - start_time

        # 读取生成的音频文件并转换为hex
        with open(result_path, "rb") as f:
            audio_bytes = f.read()
            audio_hex = audio_bytes.hex()

        # 获取音频信息
        with wave.open(result_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            audio_length = frames / float(rate)

        # 计算 RTF
        rtf = inference_time / audio_length if audio_length > 0 else 0.0

        # 清理临时文件
        os.remove(output_path)

        logger.success(
            f"TTS completed: audio_length={audio_length:.2f}s, "
            f"inference_time={inference_time:.2f}s, rtf={rtf:.4f}, "
            f"size={len(audio_hex)//2} bytes"
        )

        return TTSResponse(
            audio_hex=audio_hex,
            audio_length=audio_length,
            inference_time=inference_time,
            rtf=rtf,
            text=request.text
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS inference failed: {str(e)}")
        # 清理可能的临时文件
        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="IndexTTS API Server - Stateless",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8020,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level"
    )

    args = parser.parse_args()

    # 如果指定了多个 worker，使用 uvicorn 的多进程模式
    if args.workers > 1:
        logger.info(f"Starting server with {args.workers} workers")
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level
        )
    else:
        # 单进程模式，支持 reload
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
