from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Union, Dict
from contextlib import asynccontextmanager
from loguru import logger
import uvicorn
import os
import tempfile
import requests
import wave
import re
import threading
from enum import Enum
import json

# 延迟导入 torch 和模型相关的模块，避免在主进程中初始化 CUDA 上下文
# torch 和 IndexTTS2 将在子进程的 lifespan 中导入

# 全局变量存储模型（在子进程中初始化）
tts_model = None
USE_DEEPSPEED = None  # 将在子进程中检测

# 推理锁：确保线程安全，同一时间只有一个线程在执行推理
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期 - 在子进程中初始化所有 torch 和模型相关的内容"""
    global tts_model, USE_DEEPSPEED
    
    # 在子进程中导入 torch 和模型
    import torch
    from indextts.infer_v2 import IndexTTS2
    
    # 检测 DeepSpeed 和 CUDA 是否可用
    def check_deepspeed_availability():
        if not torch.cuda.is_available():
            logger.info("CUDA is not available, DeepSpeed will be disabled")
            return False
        try:
            import deepspeed
            logger.success(f"DeepSpeed is available (version: {deepspeed.__version__})")
            return True
        except ImportError:
            logger.warning("DeepSpeed is not installed, falling back to normal inference")
            return False
        except Exception as e:
            logger.warning(f"DeepSpeed is not available: {e}")
            return False
    
    # Startup
    try:
        worker_id = os.environ.get('WORKER_ID', 'unknown')
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'default')
        logger.info(f"Worker {worker_id} (PID: {os.getpid()}) starting, GPU: {cuda_visible}")
        
        USE_DEEPSPEED = check_deepspeed_availability()
        
        logger.info(f"Loading IndexTTS2 model (DeepSpeed: {'enabled' if USE_DEEPSPEED else 'disabled'})...")
        tts_model = IndexTTS2(
            cfg_path="models/IndexTTS/config.yaml",
            model_dir="models/IndexTTS",
            use_fp16=True,
            use_cuda_kernel=True,
            use_deepspeed=USE_DEEPSPEED
        )
        logger.success(f"Model loaded successfully on GPU: {cuda_visible}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

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
    emotion: Optional[Union[str, Dict[str, float]]] = Field(
        None, 
        description="情绪控制（可选），支持两种格式：\n"
                   "1. 字符串：单个情绪标签，支持中英文同义词（如 'happy', '高兴', 'joyful'）\n"
                   "2. 字典：多个情绪组合，如 {'happy': 0.7, '生气': 0.3}\n"
                   "标准维度：happy/高兴, angry/愤怒, sad/悲伤, afraid/恐惧, "
                   "disgusted/反感, melancholic/低落, surprised/惊讶, calm/平静。\n"
                   "优先级：emo_audio > emotion"
    )
    emo_alpha: float = Field(
        default=1.0, 
        description="情绪强度（0.0-1.0），默认1.0。\n"
                   "当emotion为字符串时：作为该情绪维度的值\n"
                   "当emotion为字典时：此参数被忽略，使用字典中的值"
    )

    @validator('emo_alpha')
    def validate_emo_alpha(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('emo_alpha must be between 0.0 and 1.0')
        return v
    
    @validator('emotion')
    def validate_emotion(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            # 验证字典中的值都是数字
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError(f"Emotion dict keys must be strings, got {type(key)}")
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Emotion dict values must be numbers, got {type(value)}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Emotion values must be between 0.0 and 1.0, got {value}")
            return v
        raise ValueError("emotion must be a string or dict")


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


@app.get("/debug/worker-info")
def worker_info():
    """
    调试端点：返回当前 worker 的信息
    
    用于验证 GPU 分配是否正确
    多次请求该端点会看到不同的 worker 响应
    
    使用方法：
        for i in {1..10}; do curl http://localhost:8020/debug/worker-info; echo ""; done
    """
    import torch
    
    worker_id = os.environ.get('WORKER_ID', 'unknown')
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    pid = os.getpid()
    
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'device_name': None,
        'device_properties': None,
    }
    
    if torch.cuda.is_available():
        gpu_info['device_count'] = torch.cuda.device_count()
        try:
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            
            # 获取设备属性
            props = torch.cuda.get_device_properties(0)
            gpu_info['device_properties'] = {
                'name': props.name,
                'total_memory': f"{props.total_memory / 1024**3:.2f} GB",
                'major': props.major,
                'minor': props.minor,
            }
        except Exception as e:
            gpu_info['error'] = str(e)
    
    model_info = {
        'loaded': tts_model is not None,
        'device': str(tts_model.device) if tts_model else 'not loaded',
        'use_fp16': tts_model.use_fp16 if tts_model else None,
        'use_deepspeed': USE_DEEPSPEED,
    }
    
    return {
        'worker_id': worker_id,
        'pid': pid,
        'cuda_visible_devices': cuda_visible,
        'gpu_info': gpu_info,
        'model_info': model_info,
    }


@app.post("/tts", response_model=TTSResponse)
def text_to_speech(request: TTSRequest):
    """
    TTS文本转语音接口（无状态版本）

    通过URL或hex编码传入参考音频进行语音合成
    - spk_audio: 说话人参考音频（URL或hex编码，必需）
    - emo_audio: 情绪参考音频（URL或hex编码，可选，优先级高）
    - emotion: 情绪标签（可选，当未提供emo_audio时使用）
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
        emo_vector = None
        
        if request.emo_audio:
            # 优先使用 emo_audio
            emo_audio_data = get_audio_data(request.emo_audio)
            logger.info(f"Using emo_audio: size={len(emo_audio_data)} bytes, alpha={request.emo_alpha}")
        elif request.emotion:
            # 如果没有 emo_audio，但提供了 emotion 标签或字典
            from emotion import create_emotion_vector
            
            if isinstance(request.emotion, str):
                # 模式1：单个情绪标签字符串
                logger.info(f"Using emotion label: '{request.emotion}', alpha={request.emo_alpha}")
                emo_vector = create_emotion_vector(request.emotion, request.emo_alpha)
                
            elif isinstance(request.emotion, dict):
                # 模式2：情绪字典（支持多个情绪组合）
                logger.info(f"Using emotion dict: {request.emotion}")
                emo_vector = create_emotion_vector(request.emotion)
            
            logger.info(f"Generated emotion vector: {emo_vector}")

        if not request.emo_audio and not request.emotion:
            logger.info(f"No emotion control specified, using default (emo_alpha={request.emo_alpha})")

        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        # 记录推理开始时间
        import time
        start_time = time.time()

        # 执行推理 - 使用锁保证线程安全，同一时间只有一个线程在执行推理
        with inference_lock:
            logger.debug("Acquired inference lock, starting model inference...")
            result_path = tts_model.infer(
                spk_audio_prompt=spk_audio_data,
                text=request.text,
                output_path=output_path,
                emo_audio_prompt=emo_audio_data if emo_audio_data else None,
                emo_alpha=request.emo_alpha if emo_audio_data else 1.0,  # emotion模式下alpha已体现在vector中
                emo_vector=emo_vector,  # 传入情绪向量（如果有）
                verbose=False
            )
            logger.debug("Model inference completed, releasing lock...")

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

    # 如果指定了多个 worker，使用 Gunicorn + UvicornWorker
    if args.workers > 1:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        gpus = [g.strip() for g in cuda_visible.split(',') if g.strip()] if cuda_visible else []
        
        logger.info(f"Starting Gunicorn server: {args.workers} workers, {args.host}:{args.port}")
        if gpus:
            logger.info(f"GPU assignment: {len(gpus)} GPU(s) [{cuda_visible}], round-robin distribution")
        else:
            logger.warning("CUDA_VISIBLE_DEVICES not set, all workers will use default device")
        
        try:
            from gunicorn.app.base import BaseApplication
            
            class StandaloneApplication(BaseApplication):
                """自定义 Gunicorn Application"""
                
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    config_file = "gunicorn_config.py"
                    if os.path.exists(config_file):
                        # 动态导入配置文件以正确注册钩子函数
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("gunicorn_config", config_file)
                        config_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(config_module)
                        
                        hook_names = ['post_fork', 'pre_fork', 'post_worker_init', 
                                     'worker_int', 'worker_abort', 'pre_exec',
                                     'when_ready', 'pre_request', 'post_request',
                                     'child_exit', 'worker_exit', 'nworkers_changed', 'on_exit']
                        
                        for key in dir(config_module):
                            if key.startswith('_'):
                                continue
                            value = getattr(config_module, key)
                            if key in self.cfg.settings:
                                self.cfg.set(key.lower(), value)
                            elif callable(value) and key in hook_names:
                                self.cfg.set(key, value)
                    
                    # 命令行参数覆盖配置文件
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            # Gunicorn 配置选项
            options = {
                'bind': f'{args.host}:{args.port}',
                'workers': args.workers,
                'worker_class': 'uvicorn.workers.UvicornWorker',
                'worker_connections': 1,  # 单连接
                'threads': 1,  # 单线程
                'timeout': 300,
                'keepalive': 5,
                'loglevel': args.log_level,
                'accesslog': '-',
                'errorlog': '-',
            }

            StandaloneApplication(app, options).run()
            
        except ImportError:
            logger.error("Gunicorn is not installed. Please install it with: pip install gunicorn")
            logger.error("Or use single worker mode: python server.py --workers 1")
            raise
            
    else:
        # 单进程模式，支持 reload
        logger.info("Starting server in single-process mode with Uvicorn")
        if args.reload:
            logger.warning("Auto-reload is enabled (development mode)")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
