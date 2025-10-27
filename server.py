from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
from loguru import logger
import uvicorn
import os
import tempfile
import base64
from pathlib import Path
import uuid
import shutil

from indextts.infer_v2 import IndexTTS2
from voice_manager import VoiceManager, VoiceInfo, Role, Gender, Age, Emotion

# 全局变量存储模型和管理器
tts_model = None
voice_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期"""
    global tts_model, voice_manager
    # Startup
    try:
        # 初始化音色管理器（PostgreSQL）
        logger.info("Initializing VoiceManager with PostgreSQL...")
        voice_manager = VoiceManager(
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "voice_tts"),
            db_user=os.getenv("DB_USER", "postgres"),
            db_password=os.getenv("DB_PASSWORD", "postgres")
        )
        logger.success("VoiceManager initialized successfully!")

        logger.info("Loading IndexTTS2 model...")
        tts_model = IndexTTS2(
            cfg_path="models/IndexTTS/config.yaml",
            model_dir="models/IndexTTS",
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False
        )
        logger.success("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


app = FastAPI(title="IndexTTS API Server", lifespan=lifespan)

# 添加CORS中间件（如需要可配置允许的源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议配置具体的允许源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)


class TTSRequest(BaseModel):
    """TTS推理请求模型"""
    text: str = Field(..., description="要合成的文本")
    voice_id: str = Field(..., description="角色voice_id")
    emotion: Emotion = Field(default=Emotion.NORMAL, description="情绪")


class TTSResponse(BaseModel):
    """TTS推理响应模型"""
    audio_hex: str  # hex编码的音频数据
    audio_length: float  # 音频长度（秒）
    inference_time: float  # 推理时间（秒）
    rtf: float  # Real-Time Factor
    voice_id: str  # 使用的角色ID
    emotion: str  # 使用的情绪


# ============ Role Management Models ============

class RoleCreateRequest(BaseModel):
    """创建角色请求模型"""
    voice_id: str = Field(..., description="角色voice_id标识")
    name: str = Field(..., description="角色名称")
    description: str = Field(default="", description="角色描述")
    gender: Gender = Field(..., description="性别")
    age: Age = Field(..., description="年龄段")


class RoleUpdateRequest(BaseModel):
    """更新角色请求模型"""
    name: Optional[str] = Field(None, description="角色名称")
    description: Optional[str] = Field(None, description="角色描述")
    gender: Optional[Gender] = Field(None, description="性别")
    age: Optional[Age] = Field(None, description="年龄段")


class RoleResponse(BaseModel):
    """角色响应模型"""
    voice_id: str
    name: str
    description: str
    gender: str
    age: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class RoleListResponse(BaseModel):
    """角色列表响应模型"""
    total: int
    roles: List[RoleResponse]


# ============ Voice Management Models ============

class VoiceResponse(BaseModel):
    """音色响应模型"""
    id: int
    voice_id: str
    emotion: str
    audio_filename: str = ""
    file_hash: str = ""
    sample_rate: int = 16000
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class VoiceListResponse(BaseModel):
    """音色列表响应模型"""
    total: int
    voices: List[VoiceResponse]


@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "running",
        "model_loaded": tts_model is not None,
        "voice_manager_loaded": voice_manager is not None,
        "service": "IndexTTS API Server"
    }


# ============ Role Management APIs ============

@app.post("/roles", response_model=RoleResponse, status_code=201)
async def create_role(request: RoleCreateRequest):
    """
    创建新的角色

    Args:
        request: 角色创建请求

    Returns:
        创建的角色信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        role = Role(
            voice_id=request.voice_id,
            name=request.name,
            gender=request.gender,
            age=request.age,
            description=request.description
        )

        voice_manager.add_role(role)

        # 获取完整信息返回
        created_role = voice_manager.get_role(request.voice_id)

        logger.success(f"Created role: voice_id={request.voice_id}, name={request.name}")

        return RoleResponse(**created_role.to_dict())

    except ValueError as e:
        logger.error(f"Failed to create role: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create role: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create role: {str(e)}")


@app.get("/roles", response_model=RoleListResponse)
async def list_roles(
    gender: Optional[str] = Query(None, description="按性别筛选"),
    age: Optional[str] = Query(None, description="按年龄段筛选"),
    name: Optional[str] = Query(None, description="按名称关键词筛选")
):
    """
    获取角色列表（支持条件筛选）

    Args:
        gender: 可选，按性别筛选
        age: 可选，按年龄段筛选
        name: 可选，按名称关键词筛选

    Returns:
        角色列表
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 转换枚举值
        gender_enum = Gender(gender) if gender else None
        age_enum = Age(age) if age else None

        # 搜索角色
        roles = voice_manager.search_roles(
            gender=gender_enum,
            age=age_enum,
            name_keyword=name
        )

        role_responses = [RoleResponse(**r.to_dict()) for r in roles]

        logger.info(f"Retrieved {len(role_responses)} roles with filters: "
                    f"gender={gender}, age={age}, name={name}")

        return RoleListResponse(
            total=len(role_responses),
            roles=role_responses
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to list roles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list roles: {str(e)}")


@app.get("/roles/{voice_id}", response_model=RoleResponse)
async def get_role(voice_id: str):
    """
    获取指定voice_id的角色信息

    Args:
        voice_id: 角色voice_id

    Returns:
        角色信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        role = voice_manager.get_role(voice_id)

        if role is None:
            raise HTTPException(status_code=404, detail=f"Role with voice_id {voice_id} not found")

        logger.info(f"Retrieved role: voice_id={voice_id}")

        return RoleResponse(**role.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get role: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get role: {str(e)}")


@app.put("/roles/{voice_id}", response_model=RoleResponse)
async def update_role(voice_id: str, request: RoleUpdateRequest):
    """
    更新角色信息

    Args:
        voice_id: 角色voice_id
        request: 更新请求

    Returns:
        更新后的角色信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 获取现有角色信息
        existing_role = voice_manager.get_role(voice_id)
        if existing_role is None:
            raise HTTPException(status_code=404, detail=f"Role with voice_id {voice_id} not found")

        # 准备更新的数据
        updated_role = Role(
            voice_id=voice_id,
            name=request.name if request.name is not None else existing_role.name,
            gender=request.gender if request.gender is not None else existing_role.gender,
            age=request.age if request.age is not None else existing_role.age,
            description=request.description if request.description is not None else existing_role.description
        )

        # 更新数据库
        success = voice_manager.update_role(voice_id, updated_role)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update role in database")

        # 获取更新后的信息
        updated_role_data = voice_manager.get_role(voice_id)

        logger.success(f"Updated role: voice_id={voice_id}")

        return RoleResponse(**updated_role_data.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update role: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update role: {str(e)}")


@app.delete("/roles/{voice_id}")
async def delete_role(
    voice_id: str,
    delete_voices: bool = Query(False, description="是否同时删除该角色的所有音色")
):
    """
    删除角色

    Args:
        voice_id: 角色voice_id
        delete_voices: 是否同时删除该角色的所有音色（默认False）

    Returns:
        删除结果
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 检查角色是否存在
        role = voice_manager.get_role(voice_id)
        if role is None:
            raise HTTPException(status_code=404, detail=f"Role with voice_id {voice_id} not found")

        # 删除角色
        success = voice_manager.delete_role(voice_id, delete_voices=delete_voices)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete role")

        logger.success(f"Deleted role: voice_id={voice_id}, delete_voices={delete_voices}")

        return {
            "success": True,
            "message": f"Role {voice_id} deleted successfully",
            "voices_deleted": delete_voices
        }

    except ValueError as e:
        # 角色有关联的音色但不允许删除
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete role: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete role: {str(e)}")


@app.get("/roles/{voice_id}/voices", response_model=VoiceListResponse)
async def get_role_voices(voice_id: str):
    """
    获取指定角色的所有音色

    Args:
        voice_id: 角色voice_id

    Returns:
        该角色的所有音色列表
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 验证角色存在
        role = voice_manager.get_role(voice_id)
        if role is None:
            raise HTTPException(status_code=404, detail=f"Role with voice_id {voice_id} not found")

        # 获取该角色的所有音色
        voices = voice_manager.get_voices_by_voice_id(voice_id)

        voice_responses = [VoiceResponse(**v.to_dict()) for v in voices]

        logger.info(f"Retrieved {len(voice_responses)} voices for role {voice_id}")

        return VoiceListResponse(
            total=len(voice_responses),
            voices=voice_responses
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get role voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@app.get("/roles/stats/summary")
async def get_role_stats():
    """
    获取角色统计信息

    Returns:
        统计信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        total_roles = voice_manager.get_role_count()
        total_voices = voice_manager.get_voice_count()

        return {
            "total_roles": total_roles,
            "total_voices": total_voices
        }

    except Exception as e:
        logger.error(f"Failed to get role stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============ Voice Management APIs ============

@app.post("/voices", response_model=VoiceResponse, status_code=201)
async def create_voice(
    voice_id: str = Form(...),
    emotion: str = Form(...),
    audio_file: UploadFile = File(...),
    sample_rate: int = Form(16000)
):
    """
    创建新的音色（需要角色已存在）
    音频数据直接存储在数据库中

    Args:
        voice_id: 角色voice_id标识
        emotion: 情绪 (normal/happy/angry/sad/fearful/disgusted/surprised)
        audio_file: 音色音频文件
        sample_rate: 采样率（默认16000）

    Returns:
        创建的音色信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 验证角色是否存在
        role = voice_manager.get_role(voice_id)
        if role is None:
            raise HTTPException(
                status_code=404,
                detail=f"Role with voice_id {voice_id} not found. Please create the role first."
            )

        # 验证枚举值
        try:
            emotion_enum = Emotion(emotion)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid emotion value: {str(e)}")

        # 读取音频文件的二进制数据
        audio_data = await audio_file.read()

        logger.info(f"Received audio file: {audio_file.filename}, size={len(audio_data)} bytes")

        # 创建音色信息（直接存储二进制数据）
        voice_info = VoiceInfo(
            voice_id=voice_id,
            emotion=emotion_enum,
            audio_data=audio_data,
            audio_filename=audio_file.filename,
            sample_rate=sample_rate
        )

        # 添加到数据库（音频数据会被存储为 BYTEA）
        new_id = voice_manager.add_voice(voice_info)

        # 获取完整信息返回（不包含音频数据以节省带宽）
        created_voice = voice_manager.get_voice(new_id, include_audio_data=False)

        logger.success(f"Created voice in database: id={new_id}, voice_id={voice_id}, emotion={emotion}, audio_size={len(audio_data)} bytes")

        return VoiceResponse(**created_voice.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create voice: {str(e)}")


@app.get("/voices", response_model=VoiceListResponse)
async def list_voices(
    voice_id: Optional[str] = Query(None, description="按voice_id筛选"),
    emotion: Optional[str] = Query(None, description="按情绪筛选"),
    gender: Optional[str] = Query(None, description="按性别筛选"),
    age: Optional[str] = Query(None, description="按年龄段筛选")
):
    """
    获取音色列表（支持条件筛选）
    支持通过角色属性（gender, age）筛选

    Args:
        voice_id: 可选，按voice_id筛选
        emotion: 可选，按情绪筛选
        gender: 可选，按性别筛选（通过关联roles表）
        age: 可选，按年龄段筛选（通过关联roles表）

    Returns:
        音色列表
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 转换枚举值
        emotion_enum = Emotion(emotion) if emotion else None
        gender_enum = Gender(gender) if gender else None
        age_enum = Age(age) if age else None

        # 搜索音色
        voices = voice_manager.search_voices(
            voice_id=voice_id,
            emotion=emotion_enum,
            gender=gender_enum,
            age=age_enum
        )

        voice_responses = [VoiceResponse(**v.to_dict()) for v in voices]

        logger.info(f"Retrieved {len(voice_responses)} voices with filters: "
                    f"voice_id={voice_id}, emotion={emotion}, gender={gender}, age={age}")

        return VoiceListResponse(
            total=len(voice_responses),
            voices=voice_responses
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to list voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@app.get("/voices/{voice_db_id}", response_model=VoiceResponse)
async def get_voice(voice_db_id: int):
    """
    获取指定ID的音色信息

    Args:
        voice_db_id: 音色数据库ID

    Returns:
        音色信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        voice = voice_manager.get_voice(voice_db_id)

        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice with id {voice_db_id} not found")

        logger.info(f"Retrieved voice: id={voice_db_id}")

        return VoiceResponse(**voice.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get voice: {str(e)}")


@app.get("/voices/{voice_db_id}/audio")
async def get_voice_audio(voice_db_id: int):
    """
    获取指定ID的音色音频数据

    Args:
        voice_db_id: 音色数据库ID

    Returns:
        音频二进制数据
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        voice = voice_manager.get_voice(voice_db_id, include_audio_data=False)

        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice with id {voice_db_id} not found")

        # 从数据库获取音频数据
        audio_data = voice_manager.get_audio_data(voice_db_id)

        if not audio_data:
            raise HTTPException(status_code=404, detail=f"Audio data not found for voice {voice_db_id}")

        logger.info(f"Serving audio data: {voice.audio_filename}, size={len(audio_data)} bytes")

        from fastapi.responses import Response
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{voice.audio_filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get audio data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio data: {str(e)}")


@app.put("/voices/{voice_db_id}", response_model=VoiceResponse)
async def update_voice(
    voice_db_id: int,
    voice_id: Optional[str] = Form(None),
    emotion: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    sample_rate: Optional[int] = Form(None)
):
    """
    更新音色信息

    Args:
        voice_db_id: 音色数据库ID
        voice_id: 可选，新的voice_id
        emotion: 可选，新的情绪
        audio_file: 可选，新的音频文件
        sample_rate: 可选，新的采样率

    Returns:
        更新后的音色信息
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 获取现有音色信息（包含数据以便可能需要保留）
        existing_voice = voice_manager.get_voice(voice_db_id, include_audio_data=True)
        if existing_voice is None:
            raise HTTPException(status_code=404, detail=f"Voice with id {voice_db_id} not found")

        # 准备更新的数据
        updated_voice_id = voice_id if voice_id is not None else existing_voice.voice_id
        updated_emotion = Emotion(emotion) if emotion is not None else existing_voice.emotion
        updated_audio_data = existing_voice.audio_data
        updated_audio_filename = existing_voice.audio_filename
        updated_sample_rate = sample_rate if sample_rate is not None else existing_voice.sample_rate

        # 如果修改了voice_id，验证新角色是否存在
        if voice_id is not None and voice_id != existing_voice.voice_id:
            role = voice_manager.get_role(updated_voice_id)
            if role is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Role with voice_id {updated_voice_id} not found"
                )

        # 如果提供了新的音频文件
        if audio_file is not None:
            updated_audio_data = await audio_file.read()
            updated_audio_filename = audio_file.filename
            logger.info(f"Updated audio data: {updated_audio_filename}, size={len(updated_audio_data)} bytes")

        # 创建更新的音色信息
        updated_voice_info = VoiceInfo(
            voice_id=updated_voice_id,
            emotion=updated_emotion,
            audio_data=updated_audio_data,
            audio_filename=updated_audio_filename,
            sample_rate=updated_sample_rate
        )

        # 更新数据库
        success = voice_manager.update_voice(voice_db_id, updated_voice_info)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update voice in database")

        # 获取更新后的信息
        updated_voice = voice_manager.get_voice(voice_db_id, include_audio_data=False)

        logger.success(f"Updated voice: id={voice_db_id}")

        return VoiceResponse(**updated_voice.to_dict())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to update voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update voice: {str(e)}")


@app.delete("/voices/{voice_db_id}")
async def delete_voice(voice_db_id: int):
    """
    删除音色（包括音频数据）

    Args:
        voice_db_id: 音色数据库ID

    Returns:
        删除结果
    """
    if voice_manager is None:
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 检查音色是否存在
        voice = voice_manager.get_voice(voice_db_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"Voice with id {voice_db_id} not found")

        # 删除音色（音频数据会随记录一起删除）
        success = voice_manager.delete_voice(voice_db_id, delete_audio=False)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete voice")

        logger.success(f"Deleted voice: id={voice_db_id}")

        return {
            "success": True,
            "message": f"Voice {voice_db_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")


# ============ TTS Inference APIs ============


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    TTS文本转语音接口

    通过角色ID和情绪自动选择对应的音色进行合成
    - spk_audio_prompt: 使用该角色的normal情绪音频
    - emo_audio_prompt: 使用该角色的指定情绪音频

    Args:
        request: TTS请求，包含文本、角色ID和情绪

    Returns:
        包含hex编码音频数据的响应
    """
    if tts_model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if voice_manager is None:
        logger.error("VoiceManager not initialized")
        raise HTTPException(status_code=503, detail="VoiceManager not initialized")

    try:
        # 验证角色是否存在
        role = voice_manager.get_role(request.voice_id)
        if role is None:
            raise HTTPException(
                status_code=404,
                detail=f"Role with voice_id '{request.voice_id}' not found"
            )

        # 获取该角色的normal情绪音频数据（直接从数据库）
        spk_audio_result = voice_manager.get_audio_data_by_voice_id(
            request.voice_id,
            Emotion.NORMAL
        )

        if not spk_audio_result:
            raise HTTPException(
                status_code=404,
                detail=f"No normal emotion voice found for role '{request.voice_id}'. Please add a normal voice first."
            )

        spk_audio_data, spk_sample_rate = spk_audio_result

        # 获取指定情绪的音频数据（如果需要）
        emo_audio_data = None
        emo_sample_rate = None
        if request.emotion != Emotion.NORMAL:
            emo_audio_result = voice_manager.get_audio_data_by_voice_id(
                request.voice_id,
                request.emotion
            )

            if not emo_audio_result:
                logger.warning(
                    f"No {request.emotion.value} emotion voice found for role '{request.voice_id}', "
                    f"will use normal emotion only"
                )
            else:
                emo_audio_data, emo_sample_rate = emo_audio_result

        logger.info(
            f"TTS request: voice_id={request.voice_id}, emotion={request.emotion.value}, "
            f"text='{request.text[:50]}...', spk_audio_size={len(spk_audio_data)} bytes, "
            f"emo_audio_size={len(emo_audio_data) if emo_audio_data else 0} bytes"
        )

        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name

        # 执行推理（直接传入音频二进制数据）
        result_path = tts_model.infer(
            spk_audio_prompt=spk_audio_data,  # 直接传入二进制数据
            text=request.text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_data if emo_audio_data else None,  # 直接传入二进制数据
            verbose=False
        )

        # 读取生成的音频文件并转换为hex
        with open(result_path, "rb") as f:
            audio_bytes = f.read()
            audio_hex = audio_bytes.hex()

        # 获取音频信息
        import wave
        with wave.open(result_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            audio_length = frames / float(rate)

        # 清理临时文件
        os.remove(output_path)

        logger.success(
            f"TTS completed: voice_id={request.voice_id}, emotion={request.emotion.value}, "
            f"audio_length={audio_length:.2f}s, size={len(audio_hex)//2} bytes"
        )

        return TTSResponse(
            audio_hex=audio_hex,
            audio_length=audio_length,
            inference_time=0.0,  # 可以从日志中提取
            rtf=0.0,  # 可以从日志中提取
            voice_id=request.voice_id,
            emotion=request.emotion.value
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS inference failed: {str(e)}")
        # 清理可能的临时文件
        try:
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="IndexTTS API Server",
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
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (only works with uvicorn module)"
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
