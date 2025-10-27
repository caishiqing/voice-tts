"""
音色管理模块

提供音色信息的存储、查询、更新和删除功能
支持 PostgreSQL 数据库和音频二进制数据存储
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


class Gender(str, Enum):
    """性别枚举"""
    MALE = "male"
    FEMALE = "female"


class Age(str, Enum):
    """年龄段枚举"""
    CHILDHOOD = "童年"
    ADOLESCENCE = "少年"
    YOUTH = "青年"
    ADULTHOOD = "成年"
    OLD_AGE = "老年"


class Emotion(str, Enum):
    """情绪枚举"""
    NORMAL = "normal"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


@dataclass
class Role:
    """角色信息数据类"""
    voice_id: str
    name: str
    gender: Gender
    age: Age
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['gender'] = self.gender.value
        data['age'] = self.age.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """从字典创建实例"""
        # 处理 datetime 对象转字符串
        created_at = data.get('created_at')
        if created_at and hasattr(created_at, 'isoformat'):
            created_at = created_at.isoformat()

        updated_at = data.get('updated_at')
        if updated_at and hasattr(updated_at, 'isoformat'):
            updated_at = updated_at.isoformat()

        return cls(
            voice_id=data['voice_id'],
            name=data['name'],
            gender=Gender(data['gender']),
            age=Age(data['age']),
            description=data.get('description', ''),
            created_at=created_at,
            updated_at=updated_at
        )


@dataclass
class VoiceInfo:
    """音色信息数据类"""
    voice_id: str
    emotion: Emotion
    audio_data: Optional[bytes] = None
    audio_filename: str = ""
    file_hash: str = ""
    sample_rate: int = 16000
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['emotion'] = self.emotion.value
        # 不包含 audio_data（太大）
        if 'audio_data' in data:
            data['audio_data'] = f"<binary data: {len(self.audio_data) if self.audio_data else 0} bytes>"
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceInfo':
        """从字典创建实例"""
        # 处理 datetime 对象转字符串
        created_at = data.get('created_at')
        if created_at and hasattr(created_at, 'isoformat'):
            created_at = created_at.isoformat()

        updated_at = data.get('updated_at')
        if updated_at and hasattr(updated_at, 'isoformat'):
            updated_at = updated_at.isoformat()

        # 处理 audio_data（可能是 memoryview 或 bytes）
        audio_data = data.get('audio_data')
        if audio_data is not None and not isinstance(audio_data, bytes):
            audio_data = bytes(audio_data)

        return cls(
            id=data.get('id'),
            voice_id=data['voice_id'],
            emotion=Emotion(data['emotion']),
            audio_data=audio_data,
            audio_filename=data.get('audio_filename', ''),
            file_hash=data.get('file_hash', ''),
            sample_rate=data.get('sample_rate', 16000),
            created_at=created_at,
            updated_at=updated_at
        )


class VoiceManager:
    """音色管理器（PostgreSQL 版本）"""

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "voice_tts",
        db_user: str = "postgres",
        db_password: str = "postgres",
        pool_minconn: int = 1,
        pool_maxconn: int = 10
    ):
        """
        初始化音色管理器

        Args:
            db_host: 数据库主机
            db_port: 数据库端口
            db_name: 数据库名称
            db_user: 数据库用户
            db_password: 数据库密码
            pool_minconn: 连接池最小连接数
            pool_maxconn: 连接池最大连接数
        """
        # 创建连接池
        try:
            self.pool = SimpleConnectionPool(
                minconn=pool_minconn,
                maxconn=pool_maxconn,
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password
            )
            logger.info(f"Connected to PostgreSQL: {db_host}:{db_port}/{db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

        # 初始化数据库
        self._init_database()

        logger.info(f"VoiceManager initialized with PostgreSQL")

    def _get_connection(self):
        """获取数据库连接"""
        return self.pool.getconn()

    def _release_connection(self, conn):
        """释放数据库连接"""
        self.pool.putconn(conn)

    def _init_database(self):
        """初始化数据库表（PostgreSQL）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 创建角色表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    voice_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT DEFAULT '',
                    gender VARCHAR(50) NOT NULL,
                    age VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建音色表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voices (
                    id SERIAL PRIMARY KEY,
                    voice_id VARCHAR(255) NOT NULL,
                    emotion VARCHAR(50) NOT NULL,
                    audio_data BYTEA NOT NULL,
                    audio_filename VARCHAR(255) DEFAULT '',
                    file_hash VARCHAR(64) DEFAULT '',
                    sample_rate INTEGER DEFAULT 16000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (voice_id) REFERENCES roles(voice_id) ON DELETE CASCADE
                )
            """)

            # 创建索引以提高查询性能
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_id ON voices(voice_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_emotion ON voices(emotion)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_role_gender ON roles(gender)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_role_age ON roles(age)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON voices(file_hash)
            """)

            conn.commit()
            logger.debug("PostgreSQL database initialized")
        finally:
            self._release_connection(conn)

    # ============ Role Management Methods ============

    def add_role(self, role: Role) -> bool:
        """
        添加角色信息

        Args:
            role: 角色信息对象

        Returns:
            是否添加成功

        Raises:
            ValueError: 如果voice_id已存在
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO roles (voice_id, name, description, gender, age)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    role.voice_id,
                    role.name,
                    role.description,
                    role.gender.value,
                    role.age.value
                ))
                conn.commit()

                logger.info(f"Added role: voice_id={role.voice_id}, name={role.name}")
                return True

            except psycopg2.IntegrityError as e:
                conn.rollback()
                logger.error(f"Failed to add role: voice_id {role.voice_id} already exists")
                raise ValueError(f"Role with voice_id {role.voice_id} already exists") from e
        finally:
            self._release_connection(conn)

    def get_role(self, voice_id: str) -> Optional[Role]:
        """
        根据voice_id获取角色信息

        Args:
            voice_id: 角色voice_id

        Returns:
            角色信息对象，如果不存在则返回None
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM roles WHERE voice_id = %s", (voice_id,))
            row = cursor.fetchone()

            if row:
                return Role.from_dict(dict(row))
            return None
        finally:
            self._release_connection(conn)

    def get_all_roles(self) -> List[Role]:
        """
        获取所有角色信息

        Returns:
            所有角色信息列表
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM roles ORDER BY created_at DESC")
            rows = cursor.fetchall()

            return [Role.from_dict(dict(row)) for row in rows]
        finally:
            self._release_connection(conn)

    def search_roles(
        self,
        gender: Optional[Gender] = None,
        age: Optional[Age] = None,
        name_keyword: Optional[str] = None
    ) -> List[Role]:
        """
        根据条件搜索角色

        Args:
            gender: 性别（可选）
            age: 年龄段（可选）
            name_keyword: 名称关键词（可选，模糊匹配）

        Returns:
            符合条件的角色信息列表
        """
        conditions = []
        params = []

        if gender is not None:
            conditions.append("gender = %s")
            params.append(gender.value)

        if age is not None:
            conditions.append("age = %s")
            params.append(age.value)

        if name_keyword is not None:
            conditions.append("name LIKE %s")
            params.append(f"%{name_keyword}%")

        query = "SELECT * FROM roles"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [Role.from_dict(dict(row)) for row in rows]
        finally:
            self._release_connection(conn)

    def update_role(self, voice_id: str, role: Role) -> bool:
        """
        更新角色信息

        Args:
            voice_id: 要更新的角色voice_id
            role: 新的角色信息

        Returns:
            是否更新成功
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE roles
                SET name = %s,
                    description = %s,
                    gender = %s,
                    age = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE voice_id = %s
            """, (
                role.name,
                role.description,
                role.gender.value,
                role.age.value,
                voice_id
            ))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Updated role: voice_id={voice_id}, name={role.name}")
            else:
                logger.warning(f"Failed to update role: voice_id={voice_id} not found")

            return success
        finally:
            self._release_connection(conn)

    def delete_role(self, voice_id: str, delete_voices: bool = False) -> bool:
        """
        删除角色信息

        Args:
            voice_id: 要删除的角色voice_id
            delete_voices: 是否同时删除该角色的所有音色（默认False）

        Returns:
            是否删除成功

        Raises:
            ValueError: 如果角色有关联的音色但delete_voices=False
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 检查是否有关联的音色
            cursor.execute("SELECT COUNT(*) FROM voices WHERE voice_id = %s", (voice_id,))
            voice_count = cursor.fetchone()[0]

            if voice_count > 0 and not delete_voices:
                raise ValueError(
                    f"Role {voice_id} has {voice_count} associated voices. "
                    "Set delete_voices=True to delete them or remove them first."
                )

            # 如果需要删除音色
            if delete_voices and voice_count > 0:
                cursor.execute("SELECT id FROM voices WHERE voice_id = %s", (voice_id,))
                voice_ids = [row[0] for row in cursor.fetchall()]
                for vid in voice_ids:
                    self.delete_voice(vid, delete_audio=False)

            # 删除角色
            cursor.execute("DELETE FROM roles WHERE voice_id = %s", (voice_id,))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Deleted role: voice_id={voice_id}, deleted {voice_count} voices")
            else:
                logger.warning(f"Failed to delete role: voice_id={voice_id} not found")

            return success
        finally:
            self._release_connection(conn)

    def get_role_count(self) -> int:
        """
        获取角色总数

        Returns:
            角色总数
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM roles")
            return cursor.fetchone()[0]
        finally:
            self._release_connection(conn)

    # ============ Voice Management Methods ============

    def add_voice(self, voice_info: VoiceInfo) -> int:
        """
        添加音色信息

        Args:
            voice_info: 音色信息对象

        Returns:
            新添加记录的ID

        Raises:
            ValueError: 如果对应的角色不存在或音频数据缺失
        """
        # 验证角色是否存在
        role = self.get_role(voice_info.voice_id)
        if role is None:
            raise ValueError(f"Role with voice_id {voice_info.voice_id} does not exist")

        # 验证音频数据
        if not voice_info.audio_data:
            raise ValueError("audio_data is required")

        # 计算文件hash（如果没有提供）
        if not voice_info.file_hash:
            import hashlib
            voice_info.file_hash = hashlib.sha256(voice_info.audio_data).hexdigest()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO voices (voice_id, emotion, audio_data, audio_filename, file_hash, sample_rate)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                voice_info.voice_id,
                voice_info.emotion.value,
                psycopg2.Binary(voice_info.audio_data),
                voice_info.audio_filename,
                voice_info.file_hash,
                voice_info.sample_rate
            ))
            voice_id = cursor.fetchone()[0]
            conn.commit()

            logger.info(f"Added voice: id={voice_id}, voice_id={voice_info.voice_id}, "
                        f"emotion={voice_info.emotion.value}, audio_size={len(voice_info.audio_data)} bytes")

            return voice_id
        finally:
            self._release_connection(conn)

    def get_voice(self, id: int, include_audio_data: bool = False) -> Optional[VoiceInfo]:
        """
        根据ID获取音色信息

        Args:
            id: 音色记录ID
            include_audio_data: 是否包含音频二进制数据（默认False，因为数据可能很大）

        Returns:
            音色信息对象，如果不存在则返回None
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            if include_audio_data:
                cursor.execute("SELECT * FROM voices WHERE id = %s", (id,))
            else:
                cursor.execute("""
                    SELECT id, voice_id, emotion, audio_filename, 
                           file_hash, sample_rate, created_at, updated_at 
                    FROM voices WHERE id = %s
                """, (id,))
            row = cursor.fetchone()

            if row:
                return VoiceInfo.from_dict(dict(row))
            return None
        finally:
            self._release_connection(conn)

    def get_audio_data(self, id: int) -> Optional[bytes]:
        """
        获取指定ID的音频二进制数据

        Args:
            id: 音色记录ID

        Returns:
            音频二进制数据，如果不存在则返回None
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT audio_data FROM voices WHERE id = %s", (id,))
            row = cursor.fetchone()

            if row and row[0]:
                return bytes(row[0])
            return None
        finally:
            self._release_connection(conn)

    def get_audio_data_by_voice_id(self, voice_id: str, emotion: Emotion) -> Optional[tuple]:
        """
        根据voice_id和情绪获取音频数据

        Args:
            voice_id: 角色ID
            emotion: 情绪

        Returns:
            (audio_data, sample_rate) 元组，如果不存在则返回None
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT audio_data, sample_rate 
                FROM voices 
                WHERE voice_id = %s AND emotion = %s 
                LIMIT 1
            """, (voice_id, emotion.value))
            row = cursor.fetchone()

            if row and row[0]:
                return (bytes(row[0]), row[1])
            return None
        finally:
            self._release_connection(conn)

    def get_voices_by_voice_id(self, voice_id: str) -> List[VoiceInfo]:
        """
        根据voice_id获取所有音色信息
        一个voice_id可能对应多个不同情绪或年龄的音色

        Args:
            voice_id: 说话人角色标识

        Returns:
            音色信息列表
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM voices WHERE voice_id = %s", (voice_id,))
            rows = cursor.fetchall()

            return [VoiceInfo.from_dict(dict(row)) for row in rows]
        finally:
            self._release_connection(conn)

    def search_voices(
        self,
        voice_id: Optional[str] = None,
        emotion: Optional[Emotion] = None,
        gender: Optional[Gender] = None,
        age: Optional[Age] = None
    ) -> List[VoiceInfo]:
        """
        根据条件搜索音色
        支持通过角色属性（gender, age）进行搜索

        Args:
            voice_id: 说话人角色标识（可选）
            emotion: 情绪（可选）
            gender: 性别（可选，通过关联roles表查询）
            age: 年龄段（可选，通过关联roles表查询）

        Returns:
            符合条件的音色信息列表
        """
        conditions = []
        params = []

        if voice_id is not None:
            conditions.append("v.voice_id = %s")
            params.append(voice_id)

        if emotion is not None:
            conditions.append("v.emotion = %s")
            params.append(emotion.value)

        if gender is not None:
            conditions.append("r.gender = %s")
            params.append(gender.value)

        if age is not None:
            conditions.append("r.age = %s")
            params.append(age.value)

        query = """
            SELECT v.* FROM voices v
            JOIN roles r ON v.voice_id = r.voice_id
        """
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY v.created_at DESC"

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [VoiceInfo.from_dict(dict(row)) for row in rows]
        finally:
            self._release_connection(conn)

    def get_all_voices(self) -> List[VoiceInfo]:
        """
        获取所有音色信息

        Returns:
            所有音色信息列表
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM voices ORDER BY created_at DESC")
            rows = cursor.fetchall()

            return [VoiceInfo.from_dict(dict(row)) for row in rows]
        finally:
            self._release_connection(conn)

    def update_voice(self, id: int, voice_info: VoiceInfo) -> bool:
        """
        更新音色信息

        Args:
            id: 要更新的音色记录ID
            voice_info: 新的音色信息

        Returns:
            是否更新成功

        Raises:
            ValueError: 如果对应的角色不存在或音频数据缺失
        """
        # 验证角色是否存在
        role = self.get_role(voice_info.voice_id)
        if role is None:
            raise ValueError(f"Role with voice_id {voice_info.voice_id} does not exist")

        # 验证音频数据
        if not voice_info.audio_data:
            raise ValueError("audio_data is required")

        # 计算文件hash（如果没有提供）
        if not voice_info.file_hash:
            import hashlib
            voice_info.file_hash = hashlib.sha256(voice_info.audio_data).hexdigest()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE voices
                SET voice_id = %s,
                    emotion = %s,
                    audio_data = %s,
                    audio_filename = %s,
                    file_hash = %s,
                    sample_rate = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (
                voice_info.voice_id,
                voice_info.emotion.value,
                psycopg2.Binary(voice_info.audio_data),
                voice_info.audio_filename,
                voice_info.file_hash,
                voice_info.sample_rate,
                id
            ))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Updated voice: id={id}, voice_id={voice_info.voice_id}, audio_size={len(voice_info.audio_data)} bytes")
            else:
                logger.warning(f"Failed to update voice: id={id} not found")

            return success
        finally:
            self._release_connection(conn)

    def delete_voice(self, id: int, delete_audio: bool = False) -> bool:
        """
        删除音色信息（音频数据随记录一起删除）

        Args:
            id: 要删除的音色记录ID
            delete_audio: 保留参数用于兼容性（音频数据总是随记录删除）

        Returns:
            是否删除成功
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM voices WHERE id = %s", (id,))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Deleted voice: id={id}")
            else:
                logger.warning(f"Failed to delete voice: id={id} not found")

            return success
        finally:
            self._release_connection(conn)

    def get_voice_count(self) -> int:
        """
        获取音色总数

        Returns:
            音色总数
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM voices")
            return cursor.fetchone()[0]
        finally:
            self._release_connection(conn)

    def get_unique_voice_ids(self) -> List[str]:
        """
        获取所有唯一的voice_id

        Returns:
            voice_id列表
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT voice_id FROM roles ORDER BY voice_id")
            return [row[0] for row in cursor.fetchall()]
        finally:
            self._release_connection(conn)
