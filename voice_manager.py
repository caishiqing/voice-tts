"""
音色管理模块

提供音色信息的存储、查询、更新和删除功能
使用SQLite作为存储后端
"""

import sqlite3
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


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
        return cls(
            voice_id=data['voice_id'],
            name=data['name'],
            gender=Gender(data['gender']),
            age=Age(data['age']),
            description=data.get('description', ''),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


@dataclass
class VoiceInfo:
    """音色信息数据类"""
    voice_id: str
    emotion: Emotion
    audio_path: str
    id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['emotion'] = self.emotion.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceInfo':
        """从字典创建实例"""
        return cls(
            id=data.get('id'),
            voice_id=data['voice_id'],
            emotion=Emotion(data['emotion']),
            audio_path=data['audio_path'],
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


class VoiceManager:
    """音色管理器"""

    def __init__(self, db_dir: str = "./data/db", voices_dir: str = "./data/voices"):
        """
        初始化音色管理器

        Args:
            db_dir: 数据库文件目录
            voices_dir: 音色音频文件目录
        """
        self.db_dir = Path(db_dir)
        self.voices_dir = Path(voices_dir)
        self.db_path = self.db_dir / "voices.db"

        # 创建必要的目录
        self._ensure_directories()

        # 初始化数据库
        self._init_database()

        logger.info(f"VoiceManager initialized with db_path={self.db_path}, voices_dir={self.voices_dir}")

    def _ensure_directories(self):
        """确保必要的目录存在"""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directories: {self.db_dir}, {self.voices_dir}")

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 允许通过列名访问
        return conn

    def _init_database(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 创建角色表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    voice_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    gender TEXT NOT NULL,
                    age TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建音色表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    voice_id TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (voice_id) REFERENCES roles(voice_id)
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

            conn.commit()
            logger.debug("Database initialized")

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
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO roles (voice_id, name, description, gender, age)
                    VALUES (?, ?, ?, ?, ?)
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

            except sqlite3.IntegrityError as e:
                logger.error(f"Failed to add role: voice_id {role.voice_id} already exists")
                raise ValueError(f"Role with voice_id {role.voice_id} already exists") from e

    def get_role(self, voice_id: str) -> Optional[Role]:
        """
        根据voice_id获取角色信息

        Args:
            voice_id: 角色voice_id

        Returns:
            角色信息对象，如果不存在则返回None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM roles WHERE voice_id = ?", (voice_id,))
            row = cursor.fetchone()

            if row:
                return Role.from_dict(dict(row))
            return None

    def get_all_roles(self) -> List[Role]:
        """
        获取所有角色信息

        Returns:
            所有角色信息列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM roles ORDER BY created_at DESC")
            rows = cursor.fetchall()

            return [Role.from_dict(dict(row)) for row in rows]

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
            conditions.append("gender = ?")
            params.append(gender.value)

        if age is not None:
            conditions.append("age = ?")
            params.append(age.value)

        if name_keyword is not None:
            conditions.append("name LIKE ?")
            params.append(f"%{name_keyword}%")

        query = "SELECT * FROM roles"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [Role.from_dict(dict(row)) for row in rows]

    def update_role(self, voice_id: str, role: Role) -> bool:
        """
        更新角色信息

        Args:
            voice_id: 要更新的角色voice_id
            role: 新的角色信息

        Returns:
            是否更新成功
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE roles
                SET name = ?,
                    description = ?,
                    gender = ?,
                    age = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE voice_id = ?
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
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 检查是否有关联的音色
            cursor.execute("SELECT COUNT(*) FROM voices WHERE voice_id = ?", (voice_id,))
            voice_count = cursor.fetchone()[0]

            if voice_count > 0 and not delete_voices:
                raise ValueError(
                    f"Role {voice_id} has {voice_count} associated voices. "
                    "Set delete_voices=True to delete them or remove them first."
                )

            # 如果需要删除音色
            if delete_voices and voice_count > 0:
                cursor.execute("SELECT id FROM voices WHERE voice_id = ?", (voice_id,))
                voice_ids = [row[0] for row in cursor.fetchall()]
                for vid in voice_ids:
                    self.delete_voice(vid, delete_audio=True)

            # 删除角色
            cursor.execute("DELETE FROM roles WHERE voice_id = ?", (voice_id,))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Deleted role: voice_id={voice_id}, deleted {voice_count} voices")
            else:
                logger.warning(f"Failed to delete role: voice_id={voice_id} not found")

            return success

    def get_role_count(self) -> int:
        """
        获取角色总数

        Returns:
            角色总数
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM roles")
            return cursor.fetchone()[0]

    # ============ Voice Management Methods ============

    def add_voice(self, voice_info: VoiceInfo) -> int:
        """
        添加音色信息

        Args:
            voice_info: 音色信息对象

        Returns:
            新添加记录的ID

        Raises:
            ValueError: 如果音频文件不存在或对应的角色不存在
        """
        # 验证角色是否存在
        role = self.get_role(voice_info.voice_id)
        if role is None:
            raise ValueError(f"Role with voice_id {voice_info.voice_id} does not exist")

        # 验证音频文件是否存在
        if not os.path.exists(voice_info.audio_path):
            raise ValueError(f"Audio file not found: {voice_info.audio_path}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO voices (voice_id, emotion, audio_path)
                VALUES (?, ?, ?)
            """, (
                voice_info.voice_id,
                voice_info.emotion.value,
                voice_info.audio_path
            ))
            conn.commit()
            voice_id = cursor.lastrowid

            logger.info(f"Added voice: id={voice_id}, voice_id={voice_info.voice_id}, "
                        f"emotion={voice_info.emotion.value}")

            return voice_id

    def get_voice(self, id: int) -> Optional[VoiceInfo]:
        """
        根据ID获取音色信息

        Args:
            id: 音色记录ID

        Returns:
            音色信息对象，如果不存在则返回None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM voices WHERE id = ?", (id,))
            row = cursor.fetchone()

            if row:
                return VoiceInfo.from_dict(dict(row))
            return None

    def get_voices_by_voice_id(self, voice_id: str) -> List[VoiceInfo]:
        """
        根据voice_id获取所有音色信息
        一个voice_id可能对应多个不同情绪或年龄的音色

        Args:
            voice_id: 说话人角色标识

        Returns:
            音色信息列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM voices WHERE voice_id = ?", (voice_id,))
            rows = cursor.fetchall()

            return [VoiceInfo.from_dict(dict(row)) for row in rows]

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
            conditions.append("v.voice_id = ?")
            params.append(voice_id)

        if emotion is not None:
            conditions.append("v.emotion = ?")
            params.append(emotion.value)

        if gender is not None:
            conditions.append("r.gender = ?")
            params.append(gender.value)

        if age is not None:
            conditions.append("r.age = ?")
            params.append(age.value)

        query = """
            SELECT v.* FROM voices v
            JOIN roles r ON v.voice_id = r.voice_id
        """
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY v.created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [VoiceInfo.from_dict(dict(row)) for row in rows]

    def get_all_voices(self) -> List[VoiceInfo]:
        """
        获取所有音色信息

        Returns:
            所有音色信息列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM voices ORDER BY created_at DESC")
            rows = cursor.fetchall()

            return [VoiceInfo.from_dict(dict(row)) for row in rows]

    def update_voice(self, id: int, voice_info: VoiceInfo) -> bool:
        """
        更新音色信息

        Args:
            id: 要更新的音色记录ID
            voice_info: 新的音色信息

        Returns:
            是否更新成功

        Raises:
            ValueError: 如果音频文件不存在或对应的角色不存在
        """
        # 验证角色是否存在
        role = self.get_role(voice_info.voice_id)
        if role is None:
            raise ValueError(f"Role with voice_id {voice_info.voice_id} does not exist")

        # 验证音频文件是否存在
        if not os.path.exists(voice_info.audio_path):
            raise ValueError(f"Audio file not found: {voice_info.audio_path}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE voices
                SET voice_id = ?,
                    emotion = ?,
                    audio_path = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                voice_info.voice_id,
                voice_info.emotion.value,
                voice_info.audio_path,
                id
            ))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Updated voice: id={id}, voice_id={voice_info.voice_id}")
            else:
                logger.warning(f"Failed to update voice: id={id} not found")

            return success

    def delete_voice(self, id: int, delete_audio: bool = False) -> bool:
        """
        删除音色信息

        Args:
            id: 要删除的音色记录ID
            delete_audio: 是否同时删除音频文件

        Returns:
            是否删除成功
        """
        # 如果需要删除音频文件，先获取音频路径
        audio_path = None
        if delete_audio:
            voice = self.get_voice(id)
            if voice:
                audio_path = voice.audio_path

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM voices WHERE id = ?", (id,))
            conn.commit()
            success = cursor.rowcount > 0

            if success:
                logger.info(f"Deleted voice: id={id}")

                # 删除音频文件
                if delete_audio and audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                        logger.info(f"Deleted audio file: {audio_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete audio file {audio_path}: {e}")
            else:
                logger.warning(f"Failed to delete voice: id={id} not found")

            return success

    def get_voice_count(self) -> int:
        """
        获取音色总数

        Returns:
            音色总数
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM voices")
            return cursor.fetchone()[0]

    def get_unique_voice_ids(self) -> List[str]:
        """
        获取所有唯一的voice_id

        Returns:
            voice_id列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT voice_id FROM roles ORDER BY voice_id")
            return [row[0] for row in cursor.fetchall()]
