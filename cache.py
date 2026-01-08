"""
Redis 缓存模块 - 用于缓存音频数据

环境变量配置:
    REDIS_HOST: Redis 服务器地址，默认 localhost
    REDIS_PORT: Redis 端口，默认 6379
    REDIS_DB: Redis 数据库编号，默认 0
    REDIS_PASSWORD: Redis 密码，默认不设置
    REDIS_AUDIO_EXPIRE: 音频缓存过期时间（秒），默认 3600（1小时）
    REDIS_ENABLED: 是否启用缓存，默认 true

使用示例:
    from cache import AudioCache
    
    cache = AudioCache()
    
    # 获取音频（优先从缓存，未命中则下载并缓存）
    audio_data = cache.get_audio_from_url(url)
    
    # 手动设置缓存
    cache.set_audio(url, audio_bytes)
    
    # 手动获取缓存
    audio_bytes = cache.get_audio(url)
"""

import os
import hashlib
from typing import Optional
from loguru import logger

# Redis 客户端（延迟导入，避免未安装时报错）
_redis_client = None


def get_redis_config() -> dict:
    """从环境变量获取 Redis 配置"""
    return {
        'host': os.environ.get('REDIS_HOST', 'localhost'),
        'port': int(os.environ.get('REDIS_PORT', 6379)),
        'db': int(os.environ.get('REDIS_DB', 0)),
        'password': os.environ.get('REDIS_PASSWORD', None) or None,
        'decode_responses': False,  # 返回 bytes，用于存储二进制音频数据
        'socket_timeout': 5,
        'socket_connect_timeout': 5,
    }


def get_cache_expire() -> int:
    """获取缓存过期时间（秒）"""
    return int(os.environ.get('REDIS_AUDIO_EXPIRE', 3600))  # 默认 1 小时


def is_cache_enabled() -> bool:
    """检查是否启用缓存"""
    enabled = os.environ.get('REDIS_ENABLED', 'true').lower()
    return enabled in ('true', '1', 'yes', 'on')


def get_redis_client():
    """获取 Redis 客户端（单例模式）"""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    if not is_cache_enabled():
        logger.info("Redis cache is disabled (REDIS_ENABLED=false)")
        return None
    
    try:
        import redis
        
        config = get_redis_config()
        logger.info(f"Connecting to Redis: {config['host']}:{config['port']}/{config['db']}")
        
        _redis_client = redis.Redis(**config)
        
        # 测试连接
        _redis_client.ping()
        logger.success(f"Redis connected successfully")
        
        return _redis_client
        
    except ImportError:
        logger.warning("Redis package not installed. Cache disabled. Install with: pip install redis")
        return None
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. Cache disabled.")
        return None


def generate_cache_key(url: str) -> str:
    """
    根据 URL 生成缓存 key
    
    使用 MD5 哈希确保 key 长度固定，避免 URL 过长问题
    """
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return f"audio:url:{url_hash}"


class AudioCache:
    """音频缓存管理器"""
    
    def __init__(self):
        self._client = None
        self._initialized = False
    
    @property
    def client(self):
        """延迟初始化 Redis 客户端"""
        if not self._initialized:
            self._client = get_redis_client()
            self._initialized = True
        return self._client
    
    @property
    def enabled(self) -> bool:
        """检查缓存是否可用"""
        return self.client is not None
    
    def get_audio(self, url: str) -> Optional[bytes]:
        """
        从缓存获取音频数据
        
        Args:
            url: 音频 URL
            
        Returns:
            音频二进制数据，未命中返回 None
        """
        if not self.enabled:
            return None
        
        try:
            key = generate_cache_key(url)
            data = self.client.get(key)
            
            if data:
                logger.info(f"Cache HIT: {url[:80]}... ({len(data)} bytes)")
                return data
            else:
                logger.debug(f"Cache MISS: {url[:80]}...")
                return None
                
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def set_audio(self, url: str, audio_data: bytes, expire: Optional[int] = None) -> bool:
        """
        将音频数据存入缓存
        
        Args:
            url: 音频 URL
            audio_data: 音频二进制数据
            expire: 过期时间（秒），默认使用环境变量配置
            
        Returns:
            是否成功
        """
        if not self.enabled:
            return False
        
        if not audio_data:
            logger.warning("Attempted to cache empty audio data")
            return False
        
        try:
            key = generate_cache_key(url)
            expire_seconds = expire if expire is not None else get_cache_expire()
            
            self.client.setex(key, expire_seconds, audio_data)
            
            logger.info(
                f"Cache SET: {url[:80]}... "
                f"({len(audio_data)} bytes, expire={expire_seconds}s)"
            )
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def delete_audio(self, url: str) -> bool:
        """
        从缓存删除音频数据
        
        Args:
            url: 音频 URL
            
        Returns:
            是否成功
        """
        if not self.enabled:
            return False
        
        try:
            key = generate_cache_key(url)
            deleted = self.client.delete(key)
            
            if deleted:
                logger.info(f"Cache DELETE: {url[:80]}...")
            
            return deleted > 0
            
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    def get_audio_from_url(
        self, 
        url: str, 
        download_func: callable,
        expire: Optional[int] = None
    ) -> bytes:
        """
        获取音频数据（优先从缓存，未命中则下载并缓存）
        
        这是主要的对外接口，实现了 cache-aside 模式
        
        Args:
            url: 音频 URL
            download_func: 下载函数，接受 url 参数，返回 bytes
            expire: 过期时间（秒），默认使用环境变量配置
            
        Returns:
            音频二进制数据
        """
        # 1. 尝试从缓存获取
        cached_data = self.get_audio(url)
        if cached_data:
            return cached_data
        
        # 2. 缓存未命中，执行下载
        logger.info(f"Downloading audio (cache miss): {url[:80]}...")
        audio_data = download_func(url)
        
        # 3. 将下载的数据存入缓存
        if audio_data:
            self.set_audio(url, audio_data, expire)
        
        return audio_data
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            包含连接状态和配置的字典
        """
        config = get_redis_config()
        
        stats = {
            'enabled': is_cache_enabled(),
            'connected': self.enabled,
            'host': config['host'],
            'port': config['port'],
            'db': config['db'],
            'expire_seconds': get_cache_expire(),
        }
        
        if self.enabled:
            try:
                info = self.client.info('memory')
                stats['memory_used'] = info.get('used_memory_human', 'unknown')
                stats['memory_peak'] = info.get('used_memory_peak_human', 'unknown')
                
                # 统计音频缓存 key 数量
                keys_count = len(list(self.client.scan_iter(match='audio:url:*', count=1000)))
                stats['cached_audio_count'] = keys_count
                
            except Exception as e:
                stats['stats_error'] = str(e)
        
        return stats


# 全局缓存实例（延迟初始化）
_audio_cache: Optional[AudioCache] = None


def get_audio_cache() -> AudioCache:
    """获取全局音频缓存实例"""
    global _audio_cache
    
    if _audio_cache is None:
        _audio_cache = AudioCache()
    
    return _audio_cache

