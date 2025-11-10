#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感标签标准化映射模块

提供将用户输入的情感标签（中英文同义词）映射到标准8维情感向量的功能。

标准情感维度：
    - happy: 高兴、快乐、开心
    - angry: 愤怒、生气、发怒
    - sad: 悲伤、难过、忧伤
    - afraid: 恐惧、害怕、恐慌
    - disgusted: 反感、厌恶、恶心
    - melancholic: 低落、忧郁、沮丧
    - surprised: 惊讶、吃惊、震惊
    - calm: 平静、自然、淡定
"""

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# 标准情感顺序（固定不变）
STANDARD_EMOTION_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


# 情感标签映射字典
EMOTION_MAPPING = {
    # happy - 高兴
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "joyful": "happy",
    "cheerful": "happy",
    "delighted": "happy",
    "pleased": "happy",
    "excited": "happy",
    "高兴": "happy",
    "快乐": "happy",
    "开心": "happy",
    "愉快": "happy",
    "欢乐": "happy",
    "喜悦": "happy",
    "兴奋": "happy",
    "欣喜": "happy",
    
    # angry - 愤怒
    "angry": "angry",
    "anger": "angry",
    "mad": "angry",
    "furious": "angry",
    "irritated": "angry",
    "annoyed": "angry",
    "enraged": "angry",
    "愤怒": "angry",
    "生气": "angry",
    "发怒": "angry",
    "恼怒": "angry",
    "气愤": "angry",
    "火大": "angry",
    
    # sad - 悲伤
    "sad": "sad",
    "sadness": "sad",
    "unhappy": "sad",
    "sorrow": "sad",
    "sorrowful": "sad",
    "grief": "sad",
    "heartbroken": "sad",
    "悲伤": "sad",
    "难过": "sad",
    "伤心": "sad",
    "忧伤": "sad",
    "哀伤": "sad",
    "痛苦": "sad",
    "悲痛": "sad",
    
    # afraid - 恐惧
    "afraid": "afraid",
    "fear": "afraid",
    "fearful": "afraid",
    "scared": "afraid",
    "frightened": "afraid",
    "terrified": "afraid",
    "anxious": "afraid",
    "nervous": "afraid",
    "panic": "afraid",
    "panicked": "afraid",
    "恐惧": "afraid",
    "害怕": "afraid",
    "恐慌": "afraid",
    "惊恐": "afraid",
    "畏惧": "afraid",
    "紧张": "afraid",
    
    # disgusted - 反感
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "disgusting": "disgusted",
    "repulsed": "disgusted",
    "revolted": "disgusted",
    "nauseated": "disgusted",
    "反感": "disgusted",
    "厌恶": "disgusted",
    "恶心": "disgusted",
    "讨厌": "disgusted",
    "反胃": "disgusted",
    "嫌弃": "disgusted",
    
    # melancholic - 低落
    "melancholic": "melancholic",
    "melancholy": "melancholic",
    "depressed": "melancholic",
    "depression": "melancholic",
    "gloomy": "melancholic",
    "downcast": "melancholic",
    "dejected": "melancholic",
    "despondent": "melancholic",
    "低落": "melancholic",
    "忧郁": "melancholic",
    "沮丧": "melancholic",
    "消沉": "melancholic",
    "抑郁": "melancholic",
    "颓废": "melancholic",
    "低沉": "melancholic",
    
    # surprised - 惊讶
    "surprised": "surprised",
    "surprise": "surprised",
    "astonished": "surprised",
    "amazed": "surprised",
    "shocked": "surprised",
    "startled": "surprised",
    "stunned": "surprised",
    "惊讶": "surprised",
    "吃惊": "surprised",
    "震惊": "surprised",
    "惊奇": "surprised",
    "诧异": "surprised",
    "惊诧": "surprised",
    "愕然": "surprised",
    
    # calm - 平静
    "calm": "calm",
    "normal": "calm",
    "calmness": "calm",
    "peaceful": "calm",
    "serene": "calm",
    "tranquil": "calm",
    "relaxed": "calm",
    "composed": "calm",
    "neutral": "calm",
    "natural": "calm",
    "平静": "calm",
    "自然": "calm",
    "淡定": "calm",
    "平和": "calm",
    "安静": "calm",
    "宁静": "calm",
    "放松": "calm",
    "冷静": "calm",
    "中性": "calm",
}


def normalize_emotion_label(label: str) -> str:
    """
    将情感标签（包括同义词）标准化到8个标准情感维度之一。
    
    标准情感维度：
    - happy: 高兴、快乐、开心
    - angry: 愤怒、生气、发怒
    - sad: 悲伤、难过、忧伤
    - afraid: 恐惧、害怕、恐慌
    - disgusted: 反感、厌恶、恶心
    - melancholic: 低落、忧郁、沮丧
    - surprised: 惊讶、吃惊、震惊
    - calm: 平静、自然、淡定
    
    Args:
        label: 输入的情感标签（中文或英文，大小写不敏感）
        
    Returns:
        标准化后的英文标签，如果无法识别则返回 "calm" 作为默认值
        
    Examples:
        >>> normalize_emotion_label("高兴")
        'happy'
        >>> normalize_emotion_label("joyful")
        'happy'
        >>> normalize_emotion_label("生气")
        'angry'
    """
    # 标准化输入：转小写并去除空格
    normalized_label = label.strip().lower()
    
    # 查找映射
    standard_label = EMOTION_MAPPING.get(normalized_label)
    
    if standard_label is None:
        logger.warning(f"Unknown emotion label '{label}', defaulting to 'calm'")
        return "calm"
    
    return standard_label


def normalize_emotion_dict(emotion_input: dict) -> dict:
    """
    将情感字典的键标准化为8个标准情感维度。
    
    Args:
        emotion_input: 输入的情感字典，键可以是任何同义词，值为情感强度（0.0-1.0）
        
    Returns:
        标准化后的情感字典，按照标准顺序：
        [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        
    Examples:
        >>> normalize_emotion_dict({"开心": 0.8, "生气": 0.2})
        {'happy': 0.8, 'angry': 0.2, 'sad': 0.0, ...}
        
        >>> normalize_emotion_dict({"happy": 0.7, "joyful": 0.5})
        {'happy': 0.7, 'angry': 0.0, ...}  # 同一标签取最大值
    """
    # 初始化标准化字典
    normalized_dict = {emotion: 0.0 for emotion in STANDARD_EMOTION_ORDER}
    
    # 标准化输入字典
    for label, value in emotion_input.items():
        standard_label = normalize_emotion_label(label)
        # 如果同一个标准标签有多个输入，取最大值
        normalized_dict[standard_label] = max(normalized_dict[standard_label], float(value))
    
    return normalized_dict


def emotion_dict_to_vector(emotion_dict: dict) -> list:
    """
    将情感字典转换为情感向量列表。
    
    Args:
        emotion_dict: 标准化的情感字典
        
    Returns:
        情感向量，顺序为 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        
    Examples:
        >>> emotion_dict_to_vector({'happy': 0.8, 'angry': 0.0, 'sad': 0.0, ...})
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """
    return [emotion_dict.get(emotion, 0.0) for emotion in STANDARD_EMOTION_ORDER]


def create_emotion_vector(emotion_input, alpha: float = 1.0) -> list:
    """
    便捷函数：直接从字符串或字典创建情感向量。
    
    Args:
        emotion_input: 情感输入，可以是字符串（单个标签）或字典（多个标签）
        alpha: 情感强度（0.0-1.0），仅当 emotion_input 为字符串时使用
        
    Returns:
        8维情感向量列表
        
    Examples:
        >>> create_emotion_vector("happy", 0.8)
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        >>> create_emotion_vector({"高兴": 0.7, "平静": 0.3})
        [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
    """
    if isinstance(emotion_input, str):
        # 单个标签字符串
        standard_emotion = normalize_emotion_label(emotion_input)
        emotion_dict = {standard_emotion: alpha}
        normalized_dict = normalize_emotion_dict(emotion_dict)
        return emotion_dict_to_vector(normalized_dict)
    
    elif isinstance(emotion_input, dict):
        # 情感字典
        normalized_dict = normalize_emotion_dict(emotion_input)
        return emotion_dict_to_vector(normalized_dict)
    
    else:
        raise TypeError(
            f"emotion_input must be str or dict, got {type(emotion_input)}"
        )


if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("情感标签标准化测试")
    print("=" * 60)
    
    # 测试单个标签
    test_labels = ["高兴", "happy", "joyful", "生气", "angry", "平静"]
    for label in test_labels:
        standard = normalize_emotion_label(label)
        print(f"'{label}' -> '{standard}'")
    
    print("\n" + "=" * 60)
    print("情感字典标准化测试")
    print("=" * 60)
    
    # 测试字典
    test_dict = {"开心": 0.7, "生气": 0.3}
    print(f"输入: {test_dict}")
    normalized = normalize_emotion_dict(test_dict)
    print(f"标准化: {normalized}")
    vector = emotion_dict_to_vector(normalized)
    print(f"向量: {vector}")
    
    print("\n" + "=" * 60)
    print("便捷函数测试")
    print("=" * 60)
    
    # 测试便捷函数
    vec1 = create_emotion_vector("happy", 0.8)
    print(f"create_emotion_vector('happy', 0.8) = {vec1}")
    
    vec2 = create_emotion_vector({"高兴": 0.7, "平静": 0.3})
    print(f"create_emotion_vector({{'高兴': 0.7, '平静': 0.3}}) = {vec2}")

