# TTS 音色管理系统

一个完整的TTS音色管理系统，包括后端API服务和前端管理界面。

## 系统架构

```
tts/
├── frontend/              # 前端应用 (React + TypeScript)
├── indextts/             # TTS模型推理模块
├── models/               # 模型文件
├── data/                 # 数据目录
│   ├── db/              # 数据库文件
│   └── voices/          # 音色音频文件
├── server.py            # FastAPI后端服务
└── voice_manager.py     # 音色管理模块
```

## 功能特性

### 后端 (FastAPI)
- 🎭 **角色管理**: CRUD操作，支持按性别、年龄段、名称筛选
- 🎵 **音色管理**: 支持多情绪音色上传和管理
- 🗣️ **TTS推理**: 基于IndexTTS2模型的语音合成
- 💾 **数据持久化**: SQLite数据库存储
- 📊 **统计信息**: 角色和音色数量统计

### 前端 (React)
- 🎨 **音色管理页面**: 
  - 创建/编辑/删除角色
  - 为角色添加不同情绪的音色
  - 统计信息展示
- 🎤 **TTS测试页面**: 
  - 选择角色和情绪
  - 文本转语音
  - 在线播放和下载

## 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+
- CUDA (可选，用于GPU加速)

### 后端启动

1. 安装Python依赖
```bash
pip install fastapi uvicorn loguru
# 安装TTS相关依赖...
```

2. 启动后端服务
```bash
python server.py
```

后端服务将运行在 http://localhost:8000

### 前端启动

1. 进入前端目录并安装依赖
```bash
cd frontend
npm install
```

2. 启动开发服务器
```bash
npm run dev
```

前端应用将运行在 http://localhost:3000

## API文档

启动后端服务后，访问以下地址查看API文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 数据库结构

### roles 表
| 字段 | 类型 | 说明 |
|------|------|------|
| voice_id | TEXT | 角色ID (主键) |
| name | TEXT | 角色名称 |
| description | TEXT | 角色描述 |
| gender | TEXT | 性别 (male/female) |
| age | TEXT | 年龄段 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### voices 表
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| voice_id | TEXT | 角色ID (外键) |
| emotion | TEXT | 情绪类型 |
| audio_path | TEXT | 音频文件路径 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

## 使用流程

1. **创建角色**: 在音色管理页面创建角色，上传normal情绪音频
2. **添加音色**: 为角色添加其他情绪的音色（happy/angry/sad等）
3. **测试TTS**: 在TTS测试页面选择角色和情绪，输入文本生成语音
4. **播放下载**: 在线播放或下载生成的音频

## 开发说明

### 后端开发
- 修改 `server.py` 添加新的API端点
- 修改 `voice_manager.py` 扩展数据库功能

### 前端开发
- 在 `frontend/src/pages/` 添加新页面
- 在 `frontend/src/services/api.ts` 添加API调用
- 在 `frontend/src/types/index.ts` 添加类型定义

## 注意事项

- 每个角色必须有normal情绪的音色才能进行TTS合成
- 音频文件格式建议使用WAV
- 数据库和音频文件存储在 `data/` 目录
- 删除角色时可选择是否删除关联的音色文件

## License

MIT
