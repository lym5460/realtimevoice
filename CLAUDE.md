# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个实时 AI 语音对话系统,允许用户通过语音与大语言模型进行自然对话。系统使用 WebSocket 实现低延迟的双向音频流传输。

**核心技术栈:**
- 后端: Python FastAPI (< 3.13)
- 前端: Vanilla JavaScript + Web Audio API
- 通信: WebSockets
- 语音识别: RealtimeSTT
- 语音合成: RealtimeTTS (支持 Kokoro、Coqui、Orpheus 引擎)
- LLM: Ollama/OpenAI
- 部署: Docker + Docker Compose

## 开发命令

### Docker 方式 (推荐)

```bash
# 构建镜像 (首次或修改配置后)
docker compose build

# 启动服务 (后台运行)
docker compose up -d

# 拉取 Ollama 模型 (首次启动后必需)
docker compose exec ollama ollama pull hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M

# 查看日志
docker compose logs -f app
docker compose logs -f ollama

# 停止服务
docker compose down
```

### 手动运行方式

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# 升级 pip
python -m pip install --upgrade pip

# 进入代码目录
cd code

# 安装 PyTorch (CUDA 12.1 示例,根据实际 GPU 调整)
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 启动服务器
python server.py
```

访问地址: `http://localhost:8000`

## 架构说明

### 核心组件及职责

1. **[server.py](code/server.py)** - FastAPI 应用主入口
   - WebSocket 端点 `/ws` 处理客户端连接
   - 管理 4 个并发 asyncio 任务:
     - `process_incoming_data`: 接收客户端音频/控制消息
     - `AudioInputProcessor.process_chunk_queue`: 处理音频队列
     - `send_text_messages`: 发送文本消息到客户端
     - `send_tts_chunks`: 发送 TTS 音频块到客户端
   - `TranscriptionCallbacks` 类管理单个连接的状态和回调

2. **[speech_pipeline_manager.py](code/speech_pipeline_manager.py)** - 语音处理管道协调器
   - 统一管理 LLM + TTS 生成流程
   - `RunningGeneration` 类追踪单次生成的完整生命周期
   - 处理快速响应(quick_answer)和完整响应(final_answer)两阶段生成
   - 实现中断检测和生成终止逻辑

3. **[audio_in.py](code/audio_in.py)** - 音频输入处理
   - `AudioInputProcessor` 类: 重采样输入音频 (48kHz → 16kHz)
   - 桥接 WebSocket 输入和转录器
   - 管理转录任务的异步循环

4. **[transcribe.py](code/transcribe.py)** - 语音识别 (RealtimeSTT)
   - `TranscriptionProcessor` 封装 RealtimeSTT 的 AudioToTextRecorder
   - 实现动态静默检测 ([turndetect.py](code/turndetect.py))
   - 触发潜在句子结束/用户回合结束的回调

5. **[audio_module.py](code/audio_module.py)** - 音频输出处理
   - `AudioProcessor` 类: 初始化并配置 TTS 引擎 (Coqui/Kokoro/Orpheus)
   - 提供文本到音频流的合成接口
   - 测量 TTFA (Time To First Audio) 延迟

6. **[llm_module.py](code/llm_module.py)** - LLM 提供商抽象
   - `LLM` 类: 统一 Ollama/OpenAI/LMStudio 的接口
   - 流式生成文本响应
   - 维护对话历史上下文

7. **[turndetect.py](code/turndetect.py)** - 智能回合检测
   - `TurnDetector`: 使用 transformer 模型分析文本完整性
   - 动态调整静默超时阈值 (根据对话速度)

8. **前端 ([static/](code/static/))**
   - [index.html](code/static/index.html): 简洁的 UI 界面
   - [app.js](code/static/app.js): WebSocket 通信和状态管理
   - [pcmWorkletProcessor.js](code/static/pcmWorkletProcessor.js): AudioWorklet 采集麦克风数据
   - [ttsPlaybackProcessor.js](code/static/ttsPlaybackProcessor.js): AudioWorklet 播放 TTS 音频

### 数据流

**用户语音 → AI 响应:**
1. 浏览器通过 WebSocket 发送 PCM 音频块 (含时间戳和 TTS 播放状态标志)
2. `AudioInputProcessor` 重采样并送入 `TranscriptionProcessor`
3. RealtimeSTT 实时转录,触发 `on_partial` 回调
4. 检测到潜在句子结束时,调用 `on_potential_sentence`
5. `SpeechPipelineManager.prepare_generation()` 创建 `RunningGeneration`
6. LLM 开始生成文本,通过 `on_partial_assistant_text` 发送中间结果
7. TTS 合成音频块放入 `audio_chunks` 队列
8. `send_tts_chunks` 任务从队列取出音频,经 `UpsampleOverlap` 处理后发送
9. 客户端使用 AudioWorklet 播放音频

**中断处理:**
- 用户说话时,`on_recording_start` 检测到 TTS 正在播放会立即中断
- 触发 `abort_generation()`,停止 LLM 和 TTS 生成
- 发送 `stop_tts` 和 `tts_interruption` 消息给客户端

## 配置要点

### 修改 TTS 引擎

在 [server.py](code/server.py) 中修改:
```python
TTS_START_ENGINE = "coqui"  # 或 "kokoro", "orpheus"
```

Orpheus 引擎特定模型:
```python
TTS_ORPHEUS_MODEL = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf"
```

### 修改 LLM 后端

在 [server.py](code/server.py) 中修改:
```python
LLM_START_PROVIDER = "ollama"  # 或 "openai", "lmstudio"
LLM_START_MODEL = "hf.co/bartowski/..."
```

### STT 模型配置

在 [transcribe.py](code/transcribe.py) 的 `DEFAULT_RECORDER_CONFIG` 中调整:
- `model`: Whisper 模型大小 (默认 `"base.en"`)
- `language`: 目标语言 (默认 `"en"`)
- `silence_limit_seconds`: 静默超时

### 系统提示词

修改 [system_prompt.txt](code/system_prompt.txt) 来定制 AI 人格。

### Docker 配置注意事项

⚠️ **重要:** 如果使用 Docker,所有配置修改必须在 `docker compose build` 之前完成,否则需要重新构建镜像。

Ollama 模型不包含在镜像中,需要在容器启动后手动拉取:
```bash
docker compose exec ollama ollama pull <model_name>
```

## 常见开发场景

### 添加新的 TTS 引擎

1. 在 [audio_module.py](code/audio_module.py) 的 `AudioProcessor.__init__` 中添加引擎初始化逻辑
2. 更新 `ENGINE_SILENCES` 字典配置静默时长
3. 在 [server.py](code/server.py) 中设置 `TTS_START_ENGINE`

### 修改中断逻辑

核心逻辑在:
- [server.py:774-814](code/server.py#L774-L814) `TranscriptionCallbacks.on_recording_start()`
- [speech_pipeline_manager.py](code/speech_pipeline_manager.py) `check_abort()` 方法

### 调整音频队列大小

设置环境变量 `MAX_AUDIO_QUEUE_SIZE` (默认 50):
```bash
export MAX_AUDIO_QUEUE_SIZE=100
```
或在 [docker-compose.yml](docker-compose.yml) 中修改。

### 启用 SSL/HTTPS

在 [server.py:30](code/server.py#L30) 设置:
```python
USE_SSL = True
SSL_CERT_PATH = "path/to/cert.pem"
SSL_KEY_PATH = "path/to/key.pem"
```

Docker 用户需同步修改 [docker-compose.yml](docker-compose.yml) 的端口映射和卷挂载。

## 依赖管理

主要依赖在 [requirements.txt](requirements.txt):
- `realtimestt==0.3.104`
- `realtimetts[kokoro,coqui,orpheus]==0.5.5`
- `fastapi`, `uvicorn`
- `ollama`, `openai`

PyTorch 需根据 CUDA 版本单独安装,不包含在 requirements.txt 中。

## GPU 要求

强烈推荐使用 NVIDIA GPU:
- Whisper (STT) 和 Coqui TTS 在 GPU 上性能显著更好
- Docker 需要 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 默认假设 CUDA 12.1,其他版本需调整 PyTorch 安装命令

## 项目状态

该项目目前由社区驱动维护,原作者不再积极开发新功能。欢迎高质量的 Pull Request。
