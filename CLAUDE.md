@../common/CLAUDE.md

# Whisper — CNPort 移植项目说明

## 项目简介

Whisper 是 OpenAI 开源的通用语音识别模型，基于 Transformer 序列到序列架构，使用大规模弱监督数据训练。
支持多语言语音识别、语音翻译（到英语）和语种识别等任务。

本仓库是 CNPort 项目对 `openai/whisper` 的移植版本，针对不同芯片平台（CUDA、AMD、Apple）维护独立分支。

- upstream: https://github.com/openai/whisper
- 公司仓库 (origin): https://codeup.aliyun.com/69b0f1246ec1ba182dc75e1c/HY_CNPort/whisper.git（~~旧：https://code.iflytek.com/HY_CNPort/whisper.git，已弃用~~）
- 主跟踪分支: `main`
- 当前版本: v20250625

## 安装 / 构建

Whisper 是纯 Python 包，无需编译，直接 pip 安装即可。

```bash
# 从本地仓库安装（开发模式，修改立即生效）
pip install -e .

# 从本地仓库安装（普通安装）
pip install .

# 安装含开发依赖（测试、格式化等）
pip install -e ".[dev]"

# 安装运行时依赖
pip install -r requirements.txt
```

### 系统依赖

```bash
# 必须安装 ffmpeg（用于音频文件读取）
sudo apt update && sudo apt install ffmpeg   # Ubuntu/Debian

# tiktoken 可能需要 Rust（如果没有预构建 wheel）
pip install setuptools-rust
```

### 依赖说明

| 依赖 | 说明 |
|------|------|
| torch | PyTorch，推理核心 |
| tiktoken | 分词器（OpenAI 实现） |
| numpy / numba | 数值计算 |
| triton | GPU kernel 加速（仅 x86_64 Linux） |
| ffmpeg | 音频解码（系统工具，非 Python 包） |

## 测试命令

测试文件位于 `tests/` 目录，使用 pytest 运行：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_audio.py
pytest tests/test_transcribe.py
pytest tests/test_tokenizer.py
pytest tests/test_normalizer.py
pytest tests/test_timing.py

# 运行单个测试（跳过耗时的全模型转录测试）
pytest tests/test_audio.py -v

# 代码格式检查
black --check whisper/
flake8 whisper/
isort --check-only whisper/
```

注意：`test_transcribe.py` 会自动下载所有 Whisper 模型文件（数 GB），在无网络或资源受限环境下会失败，建议只跑单模型测试。

## 命令行使用

```bash
# 转录音频（默认使用 turbo 模型）
whisper audio.wav

# 指定模型
whisper audio.wav --model small

# 指定语言（中文）
whisper audio.wav --model medium --language Chinese

# 翻译成英语（用多语言模型，turbo 不支持翻译）
whisper audio.wav --model medium --language Japanese --task translate

# 查看所有选项
whisper --help
```

## 重要目录结构

```
whisper/
├── whisper/                   # 核心 Python 包
│   ├── __init__.py            # 包入口，模型下载逻辑，_MODELS 映射表
│   ├── model.py               # Transformer 模型定义（Encoder/Decoder/MultiHeadAttention）
│   ├── transcribe.py          # 转录主逻辑（滑动窗口 30s 推理），CLI 入口
│   ├── decoding.py            # 解码策略（贪心/束搜索），detect_language/decode API
│   ├── audio.py               # 音频加载、Mel 频谱计算（ffmpeg 调用封装）
│   ├── tokenizer.py           # 分词器封装，多语言支持
│   ├── timing.py              # 词级时间戳（DTW 对齐）
│   ├── triton_ops.py          # Triton GPU kernel（LayerNorm 等算子加速）
│   ├── utils.py               # 通用工具函数
│   ├── version.py             # 版本号
│   ├── normalizers/           # 文本规范化器（英语/多语言）
│   └── assets/                # 内置资产文件（词表等）
├── tests/                     # 测试套件
│   ├── test_transcribe.py     # 端到端转录测试（需下载模型）
│   ├── test_audio.py          # 音频处理单元测试
│   ├── test_tokenizer.py      # 分词器测试
│   ├── test_normalizer.py     # 规范化器测试
│   ├── test_timing.py         # 时间戳对齐测试
│   ├── conftest.py            # pytest 配置
│   └── jfk.flac               # 测试用音频样本（JFK 演讲片段）
├── notebooks/                 # Jupyter 示例
│   ├── LibriSpeech.ipynb      # 英语 ASR 示例
│   └── Multilingual_ASR.ipynb # 多语言 ASR 示例
├── data/                      # 数据文件
├── pyproject.toml             # 项目元数据和构建配置（PEP 621）
├── requirements.txt           # 运行时依赖列表
├── CHANGELOG.md               # 版本变更记录
└── model-card.md              # 模型说明卡片
```

### 关键代码路径

- 模型下载缓存默认位置: `~/.cache/whisper/`
- 推理入口: `whisper.transcribe()` → `whisper/transcribe.py`
- 模型加载: `whisper.load_model()` → `whisper/__init__.py`
- 音频处理: 16kHz 采样率，30s 滑动窗口，80/128 Mel 频道
- SDPA: 如果 PyTorch 版本支持 `scaled_dot_product_attention`，自动启用

## 移植相关注意事项

### 分支结构

| 分支 | 说明 |
|------|------|
| `main` | 跟踪 upstream `openai/whisper` main 分支 |
| `platform/cuda` | NVIDIA CUDA 平台移植 |
| `platform/amd` | AMD ROCm 平台移植 |
| `platform/apple` | Apple Silicon (MPS) 平台移植 |

### 平台差异重点

**CUDA (platform/cuda)**
- 依赖 `triton` kernel 加速（`whisper/triton_ops.py`），需确认与目标 CUDA/triton 版本兼容
- `model.py` 中 SDPA 路径在 CUDA 上性能最优，优先保留
- 模型权重加载使用 `weights_only=True`（torch >= 1.13），注意版本兼容性

**AMD (platform/amd)**
- ROCm 环境下 `triton` 包可能需要替换为 ROCm 版本的 triton 或禁用
- `pyproject.toml` 中 triton 依赖条件为 `platform_machine=='x86_64' and sys_platform=='linux'`，AMD 平台满足此条件，需验证 ROCm triton 可用性
- `torch.cuda.is_available()` 在 ROCm 下返回 True，行为与 CUDA 一致

**Apple (platform/apple)**
- MPS 后端：`device = "mps"` 替代 `"cuda"`
- triton 在 macOS 不可用（pyproject.toml 已通过条件排除），无需额外处理
- 部分 numba JIT 函数在 Apple Silicon 上可能需要验证
- 注意 float16 在 MPS 上的支持情况（某些操作可能需要 float32 fallback）

### 同步上游流程

```bash
# 在 CNPort 根目录执行
cd /data/xuchen2/git/CNPort

# 同步 whisper
./sync-upstream.sh whisper

# 或手动同步
cd whisper
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main

# rebase 平台分支
git checkout platform/cuda
git rebase main
git push origin platform/cuda --force-with-lease
# 对 platform/amd、platform/apple 重复以上操作
```

**注意**: 推送到公司仓库（codeup.aliyun.com）无需绕过代理：

```bash
git push origin platform/cuda
```

### 移植原则

1. 平台分支上只保留平台特有的最小化差异（硬件加速调用、依赖替换、设备适配）
2. 避免在平台分支上修改与平台无关的代码逻辑，减少 rebase 冲突
3. triton_ops.py 是平台差异最集中的文件，AMD/Apple 移植时重点关注
4. 新增平台特有依赖在 `pyproject.toml` 中用条件标记，避免影响其他平台

### 常见问题

- **triton 版本冲突**: upstream 对 triton 要求 `>=2`，ROCm triton 版本号可能不同，必要时在 AMD 分支 pin 特定版本
- **模型下载失败**: 模型文件托管在 `openaipublic.azureedge.net`，离线环境需手动下载后通过路径加载
- **DTW 设备不一致**: `timing.py` 中 DTW cost tensor 需与输入在同一设备（已在 v20250625 修复），移植时确认该修复已包含
