# Whisper 测试套件说明

本目录包含 Whisper 的全部测试文件，使用 pytest 运行。

---

## 文件总览

| 文件 | 类型 | 需要 GPU | 需要模型文件 | 运行速度 |
|------|------|---------|------------|---------|
| `test_audio.py` | 单元测试 | 否 | 否 | 秒级 |
| `test_tokenizer.py` | 单元测试 | 否 | 否 | 秒级 |
| `test_normalizer.py` | 单元测试 | 否 | 否 | 秒级 |
| `test_timing.py` | 单元测试 | 部分 | 否 | 秒级 |
| `test_transcribe.py` | 集成测试 | 推荐 | 是（全量） | 分钟~小时级 |

---

## 各文件说明

### `conftest.py` — pytest 全局配置

注册自定义 marker `requires_cuda`，供 `test_timing.py` 中的 CUDA 测试使用，便于在无 GPU 环境中选择性跳过。

```bash
# 跳过所有 requires_cuda 测试
pytest tests/ -m "not requires_cuda"
```

---

### `jfk.flac` — 测试音频素材

肯尼迪总统演讲片段（英语），时长约 10-12 秒，采样率 16kHz。`test_audio.py` 和 `test_transcribe.py` 共用此文件作为输入，无需外部数据源。

---

### `test_audio.py` — 音频处理单元测试

**测试对象**：`whisper/audio.py` 中的 `load_audio()` 和 `log_mel_spectrogram()`

| 断言 | 验证内容 |
|------|---------|
| `audio.ndim == 1` | 音频加载后是一维数组 |
| `SAMPLE_RATE * 10 < len < SAMPLE_RATE * 12` | jfk.flac 时长在 10-12 秒之间 |
| `0 < audio.std() < 1` | 音频已归一化到 [-1, 1] 区间 |
| `np.allclose(mel_from_audio, mel_from_file)` | 从 numpy array 和从文件路径计算的 Mel 频谱结果一致 |
| `mel.max() - mel.min() <= 2.0` | Mel 频谱动态范围在合理区间内 |

---

### `test_tokenizer.py` — 分词器单元测试

**测试对象**：`whisper/tokenizer.py` 中的 `get_tokenizer()`

| 测试函数 | 验证内容 |
|---------|---------|
| `test_tokenizer` | sot token 在 sot_sequence 中；language token 数量与 code 数量一致；所有 language token 均在 timestamp_begin 之前 |
| `test_multilingual_tokenizer` | GPT2 与多语言 tokenizer 均能正确编解码韩语；多语言 tokenizer 的 token 数更少（词表更大更高效） |
| `test_split_on_unicode` | 按 Unicode 边界切分 token 的逻辑正确，含 UTF-8 解码异常字符（`\ufffd`）的处理 |

---

### `test_normalizer.py` — 文本规范化器单元测试

**测试对象**：`whisper/normalizers/english.py` 中的三个规范化器

| 测试函数 | 验证内容 |
|---------|---------|
| `test_number_normalizer` | 英文数字转阿拉伯数字，覆盖序数词、货币、百分比、复合数字等 30+ 个 case |
| `test_spelling_normalizer` | 英式→美式拼写（如 mobilisation→mobilization） |
| `test_text_normalizer` | 缩写展开（Let's→let us）、单位分离（10km→10 km）、称谓展开（Mr.→mister） |

---

### `test_timing.py` — 词级时间戳单元测试

**测试对象**：`whisper/timing.py` 中的 `dtw_cpu()`、`dtw_cuda()`、`median_filter()`

DTW（动态时间规整）是词级时间戳（word timestamps）功能的核心算法。

| 测试函数 | marker | 验证内容 |
|---------|--------|---------|
| `test_dtw` | 无 | CPU DTW 正确性：构造已知最优路径的矩阵，验证返回路径与预期一致，覆盖 4 种矩阵尺寸 |
| `test_dtw_cuda_equivalence` | `requires_cuda` | CUDA DTW 与 CPU DTW 在 4 种矩阵尺寸下结果一致 |
| `test_median_filter` | 无 | CPU median filter 正确性：与 scipy 参考实现对比，覆盖 4 种 tensor 形状 × 4 种窗宽 |
| `test_median_filter_equivalence` | `requires_cuda` | GPU median filter 与 CPU 结果一致 |

---

### `test_transcribe.py` — 端到端集成测试

**测试对象**：完整推理链路 `whisper.load_model()` + `model.transcribe()`

对所有可用模型（tiny / base / small / medium / large / turbo 等全系列）各执行一次推理，验证：

| 断言 | 验证内容 |
|------|---------|
| `result["language"] == "en"` | 语种识别正确 |
| `result["text"] == "".join(segments["text"])` | 全文与分段拼接结果一致 |
| 三个关键短语出现在转录文本中 | 转录内容语义正确 |
| `tokenizer.decode(all_tokens) == result["text"]` | token 解码与文本输出一致 |
| 时间戳 token 以 `<\|0.00\|>` 开头 | 时间戳 token 格式正确 |
| "Americans" 的词级时间戳覆盖 1.8 秒 | 词级时间戳精度正确 |

> **注意**：此测试会自动下载全部 Whisper 模型权重（数 GB），离线环境或资源受限时建议只跑单模型：
> ```bash
> pytest tests/test_transcribe.py -k "base"
> ```

---

## 运行方式

```bash
# 运行全部单元测试（不含集成测试）
pytest tests/test_audio.py tests/test_tokenizer.py tests/test_normalizer.py tests/test_timing.py -v

# 跳过需要 CUDA 的测试
pytest tests/ -m "not requires_cuda" --ignore=tests/test_transcribe.py

# 运行集成测试（单个模型，避免全量下载）
pytest tests/test_transcribe.py -k "base" -v

# 运行全部测试
pytest tests/ -v
```

---

## 测试结果归档

> **当前状态**：测试脚本本身不包含结果归档逻辑，输出仅打印到终端。
>
> 按 CNPort 项目规范，测试报告归档至本目录下：
> ```
> tests/test-result/
> ├── unit-test/
> │   └── unit-test-YYYYMMDDHHmm.log
> └── integration-test/
>     └── integration-test-YYYYMMDDHHmm.log
> ```
>
> 手动归档方式（在仓库根目录执行）：
> ```bash
> # 单元测试
> pytest tests/test_audio.py tests/test_tokenizer.py tests/test_normalizer.py tests/test_timing.py \
>     -v --tb=short 2>&1 | tee tests/test-result/unit-test/unit-test-$(date +%Y%m%d%H%M).log
>
> # 集成测试
> pytest tests/test_transcribe.py -v --tb=short \
>     2>&1 | tee tests/test-result/integration-test/integration-test-$(date +%Y%m%d%H%M).log
> ```
