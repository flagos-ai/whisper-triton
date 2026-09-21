# Whisper 测试套件说明

本目录包含 Whisper 的全部测试文件，使用 pytest 运行。

---

## 文件总览

| 文件 | 类型 | 需要加速器 | 需要模型文件 | 运行速度 |
|------|------|---------|------------|---------|
| `test_audio.py` | 单元测试 | 否 | 否 | 秒级 |
| `test_tokenizer.py` | 单元测试 | 否 | 否 | 秒级 |
| `test_normalizer.py` | 单元测试 | 否 | 否 | 秒级 |
| `test_timing.py` | 单元测试 | 部分 | 否 | 秒级 |
| `test_transcribe.py` | 集成测试 | 否 | 是（全量） | 分钟~小时级 |

---

## 各文件说明

### `conftest.py` — pytest 全局配置

注册 `requires_accelerator` marker，并保留 `requires_cuda` 作为兼容别名。测试会发现 CUDA 或由 `torch_npu`、`torch_mlu`、`torch_musa` 注册的 PrivateUse1 设备；也可用 `WHISPER_TEST_DEVICE` 显式指定厂商 PyTorch 设备。无加速器时自动跳过直连 kernel 测试。`test_transcribe.py` 单独通过 `default_device()` 探测选择设备（优先 `WHISPER_TEST_DEVICE`，无加速器时在 CPU 执行），不因缺少加速器而跳过。

```bash
# 跳过所有加速器测试
pytest tests/ -m "not requires_accelerator and not requires_cuda"
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
| `test_dtw_accelerator_equivalence` | `requires_accelerator` | 加速器 Triton DTW 与 CPU DTW 在 4 种矩阵尺寸下结果一致 |
| `test_median_filter` | 无 | CPU median filter 正确性：与 scipy 参考实现对比，覆盖 4 种 tensor 形状 × 5 种窗宽 |
| `test_median_filter_accelerator_equivalence` | `requires_accelerator` | 加速器 median filter 与 CPU 结果一致 |
| `test_median_filter_accelerator_kernel_equivalence` | `requires_accelerator` | 对多维输入和全部窗宽直接启动 Triton median kernel，避免 CPU fallback 掩盖失败 |
| `test_median_filter_accelerator_kernel_duplicate_values` | `requires_accelerator` | 验证 Triton kernel 对重复值的中位数秩判定 |
| `test_timing_falls_back_to_cpu` | `requires_accelerator` | 模拟 Triton 编译失败，验证两个 timing 算法回退到 CPU |

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

# 跳过加速器直连和集成测试
pytest tests/ -m "not requires_accelerator and not requires_cuda" --ignore=tests/test_transcribe.py

# 运行集成测试（单个模型，避免全量下载）
pytest tests/test_transcribe.py -k "base" -v

# 运行全部测试
pytest tests/ -v
```

---

## 测试结果归档

上游 pytest 本身不创建结果目录。CNPort 使用统一 runner：

```bash
python scripts/run_platform_tests.py --platform <platform-key>
```

结果写入 `tests/results/<platform>/<YYYYMMDDHHMMSS>/`，包含环境、命令、JUnit、日志、摘要和 Markdown 报告。逐次产物默认不提交 Git，格式说明见 `tests/results/README.md`。
