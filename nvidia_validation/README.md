# Whisper NVIDIA 平台环境验证

本目录包含 Whisper 在 NVIDIA GPU 平台的完整验证套件，涵盖环境检查、准确率基准测试和性能基准测试三个模块，测试结果汇总为统一的验证报告。

---

## 验证目标

1. 完成 Whisper 模型在 NVIDIA GPU 平台的编译、部署与端到端推理；
2. 音频转文字准确率与官方开源版本偏差 ≤ 5%（WER），推理性能（RTF / 吞吐量）需形成对比报告；
3. 输出完整的 NVIDIA 平台移植报告，包含环境配置清单、依赖库版本、问题记录及解决方案。

---

## 当前环境

| 配置项 | 详情 |
|--------|------|
| GPU | NVIDIA A40 × 8（单卡 46068 MiB VRAM）|
| CUDA（PyTorch） | 12.4 |
| cuDNN | 9.1.0 |
| Triton | 3.0.0 |
| PyTorch | 2.4.1+cu124 |
| Python | 3.10.12 |
| Whisper（本地） | 20231117 |

---

## 目录结构

```
nvidia_validation/
├── README.md                          # 本文件（验证说明、目标、执行方法、进度总览）
├── generate_report.py                 # 报告生成脚本（汇总所有测试结果）
│
├── env_test/                          # 阶段一：环境检查
│   └── check_env.py                   # 环境信息采集与冒烟测试脚本
│
├── accuracy_benchmark/                # 阶段二：准确率基准测试
│   └── run_wer_benchmark.py           # WER 对比测试脚本
│
├── perf_benchmark/                    # 阶段三：性能基准测试
│   ├── generate_test_audio.py         # 生成固定时长测试音频
│   └── run_perf_benchmark.py          # RTF / 吞吐量 / 显存测试脚本
│
└── results/                           # 所有测试结果输出目录
    ├── env_check.json                 # 环境检查原始数据
    ├── accuracy_results.json          # 准确率测试原始数据
    ├── perf_results.json              # 性能测试原始数据
    └── validation_report_YYYYMMDD.md  # 最终验证报告
```

---

## 前置条件

```bash
# 激活项目虚拟环境
source <venv-root>/bin/activate

# 安装本地 whisper（开发模式）
cd <repo-root>
pip install -e .

# 安装测试额外依赖
pip install jiwer soundfile

# 设置模型缓存路径（所有命令均需此环境变量）
export XDG_CACHE_HOME=<model-cache-root>

# 确认 CUDA 可用
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 执行方法

所有脚本均在 Whisper 仓库根目录（下文记为 `<repo-root>`）执行。
`XDG_CACHE_HOME` 应指向模型缓存的父目录，Whisper 权重实际位于
`$XDG_CACHE_HOME/whisper/`；未设置时使用标准的 `~/.cache/whisper/`。

### 阶段一：环境检查

```bash
XDG_CACHE_HOME=<model-cache-root> \
python3 nvidia_validation/env_test/check_env.py
```

**功能**：采集系统信息（OS、GPU、驱动、CUDA、cuDNN、Triton）、所有依赖版本，并用 tiny 模型对 `tests/jfk.flac` 进行冒烟测试，验证端到端推理可用。

**验证 Checklist**：
- [x] `pip install -e .` 安装成功
- [x] `torch.cuda.is_available() = True`
- [x] Triton 算子（`triton_ops.py`）在 CUDA 12.4 下正常激活
- [x] 所有模型（tiny → large-v3）可正常加载并完成推理

**输出**：`results/env_check.json`

---

### 阶段二：准确率基准测试

```bash
XDG_CACHE_HOME=<model-cache-root> \
python3 nvidia_validation/accuracy_benchmark/run_wer_benchmark.py

# 可选：只测试指定模型
XDG_CACHE_HOME=<model-cache-root> \
python3 nvidia_validation/accuracy_benchmark/run_wer_benchmark.py \
  --models tiny base small
```

**功能**：对 tiny / base / small / medium / large-v3 五个模型，使用 `tests/jfk.flac`（JFK 演讲片段，11s，英文）计算 WER（字错误率）和 CER（字符错误率），并与官方 `openai-whisper` 对比，验证偏差 ≤ 5%。

**验收标准**：所有模型满足 `|WER_本地 − WER_官方| ≤ 5%`

**输出**：`results/accuracy_results.json`

---

### 阶段三：性能基准测试

```bash
# 先生成测试音频（30s / 60s / 300s），只需执行一次
XDG_CACHE_HOME=<model-cache-root> \
python3 nvidia_validation/perf_benchmark/generate_test_audio.py

# 运行完整性能测试（约 20 分钟）
XDG_CACHE_HOME=<model-cache-root> \
python3 nvidia_validation/perf_benchmark/run_perf_benchmark.py

# 可选：只测试部分模型或时长
XDG_CACHE_HOME=<model-cache-root> \
python3 nvidia_validation/perf_benchmark/run_perf_benchmark.py \
  --models tiny base small --durations 30 60
```

**功能**：测量各模型在 FP32 / FP16 精度下，对 30s / 60s / 300s 三种时长音频的 RTF（实时率）、吞吐量和 GPU 峰值显存，统计 3 次取平均，1 次预热。

**输出**：`results/perf_results.json`

---

### 生成最终报告

三个阶段测试完成后，执行以下命令汇总报告：

```bash
python3 nvidia_validation/generate_report.py
```

**输出**：`results/validation_report_YYYYMMDD.md`（以执行日期命名）

---

## 一键执行（全流程）

```bash
cd <repo-root>
source <venv-root>/bin/activate
export XDG_CACHE_HOME=<model-cache-root>

python3 nvidia_validation/env_test/check_env.py 2>/dev/null
python3 nvidia_validation/accuracy_benchmark/run_wer_benchmark.py 2>/dev/null
python3 nvidia_validation/perf_benchmark/run_perf_benchmark.py 2>/dev/null
python3 nvidia_validation/generate_report.py
```

---

## 报告结构

最终报告 `results/validation_report_YYYYMMDD.md` 包含以下章节：

```
第一部分 — 环境与部署
  1.1 环境配置（OS / GPU / 驱动 / CUDA / cuDNN）
  1.2 依赖库版本表
  1.3 模型缓存
  1.4 安装与冒烟测试
  1.5 历史测试结果（pytest 单元测试 + 集成测试）

第二部分 — 准确率基准测试（WER）
  2.1 WER 对比表（各模型 vs 官方）
  2.2 转录详情
  2.3 结论

第三部分 — 性能基准测试（RTF / 吞吐量）
  3.1 RTF 汇总表
  3.2 FP16 与 FP32 对比
  3.3 结论与建议

第四部分 — 问题记录与解决方案
```

---

## 报告输出路径

| 文件 | 说明 |
|------|------|
| `results/env_check.json` | 环境检查原始数据（JSON） |
| `results/accuracy_results.json` | WER 测试原始数据（JSON） |
| `results/perf_results.json` | 性能测试原始数据（JSON） |
| `results/validation_report_YYYYMMDD.md` | 最终验证报告（Markdown，中英双语） |

---

## 验收标准

| 测试项 | 标准 |
|--------|------|
| 准确率（WER 偏差） | 本地与官方版本偏差 ≤ 5% |
| 实时性（RTF） | RTF < 1.0（推理速度快于实时） |
| 端到端推理 | 所有模型可正常加载并完成转录 |

---

## 进度总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段一 | 环境与部署验证 | ✅ 已完成 |
| 阶段二 | 准确率基准测试（WER 对比）| ✅ 已完成 |
| 阶段三 | 性能基准测试（RTF / 吞吐量）| ✅ 已完成 |
| 阶段四 | NVIDIA 平台验证报告输出 | ✅ 已完成 |

*最后更新：2026-03-19*
