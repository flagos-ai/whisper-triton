# Whisper CNPort 多平台安装与验证指南

## 移植边界

Whisper 模型的 Encoder/Decoder 推理保持上游 PyTorch 实现。各平台负责提供可用的
PyTorch、设备扩展和设备字符串；CNPort 不实现厂商推理后端，也不修改 CLI 或
`load_model()` 的上游默认设备策略。

CNPort 的运行时代码适配仅覆盖 `whisper/timing.py` 中已有 Triton 源码的
`median_kernel` 和 `dtw_kernel`。输入位于非 CPU 设备时，代码尝试用该环境的
Triton 编译并启动 kernel；Triton 不可用、编译失败或启动失败时，这两项计算回退到
CPU。平台测试可以通过 `WHISPER_TEST_DEVICE` 显式指定 PyTorch 测试设备，但该变量
只由 pytest 使用，不改变 Whisper 运行时行为。

## 为什么不再自动安装 Triton

上游依赖原来使用以下条件：

```text
triton>=2; Linux x86_64
```

该条件只能识别操作系统和 CPU 架构，无法识别 NVIDIA、昇腾、寒武纪、摩尔线程、海光或平头哥加速器。同为 Linux x86_64 的不同设备可能使用不同 Triton 发行版，但这些发行版通常都暴露同一个 `triton` Python 模块。让 Whisper 自动安装 PyPI `triton` 可能覆盖厂商版本，或形成 Torch、Triton、SDK和编译后端不匹配的环境。

因此 `requirements.txt` 与 `pyproject.toml` 都不再声明 Triton。Triton 是平台环境的一部分，而不是 Whisper 可以跨平台正确选择的普通依赖。NVIDIA 用户同样需要显式安装与其 PyTorch 版本匹配的 Triton。

## 环境匹配规则

每个平台使用独立 venv 或容器，并按以下顺序准备：

1. 安装设备驱动、运行时和编译 SDK。
2. 安装厂商提供或认可的 PyTorch。
3. 安装对应设备扩展，例如 `torch_npu`、`torch_mlu` 或 `torch_musa`。
4. 安装与该 Torch 和 SDK 组合一起验证过的 Triton；不要仅按包名或最高版本选择。
5. 用同一个解释器显式安装 Whisper 的通用依赖，再执行 `python -m pip install -e . --no-deps`。
6. 运行环境检查、kernel 直连测试和全模型集成测试。必要时设置
   `WHISPER_TEST_DEVICE=<torch-device>`，让 pytest 在厂商 PyTorch 设备上执行
   `median_kernel` 和 `dtw_kernel` 的 Triton 编译调用测试。

不要复制其他平台的 `site-packages`，也不要通过调整 `PYTHONPATH` 混用两个 Python 环境。升级 Torch、设备扩展、Triton 或 SDK 中任一项后，应把整个组合视为新环境重新验证。

## 兼容矩阵应记录的字段

每个公开验证摘要至少记录以下信息，并删除用户名、内网地址和内部绝对路径：

| 类别 | 必填字段 |
|---|---|
| 平台 | 厂商、设备型号、设备架构、操作系统架构 |
| Python | Python 版本和实现 |
| PyTorch | Torch 版本、设备扩展名称与版本、运行时设备字符串 |
| Triton | 安装发行版、`triton.__version__`、模块来源、active backend |
| SDK | 驱动/SDK版本、必要环境变量名称、设备二进制格式 |
| Whisper | 上游基线、CNPort commit、模型缓存策略 |
| 验证 | 执行时间、测试命令、通过/失败/跳过数、kernel 直连结果 |

`scripts/check_accelerator_env.py` 会采集其中可自动发现的字段，并检查模块来源和实际设备。平台专有 SDK 版本仍应按厂商命令补充到验证摘要。

## 平台提示

| 平台 | 关键检查 |
|---|---|
| NVIDIA | `torch.cuda.is_available()`、CUDA、Triton NVIDIA backend、`ptxas` |
| 昇腾 | `torch_npu`、CANN环境、`npu` 设备、Ascend backend |
| 寒武纪 | `torch_mlu`、Neuware环境、`mlu` 设备、MLU backend |
| 摩尔线程 | `torch_musa`、MUSA库路径、`musa` 设备、MTGPU backend |
| 海光 | 厂商 Torch/DTK或ROCm环境、实际 HIP兼容设备、AMD backend |
| 平头哥 | PPU SDK、厂商 Torch、CUDA兼容设备及实际加载的工具链 |

厂商包名称和下载源可能随 SDK 版本变化。本仓库记录实机验证组合，不使用未经验证的通用 `pip install` 命令代替厂商安装说明。

## 已盘点环境基线

下表用于把已有平台 Triton 环境与 Whisper 源码匹配起来。它记录的是环境盘点基线，不替代当前发布提交的全量 Whisper 验证；任何组件版本变化都必须重新执行完整验收。

| 平台环境 | Python | Torch/设备扩展 | Triton | backend | 当前证据 |
|---|---|---|---|---|---|
| NVIDIA A40 | 3.10 | `2.4.1+cu124` | `3.0.0` | `nvidia` | 已有历史全模型结果；发布前按当前模型集合重跑 |
| 华为昇腾 CANN 8.5 | 3.10 | Torch `2.5.1`、`torch_npu 2.5.1` | 厂商环境已提供，公开版本号待实机复核 | `ascend` | 环境与 kernel 已盘点；全模型待发布验收 |
| 寒武纪 MLU590 | 3.10.12 | Torch `2.9.1`、`torch_mlu 1.30.2+torch2.9.1` | `3.2.0` | `mlu` | 2026-07-23：DTW 直连通过；median 动态 kernel 编译失败并可回退 CPU；上游全模型用例在 CPU 通过，MLU 推理待验收 |
| 摩尔线程 S5000 | 3.10 | Torch `2.7.1`、配套 `torch_musa` | `3.1.0+musa1.4.6` | `mtgpu` | 环境与 timing kernel 已盘点；全模型待发布验收 |
| 海光/DTK兼容环境 | 3.10.16 | `2.4.1+das.opt1.dtk2504` | `3.0.0+das.opt3.dtk2504` | `amd` | DTK参考环境已盘点；目标海光设备仍须实机全模型验收 |
| 平头哥 PPU-ZW810E | 3.10 | Torch `2.6.0` | `3.2.0` | `nvidia`兼容 | 环境已盘点；全模型待发布验收 |

不要根据这张表单独升级某个包。实际安装时应使用平台已经成套提供的解释器、Torch、设备扩展和 Triton，再把 Whisper 以 `--no-deps` 方式装入该环境。

## 验证命令

环境检查：

```bash
python scripts/check_accelerator_env.py --platform <platform-key> \
  --json tests/results/<platform-key>/environment.json
```

统一全量测试：

```bash
python scripts/run_platform_tests.py --platform <platform-key>
```

runner 执行环境检查后运行 `pytest tests/ -v`。`test_timing.py` 同时包含直接调用 Triton kernel 的测试，Triton 失败后走 CPU fallback 不能让直连测试误报成功。集成测试覆盖当前提交中 `whisper.available_models()` 返回的所有模型，因此模型集合随上游版本变化而自动更新。

支持验收要求：环境检查通过、CPU 单测通过、两个 timing kernel 直连且与 CPU 结果等价、全模型集成测试通过、tiny 模型开启词级时间戳后端到端通过。

注意：当前上游 `tests/test_transcribe.py` 只根据 `torch.cuda.is_available()` 选择
CUDA 或 CPU，不读取 `WHISPER_TEST_DEVICE`。因此非 CUDA 平台即使设备探测成功，
该文件通过也只证明 CPU 转录路径。非 CUDA 平台的发布验收必须另行运行显式传入
厂商设备的全模型用例；后续应让转录集成测试复用 pytest 的统一 accelerator fixture，
并在测试摘要中记录每个模型的实际 `model.device`。

## 输出约定

上游 pytest 没有测试产物目录。CNPort runner 默认生成：

```text
tests/results/<platform>/<timestamp>/
├── command.json
├── environment.json
├── junit.xml
├── pytest.log
├── report.md
└── summary.json
```

这些逐次产物可能包含机器信息，默认由 `.gitignore` 排除。对外发布时只提交人工复核并脱敏的摘要，不提交模型缓存、原始日志或本机路径。
