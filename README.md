# Whisper CNPort

Whisper CNPort 基于 [OpenAI Whisper](https://github.com/openai/whisper)，目标是在保持 Whisper Python API、命令行接口、模型推理实现和模型格式不变的前提下，在 NVIDIA、华为昇腾、寒武纪、摩尔线程、海光和阿里平头哥平台完成编译与功能验证。

本仓库当前跟踪的上游基线为 `v20250625`。OpenAI 原始 README 保存在 [README_UPSTREAM.md](README_UPSTREAM.md)，许可证见 [LICENSE](LICENSE)。

## 主要兼容改动

- Whisper 的 Encoder/Decoder 仍由各平台厂商提供的 PyTorch 后端执行。
- 词级时间戳使用的 `dtw_kernel` 和 `median_kernel` 保持一份 Triton 源码，由当前环境中的平台 Triton JIT 编译。
- 仅 `whisper/timing.py` 对非 CPU 张量尝试执行这两个 Triton kernel；Whisper 不新增厂商模型推理后端，也不替用户选择厂商设备。
- Triton 不再作为 Whisper 的通用安装依赖。各平台必须预先安装与本机 PyTorch、SDK 和设备后端匹配的 Triton 发行版。
- Triton 不可用或编译失败时，词级时间戳计算会给出警告并回退到较慢的 CPU 实现。

## 支持平台

| 平台 | 常见 Torch 设备 | Triton backend/产物 | 当前支持状态 |
|---|---|---|---|
| NVIDIA | `cuda` | NVIDIA / cubin | PyTorch CUDA + PyPI/配套 Triton |
| 华为昇腾 | `npu` | Ascend / npubin | `torch_npu` + 昇腾 Triton |
| 寒武纪 | `mlu` | MLU / cnbin | **部分支持/验证中**：DTW kernel 可在 MLU 执行；median kernel 失败时自动回退 CPU |
| 摩尔线程 | `musa` | MTGPU / mubin | `torch_musa` + MUSA Triton |
| 海光 | `cuda` 或 HIP 兼容设备 | AMD/HIP / hsaco | 厂商 PyTorch + 配套 ROCm/HIP Triton |
| 阿里平头哥 | CUDA 兼容设备 | 平台工具链产物 | PPU SDK、厂商 PyTorch 和配套 Triton |

“支持”要求环境检查、Triton kernel 直连等价测试和当前版本的全模型集成测试全部通过。具体版本组合与验收方法见 [多平台安装与验证指南](docs/multi-platform.md)。

### 寒武纪 MLU590 当前限制

2026-07-23 在 MLU590-M9、Python 3.10.12、Torch 2.9.1、
`torch_mlu 1.30.2+torch2.9.1`、MLU Triton 3.2.0 组合上的实测结果为
`40 passed, 1 failed`：

- `dtw_kernel` 的 MLU 直连测试和 CPU 等价性测试通过。
- 动态生成的 `median_kernel` 在 MLU Triton 编译缓存 metadata 时失败，错误为
  `TypeError: vars() argument must have __dict__ attribute`。这是当前 MLU Triton
  编译/序列化路径与 Whisper 动态 JIT kernel 的兼容问题，不是中值滤波数值误差。
- 对外 API 的中值滤波功能测试可通过，因为编译失败会发出 `RuntimeWarning` 并回退到
  CPU；这只表示功能可用，不表示 median kernel 已在 MLU 上执行，且词级时间戳会有
  设备到 CPU 的数据传输和性能损失。
- 上游全模型转录用例的 14 个模型名/别名均通过，但当前上游用例只在
  `torch.cuda.is_available()` 为真时选择加速器，本次实际在 CPU 执行。因此该结果不能
  作为 MLU Encoder/Decoder 全模型推理已验收的证据。

基于上述结果，本仓库目前不宣称寒武纪平台已完成全栈移植。可支持的范围是：上游
Python API、CPU 推理、MLU 张量上的 DTW Triton 路径，以及 median kernel 失败后的
CPU 功能回退。发布“完整支持”前仍需修复或升级 MLU Triton 的动态 kernel 编译问题，
并让全模型转录测试显式在 `mlu` 设备执行通过。

## 安装

不要在多个厂商平台之间复用同一个 Python 环境。建议每个平台使用独立 venv 或容器，并使用同一个 Python 解释器完成安装和测试。

```bash
# 1. 按厂商文档安装驱动/SDK、PyTorch、设备扩展和配套 Triton
# 2. 安装 Whisper 的其余依赖，避免 pip 改写厂商 Torch/Triton
python -m pip install numba numpy tqdm more-itertools tiktoken
python -m pip install -e . --no-deps

# 开发和测试依赖
python -m pip install pytest scipy
```

Whisper 还需要系统命令 `ffmpeg`：

```bash
ffmpeg -version
python scripts/check_accelerator_env.py --platform auto
```

## 使用

Python API 与上游保持一致：

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

非 CUDA 平台由调用方按厂商 PyTorch 文档显式传入设备，例如
`whisper.load_model("turbo", device="<torch-device>")`。CNPort 不在产品代码中维护
NPU、MLU、MUSA 等模型推理设备的探测或调度逻辑。

命令行示例：

```bash
whisper audio.wav --model turbo --output_dir outputs
```

这里的 `--output_dir` 仅控制转录文件输出，默认是当前目录；它不是 pytest 测试报告目录。

## 测试

上游 pytest 默认只向终端输出，不创建统一报告目录。CNPort 提供统一 runner：

```bash
python scripts/run_platform_tests.py --platform nvidia-a40
```

默认产物位于：

```text
tests/results/<platform>/<YYYYMMDDHHMMSS>/
```

逐次日志和机器环境信息默认不提交 Git。发布支持声明前，应为六个平台分别保留一份经过脱敏的验证摘要。

## 上游与许可证

- 上游项目：[openai/whisper](https://github.com/openai/whisper)
- 上游说明：[README_UPSTREAM.md](README_UPSTREAM.md)
- 许可证：[MIT License](LICENSE)

本项目未修改 Whisper 模型权重的许可证或格式。使用模型前请同时阅读上游模型卡和使用限制。
