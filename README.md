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
CPU 功能回退。发布“完整支持”前仍需让重写后的 kernel 与全模型转录测试显式在
`mlu` 设备执行通过。

当前开发分支已将 `median_kernel` 重写为普通的静态 Triton JIT kernel，窗宽通过
`tl.constexpr` 特化，不再修改 `JITFunction` 源码。上面的结果仍是最近一次已完成的
MLU 实机验证基线；在 MLU590 上重新完成 kernel 直连和端到端验收前，支持状态暂不
提前调整。

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

## Triton 测试镜像

CNPort 为各平台提供成套 Triton 验证镜像。镜像预置平台运行/编译环境和 Python 3.10，用于避免手工拼接不匹配的 Torch、Triton 与 SDK。请按下表选择与实机架构一致的镜像，不要混用不同平台镜像或使用 `latest` 标签。

| 平台 | 镜像（VPC） | 标签 | 架构 |
|---|---|---|---|
| NVIDIA | `open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-nvidia-a40-py310` | `v1.0.0-amd64` | `amd64` |
| 华为昇腾 | `open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-ascend-910b-py310` | `v1.0.0-arm64` | `arm64` |
| 海光 | `open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-hygon-bw1000-py310` | `v1.0.0-amd64` | `amd64` |
| 寒武纪 | `open-audio-native-registry.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-cambricon-mlu590-py310` | `v1.0.0-amd64` | `amd64` |
| 摩尔线程 | `open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-mtt-s5000-py310` | `v1.0.0-amd64` | `amd64` |
| 阿里平头哥 | `open-audio-native-registry.cn-beijing.cr.aliyuncs.com/audio_generation/triton-t-head-zw810e-py310` | `v1.0.0-amd64` | `amd64` |

说明：

- 表中的完整镜像地址为“镜像仓库 + `:` + 标签”，例如 NVIDIA 为
  `open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-nvidia-a40-py310:v1.0.0-amd64`。
- VPC 地址仅用于已接入对应阿里云 VPC 的机器；公网机器应把仓库域名替换为
  `open-audio-native-registry.cn-beijing.cr.aliyuncs.com`，仓库路径和标签保持不变。
- 拉取私有镜像前需要使用企业账号执行 `docker login`。测试命令不会记录账号信息。
- NVIDIA、海光和平头哥镜像以 CUDA/HIP 兼容设备名暴露运行时设备；这是平台工具链的既有行为，仍需按上表选择对应平台的镜像。
- 镜像只解决软件环境。宿主机仍需正确安装驱动，并确保 `/dev` 设备、共享库和内核模块对容器可见。

## 测试

### 原生测试框架

Whisper CNPort 直接复用上游 pytest 测试套件，并补充两个层次：

1. **上游 pytest 测试**：包含音频处理（`test_audio.py`）、分词器（`test_tokenizer.py`）、文本规范化（`test_normalizer.py`）、词级时间戳（`test_timing.py`）和端到端转录（`test_transcribe.py`）。
2. **CNPort Triton kernel 直连验证**：直接调用 `whisper/triton_ops.py` 中的 `median_kernel` 和 `dtw_kernel`，验证编译、启动和与 CPU 参考实现的等价性，防止 CPU fallback 掩盖平台 Triton 失败。

测试分四个层级：

| 层级 | 覆盖内容 | 是否下载模型 |
|---|---|---|
| `timing-kernel-probe` | 两个 Triton kernel 直连、参数化形状和 CPU 等价性 | 否 |
| `unit` | 音频、分词器、规范化器、timing 单元测试 | 否 |
| `unit+smoke` | unit + 一个 tiny 模型端到端冒烟 | 是 |
| `full-integration` | `whisper.available_models()` 全部模型端到端测试 | 是 |

### 使用镜像执行测试

以下步骤在宿主机上执行；`<repo-root>` 为 Whisper CNPort 仓库路径。

1. 登录并拉取平台镜像：

```bash
# 公网机器
docker login open-audio-native-registry.cn-beijing.cr.aliyuncs.com
# 匹配 VPC 的机器
docker login open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com

# 示例：NVIDIA A40
docker pull open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-nvidia-a40-py310:v1.0.0-amd64

# 示例：昇腾 910B（VPC 地址）
docker pull open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-ascend-910b-py310:v1.0.0-arm64
```

2. 挂载仓库、模型缓存和 Triton 缓存后启动容器。以下为 NVIDIA 通用示例：

```bash
docker run -it --rm \
  --gpus all \
  --shm-size 8g \
  -v <repo-root>:/workspace/whisper \
  -v <model-cache>:/workspace/model-cache \
  -v <triton-cache>:/workspace/triton-cache \
  -e XDG_CACHE_HOME=/workspace/model-cache \
  -e TRITON_CACHE_DIR=/workspace/triton-cache/cache \
  open-audio-native-registry-vpc.cn-beijing.cr.aliyuncs.com/hy_cnport/triton-nvidia-a40-py310:v1.0.0-amd64 \
  bash
```

昇腾、寒武纪、摩尔线程、海光等平台需要按厂商容器运行规范替换设备挂载，并按需设置
`ASCEND_HOME_PATH`、`NEUWARE_HOME`、`MUSA_HOME`、`ROCM_HOME` 等环境变量。以镜像内厂商脚本为准，不要从一个平台复制另一个平台的启动参数。

3. 在容器内安装当前源码并执行测试（镜像未内置测试依赖时）：

```bash
cd /workspace/whisper
python -m pip install numba numpy tqdm more-itertools tiktoken
python -m pip install -e . --no-deps
python -m pip install pytest scipy

python scripts/check_accelerator_env.py --platform auto
python scripts/run_platform_tests.py --platform nvidia-a40
```

平台参数使用 `nvidia-a40`、`ascend`、`hygon`、`cambricon`、`musa`、`t-head`。如果 auto 检测和平头哥/CUDA 兼容设备存在歧义，请显式传入平台参数。

4. 更快的分层执行方式：

```bash
# 只验证 timing 相关测试，不下载模型
python -m pytest tests/test_timing.py -v

# 单模型冒烟
python scripts/run_platform_tests.py --platform <platform-key> \
  -- tests/test_transcribe.py -k "tiny.en" -v

# 全模型集成测试
python scripts/run_platform_tests.py --platform <platform-key>
```

`scripts/run_platform_tests.py` 会在 `tests/results/<platform>/<timestamp>/` 生成 `environment.json`、`junit.xml`、`pytest.log`、`summary.json`、`command.json` 和 `report.md`。逐次日志和机器环境默认不提交 Git。

### 平台测试结果

下表按测试层级分别记录结果。`timing-kernel-probe` 记录两个 Triton kernel 的直连状态；`unit`、`unit+smoke`、`full-integration` 分别记录 pytest 通过数/收集数。`—` 表示该层级未单独执行，不能推断为失败。

| 平台 / 设备 | 配套镜像 | Torch / Triton | `timing-kernel-probe` | `unit` | `unit+smoke` | `full-integration` | 结论 |
|---|---|---|---|---|---|---|---|
| NVIDIA A40 | `triton-nvidia-a40-py310:v1.0.0-amd64`（digest `sha256:70bf80ef66b4`） | `2.4.1+cu124` / `3.0.0` | ✅ 22/22（2026-09-21） | ✅ 31/31（2026-09-21） | ✅ 32/32，含 `tiny.en` 1/1（2026-09-21） | ✅ 45/45，含全模型 14/14（2026-09-21） | Triton 改写与全量移植已验证 |
| 华为昇腾 910B | `triton-ascend-910b-py310:v1.0.0-arm64`（digest `sha256:b5e5c7757b23`） | `2.5.1` + `torch_npu 2.5.1` / `3.2.0` | ✅ 22/22（2026-09-21） | ✅ 31/31（2026-09-21） | ✅ 32/32，含 `tiny.en` 1/1（2026-09-21） | ✅ 45/45，含全模型 14/14（2026-09-21） | Triton 改写与全量移植已验证 |
| 海光 BW1000 | `triton-hygon-bw1000-py310:v1.0.0-amd64`（digest `sha256:189ff4891b41`） | `2.5.1`（HIP）/ `3.0.0` | ✅ 22/22（2026-09-21） | ✅ 31/31（2026-09-21） | ✅ 32/32，含 `tiny.en` 1/1（2026-09-21） | ✅ 45/45，含全模型 14/14（2026-09-21） | Triton 改写与全量移植已验证 |
| 寒武纪 MLU590 | `triton-cambricon-mlu590-py310:v1.0.0-amd64` | `2.9.1` + `torch_mlu` / `3.2.0` | ⚠️ DTW 通过；median 编译失败（2026-07-22） | ✅ 27/27（2026-07-22，包含在全量执行中） | — | ⚠️ 40/41；唯一失败为 median 直连用例，功能回退通过 | 部分完成，median 未通过直连 |
| 摩尔线程 S5000 | `triton-mtt-s5000-py310:v1.0.0-amd64` | `2.7.1` + `torch_musa` / `3.1.0` | ✅ DTW、median 通过（2026-06-02） | ✅ 25/25（2026-06-02） | ✅ 26/26，含 `tiny.en` 1/1（2026-06-02） | ✅ 39/39，含全模型 14/14（2026-06-02） | Triton 改写与全量移植已验证 |
| 阿里平头哥 PPU-ZW810E | `triton-t-head-zw810e-py310:v1.0.0-amd64` | `2.6.0` / `3.2.0` | ✅ DTW、median 通过（2026-07-22） | ✅ 27/27（2026-07-22） | ✅ 28/28，含 `tiny.en` 1/1（2026-07-22） | — | Triton 改写与端到端冒烟已验证 |

层级结果说明：

- `timing-kernel-probe`：直接调用 `whisper/triton_ops.py` 的两个 kernel，并检查与 CPU 参考实现等价；✅ 表示两个 kernel 均编译并在平台设备执行。
- `unit`：只包含音频、分词器、规范化器和 timing 单元测试；数字为通过数/收集数。
- `unit+smoke`：在 unit 基础上额外执行一个 `tiny.en` 端到端转录用例；总数包含 unit 用例。
- `full-integration`：执行当时 `whisper.available_models()` 的全部模型；数字为总通过数/总收集数。
- 海光和寒武纪的全量执行已覆盖 unit 用例，因此 unit 结果来自同一轮全量执行，而非独立重复执行。
- 表中的日期为对应层级最后一次实机执行时间。不同平台的上游模型集合和用例数量不同，不能直接横向比较 collected 数量。
- NVIDIA A40 四个层级已于 2026-09-21 使用 `triton-nvidia-a40-py310:v1.0.0-amd64`（digest `sha256:70bf80ef66b483a2434aa4bcc4d0633d4455b681aa95b162b732288eaf452a9c`）实机复验，`full-integration` 全量结果为 45/45 通过。
- 华为昇腾 910B3 已于 2026-09-21 使用 `triton-ascend-910b-py310:v1.0.0-arm64`（digest `sha256:b5e5c7757b23e729eb78781fd805a74888461668fc5cec5b5cf027be6818e729`）实机复验：`timing-kernel-probe` 22/22（DTW 与 median kernel 均在 `npu` 设备直连执行并通过 CPU 等价性检查），`unit` 31/31，`unit+smoke` 32/32（含 `tiny.en` 端到端冒烟），`full-integration` 全量 45/45 通过。执行环境为镜像内置的 CANN 8.5.0、Torch 2.5.1、`torch_npu 2.5.1`、Triton 3.2.0（backend `npu`）；本机 VPC 仓库域名不可解析，实际通过公网仓库地址使用本地已拉取的同 tag 镜像。上游 `test_transcribe` 用例仅在 `torch.cuda.is_available()` 为真时选择加速器，本次 14 个模型转录均在 CPU 执行，该结果不能作为 NPU Encoder/Decoder 全模型推理已验收的证据；NPU 上已验证的部分为 `timing` 两个 Triton kernel 的直连执行。
- 海光 BW1000 四个层级已于 2026-09-21 使用 `triton-hygon-bw1000-py310:v1.0.0-amd64`（digest `sha256:189ff4891b413b44484af90d0d27795e0a2a2b09e583b4172a42245588724abc`，torch `2.5.1` HIP `6.3.26045`、Triton `3.0.0`、设备 `BW200, UBB BW1000`×8）实机复验，四个层级全部通过：`timing-kernel-probe` 22/22、`unit` 31/31、`unit+smoke` 32/32（含 `tiny.en`）、`full-integration` 45/45（含全部 14 个模型名/别名）。
- 海光镜像的 ROCm 版 PyTorch 未内置 SDPA GPU kernel（缺少 `flash_attn_2_cuda*.so`，Aotriton flash attention 编译期禁用），fp16 张量调用 `scaled_dot_product_attention` 会直接抛 `RuntimeError` 而不是回退。本次复验在 `whisper/model.py` 中为 SDPA 增加了异常回退：首次调用失败后自动切换到等价的手动 attention 路径并关闭 SDPA，对数值结果无影响。该改动同样惠及其他 SDPA kernel 不完整的平台。
- 寒武纪 MLU590 的 40 个通过用例包含 median fallback 功能正确性，但不能证明 median kernel 已在 MLU 设备执行。

## 上游与许可证

- 上游项目：[openai/whisper](https://github.com/openai/whisper)
- 上游说明：[README_UPSTREAM.md](README_UPSTREAM.md)
- 许可证：[MIT License](LICENSE)

本项目未修改 Whisper 模型权重的许可证或格式。使用模型前请同时阅读上游模型卡和使用限制。
