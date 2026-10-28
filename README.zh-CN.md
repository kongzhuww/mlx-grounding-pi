# mlx-grounding-pi

<p align="left">
  <a href="https://github.com/kongzhuww/mlx-grounding-pi/actions/workflows/ci.yml"><img src="https://github.com/kongzhuww/mlx-grounding-pi/actions/workflows/ci.yml/badge.svg" alt="CI 状态"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

> 基于 Apple MLX 的 [GroundingPI](https://huggingface.co/GroundingPI/GroundingPI) 实验性原生移植版本。

[English](README.md) | **简体中文**

短期目标是在 Apple Silicon（苹果芯片）上实现纯本地的 **图像 + 指令 -> 目标点/检测框 (points/boxes)** 视觉定位推理。不涉及任何机器人硬件、相机标定或机械控制。

**非官方社区实现；与 GroundingPI 官方无隶属或背书关系。**
初期的实验性基准与[目标不存在拒识微调研究实验室 (missing-target fine-tuning research lab)](https://github.com/kongzhuww/grounding-pi-rejection-lab) 保持独立。该实验项目目前尚未提供经过充分验证的训练后端或训练好的 Adapter 权重。

**当前状态：** 官方真实权重可在 16 GiB Mac M4 上本地运行，语言模型支持纯语言部分的 4-bit 或 8-bit 量化，视觉 Tower 及投影器 (Projector) 保持 BF16 精度。关键点定位、检测框和双目标定位已通过一组小型合成正例测试集；四个真实拍摄场景的结果也已完成人工视觉审查。**当前两种量化位宽下，在缺失目标（absent-target）场景中均会出现幻觉并报错。** 官方 CUDA BF16 基线在同样用例上也复现了该假阳性（误报）现象。在当前的 10 个测试用例中，8-bit 版本的输出在 8 个用例上与官方完整 Token 序列完全一致，而 4-bit 版本一致数量为 0；请注意 Token 序列一致**并不等同于定位准确率**。真实权重的 FP32 视觉诊断显示两端高度吻合，但 BF16 后端的数值微差以及基准评测准确率仍有待进一步解决。本项目目前属于实验性移植，非可直接投入生产的检测器。详见[实测评测报告](docs/mac-results.md)。

## 安装

环境要求：Apple Silicon 设备及 Python 3.11+（开发环境使用 Python 3.12）：

```sh
uv sync --locked
```

## 下载与转换

首先仅下载分词器及配置文件（不执行任何模型代码）：

```sh
uv run python -m mlx_grounding_pi.download --output models/groundingpi-hf
```

下载固定版本的原始官方权重（约 **9.69 GB**）：

```sh
uv run python -m mlx_grounding_pi.download \
  --output models/groundingpi-hf --weights
```

在 16 GiB 的 Mac 上，**语言部分 4-bit 量化** 是节省内存的首选方案。若希望在本阶段小型对比测试中更贴近官方输出，且内存充裕，建议使用 **8-bit 量化**（实测 MLX 显存峰值约 6.6 GB，非整机进程总内存）。视觉 Tower 及投影器均保持 BF16 精度。两种选项目前均未做标准 Benchmark 级别的准确率评定：

```sh
uv run python -m mlx_grounding_pi.convert \
  --source models/groundingpi-hf --output models/groundingpi-mlx-4bit \
  --bits 4 --group-size 64
# 备选项：--output models/groundingpi-mlx-8bit --bits 8 --group-size 64
```

转换过程单次仅读取一个张量，会校验源权重 Schema 以及官方公布的 SHA-256 校验和（若提供），并按安全大小分片保存。已存在的输出目录绝不会被静默覆盖；转换失败时会保留 `.partial-*` 临时目录供诊断分析。请确保磁盘预留充足可用空间：下载原始权重加 4-bit 转换需至少 15 GB，加 8-bit 转换需至少 18 GB；若同时导出多种位宽或不量化版本则需要更多空间。如需保留原始权重请指定 `--bits 0`，但无法保证原始无量化模型能完整载入 16 GiB 内存的 Mac。

## 本地图像目标定位推理

```sh
uv run python -m mlx_grounding_pi.generate \
  --model models/groundingpi-mlx-4bit \
  --image /path/to/image.jpg \
  --phrase "the red block" --task point \
  --output outputs/red-block
```

使用 `--task bbox` 可进行检测框定位；亦可使用 `--prompt` 替代 `--phrase` 传入完整的自定义指令文本。
输出内容为包含协议原始文本、有效性状态、0–999 归一化坐标及原图对应像素坐标的 JSON 数据；对于有效输出还会生成标注效果图 `prediction.png`。已存在的输出目录不会被覆盖。发生截断或格式非法的输出将被标记为无效（退出码 2），绝不会静默回退至 `(0,0)`。
**`valid` 仅代表输出符合完整的协议解析格式，并不证明目标一定存在或定位位置正确。** 当前量化模型在面对画面中缺失的黄色三角形时均会产生幻觉坐标。

推理过程纯本地单图贪心生成，一次处理一个请求。`--max-tokens` 默认为 256，`--max-context` 默认为 4096。可通过 `--max-image-tokens 256` 压缩默认图像 Token 配额，但这可能会影响定位精度；建议留空以保持原模型的图像处理器上限。
CLI 会保守地拒绝占用超过 GPU 推荐工作集 75% 的过大权重模型，且不会擅自修改系统内存阈值配置。

## 首次运行冒烟测试（开发验证）

`examples/synthetic_scene.png` 包含已知中心坐标的红色方块；其预期坐标与 30 像素允许误差记录于 `examples/synthetic_scene.json` 中。
运行 `scripts/first_run.py` 可等待下载任务完成、转换 4-bit 语言权重、对测试图进行推理并比对实际像素误差。遇到首个错误即刻中断。该测试不代表基准评测精度或官方完整权重的数值严格对齐。

对于后台运行的自动化流程，可监控 `models/first-run-status.json` 与 `models/first-run.log`。状态显示 `waiting_for_download` **不代表**成功；只有显示 `completed` 才表示权重转换、推理生成、严格格式解析及已知目标比对全部顺利通过。

## 复现开发评测

```sh
uv run python scripts/prepare_evaluation.py
uv run python -m mlx_grounding_pi.evaluation \
  --model models/groundingpi-mlx-4bit --cases examples/evaluation_cases.json \
  --output outputs/evaluation-new
```

该命令当前会以退出码 **2** 结束，因为缺失目标测试用例未能通过。五个正例合成场景能够通过；四个真实照片用例需要单独进行人工目视审查，无法直接通过自动化得分评判。照片数据为公开的文档测试固定素材，附带固定哈希值并存放在被忽略的 `outputs/` 目录中；评测运行器仅加载一次模型，分别独立记录各个用例的推理结果。

经授权的单次[官方 CUDA 参考环境评测](docs/cloud-reference.md)已在 20 分钟上限内完成并确认实例已关闭停止。其产出的 10 个用例 BF16 结果已与 Mac 上的两个量化版本进行了对比。在原版官方模型中，负例用例同样会出现误判；8-bit 版本下 8/10 完全一致的 Token 匹配度并不代表完全数值等价或 80% 准确率。详细的 BF16 特征微差与本地 FP32 诊断分析请参阅相关评测报告。

## 开发与验证

```sh
uv run pytest -q
uv run ruff check .
uv run python scripts/audit_checkpoint.py
# 可选的官方 PyTorch 对比测试；Torch 并非日常推理运行的依赖项：
TORCHDYNAMO_DISABLE=1 uv run --group reference pytest -q -s tests/test_reference_parity.py
```

官方参考快照已在 `upstream/manifest.json` 中锁定版本并记录 SHA-256 校验和。
官方发布的权重模型包含 **4,843,664,384 个参数、569 个张量与两个权重分片**。
Git 仅追踪元数据及源码快照；权重文件与推理生成的输出均已加入 Git 忽略清单。

更多内容请参阅 [发布说明](docs/release-notes.md)、[参与贡献指南](CONTRIBUTING.md)、[架构审计报告](docs/architecture.md)、[验证细节](docs/validation.md) 以及 [第三方开源声明](THIRD_PARTY_NOTICES.md)。源自 Kimi 的相关代码遵循额外的上游许可条款；整个模型及移植代码并非全盘单一适用 Apache-2.0 许可证。
