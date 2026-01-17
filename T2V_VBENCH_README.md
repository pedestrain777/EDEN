## T2V VBench 评估开发指引

### 快速开始
> 目标：基于仓库内现有脚本，生成评测数据、跑客观指标、调用 VBench 官方评分，并汇总报告。

1. **准备环境**
   - 激活与你训练/推理一致的 Python 环境，确保已安装 `torch`、`diffusers`、`imageio`、`imageio-ffmpeg`、`tqdm` 等依赖。
   - 进入 `VBench-master`，执行 `pip install -r requirements.txt`，并按官方说明下载所需权重（首次运行下载较慢，可提前准备）。
   - 确保系统安装 `ffmpeg`（若无，可通过 `sudo apt install ffmpeg` 或环境包管理器安装）。

2. **配置评估参数**
   - 编辑 `configs/vbench_eval.yaml`：
     - `exp_name`：评测结果输出目录名。
     - `prompt_file`：使用的 VBench prompt 列表（默认指向 `VBench-master/prompts/metadata/object_class.json`，可自定义）。
     - `reference_root`：客观指标使用的参考帧根目录。。
     - `models`、`steps`、`num_frames`、`resolution`、`guidance_scale`、`fps`：与生成脚本保持一致。
     - `baseline_latency`：如需计算 Speedup，填入基线模型的人均耗时（秒）；否则保持 `null`。
     - `vbench_metrics`：控制 VBench 评分维度（默认 `overall, quality, consistency`）。

3. **首次试跑（可选）**
   - 在根目录执行：
     ```bash
     bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml --limit 2 --dry-run
     ```
   - 该命令会串联全部步骤，但以占位数据（dry-run）运行，便于检查目录结构、配置路径是否正确。运行结束后，可在 `results/vbench/<exp_name>/` 下看到生成的视频、frames/、metrics/、vbench/ 等目录以及汇总 `report.md`（指标为 N/A）。

4. **正式评测**
   - 准备好参考帧：将真实基准视频拆帧后按 `prompt_<id>/00000.png` 命名放入 `reference_root`（脚本会与生成帧逐帧对齐）。
   - 如需减少样本，可继续带 `--limit N`；否则省略。
   - **生成真实视频**：
     ```bash
     bash scripts/vbench/run_generation.sh --config configs/vbench_eval.yaml
     ```
     可以先测试小样本 `--limit 10`，确认模型可正常生成。
   - **拆帧并计算指标**（若未使用一键脚本，可分别执行）：
     ```bash
     python3 tools/vbench/extract_frames.py --config configs/vbench_eval.yaml
     python3 evaluation/t2v_vbench/metrics.py --config configs/vbench_eval.yaml
     python3 evaluation/t2v_vbench/vbench_runner.py --config configs/vbench_eval.yaml
     python3 evaluation/t2v_vbench/latency.py --config configs/vbench_eval.yaml
     python3 evaluation/t2v_vbench/report.py --config configs/vbench_eval.yaml
     ```
   - **一键流程**（推荐）：
     ```bash
     bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml
     ```
     如需限制样本数量：`--limit 100`；如需重跑 dry-run：再加 `--dry-run`。

5. **结果检查**
   - 视频：`results/vbench/<exp_name>/videos/prompt_*/0.mp4`
   - 拆帧：`results/vbench/<exp_name>/frames/prompt_*/00000.png`
   - 客观指标：`results/vbench/<exp_name>/metrics/objective.json`
   - VBench 评分：`results/vbench/<exp_name>/vbench/vbench_scores.json`
   - 延迟统计：`results/vbench/<exp_name>/metrics/latency_summary.json`
   - 汇总报告：`results/vbench/<exp_name>/report.md`（若指标缺失会显示 `N/A`，请确认对应步骤是否完成）

6. **常见问题**
   - `ModuleNotFoundError: torch`：确认当前环境已安装 PyTorch；推理脚本需 GPU。
   - `imageio` 或 `ffmpeg` 缺失：参照第 1 步安装依赖。
   - VBench 下载报错：进入 `VBench-master` 内重试 `pip install -r requirements.txt`，并查看 README 里的权重下载指引，可提前设置国内镜像。
   - `reference_root` 找不到：确保提供真实参考帧目录，脚本会严格校验，避免客观指标误算。
   - 需注意磁盘空间：全量评测（>1k 视频）可能占用数十 GB，可在 `store_frames` 为 `false` 时跳过拆帧步骤，或在跑完客观指标后手动清理 `frames/`。

### 目标概述
- 构建一套基于 `VBench` 的文本生成视频评估流程，覆盖质量、语义一致性与性能指标。
- 与现有 T2I 评估逻辑保持一致的工程风格，便于复用监控与比对工具。
- 输出可复现的评估脚本、数据记录与报告，以支撑多模型、多步数组合对比。

### 评估流程
- **准备评估集合**：解析 `VBench` 官方 prompts、映射到待评估配置，生成 `metadata.csv`。
- **批量视频生成**：调用 `examples/hybrid_sd/hybrid_video.py`，补充参数封装，产出统一命名的 MP4 与帧数据。
- **客观指标计算**：对齐参考视频或基线输出，计算 `PSNR`、`SSIM`、`LPIPS` 等（复用 T2I 中 torchmetrics 逻辑）。
- **VBench 评分**：调用 `third_party/VBench` 脚本获取 `VBench 综合分` 及子指标。
- **性能统计**：收集推理时延、推理日志，计算 `Latency` 与 `Speedup`。
- **结果汇总**：整理为 JSON/CSV/Markdown，支持与 T2I/T2V 其他实验对比。

### 代码与任务分解
| 代码路径 | 任务 | 说明 |
| --- | --- | --- |
| `scripts/vbench/run_generation.sh` | 封装批量生成流程 | 参考 `scripts/hybrid_sd/generate_dpm_eval.sh`，设置环境变量、批量运行 `hybrid_video.py`，输出到 `results/vbench/<exp_name>/videos`。 |
| `scripts/vbench/run_eval.sh` | 串联指标计算 | 顺序调用帧拆分、客观指标脚本、VBench 评估、性能统计，生成统一日志。 |
| `tools/vbench/build_metadata.py` | 构建 VBench prompts 列表 | 解析 `third_party/VBench/assets/prompts/*.json`，输出 `metadata.csv`（包含 prompt、负面 prompt、种子、期望帧数）。 |
| `evaluation/t2v_vbench/dataset.py` | 加载生成结果与参考视频 | 提供按 `prompt_id` 读取帧序列的 Dataset，支持后续指标模块。 |
| `evaluation/t2v_vbench/metrics.py` | 实现 PSNR/SSIM/LPIPS 计算 | 复用 `evaluation/evaluation.py` 中的 torchmetrics 配置，增加视频帧维度对齐逻辑与批量处理。 |
| `evaluation/t2v_vbench/vbench_runner.py` | 调用第三方评分 | 封装 `subprocess` 调用 `third_party/VBench/evaluate.py`，并解析输出 JSON/CSV。 |
| `evaluation/t2v_vbench/report.py` | 汇总指标 | 读取上述各模块输出，生成 Markdown/CSV/JSON 表格（含 `PSNR`、`LPIPS`、`SSIM`、`VBench`、`Latency`、`Speedup`）。 |
| `results/vbench/README.md` | 结果说明模板 | 指导如何理解各指标、文件夹结构与可视化示例。 |
| `configs/vbench_eval.yaml` | 参数配置 | 统一管理模型路径、帧数、分辨率、步数组合、批量大小、VBench 指标选择等。 |

### 产物管理
- `results/vbench/<exp_name>/videos/`：最终 MP4，命名遵循 `prompt_{id}.mp4`。
- `results/vbench/<exp_name>/frames/`：可选，拆帧后的 PNG 序列，用于图像级指标。
- `results/vbench/<exp_name>/metadata.csv`：记录 prompt、负面 prompt、模型配置、随机种子、生成时间。
- `results/vbench/<exp_name>/metrics/*.json`：客观指标结果。
- `results/vbench/<exp_name>/vbench/*.json`：第三方评分结果。
- `results/vbench/<exp_name>/logs/*.log`：生成与评估日志，包含单样本耗时。
- `results/vbench/<exp_name>/report.md`：汇总表格（可由 `report.py` 自动生成）。

### 注意事项
- 统一视频编码格式为 `H.264 + yuv420p`，帧率与帧数必须与评估约定一致，否则 VBench 评分会失败。
- 计算 `PSNR/SSIM/LPIPS` 时必须与参考视频逐帧对齐；若缺少真实参照，可对比基线模型输出，需明确标记基线来源。
- 批量生成时需确保 `hybrid_video.py` 的 `guidance_scale`、步数组合、模型切换逻辑与评估配置一致。
- VBench 运行依赖的模型较多，首次执行建议预下载权重并设置缓存目录，避免脚本自动下载导致网络超时。
- 性能统计需记录 GPU 型号、批处理大小；计算 `Speedup` 时请指定参照模型（如 `CogVideoX-5B` 单模型推理）。
- 评估期间目录体积增大迅速，定期清理中间帧或使用软链接指向大容量磁盘。

### 需用户确认
- 需确认 `PSNR/SSIM/LPIPS` 对比对象：是真实参考视频还是某个 baseline（请提供路径或生成方式）。
- 需确认 VBench 任务子集与指标列表（是否采样全部 1700+ prompts，或限定在如 `M-VBench-Validation`）。
- 需确认 `Latency` 与 `Speedup` 的统计窗口（单个 prompt 平均值或整体总耗时）以及 Speedup 的对照基线。
- 需确认是否需要保存中间 latents 或仅保存视频与帧（当前设计为可选，通过配置控制）。
- 需确认评估机器 GPU/CPU 资源与可接受的最大并发，决定脚本中的 `batch_size`、`num_workers`。

### 后续拓展
- 结合 `wandb` 或内部监控系统，自动上传评估曲线和样例视频。
- 扩展 `report.py` 支持对比多次实验，自动生成雷达图或分布图。
- 接入 CI/CD，提供最小化评估集，每次模型更新自动运行关键指标回归。
- 与 T2I 评估共享公共工具（如 `MeanMetric` 封装、日志解析器），减少重复代码。


