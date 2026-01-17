# VBench 评测内容迁移说明

## 概述

本文档说明从 `Hybrid-SD-main_for_v2i` 项目迁移到 `EDEN-main` 项目的 VBench 评测相关内容。

**迁移日期**: 2024年12月6日  
**源项目**: `/home/jiayu/hengyi_zhang/Hybrid-SD-main_for_v2i`  
**目标项目**: `/home/jiayu/hengyi_zhang/EDEN-main`

---

## 迁移内容清单

### 1. 评估脚本 (`evaluation/vbench/`)

从 `Hybrid-SD-main_for_v2i/evaluation/t2v_vbench/` 迁移到 `EDEN-main/evaluation/vbench/`

| 文件 | 功能说明 |
|------|---------|
| `dataset.py` | 数据集加载工具，提供 `VideoSample` 数据类和视频帧加载功能 |
| `metrics.py` | 计算客观指标（PSNR、SSIM、LPIPS），支持按 seed 分组统计 |
| `latency.py` | 聚合延迟统计信息，计算相对于基线的加速比 |
| `report.py` | 生成汇总报告，整合客观指标、VBench 分数和延迟统计 |
| `vbench_runner.py` | 封装 VBench 官方评估脚本调用，解析评估结果 |

**注意**: 路径已从 `evaluation/t2v_vbench/` 调整为 `evaluation/vbench/` 以匹配 EDEN-main 的目录结构。

---

### 2. 运行脚本 (`scripts/vbench/`)

从 `Hybrid-SD-main_for_v2i/scripts/vbench/` 迁移到 `EDEN-main/scripts/vbench/`

| 文件 | 功能说明 |
|------|---------|
| `run_eval.sh` | 完整评估流程脚本，依次执行：构建元数据、生成视频、提取帧、计算指标、VBench 评分、延迟统计、生成报告 |
| `run_generation.sh` | 视频生成脚本，调用 `tools/vbench/run_generation.py` |
| `run_generation_vbench_format.sh` | 生成符合 VBench 格式要求的视频 |

**已更新**: 脚本中的所有路径引用已从 `evaluation/t2v_vbench/` 更新为 `evaluation/vbench/`。

---

### 3. 配置文件 (`configs/`)

从 `Hybrid-SD-main_for_v2i/configs/` 迁移到 `EDEN-main/configs/`

| 文件 | 说明 |
|------|------|
| `vbench_eval.yaml` | 主评估配置文件，包含实验名称、prompt 文件路径、模型配置、评估指标等 |
| `vbench_40_10_10files.yaml` | 特定实验配置 |
| `vbench_50_0_10files.yaml` | 特定实验配置 |
| `vbench_baseline_50_0.yaml` | 基线模型配置 |
| `vbench_compare_40_10.yaml` | 对比实验配置 |

**配置文件结构**:
```yaml
exp_name: baseline_25_0
output_root: results/vbench
prompt_file: VBench-master/prompts/metadata/object_class.json
models: [...]
steps: "25,0"
num_frames: 49
resolution: {height: 480, width: 720}
guidance_scale: 6.0
fps: 8
calc_psnr_ssim_lpips: true
run_vbench_metrics: true
measure_latency: true
vbench_metrics: [overall, quality, consistency]
```

---

### 4. Prompt 文件 (`prompts/`)

从 `Hybrid-SD-main_for_v2i/prompts/` 迁移到 `EDEN-main/prompts/`

| 文件 | 说明 |
|------|------|
| `vbench_10files_top3.txt` | VBench 评估使用的 prompt 列表（10 个文件，top 3） |

---

### 5. 文档文件

从 `Hybrid-SD-main_for_v2i/` 根目录迁移到 `EDEN-main/` 根目录

| 文件 | 说明 |
|------|------|
| `VBENCH_使用指南.md` | VBench 完整使用指南，包含安装、评估维度、使用方法、Prompt 套件、评估流程等 |
| `T2V_VBENCH_README.md` | T2V VBench 评估开发指引，包含快速开始、评估流程、代码任务分解等 |
| `T2V_VBENCH_EVAL.md` | T2V VBench 评估说明文档 |

---

### 6. 工具脚本 (`tools/vbench/`)

**状态**: 已存在且内容一致，无需迁移

`tools/vbench/` 目录在两个项目中已存在且内容相同，包含：
- `build_metadata.py` - 构建 VBench prompts 列表
- `check_vbench_ready.py` - 检查 VBench 环境准备情况
- `extract_frames.py` - 从视频中提取帧
- `prepare_vbench_videos.py` - 准备符合 VBench 格式的视频
- `run_generation_vbench_format.py` - 生成 VBench 格式视频
- `run_generation.py` - 运行视频生成

---

## 目录结构对比

### 源项目结构 (Hybrid-SD-main_for_v2i)
```
Hybrid-SD-main_for_v2i/
├── evaluation/
│   └── t2v_vbench/          # 评估脚本
├── scripts/
│   └── vbench/              # 运行脚本
├── configs/
│   └── vbench_*.yaml        # 配置文件
├── prompts/
│   └── vbench_*.txt         # Prompt 文件
└── VBENCH_使用指南.md        # 文档
```

### 目标项目结构 (EDEN-main)
```
EDEN-main/
├── evaluation/
│   └── vbench/              # 评估脚本（已迁移，路径已调整）
├── scripts/
│   └── vbench/              # 运行脚本（已迁移，路径已更新）
├── configs/
│   └── vbench_*.yaml        # 配置文件（已迁移）
├── prompts/
│   └── vbench_*.txt         # Prompt 文件（已迁移）
├── tools/
│   └── vbench/              # 工具脚本（已存在，无需迁移）
├── VBench-master/           # VBench 官方仓库（已存在）
└── VBENCH_*.md              # 文档（已迁移）
```

---

## 主要调整

### 路径更新

所有脚本中的路径引用已从 `evaluation/t2v_vbench/` 更新为 `evaluation/vbench/`：

1. **`scripts/vbench/run_eval.sh`**
   - `evaluation/t2v_vbench/metrics.py` → `evaluation/vbench/metrics.py`
   - `evaluation/t2v_vbench/vbench_runner.py` → `evaluation/vbench/vbench_runner.py`
   - `evaluation/t2v_vbench/latency.py` → `evaluation/vbench/latency.py`
   - `evaluation/t2v_vbench/report.py` → `evaluation/vbench/report.py`

2. **评估脚本中的 Usage 注释**
   - 所有 Python 脚本文件头部的 Usage 说明已更新路径

---

## 使用方法

### 快速开始

1. **配置评估参数**
   ```bash
   # 编辑配置文件
   vim configs/vbench_eval.yaml
   ```

2. **运行完整评估流程**
   ```bash
   # 运行完整评估（包括生成、指标计算、VBench 评分）
   bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml
   
   # 限制样本数量进行测试
   bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml --limit 10
   
   # 干运行（仅验证路径，不实际执行）
   bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml --dry-run
   ```

3. **单独运行各个步骤**
   ```bash
   # 步骤 1: 构建元数据
   python3 tools/vbench/build_metadata.py --config configs/vbench_eval.yaml
   
   # 步骤 2: 生成视频
   bash scripts/vbench/run_generation.sh --config configs/vbench_eval.yaml
   
   # 步骤 3: 提取帧
   python3 tools/vbench/extract_frames.py --config configs/vbench_eval.yaml
   
   # 步骤 4: 计算客观指标
   python3 evaluation/vbench/metrics.py --config configs/vbench_eval.yaml
   
   # 步骤 5: VBench 评分
   python3 evaluation/vbench/vbench_runner.py --config configs/vbench_eval.yaml
   
   # 步骤 6: 延迟统计
   python3 evaluation/vbench/latency.py --config configs/vbench_eval.yaml
   
   # 步骤 7: 生成报告
   python3 evaluation/vbench/report.py --config configs/vbench_eval.yaml
   ```

### 查看结果

评估结果保存在 `results/vbench/<exp_name>/` 目录下：

```
results/vbench/<exp_name>/
├── videos/                  # 生成的视频文件
├── frames/                  # 提取的帧序列
├── metadata.csv             # 元数据
├── metrics/
│   ├── objective.json       # 客观指标（PSNR、SSIM、LPIPS）
│   ├── latency.json         # 延迟数据
│   └── latency_summary.json # 延迟汇总
├── vbench/
│   └── vbench_scores.json   # VBench 评分结果
└── report.md                # 汇总报告
```

---

## 依赖关系

### 必需依赖

1. **VBench 官方仓库**
   - 位置: `VBench-master/`（已存在于 EDEN-main 项目）
   - 需要安装 VBench 依赖和下载预训练权重

2. **Python 包**
   - `torch`, `torchvision`
   - `torchmetrics` (用于 PSNR/SSIM/LPIPS)
   - `yaml`
   - `PIL` (Pillow)
   - `numpy`

3. **系统工具**
   - `ffmpeg` (用于视频处理)

### 配置要求

- 确保 `VBench-master/` 目录存在且已正确安装
- 确保 `configs/vbench_eval.yaml` 中的路径配置正确
- 如需计算客观指标，需要提供参考视频/帧（`reference_root`）

---

## 注意事项

1. **路径一致性**: 所有脚本已更新为使用 `evaluation/vbench/` 路径，与 EDEN-main 的目录结构一致。

2. **配置文件**: 迁移的配置文件可能需要根据 EDEN-main 项目的实际情况进行调整，特别是：
   - 模型路径
   - 输出目录
   - Prompt 文件路径

3. **参考视频**: 如果配置了 `reference_root`，需要确保参考视频/帧已准备好，否则客观指标计算会失败。

4. **VBench 环境**: 首次运行 VBench 评估时，可能需要下载预训练权重，请确保网络连接正常或提前下载。

5. **磁盘空间**: 完整评估可能产生大量视频和帧数据，请确保有足够的磁盘空间。

---

## 相关文档

- **VBench 使用指南**: 查看 `VBENCH_使用指南.md` 了解 VBench 的详细使用方法
- **T2V VBench 开发指引**: 查看 `T2V_VBENCH_README.md` 了解评估流程和代码结构
- **VBench 官方文档**: 参考 `VBench-master/` 目录中的官方文档

---

## 迁移验证

### 验证清单

- [x] 评估脚本已迁移并路径已更新
- [x] 运行脚本已迁移并路径已更新
- [x] 配置文件已迁移
- [x] Prompt 文件已迁移
- [x] 文档文件已迁移
- [x] 所有路径引用已更新为 `evaluation/vbench/`
- [x] 工具脚本已确认存在且一致

### 测试建议

1. 运行干运行模式验证路径正确性：
   ```bash
   bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml --limit 2 --dry-run
   ```

2. 小样本测试：
   ```bash
   bash scripts/vbench/run_eval.sh --config configs/vbench_eval.yaml --limit 5
   ```

---

## 问题反馈

如果在使用迁移后的 VBench 评测功能时遇到问题，请检查：

1. 路径配置是否正确
2. 依赖是否已安装
3. VBench 环境是否准备就绪
4. 配置文件中的路径是否与实际目录结构匹配

---

**迁移完成日期**: 2024年12月6日  
**迁移工具**: 自动化脚本 + 手动路径调整

