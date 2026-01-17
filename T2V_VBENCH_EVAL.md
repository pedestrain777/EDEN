## T2V VBench 评估框架

### 1. 评估目标与指标
- **目标**：基于 VBench 基准全面评估文本生成视频（T2V）系统的质量与效率。
- **核心指标**：`PSNR↑`、`LPIPS↓`、`SSIM↑`、`VBench 综合分↑`、`Latency(s)↓`、`Speedup↑`。
- **补充指标**（可选）：CLIP 相似度、FVD、Aesthetic、Consistency、Motion、用户主观评分。

### 2. 目录结构建议
- `third_party/VBench/`：VBench 官方仓库（已上传）。
- `datasets/vbench/`：评估所需的 prompts、真实参考视频（如任务需要）。
- `results/vbench/`：
  - `videos/`：生成的 MP4 视频（统一编码 H.264、24fps、名称对齐 prompt id）。
  - `frames/`：如需逐帧指标（PSNR/SSIM），保存拆帧结果。
  - `metadata.csv`：记录 prompt、负面提示、种子、模型配置、生成时间等。
  - `logs/`：推理+评估脚本日志。
  - `scores/`：评估输出（JSON/CSV/Markdown 总结）。

### 3. 环境准备
1. 激活 T2V 项目运行环境，确认 `diffusers`、`pytorch`、`ffmpeg` 等已安装。
2. 进入 `third_party/VBench/`，按照官方 README 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 如评估脚本需要额外权重（CLIP、ConvNeXt、DenseNet 等），提前下载到 `third_party/VBench/checkpoints/`。
4. 确保评估机器具备充足存储与 GPU 资源，必要时配置 `ProxyJump` 访问。

### 4. 数据与生成准备
1. **Prompt 集**：从 `VBench` 提供的 JSON/Excel 提取，生成 `metadata.csv`，字段建议：
   ```
   prompt_id,prompt_text,negative_prompt,seed,steps,scheduler,frame_num,resolution
   ```
2. **真实参照**（若需 PSNR/SSIM）：收集目标参考视频或帧，放置于 `datasets/vbench/refs/`。
3. **生成脚本**：
   - 输入 metadata，批量调用 T2V 推理接口。
   - 输出 MP4 与推理日志；记录每条样本的耗时（Latency）。
   - 保存原始生成帧或在后处理阶段拆帧，便于图像级指标计算。
4. **文件命名**：建议 `prompt_id` 对齐，例如 `000123.mp4`。

### 5. 评估流程
1. **拆帧（若需要）**：
   ```bash
   ffmpeg -i results/vbench/videos/000123.mp4 results/vbench/frames/000123/%05d.png
   ```
   - 若只统计 VBench 评分，可直接使用视频文件。
2. **客观指标**：
   - PSNR/SSIM：对比生成帧与参考帧（分辨率、帧数需一致）。
   - LPIPS：同上，确保使用正确归一化。
3. **VBench 自动评估**：
   ```bash
   cd third_party/VBench
   python evaluate.py \
     --generated_dir /home/jiayu/hengyi_zhang/Hybrid-SD-main_for_v2i/results/vbench/videos \
     --prompt_file assets/prompts/VBench_Prompts.json \
     --result_dir /home/jiayu/hengyi_zhang/Hybrid-SD-main_for_v2i/results/vbench/scores \
     --metrics overall quality consistency motion \
     --batch_size 4 \
     --device cuda
   ```
   - 根据需求调整 `--metrics`、`--prompt_file`。
4. **效率指标**：
   - Latency：从推理脚本日志中统计平均/中位数耗时。
   - Speedup：与基准模型（如纯大模型）对比，保存在 `scores/speed.csv`。
5. **汇总**：将各指标写入 `results/vbench/scores/report.md`，附上采样视频链接或帧图。

### 6. 需要额外保存的内容
- `metadata.csv`：记录所有生成配置，便于复现。
- `logs/generation.log`、`logs/eval.log`：推理与评估的详细输出。
- （可选）`checkpoints_used.txt`：列出评估时使用的模型/LoRA/配置快照。
- 脚本版本号与 Git 提交信息。

### 7. 注意事项
- **帧数与分辨率对齐**：PSNR/SSIM/LPIPS 需与参考视频严格对齐，否则结果无效。
- **编码统一**：全程使用 `yuv420p`，避免某些评估器无法读取。
- **批量处理**：生成失败的样本要记录并补齐，VBench 评估过程中会忽略缺失文件。
- **资源监控**：保证 GPU 显存和磁盘空间（视频与拆帧数据较大）。
- **复现性**：固定随机种子，记录环境依赖，对每次实验进行版本标记。

### 8. 后续工作
- 编写自动化脚本（如 `bash scripts/vbench/run_eval.sh`）串联生成与评估。
- 对接可视化面板，展示六项核心指标的对比表格与曲线。
- 若需对比多模型/多步数配置，保持统一目录深度：`results/vbench/{exp_name}/...`。


