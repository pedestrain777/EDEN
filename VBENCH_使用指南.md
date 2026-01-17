# VBench 使用指南 - 全面总结

## 一、VBench 概述

### 1.1 版本体系
- **VBench-1.0**：基础版本，评估 16 个维度，适用于短视频（< 5 秒）
- **VBench++**：扩展版本，支持 T2V 和 I2V，包含信任度评估
- **VBench-2.0**：最新版本，评估 18 个维度，专注于内在真实性（intrinsic faithfulness）
- **VBench-Long**：专门评估长视频（≥ 5 秒）
- **VBench-I2V**：专门评估图像到视频（Image-to-Video）模型

### 1.2 核心特点
- **多维度评估**：将视频生成质量分解为多个明确定义的维度
- **标准化 Prompt 套件**：为每个维度精心设计测试用例
- **自动化评估方法**：每个维度都有专门的评估方法
- **人类感知对齐**：评估结果与人类感知高度一致

---

## 二、安装方法

### 2.1 VBench-1.0 安装
```bash
# 方式 1：通过 pip 安装（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install vbench

# 方式 2：通过 git clone
git clone https://github.com/Vchitect/VBench.git
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install VBench

# 安装 detectron2（某些维度需要）
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

### 2.2 VBench-2.0 安装
```bash
conda create -n vbench2 python=3.10 -y
conda activate vbench2
conda install psutil
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1
pip install -r requirement.txt
pip install retinaface_pytorch==0.0.8 --no-deps
cd vbench2/third_party/Instance_detector
pip install -e .
cd ../../..
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html --no-cache-dir
```

### 2.3 下载预训练权重
权重默认存放在 `~/.cache/vbench` 或 `~/.cache/vbench2`。每个模型目录下有 `download.sh` 或 `model_path.txt` 文件，按指引下载即可。

---

## 三、评估维度

### 3.1 VBench-1.0 的 16 个维度

#### 技术质量维度（Quality Score）
1. **subject_consistency**（主体一致性）：评估视频中主体对象在时间上的连续性
2. **background_consistency**（背景一致性）：评估背景场景的稳定性
3. **temporal_flickering**（时间闪烁）：检测视频中的闪烁和抖动
4. **motion_smoothness**（运动平滑度）：评估运动的流畅性
5. **dynamic_degree**（动态程度）：评估视频的动态变化程度
6. **aesthetic_quality**（美学质量）：评估视频的视觉美感
7. **imaging_quality**（成像质量）：评估视频的像素级质量

#### 语义维度（Semantic Score）
8. **object_class**（对象类别）：评估模型生成特定对象类别的能力
9. **multiple_objects**（多对象）：评估同时生成多个对象的能力
10. **human_action**（人类动作）：评估生成人类动作的能力
11. **color**（颜色）：评估颜色理解和生成能力
12. **spatial_relationship**（空间关系）：评估对象间空间关系的理解
13. **scene**（场景）：评估场景生成能力
14. **temporal_style**（时间风格）：评估时间相关风格的生成
15. **appearance_style**（外观风格）：评估外观风格的生成
16. **overall_consistency**（整体一致性）：评估整体语义一致性

### 3.2 VBench-2.0 的 18 个维度

#### 创造力（Creativity）
- **Diversity**（多样性）
- **Composition**（构图）

#### 常识推理（Commonsense）
- **Motion Rationality**（运动合理性）
- **Instance Preservation**（实例保持）

#### 可控性（Controllability）
- **Dynamic Spatial Relationship**（动态空间关系）
- **Dynamic Attribute**（动态属性）
- **Motion Order Understanding**（运动顺序理解）
- **Human Interaction**（人类交互）
- **Complex Landscape**（复杂景观）
- **Complex Plot**（复杂情节）
- **Camera Motion**（相机运动）

#### 人类保真度（Human Fidelity）
- **Human Anatomy**（人体解剖）
- **Human Identity**（人类身份）
- **Human Clothes**（人类服装）

#### 物理真实性（Physics）
- **Mechanics**（力学）
- **Thermotics**（热力学）
- **Material**（材料）
- **Multi-View Consistency**（多视角一致性）

---

## 四、使用方法

### 4.1 评估标准 Prompt 套件

#### 命令行方式
```bash
# 评估单个维度
vbench evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION

# 示例
vbench evaluate --videos_path "sampled_videos/lavie/human_action" --dimension "human_action"
```

#### Python 方式
```python
from vbench import VBench

my_VBench = VBench(device, "vbench/VBench_full_info.json", "evaluation_results")
my_VBench.evaluate(
    videos_path = "sampled_videos/lavie/human_action",
    name = "lavie_human_action",
    dimension_list = ["human_action"],
)
```

### 4.2 评估自定义视频

#### 支持自定义的维度（VBench-1.0）
- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `dynamic_degree`
- `aesthetic_quality`
- `imaging_quality`

#### 支持自定义的维度（VBench-2.0）
- `Human_Anatomy`
- `Human_Identity`
- `Human_Clothes`
- `Diversity`（需要至少 20 个视频，命名格式：`prompt-index.mp4`，index 从 0 到 19）
- `Multi-View_Consistency`

#### 使用方法
```bash
# 命令行
python evaluate.py \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=custom_input

# 或使用 vbench 命令
vbench evaluate \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=custom_input
```

### 4.3 多 GPU 评估
```bash
# 使用 torchrun
torchrun --nproc_per_node=${GPUS} --standalone evaluate.py ...args...

# 或使用 vbench 命令
vbench evaluate --ngpus=${GPUS} ...args...
```

### 4.4 评估特定内容类别
```bash
vbench evaluate \
    --videos_path $VIDEO_PATH \
    --dimension $DIMENSION \
    --mode=vbench_category \
    --category=$CATEGORY
```

---

## 五、Prompt 套件

### 5.1 Prompt 文件结构

#### 按维度组织（prompts/prompts_per_dimension/）
- 每个维度有约 100 个 prompts
- 文件命名：`{dimension}.txt`
- 合并文件：`prompts/all_dimension.txt`（包含所有维度的 prompts）

#### 按类别组织（prompts/prompts_per_category/）
- 8 个内容类别：`Animal`, `Architecture`, `Food`, `Human`, `Lifestyle`, `Plant`, `Scenery`, `Vehicles`
- 每个类别 100 个 prompts
- 合并文件：`prompts/all_category.txt`

#### 元数据（prompts/metadata/）
- `object_class.json`：对象类别标签
- `multiple_objects.json`：多对象信息
- `color.json`：颜色信息
- `spatial_relationship.json`：空间关系信息
- `appearance_style.json`：外观风格信息

### 5.2 视频采样规范

#### 采样规则
1. **每个 prompt 采样 5 个视频**（`temporal_flickering` 维度采样 25 个）
2. **使用随机种子**：确保多样性，但过程可复现
3. **命名格式**：`$prompt-$index.mp4`，`$index` 取值为 `0, 1, 2, 3, 4`

#### 示例目录结构
```
sampled_videos/
├── A 3D model of a 1800s victorian house.-0.mp4
├── A 3D model of a 1800s victorian house.-1.mp4
├── A 3D model of a 1800s victorian house.-2.mp4
├── A 3D model of a 1800s victorian house.-3.mp4
├── A 3D model of a 1800s victorian house.-4.mp4
└── ...
```

### 5.3 各维度使用的 Prompt 套件

| 维度 | Prompt 套件 | Prompt 数量 |
| :---: | :---: | :---: |
| `subject_consistency` | `subject_consistency` | 72 |
| `background_consistency` | `scene` | 86 |
| `temporal_flickering` | `temporal_flickering` | 75 |
| `motion_smoothness` | `subject_consistency` | 72 |
| `dynamic_degree` | `subject_consistency` | 72 |
| `aesthetic_quality` | `overall_consistency` | 93 |
| `imaging_quality` | `overall_consistency` | 93 |
| `object_class` | `object_class` | 79 |
| `multiple_objects` | `multiple_objects` | 82 |
| `human_action` | `human_action` | 100 |
| `color` | `color` | 85 |
| `spatial_relationship` | `spatial_relationship` | 84 |
| `scene` | `scene` | 86 |
| `temporal_style` | `temporal_style` | 100 |
| `appearance_style` | `appearance_style` | 90 |
| `overall_consistency` | `overall_consistency` | 93 |

---

## 六、评估流程

### 6.1 完整评估流程

1. **准备 Prompt 套件**
   - 从 `prompts/` 目录选择或使用标准套件

2. **生成视频**
   - 按照采样规范生成视频
   - 确保命名格式正确

3. **运行评估**
   ```bash
   # 评估所有维度
   bash evaluate.sh
   
   # 或评估单个维度
   vbench evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION
   ```

4. **处理静态视频过滤**（仅 `temporal_flickering` 维度）
   ```bash
   python static_filter.py --videos_path $VIDEOS_PATH
   ```

5. **查看结果**
   - 结果保存在 `evaluation_results/` 目录

### 6.2 VBench-2.0 评估流程

```bash
# 评估所有 18 个维度（单 GPU，不推荐）
bash evaluate.sh --max_parallel_tasks 1

# 评估所有 18 个维度（多 GPU，推荐）
bash evaluate.sh --max_parallel_tasks 8
```

---

## 七、提交到排行榜

### 7.1 准备提交文件

```bash
# 打包评估结果
cd evaluation_results
zip -r ../evaluation_results.zip .

# [可选] 计算总分
python scripts/cal_final_score.py --zip_file {path_to_evaluation_results.zip} --model_name {your_model_name}
```

### 7.2 提交方式

1. **选项 1**：VBench 团队负责采样和评估（适合开源和闭源模型）
2. **选项 2**：自己采样，VBench 团队评估（通过 [Google Form](https://forms.gle/wHk1xe7ecvVNj7yAA) 提交）
3. **选项 3**：自己采样和评估（提交 `eval_results.zip` 到 [HuggingFace Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)）

### 7.3 分数计算

#### VBench-1.0 总分计算
1. **归一化**：每个维度分数归一化到 [0, 1]
2. **Quality Score**：以下维度的加权平均
   - `subject_consistency`, `background_consistency`, `temporal_flickering`, `motion_smoothness`, `aesthetic_quality`, `imaging_quality`, `dynamic_degree`
3. **Semantic Score**：以下维度的加权平均
   - `object_class`, `multiple_objects`, `human_action`, `color`, `spatial_relationship`, `scene`, `appearance_style`, `temporal_style`, `overall_consistency`
4. **Total Score** = w1 × Quality Score + w2 × Semantic Score

#### VBench-2.0 总分计算
1. **Creativity Score** = (`Diversity` + `Composition`) / 2
2. **Commonsense Score** = (`Motion Rationality` + `Instance Preservation`) / 2
3. **Controllability Score** = (`Dynamic Spatial Relationship` + `Dynamic Attribute` + `Motion Order Understanding` + `Human Interaction` + `Complex Landscape` + `Complex Plot` + `Camera Motion`) / 7
4. **Human Fidelity Score** = (`Human Anatomy` + `Human Identity` + `Human Clothes`) / 3
5. **Physics Score** = (`Mechanics` + `Thermotics` + `Material` + `Multi-View Consistency`) / 4
6. **Total Score** = 0.2 × (Creativity + Commonsense + Controllability + Human Fidelity + Physics)

---

## 八、特殊版本使用

### 8.1 VBench-Long（长视频评估）

#### 特点
- 适用于视频长度 ≥ 5 秒
- 使用 Slow-Fast 方法评估时间一致性
- 自动进行视频分割

#### 使用方法
```bash
# 评估标准 Prompt 套件
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension $dimension \
    --mode 'long_vbench_standard' \
    --dev_flag

# 评估自定义视频
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension $dimension \
    --mode 'long_custom_input' \
    --dev_flag

# 自动评估所有维度
sh vbench2_beta_long/evaluate_long.sh $VIDEOS_PATH
```

### 8.2 VBench-I2V（图像到视频评估）

#### 特点
- 提供多尺度、多宽高比的图像套件
- 自适应宽高比，适应不同模型的默认分辨率
- 高分辨率图像（主要 4K 及以上）

#### 使用方法
```bash
# 下载图像套件
pip install gdown
sh vbench2_beta_i2v/download_data.sh

# 评估（参考 VBench-1.0 的使用方法）
```

---

## 九、注意事项

### 9.1 视频要求
- **VBench-1.0**：视频长度 < 5 秒
- **VBench-Long**：视频长度 ≥ 5 秒
- **帧率**：通常为 8 fps
- **分辨率**：根据模型和评估维度而定

### 9.2 命名规范
- 标准评估：`$prompt-$index.mp4`
- 自定义评估：无特殊要求（`Diversity` 维度除外）

### 9.3 静态视频过滤
- 仅 `temporal_flickering` 维度需要
- 在评估前运行 `static_filter.py`

### 9.4 多 GPU 使用
- 建议使用多 GPU 加速评估
- 使用 `torchrun` 或 `--ngpus` 参数

---

## 十、常见问题

### Q1: 如何选择评估维度？
A: 根据你的需求选择：
- **全面评估**：使用 `all_dimension.txt` 评估所有维度
- **特定维度**：选择对应的 prompt 文件
- **快速测试**：选择几个关键维度

### Q2: 必须使用标准 Prompt 套件吗？
A: 不一定。可以使用自定义视频和 prompts，但：
- 标准套件确保公平对比
- 自定义评估结果不能直接与排行榜对比

### Q3: 评估需要多长时间？
A: 取决于：
- 视频数量和长度
- 评估维度数量
- GPU 数量和性能
- 通常每个维度需要几分钟到几小时

### Q4: 如何理解评估结果？
A: 
- 每个维度分数范围不同，需要查看具体维度的说明
- 总分是多个维度的加权平均
- 建议查看各维度详细分数，而非只看总分

---

## 十一、相关资源

- **项目主页**：https://vchitect.github.io/VBench-project/
- **GitHub 仓库**：https://github.com/Vchitect/VBench
- **排行榜**：https://huggingface.co/spaces/Vchitect/VBench_Leaderboard
- **数据集下载**：https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X
- **论文**：
  - VBench-1.0: https://arxiv.org/abs/2311.17982
  - VBench++: https://arxiv.org/abs/2411.13503
  - VBench-2.0: https://arxiv.org/abs/2503.21755

---

## 十二、总结

VBench 是一个全面的视频生成模型评估框架，提供了：
1. **多维度评估**：从技术质量到语义理解
2. **标准化流程**：确保评估的公平性和可复现性
3. **灵活使用**：支持标准评估和自定义评估
4. **持续更新**：从 VBench-1.0 到 VBench-2.0，不断扩展评估能力

使用 VBench 时，建议：
- 先进行小样本测试，验证流程正确性
- 根据需求选择合适的评估维度
- 遵循采样和命名规范
- 充分利用多 GPU 加速评估

