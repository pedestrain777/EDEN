# VBench 1.0 评估要求详细总结

## 📋 目录
1. [16 个评估维度](#16-个评估维度)
2. [Prompt 文件说明](#prompt-文件说明)
3. [视频生成要求](#视频生成要求)
4. [是否必须生成所有视频](#是否必须生成所有视频)
5. [评估流程](#评估流程)
6. [实际建议](#实际建议)

---

## 16 个评估维度

VBench 1.0 包含以下 16 个评估维度：

```python
[
    'subject_consistency',      # 主体一致性
    'background_consistency',   # 背景一致性
    'temporal_flickering',      # 时间闪烁
    'motion_smoothness',        # 运动平滑度
    'dynamic_degree',           # 动态程度
    'aesthetic_quality',        # 美学质量
    'imaging_quality',          # 成像质量
    'object_class',             # 物体类别
    'multiple_objects',         # 多个物体
    'human_action',             # 人类动作
    'color',                    # 颜色
    'spatial_relationship',    # 空间关系
    'scene',                    # 场景
    'temporal_style',           # 时间风格
    'appearance_style',         # 外观风格
    'overall_consistency'       # 整体一致性
]
```

---

## Prompt 文件说明

### 1. `all_dimension.txt` 文件

**位置**: `VBench-master/prompts/all_dimension.txt`

**内容**:
- 包含 **946 个 prompts**
- 这些 prompts 来自所有 16 个维度
- 是 `prompts_per_dimension/` 目录下所有文件的合并

**重要**: 
- ✅ `all_dimension.txt` **确实包含了所有 16 个维度的 prompts**
- ⚠️ 但**不是每个 prompt 都属于所有维度**，每个 prompt 可能只属于 1-2 个维度

### 2. `prompts_per_dimension/` 目录

**位置**: `VBench-master/prompts/prompts_per_dimension/`

**内容**: 每个维度有独立的 prompt 文件

| 维度 | 文件名 | Prompt 数量 |
|------|--------|------------|
| `subject_consistency` | `subject_consistency.txt` | 71 |
| `background_consistency` | `scene.txt` | 86 |
| `temporal_flickering` | `temporal_flickering.txt` | 74 |
| `motion_smoothness` | `subject_consistency.txt` | 72 |
| `dynamic_degree` | `subject_consistency.txt` | 72 |
| `aesthetic_quality` | `overall_consistency.txt` | 93 |
| `imaging_quality` | `overall_consistency.txt` | 93 |
| `object_class` | `object_class.txt` | 78 |
| `multiple_objects` | `multiple_objects.txt` | 81 |
| `human_action` | `human_action.txt` | 99 |
| `color` | `color.txt` | 84 |
| `spatial_relationship` | `spatial_relationship.txt` | 83 |
| `scene` | `scene.txt` | 85 |
| `temporal_style` | `temporal_style.txt` | 99 |
| `appearance_style` | `appearance_style.txt` | 89 |
| `overall_consistency` | `overall_consistency.txt` | 92 |

**注意**: 
- 不同维度可能共享相同的 prompt 文件（如 `motion_smoothness` 和 `dynamic_degree` 都使用 `subject_consistency.txt`）
- 这是因为某些维度评估的是同一组 prompts 的不同方面

### 3. `VBench_full_info.json` 文件

**位置**: `VBench-master/vbench/VBench_full_info.json`

**作用**: 
- 定义了每个 prompt 属于哪些维度
- VBench 评估时会根据这个文件自动筛选对应维度的 prompts

**示例**:
```json
{
    "prompt_en": "In a still frame, a stop sign",
    "dimension": ["temporal_flickering"]
}
```

---

## 视频生成要求

### ✅ 核心要求：每个 Prompt 生成 5 个视频

**官方要求**（来自 `prompts/README.md`）:

> For each prompt, sample 5 videos. However, for the `Temporal Flickering` dimension, sample 25 videos to ensure sufficient coverage after applying the static filter.

### 视频命名格式

**格式**: `{prompt}-{index}.mp4`

**说明**:
- `{prompt}`: prompt 的完整文本
- `{index}`: 0, 1, 2, 3, 4（共 5 个视频）

**示例**:
```
A 3D model of a 1800s victorian house.-0.mp4
A 3D model of a 1800s victorian house.-1.mp4
A 3D model of a 1800s victorian house.-2.mp4
A 3D model of a 1800s victorian house.-3.mp4
A 3D model of a 1800s victorian house.-4.mp4
```

### 特殊要求：Temporal Flickering 维度

**要求**: 生成 **25 个视频**（而不是 5 个）

**原因**: 
- Temporal Flickering 维度需要先过滤掉静态视频
- 生成 25 个视频可以确保过滤后仍有足够的样本进行评估

**命名**: 同样使用 `{prompt}-{index}.mp4`，但 index 从 0 到 24

### 随机种子要求

**要求**:
1. **每个视频使用不同的随机种子**，确保多样性
2. **记录每个视频的随机种子**，确保可复现
3. **随机种子必须是随机的**，不能是精心挑选的

**代码示例**:
```python
for prompt in prompt_list:
    for index in range(5):
        # 为每个视频设置不同的随机种子
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        
        # 生成视频
        video = sample_func(prompt, index)
        
        # 保存视频
        save_path = f'{save_path}/{prompt}-{index}.mp4'
        torchvision.io.write_video(save_path, video, fps=8)
```

---

## 是否必须生成所有视频

### 📊 理论要求

**完整评估需要**:
- 946 个 prompts（来自 `all_dimension.txt`）
- 每个 prompt 5 个视频
- **总计**: 946 × 5 = **4,730 个视频**

**Temporal Flickering 维度**:
- 74 个 prompts
- 每个 prompt 25 个视频
- **总计**: 74 × 25 = **1,850 个视频**

### ⚠️ 实际情况

**VBench 评估代码的行为**（`vbench/__init__.py` 第 153 行）:

```python
for i in range(5): # video index for the same prompt
    intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'
    if intended_video_name in video_names: # if the video exists
        intended_video_path = os.path.join(videos_path, intended_video_name)
        prompt_dict['video_list'].append(intended_video_path)
    else:
        print0(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
```

**关键发现**:
1. ✅ **VBench 不会因为缺少视频而报错**
2. ⚠️ **会显示警告**，但会继续评估已有的视频
3. ⚠️ **缺少视频会影响评估结果的公平性**

### 💡 实际建议

#### 方案 1: 完整评估（推荐用于论文发表）

**优点**:
- ✅ 结果完整、公平
- ✅ 可以提交到 VBench Leaderboard
- ✅ 结果具有可比性

**缺点**:
- ❌ 需要生成大量视频（4,730+ 个）
- ❌ 时间成本高

#### 方案 2: 部分评估（适合快速验证）

**可以只生成部分 prompts**:
- 每个维度选择部分 prompts（如每个维度 10-20 个）
- 每个 prompt 仍然生成 5 个视频
- 可以快速验证模型性能

**示例**:
```python
# 只评估 temporal_flickering 维度
dimension = "temporal_flickering"
prompts_file = f"prompts/prompts_per_dimension/{dimension}.txt"

# 只选择前 15 个 prompts
with open(prompts_file) as f:
    all_prompts = [line.strip() for line in f]
selected_prompts = all_prompts[:15]  # 只选择前 15 个

# 每个 prompt 生成 5 个视频
for prompt in selected_prompts:
    for index in range(5):
        # 生成视频...
```

**注意**:
- ⚠️ 部分评估的结果可能不够全面
- ⚠️ 不能提交到官方 Leaderboard
- ✅ 但可以用于内部对比和快速验证

#### 方案 3: 按维度评估（推荐用于研究）

**策略**:
1. 选择你关心的几个维度
2. 为每个维度生成所有 prompts 的视频
3. 每个 prompt 生成 5 个视频

**示例**:
```python
# 选择要评估的维度
dimensions = [
    "temporal_flickering",
    "subject_consistency",
    "motion_smoothness",
    "aesthetic_quality"
]

for dimension in dimensions:
    prompts_file = f"prompts/prompts_per_dimension/{dimension}.txt"
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f]
    
    for prompt in prompts:
        for index in range(5):
            # 生成视频...
```

**优点**:
- ✅ 可以专注于特定维度
- ✅ 视频数量可控
- ✅ 结果仍然有意义

---

## 评估流程

### 1. 视频准备

**目录结构**:
```
videos/
├── prompt1-0.mp4
├── prompt1-1.mp4
├── prompt1-2.mp4
├── prompt1-3.mp4
├── prompt1-4.mp4
├── prompt2-0.mp4
├── prompt2-1.mp4
└── ...
```

**要求**:
- 所有视频放在**同一个目录**下（扁平结构）
- 视频命名格式：`{prompt}-{index}.mp4`

### 2. 运行评估

**命令**:
```bash
python evaluate.py \
    --videos_path /path/to/videos/ \
    --dimension temporal_flickering \
    --output_path /path/to/output/
```

**评估多个维度**:
```bash
python evaluate.py \
    --videos_path /path/to/videos/ \
    --dimension temporal_flickering subject_consistency motion_smoothness \
    --output_path /path/to/output/
```

### 3. 评估结果

**输出文件**:
- `{name}_full_info.json`: 评估元数据
- `{name}_eval_results.json`: 评估结果

**结果格式**:
```json
{
    "temporal_flickering": [
        0.9792,  // 平均分数
        [
            {
                "video_path": "/path/to/video-0.mp4",
                "video_results": 0.9973
            },
            ...
        ]
    ]
}
```

---

## 实际建议

### 🎯 针对你的情况

**问题**: 时间不允许生成所有视频，是否可以只生成部分？

**答案**: ✅ **可以，但需要注意以下几点**

### 建议方案

#### 1. **最小可行方案**（快速验证）

**目标**: 快速验证模型性能

**策略**:
- 选择 3-5 个关键维度
- 每个维度选择 10-15 个 prompts
- 每个 prompt 生成 5 个视频

**视频数量**: 约 750-1,125 个视频

**示例维度**:
- `temporal_flickering`（时间稳定性）
- `subject_consistency`（主体一致性）
- `motion_smoothness`（运动平滑度）
- `aesthetic_quality`（美学质量）

#### 2. **中等方案**（论文实验）

**目标**: 获得有意义的评估结果

**策略**:
- 选择 8-10 个维度
- 每个维度生成所有 prompts 的视频
- 每个 prompt 生成 5 个视频

**视频数量**: 约 3,000-4,000 个视频

#### 3. **完整方案**（提交 Leaderboard）

**目标**: 完整评估，提交到官方 Leaderboard

**策略**:
- 评估所有 16 个维度
- 生成所有 prompts 的视频
- 每个 prompt 生成 5 个视频（temporal_flickering 生成 25 个）

**视频数量**: 约 4,730 个视频（temporal_flickering 额外 1,850 个）

### ⚠️ 重要注意事项

1. **每个 prompt 必须生成 5 个视频**
   - 这是 VBench 的硬性要求
   - 少于 5 个会影响评估结果的公平性

2. **视频命名格式必须正确**
   - 格式：`{prompt}-{index}.mp4`
   - index 必须是 0, 1, 2, 3, 4

3. **随机种子必须不同**
   - 每个视频使用不同的随机种子
   - 记录种子值以便复现

4. **Temporal Flickering 特殊处理**
   - 如果评估这个维度，建议生成 25 个视频
   - 或者至少生成 10-15 个，确保过滤后仍有足够样本

### 📝 总结

| 方案 | 视频数量 | 适用场景 | 评估完整性 |
|------|---------|---------|-----------|
| **最小可行** | ~750-1,125 | 快速验证 | ⭐⭐ |
| **中等** | ~3,000-4,000 | 论文实验 | ⭐⭐⭐⭐ |
| **完整** | ~4,730+ | Leaderboard | ⭐⭐⭐⭐⭐ |

**建议**: 
- 如果时间有限，选择**中等方案**
- 选择你关心的维度，生成所有 prompts 的视频
- 每个 prompt 生成 5 个视频
- 这样既能获得有意义的评估结果，又能控制视频数量

---

## 参考资源

1. **VBench 官方文档**: `VBench-master/README.md`
2. **Prompt 说明**: `VBench-master/prompts/README.md`
3. **评估代码**: `VBench-master/vbench/__init__.py`
4. **Prompt 文件**: `VBench-master/prompts/prompts_per_dimension/`
5. **元数据文件**: `VBench-master/vbench/VBench_full_info.json`

---

## 快速检查清单

在开始生成视频前，请确认：

- [ ] 已选择要评估的维度
- [ ] 已确定每个维度的 prompts 列表
- [ ] 已准备好视频生成脚本
- [ ] 已设置随机种子机制
- [ ] 已确认视频命名格式：`{prompt}-{index}.mp4`
- [ ] 已确认每个 prompt 生成 5 个视频
- [ ] 已确认所有视频保存在同一目录下
- [ ] 已准备好评估脚本和输出目录

---

**最后更新**: 2025-11-15
**基于**: VBench-master 项目代码和文档

