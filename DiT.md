查看 DiT 相关代码，梳理其设计与输入输出流程。


[5 tools called]


下面讲解 DiT（Diffusion Transformer）的设计与输入输出流程。

## DiT（Diffusion Transformer）设计详解

### 一、DiT 整体架构

DiT 在 latent 空间进行去噪，预测去噪方向（velocity）。

```
DiT架构:
┌─────────────────────────────────────┐
│  输入:                                │
│  - query_latents [1, 300, 16]        │
│  - denoise_timestep [1]               │
│  - cond_frames [2, 3, 480, 640]      │
│  - difference [1, 1]                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  1. 条件编码 (patch_cond)             │
│     - 编码条件帧为tokens              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  2. 条件Embedding生成                  │
│     - 时间步embedding                  │
│     - 差异embedding                    │
│     - AdaLN调制参数                    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  3. Query处理                          │
│     - 投影到高维                       │
│     - 添加位置编码                     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  4. DiT Blocks (12层)                 │
│     - Self-Attention                  │
│     - Temporal Attention              │
│     - MLP                             │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  5. 输出投影                           │
│     - 归一化                           │
│     - 投影回latent空间                 │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  输出: query_latents [1, 300, 16]     │
│  (预测的去噪方向)                      │
└──────────────────────────────────────┘
```

---

### 二、DiT 核心组件详解

#### 组件1：条件编码（Condition Encoding）

位置：`denoise()` 方法第78行

```python
cond_dit, self.cond_dec = self.patch_cond(cond_frames)
tokens_0, tokens_1 = cond_dit.chunk(2, dim=0)
```

作用：
- 将条件帧编码为 tokens
- 分离两帧的 tokens

数据形状：
```
输入: cond_frames [2, 3, 480, 640]
  ↓ patch_cond()
输出: cond_dit [2, 1200, 768]
  ↓ chunk(2, dim=0)
tokens_0: [1, 1200, 768] (frame_0)
tokens_1: [1, 1200, 768] (frame_1)
```

---

#### 组件2：时间步编码（Timestep Embedding）

位置：`denoise()` 方法第80行

```python
denoise_timestep_embedding = self.denoise_timestep_embedder(denoise_timestep)
```

工作原理：
```python
# TimestepEmbedder内部:
# 1. 生成sin/cos编码
t_freq = timestep_embedding(t, 256)  # [1, 256]
# 2. 通过MLP
t_emb = mlp(t_freq)  # [1, 768]
```

作用：
- 告诉模型当前去噪进度
- t=1.0（完全噪声）→ t=0.0（干净图像）

示例：
```
t=1.0 → embedding_A (表示"完全噪声")
t=0.5 → embedding_B (表示"一半噪声")
t=0.0 → embedding_C (表示"干净图像")
```

---

#### 组件3：差异编码（Difference Embedding）

位置：`denoise()` 方法第81行

```python
difference_embedding = self.difference_embedder(difference)
```

作用：
- 编码两帧的相似程度
- 帮助模型理解运动幅度

```
difference = 0.99 → 两帧很相似，运动小
difference = 0.50 → 两帧差异大，运动大
```

---

#### 组件4：AdaLN 调制（Adaptive Layer Normalization Modulation）

位置：`denoise()` 方法第82-83行

```python
condition_embedding = denoise_timestep_embedding + difference_embedding
modulations = self.adaLN_modulation(condition_embedding)
```

AdaLN 是什么？
- 根据条件动态调整 LayerNorm 的参数
- 让模型根据时间步和差异调整行为

```python
# adaLN_modulation结构:
# SiLU → Linear(768 → 4608)
# 输出6组参数，每组768维:
modulations = [shift_msa, scale_msa, gate_msa, 
               shift_mlp, scale_mlp, gate_mlp]
# 每组: [1, 768]
```

作用：
- `shift` 和 `scale`：调整 LayerNorm
- `gate`：控制信息流（类似门控）

---

#### 组件5：Query 处理

位置：`denoise()` 方法第84-85行

```python
self.query_pos_embedding = get_pos_embedding(self.ph, self.pw, 2, self.dim)
query_embedding = self.proj_in(query_latents) + self.query_pos_embedding
```

数据变换：
```
输入: query_latents [1, 300, 16]
  ↓ proj_in (Linear: 16 → 768)
[1, 300, 768]
  ↓ + query_pos_embedding [1, 300, 768]
query_embedding: [1, 300, 768]
```

为什么需要位置编码？
- Query tokens 也需要知道位置信息
- `scale=2` 表示 query 空间是条件空间的一半分辨率

---

### 三、DiTBlock 详细设计

DiTBlock 是 DiT 的核心处理单元，包含三个部分：

#### DiTBlock 结构（blocks.py 第67-86行）

```python
class DiTBlock(nn.Module):
    def __init__(self, ...):
        self.norm_self_attn = LayerNorm(768)
        self.self_attn = SelfAttention(...)
        self.norm_temp_attn = LayerNorm(768)
        self.temp_attn = TemporalAttention(...)
        self.norm_mlp = LayerNorm(768)
        self.mlp = Mlp(...)
```

#### DiTBlock 前向传播（第78-86行）

```python
def forward(self, query_tokens, tokens0, tokens1, ph, pw, modulations):
    # 1. 分解调制参数
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        modulations.chunk(6, dim=1)
    
    # 2. Self-Attention (带AdaLN调制)
    query_tokens = query_tokens + gate_msa.unsqueeze(1) * \
        self.self_attn(modulate(self.norm_self_attn(query_tokens), 
                                 shift_msa, scale_msa))
    
    # 3. Temporal Attention
    query_tokens = query_tokens + \
        self.temp_attn(self.norm_temp_attn(query_tokens), 
                      x0=tokens0, x1=tokens1, ph=ph, pw=pw)
    
    # 4. MLP (带AdaLN调制)
    query_tokens = query_tokens + gate_mlp.unsqueeze(1) * \
        self.mlp(modulate(self.norm_mlp(query_tokens), 
                         shift_mlp, scale_mlp))
    
    return query_tokens
```

---

### 四、注意力机制详解

#### 1. Self-Attention（自注意力）

作用：让 query tokens 之间相互关注

```python
# SelfAttention内部:
# 1. 生成Q, K, V
qkv = self.qkv(query_tokens)  # [1, 300, 2304]
q, k, v = qkv.chunk(3, dim=-1)  # 各[1, 300, 768]

# 2. 多头分割
q = q.reshape(1, 300, 12, 64)  # 12个头，每个64维
k = k.reshape(1, 300, 12, 64)
v = v.reshape(1, 300, 12, 64)

# 3. 计算注意力
attn = (q @ k.transpose(-2, -1)) * scale  # [1, 12, 300, 300]
attn = attn.softmax(dim=-1)
output = attn @ v  # [1, 300, 768]
```

注意力矩阵的含义：
```
attn[i, j] = Token i 对 Token j 的关注程度
```

---

#### 2. Temporal Attention（时间注意力）

作用：让 query tokens 同时关注 frame_0 和 frame_1

位置：`attention.py` 第46-74行

```python
def forward(self, x, x0=None, x1=None, ...):
    # x: query_tokens [1, 300, 768]
    # x0: tokens0 [1, 1200, 768] (frame_0)
    # x1: tokens1 [1, 1200, 768] (frame_1)
    
    if x0.shape[1] != x.shape[1]:
        # 如果尺寸不匹配，需要reshape
        x0 = self.reshape_cond(x0, ph, pw)  # [1, 300, 768]
        x1 = self.reshape_cond(x1, ph, pw)  # [1, 300, 768]
        x = x.unsqueeze(2)  # [1, 300, 1, 768]
        # 拼接: [frame_0, query, frame_1]
        x = torch.cat((x0, x, x1), dim=2)  # [1, 300, 3, 768]
        x = x.reshape(1 * 300, 3, 768)  # [300, 3, 768]
    else:
        # 直接拼接
        x = torch.stack((x0, x, x1), dim=2)  # [1, 300, 3, 768]
        x = x.reshape(1 * 300, 3, 768)  # [300, 3, 768]
    
    # Self-Attention处理
    qkv = self.qkv(x)  # [300, 3, 2304]
    q, k, v = qkv.chunk(3, dim=-1)  # 各[300, 3, 768]
    # ... 计算注意力 ...
    
    # 只取中间位置（query对应的输出）
    x = x[:, 1, :]  # [300, 768] (取中间位置)
    x = x.reshape(1, 300, 768)
    return x
```

工作原理：
```
对于每个query token:
┌─────────┬─────────┬─────────┐
│ Frame_0 │  Query  │ Frame_1 │
│ Token   │  Token  │  Token  │
└─────────┴─────────┴─────────┘
     ↑         ↑         ↑
     关注      当前      关注

Query token通过注意力机制，同时关注：
- Frame_0的对应位置
- Frame_1的对应位置
- 理解两帧之间的运动
```

---

#### 3. AdaLN 调制（modulate 函数）

位置：`blocks.py` 第7-8行

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

作用：
- 动态调整 LayerNorm 的输出
- 根据条件（时间步+差异）调整特征

```
标准LayerNorm:
x_norm = (x - mean) / std

AdaLN调制:
x_mod = (x_norm * (1 + scale)) + shift
```

---

### 五、完整输入输出流程（数值示例）

假设输入：480×640 的图像

#### 步骤1：输入准备

```python
# 输入
query_latents: [1, 300, 16]  # 噪声latent
denoise_timestep: [1] = 0.5  # 时间步
cond_frames: [2, 3, 480, 640]  # 条件帧
difference: [1, 1] = 0.85  # 帧差异
```

#### 步骤2：条件编码（第78行）

```python
cond_dit, self.cond_dec = self.patch_cond(cond_frames)
# cond_dit: [2, 1200, 768]
tokens_0, tokens_1 = cond_dit.chunk(2, dim=0)
# tokens_0: [1, 1200, 768] (frame_0)
# tokens_1: [1, 1200, 768] (frame_1)
```

#### 步骤3：条件 Embedding（第80-83行）

```python
# 3.1 时间步embedding
denoise_timestep_embedding = self.denoise_timestep_embedder(0.5)
# [1, 768]

# 3.2 差异embedding
difference_embedding = self.difference_embedder(0.85)
# [1, 768]

# 3.3 合并
condition_embedding = denoise_timestep_embedding + difference_embedding
# [1, 768]

# 3.4 生成调制参数
modulations = self.adaLN_modulation(condition_embedding)
# [1, 4608] → chunk(6) → 6个[1, 768]
```

#### 步骤4：Query 处理（第84-85行）

```python
# 4.1 生成query位置编码
self.query_pos_embedding = get_pos_embedding(30, 40, 2, 768)
# [1, 300, 768] (scale=2，因为query空间是条件空间的一半)

# 4.2 投影并添加位置编码
query_embedding = self.proj_in(query_latents) + self.query_pos_embedding
# query_latents: [1, 300, 16]
#   ↓ proj_in (16 → 768)
# [1, 300, 768]
#   ↓ + query_pos_embedding
# query_embedding: [1, 300, 768]
```

#### 步骤5：DiT Blocks 处理（第86-87行）

```python
for blk in self.dit_blocks:  # 12层
    query_embedding = blk(query_embedding, tokens_0, tokens_1, 
                         self.ph, self.pw, modulations)
```

##### DiTBlock 内部流程（第1层示例）

```python
# 输入: query_embedding [1, 300, 768]

# 5.1 Self-Attention (带AdaLN)
x = self.norm_self_attn(query_embedding)  # LayerNorm
x = modulate(x, shift_msa, scale_msa)  # AdaLN调制
x = self.self_attn(x)  # Self-Attention
query_embedding = query_embedding + gate_msa * x  # 残差连接
# [1, 300, 768]

# 5.2 Temporal Attention
x = self.norm_temp_attn(query_embedding)  # LayerNorm
x = self.temp_attn(x, x0=tokens_0, x1=tokens_1, ph=30, pw=40)
# 内部: reshape tokens_0和tokens_1到[1, 300, 768]
#       拼接为[1, 300, 3, 768]
#       计算注意力，取中间位置
query_embedding = query_embedding + x  # 残差连接
# [1, 300, 768]

# 5.3 MLP (带AdaLN)
x = self.norm_mlp(query_embedding)  # LayerNorm
x = modulate(x, shift_mlp, scale_mlp)  # AdaLN调制
x = self.mlp(x)  # MLP: [1, 300, 768] → [1, 300, 3072] → [1, 300, 768]
query_embedding = query_embedding + gate_mlp * x  # 残差连接
# [1, 300, 768]

# 重复12次...
```

#### 步骤6：输出投影（第88-89行）

```python
# 6.1 归一化
query_embedding = self.norm_out(query_embedding)
# [1, 300, 768]

# 6.2 投影回latent空间
query_latents = self.proj_out(query_embedding)
# proj_out: Linear(768 → 32)
# [1, 300, 768] → [1, 300, 32]

# 6.3 只取一半（为什么？可能是设计选择）
query_latents, _ = query_latents.chunk(2, dim=-1)
# [1, 300, 32] → [1, 300, 16]
```

#### 步骤7：返回结果

```python
return query_latents  # [1, 300, 16]
# 这是预测的去噪方向（velocity）
```

---

### 六、数据流图（完整）

```
┌─────────────────────────────────────────────────────────────┐
│                    DiT完整流程                                │
└─────────────────────────────────────────────────────────────┘

输入:
- query_latents [1, 300, 16]
- denoise_timestep [1] = 0.5
- cond_frames [2, 3, 480, 640]
- difference [1, 1] = 0.85
  │
  ├─> 条件编码
  │   cond_dit [2, 1200, 768]
  │   ↓ chunk(2)
  │   tokens_0 [1, 1200, 768]
  │   tokens_1 [1, 1200, 768]
  │
  ├─> 条件Embedding
  │   timestep_emb [1, 768]
  │   diff_emb [1, 768]
  │   ↓ 相加
  │   condition_emb [1, 768]
  │   ↓ adaLN_modulation
  │   modulations [1, 4608] → 6个[1, 768]
  │
  ├─> Query处理
  │   query_latents [1, 300, 16]
  │   ↓ proj_in
  │   [1, 300, 768]
  │   ↓ + query_pos_embedding
  │   query_embedding [1, 300, 768]
  │
  └─> DiT Blocks (12层)
      query_embedding [1, 300, 768]
      ↓
      ┌─────────────────────────┐
      │ DiTBlock 1               │
      │ 1. Self-Attention        │
      │ 2. Temporal Attention   │
      │ 3. MLP                   │
      └─────────────────────────┘
      query_embedding [1, 300, 768]
      ↓
      ┌─────────────────────────┐
      │ DiTBlock 2               │
      │ ...                      │
      └─────────────────────────┘
      ...
      ↓
      ┌─────────────────────────┐
      │ DiTBlock 12              │
      │ ...                      │
      └─────────────────────────┘
      query_embedding [1, 300, 768]
      │
      ↓
      输出投影
      norm_out → proj_out → chunk(2)
      │
      └─> query_latents [1, 300, 16]
          (预测的去噪方向)
```

---

### 七、关键设计要点

#### 1. 为什么用 AdaLN？

- 根据时间步和差异动态调整模型行为
- 不同时间步需要不同的去噪策略

#### 2. 为什么需要 Temporal Attention？

- 同时关注两帧，理解运动
- 帮助生成合理的中间帧

#### 3. 为什么 Query 空间是条件空间的一半？

- 降低计算量
- Latent 空间本身就是压缩表示

#### 4. 残差连接的作用

- 稳定训练
- 允许信息直接传递

---

### 八、总结

DiT 的核心功能：
1. 条件编码：将条件帧编码为 tokens
2. 条件 Embedding：生成时间步和差异的 embedding
3. AdaLN 调制：根据条件动态调整模型
4. Self-Attention：Query tokens 之间交互
5. Temporal Attention：同时关注两帧
6. MLP：进一步特征处理
7. 输出投影：预测去噪方向

输出：
- `query_latents [1, 300, 16]`：预测的去噪方向（velocity）
- 用于 ODE 求解器逐步去噪

如需进一步解释某个部分，请指出。