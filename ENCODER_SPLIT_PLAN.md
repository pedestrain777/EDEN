# EDEN Encoder 拆分方案

## 一、方案概述

将EDEN模型拆分为两个部分：
- **云端（Cloud）**：条件编码器 `E_cond` - 对输入帧 I0, I1 进行编码
- **边缘（Edge）**：扩散Transformer `G_DiT` + 解码器 `D_dec` - 使用编码后的条件进行扩散生成和解码

---

## 二、模块划分

### 2.1 云端模块：条件编码器 `E_cond`

#### 2.1.1 功能定位
- **输入**：原始图像帧 `frame_0` 和 `frame_1`
- **输出**：编码后的条件表示，供边缘端使用
- **作用**：将图像帧编码为tokens，提取条件信息

#### 2.1.2 需要提取的组件

从 `EDEN.py` 中提取以下组件：

1. **图像预处理组件**
   - `preprocess_cond()` 函数（`src/utils/__init__.py` 第24-33行）
   - `InputPadder` 类（`src/utils/__init__.py` 第8-21行）

2. **Patch编码组件**
   - `patch_cond_dit`（`EDEN.py` 第19行）
     - `nn.Conv2d(3, 768, kernel_size=16, stride=16)`
     - 用于DiT的条件编码
   - `patch_cond_dec`（`EDEN.py` 第30行）
     - `nn.Conv2d(3, 768, kernel_size=16, stride=16)`
     - 用于Decoder的条件编码
   - `norm_cond_dit`（`EDEN.py` 第20行）
     - `nn.LayerNorm(768)`
   - `norm_cond_dec`（`EDEN.py` 第31行）
     - `nn.LayerNorm(768)`

3. **位置编码组件**
   - `get_pos_embedding()` 函数（`src/utils/embedding.py` 第58-64行）
   - 相关辅助函数：`get_2d_sincos_pos_embed()` 等

#### 2.1.3 云端编码流程

```python
# 伪代码：云端编码流程
def cloud_encode(frame_0, frame_1):
    # 1. 图像预处理
    cond_frames = torch.cat((frame_0, frame_1), dim=0)  # [2, 3, H, W]
    padder = InputPadder(image_size)
    cond_frames = padder.pad(cond_frames)
    
    # 2. 计算patch尺寸
    ph = cond_frames.shape[-2] // 16
    pw = cond_frames.shape[-1] // 16
    
    # 3. 归一化并保存统计信息
    x_norm, stats = preprocess_cond(cond_frames)
    # stats = (mean, std) 用于后续反归一化
    
    # 4. 生成位置编码
    pos_embedding = get_pos_embedding(ph, pw, scale=1, dim=768)
    
    # 5. DiT条件编码
    x_dit = patch_cond_dit(x_norm)  # [2, 768, ph, pw]
    x_dit = x_dit.flatten(2).transpose(1, 2)  # [2, ph*pw, 768]
    x_dit = norm_cond_dit(x_dit + pos_embedding)
    cond_dit = x_dit  # [2, ph*pw, 768]
    
    # 6. Decoder条件编码
    x_dec = patch_cond_dec(x_norm)  # [2, 768, ph, pw]
    x_dec = x_dec.flatten(2).transpose(1, 2)  # [2, ph*pw, 768]
    x_dec = norm_cond_dec(x_dec + pos_embedding)
    cond_dec = x_dec  # [2, ph*pw, 768]
    
    # 7. 返回编码结果
    return {
        'cond_dit': cond_dit,      # [2, ph*pw, 768] - 用于DiT
        'cond_dec': cond_dec,      # [2, ph*pw, 768] - 用于Decoder
        'stats': stats,            # (mean, std) - 用于反归一化
        'ph': ph,                  # int - patch高度数量
        'pw': pw,                  # int - patch宽度数量
        'pos_embedding': pos_embedding,  # [1, ph*pw, 768] - 可选
        'padder_info': padder._pad  # 填充信息，用于后续unpad
    }
```

#### 2.1.4 云端输出数据结构

```python
# 云端需要传输给边缘的数据结构
cloud_output = {
    # 必需数据
    'cond_dit': torch.Tensor,      # [2, num_patches, 768]
    'cond_dec': torch.Tensor,      # [2, num_patches, 768]
    'stats': tuple,                # ((mean, std), ...)
    'ph': int,                     # patch高度数量
    'pw': int,                     # patch宽度数量
    
    # 可选数据（可以重新计算，但传输可以节省计算）
    'pos_embedding': torch.Tensor, # [1, num_patches, 768]
    'padder_info': list,           # [pad_w_left, pad_w_right, pad_h_top, pad_h_bottom]
    'original_size': tuple,        # (H, W) - 原始图像尺寸
}
```

#### 2.1.5 数据压缩建议

为了减少传输量，可以考虑：

1. **量化**：将float32转为float16或int8
2. **压缩**：使用gzip或其他压缩算法
3. **选择性传输**：
   - `pos_embedding` 可以在边缘重新计算（根据ph, pw）
   - `padder_info` 可以根据原始尺寸重新计算

---

### 2.2 边缘模块：DiT + Decoder

#### 2.2.1 DiT模块 `G_DiT`

**功能**：在latent空间进行扩散去噪

**输入**：
- `query_latents`: [1, num_query_patches, 16] - 噪声latent
- `denoise_timestep`: [1] - 时间步
- `cond_dit`: [2, num_patches, 768] - 从云端接收的DiT条件tokens
- `difference`: [1, 1] - 帧差异（可在边缘计算）

**输出**：
- `query_latents`: [1, num_query_patches, 16] - 去噪后的latent

**需要的组件**（从 `EDEN.py` 提取）：
- `proj_in`（第18行）
- `dit_blocks`（第21-24行）
- `norm_out`（第25行）
- `proj_out`（第26行）
- `adaLN_modulation`（第27行）
- `difference_embedder`（第28行）
- `denoise_timestep_embedder`（第29行）

**边缘DiT流程**：

```python
# 伪代码：边缘DiT流程
def edge_denoise(query_latents, denoise_timestep, cond_dit, difference, ph, pw):
    # 1. 分离条件tokens
    tokens_0, tokens_1 = cond_dit.chunk(2, dim=0)  # 各[1, num_patches, 768]
    
    # 2. 生成条件embedding
    denoise_timestep_embedding = denoise_timestep_embedder(denoise_timestep)
    difference_embedding = difference_embedder(difference)
    condition_embedding = denoise_timestep_embedding + difference_embedding
    modulations = adaLN_modulation(condition_embedding)
    
    # 3. 生成query位置编码（可以重新计算）
    query_pos_embedding = get_pos_embedding(ph, pw, scale=2, dim=768)
    
    # 4. 处理query
    query_embedding = proj_in(query_latents) + query_pos_embedding
    
    # 5. 通过DiT blocks
    for blk in dit_blocks:
        query_embedding = blk(query_embedding, tokens_0, tokens_1, ph, pw, modulations)
    
    # 6. 输出投影
    query_latents = proj_out(norm_out(query_embedding))
    query_latents, _ = query_latents.chunk(2, dim=-1)
    
    return query_latents
```

#### 2.2.2 Decoder模块 `D_dec`

**功能**：将去噪后的latent解码为图像

**输入**：
- `query_latents`: [1, num_query_patches, 16] - 去噪后的latent
- `cond_dec`: [2, num_patches, 768] - 从云端接收的Decoder条件tokens
- `stats`: (mean, std) - 从云端接收的统计信息
- `ph`, `pw`: int - 从云端接收的patch尺寸
- `pos_embedding`: [1, num_patches, 768] - 位置编码（可选，可重新计算）

**输出**：
- `generated_frame`: [1, 3, H, W] - 生成的中间帧

**需要的组件**（从 `EDEN.py` 提取）：
- `proj_token`（第32行）
- `decoder_blocks`（第33-37行）
- `unpatchify`（第39-42行）
- `postprocess()` 方法（第54-55行）
- `pixel_shuffle()` 方法（第57-61行）

**边缘Decoder流程**：

```python
# 伪代码：边缘Decoder流程
def edge_decode(query_latents, cond_dec, stats, ph, pw, pos_embedding, padder_info):
    # 1. 分离条件tokens
    tokens_0, tokens_1 = cond_dec.chunk(2, dim=0)
    cond_tokens = torch.cat((tokens_0, tokens_1), dim=1)  # [1, 2*num_patches, 768]
    
    # 2. 投影latent
    query_tokens = proj_token(query_latents)  # [1, num_query_patches, 768]
    
    # 3. 上采样
    query_frames = rearrange(query_tokens, "b (ph pw) d -> b d ph pw", 
                            ph=ph//2, pw=pw//2)
    recon_frames = interpolate(query_frames, scale_factor=2., mode="bicubic")
    recon_tokens = recon_frames.flatten(2).transpose(1, 2) + pos_embedding
    
    # 4. 生成query位置编码
    query_pos_embedding = get_pos_embedding(ph, pw, scale=2, dim=768)
    query_tokens = query_tokens + query_pos_embedding
    
    # 5. 通过Decoder blocks
    for blk in decoder_blocks:
        recon_tokens = blk(query_tokens, recon_tokens, cond_tokens, ph, pw)
    
    # 6. 还原为像素
    recon_frames = postprocess(pixel_shuffle(recon_tokens))
    
    # 7. 反填充
    padder = InputPadder.from_info(padder_info)
    generated_frame = padder.unpad(recon_frames)
    
    return generated_frame
```

---

## 三、接口设计

### 3.1 云端API接口

```python
# 云端服务接口
class CloudEncoderService:
    def __init__(self, model_path):
        # 加载encoder相关权重
        self.encoder = load_encoder_weights(model_path)
    
    def encode(self, frame_0, frame_1):
        """
        编码两帧图像
        
        Args:
            frame_0: [1, 3, H, W] - 第一帧
            frame_1: [1, 3, H, W] - 第二帧
        
        Returns:
            dict: 包含编码后的条件信息
        """
        return cloud_encode(frame_0, frame_1)
    
    def encode_batch(self, frames_list):
        """
        批量编码
        
        Args:
            frames_list: List of (frame_0, frame_1) tuples
        
        Returns:
            List of encoded conditions
        """
        pass
```

### 3.2 边缘API接口

```python
# 边缘服务接口
class EdgeInferenceService:
    def __init__(self, model_path):
        # 加载DiT和Decoder相关权重
        self.dit = load_dit_weights(model_path)
        self.decoder = load_decoder_weights(model_path)
        self.transport = create_transport("Linear", "velocity")
        self.sampler = Sampler(self.transport)
        self.sample_fn = self.sampler.sample_ode(...)
    
    def interpolate(self, encoded_cond):
        """
        使用编码后的条件进行插值
        
        Args:
            encoded_cond: dict - 从云端接收的编码条件
        
        Returns:
            generated_frame: [1, 3, H, W] - 生成的中间帧
        """
        # 1. 提取条件
        cond_dit = encoded_cond['cond_dit']
        cond_dec = encoded_cond['cond_dec']
        stats = encoded_cond['stats']
        ph = encoded_cond['ph']
        pw = encoded_cond['pw']
        
        # 2. 计算difference（如果需要）
        # difference可以在云端计算并传输，也可以在边缘计算
        
        # 3. 生成初始噪声
        num_query_patches = (ph // 2) * (pw // 2)
        noise = torch.randn([1, num_query_patches, 16])
        
        # 4. 扩散采样
        denoise_kwargs = {
            'cond_dit': cond_dit,
            'difference': difference,  # 从云端传输或本地计算
            'ph': ph,
            'pw': pw
        }
        samples = sample_fn(noise, self.dit.denoise, **denoise_kwargs)[-1]
        
        # 5. 反归一化
        denoise_latents = samples / vae_scaler + vae_shift
        
        # 6. 解码
        generated_frame = self.decoder.decode(
            denoise_latents, 
            cond_dec, 
            stats, 
            ph, 
            pw
        )
        
        return generated_frame
```

---

## 四、数据传输格式

### 4.1 序列化格式

建议使用以下格式传输数据：

```python
# 方案1：使用pickle（简单但不安全）
import pickle
encoded_data = pickle.dumps(cloud_output)

# 方案2：使用JSON + base64（跨平台）
import json
import base64
# 将tensor转为numpy，再转为base64
encoded_data = {
    'cond_dit': base64.b64encode(cond_dit.numpy().tobytes()).decode(),
    'cond_dit_shape': cond_dit.shape,
    'cond_dit_dtype': str(cond_dit.dtype),
    # ... 其他字段
}
json_data = json.dumps(encoded_data)

# 方案3：使用protobuf（推荐，高效且类型安全）
# 定义.proto文件，生成Python代码
```

### 4.2 数据大小估算

假设输入图像 480×640：

```
原始图像大小：
- frame_0: 480 × 640 × 3 × 4 bytes = 3.69 MB
- frame_1: 480 × 640 × 3 × 4 bytes = 3.69 MB
- 总计: 7.38 MB

编码后大小：
- ph = 30, pw = 40, num_patches = 1200
- cond_dit: 2 × 1200 × 768 × 4 bytes = 7.37 MB
- cond_dec: 2 × 1200 × 768 × 4 bytes = 7.37 MB
- stats: 很小，可忽略
- 总计: ~14.74 MB (float32) 或 ~7.37 MB (float16)

压缩后（使用float16 + gzip）：
- 预计: ~3-4 MB
```

---

## 五、实现步骤

### 5.1 阶段1：模块提取

1. **创建云端Encoder类**
   - 文件：`src/models/EDEN_Encoder.py`
   - 提取 `patch_cond()` 相关代码
   - 提取必要的工具函数

2. **创建边缘DiT类**
   - 文件：`src/models/EDEN_DiT_Edge.py`
   - 提取 `denoise()` 相关代码
   - 修改为接收编码后的条件

3. **创建边缘Decoder类**
   - 文件：`src/models/EDEN_Decoder_Edge.py`
   - 提取 `decode()` 相关代码
   - 修改为接收编码后的条件

### 5.2 阶段2：权重分离

1. **提取Encoder权重**
   ```python
   encoder_state_dict = {
       'patch_cond_dit.weight': eden.patch_cond_dit.weight,
       'patch_cond_dit.bias': eden.patch_cond_dit.bias,
       'norm_cond_dit.weight': eden.norm_cond_dit.weight,
       'norm_cond_dit.bias': eden.norm_cond_dit.bias,
       'patch_cond_dec.weight': eden.patch_cond_dec.weight,
       'patch_cond_dec.bias': eden.patch_cond_dec.bias,
       'norm_cond_dec.weight': eden.norm_cond_dec.weight,
       'norm_cond_dec.bias': eden.norm_cond_dec.bias,
   }
   ```

2. **提取DiT权重**
   ```python
   dit_state_dict = {
       'proj_in.weight': eden.proj_in.weight,
       'proj_in.bias': eden.proj_in.bias,
       'dit_blocks.*': eden.dit_blocks.state_dict(),
       'norm_out.weight': eden.norm_out.weight,
       'norm_out.bias': eden.norm_out.bias,
       'proj_out.weight': eden.proj_out.weight,
       'proj_out.bias': eden.proj_out.bias,
       'adaLN_modulation.*': eden.adaLN_modulation.state_dict(),
       'difference_embedder.*': eden.difference_embedder.state_dict(),
       'denoise_timestep_embedder.*': eden.denoise_timestep_embedder.state_dict(),
   }
   ```

3. **提取Decoder权重**
   ```python
   decoder_state_dict = {
       'proj_token.weight': eden.proj_token.weight,
       'proj_token.bias': eden.proj_token.bias,
       'decoder_blocks.*': eden.decoder_blocks.state_dict(),
       'unpatchify.*': eden.unpatchify.state_dict(),
   }
   ```

### 5.3 阶段3：接口实现

1. **云端服务**
   - 实现HTTP/gRPC接口
   - 实现数据序列化/反序列化
   - 实现批量处理

2. **边缘服务**
   - 实现数据接收接口
   - 实现推理流程
   - 实现结果返回

### 5.4 阶段4：测试验证

1. **功能测试**
   - 验证拆分后的结果与原始EDEN一致
   - 验证数据传输正确性

2. **性能测试**
   - 测试传输延迟
   - 测试推理速度
   - 测试内存占用

---

## 六、关键注意事项

### 6.1 依赖关系

1. **位置编码**
   - `pos_embedding` 可以根据 `ph, pw` 重新计算
   - 建议：传输 `ph, pw`，在边缘重新计算以节省传输

2. **统计信息**
   - `stats` 必须从云端传输
   - 用于Decoder的反归一化

3. **填充信息**
   - `padder_info` 可以根据原始尺寸重新计算
   - 建议：传输原始尺寸 `(H, W)`，在边缘重新计算

### 6.2 计算差异

1. **difference计算**
   - 可以在云端计算并传输
   - 也可以在边缘计算（需要原始图像，但会增加传输量）
   - 建议：在云端计算并传输

2. **query位置编码**
   - 可以在边缘根据 `ph, pw` 重新计算
   - 不需要从云端传输

### 6.3 兼容性

1. **模型权重**
   - 使用原始EDEN的预训练权重
   - 只需提取对应的权重部分

2. **版本控制**
   - 确保云端和边缘使用相同版本的编码格式
   - 建议使用版本号标识

---

## 七、优化建议

### 7.1 传输优化

1. **量化**
   - float32 → float16：减少50%传输量
   - float16 → int8：进一步减少，但可能影响精度

2. **压缩**
   - 使用gzip压缩
   - 使用更高效的压缩算法（如zstd）

3. **选择性传输**
   - 只传输必需数据
   - 可计算的数据在边缘重新计算

### 7.2 计算优化

1. **缓存**
   - 缓存位置编码（根据ph, pw）
   - 缓存常用配置

2. **批处理**
   - 云端支持批量编码
   - 边缘支持批量推理

---

## 八、文件结构建议

```
eden_split/
├── cloud/
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── eden_encoder.py      # Encoder类
│   │   └── encoder_utils.py     # 工具函数
│   ├── service/
│   │   ├── __init__.py
│   │   ├── api_server.py        # API服务
│   │   └── serialization.py    # 序列化
│   └── weights/
│       └── encoder.pt           # Encoder权重
│
├── edge/
│   ├── dit/
│   │   ├── __init__.py
│   │   ├── eden_dit_edge.py     # DiT类（边缘版）
│   │   └── dit_utils.py
│   ├── decoder/
│   │   ├── __init__.py
│   │   ├── eden_decoder_edge.py # Decoder类（边缘版）
│   │   └── decoder_utils.py
│   ├── service/
│   │   ├── __init__.py
│   │   ├── inference_service.py # 推理服务
│   │   └── deserialization.py   # 反序列化
│   └── weights/
│       ├── dit.pt                # DiT权重
│       └── decoder.pt           # Decoder权重
│
└── shared/
    ├── __init__.py
    ├── data_format.py            # 数据格式定义
    └── utils.py                  # 共享工具函数
```

---

## 九、总结

本方案将EDEN拆分为：
- **云端**：条件编码器，负责将图像编码为tokens
- **边缘**：DiT + Decoder，负责使用编码条件进行生成

**关键点**：
1. 保持与原始EDEN的兼容性
2. 最小化数据传输量
3. 保持推理结果的准确性
4. 支持灵活的部署方式

**下一步**：
1. 实现模块提取
2. 实现权重分离脚本
3. 实现接口服务
4. 进行测试验证

