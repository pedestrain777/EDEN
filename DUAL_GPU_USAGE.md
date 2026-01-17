# 双GPU模式使用说明

## 功能说明

已实现双GPU模式，可以在同一进程中模拟"云端encoder + 边缘DiT+decoder"的场景：

- **Encoder (cuda:0)**: 负责将输入帧编码为tokens
- **DiT+Decoder (cuda:1)**: 负责扩散去噪和解码生成

通过打包/解包机制模拟网络传输，为后续真正的HTTP服务做准备。

## 使用方法

### 1. 单GPU模式（默认）

```bash
# 图像插值
python inference.py \
    --frame_0_path examples/frame_0.jpg \
    --frame_1_path examples/frame_1.jpg

# 视频插值
python inference.py \
    --video_path examples/0.mp4
```

### 2. 双GPU模式

添加 `--use_split_gpu` 参数：

```bash
# 图像插值（双GPU）
python inference.py \
    --frame_0_path examples/frame_0.jpg \
    --frame_1_path examples/frame_1.jpg \
    --use_split_gpu

# 视频插值（双GPU）
python inference.py \
    --video_path examples/0.mp4 \
    --use_split_gpu
```

## 系统要求

- **双GPU模式需要至少2张GPU**
- 如果只有1张GPU，会自动回退到单GPU模式

## 工作流程

### 双GPU模式流程：

```
1. 输入帧 (CPU)
   ↓
2. Encoder (cuda:0)
   - 编码为tokens
   - 打包成bytes（所有tensor移到CPU）
   ↓
3. 模拟传输（内存中的bytes）
   ↓
4. 解包到cuda:1
   ↓
5. DiT+Decoder (cuda:1)
   - 扩散去噪
   - 解码生成
   ↓
6. 输出帧 (CPU)
```

## 关键文件

1. **`src/utils/encode_transfer.py`**
   - `pack_enc_out()`: 将encoder输出打包成bytes
   - `unpack_enc_out()`: 从bytes还原encoder输出

2. **`inference.py`**
   - 修改了`interpolate()`函数，支持双GPU模式
   - 添加了`--use_split_gpu`参数

## 技术细节

### 打包/解包机制

```python
# 打包（在encoder GPU上）
enc_out = eden_enc.encode(cond_frames)  # dict
blob = pack_enc_out(enc_out)  # bytes

# 解包（在DiT+decoder GPU上）
enc_out_dit = unpack_enc_out(blob, device_ditdec)  # dict，tensor在cuda:1上
```

### 设备分配

- **Encoder**: 所有操作在 `cuda:0`
- **DiT+Decoder**: 所有操作在 `cuda:1`
- **数据传输**: 通过CPU（打包/解包）模拟网络传输

## 验证方法

运行双GPU模式后，应该看到类似输出：

```
============================================================
使用双GPU模式：
  - Encoder: cuda:0 (模拟云端)
  - DiT+Decoder: cuda:1 (模拟边缘)
============================================================

加载Encoder模型到 cuda:0...
✓ Encoder已加载到 cuda:0

加载DiT+Decoder模型到 cuda:1...
✓ DiT+Decoder已加载到 cuda:1
```

## 下一步

1. ✅ 双GPU模式已实现
2. ⏳ 测试验证功能正确性
3. ⏳ 如果测试通过，可以开始实现HTTP服务（将bytes通过HTTP传输）

## 注意事项

- 双GPU模式会占用两张GPU的显存
- 打包/解包过程会有少量CPU内存开销
- 如果GPU数量不足，会自动回退到单GPU模式

