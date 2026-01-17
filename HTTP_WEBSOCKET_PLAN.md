# HTTP/WebSocket 通信方案详细计划

## 📋 一、整体架构设计

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     客户端 (Client)                          │
│  - 读取视频/图像                                             │
│  - 发送请求到云端                                            │
│  - 接收最终结果                                              │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/WebSocket
                     │
        ┌────────────▼────────────┐
        │   云端服务器 (Cloud)     │
        │  - Encoder服务           │
        │  - GPU: cuda:0           │
        │  - 接收: 原始帧对         │
        │  - 返回: 编码结果         │
        └────────────┬─────────────┘
                     │ HTTP/WebSocket
                     │
        ┌────────────▼────────────┐
        │  边缘服务器 (Edge)       │
        │  - DiT + Decoder服务     │
        │  - GPU: cuda:1           │
        │  - 接收: 编码结果         │
        │  - 返回: 生成帧          │
        └────────────┬─────────────┘
                     │
        ┌────────────▼────────────┐
        │      客户端 (Client)      │
        │  - 接收最终结果           │
        │  - 保存视频/图像          │
        └─────────────────────────┘
```

### 1.2 通信流程

**方案A：客户端直接连接两个服务器（推荐）**
```
客户端 → 云端(Encoder) → 客户端 → 边缘(DiT+Decoder) → 客户端
```

**方案B：边缘服务器作为代理**
```
客户端 → 边缘 → 云端(Encoder) → 边缘(DiT+Decoder) → 客户端
```

**推荐方案A**，因为：
- 客户端可以并行处理多个请求
- 边缘服务器不需要等待云端响应
- 更符合真实的云端-边缘架构

---

## 📦 二、服务端设计

### 2.1 云端服务器（Encoder服务）

#### 2.1.1 服务职责
- 接收原始帧对（frame0, frame1）
- 执行encoder编码
- 返回编码结果（cond_dit, cond_dec, stats等）

#### 2.1.2 API设计

**HTTP RESTful API（推荐用于单次请求）**

```
POST /api/v1/encode
Content-Type: multipart/form-data 或 application/json

请求体：
{
    "frame0": base64编码的图像数据 或 文件上传,
    "frame1": base64编码的图像数据 或 文件上传,
    "image_size": [H, W],  # 可选，用于padder
    "metadata": {
        "fps": 30,  # 可选
        "frame_id": 0  # 可选，用于追踪
    }
}

响应：
{
    "status": "success",
    "data": {
        "enc_out": base64编码的打包数据,
        "enc_out_size": 123456,  # 字节数
        "metadata": {
            "ph": 32,
            "pw": 32,
            "processing_time": 0.05  # 秒
        }
    },
    "error": null
}
```

**WebSocket API（推荐用于流式处理）**

```
连接: ws://cloud-server:8000/ws/encode

客户端发送：
{
    "type": "encode",
    "frame0": base64编码的图像数据,
    "frame1": base64编码的图像数据,
    "request_id": "unique_id_123",
    "metadata": {...}
}

服务端响应：
{
    "type": "encode_result",
    "request_id": "unique_id_123",
    "status": "success",
    "data": {
        "enc_out": base64编码的打包数据,
        "metadata": {...}
    }
}
```

#### 2.1.3 技术选型

**框架选择：**
- **FastAPI**（推荐）
  - 支持HTTP和WebSocket
  - 自动生成API文档
  - 异步支持好
  - 类型检查

- **Flask + Flask-SocketIO**（备选）
  - 简单易用
  - 但异步性能不如FastAPI

**依赖库：**
```python
fastapi
uvicorn[standard]  # ASGI服务器
python-multipart   # 文件上传支持
websockets         # WebSocket支持（FastAPI内置）
```

#### 2.1.4 服务端代码结构

```
cloud_server/
├── main.py                 # FastAPI应用入口
├── models/
│   ├── encoder_service.py  # Encoder服务类
│   └── eden_encoder.py     # EDEN encoder封装
├── api/
│   ├── routes.py           # HTTP路由
│   └── websocket.py        # WebSocket路由
├── utils/
│   ├── image_utils.py      # 图像处理工具
│   └── serialization.py    # 序列化工具
└── config.py               # 配置管理
```

---

### 2.2 边缘服务器（DiT+Decoder服务）

#### 2.2.1 服务职责
- 接收编码结果（enc_out）
- 执行DiT扩散采样
- 执行Decoder解码
- 返回生成的中间帧

#### 2.2.2 API设计

**HTTP RESTful API**

```
POST /api/v1/interpolate
Content-Type: application/json

请求体：
{
    "enc_out": base64编码的打包数据,
    "difference": 0.5,  # 余弦相似度（可选，可在边缘计算）
    "vae_scaler": 0.18215,  # VAE参数
    "vae_shift": 0.0,
    "model_args": {
        "latent_dim": 16,
        ...
    },
    "sampling_config": {
        "num_steps": 2,
        "method": "euler",
        ...
    },
    "metadata": {
        "ph": 32,
        "pw": 32,
        "request_id": "unique_id_123"
    }
}

响应：
{
    "status": "success",
    "data": {
        "generated_frame": base64编码的图像数据,
        "metadata": {
            "processing_time": 0.25,  # 秒
            "request_id": "unique_id_123"
        }
    },
    "error": null
}
```

**WebSocket API**

```
连接: ws://edge-server:8001/ws/interpolate

客户端发送：
{
    "type": "interpolate",
    "enc_out": base64编码的打包数据,
    "request_id": "unique_id_123",
    "metadata": {...}
}

服务端响应：
{
    "type": "interpolate_result",
    "request_id": "unique_id_123",
    "status": "success",
    "data": {
        "generated_frame": base64编码的图像数据,
        "metadata": {...}
    }
}
```

#### 2.2.3 技术选型

同云端服务器，使用FastAPI。

#### 2.2.4 服务端代码结构

```
edge_server/
├── main.py                 # FastAPI应用入口
├── models/
│   ├── dit_decoder_service.py  # DiT+Decoder服务类
│   └── eden_dit_decoder.py    # EDEN DiT+Decoder封装
├── api/
│   ├── routes.py           # HTTP路由
│   └── websocket.py        # WebSocket路由
├── utils/
│   ├── image_utils.py      # 图像处理工具
│   └── serialization.py    # 序列化工具
└── config.py               # 配置管理
```

---

## 🔄 三、客户端设计

### 3.1 客户端职责

- 读取视频/图像
- 连接云端服务器（发送帧对，接收编码结果）
- 连接边缘服务器（发送编码结果，接收生成帧）
- 组装最终结果
- 保存视频/图像

### 3.2 客户端代码结构

```
client/
├── inference_client.py      # 主客户端类
├── cloud_client.py          # 云端客户端封装
├── edge_client.py           # 边缘客户端封装
├── video_processor.py       # 视频处理逻辑
└── utils/
    ├── image_utils.py       # 图像处理工具
    └── serialization.py    # 序列化工具
```

### 3.3 客户端使用方式

**方式1：修改现有inference.py**
- 将`interpolate()`函数改为调用HTTP/WebSocket客户端
- 保持原有命令行接口

**方式2：创建新的客户端脚本**
- `inference_http.py` - 使用HTTP
- `inference_ws.py` - 使用WebSocket

---

## 📡 四、数据传输设计

### 4.1 数据序列化方案

#### 方案1：使用现有的pack_enc_out（推荐）

**优点：**
- 已经实现
- 使用torch.save，兼容性好
- 压缩效率高

**缺点：**
- 需要base64编码（增加33%体积）
- 二进制数据，调试困难

**实现：**
```python
# 编码端
enc_out = eden.encode(cond_frames)
blob = pack_enc_out(enc_out)  # bytes
blob_b64 = base64.b64encode(blob).decode('utf-8')  # base64字符串

# 解码端
blob = base64.b64decode(blob_b64)  # bytes
enc_out = unpack_enc_out(blob, device)  # dict
```

#### 方案2：JSON + base64（备选）

**优点：**
- 人类可读
- 调试方便
- 跨语言支持好

**缺点：**
- 体积大（base64增加33%）
- 序列化/反序列化慢

**实现：**
```python
# 编码端
enc_out_json = {
    "cond_dit": base64.b64encode(cond_dit.cpu().numpy().tobytes()).decode(),
    "cond_dit_shape": list(cond_dit.shape),
    "cond_dit_dtype": str(cond_dit.dtype),
    # ... 其他字段
}

# 解码端
cond_dit_bytes = base64.b64decode(enc_out_json["cond_dit"])
cond_dit = torch.frombuffer(cond_dit_bytes, dtype=...).reshape(...)
```

#### 方案3：MessagePack（高性能备选）

**优点：**
- 比JSON快
- 比JSON体积小
- 支持二进制数据

**缺点：**
- 需要额外依赖
- 调试不如JSON方便

**推荐：方案1（pack_enc_out + base64）**

---

### 4.2 图像数据传输

**方式1：base64编码（简单）**
```python
# 编码
import base64
from PIL import Image
import io

img_bytes = io.BytesIO()
Image.fromarray(frame).save(img_bytes, format='PNG')
img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

# 解码
img_bytes = base64.b64decode(img_b64)
frame = Image.open(io.BytesIO(img_bytes))
```

**方式2：multipart/form-data（文件上传）**
```python
# 客户端
files = {
    'frame0': ('frame0.png', frame0_bytes, 'image/png'),
    'frame1': ('frame1.png', frame1_bytes, 'image/png')
}
response = requests.post(url, files=files)

# 服务端
frame0_file = request.files['frame0']
frame0 = Image.open(frame0_file.stream)
```

**推荐：方式1（base64），简单直接**

---

### 4.3 请求/响应格式

#### HTTP请求格式

```python
# 请求头
{
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-Request-ID": "unique_id_123",  # 用于追踪
    "X-Client-Version": "1.0.0"
}

# 请求体（Encoder）
{
    "frame0": "base64_string...",
    "frame1": "base64_string...",
    "metadata": {
        "image_size": [480, 640],
        "frame_id": 0
    }
}

# 请求体（DiT+Decoder）
{
    "enc_out": "base64_string...",
    "difference": 0.5,
    "vae_scaler": 0.18215,
    "vae_shift": 0.0,
    "metadata": {
        "ph": 32,
        "pw": 32,
        "request_id": "unique_id_123"
    }
}
```

#### HTTP响应格式

```python
# 成功响应
{
    "status": "success",
    "data": {
        "enc_out": "base64_string...",  # 或 "generated_frame": "..."
        "metadata": {
            "processing_time": 0.05,
            "request_id": "unique_id_123"
        }
    },
    "error": null
}

# 错误响应
{
    "status": "error",
    "data": null,
    "error": {
        "code": "ENCODE_ERROR",
        "message": "Failed to encode frames",
        "details": "..."
    }
}
```

#### WebSocket消息格式

```python
# 客户端发送
{
    "type": "encode",  # 或 "interpolate"
    "request_id": "unique_id_123",
    "payload": {
        "frame0": "base64_string...",
        "frame1": "base64_string..."
    },
    "metadata": {...}
}

# 服务端响应
{
    "type": "encode_result",  # 或 "interpolate_result"
    "request_id": "unique_id_123",
    "status": "success",
    "payload": {
        "enc_out": "base64_string..."
    },
    "metadata": {...}
}

# 错误响应
{
    "type": "error",
    "request_id": "unique_id_123",
    "status": "error",
    "error": {
        "code": "ENCODE_ERROR",
        "message": "..."
    }
}
```

---

## 🛠️ 五、实现步骤

### 阶段1：基础HTTP服务（推荐先实现）

#### 步骤1.1：创建云端Encoder服务
1. 创建`cloud_server/`目录结构
2. 实现FastAPI应用（`main.py`）
3. 实现Encoder服务类（加载模型、编码逻辑）
4. 实现HTTP路由（`POST /api/v1/encode`）
5. 实现图像序列化/反序列化工具
6. 实现错误处理和日志

#### 步骤1.2：创建边缘DiT+Decoder服务
1. 创建`edge_server/`目录结构
2. 实现FastAPI应用（`main.py`）
3. 实现DiT+Decoder服务类（加载模型、采样+解码逻辑）
4. 实现HTTP路由（`POST /api/v1/interpolate`）
5. 实现数据序列化/反序列化工具
6. 实现错误处理和日志

#### 步骤1.3：创建客户端
1. 创建`client/`目录结构
2. 实现云端客户端（发送帧对，接收编码结果）
3. 实现边缘客户端（发送编码结果，接收生成帧）
4. 实现视频处理逻辑（循环处理帧对）
5. 修改或创建`inference_http.py`

#### 步骤1.4：测试和验证
1. 启动云端服务（`uvicorn cloud_server.main:app --port 8000`）
2. 启动边缘服务（`uvicorn edge_server.main:app --port 8001`）
3. 运行客户端测试单对帧
4. 运行客户端测试视频
5. 对比结果和性能

---

### 阶段2：WebSocket支持（可选，用于流式处理）

#### 步骤2.1：云端WebSocket支持
1. 实现WebSocket路由（`/ws/encode`）
2. 实现消息处理逻辑
3. 实现连接管理（多客户端支持）
4. 实现心跳机制（保持连接）

#### 步骤2.2：边缘WebSocket支持
1. 实现WebSocket路由（`/ws/interpolate`）
2. 实现消息处理逻辑
3. 实现连接管理
4. 实现心跳机制

#### 步骤2.3：客户端WebSocket支持
1. 实现WebSocket客户端类
2. 实现异步消息处理
3. 实现重连机制
4. 创建`inference_ws.py`

---

### 阶段3：优化和增强

#### 步骤3.1：性能优化
1. 实现请求批处理（多个帧对一起处理）
2. 实现异步处理（使用asyncio）
3. 实现连接池（复用HTTP连接）
4. 实现数据压缩（gzip）

#### 步骤3.2：可靠性增强
1. 实现重试机制（网络错误自动重试）
2. 实现超时处理
3. 实现请求队列（防止过载）
4. 实现健康检查接口（`/health`）

#### 步骤3.3：监控和日志
1. 实现请求日志（记录每个请求）
2. 实现性能监控（处理时间、吞吐量）
3. 实现错误追踪
4. 实现指标收集（Prometheus格式）

---

## 🔧 六、技术细节

### 6.1 服务启动方式

**云端服务：**
```bash
# 开发模式
uvicorn cloud_server.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
gunicorn cloud_server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**边缘服务：**
```bash
# 开发模式
uvicorn edge_server.main:app --host 0.0.0.0 --port 8001 --reload

# 生产模式
gunicorn edge_server.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

### 6.2 配置管理

**使用环境变量或配置文件：**
```python
# config.py
import os

CLOUD_SERVER_URL = os.getenv("CLOUD_SERVER_URL", "http://localhost:8000")
EDGE_SERVER_URL = os.getenv("EDGE_SERVER_URL", "http://localhost:8001")

MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoints/eden.pth")
DEVICE = os.getenv("DEVICE", "cuda:0")  # 云端用cuda:0，边缘用cuda:1
```

### 6.3 错误处理策略

**客户端错误处理：**
```python
try:
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    return response.json()
except requests.exceptions.Timeout:
    # 超时重试
    retry()
except requests.exceptions.ConnectionError:
    # 连接错误重试
    retry()
except requests.exceptions.HTTPError as e:
    # HTTP错误（4xx, 5xx）
    handle_error(e)
```

**服务端错误处理：**
```python
try:
    result = process_request(data)
    return {"status": "success", "data": result}
except ModelError as e:
    return {"status": "error", "error": {"code": "MODEL_ERROR", "message": str(e)}}
except Exception as e:
    logger.exception("Unexpected error")
    return {"status": "error", "error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}
```

### 6.4 并发处理

**服务端并发：**
- 使用FastAPI的异步支持
- 使用`asyncio`处理多个请求
- 注意GPU资源竞争（可能需要请求队列）

**客户端并发：**
- 可以并行发送多个请求（如果服务端支持）
- 使用`asyncio`或`concurrent.futures`

---

## 📊 七、性能考虑

### 7.1 网络延迟

**HTTP：**
- 每次请求都有TCP握手开销
- 适合单次请求，不适合高频请求

**WebSocket：**
- 建立连接后复用，延迟低
- 适合流式处理、高频请求

### 7.2 数据传输量

**单对帧的encoder输出：**
- 假设：cond_dit [2, 1024, 768] + cond_dec [2, 1024, 768]
- 大小：约 2 × 1024 × 768 × 4 bytes × 2 = ~12.5 MB
- base64编码后：~16.7 MB

**优化建议：**
- 使用gzip压缩（可减少50-70%）
- 使用更高效的序列化格式（MessagePack）

### 7.3 服务端资源

**GPU内存：**
- 每个请求需要加载模型到GPU
- 考虑模型常驻内存（避免重复加载）

**CPU资源：**
- 序列化/反序列化消耗CPU
- 考虑使用多进程处理

---

## 🎯 八、实施优先级

### 高优先级（必须实现）
1. ✅ 云端HTTP服务（Encoder）
2. ✅ 边缘HTTP服务（DiT+Decoder）
3. ✅ HTTP客户端
4. ✅ 基础错误处理
5. ✅ 单对帧测试

### 中优先级（重要功能）
1. ⏳ 视频处理支持
2. ⏳ 请求重试机制
3. ⏳ 超时处理
4. ⏳ 健康检查接口
5. ⏳ 日志系统

### 低优先级（优化功能）
1. ⏳ WebSocket支持
2. ⏳ 批处理支持
3. ⏳ 性能监控
4. ⏳ 数据压缩
5. ⏳ 连接池

---

## 📝 九、文件结构总结

```
EDEN-main/
├── cloud_server/              # 云端服务（新增）
│   ├── main.py
│   ├── models/
│   │   ├── encoder_service.py
│   │   └── eden_encoder.py
│   ├── api/
│   │   ├── routes.py
│   │   └── websocket.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── serialization.py
│   └── config.py
│
├── edge_server/               # 边缘服务（新增）
│   ├── main.py
│   ├── models/
│   │   ├── dit_decoder_service.py
│   │   └── eden_dit_decoder.py
│   ├── api/
│   │   ├── routes.py
│   │   └── websocket.py
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── serialization.py
│   └── config.py
│
├── client/                    # 客户端（新增）
│   ├── inference_client.py
│   ├── cloud_client.py
│   ├── edge_client.py
│   ├── video_processor.py
│   └── utils/
│       ├── image_utils.py
│       └── serialization.py
│
├── inference_http.py          # HTTP版本推理脚本（新增）
├── inference_ws.py            # WebSocket版本推理脚本（新增，可选）
│
├── inference.py               # 原有脚本（保持不变）
├── src/                       # 原有代码（保持不变）
└── ...
```

---

## ✅ 十、验证和测试计划

### 10.1 单元测试
- 测试序列化/反序列化
- 测试图像编码/解码
- 测试服务端API

### 10.2 集成测试
- 测试完整流程（客户端→云端→边缘→客户端）
- 测试错误处理
- 测试并发请求

### 10.3 性能测试
- 对比HTTP vs WebSocket性能
- 对比单GPU vs 双服务器性能
- 测试吞吐量和延迟

### 10.4 正确性验证
- 对比HTTP/WebSocket结果 vs 单GPU结果
- 确保生成的视频质量一致

---

## 🎓 总结

这个方案提供了完整的HTTP/WebSocket通信架构，包括：

1. **清晰的架构设计**：云端-边缘分离
2. **详细的API设计**：HTTP RESTful + WebSocket
3. **完整的数据传输方案**：序列化、图像编码
4. **分阶段实施计划**：从基础到高级
5. **技术选型建议**：FastAPI + 现有序列化工具

**下一步**：按照阶段1开始实现基础HTTP服务。

##运行
安装依赖（已在本地跑过）：/home/jiayu/miniconda3/envs/eden/bin/pip install fastapi 'uvicorn[standard]' pillow requests。
启动云端服务（终端1）：
   cd /home/jiayu/hengyi_zhang/EDEN-main   /home/jiayu/miniconda3/envs/eden/bin/uvicorn cloud_server.main:app --host 0.0.0.0 --port 8000
启动边缘服务（终端2）：
   cd /home/jiayu/hengyi_zhang/EDEN-main   /home/jiayu/miniconda3/envs/eden/bin/uvicorn edge_server.main:app --host 0.0.0.0 --port 8001
运行客户端（终端3）：
视频：/home/jiayu/miniconda3/envs/eden/bin/python client.py --video_path examples/0.mp4 --output_dir interpolation_outputs/http_client_test
单帧对：python client.py --frame_0_path path0 --frame_1_path path1