from src.models import load_model
from src.utils import InputPadder
from src.utils.encode_transfer import pack_enc_out, unpack_enc_out
from src.transport import create_transport, Sampler
import torchvision
import torch
import argparse
import yaml
import os
import numpy as np


def interpolate(frame0, frame1, use_split_gpu=False):
    """
    插值函数，支持单GPU和双GPU（encoder/DiT+decoder分离）模式
    
    Args:
        frame0: [1, 3, H, W] 第一帧
        frame1: [1, 3, H, W] 第二帧
        use_split_gpu: 是否使用双GPU模式（encoder在cuda:0，DiT+decoder在cuda:1）
    """
    h, w = frame0.shape[2:]
    image_size = [h, w]
    padder = InputPadder(image_size)
    
    if use_split_gpu:
        # ========== 双GPU模式：encoder在cuda:0，DiT+decoder在cuda:1 ==========
        device_enc = torch.device("cuda:0")
        device_ditdec = torch.device("cuda:1")
        
        # 1. 在encoder GPU上准备输入
        frame0_enc = frame0.to(device_enc)
        frame1_enc = frame1.to(device_enc)
        cond_frames = padder.pad(torch.cat((frame0_enc, frame1_enc), dim=0))  # [2, 3, H', W']
        
        # 计算difference（在encoder侧）
        difference = ((torch.mean(torch.cosine_similarity(frame0_enc, frame1_enc),
                                  dim=[1, 2]) - args.cos_sim_mean) / args.cos_sim_std).unsqueeze(1)
        
        # 2. 在encoder GPU上执行encode
        with torch.no_grad():
            enc_out = eden_enc.encode(cond_frames)  # dict，包含cond_dit, cond_dec, stats等
            
            # 3. 打包成bytes（模拟网络传输）
            blob = pack_enc_out(enc_out)  # 所有tensor已移到CPU
        
        # 4. 在DiT+decoder GPU上解包并执行扩散+解码
        with torch.no_grad():
            # 解包到cuda:1
            enc_out_dit = unpack_enc_out(blob, device_ditdec)  # 所有tensor都在cuda:1上了
            difference_dit = difference.to(device_ditdec)
            
            # 生成初始噪声（在cuda:1上）
            new_h, new_w = cond_frames.shape[2:]
            noise = torch.randn([1, new_h // 32 * new_w // 32, args.model_args["latent_dim"]], 
                              device=device_ditdec)
            
            # 构造denoise wrapper
            def denoise_wrapper(query_latents, t):
                """Wrapper函数，使用eden_ditdec进行去噪"""
                if isinstance(t, torch.Tensor):
                    if t.numel() == 1:
                        denoise_timestep = t.unsqueeze(0) if t.dim() == 0 else t
                    else:
                        denoise_timestep = t[0:1]
                else:
                    denoise_timestep = torch.tensor([t], device=device_ditdec, dtype=torch.float32)
                
                if denoise_timestep.dim() == 0:
                    denoise_timestep = denoise_timestep.unsqueeze(0)
                
                return eden_ditdec.denoise_from_tokens(query_latents, denoise_timestep, enc_out_dit, difference_dit)
            
            # 扩散采样（在cuda:1上）
            samples = sample_fn_ditdec(noise, denoise_wrapper)[-1]
            
            # 反归一化并解码（在cuda:1上）
            denoise_latents = samples / args.vae_scaler + args.vae_shift
            generated_frame = eden_ditdec.decode(denoise_latents)
            generated_frame = generated_frame.clamp(0., 1.)
            
            # 移回CPU并unpad（因为最终结果需要保存）
            generated_frame = padder.unpad(generated_frame.cpu())
        
        return generated_frame
    
    else:
        # ========== 单GPU模式（原版逻辑） ==========
        # 确保输入帧与模型在同一设备上
        frame0 = frame0.to(device)
        frame1 = frame1.to(device)

        difference = (
            (
                torch.mean(torch.cosine_similarity(frame0, frame1), dim=[1, 2])
                - args.cos_sim_mean
            )
            / args.cos_sim_std
        ).unsqueeze(1).to(device)
        
        # 1. Encoder: 将条件帧编码为tokens
        cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
        enc_out = eden.encode(cond_frames)
        
        # 2. 生成初始噪声
        new_h, new_w = cond_frames.shape[2:]
        noise = torch.randn([1, new_h // 32 * new_w // 32, args.model_args["latent_dim"]]).to(device)
        
        # 3. 构造denoise wrapper，使用denoise_from_tokens
        def denoise_wrapper(query_latents, t):
            """Wrapper函数，将ODE求解器的时间步t转换为denoise_from_tokens需要的格式"""
            if isinstance(t, torch.Tensor):
                if t.numel() == 1:
                    denoise_timestep = t.unsqueeze(0) if t.dim() == 0 else t
                else:
                    denoise_timestep = t[0:1]
            else:
                denoise_timestep = torch.tensor([t], device=query_latents.device, dtype=torch.float32)
            
            if denoise_timestep.dim() == 0:
                denoise_timestep = denoise_timestep.unsqueeze(0)
            
            return eden.denoise_from_tokens(query_latents, denoise_timestep, enc_out, difference)
        
        # 4. 扩散采样（使用新的denoise_wrapper）
        samples = sample_fn(noise, denoise_wrapper)[-1]
        
        # 5. 反归一化并解码
        denoise_latents = samples / args.vae_scaler + args.vae_shift
        generated_frame = eden.decode(denoise_latents)
        generated_frame = padder.unpad(generated_frame.clamp(0., 1.))
        return generated_frame


def compute_motion_scores(video_frames, topk_ratio=0.1):
    """
    计算当前视频中每一对相邻帧的运动分数（0~1，越大越动）。

    Args:
        video_frames: [T, 3, H, W] 的张量，已经是 float 且归一化到 [0,1]
        topk_ratio: 计算局部运动时选取的 top-k 像素比例

    Returns:
        motion_scores: 长度为 T-1 的 list，每个元素 ∈ [0,1]
    """
    num_frames = video_frames.shape[0]
    cos_vals = []
    local_vals = []

    for i in range(num_frames - 1):
        f0 = video_frames[i].unsqueeze(0)  # [1,3,H,W]
        f1 = video_frames[i + 1].unsqueeze(0)

        # --- 全局 cosine 相似度 ---
        cos = torch.cosine_similarity(f0, f1, dim=1)  # [1,H,W]
        cos_mean = cos.mean().item()
        cos_vals.append(cos_mean)

        # --- 局部 top-k 像素差 ---
        delta = (f0 - f1).abs().mean(dim=1)  # [1,H,W]
        delta_flat = delta.view(-1)
        k = max(1, int(topk_ratio * delta_flat.numel()))
        topk_vals, _ = torch.topk(delta_flat, k)
        local_mean = topk_vals.mean().item()
        local_vals.append(local_mean)

    cos_arr = np.array(cos_vals, dtype=np.float32)
    local_arr = np.array(local_vals, dtype=np.float32)

    # 1) 在本视频内部对 cos 做标准化，得到“全局运动”分数
    cos_mean_v = float(cos_arr.mean())
    cos_std_v = float(cos_arr.std() + 1e-6)
    diff_v = (cos_arr - cos_mean_v) / cos_std_v  # 越小/越负表示越动

    diff_min = float(diff_v.min())
    diff_max = float(diff_v.max())
    if diff_max - diff_min < 1e-6:
        global_motion = np.zeros_like(diff_v)
    else:
        diff_norm = (diff_v - diff_min) / (diff_max - diff_min)
        # 1 - diff_norm: 越大表示全局越动
        global_motion = 1.0 - diff_norm

    # 2) 对局部差分做 min-max 标准化
    local_min = float(local_arr.min())
    local_max = float(local_arr.max())
    if local_max - local_min < 1e-6:
        local_norm = np.zeros_like(local_arr)
    else:
        local_norm = (local_arr - local_min) / (local_max - local_min)

    # 3) 融合全局&局部，得到最终 motion_scores
    alpha = 0.5  # 全局/局部权重
    motion_scores = alpha * global_motion + (1.0 - alpha) * local_norm
    motion_scores = np.clip(motion_scores, 0.0, 1.0)
    return motion_scores.tolist()


def recursive_interp(frame0, frame1, depth, use_split_gpu=False):
    """
    递归插帧：
        depth = 0 -> 不插帧，返回 []
        depth = 1 -> 插 1 帧（中点），返回 [M]
        depth = 2 -> 插 3 帧（M0, M, M1）
    仍然复用 interpolate()，兼容单/双 GPU 模式。
    """
    if depth == 0:
        return []

    # 先算中点
    mid = interpolate(frame0, frame1, use_split_gpu=use_split_gpu)

    if depth == 1:
        return [mid]

    # depth >= 2: 对两侧再各插一层
    left_mids = recursive_interp(
        frame0, mid, depth - 1, use_split_gpu=use_split_gpu
    )
    right_mids = recursive_interp(
        mid, frame1, depth - 1, use_split_gpu=use_split_gpu
    )
    return left_mids + [mid] + right_mids


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/eval_eden.yaml")
parser.add_argument("--frame_0_path", type=str, default="examples/frame_0.jpg")
parser.add_argument("--frame_1_path", type=str, default="examples/frame_1.jpg")
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--interpolated_results_dir", type=str, default="interpolation_outputs")
parser.add_argument("--use_split_gpu", action="store_true", help="使用双GPU模式：encoder在cuda:0，DiT+decoder在cuda:1")
args = parser.parse_args()
with open(args.config, "r") as f:
    update_args = yaml.unsafe_load(f)
parser.set_defaults(**update_args)
args = parser.parse_args()

model_name = args.model_name
ckpt = torch.load(args.pretrained_eden_path, map_location="cpu")

if args.use_split_gpu:
    # ========== 双GPU模式：encoder和DiT+decoder分离 ==========
    print("=" * 60)
    print("使用双GPU模式：")
    print("  - Encoder: cuda:0 (模拟云端)")
    print("  - DiT+Decoder: cuda:1 (模拟边缘)")
    print("=" * 60)
    
    # 检查是否有两张GPU
    if torch.cuda.device_count() < 2:
        print(f"警告: 检测到只有 {torch.cuda.device_count()} 张GPU，需要至少2张GPU才能使用双GPU模式")
        print("将回退到单GPU模式")
        args.use_split_gpu = False
    else:
        device_enc = torch.device("cuda:0")
        device_ditdec = torch.device("cuda:1")
        
        # 创建encoder模型（在cuda:0）
        print(f"\n加载Encoder模型到 {device_enc}...")
        eden_enc = load_model(model_name, **args.model_args)
        eden_enc.load_state_dict(ckpt["eden"])
        eden_enc.to(device_enc)
        eden_enc.eval()
        print(f"✓ Encoder已加载到 {device_enc}")
        
        # 创建DiT+decoder模型（在cuda:1）
        print(f"\n加载DiT+Decoder模型到 {device_ditdec}...")
        eden_ditdec = load_model(model_name, **args.model_args)
        eden_ditdec.load_state_dict(ckpt["eden"])
        eden_ditdec.to(device_ditdec)
        eden_ditdec.eval()
        print(f"✓ DiT+Decoder已加载到 {device_ditdec}")
        
        # 创建sampler（用于DiT+decoder GPU）
        transport_ditdec = create_transport("Linear", "velocity")
        sampler_ditdec = Sampler(transport_ditdec)
        sample_fn_ditdec = sampler_ditdec.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)
        
        # 为了兼容性，也保留单GPU的变量（但不会用到）
        device = device_enc
        eden = eden_enc
        transport = create_transport("Linear", "velocity")
        sampler = Sampler(transport)
        sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)

if not args.use_split_gpu:
    # ========== 单GPU模式（原版） ==========
    device = "cuda:0"
    eden = load_model(model_name, **args.model_args)
    eden.load_state_dict(ckpt["eden"])
    eden.to(device)
    eden.eval()
    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3
    )

del ckpt
video_path = args.video_path
interpolated_results_dir = args.interpolated_results_dir
os.makedirs(interpolated_results_dir, exist_ok=True)
frame_0_path, frame_1_path = args.frame_0_path, args.frame_1_path
if video_path:
    print(f"Interpolating Video ({video_path}) ...")
    interpolated_video_save_path = f"{interpolated_results_dir}/interpolated.mp4"

    # 1. 读取视频
    video_frames, _, video_info = torchvision.io.read_video(video_path)
    video_frames = video_frames.float().permute(0, 3, 1, 2) / 255.0
    fps = float(video_info["video_fps"])
    frames_num = video_frames.shape[0]
    print(f"Input video: {frames_num} frames, fps={fps}")

    # 2. 第一遍：运动分析（per-video 标准化 + 局部 top-k）
    print("Computing motion scores for each frame interval...")
    motion_scores = compute_motion_scores(video_frames, topk_ratio=0.1)
    motion_arr = np.array(motion_scores, dtype=np.float32)

    if motion_arr.max() - motion_arr.min() < 1e-6:
        # 所有区段运动差不多：统一插 1 帧
        depths = [1] * (frames_num - 1)
    else:
        # 用中位数把区段切成两档：低/高 运动
        mid_th = float(np.quantile(motion_arr, 0.5))
        depths = []
        for s in motion_scores:
            if s >= mid_th:
                depth = 2    # 高运动：插 3 帧
            else:
                depth = 1    # 低/中运动：插 1 帧
            depths.append(depth)

    print(
        "Motion score stats:",
        f"min={motion_arr.min():.4f}, max={motion_arr.max():.4f}, mean={motion_arr.mean():.4f}",
    )
    print(
        "Depth distribution (1/2):",
        depths.count(1),
        depths.count(2),
    )

    # 3. 第二遍：根据 depth 做递归插帧
    interpolated_video = []
    for i in range(frames_num - 1):
        frame_0 = video_frames[i].unsqueeze(0)  # [1,3,H,W]
        frame_1 = video_frames[i + 1].unsqueeze(0)
        depth = depths[i]

        # 先放原始帧0
        interpolated_video.append(frame_0.cpu())

        if depth > 0:
            with torch.no_grad():
                mids = recursive_interp(
                    frame_0,
                    frame_1,
                    depth,
                    use_split_gpu=args.use_split_gpu,
                )
            for m in mids:
                interpolated_video.append(m.cpu())
            del mids

        # 清理显存
        if args.use_split_gpu:
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()

        del frame_0, frame_1

    # 最后一帧
    interpolated_video.append(video_frames[-1].unsqueeze(0))

    # 4. 计算新的 fps，使总时长尽量保持不变
    orig_frames = frames_num
    new_frames = len(interpolated_video)
    fps_out = fps * (new_frames - 1) / max(1, (orig_frames - 1))
    print(
        f"Original frames: {orig_frames}, New frames: {new_frames}, fps_out ≈ {fps_out:.4f}"
    )

    # 5. 写回视频
    interpolated_video = (
        torch.cat(interpolated_video, dim=0).permute(0, 2, 3, 1) * 255.0
    ).cpu()
    torchvision.io.write_video(
        interpolated_video_save_path, interpolated_video, fps=fps_out
    )
    print(f"Saved interpolated video in {interpolated_video_save_path}.")
elif frame_0_path and frame_1_path:
    print(f"Interpolating Image-pairs {frame_0_path}-{frame_1_path} ...")
    frame_0 = (torchvision.io.read_image(frame_0_path) / 255.).unsqueeze(0)
    frame_1 = (torchvision.io.read_image(frame_1_path) / 255.).unsqueeze(0)
    interpolated_frame = interpolate(frame_0, frame_1, use_split_gpu=args.use_split_gpu)
    interpolated_frame_path = f"{interpolated_results_dir}/interpolated.png"
    torchvision.utils.save_image(interpolated_frame, interpolated_frame_path)
    print(f"Saved interpolated image in {interpolated_frame_path}.")
else:
    assert "There are no images or videos to be interpolated!"

