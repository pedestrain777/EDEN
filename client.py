import argparse
import base64
import io
import os

import numpy as np
import requests
import torch
import torchvision
from PIL import Image


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

    # 1) 在本视频内部对 cos 做标准化，得到"全局运动"分数
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


def tensor_to_b64_png(tensor: torch.Tensor) -> str:
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise ValueError("Expected tensor shape [1, 3, H, W]")
    array = (tensor.clamp(0.0, 1.0)[0] * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def b64_png_to_tensor(b64_str: str) -> torch.Tensor:
    data = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        np_img = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


def call_encoder(url: str, frame0: torch.Tensor, frame1: torch.Tensor, max_retries: int = 3) -> dict:
    """调用encoder服务，带重试机制"""
    payload = {
        "frame0": tensor_to_b64_png(frame0),
        "frame1": tensor_to_b64_png(frame1),
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                raise ConnectionError(f"无法连接到encoder服务 {url}。请确保服务器已启动。") from e
            print(f"连接失败，重试 {attempt + 1}/{max_retries}...")
            import time
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Encoder服务请求失败: {e}") from e


def call_ditdec(url: str, blob: str, difference: float, height: int, width: int, max_retries: int = 3) -> torch.Tensor:
    """调用DiT+decoder服务，带重试机制"""
    payload = {
        "blob": blob,
        "difference": difference,
        "height": height,
        "width": width,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            frame_tensor = b64_png_to_tensor(data["frame"])
            return frame_tensor
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                raise ConnectionError(f"无法连接到DiT+decoder服务 {url}。请确保服务器已启动。") from e
            print(f"连接失败，重试 {attempt + 1}/{max_retries}...")
            import time
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"DiT+decoder服务请求失败: {e}") from e


def interpolate_http(encoder_url: str, ditdec_url: str, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
    """
    通过HTTP调用实现单帧插值（中点）。
    
    Args:
        encoder_url: encoder服务URL
        ditdec_url: DiT+decoder服务URL
        frame0: [1, 3, H, W] 第一帧
        frame1: [1, 3, H, W] 第二帧
    
    Returns:
        mid: [1, 3, H, W] 中间帧
    """
    enc_response = call_encoder(encoder_url, frame0, frame1)
    mid_tensor = call_ditdec(
        ditdec_url,
        blob=enc_response["blob"],
        difference=enc_response["difference"],
        height=enc_response["height"],
        width=enc_response["width"],
    )
    return mid_tensor


def recursive_interp_http(encoder_url: str, ditdec_url: str, frame0: torch.Tensor, frame1: torch.Tensor, depth: int) -> list:
    """
    通过HTTP调用实现递归插帧：
        depth = 1 -> 插 1 帧（中点），返回 [M]
        depth = 2 -> 插 3 帧（M0, M, M1）
    
    Args:
        encoder_url: encoder服务URL
        ditdec_url: DiT+decoder服务URL
        frame0: [1, 3, H, W] 第一帧
        frame1: [1, 3, H, W] 第二帧
        depth: 插帧深度（1或2）
    
    Returns:
        list of [1, 3, H, W] tensors，插值得到的中间帧列表
    """
    if depth == 0:
        return []
    
    # 先算中点
    mid = interpolate_http(encoder_url, ditdec_url, frame0, frame1)
    
    if depth == 1:
        return [mid]
    
    # depth == 2: 对两侧再各插一层
    left_mids = recursive_interp_http(encoder_url, ditdec_url, frame0, mid, depth - 1)
    right_mids = recursive_interp_http(encoder_url, ditdec_url, mid, frame1, depth - 1)
    return left_mids + [mid] + right_mids


def process_video(encoder_url: str, ditdec_url: str, video_path: str, output_dir: str, use_adaptive: bool = True):
    """
    处理视频，支持自适应插帧（根据运动分数决定插1帧还是3帧）。
    
    Args:
        encoder_url: encoder服务URL
        ditdec_url: DiT+decoder服务URL
        video_path: 输入视频路径
        output_dir: 输出目录
        use_adaptive: 是否使用自适应插帧（True=动态插帧，False=固定插1帧）
    """
    print(f"Loading video: {video_path}")
    frames, _, info = torchvision.io.read_video(video_path)
    frames = frames.float().permute(0, 3, 1, 2) / 255.0
    fps = float(info["video_fps"])
    frames_num = frames.shape[0]
    print(f"Input video: {frames_num} frames, fps={fps}")

    if use_adaptive:
        # ========== 自适应插帧模式 ==========
        # 1. 第一遍：运动分析
        print("Computing motion scores for each frame interval...")
        motion_scores = compute_motion_scores(frames, topk_ratio=0.1)
        motion_arr = np.array(motion_scores, dtype=np.float32)
        
        # 2. 根据中位数决定depth（只区分插1帧和插3帧）
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
        interpolated = []
        for i in range(frames_num - 1):
            frame0 = frames[i].unsqueeze(0)  # [1,3,H,W]
            frame1 = frames[i + 1].unsqueeze(0)
            depth = depths[i]
            
            # 先放原始帧0
            interpolated.append(frame0.cpu())
            
            # 递归插帧
            print(f"Processing segment {i+1}/{frames_num-1} (depth={depth})...")
            mids = recursive_interp_http(encoder_url, ditdec_url, frame0, frame1, depth)
            for m in mids:
                interpolated.append(m.cpu())
            
            del frame0, frame1, mids
        
        # 最后一帧
        interpolated.append(frames[-1].unsqueeze(0).cpu())
        
        # 4. 计算新的 fps，使总时长尽量保持不变
        orig_frames = frames_num
        new_frames = len(interpolated)
        fps_out = fps * (new_frames - 1) / max(1, (orig_frames - 1))
        print(
            f"Original frames: {orig_frames}, New frames: {new_frames}, fps_out ≈ {fps_out:.4f}"
        )
    else:
        # ========== 固定插帧模式（每对帧之间插1帧） ==========
        print("Using fixed interpolation (1 frame per segment)...")
        interpolated = []
        for idx in range(frames_num - 1):
            frame0 = frames[idx].unsqueeze(0)
            frame1 = frames[idx + 1].unsqueeze(0)
            print(f"Processing segment {idx+1}/{frames_num-1}...")
            mid_tensor = interpolate_http(encoder_url, ditdec_url, frame0, frame1)
            interpolated.append(frame0.cpu())
            interpolated.append(mid_tensor.cpu())
            del frame0, frame1, mid_tensor
        interpolated.append(frames[-1].unsqueeze(0).cpu())
        fps_out = 2 * fps  # 固定插1帧，fps翻倍
    
    # 5. 写回视频
    video_tensor = torch.cat(interpolated, dim=0).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "interpolated_http.mp4")
    torchvision.io.write_video(save_path, (video_tensor * 255.0).byte(), fps=fps_out)
    print(f"Saved interpolated video to {save_path}")


def process_pair(encoder_url: str, ditdec_url: str, frame0_path: str, frame1_path: str, output_dir: str):
    frame0 = (torchvision.io.read_image(frame0_path).float() / 255.0).unsqueeze(0)
    frame1 = (torchvision.io.read_image(frame1_path).float() / 255.0).unsqueeze(0)
    enc_response = call_encoder(encoder_url, frame0, frame1)
    mid_tensor = call_ditdec(
        ditdec_url,
        blob=enc_response["blob"],
        difference=enc_response["difference"],
        height=enc_response["height"],
        width=enc_response["width"],
    )
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "interpolated_http.png")
    torchvision.utils.save_image(mid_tensor, save_path)
    print(f"Saved interpolated image to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="EDEN HTTP client")
    parser.add_argument("--encoder_url", default="http://127.0.0.1:8000/encode")
    parser.add_argument("--ditdec_url", default="http://127.0.0.1:8001/interpolate")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--frame_0_path", type=str, default=None)
    parser.add_argument("--frame_1_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="interpolation_outputs/http_client")
    parser.add_argument("--use_adaptive", action="store_true", help="使用自适应插帧（根据运动分数动态决定插1帧或3帧）")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.video_path:
        process_video(args.encoder_url, args.ditdec_url, args.video_path, args.output_dir, use_adaptive=args.use_adaptive)
    elif args.frame_0_path and args.frame_1_path:
        process_pair(args.encoder_url, args.ditdec_url, args.frame_0_path, args.frame_1_path, args.output_dir)
    else:
        raise ValueError("Please provide either --video_path or both --frame_0_path and --frame_1_path.")


if __name__ == "__main__":
    main()

