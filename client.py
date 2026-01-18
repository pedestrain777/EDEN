import argparse
import base64
import io
import os

import numpy as np
import requests
import torch
import torchvision
from PIL import Image

from src.scheduler.greedy_refine import greedy_refine


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


def compute_pair_score(frame0: torch.Tensor, frame1: torch.Tensor, topk_ratio: float = 0.1, alpha: float = 0.5) -> float:
    """
    给任意两帧打一个"需要插帧"的分数（越大越需要优先插）。

    frame0/frame1: [1,3,H,W]，值域[0,1]
    """
    with torch.no_grad():
        # global: 1 - cosine similarity
        cos = torch.cosine_similarity(frame0, frame1, dim=1).mean().item()
        # 对于正值域图像，cos通常在[0,1]，直接用 1-cos 更直观
        global_motion = 1.0 - max(0.0, min(1.0, cos))

        # local: top-k pixel RGB difference
        delta = (frame0 - frame1).abs().mean(dim=1)  # [1,H,W]
        delta_flat = delta.view(-1)
        k = max(1, int(topk_ratio * delta_flat.numel()))
        topk_vals, _ = torch.topk(delta_flat, k)
        local_mean = topk_vals.mean().item()  # 理论上就在[0,1]

        s = alpha * global_motion + (1.0 - alpha) * local_mean
        return float(max(0.0, min(1.0, s)))


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
        # ========== 自适应插帧模式（Greedy refine，直到达到24fps） ==========
        # 目标：填充成每秒24帧（24fps），而不是总帧数24
        # 计算视频时长：duration = frames_num / fps
        # 计算目标帧数：target_frames = duration * 24 = frames_num * 24 / fps
        target_fps = 24
        video_duration = frames_num / fps  # 视频时长（秒）
        target_len = int(video_duration * target_fps)  # 达到24fps需要的总帧数
        
        # 1) 构造初始帧序列（这里用视频输入的全部帧作为初始帧）
        #    如果你后面要"随机取帧/均匀取帧"当关键帧，这里换成你挑出来的 keyframes 即可
        init_frames = [frames[i].unsqueeze(0).cpu() for i in range(frames_num)]
        if len(init_frames) > target_len:
            raise ValueError(f"initial frames ({len(init_frames)}) > target_len ({target_len}). "
                             f"请先做关键帧/稀疏采样，或调整输入视频。")
        
        print(f"Greedy refinement: {len(init_frames)} frames -> {target_len} frames (24fps)")
        print(f"Video duration: {video_duration:.2f}s, Target FPS: {target_fps}")
        
        def score_fn(a: torch.Tensor, b: torch.Tensor) -> float:
            return compute_pair_score(a, b, topk_ratio=0.1, alpha=0.5)
        
        def interp_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # 这里直接复用你已有的 HTTP 单次中点插帧
            return interpolate_http(encoder_url, ditdec_url, a, b)
        
        # 2) Greedy refine：每次挑"最需要插"的区间插一张中点，直到达到24fps的帧数
        # 生成log文件路径（与输出视频在同一目录）
        log_file_path = os.path.join(output_dir, "greedy_refinement.log")
        
        refined = greedy_refine(
            init_frames,
            target_len=target_len,
            score_fn=score_fn,
            interp_fn=interp_fn,
            verbose=True,
            log_file=log_file_path,
        )
        
        # 3) fps_out：固定为24fps（目标是24fps）
        orig_frames = len(init_frames)
        new_frames = len(refined)
        fps_out = target_fps  # 固定输出为24fps
        interpolated = refined  # 直接作为输出帧序列
        print(
            f"Original: {orig_frames} frames @ {fps:.2f}fps, "
            f"New: {new_frames} frames @ {fps_out:.2f}fps"
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

