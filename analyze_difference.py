import os
import argparse

import torch
import torchvision
import yaml

from src.models import load_model
from src.utils import InputPadder
from src.transport import create_transport, Sampler


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_eden.yaml",
        help="配置文件路径（与 inference.py 相同）",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="要分析的输入视频路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="difference_analysis_outputs",
        help="插帧结果输出目录",
    )
    parser.add_argument(
        "--max_offset",
        type=int,
        default=10,
        help="最多取多少帧间隔（1 表示相邻，2 表示隔 1 帧，...）",
    )
    args = parser.parse_args()

    # 加载配置，与 inference.py 保持一致
    with open(args.config, "r") as f:
        update_args = yaml.unsafe_load(f)
    for k, v in update_args.items():
        setattr(args, k, v)

    return args


def load_eden_and_sampler(args, device):
    model_name = args.model_name
    ckpt = torch.load(args.pretrained_eden_path, map_location="cpu")

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
    return eden, sample_fn


def compute_difference(frame0, frame1, args, device):
    """
    按 inference.py 中的公式计算 difference。

    frame0, frame1: [1, 3, H, W], 已经是 0~1 的 float
    """
    # 在 CPU 上计算没问题，最后再 to(device)
    diff = (
        (
            torch.mean(torch.cosine_similarity(frame0, frame1), dim=[1, 2])
            - args.cos_sim_mean
        )
        / args.cos_sim_std
    ).unsqueeze(1)
    return diff.to(device)


def interpolate_pair(frame0, frame1, eden, sample_fn, args, device):
    """
    复用当前单 GPU 的 encode + denoise_from_tokens + decode 逻辑，
    对一对帧做一次插帧。
    """
    h, w = frame0.shape[2:]
    padder = InputPadder([h, w])

    frame0 = frame0.to(device)
    frame1 = frame1.to(device)

    # difference
    difference = compute_difference(frame0, frame1, args, device)

    # encoder
    cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
    enc_out = eden.encode(cond_frames)

    # 初始噪声
    new_h, new_w = cond_frames.shape[2:]
    noise = torch.randn(
        [1, new_h // 32 * new_w // 32, args.model_args["latent_dim"]],
        device=device,
    )

    # denoise wrapper
    def denoise_wrapper(query_latents, t):
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                denoise_timestep = t.unsqueeze(0) if t.dim() == 0 else t
            else:
                denoise_timestep = t[0:1]
        else:
            denoise_timestep = torch.tensor(
                [t], device=query_latents.device, dtype=torch.float32
            )

        if denoise_timestep.dim() == 0:
            denoise_timestep = denoise_timestep.unsqueeze(0)

        return eden.denoise_from_tokens(
            query_latents, denoise_timestep, enc_out, difference
        )

    samples = sample_fn(noise, denoise_wrapper)[-1]

    # 解码
    denoise_latents = samples / args.vae_scaler + args.vae_shift
    generated_frame = eden.decode(denoise_latents)
    generated_frame = padder.unpad(generated_frame.clamp(0.0, 1.0))

    return generated_frame.cpu()


def main():
    args = build_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载 EDEN 模型和采样器...")
    eden, sample_fn = load_eden_and_sampler(args, device)
    print("模型加载完成。")

    print(f"读取视频: {args.video_path}")
    frames, _, info = torchvision.io.read_video(args.video_path)
    # [T, H, W, C] -> [T, 3, H, W], 0~1
    frames = frames.float().permute(0, 3, 1, 2) / 255.0
    num_frames = frames.shape[0]
    fps = info.get("video_fps", 0)
    print(f"视频总帧数: {num_frames}, fps: {fps}")

    if num_frames < 2:
        print("视频帧数小于 2，无法分析。")
        return

    # 最多 offset 不能超过 num_frames-1
    max_offset = min(args.max_offset, num_frames - 1)
    base_idx = 0

    print(
        f"以第 {base_idx} 帧为基准，截取 offset=1..{max_offset} 的帧对 "
        "(相邻、隔1帧、隔2帧...)"
    )

    results_txt_path = os.path.join(args.output_dir, "differences.txt")
    with open(results_txt_path, "w") as ftxt:
        ftxt.write(
            "# pair_index, frame_i, frame_j, offset, difference_value\n"
        )

        for offset in range(1, max_offset + 1):
            i = base_idx
            j = base_idx + offset

            frame0 = frames[i].unsqueeze(0)  # [1,3,H,W]
            frame1 = frames[j].unsqueeze(0)

            # 先单独算 difference（不走 interpolate）
            diff = compute_difference(
                frame0.to(device), frame1.to(device), args, device
            )
            diff_val = diff.item()

            print(
                f"帧对 {i}-{j} (offset={offset}) 的 difference = {diff_val:.4f}"
            )
            ftxt.write(
                f"{offset},{i},{j},{offset},{diff_val:.6f}\n"
            )
            ftxt.flush()

            # 执行一次插帧
            with torch.no_grad():
                mid = interpolate_pair(
                    frame0, frame1, eden, sample_fn, args, device
                )

            # 保存三张图：frame0, mid, frame1，方便肉眼对比
            out_subdir = os.path.join(
                args.output_dir, f"pair_{i}_{j}_off{offset}_diff_{diff_val:.3f}"
            )
            os.makedirs(out_subdir, exist_ok=True)

            torchvision.utils.save_image(
                frame0, os.path.join(out_subdir, "frame0.png")
            )
            torchvision.utils.save_image(
                mid, os.path.join(out_subdir, "mid.png")
            )
            torchvision.utils.save_image(
                frame1, os.path.join(out_subdir, "frame1.png")
            )

    print(f"\n所有 difference 已写入: {results_txt_path}")
    print(f"每对帧的三个图像保存在子文件夹，位于: {args.output_dir}")


if __name__ == "__main__":
    main()


