#!/usr/bin/env python3
"""
Compute objective metrics (PSNR, SSIM, LPIPS) for T2V VBench evaluation.

Usage:
    python evaluation/vbench/metrics.py --config configs/vbench_eval.yaml [--limit 2] [--dry_run]
"""

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS for generated videos.")
    parser.add_argument("--config", type=str, required=True, help="Path to vbench eval yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts for testing.")
    parser.add_argument("--dry_run", action="store_true", help="Skip metric computation, only validate IO.")
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(samples, device, dry_run: bool = False) -> Dict:
    if dry_run:
        return {"dry_run": True, "count": len(samples)}

    if not samples:
        raise ValueError("No samples provided for metric computation.")

    import torch
    from collections import defaultdict
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )

    # 按 seed 分组
    samples_by_seed = defaultdict(list)
    for sample in samples:
        samples_by_seed[sample.seed].append(sample)

    # 为每个 seed 单独计算指标
    per_seed_metrics = []
    for seed in sorted(samples_by_seed.keys()):
        seed_samples = samples_by_seed[seed]
        
        # 为当前 seed 创建新的 metric 实例
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    frame_count = 0
    per_prompt = []
        for sample in seed_samples:
        generated = sample.generated.to(device=device, dtype=torch.float32)
        reference = sample.reference.to(device=device, dtype=torch.float32)

        prompt_frames = generated.shape[0]
        for idx in range(prompt_frames):
            g = generated[idx].unsqueeze(0)
            r = reference[idx].unsqueeze(0)
            psnr.update(g, r)
            ssim.update(g, r)
            lpips.update(g, r)
        frame_count += prompt_frames
            per_prompt.append({
                "prompt_id": sample.prompt_id,
                "frames": prompt_frames
            })

    psnr_value = psnr.compute().item()
    ssim_value = ssim.compute().item()
    lpips_value = lpips.compute().item()

        per_seed_metrics.append({
            "seed": seed,
            "count": len(seed_samples),
        "frame_count": frame_count,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": lpips_value,
        "per_prompt": per_prompt,
        })

    # 计算总体平均（可选，保留用于兼容性）
    total_frame_count = sum(m["frame_count"] for m in per_seed_metrics)
    total_count = sum(m["count"] for m in per_seed_metrics)
    
    # 加权平均（按帧数加权）
    overall_psnr = sum(m["psnr"] * m["frame_count"] for m in per_seed_metrics) / total_frame_count if total_frame_count > 0 else 0.0
    overall_ssim = sum(m["ssim"] * m["frame_count"] for m in per_seed_metrics) / total_frame_count if total_frame_count > 0 else 0.0
    overall_lpips = sum(m["lpips"] * m["frame_count"] for m in per_seed_metrics) / total_frame_count if total_frame_count > 0 else 0.0

    return {
        "count": total_count,
        "frame_count": total_frame_count,
        "psnr": overall_psnr,
        "ssim": overall_ssim,
        "lpips": overall_lpips,
        "per_seed": per_seed_metrics,  # 按 seed 分组的结果
    }


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    if not cfg.get("calc_psnr_ssim_lpips", True):
        print("[metrics] calc_psnr_ssim_lpips disabled in config; exiting.")
        return

    exp_name = cfg.get("exp_name", "default_exp")
    output_root = cfg.get("output_root", "results/vbench")
    output_root = Path(output_root) if Path(output_root).is_absolute() else (cfg_path.parent.parent / output_root)
    exp_dir = output_root / exp_name

    metadata_path = exp_dir / "metadata.csv"
    frames_root = exp_dir / "frames"
    reference_root_cfg = cfg.get("reference_root")
    if not reference_root_cfg:
        raise ValueError(
            "reference_root is not specified in config. Provide path to reference frames for objective metrics."
        )
    reference_root = Path(reference_root_cfg)
    if not reference_root.is_absolute():
        reference_root = (cfg_path.parent.parent / reference_root).resolve()

    frame_format = cfg.get("frame_format", "png")

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_root}. Run extract_frames first.")

    if args.dry_run:
        # Only verify paths exist and write placeholder result
        placeholder = {
            "dry_run": True,
            "metadata_exists": metadata_path.exists(),
            "frames_root": str(frames_root),
            "reference_root": str(reference_root),
        }
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_path = metrics_dir / "objective.json"
        output_path.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        print(f"[metrics] Dry-run placeholder saved to {output_path}")
        return

    import sys
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import load_video_samples
    import torch

    samples = load_video_samples(
        metadata_path=metadata_path,
        frames_root=frames_root,
        reference_root=reference_root,
        frame_format=frame_format,
        limit=args.limit,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = compute_metrics(samples, device=device, dry_run=args.dry_run)

    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "objective.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[metrics] Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

