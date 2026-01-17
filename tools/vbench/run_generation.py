#!/usr/bin/env python3
"""
Run T2V generation for VBench evaluation based on metadata.csv and yaml config.

Usage:
    python tools/vbench/run_generation.py --config configs/vbench_eval.yaml [--limit 2] [--dry_run]
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T2V generation for VBench.")
    parser.add_argument("--config", type=str, required=True, help="Path to vbench eval yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Limit prompts to generate for smoke test.")
    parser.add_argument("--dry_run", action="store_true", help="If set, skip actual generation and only touch files.")
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_metadata(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    videos = base_dir / "videos"
    frames = base_dir / "frames"
    metrics = base_dir / "metrics"
    logs = base_dir / "logs"
    for p in (videos, frames, metrics, logs):
        p.mkdir(parents=True, exist_ok=True)
    return {"videos": videos, "frames": frames, "metrics": metrics, "logs": logs}


def build_pipeline(cfg: Dict):
    # Import heavy deps lazily to support --dry_run without torch/diffusers
    from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline  # type: ignore
    weight_folders = cfg.get("models") or []
    if not weight_folders:
        raise ValueError("models is empty in config.")
    device = cfg.get("device", "cuda:0")
    steps_cfg = cfg.get("steps", "10,15")
    seed_list = cfg.get("seed_list") or [cfg.get("seed", 1234)]
    base_seed = int(seed_list[0])

    # Prepare args-like object for pipeline constructor
    class Args:  # minimal adapter to satisfy current pipeline API
        def __init__(self) -> None:
            self.use_dpm_solver = True
            self.logger = None
            self.enable_xformers_memory_efficient_attention = False
            self.steps = [int(x) for x in str(steps_cfg).split(",")]

    args = Args()
    pipe = HybridVideoInferencePipeline(
        weight_folders=weight_folders,
        seed=base_seed,
        device=device,
        args=args,
    )
    pipe.set_pipe_and_generator()

    # Store parsed steps in pipeline args if needed by downstream
    return pipe


def generate_one(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    out_video_path: Path,
    num_frames: int,
    height: int,
    width: int,
    guidance_scale: float,
    fps: int,
    dry_run: bool = False,
) -> float:
    t0 = time.perf_counter()
    if dry_run:
        # Simulate work and create an empty file placeholder
        time.sleep(0.01)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_video_path.touch()
        latency = time.perf_counter() - t0
        return latency

    # Import exporter lazily
    from diffusers.utils import export_to_video  # type: ignore
    video_frames = pipe.generate(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_frames=int(num_frames),
        height=int(height),
        width=int(width),
        guidance_scale=float(guidance_scale),
        num_videos_per_prompt=1,
        output_type="pil",
    )
    # Select first video if nested
    if isinstance(video_frames, list) and video_frames and isinstance(video_frames[0], list):
        video = video_frames[0]
    else:
        video = video_frames

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(out_video_path), fps=int(fps))

    latency = time.perf_counter() - t0
    return latency


def setup_logging(log_file: Path) -> logging.Logger:
    """设置日志记录器，同时输出到文件和控制台"""
    logger = logging.getLogger("vbench_generation")
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    exp_name = cfg.get("exp_name", "default_exp")
    output_root = cfg.get("output_root", "results/vbench")
    output_root = Path(output_root) if Path(output_root).is_absolute() else (cfg_path.parent.parent / output_root)
    exp_dir = output_root / exp_name
    dirs = ensure_dirs(exp_dir)

    # 设置日志
    log_file = dirs["logs"] / "generation_progress.log"
    logger = setup_logging(log_file)
    logger.info(f"开始生成任务: {exp_name}")
    logger.info(f"配置文件: {cfg_path}")
    logger.info(f"输出目录: {exp_dir}")

    metadata_path = exp_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}. Please run tools/vbench/build_metadata.py first.")
    rows = read_metadata(metadata_path)
    if args.limit is not None:
        rows = rows[: args.limit]
        logger.info(f"限制生成数量: {args.limit} 条记录")

    # Generation params
    num_frames = int(cfg.get("num_frames", 49))
    resolution = cfg.get("resolution") or {}
    height = int(resolution.get("height", 480))
    width = int(resolution.get("width", 720))
    guidance_scale = float(cfg.get("guidance_scale", 6.0))
    fps = int(cfg.get("fps", 8))

    logger.info(f"生成参数: {num_frames}帧, {height}x{width}, fps={fps}, guidance_scale={guidance_scale}")
    logger.info(f"总共需要生成: {len(rows)} 个视频")

    # Initialize pipeline unless dry-run
    pipe = None
    if not args.dry_run:
        logger.info("正在加载模型管道...")
        pipe = build_pipeline(cfg)
        logger.info("模型管道加载完成")

    # 按 prompt_id 分组，为每个 prompt 的多个 seed 分配索引
    prompt_groups = defaultdict(list)
    for row in rows:
        prompt_id = row["prompt_id"]
        prompt_groups[prompt_id].append(row)
    
    total_prompts = len(prompt_groups)
    logger.info(f"共有 {total_prompts} 个不同的 prompt，每个 prompt 将生成多个视频")

    latencies: List[Tuple[str, int, float]] = []  # (prompt_id, video_idx, latency)
    current_idx = 0
    
    for prompt_idx, (prompt_id, prompt_rows) in enumerate(sorted(prompt_groups.items()), 1):
        prompt = prompt_rows[0]["prompt"]
        logger.info(f"[{prompt_idx}/{total_prompts}] 开始处理 prompt_{prompt_id}: {prompt[:60]}...")
        
        for video_idx, row in enumerate(prompt_rows):
            current_idx += 1
            seed = int(row.get("seed", cfg.get("seed_list", [1234])[0]))
            neg_prompt = row.get("negative_prompt") or ""
            
            out_dir = dirs["videos"] / f"prompt_{prompt_id}"
            # 使用索引作为文件名，确保每个 seed 生成不同的文件
            out_path = out_dir / f"{video_idx}.mp4"
            
            logger.info(f"  [{current_idx}/{len(rows)}] 生成视频 {video_idx}/4 (seed={seed}): {out_path.name}")
            
            # 检查文件是否已存在
            if out_path.exists() and not args.dry_run:
                logger.info(f"    文件已存在，跳过: {out_path}")
                continue
            
            if pipe is not None and hasattr(pipe, "generator") and pipe.generator is not None:
                pipe.generator.manual_seed(seed)

            t_start = time.time()
            latency = generate_one(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=neg_prompt,
                out_video_path=out_path,
                num_frames=int(row.get("num_frames", num_frames)),
                height=int(row.get("height", height)),
                width=int(row.get("width", width)),
                guidance_scale=guidance_scale,
                fps=int(row.get("fps", fps)),
                dry_run=args.dry_run,
            )
            t_elapsed = time.time() - t_start
            
            latencies.append((prompt_id, video_idx, latency))
            logger.info(f"    完成! 耗时: {t_elapsed:.2f}秒 (latency: {latency:.2f}秒)")
        
        logger.info(f"  prompt_{prompt_id} 的所有视频生成完成")

    # Save latency stats
    metrics_dir = dirs["metrics"]
    latency_json = {
        "exp_name": exp_name,
        "count": len(latencies),
        "avg_latency": (sum(x[2] for x in latencies) / len(latencies)) if latencies else None,
        "items": [{"prompt_id": pid, "video_idx": vidx, "latency": lat} for pid, vidx, lat in latencies],
    }
    (metrics_dir / "latency.json").write_text(json.dumps(latency_json, indent=2), encoding="utf-8")
    logger.info(f"延迟指标已保存到: {metrics_dir / 'latency.json'}")
    logger.info(f"平均延迟: {latency_json['avg_latency']:.2f}秒")
    logger.info("所有视频生成完成!")


if __name__ == "__main__":
    main()


