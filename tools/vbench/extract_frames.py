#!/usr/bin/env python3
"""
Extract frames from generated videos for VBench evaluation.

Usage:
    python tools/vbench/extract_frames.py --config configs/vbench_eval.yaml [--limit 2] [--dry_run]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from generated videos.")
    parser.add_argument("--config", type=str, required=True, help="Path to vbench eval yaml config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts for quick tests.")
    parser.add_argument("--dry_run", action="store_true", help="Do not read videos, only simulate outputs.")
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    expected_frames: int,
    frame_format: str,
    dry_run: bool,
) -> int:
    ensure_dir(output_dir)
    for old_frame in output_dir.glob(f"*.{frame_format}"):
        old_frame.unlink()
    if dry_run:
        for idx in range(expected_frames):
            output_file = output_dir / f"{idx:05d}.{frame_format}"
            dummy_image = Image.new("RGB", (2, 2), color=(0, 0, 0))
            dummy_image.save(output_file)
        return expected_frames

    try:
        import imageio.v2 as imageio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "imageio is required for frame extraction. Install via `pip install imageio imageio-ffmpeg`."
        ) from exc

    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    count = 0
    try:
        for idx, frame in enumerate(reader):
            output_file = output_dir / f"{idx:05d}.{frame_format}"
            imageio.imwrite(output_file, frame)
            count += 1
    finally:
        reader.close()
    return count


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    exp_name = cfg.get("exp_name", "default_exp")
    output_root = cfg.get("output_root", "results/vbench")
    output_root = Path(output_root) if Path(output_root).is_absolute() else (cfg_path.parent.parent / output_root)
    exp_dir = output_root / exp_name

    metadata_path = exp_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}. Run build_metadata first.")
    metadata = read_metadata(metadata_path)
    if args.limit is not None:
        metadata = metadata[: args.limit]

    videos_dir = exp_dir / "videos"
    frames_dir = exp_dir / "frames"
    ensure_dir(frames_dir)

    frame_format = cfg.get("frame_format", "png")
    frame_format = frame_format.lower()
    supported_formats = {"png", "jpg", "jpeg"}
    if frame_format not in supported_formats:
        raise ValueError(f"Unsupported frame_format: {frame_format}. Supported: {supported_formats}")

    # 按 prompt_id 分组，以便为每个视频分配正确的索引
    prompt_groups = defaultdict(list)
    for row in metadata:
        prompt_id = row["prompt_id"]
        prompt_groups[prompt_id].append(row)
    
    failures = []
    for prompt_id, prompt_rows in sorted(prompt_groups.items()):
        for video_idx, row in enumerate(prompt_rows):
            video_path = videos_dir / f"prompt_{prompt_id}" / f"{video_idx}.mp4"
            # 为每个视频创建单独的子目录
            target_dir = frames_dir / f"prompt_{prompt_id}" / str(video_idx)

        if not video_path.exists() and not args.dry_run:
                failures.append(f"{prompt_id}/{video_idx}: missing video {video_path}")
            continue

        expected_frames = int(row.get("num_frames", cfg.get("num_frames", 49)))
        extracted = extract_frames_from_video(
            video_path=video_path,
            output_dir=target_dir,
            expected_frames=expected_frames,
            frame_format=frame_format,
            dry_run=args.dry_run,
        )

        if extracted != expected_frames:
                print(f"[extract_frames] Warning: prompt {prompt_id} video {video_idx} expected {expected_frames} frames, got {extracted}.")

    if failures:
        print("[extract_frames] Missing videos:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)

    print(f"[extract_frames] Completed. Frames stored in {frames_dir}")


if __name__ == "__main__":
    main()

