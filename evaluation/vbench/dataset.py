#!/usr/bin/env python3
"""
Dataset utilities for T2V VBench evaluation.
"""

from collections import defaultdict
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class VideoSample:
    prompt_id: str
    prompt: str
    seed: int
    generated: torch.Tensor  # Shape: (T, C, H, W), float32 in [0,1]
    reference: torch.Tensor  # Same shape/dtype


def _load_frame_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = torch.from_numpy(np.asarray(img))  # HWC, uint8
    tensor = arr.permute(2, 0, 1).float() / 255.0
    return tensor


def _collect_frame_paths(directory: Path, frame_format: str, expected: int) -> List[Path]:
    frame_paths = sorted(directory.glob(f"*.{frame_format}"))
    if len(frame_paths) < expected:
        raise FileNotFoundError(
            f"Expected at least {expected} frames in {directory}, found {len(frame_paths)}."
        )
    return frame_paths[:expected]


def load_metadata(metadata_path: Path) -> List[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_video_samples(
    metadata_path: Path,
    frames_root: Path,
    reference_root: Path,
    frame_format: str = "png",
    limit: Optional[int] = None,
) -> List[VideoSample]:
    metadata = load_metadata(metadata_path)
    samples: List[VideoSample] = []
    if limit is not None:
        metadata = metadata[:limit]

    # 按 prompt_id 分组，以便为每个视频分配正确的索引
    prompt_groups = defaultdict(list)
    for row in metadata:
        prompt_id = row["prompt_id"]
        prompt_groups[prompt_id].append(row)

    for prompt_id, prompt_rows in sorted(prompt_groups.items()):
        for video_idx, row in enumerate(prompt_rows):
        prompt = row.get("prompt", "")
        seed = int(row.get("seed", 0))
        expected_frames = int(row.get("num_frames", 0))

            # 新的目录结构：frames/prompt_{prompt_id}/{video_idx}/
            gen_dir = frames_root / f"prompt_{prompt_id}" / str(video_idx)
            # 参考帧目录：如果存在子目录结构，则使用对应的 video_idx，否则使用根目录
            ref_base_dir = reference_root / f"prompt_{prompt_id}"
            if (ref_base_dir / str(video_idx)).exists():
                # 参考帧也有多个视频，使用对应的 video_idx
                ref_dir = ref_base_dir / str(video_idx)
            else:
                # 参考帧只有一个视频，使用根目录
                ref_dir = ref_base_dir

        if not gen_dir.exists():
            raise FileNotFoundError(f"Generated frame directory missing: {gen_dir}")
        if not ref_dir.exists():
            raise FileNotFoundError(
                f"Reference frame directory missing: {ref_dir}. "
                "Please provide reference videos/frames or update config.reference_root."
            )

        gen_paths = _collect_frame_paths(gen_dir, frame_format, expected_frames)
        ref_paths = _collect_frame_paths(ref_dir, frame_format, expected_frames)

        gen_tensor = torch.stack([_load_frame_tensor(p) for p in gen_paths], dim=0)
        ref_tensor = torch.stack([_load_frame_tensor(p) for p in ref_paths], dim=0)

        if gen_tensor.shape != ref_tensor.shape:
            raise ValueError(
                    f"Shape mismatch for prompt {prompt_id} video {video_idx}: generated {gen_tensor.shape}, reference {ref_tensor.shape}"
            )

        samples.append(
            VideoSample(
                prompt_id=prompt_id,
                prompt=prompt,
                seed=seed,
                generated=gen_tensor,
                reference=ref_tensor,
            )
        )
    return samples


def iter_video_samples(*args, **kwargs) -> Iterable[VideoSample]:
    samples = load_video_samples(*args, **kwargs)
    for sample in samples:
        yield sample

