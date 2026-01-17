#!/usr/bin/env python3
"""
Generate VBench metadata CSV from configuration.

Usage:
    python tools/vbench/build_metadata.py --config configs/vbench_eval.yaml
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata.csv for VBench evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to vbench evaluation yaml config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional override for metadata.csv output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts (useful for smoke tests).",
    )
    return parser.parse_args()


def _load_yaml(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            prompts: List[str] = []
            for item in data:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict):
                    prompt = None
                    for key in ("prompt", "prompt_en", "text", "caption", "description"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            prompt = value.strip()
                            break
                    if prompt is None:
                        raise ValueError(
                            f"Cannot locate prompt text in entry: {item}"
                        )
                    prompts.append(prompt)
                else:
                    raise ValueError(f"Unsupported JSON prompt entry type: {type(item)}")
            return prompts
        raise ValueError("JSON prompt file must contain a list.")

    if suffix == ".txt":
        with file_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            prompts = []
            for row in reader:
                if not row:
                    continue
                if header:
                    # Try common column names
                    lookup_order = ["prompt", "text", "caption", "prompt_en"]
                    prompt = None
                    for col in lookup_order:
                        if col in header:
                            idx = header.index(col)
                            prompt = row[idx].strip()
                            break
                    if prompt is None:
                        prompt = row[0].strip()
                else:
                    prompt = row[0].strip()
                if prompt:
                    prompts.append(prompt)
            return prompts

    raise ValueError(f"Unsupported prompt file format: {file_path}")


def _load_negative_prompts(file_path: Optional[Path], expected: int) -> List[str]:
    if file_path is None:
        return [""] * expected

    if not file_path.exists():
        raise FileNotFoundError(f"Negative prompt file not found: {file_path}")

    suffix = file_path.suffix.lower()
    neg_prompts: List[str]
    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            neg_prompts = []
            for item in data:
                if isinstance(item, str):
                    neg_prompts.append(item.strip())
                elif isinstance(item, dict):
                    val = (
                        item.get("negative_prompt")
                        or item.get("negative_prompt_en")
                        or item.get("prompt")
                        or item.get("text")
                    )
                    if isinstance(val, str):
                        neg_prompts.append(val.strip())
                    else:
                        neg_prompts.append("")
                else:
                    neg_prompts.append("")
        else:
            raise ValueError("Negative prompt JSON must be a list.")
    else:
        with file_path.open("r", encoding="utf-8") as f:
            neg_prompts = [line.strip() for line in f if line.strip()]

    if len(neg_prompts) < expected:
        # pad with empty strings
        neg_prompts.extend([""] * (expected - len(neg_prompts)))
    elif len(neg_prompts) > expected:
        neg_prompts = neg_prompts[:expected]

    return neg_prompts


def _iter_rows(
    prompts: Iterable[str],
    neg_prompts: Iterable[str],
    seeds: Iterable[int],
    num_frames: int,
    height: int,
    width: int,
    fps: int,
) -> Iterable[List[str]]:
    for prompt_id, (prompt, neg_prompt) in enumerate(zip(prompts, neg_prompts)):
        for seed in seeds:
            yield [
                f"{prompt_id:05d}",
                prompt,
                neg_prompt,
                str(seed),
                str(num_frames),
                str(height),
                str(width),
                str(fps),
            ]


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(config_path)

    prompt_file = Path(cfg["prompt_file"])
    if not prompt_file.is_absolute():
        prompt_file = config_path.parent.parent / prompt_file
    prompts = _load_prompts(prompt_file)

    if args.limit is not None:
        prompts = prompts[: args.limit]

    negative_prompt_cfg = cfg.get("negative_prompt_file")
    neg_file: Optional[Path] = None
    if negative_prompt_cfg and str(negative_prompt_cfg).lower() != "null":
        neg_file = Path(negative_prompt_cfg)
        if not neg_file.is_absolute():
            neg_file = config_path.parent.parent / neg_file
    negative_prompts = _load_negative_prompts(neg_file, len(prompts))

    seeds = cfg.get("seed_list") or [cfg.get("seed", 1234)]
    seeds = [int(s) for s in seeds]

    resolution = cfg.get("resolution") or {}
    height = int(resolution.get("height", 480))
    width = int(resolution.get("width", 720))

    output_root = cfg.get("output_root", "results/vbench")
    if not Path(output_root).is_absolute():
        output_root = str(config_path.parent.parent / output_root)
    exp_name = cfg.get("exp_name", "default_exp")
    output_dir = Path(output_root) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        Path(args.output) if args.output else output_dir / "metadata.csv"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    num_frames = int(cfg.get("num_frames", 49))
    fps = int(cfg.get("fps", 8))

    header = [
        "prompt_id",
        "prompt",
        "negative_prompt",
        "seed",
        "num_frames",
        "height",
        "width",
        "fps",
    ]

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in _iter_rows(
            prompts,
            negative_prompts,
            seeds,
            num_frames,
            height,
            width,
            fps,
        ):
            writer.writerow(row)

    print(f"Wrote {metadata_path} with {len(prompts) * len(seeds)} rows.")


if __name__ == "__main__":
    main()

