#!/usr/bin/env python3
"""
Wrapper to invoke VBench official evaluation scripts and collect scores.

Usage:
    python evaluation/vbench/vbench_runner.py --config configs/vbench_eval.yaml [--limit 10] [--dry_run]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VBench evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to vbench eval yaml config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos (if supported).")
    parser.add_argument("--dry_run", action="store_true", help="Skip subprocess execution; only verify paths.")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Override metrics list for VBench (comma separated). Default from config or VBench.",
    )
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_command(
    vbench_repo: Path,
    generated_dir: Path,
    prompt_file: Path,
    result_dir: Path,
    metrics: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[str]:
    script = vbench_repo / "evaluate.py"
    if not script.exists():
        raise FileNotFoundError(f"VBench evaluate.py not found at {script}")

    cmd = [
        "python3",
        str(script),
        "--videos_path",
        str(generated_dir),
        "--output_path",
        str(result_dir),
        "--load_ckpt_from_local",
        "True",  # Use local mode to avoid GitHub API rate limits
    ]

    if metrics:
        cmd += ["--dimension"] + metrics
    if limit is not None:
        # VBench doesn't have --limit, but we can filter in the wrapper if needed
        pass

    return cmd


def parse_vbench_outputs(result_dir: Path) -> Dict:
    """解析 VBench 评估输出文件
    
    VBench 会生成以下文件：
    - {name}_full_info.json: 评估元数据（prompts、维度、视频路径）
    - {name}_eval_results.json: 评估结果（每个维度的分数）
    - scores.json: 可能存在的汇总文件（如果 VBench 生成）
    - scores.csv: 可能存在的 CSV 汇总文件（如果 VBench 生成）
    """
    summary: Dict = {}
    
    # 查找最新的 eval_results.json 文件
    eval_results_files = sorted(result_dir.glob("*_eval_results.json"))
    if eval_results_files:
        latest_eval = eval_results_files[-1]
        summary["eval_results_json"] = str(latest_eval)
        try:
            eval_data = json.loads(latest_eval.read_text(encoding="utf-8"))
            # 提取每个维度的平均分数
            dimension_scores = {}
            for dimension, result in eval_data.items():
                if isinstance(result, (list, tuple)) and len(result) >= 1:
                    # 第一个元素通常是平均分数
                    dimension_scores[dimension] = result[0] if isinstance(result[0], (int, float)) else None
                elif isinstance(result, dict) and "average" in result:
                    dimension_scores[dimension] = result["average"]
                else:
                    dimension_scores[dimension] = result
            summary["dimension_scores"] = dimension_scores
        except Exception as e:
            summary["parse_error"] = str(e)
    
    # 查找最新的 full_info.json 文件
    full_info_files = sorted(result_dir.glob("*_full_info.json"))
    if full_info_files:
        latest_full_info = full_info_files[-1]
        summary["full_info_json"] = str(latest_full_info)
    
    # 检查是否有 scores.json 或 scores.csv（VBench 可能生成）
    result_json = result_dir / "scores.json"
    result_csv = result_dir / "scores.csv"

    if result_json.exists():
        summary["scores_json"] = json.loads(result_json.read_text(encoding="utf-8"))
    if result_csv.exists():
        summary["scores_csv_path"] = str(result_csv)

    return summary


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    if not cfg.get("run_vbench_metrics", True):
        print("[vbench_runner] run_vbench_metrics disabled in config; exiting.")
        return

    project_root = cfg_path.parent.parent
    vbench_repo = project_root / "VBench-master"
    if not vbench_repo.exists():
        raise FileNotFoundError(f"VBench repository not found at {vbench_repo}")

    exp_name = cfg.get("exp_name", "default_exp")
    output_root = cfg.get("output_root", "results/vbench")
    output_root = Path(output_root) if Path(output_root).is_absolute() else (project_root / output_root)
    exp_dir = output_root / exp_name
    
    # 检查并准备 VBench 格式的视频目录
    videos_vbench_format_dir = exp_dir / "videos_vbench_format"
    original_videos_dir = exp_dir / "videos"
    
    # 如果 VBench 格式目录不存在，自动创建
    if not videos_vbench_format_dir.exists():
        if not original_videos_dir.exists():
            raise FileNotFoundError(f"Generated videos directory not found: {original_videos_dir}")
        
        # 检查原始视频目录是否有子目录结构
        has_subdirs = any(d.is_dir() for d in original_videos_dir.iterdir())
        
        if has_subdirs:
            # 需要准备 VBench 格式
            print(f"[vbench_runner] Preparing videos for VBench format...")
            metadata_path = exp_dir / "metadata.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"metadata.csv not found: {metadata_path}\n"
                    f"Please run build_metadata.py first to generate metadata.csv"
                )
            
            # 导入并运行 prepare_vbench_videos
            import sys
            prepare_script = project_root / "tools" / "vbench" / "prepare_vbench_videos.py"
            if not prepare_script.exists():
                raise FileNotFoundError(
                    f"prepare_vbench_videos.py not found: {prepare_script}\n"
                    f"Please ensure the script exists to prepare videos for VBench"
                )
            
            # 运行准备脚本
            import subprocess as sp
            prepare_cmd = [
                "python3",
                str(prepare_script),
                "--videos_dir", str(original_videos_dir),
                "--metadata", str(metadata_path),
                "--output_dir", str(videos_vbench_format_dir),
            ]
            print(f"[vbench_runner] Running: {' '.join(prepare_cmd)}")
            result = sp.run(prepare_cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                raise RuntimeError(f"Failed to prepare videos for VBench format")
            print(f"[vbench_runner] Videos prepared successfully")
        else:
            # 原始目录已经是正确格式，直接使用
            videos_vbench_format_dir = original_videos_dir
            print(f"[vbench_runner] Using videos directory (already in correct format): {videos_vbench_format_dir}")
    
    videos_dir = videos_vbench_format_dir
    print(f"[vbench_runner] Using VBench format directory: {videos_dir}")

    if not videos_dir.exists():
        raise FileNotFoundError(f"VBench format videos directory not found: {videos_dir}")

    vbench_output_dir = exp_dir / "vbench"
    ensure_dir(vbench_output_dir)

    # Determine prompt file
    prompt_file_cfg = cfg.get("prompt_file")
    if not prompt_file_cfg:
        raise ValueError("prompt_file not specified in config.")
    prompt_file = Path(prompt_file_cfg)
    if not prompt_file.is_absolute():
        prompt_file = (project_root / prompt_file_cfg).resolve()
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    metrics_list = None
    if args.metrics:
        metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    elif cfg.get("vbench_metrics"):
        metrics_list = cfg["vbench_metrics"]

    cmd = build_command(
        vbench_repo=vbench_repo,
        generated_dir=videos_dir,
        prompt_file=prompt_file,
        result_dir=vbench_output_dir,
        metrics=metrics_list,
        limit=args.limit,
    )

    if args.dry_run:
        print("[vbench_runner] Dry run. Command would be:")
        print("  ", " ".join(cmd))
        return

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{vbench_repo}:{env.get('PYTHONPATH', '')}"
    print("[vbench_runner] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=vbench_repo, env=env, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(proc.returncode)

    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    summary = parse_vbench_outputs(vbench_output_dir)
    (vbench_output_dir / "vbench_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[vbench_runner] Stored summary to {vbench_output_dir / 'vbench_scores.json'}")


if __name__ == "__main__":
    main()

