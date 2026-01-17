#!/usr/bin/env python3
"""
Generate summary report combining objective metrics, VBench scores, and latency stats.

Usage:
    python evaluation/vbench/report.py --config configs/vbench_eval.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VBench evaluation report.")
    parser.add_argument("--config", type=str, required=True, help="Path to vbench eval yaml.")
    parser.add_argument("--output", type=str, default=None, help="Optional override path for report markdown.")
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_value(value, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)


def extract_vbench_score(vbench_data: Optional[Dict]) -> Dict[str, str]:
    if not vbench_data:
        return {}
    scores_json = vbench_data.get("scores_json")
    if not scores_json:
        return {}
    result = {}
    for key, value in scores_json.items():
        if isinstance(value, (int, float)):
            result[key] = format_value(value, precision=4)
    return result


def build_summary_table(
    objective: Optional[Dict],
    vbench_scores: Dict[str, str],
    latency: Optional[Dict],
) -> List[List[str]]:
    psnr = format_value(objective.get("psnr")) if objective and not objective.get("dry_run") else "N/A"
    ssim = format_value(objective.get("ssim")) if objective and not objective.get("dry_run") else "N/A"
    lpips = format_value(objective.get("lpips")) if objective and not objective.get("dry_run") else "N/A"

    overall = vbench_scores.get("overall", "N/A")

    avg_latency = format_value(latency.get("avg_latency")) if latency else "N/A"
    speedup = format_value(latency.get("speedup")) if latency else "N/A"

    return [
        ["Metric", "Value"],
        ["PSNR ↑", psnr],
        ["SSIM ↑", ssim],
        ["LPIPS ↓", lpips],
        ["VBench Overall ↑", overall],
        ["Latency (s) ↓", avg_latency],
        ["Speedup ↑", speedup],
    ]


def make_markdown_table(rows: List[List[str]]) -> str:
    lines = []
    header = rows[0]
    lines.append(f"| {header[0]} | {header[1]} |")
    lines.append("| --- | --- |")
    for metric, value in rows[1:]:
        lines.append(f"| {metric} | {value} |")
    return "\n".join(lines)


def render_prompt_details(objective: Optional[Dict], max_rows: int = 5) -> str:
    if not objective or objective.get("dry_run"):
        return "_Objective metrics未实际计算，暂无逐条详情。_"

    per_prompt = objective.get("per_prompt")
    if not per_prompt:
        return "_未提供 per-prompt 指标。_"

    rows = [["Prompt ID", "Frames"]]
    for item in per_prompt[:max_rows]:
        rows.append([item.get("prompt_id", "N/A"), str(item.get("frames", "N/A"))])
    return make_markdown_table(rows)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    exp_name = cfg.get("exp_name", "default_exp")
    output_root = cfg.get("output_root", "results/vbench")
    project_root = cfg_path.parent.parent
    output_root = Path(output_root) if Path(output_root).is_absolute() else (project_root / output_root)
    exp_dir = output_root / exp_name

    objective = load_json(exp_dir / "metrics" / "objective.json")
    latency = load_json(exp_dir / "metrics" / "latency_summary.json")
    vbench_data = load_json(exp_dir / "vbench" / "vbench_scores.json")
    vbench_scores = extract_vbench_score(vbench_data)

    summary_table = build_summary_table(objective, vbench_scores, latency)
    summary_md = make_markdown_table(summary_table)
    detail_md = render_prompt_details(objective)

    report_lines = [
        f"# VBench 评估报告 - {exp_name}",
        "",
        "## 汇总指标",
        "",
        summary_md,
        "",
        "## 逐条详情（Top 5）",
        "",
        detail_md,
        "",
        "## 附加信息",
        "",
        f"- 评估配置：`{args.config}`",
        f"- 结果目录：`{exp_dir}`",
        "- 若某些指标为 `N/A`，表示对应步骤尚未执行或处于 dry-run 模式。",
    ]

    report_md = "\n".join(report_lines)
    output_path = Path(args.output) if args.output else (exp_dir / "report.md")
    output_path.write_text(report_md, encoding="utf-8")
    print(f"[report] Report written to {output_path}")


if __name__ == "__main__":
    main()

