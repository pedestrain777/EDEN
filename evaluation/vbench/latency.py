#!/usr/bin/env python3
"""
Aggregate latency statistics and compute speedup over baseline.

Usage:
    python evaluation/vbench/latency.py --config configs/vbench_eval.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate latency stats.")
    parser.add_argument("--config", type=str, required=True, help="Path to vbench eval config.")
    return parser.parse_args()


def load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)

    if not cfg.get("measure_latency", True):
        print("[latency] measure_latency disabled in config; exiting.")
        return

    exp_name = cfg.get("exp_name", "default_exp")
    output_root = cfg.get("output_root", "results/vbench")
    project_root = cfg_path.parent.parent
    output_root = Path(output_root) if Path(output_root).is_absolute() else (project_root / output_root)
    exp_dir = output_root / exp_name

    latency_json_path = exp_dir / "metrics" / "latency.json"
    if not latency_json_path.exists():
        raise FileNotFoundError(f"latency.json not found at {latency_json_path}. Run generation first.")

    latency_data = json.loads(latency_json_path.read_text(encoding="utf-8"))
    avg_latency = latency_data.get("avg_latency")
    speedup = None
    baseline_latency = cfg.get("baseline_latency")
    if baseline_latency:
        try:
            baseline_latency = float(baseline_latency)
            if baseline_latency > 0 and avg_latency:
                speedup = baseline_latency / avg_latency
        except (TypeError, ValueError):
            print(f"[latency] Invalid baseline_latency value: {baseline_latency}")

    summary = {
        "avg_latency": avg_latency,
        "baseline_latency": baseline_latency,
        "speedup": speedup,
        "count": latency_data.get("count"),
        "items": latency_data.get("items"),
    }

    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "latency_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[latency] Summary written to {output_path}")


if __name__ == "__main__":
    main()

