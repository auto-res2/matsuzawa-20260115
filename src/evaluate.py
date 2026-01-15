"""Independent evaluation & visualisation script.
Run:
    uv run python -m src.evaluate results_dir=./outputs run_ids='["runA", "runB"]'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------


def _parse_args():
    import sys

    p = argparse.ArgumentParser("Few-shot evaluation summariser")

    # Parse key=value style arguments if present
    parsed_kv = {}
    remaining_args = []
    for arg in sys.argv[1:]:
        if "=" in arg and not arg.startswith("-"):
            key, val = arg.split("=", 1)
            parsed_kv[key] = val
        else:
            remaining_args.append(arg)

    # Setup parser with optional arguments
    p.add_argument("--results_dir", type=str, help="Output directory for metrics/figures", default=None)
    p.add_argument("--run_ids", type=str, help="JSON list of WandB run IDs", default=None)
    p.add_argument("results_dir_pos", nargs="?", type=str, help="Output directory (positional)")
    p.add_argument("run_ids_pos", nargs="?", type=str, help="Run IDs (positional)")

    # Parse remaining args
    args = p.parse_args(remaining_args)

    # Merge key=value style args
    if "results_dir" in parsed_kv:
        args.results_dir = parsed_kv["results_dir"]
    if "run_ids" in parsed_kv:
        args.run_ids = parsed_kv["run_ids"]

    # Support positional fallback
    if args.results_dir is None and args.results_dir_pos is not None:
        args.results_dir = args.results_dir_pos
    if args.run_ids is None and args.run_ids_pos is not None:
        args.run_ids = args.run_ids_pos

    if not args.results_dir or not args.run_ids:
        p.error("Both results_dir and run_ids are required")

    return args


# -----------------------------------------------------------------------------
# WandB helpers ----------------------------------------------------------------


def _load_wandb_cfg() -> Dict[str, str]:
    cfg = OmegaConf.load("config/config.yaml")
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project}


def _download_run_history(run) -> pd.DataFrame:
    """Safely download full history (stream=True skips huge memory allocations)."""
    return run.history(keys=None, samples=10000)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Per-run processing -----------------------------------------------------------


def _save_metrics_json(path: Path, summary: dict, history_df: pd.DataFrame):
    out = {
        "summary": summary,
        "history_columns": list(history_df.columns),
    }
    path.write_text(json.dumps(out, indent=2))


def _plot_learning_curve(df: pd.DataFrame, metric: str, out_path: Path, y_lim=None):
    if metric not in df.columns:
        return  # metric absent in this run
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x=df.index, y=metric, linewidth=2)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(metric, fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', labelsize=12)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Aggregation helpers ----------------------------------------------------------


def _collect_primary_metric(summary: dict) -> float:
    for key in ("accuracy", "best_val_acc", "val/acc"):
        if key in summary:
            return float(summary[key])
    raise KeyError("Could not locate primary accuracy metric in WandB summary")


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------


def main():
    args = _parse_args()
    results_dir = Path(args.results_dir).expanduser().absolute()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_ids: List[str] = json.loads(args.run_ids)

    wb_cfg = _load_wandb_cfg()
    api = wandb.Api()

    primary_metric_name = "accuracy"
    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    per_run_primary: Dict[str, float] = {}

    generated_paths: List[str] = []

    # First pass: collect all histories to determine consistent Y-axis ranges
    all_histories: Dict[str, pd.DataFrame] = {}
    metric_ranges: Dict[str, tuple] = {}

    for rid in run_ids:
        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{rid}")
        history = _download_run_history(run)
        all_histories[rid] = history

        # Track min/max for each metric across all runs
        for metric in ("train/loss", "val/loss", "train/acc", "val/acc"):
            if metric in history.columns:
                metric_min = history[metric].min()
                metric_max = history[metric].max()
                if metric in metric_ranges:
                    curr_min, curr_max = metric_ranges[metric]
                    metric_ranges[metric] = (min(curr_min, metric_min), max(curr_max, metric_max))
                else:
                    metric_ranges[metric] = (metric_min, metric_max)

    # Add padding to ranges for better visualization
    for metric in metric_ranges:
        min_val, max_val = metric_ranges[metric]
        padding = (max_val - min_val) * 0.1
        metric_ranges[metric] = (max(0, min_val - padding), max_val + padding)

    # -------------------------------- per run --------------------------------
    for rid in run_ids:
        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{rid}")
        history = all_histories[rid]
        summary = run.summary._json_dict
        run_cfg = dict(run.config)

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON ---------------------------------------------------
        m_path = run_dir / "metrics.json"
        _save_metrics_json(m_path, summary, history)
        generated_paths.append(str(m_path))

        # Figures -------------------------------------------------------------
        for metric in [m for m in ("train/loss", "val/loss", "train/acc", "val/acc") if m in history.columns]:
            fig_p = run_dir / f"{rid}_{metric.replace('/', '_')}.pdf"
            y_lim = metric_ranges.get(metric)
            _plot_learning_curve(history, metric, fig_p, y_lim=y_lim)
            generated_paths.append(str(fig_p))

        # Aggregate -----------------------------------------------------------
        primary_val = _collect_primary_metric(summary)
        per_run_primary[rid] = primary_val
        aggregated_metrics.setdefault(primary_metric_name, {})[rid] = primary_val

    # ----------------------- Cross-run comparison ---------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart of primary metric -------------------------------------------
    plt.figure(figsize=(10, 6))
    order = sorted(per_run_primary.keys())
    values = [per_run_primary[o] for o in order]

    # Create shortened labels for better readability
    labels = []
    for o in order:
        if "comparative" in o or "baseline" in o:
            labels.append("Comparative")
        elif "proposed" in o:
            labels.append("Proposed")
        else:
            labels.append(o)

    # Create bar plot with better styling
    colors = ['#1f77b4' if 'comparative' in o or 'baseline' in o else '#ff7f0e' for o in order]
    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    plt.ylabel(primary_metric_name.capitalize(), fontsize=16, fontweight='bold')
    plt.xlabel("Method", fontsize=16, fontweight='bold')

    # Set Y-axis range to focus on the data (start from 80% of min value)
    min_val = min(values)
    max_val = max(values)
    y_range = max_val - min_val
    plt.ylim(max(0, min_val - y_range * 0.2), max_val + y_range * 0.15)

    plt.title(f"{primary_metric_name.capitalize()} Comparison", fontsize=18, fontweight='bold', pad=20)
    plt.tick_params(axis='both', labelsize=14)

    # Add value labels on top of bars
    for idx, (bar, v) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{v:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    comp_fig = comp_dir / "comparison_accuracy_bar_chart.pdf"
    plt.savefig(comp_fig, dpi=300, bbox_inches='tight')
    plt.close()
    generated_paths.append(str(comp_fig))

    # --------------------- Compute best runs & gap --------------------------
    best_proposed_id, best_baseline_id = None, None
    best_proposed_val, best_baseline_val = -1.0, -1.0
    for rid, val in per_run_primary.items():
        if "proposed" in rid and val > best_proposed_val:
            best_proposed_id, best_proposed_val = rid, val
        if any(tag in rid for tag in ("comparative", "baseline")) and val > best_baseline_val:
            best_baseline_id, best_baseline_val = rid, val

    gap = None
    if best_proposed_val > 0 and best_baseline_val > 0:
        gap = (best_proposed_val - best_baseline_val) / best_baseline_val * 100.0

    # Aggregated JSON --------------------------------------------------------
    agg_json = {
        "primary_metric": primary_metric_name,
        "metrics": aggregated_metrics,
        "best_proposed": {"run_id": best_proposed_id, "value": best_proposed_val},
        "best_baseline": {"run_id": best_baseline_id, "value": best_baseline_val},
        "gap": gap,
    }
    agg_path = comp_dir / "aggregated_metrics.json"
    agg_path.write_text(json.dumps(agg_json, indent=2))
    generated_paths.append(str(agg_path))

    # ---------------------------- STDOUT ------------------------------------
    print("\n[ evaluate ] Generated files:")
    for p in generated_paths:
        print(p)


if __name__ == "__main__":
    main()
