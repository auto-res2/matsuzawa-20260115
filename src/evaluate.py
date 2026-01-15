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
    p = argparse.ArgumentParser("Few-shot evaluation summariser")
    p.add_argument("results_dir", type=str, help="Output directory for metrics/figures")
    p.add_argument("run_ids", type=str, help="JSON list of WandB run IDs")
    return p.parse_args()


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


def _plot_learning_curve(df: pd.DataFrame, metric: str, out_path: Path):
    if metric not in df.columns:
        return  # metric absent in this run
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x=df.index, y=metric)
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
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

    # -------------------------------- per run --------------------------------
    for rid in run_ids:
        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{rid}")
        history = _download_run_history(run)
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
            _plot_learning_curve(history, metric, fig_p)
            generated_paths.append(str(fig_p))

        # Aggregate -----------------------------------------------------------
        primary_val = _collect_primary_metric(summary)
        per_run_primary[rid] = primary_val
        aggregated_metrics.setdefault(primary_metric_name, {})[rid] = primary_val

    # ----------------------- Cross-run comparison ---------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart of primary metric -------------------------------------------
    plt.figure(figsize=(8, 4))
    order = sorted(per_run_primary.keys())
    values = [per_run_primary[o] for o in order]
    sns.barplot(x=order, y=values)
    plt.ylabel(primary_metric_name)
    plt.ylim(0, max(values) * 1.1)
    plt.title(f"{primary_metric_name} comparison")
    for idx, v in enumerate(values):
        plt.text(idx, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    comp_fig = comp_dir / "comparison_accuracy_bar_chart.pdf"
    plt.savefig(comp_fig, dpi=200)
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
