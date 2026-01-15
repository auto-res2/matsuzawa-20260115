"""Main orchestrator – launches *train* as a subprocess under the correct Hydra run config.
CLI (from repo root):
    uv run python -u -m src.main run=<run_id> results_dir=<path> mode=full|trial
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig


# -----------------------------------------------------------------------------
# Helper -----------------------------------------------------------------------


def _build_cmd(cfg: DictConfig) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]


# -----------------------------------------------------------------------------
# Entry-point ------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):  # pragma: no cover
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    cfg.results_dir = str(Path(cfg.results_dir).expanduser().absolute())

    cmd = _build_cmd(cfg)
    print("[main] Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
