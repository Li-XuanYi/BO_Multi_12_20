"""
plot_disk_pimd_hv.py - HV convergence plot for DISK vs PIMD
===========================================================

Creates hv_convergence_5.png with green (DISK) and purple (PIMD) lines
plus shaded confidence bands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_hv_trace(summary_path: Path) -> List[Dict]:
    """Load hv_trace from summary.json."""
    with open(summary_path, "r") as f:
        data = json.load(f)
    return data.get("hv_trace", [])


def collect_algorithm_data(exp_dir: Path, algorithm: str) -> Dict[int, List[float]]:
    """
    Collect canonical_hv values per eval_index across all seeds.
    Returns: {eval_index: [hv_seed1, hv_seed2, ...]}
    """
    hv_by_eval: Dict[int, List[float]] = {}

    for seed_dir in sorted(exp_dir.glob("seed*")):
        summary_path = seed_dir / f"{algorithm}_Chen2020" / "summary.json"
        if not summary_path.exists():
            continue

        hv_trace = load_hv_trace(summary_path)
        for entry in hv_trace:
            eval_idx = entry["eval_index"]
            canonical_hv = entry["canonical_hv"]
            if eval_idx not in hv_by_eval:
                hv_by_eval[eval_idx] = []
            hv_by_eval[eval_idx].append(canonical_hv)

    return hv_by_eval


def compute_stats(hv_by_eval: Dict[int, List[float]]) -> tuple:
    """Compute mean and std for each eval_index."""
    eval_indices = sorted(hv_by_eval.keys())
    means = []
    stds = []

    for idx in eval_indices:
        values = hv_by_eval[idx]
        means.append(np.mean(values))
        stds.append(np.std(values))

    return np.array(eval_indices), np.array(means), np.array(stds)


def main():
    exp_root = Path(__file__).parent / "experiment_records"

    # Find experiment directories
    disk_dirs = list(exp_root.glob("disk_python_*_5seeds_*"))
    pimd_dirs = list(exp_root.glob("pimd_python_*_5seeds_*"))

    if not disk_dirs:
        print("Error: No DISK experiment directory found")
        return 1
    if not pimd_dirs:
        print("Error: No PIMD experiment directory found")
        return 1

    disk_dir = disk_dirs[0]
    pimd_dir = pimd_dirs[0]

    print(f"Loading DISK data from: {disk_dir}")
    print(f"Loading PIMD data from: {pimd_dir}")

    # Collect data
    disk_hv = collect_algorithm_data(disk_dir, "disk")
    pimd_hv = collect_algorithm_data(pimd_dir, "pimd")

    # Compute statistics
    disk_x, disk_mean, disk_std = compute_stats(disk_hv)
    pimd_x, pimd_mean, pimd_std = compute_stats(pimd_hv)

    print(f"DISK: {len(disk_x)} evaluation points, {len(disk_hv[1])} seeds")
    print(f"PIMD: {len(pimd_x)} evaluation points, {len(pimd_hv[1])} seeds")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # DISK - Green line with shaded band
    ax.plot(disk_x, disk_mean, color="#2E8B57", linewidth=2, label="DISK", linestyle="-")
    ax.fill_between(
        disk_x,
        disk_mean - disk_std,
        disk_mean + disk_std,
        color="#2E8B57",
        alpha=0.3,
        label="DISK ±1 std"
    )

    # PIMD - Purple line with shaded band
    ax.plot(pimd_x, pimd_mean, color="#8A2BE2", linewidth=2, label="PIMD", linestyle="-")
    ax.fill_between(
        pimd_x,
        pimd_mean - pimd_std,
        pimd_mean + pimd_std,
        color="#8A2BE2",
        alpha=0.3,
        label="PIMD ±1 std"
    )

    # Styling
    ax.set_xlabel("Number of Evaluations", fontsize=12)
    ax.set_ylabel("Hypervolume (Canonical)", fontsize=12)
    ax.set_title("HV Convergence: DISK vs PIMD (5 seeds, 50 evals)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, max(max(disk_x), max(pimd_x)) + 1)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "images" / "hv_convergence_5.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics")
    print("=" * 50)
    print(f"DISK Final HV: {disk_mean[-1]:.4f} ± {disk_std[-1]:.4f}")
    print(f"PIMD Final HV: {pimd_mean[-1]:.4f} ± {pimd_std[-1]:.4f}")

    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
