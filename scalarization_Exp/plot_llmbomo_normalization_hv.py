from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_scalarization_hv import _load_json, _stack_mode, _summary_paths_by_mode


MODES = ("minmax", "zscore", "none")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "minmax": "#b2182b",
    "zscore": "#2166ac",
    "none": "#7b2cbf",
}
LABELS = {
    "minmax": "Min-max",
    "zscore": "Z-score",
    "none": "No normalization",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the LLMBO-MO objective-normalization HV curves."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iteration", type=int, default=50)
    parser.add_argument("--max-evaluations", type=int, default=50)
    parser.add_argument("--output-stem", type=str, default="llmbomo_normalization_hv_first50")
    parser.add_argument("--y-label", type=str, default="HV")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d8d8d8",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    output_dir = args.output_dir.resolve()
    report = _load_json(report_path)
    summary_paths = _summary_paths_by_mode(report, list(MODES))
    summary_paths = {
        mode: [path if path.is_absolute() else PROJECT_ROOT / path for path in paths]
        for mode, paths in summary_paths.items()
    }

    traces: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        trace = _stack_mode(summary_paths[mode], max_iteration=args.max_iteration)
        if trace is None:
            raise RuntimeError(f"No valid trace for mode={mode}")
        keep = np.asarray(trace["x"], dtype=int) <= int(args.max_evaluations)
        if not np.any(keep):
            raise RuntimeError(
                f"No evaluations at or below {args.max_evaluations} for mode={mode}"
            )
        trace = {
            **trace,
            "x": np.asarray(trace["x"])[keep],
            "mean": np.asarray(trace["mean"])[keep],
            "std": np.asarray(trace["std"])[keep],
        }
        traces[mode] = trace

    configure_style()
    fig, ax = plt.subplots(figsize=(4.55, 3.35))
    for mode in MODES:
        trace = traces[mode]
        x = np.asarray(trace["x"], dtype=float)
        mean = np.asarray(trace["mean"], dtype=float)
        std = np.asarray(trace["std"], dtype=float)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[mode],
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(x, mean, color=COLORS[mode], linewidth=1.65, label=LABELS[mode])

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel(args.y_label)
    ax.set_xlim(1, max(float(traces[mode]["x"][-1]) for mode in MODES))
    tick_candidates = [1, 10, 20, 30, 40, 50]
    ax.set_xticks([tick for tick in tick_candidates if tick <= args.max_evaluations])
    ax.grid(True)
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#888888",
        borderpad=0.35,
        handlelength=2.2,
        labelspacing=0.25,
    )
    fig.tight_layout(pad=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.output_stem}.png"
    pdf_path = output_dir / f"{args.output_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"{args.output_stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "evaluation",
                "minmax_mean",
                "minmax_sample_sd",
                "zscore_mean",
                "zscore_sample_sd",
                "none_mean",
                "none_sample_sd",
            ]
        )
        common_x = traces["minmax"]["x"]
        for idx, evaluation in enumerate(common_x):
            writer.writerow(
                [
                    int(evaluation),
                    *[
                        value
                        for mode in MODES
                        for value in (
                            f"{float(traces[mode]['mean'][idx]):.10f}",
                            f"{float(traces[mode]['std'][idx]):.10f}",
                        )
                    ],
                ]
            )

    manifest = {
        "source_report": str(report_path),
        "maximum_evaluation": int(args.max_evaluations),
        "n_seeds": {mode: int(traces[mode]["n_runs"]) for mode in MODES},
        "uncertainty": "mean +/- 1 sample standard deviation across seeds",
        "final_hv": {
            mode: {
                "mean": float(traces[mode]["mean"][-1]),
                "sample_sd": float(traces[mode]["std"][-1]),
            }
            for mode in MODES
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "csv": str(csv_path),
        },
    }
    manifest_path = output_dir / f"{args.output_stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
