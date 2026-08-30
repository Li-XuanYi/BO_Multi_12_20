"""Create the same-batch Chen2020 ablation figure used by the paper.

The script intentionally reads only the 2026-05-22 ``adaptive4`` experiment.
It does not use the later paired rerun.  The four variants therefore share
the same experiment batch, seeds, simulator budget, and reporting pipeline.

Outputs
-------
figures/ablation_same_batch.pdf
figures/ablation_same_batch.png

Run from any directory:

    python scripts/make_ablation_figure.py

Use ``--source-root`` if the experiment archive has been moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "Ablation_Exp"
    / "experiment_records"
    / "adaptive4_5seeds_50iter_deepseek_v3_2026_05_22"
)
EXPECTED_SEEDS = (8409, 8410, 8411, 8412, 8413)
EXPECTED_EVALUATIONS = 56

VARIANTS = (
    {
        "key": "baseline",
        "label": "Plain BO",
        "color": "#4C566A",
        "marker": "o",
    },
    {
        "key": "baseline_warmstart",
        "label": "Warm start",
        "color": "#0072B2",
        "marker": "s",
    },
    {
        "key": "baseline_llm_region",
        "label": "Region lift",
        "color": "#E69F00",
        "marker": "^",
    },
    {
        "key": "llmbo_mo",
        "label": "Full (warm + lift)",
        "color": "#CC6677",
        "marker": "D",
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_trace(summary: dict[str, Any], path: Path) -> tuple[np.ndarray, np.ndarray]:
    entries = summary.get("hv_trace")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Missing hv_trace in {path}")

    indices: list[int] = []
    values: list[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed hv_trace entry in {path}")
        index = entry.get("eval_index")
        value = entry.get("canonical_hv", entry.get("hypervolume_canonical"))
        if index is None or value is None:
            raise ValueError(f"Missing eval_index/canonical_hv in {path}")
        indices.append(int(index))
        values.append(float(value))

    return np.asarray(indices, dtype=int), np.asarray(values, dtype=float)


def load_same_batch(
    source_root: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load per-seed traces and final HV values from one experiment batch."""

    report_path = source_root / "report_5seeds.json"
    report = _load_json(report_path)
    meta = report.get("meta", {})

    if int(meta.get("iterations", -1)) != 50:
        raise ValueError(f"Expected 50 BO iterations in {report_path}")
    if tuple(int(seed) for seed in meta.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seed set in {report_path}")
    if str(meta.get("hv_metric")) != "canonical_hv":
        raise ValueError(f"Expected canonical_hv metric in {report_path}")

    traces: dict[str, np.ndarray] = {}
    finals: dict[str, np.ndarray] = {}
    shared_indices: np.ndarray | None = None

    for variant in VARIANTS:
        key = str(variant["key"])
        variant_traces: list[np.ndarray] = []
        variant_finals: list[float] = []

        for seed in EXPECTED_SEEDS:
            summary_path = source_root / f"seed{seed}" / key / "summary.json"
            summary = _load_json(summary_path)
            config = summary.get("config", {})

            if config.get("battery_param_set") != "Chen2020":
                raise ValueError(f"Expected Chen2020 in {summary_path}")
            if int(config.get("max_iterations", -1)) != 50:
                raise ValueError(f"Expected 50 iterations in {summary_path}")
            if int(summary.get("n_total", -1)) != EXPECTED_EVALUATIONS:
                raise ValueError(f"Expected 56 evaluations in {summary_path}")

            indices, trace = _canonical_trace(summary, summary_path)
            if len(indices) != EXPECTED_EVALUATIONS:
                raise ValueError(f"Expected 56 trace entries in {summary_path}")
            if not np.array_equal(indices, np.arange(1, EXPECTED_EVALUATIONS + 1)):
                raise ValueError(f"Non-contiguous evaluation indices in {summary_path}")

            final = float(summary["canonical_hv"])
            if not np.isclose(trace[-1], final, rtol=0.0, atol=1e-12):
                raise ValueError(f"Final trace value differs from summary in {summary_path}")

            if shared_indices is None:
                shared_indices = indices
            elif not np.array_equal(shared_indices, indices):
                raise ValueError(f"Trace indices differ in {summary_path}")

            variant_traces.append(trace)
            variant_finals.append(final)

        traces[key] = np.vstack(variant_traces)
        finals[key] = np.asarray(variant_finals, dtype=float)

    if shared_indices is None:
        raise RuntimeError("No traces were loaded")
    return shared_indices, traces, finals


def summarize(finals: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    baseline = finals["baseline"]
    rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        key = str(variant["key"])
        values = finals[key]
        delta = None if key == "baseline" else float(np.mean(values) - np.mean(baseline))
        wins = None if key == "baseline" else int(np.sum(values > baseline))
        rows.append(
            {
                "key": key,
                "label": variant["label"],
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "delta_vs_baseline": delta,
                "wins_vs_baseline": wins,
                "seeds": list(EXPECTED_SEEDS),
                "values": [float(value) for value in values],
            }
        )
    return rows


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_figure(
    indices: np.ndarray,
    traces: dict[str, np.ndarray],
    finals: dict[str, np.ndarray],
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    _set_style()
    fig, (ax_curve, ax_box) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.15),
        gridspec_kw={"width_ratios": (1.58, 1.0), "wspace": 0.26},
    )

    # (a) Mean convergence with sample-standard-deviation bands.
    for variant in VARIANTS:
        key = str(variant["key"])
        stack = traces[key]
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0, ddof=1)
        color = str(variant["color"])

        ax_curve.fill_between(
            indices,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )
        ax_curve.plot(
            indices,
            mean,
            color=color,
            marker=str(variant["marker"]),
            markevery=(5, 10),
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=str(variant["label"]),
            zorder=3,
        )

    ax_curve.axvline(6.5, color="#777777", linestyle=(0, (2, 2)), linewidth=0.75)
    ax_curve.text(
        6.5,
        0.018,
        "BO iterations",
        color="#555555",
        fontsize=7.8,
        ha="left",
        va="bottom",
    )
    ax_curve.set_xlim(1, EXPECTED_EVALUATIONS)
    # Starting from zero prevents small final-HV differences from being exaggerated.
    ax_curve.set_ylim(0.0, 0.42)
    ax_curve.xaxis.set_major_locator(MultipleLocator(10))
    ax_curve.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_curve.set_xlabel("Cumulative simulator evaluations")
    ax_curve.set_ylabel("Scaled hypervolume")
    ax_curve.set_title("(a) Search progress", loc="left", fontweight="bold", pad=4)
    ax_curve.grid(axis="both", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax_curve.set_axisbelow(True)
    ax_curve.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#A0A0A0",
        facecolor="white",
        framealpha=0.96,
        handlelength=2.2,
        columnspacing=0.9,
        borderpad=0.45,
    )

    # (b) Final-HV distribution with paired seed traces and visible seed points.
    positions = np.arange(1, len(VARIANTS) + 1, dtype=float)
    box_data = [finals[str(variant["key"])] for variant in VARIANTS]
    colors = [str(variant["color"]) for variant in VARIANTS]

    for seed_index in range(len(EXPECTED_SEEDS)):
        seed_values = [values[seed_index] for values in box_data]
        ax_box.plot(
            positions,
            seed_values,
            color="#C7C7C7",
            linewidth=0.55,
            alpha=0.8,
            zorder=1,
        )

    box = ax_box.boxplot(
        box_data,
        positions=positions,
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"linewidth": 0.9},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.16)

    # The fixed offsets make the strip plot deterministic and keep all seeds visible.
    offsets = np.linspace(-0.10, 0.10, len(EXPECTED_SEEDS))
    for position, values, variant, color in zip(positions, box_data, VARIANTS, colors):
        ax_box.scatter(
            position + offsets,
            values,
            s=20,
            marker=str(variant["marker"]),
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.35,
            zorder=4,
        )
        mean = float(np.mean(values))
        ax_box.scatter(
            [position],
            [mean],
            s=30,
            marker="*",
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=5,
        )

    ax_box.set_xlim(0.55, 4.45)
    # A distribution plot may use the observed range; all points and whiskers remain visible.
    ax_box.set_ylim(0.364, 0.4075)
    ax_box.yaxis.set_major_locator(MultipleLocator(0.01))
    ax_box.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(
        ["Plain\nBO", "Warm\nstart", "Region\nlift", "Full\n(warm + lift)"],
        rotation=0,
    )
    ax_box.set_ylabel("Final scaled hypervolume")
    ax_box.set_title("(b) Final values by seed", loc="left", fontweight="bold", pad=4)
    ax_box.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
    ax_box.set_axisbelow(True)
    ax_box.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markersize=6.5,
                label="mean",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#C7C7C7",
                markerfacecolor="#777777",
                markeredgecolor="#222222",
                linewidth=0.7,
                markersize=4.0,
                label="paired seed",
            ),
        ],
        loc="lower left",
        frameon=False,
        handlelength=1.4,
        borderpad=0.1,
        labelspacing=0.25,
    )

    for axis in (ax_curve, ax_box):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.7)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.91)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Path to adaptive4_5seeds_50iter_deepseek_v3_2026_05_22.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=PAPER_ROOT / "Section" / "figures" / "ablation_same_batch",
        help="Output path without extension.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_base = args.output_base.resolve()

    indices, traces, finals = load_same_batch(source_root)
    rows = summarize(finals)
    make_figure(
        indices,
        traces,
        finals,
        output_base.with_suffix(".pdf"),
        output_base.with_suffix(".png"),
        args.dpi,
    )

    payload = {
        "source_root": str(source_root),
        "dataset": "Chen2020",
        "seeds": list(EXPECTED_SEEDS),
        "bo_iterations": 50,
        "total_simulator_evaluations": EXPECTED_EVALUATIONS,
        "metric": "benchmark_scaled_hv (archive key: canonical_hv)",
        "uncertainty": "sample standard deviation (ddof=1)",
        "statistics": rows,
        "outputs": {
            "pdf": str(output_base.with_suffix(".pdf")),
            "png": str(output_base.with_suffix(".png")),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
