"""Rebuild the paper's benchmark and Pareto figures from archived runs.

The benchmark figure uses five matched seeds (8409--8413). Chen2020 traces
are replayed from each database because an archived ParEGO summary stored the
final HV at every trace entry. Ecker2015 traces are read from the archived
per-seed CSV files produced by the same database replay routine.

Outputs
-------
paper/Section/figures/benchmark_hv.{pdf,png}
paper/Section/figures/pareto_projections.{pdf,png}
paper/Section/figures/comparison_figure_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np


SEEDS = (8409, 8410, 8411, 8412, 8413)
MAX_EVALS = 56

NSGA2_COLOR = "#666666"
PAREGO_COLOR = "#2B6F9E"
LLMBO_COLOR = "#C74747"
GRID_COLOR = "#D8D8D8"

CHEN_EXPECTED = {
    "NSGA-II": (0.322238565880124, 0.0236750135547637),
    "ParEGO": (0.3853050989788063, 0.009357156930015625),
    "LLMBO-MO": (0.3835295901394017, 0.007890651338768746),
}
ECKER_EXPECTED = {
    "ParEGO": (1.5865626975702702, 0.012992483331202098),
    "LLMBO-MO": (1.8684172232658676, 0.002730628128270058),
}
EXPECTED_PARETO_COUNTS = {"ParEGO": 45, "LLMBO-MO": 37}


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "Compare_Exp").is_dir() and (candidate / "paper").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pymoo.config import Config

    Config.warnings["not_compiled"] = False
except Exception:
    pass

from DataBase.database import ObservationDB  # noqa: E402


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _prefix_trace(database_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Replay the first 56 observations with the database's stored HV box."""
    db = ObservationDB.load(str(database_path))
    observations = db.get_all()
    if len(observations) < MAX_EVALS:
        raise ValueError(
            f"{database_path} contains {len(observations)} observations; "
            f"expected at least {MAX_EVALS}"
        )

    replay = ObservationDB(
        param_bounds=db.param_bounds,
        ref_point=db.ref_point.copy(),
        ideal_point=db.ideal_point.copy(),
        normalize=db.normalize,
    )
    values: list[float] = []
    for eval_index, observation in enumerate(observations[:MAX_EVALS], start=1):
        replay.add_observation(
            theta=observation.theta,
            objectives=observation.objectives,
            feasible=observation.feasible,
            violation=observation.violation,
            source=observation.source,
            iteration=eval_index,
            acq_value=observation.acq_value,
            acq_type=observation.acq_type,
            gp_pred=observation.gp_pred,
            llm_rationale=observation.llm_rationale,
            details=observation.details,
        )
        values.append(float(replay.compute_hypervolume_canonical()))

    x = np.arange(1, MAX_EVALS + 1, dtype=int)
    y = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(y)) or np.any(np.diff(y) < -1e-12):
        raise ValueError(f"Invalid replayed HV trace in {database_path}")
    return x, y


def _stack_database_traces(
    root: Path, variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    traces: list[np.ndarray] = []
    inputs: list[str] = []
    x_ref: np.ndarray | None = None
    for seed in SEEDS:
        path = root / f"seed{seed}" / variant / "database.json"
        x, trace = _prefix_trace(path)
        if x_ref is None:
            x_ref = x
        elif not np.array_equal(x_ref, x):
            raise ValueError(f"Evaluation axes differ in {path}")
        traces.append(trace)
        inputs.append(str(path))

    stack = np.vstack(traces)
    assert x_ref is not None
    return (
        x_ref,
        stack.mean(axis=0),
        stack.std(axis=0, ddof=1),
        stack[:, -1],
        inputs,
    )


def _load_ecker_traces(
    root: Path, method_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    traces: list[np.ndarray] = []
    inputs: list[str] = []
    x_ref: np.ndarray | None = None
    for seed in SEEDS:
        path = root / method_dir / f"seed{seed}.csv"
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        x = np.asarray([int(row["eval_index"]) for row in rows], dtype=int)
        y = np.asarray([float(row["canonical_hv"]) for row in rows], dtype=float)
        expected = np.arange(1, MAX_EVALS + 1, dtype=int)
        if not np.array_equal(x, expected):
            raise ValueError(f"Unexpected evaluation axis in {path}")
        if not np.all(np.isfinite(y)) or np.any(np.diff(y) < -1e-12):
            raise ValueError(f"Invalid archived HV trace in {path}")
        if x_ref is None:
            x_ref = x
        traces.append(y)
        inputs.append(str(path))

    stack = np.vstack(traces)
    assert x_ref is not None
    return (
        x_ref,
        stack.mean(axis=0),
        stack.std(axis=0, ddof=1),
        stack[:, -1],
        inputs,
    )


def _validate_stats(
    name: str, finals: np.ndarray, expected: tuple[float, float]
) -> None:
    actual = (float(np.mean(finals)), float(np.std(finals, ddof=1)))
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} statistics changed: {actual} != {expected}")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.7)


def _plot_mean_band(
    ax: plt.Axes,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
    *,
    label: str,
    color: str,
    marker: str,
) -> Line2D:
    x, mean, std, _, _ = data
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.13,
        linewidth=0.0,
        zorder=1,
    )
    (line,) = ax.plot(
        x,
        mean,
        color=color,
        marker=marker,
        markevery=(5, 10),
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label=label,
        zorder=3,
    )
    return line


def make_benchmark_figure(
    project_root: Path, output_dir: Path, dpi: int
) -> dict[str, Any]:
    chen_root = (
        project_root
        / "Compare_Exp"
        / "experiment_records"
        / "computational_time_3algo_5seeds_50iter_2026_05_12"
    )
    chen = {
        "NSGA-II": _stack_database_traces(chen_root, "nsga2"),
        "ParEGO": _stack_database_traces(chen_root, "parego"),
        "LLMBO-MO": _stack_database_traces(chen_root, "llmbo_mo"),
    }
    for method, data in chen.items():
        _validate_stats(f"Chen2020/{method}", data[3], CHEN_EXPECTED[method])

    ecker_root = (
        project_root
        / "Compare_Exp"
        / "experiment_records"
        / "Ecker2015_HV05-12"
        / "curve_data"
        / "per_seed_traces"
    )
    ecker = {
        "ParEGO": _load_ecker_traces(ecker_root, "ParEGO"),
        "LLMBO-MO": _load_ecker_traces(ecker_root, "LLMBO-MO"),
    }
    for method, data in ecker.items():
        _validate_stats(f"Ecker2015/{method}", data[3], ECKER_EXPECTED[method])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.82, wspace=0.25)

    chen_specs = (
        ("NSGA-II", NSGA2_COLOR, "^"),
        ("ParEGO", PAREGO_COLOR, "s"),
        ("LLMBO-MO", LLMBO_COLOR, "o"),
    )
    handles = [
        _plot_mean_band(
            axes[0],
            chen[method],
            label=method,
            color=color,
            marker=marker,
        )
        for method, color, marker in chen_specs
    ]
    axes[0].set_title("(a) Chen2020", loc="left", fontweight="bold", pad=4)
    axes[0].set_xlabel("Cumulative simulator evaluations")
    axes[0].set_ylabel("Scaled hypervolume")
    axes[0].set_xlim(1, MAX_EVALS)
    axes[0].set_ylim(0.0, 0.42)
    axes[0].xaxis.set_major_locator(MultipleLocator(10))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.1))
    _style_axis(axes[0])

    for method, color, marker in (
        ("ParEGO", PAREGO_COLOR, "s"),
        ("LLMBO-MO", LLMBO_COLOR, "o"),
    ):
        _plot_mean_band(
            axes[1],
            ecker[method],
            label=method,
            color=color,
            marker=marker,
        )
    axes[1].set_title("(b) Ecker2015", loc="left", fontweight="bold", pad=4)
    axes[1].set_xlabel("Cumulative simulator evaluations")
    axes[1].set_ylabel("Scaled hypervolume")
    axes[1].set_xlim(1, MAX_EVALS)
    axes[1].set_ylim(0.0, 1.95)
    axes[1].xaxis.set_major_locator(MultipleLocator(10))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.4))
    _style_axis(axes[1])

    for ax in axes:
        ax.axvline(6.5, color="#777777", linestyle=(0, (2, 2)), linewidth=0.7)

    fig.legend(
        handles=handles,
        labels=["NSGA-II", "ParEGO", "LLMBO-MO"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.7,
        handlelength=2.2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "benchmark_hv.pdf"
    png_path = output_dir / "benchmark_hv.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    return {
        "inputs": [
            source
            for benchmark in (chen, ecker)
            for data in benchmark.values()
            for source in data[4]
        ],
        "outputs": [str(pdf_path), str(png_path)],
        "statistics": {
            "Chen2020": {
                method: {
                    "mean": float(np.mean(data[3])),
                    "sample_sd": float(np.std(data[3], ddof=1)),
                    "values": data[3].tolist(),
                }
                for method, data in chen.items()
            },
            "Ecker2015": {
                method: {
                    "mean": float(np.mean(data[3])),
                    "sample_sd": float(np.std(data[3], ddof=1)),
                    "values": data[3].tolist(),
                }
                for method, data in ecker.items()
            },
        },
    }


def _load_pareto_set(path: Path) -> np.ndarray:
    rows = _load_json(path)
    objectives = np.asarray([row["objectives"] for row in rows], dtype=float)
    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 objective array in {path}")
    if not np.all(np.isfinite(objectives)):
        raise ValueError(f"Non-finite objective in {path}")
    for index, point in enumerate(objectives):
        weakly_better = np.all(objectives <= point, axis=1)
        strictly_better = np.any(objectives < point, axis=1)
        if np.any(weakly_better & strictly_better):
            raise ValueError(f"Dominated point {index} in {path}")
    return objectives


def _scatter_projection(
    ax: plt.Axes,
    parego: np.ndarray,
    llmbo: np.ndarray,
    x_index: int,
    y_index: int,
) -> None:
    ax.scatter(
        parego[:, x_index],
        parego[:, y_index],
        s=17,
        marker="s",
        facecolors="none",
        edgecolors=PAREGO_COLOR,
        linewidths=0.75,
        alpha=0.88,
        zorder=2,
    )
    ax.scatter(
        llmbo[:, x_index],
        llmbo[:, y_index],
        s=17,
        marker="o",
        facecolors=LLMBO_COLOR,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.72,
        zorder=3,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.045, y=0.075)
    _style_axis(ax)


def make_pareto_figure(
    project_root: Path, output_dir: Path, dpi: int
) -> dict[str, Any]:
    root = (
        project_root
        / "Compare_Exp"
        / "experiment_records"
        / "seed8409_llmbo_vs_parego_50iter"
        / "seed8409"
    )
    inputs = {
        "ParEGO": root / "parego_matlab_reference" / "pareto_front.json",
        "LLMBO-MO": root / "llmbo_mo" / "pareto_front.json",
    }
    sets = {method: _load_pareto_set(path) for method, path in inputs.items()}
    for method, points in sets.items():
        if points.shape[0] != EXPECTED_PARETO_COUNTS[method]:
            raise ValueError(f"Unexpected {method} Pareto count: {points.shape[0]}")

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    fig.subplots_adjust(left=0.073, right=0.99, bottom=0.23, top=0.78, wspace=0.30)
    labels = {
        0: "Charging time (s)",
        1: "Peak temperature rise (K)",
        2: r"Degradation proxy, $D_{\mathrm{chg}}$ (a.u.)",
    }
    projections = (
        (0, 1, "(a) Time--temperature"),
        (0, 2, "(b) Time--proxy"),
        (1, 2, "(c) Temperature--proxy"),
    )
    for ax, (x_index, y_index, title) in zip(axes, projections):
        _scatter_projection(
            ax,
            sets["ParEGO"],
            sets["LLMBO-MO"],
            x_index,
            y_index,
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_xlabel(labels[x_index])
        ax.set_ylabel(labels[y_index])

    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markersize=4.5,
            markerfacecolor="none",
            markeredgecolor=PAREGO_COLOR,
            markeredgewidth=0.8,
            label="ParEGO",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor=LLMBO_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label="LLMBO-MO",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "pareto_projections.pdf"
    png_path = output_dir / "pareto_projections.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    return {
        "inputs": [str(path) for path in inputs.values()],
        "outputs": [str(pdf_path), str(png_path)],
        "pareto_counts": {
            method: int(points.shape[0]) for method, points in sets.items()
        },
    }


def _check_outputs(paths: Iterable[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="New_LLMBO project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "Section" / "figures",
        help="Output directory for paper figures.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    _configure_style()

    benchmark = make_benchmark_figure(project_root, output_dir, args.dpi)
    pareto = make_pareto_figure(project_root, output_dir, args.dpi)
    _check_outputs(benchmark["outputs"] + pareto["outputs"])

    manifest = {
        "metric": (
            "Benchmark-scaled hypervolume: log10 transforms are applied to "
            "charging time and the degradation proxy, then raw HV is divided by the "
            "benchmark's nominal ideal-to-reference box volume. No clipping "
            "is applied, so values can exceed one."
        ),
        "seeds": list(SEEDS),
        "evaluations": MAX_EVALS,
        "uncertainty": "sample standard deviation (ddof=1)",
        "benchmark_hv": benchmark,
        "pareto_projections": pareto,
    }
    manifest_path = output_dir / "comparison_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
