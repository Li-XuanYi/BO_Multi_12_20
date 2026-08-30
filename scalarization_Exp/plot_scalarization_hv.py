from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.scalarization import OBJECTIVE_PREPROCESS_MODES, canonicalize_objective_preprocess_mode


COLORS = {
    "minmax": "#0072B2",
    "zscore": "#E69F00",
    "none": "#009E73",
}
LABELS = {
    "minmax": "Dynamic min-max",
    "zscore": "Z-score scaling",
    "none": "No objective scaling",
}
MARKERS = {
    "minmax": "o",
    "zscore": "s",
    "none": "^",
}
LINESTYLES = {
    "minmax": "-",
    "zscore": "--",
    "none": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scalarization preprocessing HV comparison.")
    parser.add_argument("--exp-root", type=Path, default=None, help="Experiment root with report_5seeds.json.")
    parser.add_argument("--report", type=Path, default=None, help="Explicit report_5seeds.json path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--modes", type=str, nargs="+", default=None, choices=OBJECTIVE_PREPROCESS_MODES)
    parser.add_argument("--max-iteration", type=int, default=50)
    return parser.parse_args()


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 19,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#666666",
            "grid.color": "#cfcfcf",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
        }
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_report(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.report is not None:
        report_path = Path(args.report)
        return report_path, report_path.parent
    if args.exp_root is None:
        raise ValueError("Provide --exp-root or --report.")
    exp_root = Path(args.exp_root)
    return exp_root / "report_5seeds.json", exp_root


def _summary_paths_by_mode(report: Dict[str, Any], modes: List[str]) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {mode: [] for mode in modes}
    for record in report.get("records") or []:
        if record.get("status") != "ok":
            continue
        mode = canonicalize_objective_preprocess_mode(record.get("mode"))
        if mode not in result:
            continue
        result[mode].append(Path(str(record["summary_path"])))
    return result


def _extract_evaluation_trace(summary: Dict[str, Any], max_iteration: int) -> Optional[Dict[str, np.ndarray]]:
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        return None

    by_eval: Dict[int, float] = {}
    for item in hv_trace:
        iteration = int(item.get("iteration", 0))
        if str(item.get("phase", "")) == "bo" and iteration > int(max_iteration):
            continue
        eval_index = int(item.get("eval_index", item.get("n_total", 0)))
        if eval_index <= 0:
            continue
        by_eval[eval_index] = float(item.get("canonical_hv", item.get("hypervolume_canonical", 0.0)))

    if not by_eval:
        return None

    xs = np.array(sorted(by_eval), dtype=int)
    ys = np.array([by_eval[int(x)] for x in xs], dtype=float)
    return {"x": xs, "hv": ys}


def _stack_mode(summary_paths: List[Path], max_iteration: int) -> Optional[Dict[str, Any]]:
    traces = []
    for path in summary_paths:
        if not path.exists():
            continue
        trace = _extract_evaluation_trace(_load_json(path), max_iteration=max_iteration)
        if trace is not None and len(trace["x"]) > 0:
            traces.append(trace)

    if not traces:
        return None

    common_x = sorted(set.intersection(*(set(map(int, trace["x"])) for trace in traces)))
    if not common_x:
        return None

    x_arr = np.asarray(common_x, dtype=int)
    rows = []
    for trace in traces:
        values = {int(x): float(y) for x, y in zip(trace["x"], trace["hv"])}
        rows.append([values[int(x)] for x in x_arr])
    matrix = np.asarray(rows, dtype=float)
    return {
        "x": x_arr,
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1]),
        "n_runs": int(matrix.shape[0]),
    }


def _plot_band(ax: plt.Axes, mode: str, data: Dict[str, Any]) -> None:
    x = np.asarray(data["x"], dtype=float)
    mean = np.asarray(data["mean"], dtype=float)
    std = np.asarray(data["std"], dtype=float)
    color = COLORS[mode]
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0.0)
    ax.plot(
        x,
        mean,
        color=color,
        linestyle=LINESTYLES[mode],
        lw=2.7,
        marker=MARKERS[mode],
        markevery=max(1, len(x) // 8),
        markersize=5.5,
        label=f"{LABELS[mode]}",
    )


def plot(report_path: Path, exp_root: Path, output_dir: Path, modes: List[str], max_iteration: int) -> Dict[str, Any]:
    report = _load_json(report_path)
    paths = _summary_paths_by_mode(report, modes)

    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    plotted: Dict[str, Any] = {}
    y_values: List[np.ndarray] = []

    for mode in modes:
        data = _stack_mode(paths.get(mode, []), max_iteration=max_iteration)
        if data is None:
            print(f"[warn] no plottable data for mode={mode}")
            continue
        _plot_band(ax, mode, data)
        plotted[mode] = {
            "n_runs": int(data["n_runs"]),
            "iterations": [int(x) for x in data["x"]],
            "final_mean": float(data["mean"][-1]),
            "final_std": float(data["std"][-1]),
        }
        y_values.extend([data["mean"] - data["std"], data["mean"] + data["std"]])

    ax.axvline(6.5, color="#666666", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("Simulator evaluations")
    ax.set_ylabel("Scaled hypervolume (sHV)")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if y_values:
        y_all = np.concatenate(y_values)
        y_min = max(0.0, float(np.floor((y_all.min() - 0.01) * 100.0) / 100.0))
        y_max = float(np.ceil((y_all.max() + 0.01) * 100.0) / 100.0)
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#777777")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for ext in ("png", "pdf"):
        path = output_dir / f"normalization_hv_curve.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs[ext] = str(path)
        print(f"[save] {path}")
    plt.close(fig)

    manifest = {
        "report_path": str(report_path),
        "exp_root": str(exp_root),
        "outputs": outputs,
        "modes": modes,
        "metric": "canonical_hv",
        "uncertainty": "mean +/- 1 sample standard deviation across seeds",
        "summary_sha256": {
            mode: {str(path): _sha256(path) for path in paths.get(mode, []) if path.exists()}
            for mode in modes
        },
        "plotted": plotted,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {manifest_path}")
    return manifest


def main() -> int:
    args = parse_args()
    report_path, exp_root = _resolve_report(args)
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = _load_json(report_path)
    report_modes = [canonicalize_objective_preprocess_mode(mode) for mode in (report.get("meta") or {}).get("modes", [])]
    modes = [canonicalize_objective_preprocess_mode(mode) for mode in (args.modes or report_modes or ["minmax", "zscore", "none"])]
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "scalarization_Exp" / "images" / exp_root.name
    plot(report_path, exp_root, output_dir, modes, max_iteration=int(args.max_iteration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
