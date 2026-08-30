"""Build the paper's controlled hyperparameter and preprocessing sensitivity figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper"
MODE_ORDER = ("minmax", "zscore", "none")
MODE_LABELS = {
    "minmax": "Dynamic min-max",
    "zscore": "Z-score scaling",
    "none": "No objective scaling",
}
MODE_COLORS = {"minmax": "#0072B2", "zscore": "#E69F00", "none": "#009E73"}
MODE_STYLES = {"minmax": "-", "zscore": "--", "none": "-."}
MODE_MARKERS = {"minmax": "o", "zscore": "s", "none": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-report", type=Path, required=True)
    parser.add_argument("--normalization-report", type=Path, required=True)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=PAPER_ROOT / "Section" / "figures" / "sensitivity_hv",
    )
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_record_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _budget_points(report: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float, float]] = []
    for item in (report.get("settings") or {}).values():
        setting = item.get("setting") or {}
        if "shift_budget" not in (setting.get("panels") or []):
            continue
        if str(setting.get("normalization")) != "minmax":
            continue
        stats = item.get("final_sHV") or {}
        if stats.get("mean") is None:
            continue
        rows.append(
            (
                float(setting["shift_budget"]),
                float(stats["mean"]),
                float(stats.get("sample_std", 0.0)),
            )
        )
    if not rows:
        raise ValueError("No completed shift-budget settings found")
    rows.sort(key=lambda row: row[0])
    return tuple(np.asarray(values, dtype=float) for values in zip(*rows))


def _normalization_paths(report: Mapping[str, Any]) -> Dict[str, List[Path]]:
    result = {mode: [] for mode in MODE_ORDER}
    for record in report.get("records") or []:
        mode = str(record.get("mode"))
        if record.get("status") == "ok" and mode in result:
            result[mode].append(_resolve_record_path(str(record["summary_path"])))
    return result


def _trace(summary_path: Path) -> Dict[int, float]:
    summary = _load(summary_path)
    return {
        int(item.get("eval_index", item.get("n_total", 0))): float(
            item.get("canonical_hv", item.get("hypervolume_canonical", 0.0))
        )
        for item in (summary.get("hv_trace") or [])
        if int(item.get("eval_index", item.get("n_total", 0))) > 0
    }


def _normalization_curve(paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    traces = [_trace(path) for path in paths]
    if not traces:
        raise ValueError("No normalization traces found")
    common = sorted(set.intersection(*(set(trace) for trace in traces)))
    matrix = np.asarray([[trace[index] for index in common] for trace in traces], dtype=float)
    std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
    return np.asarray(common, dtype=int), matrix.mean(axis=0), std


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#555555",
            "grid.color": "#cccccc",
            "grid.alpha": 0.45,
        }
    )


def make_figure(budget_report_path: Path, normalization_report_path: Path, output_base: Path) -> None:
    budget_report = _load(budget_report_path)
    normalization_report = _load(normalization_report_path)
    budget_x, budget_mean, budget_std = _budget_points(budget_report)
    normalization_paths = _normalization_paths(normalization_report)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.38))

    categorical_x = np.arange(len(budget_x))
    axes[0].errorbar(
        categorical_x,
        budget_mean,
        yerr=budget_std,
        color="#0072B2",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        capsize=3,
    )
    axes[0].set_xticks(categorical_x, [f"{value:g}" for value in budget_x])
    axes[0].set_xlabel(r"Posterior-mean budget $B_\mu$")
    axes[0].set_ylabel("Final sHV")
    axes[0].grid(True)

    curve_manifest: Dict[str, Any] = {}
    for mode in MODE_ORDER:
        x, mean, std = _normalization_curve(normalization_paths[mode])
        color = MODE_COLORS[mode]
        axes[1].fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        axes[1].plot(
            x,
            mean,
            color=color,
            linestyle=MODE_STYLES[mode],
            marker=MODE_MARKERS[mode],
            markevery=max(1, len(x) // 7),
            markersize=3.0,
            linewidth=1.8,
            label=MODE_LABELS[mode],
        )
        curve_manifest[mode] = {
            "n_runs": len(normalization_paths[mode]),
            "final_mean": float(mean[-1]),
            "final_sample_std": float(std[-1]),
        }
    axes[1].axvline(6.5, color="#666666", linestyle="--", linewidth=0.9, alpha=0.75)
    axes[1].set_xlim(1, max(x))
    axes[1].set_xlabel("Simulator evaluations")
    axes[1].set_ylabel("sHV")
    axes[1].grid(True)
    axes[1].legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#777777")

    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(0.5, -0.30, label, transform=axis.transAxes, ha="center", va="top")

    fig.subplots_adjust(left=0.085, right=0.99, top=0.98, bottom=0.31, wspace=0.30)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "png"):
        path = output_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=400, bbox_inches="tight")
        outputs[suffix] = str(path)
    plt.close(fig)

    # Keep the parameter study as a separate single-column figure so the
    # objective-normalization convergence curve can be discussed independently.
    budget_base = output_base.with_name("shift_budget_hv")
    budget_fig, budget_ax = plt.subplots(figsize=(3.45, 2.35))
    budget_ax.errorbar(
        categorical_x,
        budget_mean,
        yerr=budget_std,
        color="#0072B2",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        capsize=3,
    )
    budget_ax.set_xticks(categorical_x, [f"{value:g}" for value in budget_x])
    budget_ax.set_xlabel(r"Posterior-mean budget $B_\mu$")
    budget_ax.set_ylabel("Final sHV")
    budget_ax.grid(True)
    budget_fig.tight_layout(pad=0.45)
    budget_outputs = {}
    for suffix in ("pdf", "png"):
        path = budget_base.with_suffix(f".{suffix}")
        budget_fig.savefig(path, dpi=400, bbox_inches="tight")
        budget_outputs[suffix] = str(path)
    plt.close(budget_fig)

    manifest = {
        "budget_report": str(budget_report_path),
        "budget_report_sha256": _sha256(budget_report_path),
        "normalization_report": str(normalization_report_path),
        "normalization_report_sha256": _sha256(normalization_report_path),
        "uncertainty": "mean +/- 1 sample standard deviation across five seeds",
        "budget": {
            "values": budget_x.tolist(),
            "mean": budget_mean.tolist(),
            "sample_std": budget_std.tolist(),
        },
        "normalization": curve_manifest,
        "summary_sha256": {
            mode: {str(path): _sha256(path) for path in normalization_paths[mode]}
            for mode in MODE_ORDER
        },
        "outputs": outputs,
        "budget_outputs": budget_outputs,
    }
    output_base.with_name(f"{output_base.name}_manifest").with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    make_figure(args.budget_report, args.normalization_report, args.output_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
