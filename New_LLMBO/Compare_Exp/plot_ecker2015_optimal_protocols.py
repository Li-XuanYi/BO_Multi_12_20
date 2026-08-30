from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "Compare_Exp").is_dir() and (candidate / "DataBase").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Compare_Exp.plot_ecker2015_hv_convergence import (  # noqa: E402
    ARCHIVE_DIR,
    BAND_ALPHA,
    IMAGE_DIR,
    LLMBO_COLOR,
    LLMBO_ROOT,
    LLMBO_VARIANT,
    MAX_EVALS,
    PAREGO_COLOR,
    PAREGO_ROOT,
    PAREGO_VARIANT,
    SEEDS,
    _configure_plot_style,
    _relative,
    _stack_algorithm,
    _validate_sources,
)


CURVE_DIR = ARCHIVE_DIR / "curve_data"
OUTPUT_STEM = "optimal_protocols_parego_vs_llmbo"
ADD_COMMON_ORIGIN = True


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _pareto_plot_series(algorithm: Dict[str, Any]) -> Dict[str, np.ndarray]:
    x = np.asarray(algorithm["x"], dtype=int)
    mean = np.asarray(algorithm["pareto_mean"], dtype=float)
    std = np.asarray(algorithm["pareto_std"], dtype=float)

    if ADD_COMMON_ORIGIN and (x.size == 0 or int(x[0]) != 0):
        x = np.concatenate([np.asarray([0], dtype=int), x])
        mean = np.concatenate([np.asarray([0.0], dtype=float), mean])
        std = np.concatenate([np.asarray([0.0], dtype=float), std])

    return {"x": x, "mean": mean, "std": std}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_curve_data(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> Dict[str, Any]:
    parego_plot = _pareto_plot_series(parego)
    llmbo_plot = _pareto_plot_series(llmbo)

    if not np.array_equal(parego_plot["x"], llmbo_plot["x"]):
        raise ValueError("ParEGO and LLMBO-MO x-axis mismatch")

    rows = []
    for idx, x in enumerate(parego_plot["x"]):
        parego_mean = float(parego_plot["mean"][idx])
        parego_std = float(parego_plot["std"][idx])
        llmbo_mean = float(llmbo_plot["mean"][idx])
        llmbo_std = float(llmbo_plot["std"][idx])
        rows.append(
            {
                "eval_index": int(x),
                "parego_mean_optimal_protocols": parego_mean,
                "parego_std_optimal_protocols": parego_std,
                "parego_lower": parego_mean - parego_std,
                "parego_upper": parego_mean + parego_std,
                "llmbo_mo_mean_optimal_protocols": llmbo_mean,
                "llmbo_mo_std_optimal_protocols": llmbo_std,
                "llmbo_mo_lower": llmbo_mean - llmbo_std,
                "llmbo_mo_upper": llmbo_mean + llmbo_std,
            }
        )

    csv_path = CURVE_DIR / f"{OUTPUT_STEM}.csv"
    json_path = CURVE_DIR / f"{OUTPUT_STEM}.json"
    _write_csv(
        csv_path,
        rows,
        [
            "eval_index",
            "parego_mean_optimal_protocols",
            "parego_std_optimal_protocols",
            "parego_lower",
            "parego_upper",
            "llmbo_mo_mean_optimal_protocols",
            "llmbo_mo_std_optimal_protocols",
            "llmbo_mo_lower",
            "llmbo_mo_upper",
        ],
    )

    payload = {
        "eval_index": [int(x) for x in parego_plot["x"]],
        "common_origin": {
            "enabled": ADD_COMMON_ORIGIN,
            "eval_index": 0,
            "optimal_protocols": 0,
        },
        "ParEGO": {
            "mean_optimal_protocols": parego_plot["mean"].tolist(),
            "std_optimal_protocols": parego_plot["std"].tolist(),
            "lower": (parego_plot["mean"] - parego_plot["std"]).tolist(),
            "upper": (parego_plot["mean"] + parego_plot["std"]).tolist(),
        },
        "LLMBO-MO": {
            "mean_optimal_protocols": llmbo_plot["mean"].tolist(),
            "std_optimal_protocols": llmbo_plot["std"].tolist(),
            "lower": (llmbo_plot["mean"] - llmbo_plot["std"]).tolist(),
            "upper": (llmbo_plot["mean"] + llmbo_plot["std"]).tolist(),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def _plot_line_with_band(
    ax: plt.Axes,
    series: Dict[str, np.ndarray],
    *,
    color: str,
    label: str,
) -> None:
    x = np.asarray(series["x"], dtype=float)
    mean = np.asarray(series["mean"], dtype=float)
    std = np.asarray(series["std"], dtype=float)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=BAND_ALPHA, linewidth=0.0)
    ax.plot(x, mean, color=color, lw=2.4, solid_capstyle="round", label=label)


def _plot_optimal_protocols(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> Dict[str, Path]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))

    parego_plot = _pareto_plot_series(parego)
    llmbo_plot = _pareto_plot_series(llmbo)

    _plot_line_with_band(ax, parego_plot, color=PAREGO_COLOR, label="ParEGO")
    _plot_line_with_band(ax, llmbo_plot, color=LLMBO_COLOR, label="LLMBO-MO")

    all_y = np.concatenate(
        [
            parego_plot["mean"] - parego_plot["std"],
            parego_plot["mean"] + parego_plot["std"],
            llmbo_plot["mean"] - llmbo_plot["std"],
            llmbo_plot["mean"] + llmbo_plot["std"],
        ]
    )
    y_min = 0.0
    y_max = float(np.ceil((np.nanmax(all_y) + 2.0) / 5.0) * 5.0)
    if y_max <= y_min:
        y_max = 5.0

    ax.set_xlim(0, MAX_EVALS)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Number of optimal charging protocols")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        handlelength=2.6,
        handletextpad=0.6,
    )
    ax.text(0.5, -0.18, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=18)
    fig.tight_layout(pad=0.6)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    png = IMAGE_DIR / f"{OUTPUT_STEM}.png"
    pdf = IMAGE_DIR / f"{OUTPUT_STEM}.pdf"
    legacy_png = IMAGE_DIR / "ecker2015_optimal_protocols_parego_vs_llmbo.png"
    legacy_pdf = IMAGE_DIR / "ecker2015_optimal_protocols_parego_vs_llmbo.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(legacy_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(legacy_pdf, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    return {"png": png, "pdf": pdf, "legacy_png": legacy_png, "legacy_pdf": legacy_pdf}


def _write_manifest(
    parego: Dict[str, Any],
    llmbo: Dict[str, Any],
    figure_paths: Dict[str, Path],
    curve_paths: Dict[str, Path],
) -> Path:
    manifest = {
        "figure_family": "ecker2015_optimal_protocols_parego_vs_llmbo",
        "generation_script": _relative(Path(__file__)),
        "seeds": SEEDS,
        "max_evals": MAX_EVALS,
        "metric": "Number of optimal charging protocols, computed as ObservationDB.pareto_size after replaying each database prefix.",
        "notes": [
            "Centerline is the five-seed mean at each evaluation.",
            "Shaded band is +/- one across-seed standard deviation.",
            "A common visual origin at evaluation 0 and count 0 is prepended to match the Ecker-style curve layout.",
        ],
        "lines": {
            "ParEGO": {
                "color": PAREGO_COLOR,
                "source_root": _relative(PAREGO_ROOT),
                "variant_dir": PAREGO_VARIANT,
                "final_optimal_protocols": _stats(parego["final"]["pareto_values"]),
            },
            "LLMBO-MO": {
                "color": LLMBO_COLOR,
                "source_root": _relative(LLMBO_ROOT),
                "variant_dir": LLMBO_VARIANT,
                "final_optimal_protocols": _stats(llmbo["final"]["pareto_values"]),
            },
        },
        "artifacts": {
            key: _relative(path)
            for key, path in {
                "figure_png": figure_paths["png"],
                "figure_pdf": figure_paths["pdf"],
                "legacy_figure_png": figure_paths["legacy_png"],
                "legacy_figure_pdf": figure_paths["legacy_pdf"],
                "curve_csv": curve_paths["csv"],
                "curve_json": curve_paths["json"],
            }.items()
        },
    }

    manifest_path = ARCHIVE_DIR / f"{OUTPUT_STEM}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    code_dir = ARCHIVE_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve()
    script_dst = (code_dir / Path(__file__).name).resolve()
    if script_src != script_dst:
        shutil.copy2(script_src, script_dst)

    return manifest_path


def _print_summary(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> None:
    print("\nFinal optimal-protocol summary")
    print("=" * 72)
    for key, algorithm in [("ParEGO", parego), ("LLMBO-MO", llmbo)]:
        stats = _stats(algorithm["final"]["pareto_values"])
        print(
            f"{key}: {stats['mean']:.2f} +/- {stats['std']:.2f} "
            f"(min={stats['min']:.0f}, max={stats['max']:.0f})"
        )


def main() -> None:
    _validate_sources()
    parego = _stack_algorithm(PAREGO_ROOT, PAREGO_VARIANT, "ParEGO")
    llmbo = _stack_algorithm(LLMBO_ROOT, LLMBO_VARIANT, "LLMBO-MO")

    curve_paths = _write_curve_data(parego, llmbo)
    figure_paths = _plot_optimal_protocols(parego, llmbo)
    manifest_path = _write_manifest(parego, llmbo, figure_paths, curve_paths)

    for path in figure_paths.values():
        print(f"Saved: {path}")
    print(f"Saved: {curve_paths['csv']}")
    print(f"Saved: {curve_paths['json']}")
    print(f"Saved: {manifest_path}")
    _print_summary(parego, llmbo)


if __name__ == "__main__":
    main()
