from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from pymoo.config import Config

    Config.warnings["not_compiled"] = False
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "Compare_Exp").is_dir() and (candidate / "DataBase").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataBase.database import ObservationDB


SEEDS = [8409, 8410, 8411, 8412, 8413]
MAX_EVALS = 56

PAREGO_ROOT = PROJECT_ROOT / "optimized_experiments" / "parego_ecker_5seeds_56evals_2026_05_11"
PAREGO_VARIANT = "parego_matlab_reference_Ecker2015"
PAREGO_REPORT = "report.json"

LLMBO_ROOT = (
    PROJECT_ROOT
    / "scalarization_Exp"
    / "experiment_records"
    / "ecker_llmbo_5seeds_50iter_fixed_2026_05_11"
)
LLMBO_VARIANT = "minmax"
LLMBO_REPORT = "report_5seeds.json"

IMAGE_DIR = PROJECT_ROOT / "Compare_Exp" / "images" / "Ecker2015_HV05-12"
ARCHIVE_DIR = PROJECT_ROOT / "Compare_Exp" / "experiment_records" / "Ecker2015_HV05-12"
CURVE_DIR = ARCHIVE_DIR / "curve_data"

PAREGO_COLOR = "#1f77b4"
LLMBO_COLOR = "#d62728"
BAND_ALPHA = 0.16
ADD_COMMON_ORIGIN = False
COMMON_ORIGIN_HV = 0.0
PAREGO_BAND_SCALE = 1.0
LLMBO_BAND_SCALE = 1.0
Y_AXIS_TICK_SCALE = 0.3


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#666666",
            "grid.color": "#cfcfcf",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
        }
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scaled_tick_label(value: float, _pos: int) -> str:
    scaled = float(value) * Y_AXIS_TICK_SCALE
    return f"{scaled:.2f}".rstrip("0").rstrip(".")


def _validate_sources() -> None:
    required = [
        PAREGO_ROOT / PAREGO_REPORT,
        LLMBO_ROOT / LLMBO_REPORT,
    ]
    for seed in SEEDS:
        required.append(PAREGO_ROOT / f"seed{seed}" / PAREGO_VARIANT / "database.json")
        required.append(LLMBO_ROOT / f"seed{seed}" / LLMBO_VARIANT / "database.json")

    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {_relative(path)}" for path in missing)
        raise FileNotFoundError(f"Missing Ecker2015 experiment data:\n{formatted}")


def _prefix_trace_from_database(database_path: Path, *, max_evals: int = MAX_EVALS) -> Dict[str, np.ndarray]:
    db = ObservationDB.load(str(database_path))
    observations = db.get_all()[: int(max_evals)]
    prefix_db = ObservationDB(
        param_bounds=db.param_bounds,
        ref_point=db.ref_point.copy(),
        ideal_point=db.ideal_point.copy(),
        normalize=db.normalize,
    )

    x_values: List[int] = []
    canonical_values: List[float] = []
    display_values: List[float] = []
    pareto_sizes: List[int] = []

    for eval_index, obs in enumerate(observations, start=1):
        prefix_db.add_observation(
            theta=obs.theta,
            objectives=obs.objectives,
            feasible=obs.feasible,
            violation=obs.violation,
            source=obs.source,
            iteration=eval_index,
            acq_value=obs.acq_value,
            acq_type=obs.acq_type,
            gp_pred=obs.gp_pred,
            llm_rationale=obs.llm_rationale,
            details=obs.details,
        )
        x_values.append(eval_index)
        canonical_values.append(float(prefix_db.compute_hypervolume_canonical()))
        display_values.append(float(prefix_db.compute_hypervolume()))
        pareto_sizes.append(int(prefix_db.pareto_size))

    return {
        "x": np.asarray(x_values, dtype=int),
        "canonical_hv": np.asarray(canonical_values, dtype=float),
        "display_hv": np.asarray(display_values, dtype=float),
        "pareto_size": np.asarray(pareto_sizes, dtype=int),
    }


def _load_seed_trace(root: Path, variant: str, seed: int) -> Dict[str, np.ndarray]:
    return _prefix_trace_from_database(root / f"seed{seed}" / variant / "database.json")


def _stack_algorithm(root: Path, variant: str, label: str) -> Dict[str, Any]:
    traces: List[Dict[str, np.ndarray]] = []
    x_ref: np.ndarray | None = None

    for seed in SEEDS:
        trace = _load_seed_trace(root, variant, seed)
        if x_ref is None:
            x_ref = trace["x"]
        elif trace["x"].shape != x_ref.shape or not np.array_equal(trace["x"], x_ref):
            raise ValueError(f"{label} trace alignment mismatch for seed {seed}")
        traces.append(trace)

    if x_ref is None:
        raise ValueError(f"No traces loaded for {label}")

    canonical = np.vstack([trace["canonical_hv"] for trace in traces])
    display = np.vstack([trace["display_hv"] for trace in traces])
    pareto = np.vstack([trace["pareto_size"] for trace in traces])

    return {
        "label": label,
        "x": x_ref,
        "seed_traces": dict(zip(SEEDS, traces)),
        "canonical_mean": canonical.mean(axis=0),
        "canonical_std": canonical.std(axis=0),
        "display_mean": display.mean(axis=0),
        "display_std": display.std(axis=0),
        "pareto_mean": pareto.mean(axis=0),
        "pareto_std": pareto.std(axis=0),
        "final": {
            "canonical_values": canonical[:, -1],
            "display_values": display[:, -1],
            "pareto_values": pareto[:, -1],
        },
    }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _final_summary(algorithm: Dict[str, Any]) -> Dict[str, Any]:
    final = algorithm["final"]
    return {
        "canonical_hv": _stats(final["canonical_values"]),
        "display_hv": _stats(final["display_values"]),
        "pareto_size": _stats(final["pareto_values"]),
        "per_seed": [
            {
                "seed": int(seed),
                "canonical_hv": float(final["canonical_values"][idx]),
                "display_hv": float(final["display_values"][idx]),
                "pareto_size": int(final["pareto_values"][idx]),
            }
            for idx, seed in enumerate(SEEDS)
        ],
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_seed_curves(algorithm_key: str, algorithm: Dict[str, Any]) -> None:
    out_dir = CURVE_DIR / "per_seed_traces" / algorithm_key
    fields = ["eval_index", "canonical_hv", "display_hv", "pareto_size"]
    for seed, trace in algorithm["seed_traces"].items():
        rows = [
            {
                "eval_index": int(x),
                "canonical_hv": float(canonical),
                "display_hv": float(display),
                "pareto_size": int(pareto),
            }
            for x, canonical, display, pareto in zip(
                trace["x"],
                trace["canonical_hv"],
                trace["display_hv"],
                trace["pareto_size"],
            )
        ]
        _write_csv(out_dir / f"seed{seed}.csv", rows, fields)


def _write_algorithm_curve(filename: str, algorithm: Dict[str, Any]) -> None:
    rows = []
    for idx, x in enumerate(algorithm["x"]):
        mean = float(algorithm["canonical_mean"][idx])
        std = float(algorithm["canonical_std"][idx])
        rows.append(
            {
                "eval_index": int(x),
                "canonical_mean_hv": mean,
                "canonical_std_hv": std,
                "canonical_lower": mean - std,
                "canonical_upper": mean + std,
                "display_mean_hv": float(algorithm["display_mean"][idx]),
                "display_std_hv": float(algorithm["display_std"][idx]),
                "pareto_mean_size": float(algorithm["pareto_mean"][idx]),
                "pareto_std_size": float(algorithm["pareto_std"][idx]),
            }
        )
    _write_csv(
        CURVE_DIR / filename,
        rows,
        [
            "eval_index",
            "canonical_mean_hv",
            "canonical_std_hv",
            "canonical_lower",
            "canonical_upper",
            "display_mean_hv",
            "display_std_hv",
            "pareto_mean_size",
            "pareto_std_size",
        ],
    )


def _plot_series(
    algorithm: Dict[str, Any],
    *,
    band_scale: float,
) -> Dict[str, np.ndarray]:
    x = np.asarray(algorithm["x"], dtype=int)
    mean = np.asarray(algorithm["canonical_mean"], dtype=float)
    raw_std = np.asarray(algorithm["canonical_std"], dtype=float)
    band = raw_std * float(band_scale)

    if ADD_COMMON_ORIGIN and (x.size == 0 or int(x[0]) != 0):
        x = np.concatenate([np.asarray([0], dtype=int), x])
        mean = np.concatenate([np.asarray([COMMON_ORIGIN_HV], dtype=float), mean])
        raw_std = np.concatenate([np.asarray([0.0], dtype=float), raw_std])
        band = np.concatenate([np.asarray([0.0], dtype=float), band])

    return {"x": x, "mean": mean, "raw_std": raw_std, "band": band}


def _write_combined_curves(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> None:
    if not np.array_equal(parego["x"], llmbo["x"]):
        raise ValueError("ParEGO and LLMBO-MO x-axis mismatch")

    parego_plot = _plot_series(parego, band_scale=PAREGO_BAND_SCALE)
    llmbo_plot = _plot_series(llmbo, band_scale=LLMBO_BAND_SCALE)
    if not np.array_equal(parego_plot["x"], llmbo_plot["x"]):
        raise ValueError("ParEGO and LLMBO-MO plotted x-axis mismatch")

    rows = []
    combined_json: Dict[str, Any] = {
        "eval_index": [int(x) for x in parego_plot["x"]],
        "common_origin": {
            "enabled": ADD_COMMON_ORIGIN,
            "eval_index": 0,
            "canonical_hv": COMMON_ORIGIN_HV,
        },
        "ParEGO": {},
        "LLMBO-MO": {},
    }

    for key, algorithm, plotted, band_scale in [
        ("ParEGO", parego, parego_plot, PAREGO_BAND_SCALE),
        ("LLMBO-MO", llmbo, llmbo_plot, LLMBO_BAND_SCALE),
    ]:
        mean = np.asarray(plotted["mean"], dtype=float)
        raw_std = np.asarray(plotted["raw_std"], dtype=float)
        band = np.asarray(plotted["band"], dtype=float)
        combined_json[key] = {
            "type": "multiseed_mean_with_scaled_plot_band_from_database_prefix_hv",
            "band_scale": float(band_scale),
            "canonical_mean_hv": mean.tolist(),
            "canonical_raw_std_hv": raw_std.tolist(),
            "canonical_plot_band": band.tolist(),
            "canonical_lower": (mean - band).tolist(),
            "canonical_upper": (mean + band).tolist(),
            "display_mean_hv": np.asarray(algorithm["display_mean"], dtype=float).tolist(),
            "display_std_hv": np.asarray(algorithm["display_std"], dtype=float).tolist(),
            "pareto_mean_size": np.asarray(algorithm["pareto_mean"], dtype=float).tolist(),
            "pareto_std_size": np.asarray(algorithm["pareto_std"], dtype=float).tolist(),
        }

    for idx, x in enumerate(parego_plot["x"]):
        parego_mean = float(parego_plot["mean"][idx])
        parego_raw_std = float(parego_plot["raw_std"][idx])
        parego_band = float(parego_plot["band"][idx])
        llmbo_mean = float(llmbo_plot["mean"][idx])
        llmbo_raw_std = float(llmbo_plot["raw_std"][idx])
        llmbo_band = float(llmbo_plot["band"][idx])
        rows.append(
            {
                "eval_index": int(x),
                "parego_mean_hv": parego_mean,
                "parego_raw_std_hv": parego_raw_std,
                "parego_plot_band": parego_band,
                "parego_lower": parego_mean - parego_band,
                "parego_upper": parego_mean + parego_band,
                "llmbo_mo_mean_hv": llmbo_mean,
                "llmbo_mo_raw_std_hv": llmbo_raw_std,
                "llmbo_mo_plot_band": llmbo_band,
                "llmbo_mo_lower": llmbo_mean - llmbo_band,
                "llmbo_mo_upper": llmbo_mean + llmbo_band,
            }
        )

    _write_csv(
        CURVE_DIR / "hv_convergence_parego_vs_llmbo.csv",
        rows,
        [
            "eval_index",
            "parego_mean_hv",
            "parego_raw_std_hv",
            "parego_plot_band",
            "parego_lower",
            "parego_upper",
            "llmbo_mo_mean_hv",
            "llmbo_mo_raw_std_hv",
            "llmbo_mo_plot_band",
            "llmbo_mo_lower",
            "llmbo_mo_upper",
        ],
    )
    (CURVE_DIR / "hv_convergence_parego_vs_llmbo.json").write_text(
        json.dumps(combined_json, indent=2),
        encoding="utf-8",
    )


def _write_final_summary(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "seeds": SEEDS,
        "max_evals": MAX_EVALS,
        "ParEGO": _final_summary(parego),
        "LLMBO-MO": _final_summary(llmbo),
    }
    (CURVE_DIR / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for algorithm_key in ["ParEGO", "LLMBO-MO"]:
        for record in summary[algorithm_key]["per_seed"]:
            rows.append({"algorithm": algorithm_key, **record})
    _write_csv(
        CURVE_DIR / "final_summary.csv",
        rows,
        ["algorithm", "seed", "canonical_hv", "display_hv", "pareto_size"],
    )
    return summary


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _archive_raw_sources(root: Path, variant: str, report_name: str, archive_name: str) -> Dict[str, Any]:
    dest_root = ARCHIVE_DIR / "raw_sources" / archive_name
    _copy_if_exists(root / report_name, dest_root / report_name)

    seed_sources = []
    for seed in SEEDS:
        src_dir = root / f"seed{seed}" / variant
        dst_dir = dest_root / f"seed{seed}" / variant
        copied = []
        for filename in ["summary.json", "database.json", "db_final.json", "pareto_front.json"]:
            src = src_dir / filename
            if src.exists():
                _copy_if_exists(src, dst_dir / filename)
                copied.append(_relative(src))
        seed_sources.append({"seed": seed, "source_dir": _relative(src_dir), "copied_files": copied})

    return {
        "source_root": _relative(root),
        "variant_dir": variant,
        "report": _relative(root / report_name),
        "archived_raw_source": _relative(dest_root),
        "seed_sources": seed_sources,
    }


def _write_manifest(
    parego: Dict[str, Any],
    llmbo: Dict[str, Any],
    final_summary: Dict[str, Any],
    parego_sources: Dict[str, Any],
    llmbo_sources: Dict[str, Any],
) -> None:
    manifest = {
        "archive_name": ARCHIVE_DIR.name,
        "created_for": _relative(IMAGE_DIR / "hv_convergence_parego_vs_llmbo.png"),
        "generation_script": _relative(Path(__file__)),
        "seeds": SEEDS,
        "max_evals": MAX_EVALS,
        "curve_method": "Canonical HV traces recomputed from each seed database.json by replaying prefix observations.",
        "plot_band_notes": {
            "ParEGO": f"Centerline is mean across 5 seeds; plotted band is +/- {PAREGO_BAND_SCALE:.2f}x raw standard deviation.",
            "LLMBO-MO": f"Centerline is mean across 5 seeds; plotted band is +/- {LLMBO_BAND_SCALE:.2f}x raw standard deviation.",
            "common_origin": "Disabled; plotted curves start from the first actual evaluation point.",
            "y_axis": f"Axis label is HV; tick labels are displayed as plotted canonical values multiplied by {Y_AXIS_TICK_SCALE:.1f}.",
        },
        "lines": {
            "ParEGO": {
                "label_in_plot": "ParEGO",
                "color": PAREGO_COLOR,
                **parego_sources,
                "curve_csv": "curve_data/parego_curve.csv",
                "per_seed_csv_dir": "curve_data/per_seed_traces/ParEGO",
                "plot_band_scale": PAREGO_BAND_SCALE,
                "final_mean_hv": final_summary["ParEGO"]["canonical_hv"]["mean"],
                "final_std_hv": final_summary["ParEGO"]["canonical_hv"]["std"],
            },
            "LLMBO-MO": {
                "label_in_plot": "LLMBO-MO",
                "color": LLMBO_COLOR,
                **llmbo_sources,
                "curve_csv": "curve_data/llmbo_mo_curve.csv",
                "per_seed_csv_dir": "curve_data/per_seed_traces/LLMBO-MO",
                "plot_band_scale": LLMBO_BAND_SCALE,
                "final_mean_hv": final_summary["LLMBO-MO"]["canonical_hv"]["mean"],
                "final_std_hv": final_summary["LLMBO-MO"]["canonical_hv"]["std"],
            },
        },
        "derived_files": {
            "figure_png": _relative(IMAGE_DIR / "hv_convergence_parego_vs_llmbo.png"),
            "figure_pdf": _relative(IMAGE_DIR / "hv_convergence_parego_vs_llmbo.pdf"),
            "combined_csv": "curve_data/hv_convergence_parego_vs_llmbo.csv",
            "combined_json": "curve_data/hv_convergence_parego_vs_llmbo.json",
            "final_summary_csv": "curve_data/final_summary.csv",
            "final_summary_json": "curve_data/final_summary.json",
        },
    }
    (ARCHIVE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_readme(final_summary: Dict[str, Any]) -> None:
    parego = final_summary["ParEGO"]["canonical_hv"]
    llmbo = final_summary["LLMBO-MO"]["canonical_hv"]
    content = f"""# Ecker2015 HV05-12

This archive backs `Compare_Exp/images/Ecker2015_HV05-12/hv_convergence_parego_vs_llmbo.png`.

Curves are recomputed from per-seed `database.json` files by replaying prefix observations, then plotted as mean canonical HV with +/- 1 std shaded bands across seeds {SEEDS}. The y-axis label is `HV`, and tick labels are shown as canonical values multiplied by {Y_AXIS_TICK_SCALE:.1f}.

- ParEGO final canonical HV: {parego["mean"]:.6f} +/- {parego["std"]:.6f}
- LLMBO-MO final canonical HV: {llmbo["mean"]:.6f} +/- {llmbo["std"]:.6f}
"""
    (ARCHIVE_DIR / "README.md").write_text(content, encoding="utf-8")


def _archive_outputs(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> Dict[str, Any]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    _write_per_seed_curves("ParEGO", parego)
    _write_per_seed_curves("LLMBO-MO", llmbo)
    _write_algorithm_curve("parego_curve.csv", parego)
    _write_algorithm_curve("llmbo_mo_curve.csv", llmbo)
    _write_combined_curves(parego, llmbo)
    final_summary = _write_final_summary(parego, llmbo)

    parego_sources = _archive_raw_sources(PAREGO_ROOT, PAREGO_VARIANT, PAREGO_REPORT, "ParEGO")
    llmbo_sources = _archive_raw_sources(LLMBO_ROOT, LLMBO_VARIANT, LLMBO_REPORT, "LLMBO-MO")

    code_dir = ARCHIVE_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve()
    script_dst = (code_dir / Path(__file__).name).resolve()
    if script_src != script_dst:
        shutil.copy2(script_src, script_dst)

    _write_manifest(parego, llmbo, final_summary, parego_sources, llmbo_sources)
    _write_readme(final_summary)
    return final_summary


def _plot_line_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    band: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str,
) -> None:
    lower = mean - band
    upper = mean + band
    ax.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0.0)
    ax.plot(
        x,
        mean,
        color=color,
        lw=2.8,
        solid_capstyle="round",
        label=label,
        marker=marker,
        markevery=7,
        markersize=6,
    )


def _plot_hv_convergence(parego: Dict[str, Any], llmbo: Dict[str, Any]) -> None:
    _configure_plot_style()

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    parego_plot = _plot_series(parego, band_scale=PAREGO_BAND_SCALE)
    llmbo_plot = _plot_series(llmbo, band_scale=LLMBO_BAND_SCALE)

    _plot_line_with_band(
        ax,
        parego_plot["x"],
        parego_plot["mean"],
        parego_plot["band"],
        color=PAREGO_COLOR,
        label="ParEGO",
        marker="s",
    )
    _plot_line_with_band(
        ax,
        llmbo_plot["x"],
        llmbo_plot["mean"],
        llmbo_plot["band"],
        color=LLMBO_COLOR,
        label="LLMBO-MO",
        marker="o",
    )

    all_y = np.concatenate(
        [
            parego_plot["mean"] - parego_plot["band"],
            parego_plot["mean"] + parego_plot["band"],
            llmbo_plot["mean"] - llmbo_plot["band"],
            llmbo_plot["mean"] + llmbo_plot["band"],
        ]
    )
    y_min = max(0.0, float(np.nanmin(all_y)) - 0.06)
    y_max = float(np.nanmax(all_y)) + 0.06

    ax.set_xlim(0, MAX_EVALS)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("HV")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(_scaled_tick_label))
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        handlelength=2.6,
        handletextpad=0.6,
    )
    fig.tight_layout()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    png = IMAGE_DIR / "hv_convergence_parego_vs_llmbo.png"
    pdf = IMAGE_DIR / "hv_convergence_parego_vs_llmbo.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, dpi=240, bbox_inches="tight")

    legacy_png = IMAGE_DIR / "ecker2015_hv_convergence_parego_vs_llmbo.png"
    legacy_pdf = IMAGE_DIR / "ecker2015_hv_convergence_parego_vs_llmbo.pdf"
    fig.savefig(legacy_png, dpi=240, bbox_inches="tight")
    fig.savefig(legacy_pdf, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {legacy_png}")
    print(f"Saved: {legacy_pdf}")


def _print_summary(final_summary: Dict[str, Any]) -> None:
    print("\nFinal HV summary")
    print("=" * 72)
    for algorithm_key in ["ParEGO", "LLMBO-MO"]:
        stats = final_summary[algorithm_key]
        canonical = stats["canonical_hv"]
        display = stats["display_hv"]
        pareto = stats["pareto_size"]
        print(f"\n{algorithm_key}")
        print(
            f"  Canonical HV: {canonical['mean']:.6f} +/- {canonical['std']:.6f} "
            f"(min={canonical['min']:.6f}, max={canonical['max']:.6f})"
        )
        print(
            f"  Display HV:   {display['mean']:.6f} +/- {display['std']:.6f} "
            f"(min={display['min']:.6f}, max={display['max']:.6f})"
        )
        print(f"  Pareto size:  {pareto['mean']:.2f} +/- {pareto['std']:.2f}")
        for record in stats["per_seed"]:
            print(
                f"    seed {record['seed']}: canonical={record['canonical_hv']:.6f}, "
                f"display={record['display_hv']:.6f}, pareto={record['pareto_size']}"
            )


def main() -> None:
    _validate_sources()
    parego = _stack_algorithm(PAREGO_ROOT, PAREGO_VARIANT, "ParEGO")
    llmbo = _stack_algorithm(LLMBO_ROOT, LLMBO_VARIANT, "LLMBO-MO")
    _plot_hv_convergence(parego, llmbo)
    final_summary = _archive_outputs(parego, llmbo)
    _print_summary(final_summary)
    print(f"\nArchived data: {ARCHIVE_DIR}")


if __name__ == "__main__":
    main()
