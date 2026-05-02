from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pybamm_simulator import PyBaMMSimulator


WINNER_SUMMARY = PROJECT_ROOT / "optimized_experiments" / "region_lift_force_pool_local_sweep_seed8409_2026_05_01" / "seed8409" / "wider_active16_ext32" / "summary.json"
WINNER_DB = PROJECT_ROOT / "optimized_experiments" / "region_lift_force_pool_local_sweep_seed8409_2026_05_01" / "seed8409" / "wider_active16_ext32" / "database.json"
WINNER_DB_FINAL = PROJECT_ROOT / "optimized_experiments" / "region_lift_force_pool_local_sweep_seed8409_2026_05_01" / "seed8409" / "wider_active16_ext32" / "db_final.json"
WINNER_PARETO = PROJECT_ROOT / "optimized_experiments" / "region_lift_force_pool_local_sweep_seed8409_2026_05_01" / "seed8409" / "wider_active16_ext32" / "pareto_front.json"

PLAIN_SUMMARY = PROJECT_ROOT / "optimized_experiments" / "region_lift_fix_seed8409_50iter_2026_05_01" / "seed8409" / "warmstart_plain_ei" / "summary.json"
STRICT_SUMMARY = PROJECT_ROOT / "optimized_experiments" / "region_lift_fix_seed8409_50iter_2026_05_01" / "seed8409" / "strict_baseline" / "summary.json"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fixed_experiments" / "fixed_seed8409_llmgp_winner_2026_05_01"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_final_theta(summary: Dict[str, Any]) -> np.ndarray:
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        raise ValueError("summary.json missing hv_trace")
    theta = hv_trace[-1].get("theta")
    if theta is None:
        raise ValueError("last hv_trace item missing theta")
    return np.asarray(theta, dtype=float)


def _extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "canonical_hv": float(summary.get("hypervolume_canonical", 0.0)),
        "display_hv": float(summary.get("display_hv", summary.get("hypervolume", 0.0))),
        "raw_hv": float(summary.get("hypervolume_raw", 0.0)),
        "pareto_size": int(summary.get("pareto_size", 0)),
        "n_total": int(summary.get("n_total", 0)),
        "n_feasible": int(summary.get("n_feasible", 0)),
    }


def _simulate_profile(
    simulator: PyBaMMSimulator,
    theta: np.ndarray,
    loss_simulator: Optional[PyBaMMSimulator] = None,
) -> Dict[str, Any]:
    result = simulator.evaluate(theta)
    if not bool(result.get("feasible", False)):
        raise RuntimeError(f"Simulation failed for theta={theta.tolist()}: {result.get('violation')}")
    traj = result["trajectories"]
    profile = {
        "theta": np.asarray(theta, dtype=float).round(12).tolist(),
        "objectives": np.asarray(result["raw_objectives"], dtype=float).round(12).tolist(),
        "soc_final": float(result.get("soc_final", 0.0)),
        "time_s": np.asarray(traj["time"], dtype=float).tolist(),
        "voltage_v": np.asarray(traj["V"], dtype=float).tolist(),
        "temperature_k": np.asarray(traj["T"], dtype=float).tolist(),
        "soc_frac": np.asarray(traj["SOC"], dtype=float).tolist(),
        "current_a": np.asarray(traj["I"], dtype=float).tolist(),
    }
    if loss_simulator is not None:
        loss_result = loss_simulator.evaluate(theta)
        if not bool(loss_result.get("feasible", False)):
            raise RuntimeError(f"Physical-loss simulation failed for theta={theta.tolist()}: {loss_result.get('violation')}")
        loss_traj = loss_result["trajectories"]
        profile["loss_time_s"] = np.asarray(loss_traj["time"], dtype=float).tolist()
        profile["loss_pct"] = np.asarray(loss_traj.get("loss_pct", []), dtype=float).tolist()
        profile["aging_physical_pct_final"] = float(loss_result.get("aging_physical", 0.0) or 0.0)
    return profile


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_profiles(
    profiles: List[Tuple[str, Dict[str, Any], str]],
    output_dir: Path,
    lower_right_key: str,
    lower_right_label: str,
    filename_stem: str,
    lower_right_scale: float = 1.0,
    lower_right_time_key: str = "time_s",
) -> Tuple[Path, Path]:
    _configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    axes = axes.ravel()

    voltage_ax, temp_ax, current_ax, lower_right_ax = axes
    vmax_line = 4.3

    for label, profile, color in profiles:
        t = np.asarray(profile["time_s"], dtype=float)
        voltage_ax.plot(t, np.asarray(profile["voltage_v"], dtype=float), lw=2.0, color=color, label=label)
        temp_ax.plot(t, np.asarray(profile["temperature_k"], dtype=float), lw=2.0, color=color, label=label)
        current_ax.plot(t, np.asarray(profile["current_a"], dtype=float), lw=2.0, color=color, label=label)
        lower_t = np.asarray(profile.get(lower_right_time_key, profile["time_s"]), dtype=float)
        lower_y = np.asarray(profile[lower_right_key], dtype=float) * lower_right_scale
        lower_right_ax.plot(lower_t, lower_y, lw=2.0, color=color, label=label)

    voltage_ax.axhline(vmax_line, color="#d62728", ls="--", lw=1.2, alpha=0.75)
    voltage_ax.set_ylabel("Voltage/V")
    temp_ax.set_ylabel("Temperature/K")
    current_ax.set_ylabel("Input Current/A")
    lower_right_ax.set_ylabel(lower_right_label)

    temp_values = np.concatenate([np.asarray(profile["temperature_k"], dtype=float) for _, profile, _ in profiles])
    temp_bottom = 2.0 * np.floor(np.min(temp_values) / 2.0)
    temp_top = 2.0 * np.ceil((np.max(temp_values) + 0.6) / 2.0)
    if temp_top <= temp_bottom:
        temp_top = temp_bottom + 2.0
    temp_ax.set_ylim(temp_bottom, temp_top)
    temp_ax.yaxis.set_major_locator(MultipleLocator(2.0))

    if lower_right_key == "loss_pct":
        loss_values = np.concatenate([np.asarray(profile["loss_pct"], dtype=float) for _, profile, _ in profiles])
        loss_top = max(0.03, 0.01 * np.ceil((np.max(loss_values) + 0.001) / 0.01))
        lower_right_ax.set_ylim(0.0, loss_top)
        lower_right_ax.yaxis.set_major_locator(MultipleLocator(0.01))
        lower_right_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for ax in axes:
        ax.set_xlabel("Time/s")
        ax.grid(True, alpha=0.22)

    voltage_ax.legend(frameon=True, loc="lower right")

    voltage_ax.text(0.5, -0.22, "(a)", transform=voltage_ax.transAxes, ha="center", va="top", fontsize=16)
    temp_ax.text(0.5, -0.22, "(b)", transform=temp_ax.transAxes, ha="center", va="top", fontsize=16)
    current_ax.text(0.5, -0.22, "(c)", transform=current_ax.transAxes, ha="center", va="top", fontsize=16)
    lower_right_ax.text(0.5, -0.22, "(d)", transform=lower_right_ax.transAxes, ha="center", va="top", fontsize=16)

    fig.tight_layout()
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _copy_sources(output_dir: Path) -> None:
    snapshot_dir = output_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copy_pairs = [
        (WINNER_SUMMARY, snapshot_dir / "winner_summary.json"),
        (WINNER_DB, snapshot_dir / "winner_database.json"),
        (WINNER_DB_FINAL, snapshot_dir / "winner_db_final.json"),
        (WINNER_PARETO, snapshot_dir / "winner_pareto_front.json"),
        (PLAIN_SUMMARY, snapshot_dir / "warmstart_summary.json"),
        (STRICT_SUMMARY, snapshot_dir / "baseline_summary.json"),
    ]
    for src, dst in copy_pairs:
        shutil.copy2(src, dst)


def build_fixed_seed8409_bundle(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_sources(output_dir)

    winner_summary = _load_json(WINNER_SUMMARY)
    plain_summary = _load_json(PLAIN_SUMMARY)
    strict_summary = _load_json(STRICT_SUMMARY)

    winner_theta = _extract_final_theta(winner_summary)
    plain_theta = _extract_final_theta(plain_summary)
    strict_theta = _extract_final_theta(strict_summary)

    simulator = PyBaMMSimulator()
    loss_simulator = PyBaMMSimulator(aging_mode="both")
    winner_profile = _simulate_profile(simulator, winner_theta, loss_simulator=loss_simulator)
    plain_profile = _simulate_profile(simulator, plain_theta, loss_simulator=loss_simulator)
    strict_profile = _simulate_profile(simulator, strict_theta, loss_simulator=loss_simulator)

    trajectories_payload = {
        "winner": winner_profile,
        "warmstart_plain_ei": plain_profile,
        "strict_baseline": strict_profile,
    }
    trajectories_path = output_dir / "seed8409_profiles.json"
    trajectories_path.write_text(json.dumps(trajectories_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    profiles = [
        ("LLMGP Tuned", winner_profile, "#2ca02c"),
        ("WarmStart", plain_profile, "#1f77b4"),
        ("Baseline", strict_profile, "#d62728"),
    ]
    soc_png_path, soc_pdf_path = _plot_profiles(
        profiles,
        output_dir,
        lower_right_key="soc_frac",
        lower_right_label="SOC/%",
        filename_stem="seed8409_voltage_current_soc_temperature_profiles",
        lower_right_scale=100.0,
    )
    loss_png_path, loss_pdf_path = _plot_profiles(
        profiles,
        output_dir,
        lower_right_key="loss_pct",
        lower_right_label="Lithium-Ion Loss/%",
        filename_stem="seed8409_voltage_current_loss_temperature_profiles",
        lower_right_scale=1.0,
        lower_right_time_key="loss_time_s",
    )

    manifest = {
        "seed": 8409,
        "fixed_run_name": "fixed_seed8409_llmgp_winner_2026_05_01",
        "source_paths": {
            "winner_summary": str(WINNER_SUMMARY.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "winner_database": str(WINNER_DB.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "winner_db_final": str(WINNER_DB_FINAL.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "winner_pareto_front": str(WINNER_PARETO.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "warmstart_summary": str(PLAIN_SUMMARY.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "baseline_summary": str(STRICT_SUMMARY.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "metrics": {
            "winner": _extract_metrics(winner_summary),
            "warmstart_plain_ei": _extract_metrics(plain_summary),
            "strict_baseline": _extract_metrics(strict_summary),
        },
        "protocols": {
            "winner": winner_theta.round(12).tolist(),
            "warmstart_plain_ei": plain_theta.round(12).tolist(),
            "strict_baseline": strict_theta.round(12).tolist(),
        },
        "artifacts": {
            "profiles_json": str(trajectories_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "profiles_png": str(soc_png_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "profiles_pdf": str(soc_pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "profiles_loss_png": str(loss_png_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "profiles_loss_pdf": str(loss_pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the winning seed8409 run and plot profile curves.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will store the frozen run snapshot and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_fixed_seed8409_bundle(args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
