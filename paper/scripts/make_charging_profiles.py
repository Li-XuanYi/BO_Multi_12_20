r"""Replay archived Chen2020 protocols and plot their charging trajectories.

This script produces the publication artifacts

    paper/figures/charging_profiles.pdf
    paper/figures/charging_profiles.png

from the same ``PyBaMMSimulator`` used by the optimization code. The three
decision vectors are illustrative successfully simulated protocols from the
archived LLMBO-MO Chen2020 seed-8409 run; the low-temperature example is not
in that run's final nondominated set. No interpolated or synthetic
trajectories are used.

Run from any directory with the project ``llambo`` environment, for example:

    C:\Users\aa133\miniconda3\envs\llambo\python.exe \
        paper\scripts\make_charging_profiles.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from pybamm_simulator import PyBaMMSimulator  # noqa: E402


@dataclass(frozen=True)
class Protocol:
    label: str
    theta: tuple[float, float, float, float, float]
    archived_objectives: tuple[float, float, float]
    color: str
    linestyle: str


# Source:
# optimized_experiments/region_lift_force_pool_local_sweep_seed8409_2026_05_01/
# seed8409/wider_active16_ext32/database.json
PROTOCOLS: tuple[Protocol, ...] = (
    Protocol(
        label="Fast",
        theta=(6.0, 5.0, 3.0, 0.4, 0.2999951281178859),
        archived_objectives=(2880.0, 7.566964191322911, 1.2606511928531257),
        color="#D55E00",
        linestyle="-",
    ),
    Protocol(
        label="Intermediate",
        theta=(2.0, 2.0, 3.0, 0.31974642677374054, 0.11767167837384315),
        archived_objectives=(6112.0, 2.8574864297119125, 0.5713455870696137),
        color="#0072B2",
        linestyle="--",
    ),
    Protocol(
        label="Low-temperature",
        theta=(2.0, 2.0, 2.0, 0.1, 0.3),
        archived_objectives=(7200.0, 1.5285791733729015, 0.6401557211840261),
        color="#009E73",
        linestyle="-.",
    ),
)

SOURCE_DATABASE = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "region_lift_force_pool_local_sweep_seed8409_2026_05_01"
    / "seed8409"
    / "wider_active16_ext32"
    / "database.json"
)


def _replay_protocols() -> Dict[str, Mapping[str, np.ndarray]]:
    """Replay every protocol with the paper's semi-empirical degradation proxy."""

    simulator = PyBaMMSimulator(
        param_set="Chen2020",
        aging_mode="empirical",
    )
    profiles: Dict[str, Mapping[str, np.ndarray]] = {}

    for protocol in PROTOCOLS:
        theta = np.asarray(protocol.theta, dtype=float)
        result = simulator.evaluate(theta)
        if not bool(result.get("feasible", False)):
            raise RuntimeError(
                f"Chen2020 replay failed for {protocol.label}: "
                f"{result.get('violation', 'unknown simulator error')}"
            )

        objectives = np.asarray(result["raw_objectives"], dtype=float)
        archived = np.asarray(protocol.archived_objectives, dtype=float)
        if not np.allclose(objectives, archived, rtol=1e-7, atol=1e-7):
            raise RuntimeError(
                f"{protocol.label} replay no longer matches the archived run. "
                f"replayed={objectives.tolist()}, archived={archived.tolist()}"
            )

        trajectories = result.get("trajectories")
        if not isinstance(trajectories, Mapping):
            raise RuntimeError(f"{protocol.label} replay returned no trajectories")

        profile = {
            "time_s": np.asarray(trajectories["time"], dtype=float),
            "current_a": np.asarray(trajectories["I"], dtype=float),
            "voltage_v": np.asarray(trajectories["V"], dtype=float),
            "temperature_k": np.asarray(trajectories["T"], dtype=float),
            "soc_pct": 100.0 * np.asarray(trajectories["SOC"], dtype=float),
            "objectives": objectives,
        }
        lengths = {array.size for key, array in profile.items() if key != "objectives"}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise RuntimeError(
                f"{protocol.label} replay returned misaligned trajectory arrays"
            )
        if not all(np.all(np.isfinite(array)) for array in profile.values()):
            raise RuntimeError(
                f"{protocol.label} replay returned non-finite trajectory values"
            )
        profiles[protocol.label] = profile

    return profiles


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.7,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.3,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _plot_profiles(
    profiles: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.65))
    current_ax, voltage_ax, temperature_ax, soc_ax = axes.ravel()

    panel_specs = (
        (current_ax, "current_a", "Applied current (A)", "(a)"),
        (voltage_ax, "voltage_v", "Terminal voltage (V)", "(b)"),
        (temperature_ax, "temperature_k", "Cell temperature (K)", "(c)"),
        (soc_ax, "soc_pct", "State of charge (%)", "(d)"),
    )

    handles = []
    for protocol in PROTOCOLS:
        profile = profiles[protocol.label]
        time_h = profile["time_s"] / 3600.0
        for axis, key, _, _ in panel_specs:
            (line,) = axis.plot(
                time_h,
                profile[key],
                color=protocol.color,
                linestyle=protocol.linestyle,
                label=protocol.label,
                solid_capstyle="round",
            )
            if axis is current_ax:
                handles.append(line)

    max_time_h = max(
        float(profile["time_s"][-1]) / 3600.0 for profile in profiles.values()
    )
    x_max = float(np.ceil(max_time_h * 2.0) / 2.0)
    x_ticks = np.arange(0.0, x_max + 0.01, 0.5)

    for axis, _, ylabel, panel_label in panel_specs:
        axis.set_xlim(0.0, x_max)
        axis.set_xticks(x_ticks)
        axis.set_xlabel("Time (h)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.7)
        axis.text(
            0.02,
            0.95,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
        )

    current_ax.set_ylim(bottom=0.0)
    voltage_ax.set_ylim(2.75, 4.45)
    voltage_ax.axhline(
        4.4,
        color="#555555",
        linestyle=(0, (2, 2)),
        linewidth=0.8,
        zorder=0,
    )
    voltage_ax.text(
        0.985,
        4.4,
        "4.4-V limit",
        transform=voltage_ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    soc_ax.set_ylim(0.0, 100.0)

    fig.legend(
        handles=handles,
        labels=[protocol.label for protocol in PROTOCOLS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.0,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.895,
        wspace=0.28,
        hspace=0.40,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "charging_profiles.pdf"
    png_path = output_dir / "charging_profiles.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


def _write_manifest(
    profiles: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    pdf_path: Path,
    png_path: Path,
) -> Path:
    try:
        import pybamm

        pybamm_version = pybamm.__version__
    except Exception:
        pybamm_version = "unknown"

    payload = {
        "figure": "charging_profiles",
        "source_database": str(SOURCE_DATABASE.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "battery_parameter_set": "Chen2020",
        "simulator": "PyBaMMSimulator (SPMe, lumped thermal)",
        "aging_mode": "empirical",
        "note": (
            "The figure contains current, voltage, temperature, and SOC only. "
            "No physical lithium-loss trajectory is presented as empirical "
            "relative degradation proxy."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pybamm": pybamm_version,
        },
        "protocols": {
            protocol.label: {
                "theta": list(protocol.theta),
                "archived_objectives": list(protocol.archived_objectives),
                "replayed_objectives": profiles[protocol.label][
                    "objectives"
                ].tolist(),
                "n_time_points": int(profiles[protocol.label]["time_s"].size),
            }
            for protocol in PROTOCOLS
        },
        "artifacts": {
            "pdf": str(pdf_path.relative_to(PAPER_ROOT)).replace("\\", "/"),
            "png": str(png_path.relative_to(PAPER_ROOT)).replace("\\", "/"),
        },
    }
    manifest_path = output_dir / "charging_profiles_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay three archived Chen2020 seed-8409 LLMBO-MO protocols and "
            "generate a publication-ready 2x2 charging-profile figure."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_ROOT / "Section" / "figures",
        help="Output directory (default: paper/figures).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution in dots per inch (default: 300).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PAPER_ROOT / output_dir

    if not SOURCE_DATABASE.is_file():
        raise FileNotFoundError(
            "Archived Chen2020 seed-8409 source database is missing: "
            f"{SOURCE_DATABASE}"
        )

    profiles = _replay_protocols()
    pdf_path, png_path = _plot_profiles(profiles, output_dir, dpi=args.dpi)
    manifest_path = _write_manifest(
        profiles,
        output_dir,
        pdf_path,
        png_path,
    )

    print(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "png": str(png_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
