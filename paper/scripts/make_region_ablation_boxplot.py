"""Create an audit-safe boxplot for the archived Region-lift ablation.

The two Region contrasts come from separate, internally paired experiment
batches:

* no warm start: Region lift versus Plain BO from the adaptive4 batch;
* warm start: Full versus Warm-start BO from the dedicated paired rerun.

They must not be combined into a cross-batch factorial interaction.  The
figure therefore presents the two cohorts in separate panels and reports only
their within-seed Region contrasts.

Outputs (by default)
--------------------
Section/figures/ablation_region_boxplot_historical.pdf
Section/figures/ablation_region_boxplot_historical.png
Section/figures/ablation_region_boxplot_historical_manifest.json

Run from any directory:

    python paper/scripts/make_region_ablation_boxplot.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import MultipleLocator


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
REPORT_ROOT = PROJECT_ROOT / "Ablation_Exp" / "Ablation523_4group" / "source_reports"
DEFAULT_NO_WARM_REPORT = REPORT_ROOT / "adaptive4_5seeds_50iter_report.json"
DEFAULT_WARM_REPORT = REPORT_ROOT / "warmstart_vs_llmbo_paired_5seeds_50iter_report.json"
DEFAULT_OUTPUT_BASE = (
    PAPER_ROOT / "Section" / "figures" / "ablation_region_boxplot_historical"
)

EXPECTED_SEEDS = (8409, 8410, 8411, 8412, 8413)
REGION_OFF = "#4C566A"
REGION_ON = "#E69F00"
GRID_COLOR = "#D9D9D9"
PAIR_COLOR = "#B9B9B9"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_report(report: dict[str, Any], path: Path, variant_set: str) -> None:
    meta = report.get("meta", {})
    if str(meta.get("variant_set")) != variant_set:
        raise ValueError(f"Unexpected variant_set in {path}: {meta.get('variant_set')!r}")
    if int(meta.get("iterations", -1)) != 50:
        raise ValueError(f"Expected 50 BO iterations in {path}")
    if tuple(int(seed) for seed in meta.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seed set in {path}")
    if str(meta.get("hv_metric")) != "canonical_hv":
        raise ValueError(f"Expected canonical_hv in {path}")


def _variant_values(
    report: dict[str, Any], path: Path, variant: str
) -> np.ndarray:
    by_seed: dict[int, float] = {}
    for record in report.get("records", []):
        if record.get("variant") != variant:
            continue
        seed = int(record["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate {variant}/seed{seed} record in {path}")
        if record.get("status") != "ok":
            raise ValueError(f"Failed {variant}/seed{seed} record in {path}")
        by_seed[seed] = float(record["canonical_hv"])

    missing = [seed for seed in EXPECTED_SEEDS if seed not in by_seed]
    if missing:
        raise ValueError(f"Missing {variant} seeds {missing} in {path}")
    values = np.asarray([by_seed[seed] for seed in EXPECTED_SEEDS], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite {variant} values in {path}")
    return values


def _validate_reported_comparison(
    report: dict[str, Any],
    path: Path,
    comparison_key: str,
    computed_delta: np.ndarray,
) -> None:
    comparison = report.get("comparisons", {}).get(comparison_key)
    if not isinstance(comparison, dict):
        raise ValueError(f"Missing comparison {comparison_key!r} in {path}")
    shared_seeds = tuple(int(seed) for seed in comparison.get("shared_seeds", []))
    if shared_seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected paired seeds for {comparison_key!r} in {path}")
    reported_entries = comparison.get("per_seed_delta", [])
    reported_by_seed = {
        int(entry["seed"]): float(entry["delta"]) for entry in reported_entries
    }
    try:
        reported_delta = np.asarray(
            [reported_by_seed[seed] for seed in EXPECTED_SEEDS], dtype=float
        )
    except KeyError as error:
        raise ValueError(
            f"Missing per-seed delta for {comparison_key!r} in {path}"
        ) from error
    if not np.allclose(reported_delta, computed_delta, rtol=0.0, atol=1e-12):
        raise ValueError(f"Reported deltas disagree for {comparison_key!r} in {path}")
    reported_mean = float(comparison["mean_canonical_hv_delta"])
    if not np.isclose(reported_mean, np.mean(computed_delta), rtol=0.0, atol=1e-12):
        raise ValueError(f"Reported mean delta disagrees for {comparison_key!r} in {path}")


def load_paired_cohorts(
    no_warm_path: Path, warm_path: Path
) -> dict[str, dict[str, Any]]:
    """Load the two independently paired Region cohorts."""

    no_warm_report = _load_json(no_warm_path)
    warm_report = _load_json(warm_path)
    _validate_report(no_warm_report, no_warm_path, "adaptive4")
    _validate_report(warm_report, warm_path, "warmstart_llmbo_paired")

    plain = _variant_values(no_warm_report, no_warm_path, "baseline")
    region = _variant_values(no_warm_report, no_warm_path, "baseline_llm_region")
    warm = _variant_values(warm_report, warm_path, "baseline_warmstart")
    full = _variant_values(warm_report, warm_path, "llmbo_mo")
    _validate_reported_comparison(
        no_warm_report,
        no_warm_path,
        "baseline_llm_region_vs_baseline",
        region - plain,
    )
    _validate_reported_comparison(
        warm_report,
        warm_path,
        "llmbo_mo_vs_baseline_warmstart",
        full - warm,
    )

    return {
        "no_warm": {
            "source": no_warm_path,
            "source_label": "archived adaptive4 batch (2026-05-22)",
            "left_key": "baseline",
            "left_label": "Plain\nBO",
            "left_values": plain,
            "right_key": "baseline_llm_region",
            "right_label": "Region\nlift",
            "right_values": region,
            "delta_label": "No warm start\nRegion - Plain",
            "pairing": "shared six-point random initialization within each seed",
        },
        "warm": {
            "source": warm_path,
            "source_label": "archived paired rerun (2026-05-23)",
            "left_key": "baseline_warmstart",
            "left_label": "Warm-start\nBO",
            "left_values": warm,
            "right_key": "llmbo_mo",
            "right_label": "Warm start\n+ Region",
            "right_values": full,
            "delta_label": "Warm start\nFull - Warm",
            "pairing": "shared selected warm-start cache within each seed",
        },
    }


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _box(
    ax: plt.Axes,
    position: float,
    values: np.ndarray,
    color: str,
) -> None:
    artists = ax.boxplot(
        [values],
        positions=[position],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        manage_ticks=False,
        medianprops={"color": "#202020", "linewidth": 1.15},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.8},
        capprops={"color": "#4A4A4A", "linewidth": 0.8},
        boxprops={"color": color, "linewidth": 0.95},
    )
    artists["boxes"][0].set_facecolor(to_rgba(color, 0.17))


def _draw_pair_panel(
    ax: plt.Axes,
    cohort: dict[str, Any],
    title: str,
    show_ylabel: bool,
) -> None:
    left = np.asarray(cohort["left_values"], dtype=float)
    right = np.asarray(cohort["right_values"], dtype=float)
    positions = (1.0, 2.0)
    offsets = np.linspace(-0.095, 0.095, len(EXPECTED_SEEDS))

    _box(ax, positions[0], left, REGION_OFF)
    _box(ax, positions[1], right, REGION_ON)

    for index, offset in enumerate(offsets):
        x_values = [positions[0] + offset, positions[1] + offset]
        y_values = [left[index], right[index]]
        ax.plot(
            x_values,
            y_values,
            color=PAIR_COLOR,
            linewidth=0.65,
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            x_values[0],
            y_values[0],
            s=22,
            marker="o",
            facecolor=REGION_OFF,
            edgecolor="#202020",
            linewidth=0.4,
            zorder=3,
        )
        ax.scatter(
            x_values[1],
            y_values[1],
            s=26,
            marker="^",
            facecolor=REGION_ON,
            edgecolor="#202020",
            linewidth=0.4,
            zorder=3,
        )

    for position, values, color in zip(positions, (left, right), (REGION_OFF, REGION_ON)):
        ax.scatter(
            position,
            float(np.mean(values)),
            s=31,
            marker="D",
            facecolor="white",
            edgecolor=color,
            linewidth=1.05,
            zorder=4,
        )

    ax.set_xlim(0.55, 2.45)
    ax.set_ylim(0.364, 0.4105)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.set_xticks(positions)
    ax.set_xticklabels([cohort["left_label"], cohort["right_label"]])
    if show_ylabel:
        ax.set_ylabel(r"Final scaled hypervolume $\uparrow$")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.text(
        0.0,
        1.015,
        str(cohort["source_label"]),
        transform=ax.transAxes,
        color="#666666",
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def _draw_delta_panel(ax: plt.Axes, cohorts: dict[str, dict[str, Any]]) -> None:
    deltas = [
        np.asarray(cohorts["no_warm"]["right_values"], dtype=float)
        - np.asarray(cohorts["no_warm"]["left_values"], dtype=float),
        np.asarray(cohorts["warm"]["right_values"], dtype=float)
        - np.asarray(cohorts["warm"]["left_values"], dtype=float),
    ]
    positions = (1.0, 2.0)
    offsets = np.linspace(-0.095, 0.095, len(EXPECTED_SEEDS))

    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=0)
    for position, values in zip(positions, deltas):
        _box(ax, position, values, REGION_ON)
        ax.scatter(
            position + offsets,
            values,
            s=27,
            marker="^",
            facecolor=REGION_ON,
            edgecolor="#202020",
            linewidth=0.4,
            zorder=3,
        )
        ax.scatter(
            position,
            float(np.mean(values)),
            s=33,
            marker="D",
            facecolor="white",
            edgecolor=REGION_ON,
            linewidth=1.05,
            zorder=4,
        )

    for position, values in zip(positions, deltas):
        mean = float(np.mean(values))
        wins = int(np.sum(values > 0.0))
        sign = "+" if mean >= 0.0 else ""
        ax.text(
            position,
            0.0250,
            f"mean {sign}{mean:.4f}\n{wins}/5 positive",
            ha="center",
            va="top",
            fontsize=7.0,
            color="#333333",
        )

    ax.set_xlim(0.55, 2.45)
    ax.set_ylim(-0.026, 0.026)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [cohorts["no_warm"]["delta_label"], cohorts["warm"]["delta_label"]]
    )
    ax.set_ylabel(r"Paired $\Delta$ scaled hypervolume")
    ax.set_title("(c) Region effect by initialization", loc="left", fontweight="bold", pad=14)
    ax.text(
        0.0,
        1.015,
        "within-seed contrasts; $n=5$",
        transform=ax.transAxes,
        color="#666666",
        fontsize=7.0,
        ha="left",
        va="bottom",
    )
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)


def make_figure(
    cohorts: dict[str, dict[str, Any]],
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    _set_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.85),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.18), "wspace": 0.32},
    )

    _draw_pair_panel(
        axes[0],
        cohorts["no_warm"],
        "(a) Random initialization",
        show_ylabel=True,
    )
    _draw_pair_panel(
        axes[1],
        cohorts["warm"],
        "(b) Warm-start initialization",
        show_ylabel=False,
    )
    _draw_delta_panel(axes[2], cohorts)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.7)

    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.205, top=0.84)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "values_by_seed": {
            str(seed): float(value) for seed, value in zip(EXPECTED_SEEDS, values)
        },
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_std": float(np.std(values, ddof=1)),
    }


def _exact_sign_flip_p(delta: np.ndarray) -> float:
    """Two-sided exact paired sign-flip p-value over all 2**n assignments."""

    observed = abs(float(np.mean(delta)))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta)):
        permuted.append(abs(float(np.mean(delta * np.asarray(signs, dtype=float)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def build_manifest(
    cohorts: dict[str, dict[str, Any]],
    output_pdf: Path,
    output_png: Path,
) -> dict[str, Any]:
    cohort_payload: dict[str, Any] = {}
    raw_p_values: list[float] = []

    for key in ("no_warm", "warm"):
        cohort = cohorts[key]
        left = np.asarray(cohort["left_values"], dtype=float)
        right = np.asarray(cohort["right_values"], dtype=float)
        delta = right - left
        exact_p = _exact_sign_flip_p(delta)
        raw_p_values.append(exact_p)
        cohort_payload[key] = {
            "source": str(Path(cohort["source"]).resolve()),
            "source_sha256": _sha256(Path(cohort["source"])),
            "pairing": cohort["pairing"],
            "left_variant": cohort["left_key"],
            "left": _summary(left),
            "right_variant": cohort["right_key"],
            "right": _summary(right),
            "paired_delta_right_minus_left": {
                **_summary(delta),
                "wins_positive": int(np.sum(delta > 0.0)),
                "exact_two_sided_sign_flip_p": exact_p,
            },
        }

    # Holm adjustment over the two prespecified Region contrasts.  These values
    # are retained for auditability, not displayed as significance annotations.
    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(raw_p_values) - rank) * raw_p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    for key, adjusted_p in zip(("no_warm", "warm"), adjusted):
        cohort_payload[key]["paired_delta_right_minus_left"][
            "holm_adjusted_p_over_two_contrasts"
        ] = float(adjusted_p)

    caption = (
        "Archived Region-lift ablation at the final canonical hypervolume after "
        "50 BO iterations. Panels (a) and (b) are separate, internally paired "
        "five-seed cohorts: Plain BO versus Region lift under shared random "
        "initialization, and Warm-start BO versus Warm start + Region under a "
        "shared selected warm-start cache. Boxes show the median and interquartile "
        "range, whiskers span the observed range, dots show all seeds 8409--8413, "
        "thin segments connect paired runs, and white diamonds mark means. Panel "
        "(c) shows the corresponding within-seed Region differences; the horizontal "
        "line denotes zero. The cohorts come from different archived batches, so "
        "no cross-batch factorial or interaction contrast is implied. Results are "
        "descriptive (n=5); BO iterations are not treated as independent samples."
    )

    return {
        "figure": "historical Region-lift paired ablation boxplot",
        "method_status": (
            "archived adaptive-posterior Region implementation; not the corrected "
            "Region_Lift preset introduced after these runs"
        ),
        "dataset": "Chen2020",
        "metric": "canonical_hv (plotted as scaled hypervolume)",
        "bo_iterations": 50,
        "initial_points": 6,
        "seeds": list(EXPECTED_SEEDS),
        "box_definition": "median/IQR; whiskers=min/max; all raw points shown",
        "cohorts": cohort_payload,
        "claim_boundary": (
            "The no-warm cohort supports only a descriptive Region gain; the warm "
            "cohort is mixed. This figure is not evidence for the corrected preset."
        ),
        "recommended_caption": caption,
        "outputs": {
            "pdf": {
                "path": str(output_pdf.resolve()),
                "sha256": _sha256(output_pdf),
                "bytes": output_pdf.stat().st_size,
            },
            "png": {
                "path": str(output_png.resolve()),
                "sha256": _sha256(output_png),
                "bytes": output_png.stat().st_size,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-warm-report",
        type=Path,
        default=DEFAULT_NO_WARM_REPORT,
        help="adaptive4 report containing the paired Plain/Region runs",
    )
    parser.add_argument(
        "--warm-report",
        type=Path,
        default=DEFAULT_WARM_REPORT,
        help="dedicated paired Warm-start/Full report",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="output path without extension",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_warm_path = args.no_warm_report.resolve()
    warm_path = args.warm_report.resolve()
    output_base = args.output_base.resolve()
    output_pdf = output_base.with_suffix(".pdf")
    output_png = output_base.with_suffix(".png")
    output_manifest = output_base.with_name(output_base.name + "_manifest.json")

    cohorts = load_paired_cohorts(no_warm_path, warm_path)
    make_figure(cohorts, output_pdf, output_png, args.dpi)
    manifest = build_manifest(cohorts, output_pdf, output_png)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest: {output_manifest}")


if __name__ == "__main__":
    main()
