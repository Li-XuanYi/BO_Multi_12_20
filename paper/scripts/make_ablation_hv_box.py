"""Recreate the requested short-budget ablation boxplot for the paper.

The plotted values come from the archived three-seed, ten-iteration
Chen2020 run.  The archive retains its internal ``canonical_hv`` key for
reproducibility, while the reader-facing axis is deliberately labeled ``HV``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "Ablation_Exp"
    / "experiment_records"
    / "ablation_4way_3seeds_10iter_deepseek_v3_codex_2026_05_17"
)
DEFAULT_OUTPUT_BASE = PAPER_ROOT / "Section" / "figures" / "ablation_hv_box"
EXPECTED_SEEDS = (8409, 8410, 8411)
EXPECTED_ITERATIONS = 10
EXPECTED_KEYS = (
    "baseline",
    "baseline_warmstart",
    "baseline_llm_region",
    "llmbo_mo",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_archive(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiment_path = source_root / "manifest.json"
    plot_path = source_root / "plot_manifest.json"
    experiment = _load_json(experiment_path)
    plot_manifest = _load_json(plot_path)

    seeds = tuple(int(seed) for seed in experiment.get("seeds", []))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seeds in {experiment_path}: {seeds}")
    if int(experiment.get("iterations", -1)) != EXPECTED_ITERATIONS:
        raise ValueError(f"Expected 10 BO iterations in {experiment_path}")
    if str(plot_manifest.get("metric")) != "canonical_hv":
        raise ValueError(f"Unexpected archive metric in {plot_path}")

    groups = plot_manifest.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"Missing groups in {plot_path}")
    keys = tuple(str(group.get("key")) for group in groups)
    if keys != EXPECTED_KEYS:
        raise ValueError(f"Unexpected group order in {plot_path}: {keys}")
    for group in groups:
        values = np.asarray(group.get("values", []), dtype=float)
        if values.shape != (len(EXPECTED_SEEDS),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid values for {group.get('key')} in {plot_path}")

    provenance = {
        "experiment_manifest": str(experiment_path.resolve()),
        "experiment_manifest_sha256": _sha256(experiment_path),
        "plot_manifest": str(plot_path.resolve()),
        "plot_manifest_sha256": _sha256(plot_path),
    }
    return groups, provenance


def make_figure(
    groups: list[dict[str, Any]], output_png: Path, output_pdf: Path, dpi: int
) -> None:
    labels = [str(group["label"]) for group in groups]
    colors = [str(group["color"]) for group in groups]
    data = [np.asarray(group["values"], dtype=float) for group in groups]

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(1, len(data) + 1)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor("#222222")

    rng = np.random.default_rng(8409)
    for position, values, color in zip(positions, data, colors):
        offsets = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(
            position + offsets,
            values,
            s=42,
            color=color,
            edgecolor="#222222",
            linewidth=0.35,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("HV")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_manifest(
    groups: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_png: Path,
    output_pdf: Path,
) -> dict[str, Any]:
    return {
        "figure": "short-budget four-way ablation boxplot",
        "dataset": "Chen2020",
        "seeds": list(EXPECTED_SEEDS),
        "bo_iterations": EXPECTED_ITERATIONS,
        "initial_evaluations": 6,
        "total_evaluations": 16,
        "archive_metric_key": "canonical_hv",
        "reader_facing_axis_label": "HV",
        "groups": groups,
        "provenance": provenance,
        "outputs": {
            "png": {
                "path": str(output_png.resolve()),
                "sha256": _sha256(output_png),
                "bytes": output_png.stat().st_size,
            },
            "pdf": {
                "path": str(output_pdf.resolve()),
                "sha256": _sha256(output_pdf),
                "bytes": output_pdf.stat().st_size,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_base = args.output_base.resolve()
    output_png = output_base.with_suffix(".png")
    output_pdf = output_base.with_suffix(".pdf")
    output_manifest = output_base.with_name(output_base.name + "_manifest.json")

    groups, provenance = load_archive(source_root)
    make_figure(groups, output_png, output_pdf, args.dpi)
    manifest = build_manifest(groups, provenance, output_png, output_pdf)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
