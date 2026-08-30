from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import plot_hv_convergence_5way as base


TARGET_PNG = base.TARGET_DIR / "hv_convergence_5way_distinct_start.png"
TARGET_PDF = base.TARGET_DIR / "hv_convergence_5way_distinct_start.pdf"
TARGET_JSON = base.TARGET_DIR / "hv_convergence_5way_distinct_start_starts.json"


def _record_start(label: str, x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "label": label,
        "first_eval_index": int(np.asarray(x)[0]),
        "first_canonical_hv": float(np.asarray(y, dtype=float)[0]),
    }


def main() -> None:
    base._configure_plot_style()

    llmbo_trace = base._extract_canonical_trace(base.LLMBO_SINGLE_SUMMARY)
    parego_trace = base._extract_canonical_trace(base.PAREGO_SINGLE_SUMMARY)
    nsga2 = base._stack_nsga2_mean_std()
    disk = base._stack_multiseed_mean_std_from_database(
        base._find_latest_dir("disk_python_Chen2020_5seeds_50evals_*"),
        "disk_Chen2020",
    )
    pimd = base._stack_multiseed_mean_std_from_database(
        base._find_latest_dir("pimd_python_Chen2020_5seeds_50evals_*"),
        "pimd_Chen2020",
    )

    llmbo_sources, llmbo_params = base._resolve_proxy_sources("llmbo_band_profile")
    parego_sources, parego_params = base._resolve_proxy_sources("parego_band_profile")
    llmbo_band = base._build_estimated_canonical_band(llmbo_sources, **llmbo_params)
    parego_band = base._build_estimated_canonical_band(parego_sources, **parego_params)

    if not np.array_equal(llmbo_trace["x"], llmbo_band["x"]):
        raise ValueError("LLMBO single trace and proxy band x-axis mismatch")
    if not np.array_equal(parego_trace["x"], parego_band["x"]):
        raise ValueError("ParEGO single trace and proxy band x-axis mismatch")

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    base._plot_line_with_band(
        ax,
        parego_trace["x"],
        parego_trace["hv"],
        parego_band["band"] * base.PAREGO_BAND_SCALE,
        color=base.PAREGO_COLOR,
        label="ParEGO",
        marker="s",
        alpha=base.ESTIMATED_BAND_ALPHA,
    )
    base._plot_line_with_band(
        ax,
        llmbo_trace["x"],
        llmbo_trace["hv"],
        llmbo_band["band"] * base.LLMBO_BAND_SCALE,
        color=base.LLMBO_COLOR,
        label="LLAMBO-MO",
        marker="o",
        alpha=base.ESTIMATED_BAND_ALPHA,
    )
    base._plot_line_with_band(
        ax,
        nsga2["x"],
        nsga2["mean"],
        nsga2["std"],
        color=base.NSGA2_COLOR,
        label="NSGA-II",
        marker="v",
        alpha=base.MULTISEED_BAND_ALPHA,
    )
    base._plot_line_with_band(
        ax,
        disk["x"],
        disk["mean"],
        disk["std"],
        color=base.DISK_COLOR,
        label="DISK",
        marker="^",
        alpha=base.MULTISEED_BAND_ALPHA,
    )
    base._plot_line_with_band(
        ax,
        pimd["x"],
        pimd["mean"],
        pimd["std"],
        color=base.PIMD_COLOR,
        label="PIMD",
        marker="D",
        alpha=base.MULTISEED_BAND_ALPHA,
    )

    ax.set_xlim(0, base.MAX_EVALS)
    ax.set_ylim(0.12, 0.40)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Canonical HV")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        handlelength=2.6,
        handletextpad=0.6,
    )
    fig.tight_layout()

    base.TARGET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(TARGET_PDF, dpi=240, bbox_inches="tight")
    plt.close(fig)

    starts = {
        "note": "Distinct-start version. No shared artificial starting point was prepended.",
        "output_png": str(TARGET_PNG),
        "output_pdf": str(TARGET_PDF),
        "starts": [
            _record_start("ParEGO", parego_trace["x"], parego_trace["hv"]),
            _record_start("LLAMBO-MO", llmbo_trace["x"], llmbo_trace["hv"]),
            _record_start("NSGA-II", nsga2["x"], nsga2["mean"]),
            _record_start("DISK", disk["x"], disk["mean"]),
            _record_start("PIMD", pimd["x"], pimd["mean"]),
        ],
    }
    TARGET_JSON.write_text(json.dumps(starts, indent=2), encoding="utf-8")

    print(f"Saved: {TARGET_PNG}")
    print(f"Saved: {TARGET_PDF}")
    print(f"Saved: {TARGET_JSON}")


if __name__ == "__main__":
    main()
