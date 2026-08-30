"""Alternative convergence plot styles for ablation study comparison.

Generates three variants for visual comparison:
  A) Inset zoom + progressive colors
  B) Error bars instead of shaded CI
  C) Convergence + final HV bar chart (combined)

Run:  python plot_4group_variants.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

# Same-directory import
import plot_4group_compare_style as base

OUT_DIR = base.OUT_DIR

# Progressive style: gray → light blue → deep blue → bright red
GROUPS_PROGRESSIVE = [
    {"key": "baseline", "label": "Baseline", "source": "adaptive", "variant": "baseline",
     "color": "#A0A0A0", "marker": "o", "ls": "--", "lw": 1.5, "zo": 1},
    {"key": "warmstart", "label": "+ WarmStart", "source": "adaptive", "variant": "baseline_warmstart",
     "color": "#6baed6", "marker": "s", "ls": "-", "lw": 1.8, "zo": 2},
    {"key": "llm_region", "label": "+ LLM_Region", "source": "adaptive", "variant": "baseline_llm_region",
     "color": "#2171b5", "marker": "^", "ls": "-", "lw": 2.0, "zo": 3},
    {"key": "llmbo", "label": "LLMBO (Ours)", "source": "paired", "variant": "llmbo_mo",
     "color": "#e31a1c", "marker": "D", "ls": "-", "lw": 2.5, "zo": 5},
]


def _trace_data(rows: List[Mapping[str, Any]],
                style_map: Dict[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Load traces and compute mean/std for each group."""
    data: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row["key"] not in style_map:
            continue
        traces = base._load_group_traces(row)
        if not traces:
            continue
        ml = min(len(t) for t in traces)
        stack = np.vstack([t[:ml] for t in traces])
        data[row["key"]] = {
            "x": np.arange(1, ml + 1),
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0),
            "style": style_map[row["key"]],
        }
    return data


# ── Variant A: Inset + Progressive Colors ──────────────────────────────────

def _plot_convergence_inset(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    base._set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    style_map = {g["key"]: g for g in GROUPS_PROGRESSIVE}
    data = _trace_data(rows, style_map)
    if not data:
        return {}

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for key, d in data.items():
        s = d["style"]
        alpha = 0.06 if key in ("baseline", "llm_region") else 0.13
        ax.plot(d["x"], d["mean"], color=s["color"], lw=s["lw"], marker=s["marker"],
                markevery=max(len(d["x"]) // 7, 1), ms=5 if key == "llmbo" else 4,
                label=s["label"], ls=s["ls"], solid_capstyle="round", zorder=s["zo"])
        ax.fill_between(d["x"], d["mean"] - d["std"], d["mean"] + d["std"],
                        color=s["color"], alpha=alpha, lw=0, zorder=s["zo"] - 0.5)

    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Canonical HV")
    ax.set_title("Ablation HV Convergence (Inset Zoom)")
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.grid(True, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777777",
              handlelength=2.6, loc="lower right")

    # ── Inset: zoom into final convergence region ──
    axins = inset_axes(ax, width="42%", height="35%", loc="center right",
                       bbox_to_anchor=(0.01, 0.06, 1, 1), bbox_transform=ax.transAxes)
    max_len = max(len(d["x"]) for d in data.values())
    x_start = int(max_len * 0.65)

    # Only plot mean lines (no CI) in inset for clarity
    for key, d in data.items():
        s = d["style"]
        mask = d["x"] >= x_start
        axins.plot(d["x"][mask], d["mean"][mask], color=s["color"], lw=s["lw"] * 0.9,
                   marker=s["marker"], markevery=2, ms=3.5,
                   ls=s["ls"], zorder=s["zo"])

    # Tight y-range based on mean curves only
    y_vals: List[float] = []
    for d in data.values():
        mask = d["x"] >= x_start
        if mask.any():
            y_vals.extend(d["mean"][mask].tolist())
    if y_vals:
        margin = (max(y_vals) - min(y_vals)) * 0.10
        axins.set_xlim(x_start, max_len + 1)
        axins.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

    axins.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    axins.tick_params(labelsize=7)
    axins.set_title("Zoomed (last 35%)", fontsize=8, pad=2)
    axins.grid(True, alpha=0.5, lw=0.5)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#888888", lw=0.8, ls="--")

    fig.tight_layout()
    png = OUT_DIR / "ablation_convergence_inset.png"
    pdf = OUT_DIR / "ablation_convergence_inset.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


# ── Variant B: Error Bars ──────────────────────────────────────────────────

def _plot_convergence_errorbar(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    base._set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    style_map = {g["key"]: g for g in GROUPS_PROGRESSIVE}
    data = _trace_data(rows, style_map)
    if not data:
        return {}

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for key, d in data.items():
        s = d["style"]
        step = max(len(d["x"]) // 8, 1)
        ax.plot(d["x"], d["mean"], color=s["color"], lw=s["lw"], marker=s["marker"],
                markevery=step, ms=5 if key == "llmbo" else 4,
                label=s["label"], ls=s["ls"], solid_capstyle="round", zorder=s["zo"])
        idx = np.arange(step - 1, len(d["x"]), step)
        ax.errorbar(d["x"][idx], d["mean"][idx], yerr=d["std"][idx],
                    fmt="none", ecolor=s["color"], elinewidth=1.0,
                    capsize=2.5, capthick=0.8, alpha=0.55, zorder=s["zo"] - 0.5)

    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Canonical HV")
    ax.set_title("Ablation HV Convergence (Error Bars)")
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.grid(True, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777777",
              handlelength=2.6, loc="lower right")
    fig.tight_layout()

    png = OUT_DIR / "ablation_convergence_errorbar.png"
    pdf = OUT_DIR / "ablation_convergence_errorbar.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


# ── Variant C: Convergence + Final Bar Chart ───────────────────────────────

def _plot_convergence_combined(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    from matplotlib.gridspec import GridSpec

    base._set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    style_map = {g["key"]: g for g in GROUPS_PROGRESSIVE}
    data = _trace_data(rows, style_map)
    if not data:
        return {}

    fig = plt.figure(figsize=(12.0, 5.0))
    gs = GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.30)
    ax_conv = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # Left: convergence
    for key, d in data.items():
        s = d["style"]
        alpha = 0.06 if key in ("baseline", "llm_region") else 0.13
        ax_conv.plot(d["x"], d["mean"], color=s["color"], lw=s["lw"], marker=s["marker"],
                     markevery=max(len(d["x"]) // 7, 1), ms=5 if key == "llmbo" else 4,
                     label=s["label"], ls=s["ls"], zorder=s["zo"])
        ax_conv.fill_between(d["x"], d["mean"] - d["std"], d["mean"] + d["std"],
                             color=s["color"], alpha=alpha, lw=0, zorder=s["zo"] - 0.5)
    ax_conv.set_xlabel("Evaluation Index")
    ax_conv.set_ylabel("Canonical HV")
    ax_conv.set_title("HV Convergence")
    ax_conv.yaxis.set_major_locator(MultipleLocator(0.01))
    ax_conv.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax_conv.grid(True, alpha=0.85)
    ax_conv.set_axisbelow(True)
    ax_conv.legend(frameon=True, fancybox=False, edgecolor="#777777",
                   handlelength=2.6, loc="lower right")

    # Right: final HV bar chart
    keys_ordered = [g["key"] for g in GROUPS_PROGRESSIVE if g["key"] in data]
    labels = [style_map[k]["label"] for k in keys_ordered]
    colors = [style_map[k]["color"] for k in keys_ordered]
    means = np.array([data[k]["mean"][-1] for k in keys_ordered])
    stds = np.array([data[k]["std"][-1] for k in keys_ordered])

    x_bar = np.arange(len(keys_ordered))
    ax_bar.bar(x_bar, means, yerr=stds, capsize=4, color=colors, alpha=0.86,
               edgecolor="#555555", lw=0.8,
               error_kw={"elinewidth": 1.2, "ecolor": "#444444"})
    for i, (m, s) in enumerate(zip(means, stds)):
        ax_bar.text(i, m + s + 0.001, f"{m:.4f}", ha="center", va="bottom",
                    fontsize=9.5, color="#333333")

    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.set_ylabel("Final Canonical HV")
    ax_bar.set_title("Final Performance")
    ax_bar.grid(True, axis="y", alpha=0.85)
    ax_bar.set_axisbelow(True)
    y_lo = min(means - stds) - 0.006
    ax_bar.set_ylim(y_lo, max(means + stds) + 0.012)

    fig.tight_layout()
    png = OUT_DIR / "ablation_convergence_combined.png"
    pdf = OUT_DIR / "ablation_convergence_combined.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


# ── Variant D: Delta Plot (all curves minus Baseline) ──────────────────────

def _plot_delta(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    base._set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    style_map = {g["key"]: g for g in GROUPS_PROGRESSIVE}
    data = _trace_data(rows, style_map)
    if not data or "baseline" not in data:
        return {}

    bl = data["baseline"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for key, d in data.items():
        s = d["style"]
        delta = d["mean"] - bl["mean"]
        # propagate std: sqrt(std_a^2 + std_b^2) for independent seeds
        combined_std = np.sqrt(d["std"]**2 + bl["std"]**2)
        ax.plot(d["x"], delta, color=s["color"], lw=s["lw"], marker=s["marker"],
                markevery=max(len(d["x"]) // 7, 1), ms=5 if key == "llmbo" else 4,
                label=s["label"], ls=s["ls"], zorder=s["zo"])
        if key != "baseline":
            ax.fill_between(d["x"], delta - combined_std, delta + combined_std,
                            color=s["color"], alpha=0.10, lw=0, zorder=s["zo"] - 0.5)

    ax.axhline(0, color="#555555", lw=1.0, ls="-", zorder=0)
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Δ Canonical HV vs Baseline")
    ax.set_title("Ablation: HV Gain Over Baseline")
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
    ax.grid(True, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777777",
              handlelength=2.6, loc="upper left")
    fig.tight_layout()

    png = OUT_DIR / "ablation_convergence_delta.png"
    pdf = OUT_DIR / "ablation_convergence_delta.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


# ── Variant E: Delta + Fill Gain Region + Annotation ───────────────────────

def _plot_delta_fill(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    base._set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    style_map = {g["key"]: g for g in GROUPS_PROGRESSIVE}
    data = _trace_data(rows, style_map)
    if not data or "baseline" not in data:
        return {}

    bl = data["baseline"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # Shaded gain region between LLMBO and baseline
    if "llmbo" in data:
        delta_llmbo = data["llmbo"]["mean"] - bl["mean"]
        x = data["llmbo"]["x"]
        ax.fill_between(x, 0, delta_llmbo, color="#e31a1c", alpha=0.12,
                        lw=0, zorder=0, label="LLMBO gain region")

    for key, d in data.items():
        s = d["style"]
        delta = d["mean"] - bl["mean"]
        ax.plot(d["x"], delta, color=s["color"], lw=s["lw"], marker=s["marker"],
                markevery=max(len(d["x"]) // 7, 1), ms=5 if key == "llmbo" else 4,
                label=s["label"], ls=s["ls"], zorder=s["zo"])

    # Annotate final delta
    if "llmbo" in data:
        final_delta = float(data["llmbo"]["mean"][-1] - bl["mean"][-1])
        ax.annotate(
            f"ΔHV = +{final_delta:.4f}",
            xy=(len(data["llmbo"]["x"]), final_delta),
            xytext=(-80, 15), textcoords="offset points",
            fontsize=10, fontweight="bold", color="#e31a1c",
            arrowprops=dict(arrowstyle="->", color="#e31a1c", lw=1.2),
        )

    ax.axhline(0, color="#555555", lw=1.0, zorder=0)
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Δ Canonical HV vs Baseline")
    ax.set_title("Ablation: Cumulative Gain of LLMBO")
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
    ax.grid(True, alpha=0.85)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, fancybox=False, edgecolor="#777777",
              handlelength=2.6, loc="upper left")
    fig.tight_layout()

    png = OUT_DIR / "ablation_convergence_delta_fill.png"
    pdf = OUT_DIR / "ablation_convergence_delta_fill.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


# ── Variant F: Last Half Only + Tight Y (no broken axis needed) ────────────

def _plot_half_zoom(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    base._set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    style_map = {g["key"]: g for g in GROUPS_PROGRESSIVE}
    data = _trace_data(rows, style_map)
    if not data:
        return {}

    max_len = max(len(d["x"]) for d in data.values())
    x_start = int(max_len * 0.45)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y_all: List[float] = []
    for key, d in data.items():
        s = d["style"]
        mask = d["x"] >= x_start
        ax.plot(d["x"][mask], d["mean"][mask], color=s["color"], lw=s["lw"],
                marker=s["marker"], markevery=3, ms=5 if key == "llmbo" else 4,
                label=s["label"], ls=s["ls"], zorder=s["zo"])
        alpha = 0.06 if key in ("baseline", "llm_region") else 0.13
        ax.fill_between(d["x"][mask], d["mean"][mask] - d["std"][mask],
                        d["mean"][mask] + d["std"][mask],
                        color=s["color"], alpha=alpha, lw=0, zorder=s["zo"] - 0.5)
        y_all.extend(d["mean"][mask].tolist())

    y_margin = (max(y_all) - min(y_all)) * 0.15
    ax.set_xlim(x_start, max_len + 1)
    ax.set_ylim(min(y_all) - y_margin, max(y_all) + y_margin)
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Canonical HV")
    ax.set_title("Ablation HV Convergence (Last 55%)")
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
    ax.grid(True, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777777",
              handlelength=2.6, loc="lower right")
    fig.tight_layout()

    png = OUT_DIR / "ablation_convergence_half.png"
    pdf = OUT_DIR / "ablation_convergence_half.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def main() -> None:
    rows = base._build_table()
    funcs = {
        "inset": _plot_convergence_inset,
        "errorbar": _plot_convergence_errorbar,
        "combined": _plot_convergence_combined,
        "delta": _plot_delta,
        "delta_fill": _plot_delta_fill,
        "half_zoom": _plot_half_zoom,
    }
    for name, fn in funcs.items():
        paths = fn(rows)
        print(f"[{name}] → {paths.get('png', 'N/A')}")


if __name__ == "__main__":
    main()
