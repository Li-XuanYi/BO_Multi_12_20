from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images" / "summary_compare_style"
EXP_RECORDS = ROOT.parent / "experiment_records"
ADAPTIVE_REPORT = ROOT / "source_reports" / "adaptive4_5seeds_50iter_report.json"
PAIRED_REPORT = ROOT / "source_reports" / "warmstart_vs_llmbo_paired_5seeds_50iter_report.json"

GROUPS = [
    {
        "key": "baseline",
        "label": "Baseline",
        "source": "adaptive",
        "variant": "baseline",
        "color": "#4e8fb5",
        "marker": "o",
    },
    {
        "key": "warmstart",
        "label": "WarmStart",
        "source": "adaptive",
        "variant": "baseline_warmstart",
        "color": "#2ecc71",
        "marker": "s",
    },
    {
        "key": "llm_region",
        "label": "LLM_Region",
        "source": "adaptive",
        "variant": "baseline_llm_region",
        "color": "#e67e22",
        "marker": "^",
    },
    {
        "key": "llmbo",
        "label": "LLMBO",
        "source": "paired",
        "variant": "llmbo_mo",
        "color": "#c85d6b",
        "marker": "D",
    },
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_compare_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.9,
            "grid.color": "#cfcfcf",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _records_by_variant(report: Mapping[str, Any]) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    by_variant: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for row in report.get("records", []):
        if row.get("status") != "ok":
            continue
        variant = str(row.get("variant"))
        seed = int(row.get("seed"))
        by_variant.setdefault(variant, {})[seed] = row
    return by_variant


def _build_table() -> List[Dict[str, Any]]:
    adaptive = _load_json(ADAPTIVE_REPORT)
    paired = _load_json(PAIRED_REPORT)
    reports = {"adaptive": adaptive, "paired": paired}
    record_maps = {name: _records_by_variant(report) for name, report in reports.items()}

    baseline_values = [
        float(row["canonical_hv"])
        for _, row in sorted(record_maps["adaptive"]["baseline"].items())
    ]
    baseline_mean = float(np.mean(baseline_values))

    rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        records = record_maps[group["source"]][group["variant"]]
        sorted_records = [row for _, row in sorted(records.items())]
        values = [float(row["canonical_hv"]) for row in sorted_records]
        mean = float(np.mean(values))
        std = float(np.std(values))
        delta = None if group["key"] == "baseline" else mean - baseline_mean
        wins = None
        if group["key"] != "baseline":
            wins = int(sum(v > b for v, b in zip(values, baseline_values)))
        rows.append(
            {
                **group,
                "values": values,
                "mean": mean,
                "std": std,
                "delta_vs_baseline": delta,
                "wins_vs_baseline": wins,
                "n": len(values),
                "summary_paths": [str(row["summary_path"]) for row in sorted_records if row.get("summary_path")],
            }
        )
    return rows


def _write_csv(rows: List[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "mean_canonical_hv", "std", "vs_baseline", "wins_vs_baseline", "values"])
        for row in rows:
            wins = "" if row["wins_vs_baseline"] is None else f"{row['wins_vs_baseline']}/{row['n']}"
            delta = "" if row["delta_vs_baseline"] is None else f"{row['delta_vs_baseline']:.9f}"
            writer.writerow(
                [
                    row["label"],
                    f"{row['mean']:.9f}",
                    f"{row['std']:.9f}",
                    delta,
                    wins,
                    ";".join(f"{v:.9f}" for v in row["values"]),
                ]
            )


def _plot_hv_summary(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(rows))
    means = np.array([row["mean"] for row in rows], dtype=float)
    stds = np.array([row["std"] for row in rows], dtype=float)
    colors = [row["color"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        alpha=0.86,
        edgecolor="#555555",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#444444"},
    )
    rng = np.random.default_rng(523)
    for i, row in enumerate(rows):
        vals = np.asarray(row["values"], dtype=float)
        jitter = rng.normal(0.0, 0.045, size=len(vals))
        ax.scatter(
            np.full_like(vals, x[i], dtype=float) + jitter,
            vals,
            s=42,
            marker=row["marker"],
            color="#ffffff",
            edgecolor="#222222",
            linewidth=0.65,
            zorder=4,
        )
        ax.text(
            bars[i].get_x() + bars[i].get_width() / 2,
            means[i] + stds[i] + 0.003,
            f"{means[i]:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows], rotation=12, ha="right")
    ax.set_ylabel("Canonical HV")
    ax.set_title("Ablation Study: Component Contributions")
    ax.grid(True, axis="y", alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylim(0.365, max(means + stds) + 0.014)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=row["color"], lw=7, label=row["label"])
            for row in rows
        ],
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
    )
    fig.tight_layout()

    png = OUT_DIR / "ablation523_4group_canonical_hv_compare_style.png"
    pdf = OUT_DIR / "ablation523_4group_canonical_hv_compare_style.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _plot_hv_box(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [row["label"] for row in rows]
    colors = [row["color"] for row in rows]
    data = [np.asarray(row["values"], dtype=float) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(1, len(data) + 1)
    box = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.9)
    for key in ["whiskers", "caps", "medians"]:
        for artist in box[key]:
            artist.set_color("#333333")
            artist.set_linewidth(0.9)

    rng = np.random.default_rng(8409)
    for pos, values, color, row in zip(positions, data, colors, rows):
        offsets = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(
            pos + offsets,
            values,
            s=42,
            marker=row["marker"],
            color=color,
            edgecolor="#222222",
            linewidth=0.35,
            zorder=3,
        )
        ax.text(
            pos,
            float(np.max(values)) + 0.0025,
            f"{row['mean']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#333333",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Canonical HV")
    ax.set_title("Ablation Canonical HV Distribution")
    ax.grid(True, axis="y", alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylim(0.36, 0.415)
    fig.tight_layout()

    png = OUT_DIR / "ablation_canonical_hv_box.png"
    pdf = OUT_DIR / "ablation_canonical_hv_box.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _extract_trace(summary: Mapping[str, Any]) -> np.ndarray:
    trace = summary.get("hv_trace") or []
    values: List[float] = []
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        value = item.get("canonical_hv", item.get("hypervolume_canonical", None))
        if value is None:
            raw = item.get("hypervolume_raw")
            value = item.get("hypervolume") if raw is None else None
        if value is not None:
            values.append(float(value))
    if values:
        return np.asarray(values, dtype=float)
    final_value = summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0))
    return np.asarray([float(final_value)], dtype=float)


def _resolve_summary_path(path_text: str) -> Path | None:
    """Resolve a summary path to work on both Linux and Windows.

    The report may contain Windows paths (D:\\Users\\...). We extract the
    relative suffix after ``experiment_records`` and resolve it locally.
    """
    path = Path(path_text)
    if path.exists():
        return path
    # Normalize separators and find 'experiment_records' anchor
    parts = Path(path_text.replace("\\", "/")).parts
    for i, part in enumerate(parts):
        if part == "experiment_records" and i + 1 < len(parts):
            rel = Path(*parts[i + 1:])
            local = EXP_RECORDS / rel
            if local.exists():
                return local
    return None


def _load_group_traces(row: Mapping[str, Any]) -> List[np.ndarray]:
    traces: List[np.ndarray] = []
    for path_text in row.get("summary_paths", []):
        path = _resolve_summary_path(str(path_text))
        if path is None:
            continue
        traces.append(_extract_trace(_load_json(path)))
    return traces


def _plot_hv_convergence(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    _set_compare_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for row in rows:
        traces = _load_group_traces(row)
        if not traces:
            continue
        min_len = min(len(trace) for trace in traces)
        stack = np.vstack([trace[:min_len] for trace in traces])
        x = np.arange(1, min_len + 1)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(
            x,
            mean,
            color=row["color"],
            linewidth=2.2,
            marker=row["marker"],
            markevery=max(min_len // 7, 1),
            markersize=4.5,
            label=row["label"],
            solid_capstyle="round",
        )
        ax.fill_between(x, mean - std, mean + std, color=row["color"], alpha=0.13, linewidth=0)

    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Canonical HV")
    ax.set_title("Ablation HV Convergence")
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.grid(True, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=False, edgecolor="#777777", handlelength=2.6)
    fig.tight_layout()

    png = OUT_DIR / "ablation_hv_convergence.png"
    pdf = OUT_DIR / "ablation_hv_convergence.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _plot_delta_summary(rows: List[Mapping[str, Any]]) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_compare_style()
    comparison_rows = [row for row in rows if row["key"] != "baseline"]
    x = np.arange(len(comparison_rows))
    deltas = np.array([row["delta_vs_baseline"] for row in comparison_rows], dtype=float)
    colors = [row["color"] for row in comparison_rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    bars = ax.bar(
        x,
        deltas,
        color=colors,
        alpha=0.88,
        edgecolor="#555555",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    for bar, row, delta in zip(bars, comparison_rows, deltas):
        wins = row["wins_vs_baseline"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            delta + 0.00045,
            f"+{delta:.4f}\n{wins}/{row['n']} wins",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in comparison_rows], rotation=12, ha="right")
    ax.set_ylabel("Mean canonical HV gain vs Baseline")
    ax.set_title("Ablation Gain Over Baseline")
    ax.grid(True, axis="y", alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, max(deltas) + 0.0032)
    fig.tight_layout()

    png = OUT_DIR / "ablation523_4group_delta_vs_baseline_compare_style.png"
    pdf = OUT_DIR / "ablation523_4group_delta_vs_baseline_compare_style.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def main() -> None:
    rows = _build_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, OUT_DIR / "ablation523_4group_plot_values.csv")
    artifacts = {
        "hv_summary": _plot_hv_summary(rows),
        "delta_summary": _plot_delta_summary(rows),
        "canonical_hv_box": _plot_hv_box(rows),
        "hv_convergence": _plot_hv_convergence(rows),
        "values_csv": str(OUT_DIR / "ablation523_4group_plot_values.csv"),
    }
    manifest = {
        "figure_style": "Compare_Exp-like: serif font, gray axes, light grid, solid colors, PNG/PDF outputs.",
        "source_reports": {
            "adaptive4": str(ADAPTIVE_REPORT),
            "paired_llmbo": str(PAIRED_REPORT),
        },
        "artifacts": artifacts,
    }
    (OUT_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
