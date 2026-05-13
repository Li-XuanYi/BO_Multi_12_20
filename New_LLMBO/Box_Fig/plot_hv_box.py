from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOX_FIG_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BOX_FIG_ROOT / "demo_config.json"
DEFAULT_OUTPUT_DIR = BOX_FIG_ROOT / "output"
DEFAULT_DEMO_DATA_DIR = BOX_FIG_ROOT / "demo_data"

DEFAULT_COLORS = ["#D45162", "#2E8BC8", "#F0A33A", "#5C9E45", "#8A52CC"]


@dataclass
class GroupSpec:
    label: str
    color: str
    metric: str
    value_scale: float = 1.0
    path: Optional[Path] = None
    values: Optional[np.ndarray] = None
    variant: Optional[str] = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    rows: List[float] = []
    for value in values:
        scalar = _to_float(value)
        if scalar is not None:
            rows.append(scalar)
    if not rows:
        raise ValueError("Could not extract any numeric HV values.")
    return np.asarray(rows, dtype=float)


def _extract_metric_values(
    payload: Any,
    *,
    metric_key: str = "display_hv",
    variant: Optional[str] = None,
    ok_only: bool = True,
) -> np.ndarray:
    if isinstance(payload, Mapping):
        if "values" in payload and isinstance(payload["values"], Sequence):
            return _as_float_array(payload["values"])

        if "records" in payload and isinstance(payload["records"], Sequence):
            values: List[float] = []
            for item in payload["records"]:
                if not isinstance(item, Mapping):
                    continue
                if variant is not None and str(item.get("variant", "")) != variant:
                    continue
                status = item.get("status")
                if ok_only and status not in (None, "ok"):
                    continue
                scalar = _to_float(item.get(metric_key))
                if scalar is not None:
                    values.append(scalar)
            if values:
                return np.asarray(values, dtype=float)
            if payload.get("failures"):
                raise ValueError(
                    f"No usable '{metric_key}' records were found. The report only contains failures."
                )

        scalar = _to_float(payload.get(metric_key))
        if scalar is not None:
            return np.asarray([scalar], dtype=float)

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if all(not isinstance(item, Mapping) for item in payload):
            return _as_float_array(payload)

        values = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            if variant is not None and str(item.get("variant", "")) != variant:
                continue
            scalar = _to_float(item.get(metric_key))
            if scalar is not None:
                values.append(scalar)
        if values:
            return np.asarray(values, dtype=float)

    raise ValueError(f"Could not find metric '{metric_key}' in the JSON payload.")


def _load_group_specs(config: Mapping[str, Any], config_dir: Path) -> List[GroupSpec]:
    groups_cfg = config.get("groups", [])
    if not isinstance(groups_cfg, Sequence) or not groups_cfg:
        raise ValueError("Config must contain a non-empty 'groups' list.")

    specs: List[GroupSpec] = []
    for idx, item in enumerate(groups_cfg):
        if not isinstance(item, Mapping):
            raise ValueError("Each group config must be a JSON object.")
        label = str(item.get("label", f"Group {idx + 1}"))
        color = str(item.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]))
        metric = str(item.get("metric", config.get("metric", "display_hv")))
        value_scale = _to_float(item.get("value_scale", config.get("value_scale", 1.0)))
        if value_scale is None:
            raise ValueError(f"Group '{label}' has an invalid 'value_scale'.")
        variant = item.get("variant")
        values = item.get("values")
        path_text = item.get("path")

        if values is not None:
            spec = GroupSpec(
                label=label,
                color=color,
                metric=metric,
                value_scale=value_scale,
                values=_as_float_array(values),
                variant=str(variant) if variant is not None else None,
            )
        elif path_text:
            spec = GroupSpec(
                label=label,
                color=color,
                metric=metric,
                value_scale=value_scale,
                path=_resolve_path(str(path_text), config_dir),
                variant=str(variant) if variant is not None else None,
            )
        else:
            raise ValueError(f"Group '{label}' needs either 'path' or 'values'.")
        specs.append(spec)
    return specs


def _load_group_values(spec: GroupSpec) -> Dict[str, Any]:
    if spec.values is not None:
        values = np.asarray(spec.values, dtype=float)
    elif spec.path is not None:
        payload = _read_json(spec.path)
        values = _extract_metric_values(payload, metric_key=spec.metric, variant=spec.variant)
    else:
        raise ValueError(f"Group '{spec.label}' does not have any data source.")

    if values.size == 0:
        raise ValueError(f"Group '{spec.label}' does not contain any HV values.")
    values = values * spec.value_scale

    return {
        "label": spec.label,
        "color": spec.color,
        "metric": spec.metric,
        "value_scale": spec.value_scale,
        "variant": spec.variant,
        "values": values,
        "path": spec.path,
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 20,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
        }
    )


def _format_tick(value: float, _: int) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def _plot_box(config: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], output_dir: Path) -> Dict[str, str]:
    plot_cfg = config.get("plot", {})
    output_cfg = config.get("output", {})
    figure_size = plot_cfg.get("figure_size", [4.8, 4.0])
    dpi = int(plot_cfg.get("dpi", 300))
    title = str(plot_cfg.get("title", "")).strip()
    y_label = str(plot_cfg.get("y_label", "HV"))
    x_rotation = float(plot_cfg.get("x_rotation", 28.0))
    jitter = float(plot_cfg.get("jitter", 0.055))
    point_size = float(plot_cfg.get("point_size", 38.0))
    box_width = float(plot_cfg.get("box_width", 0.36))
    basename = str(output_cfg.get("basename", "hv_box_comparison"))

    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(float(figure_size[0]), float(figure_size[1])))

    positions = np.arange(1, len(groups) + 1, dtype=float)
    data = [np.asarray(item["values"], dtype=float) for item in groups]
    labels = [str(item["label"]) for item in groups]
    colors = [str(item["color"]) for item in groups]

    artists = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.35},
        whiskerprops={"color": "black", "linewidth": 1.25},
        capprops={"color": "black", "linewidth": 1.25},
        boxprops={"edgecolor": "black", "linewidth": 1.2},
    )

    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(to_rgba(color, 0.16))

    rng = np.random.default_rng(7)
    for pos, values, color in zip(positions, data, colors):
        if len(values) == 1:
            offsets = np.zeros(1, dtype=float)
        else:
            offsets = np.linspace(-jitter, jitter, len(values))
            offsets += rng.normal(loc=0.0, scale=jitter * 0.12, size=len(values))
            offsets = np.clip(offsets, -jitter * 1.15, jitter * 1.15)
        ax.scatter(
            pos + offsets,
            values,
            s=point_size,
            facecolors=to_rgba(color, 0.85),
            edgecolors=to_rgba(color, 1.0),
            linewidths=0.55,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=x_rotation, ha="right", rotation_mode="anchor")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, pad=10)

    if "y_min" in plot_cfg or "y_max" in plot_cfg:
        y_min = float(plot_cfg.get("y_min", np.min(np.concatenate(data))))
        y_max = float(plot_cfg.get("y_max", np.max(np.concatenate(data))))
        ax.set_ylim(y_min, y_max)

    if "y_ticks" in plot_cfg:
        ax.set_yticks([float(value) for value in plot_cfg["y_ticks"]])
    ax.yaxis.set_major_formatter(FuncFormatter(_format_tick))

    ax.grid(True, axis="both", color="#777777", alpha=0.23)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    ax.tick_params(axis="both", direction="in", length=3.2, width=0.9)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    pdf_path = output_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _write_demo_assets() -> Dict[str, Path]:
    demo_config = {
        "metric": "canonical_hv",
        "value_scale": 0.2,
        "source_notes": [
            "Ecker2015 figure values are canonical_hv multiplied by 0.2.",
            "LLMBO-MO and ParEGO are 5-seed Ecker2015 runs with n_total=56.",
            "NSGA-II, PIMD, and DISK are 5-seed Ecker2015 external baselines with n_total=60.",
        ],
        "plot": {
            "title": "",
            "y_label": "HV",
            "figure_size": [5.9, 4.0],
            "dpi": 300,
            "x_rotation": 28,
            "jitter": 0.055,
            "point_size": 38,
            "box_width": 0.36,
            "y_min": 0.18,
            "y_max": 0.39,
            "y_ticks": [0.18, 0.22, 0.26, 0.30, 0.34, 0.38],
        },
        "groups": [
            {
                "label": "LLMBO-MO",
                "path": "../scalarization_Exp/experiment_records/ecker_llmbo_5seeds_50iter_fixed_2026_05_11/report_5seeds.json",
                "color": "#D45162",
            },
            {
                "label": "ParEGO",
                "path": "../optimized_experiments/parego_ecker_5seeds_56evals_2026_05_11/report.json",
                "color": "#2E8BC8",
            },
            {
                "label": "NSGA-II",
                "path": "../optimized_experiments/nsga2_ecker_5seeds_50evals_2026_05_13/report.json",
                "color": "#F0A33A",
            },
            {
                "label": "PIMD",
                "path": "../optimized_experiments/pimd_ecker_5seeds_50evals_2026_05_13/report_5seeds.json",
                "color": "#5C9E45",
            },
            {
                "label": "DISK",
                "path": "../optimized_experiments/disk_ecker_5seeds_50evals_2026_05_13/report_5seeds.json",
                "color": "#8A52CC",
            },
        ],
        "output": {"basename": "hv_box_ecker2015_scaled"},
    }
    _write_json(DEFAULT_CONFIG_PATH, demo_config)

    return {
        "config": DEFAULT_CONFIG_PATH,
        "data_dir": DEFAULT_DEMO_DATA_DIR,
        "output_dir": DEFAULT_OUTPUT_DIR,
    }


def run_from_config(config_path: Path) -> Dict[str, str]:
    config = _read_json(config_path)
    if not isinstance(config, Mapping):
        raise ValueError("Config JSON must be an object.")
    specs = _load_group_specs(config, config_path.parent)
    groups = [_load_group_values(spec) for spec in specs]
    return _plot_box(config, groups, DEFAULT_OUTPUT_DIR)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a reference-style HV box plot.")
    parser.add_argument("--config", type=str, default=None, help="Path to the JSON config file.")
    parser.add_argument(
        "--make-demo",
        action="store_true",
        help="Write bundled demo assets first, then render the demo figure.",
    )
    args = parser.parse_args(argv)

    if args.make_demo:
        _write_demo_assets()

    config_path = _resolve_path(args.config, PROJECT_ROOT) if args.config else DEFAULT_CONFIG_PATH
    outputs = run_from_config(config_path)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
