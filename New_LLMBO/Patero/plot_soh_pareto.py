from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATERO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PATERO_ROOT / "demo_config.json"
DEFAULT_OUTPUT_DIR = PATERO_ROOT / "output"
DEFAULT_DEMO_DATA_DIR = PATERO_ROOT / "demo_data"

DEFAULT_COLORS = ["#8E73F6", "#76E1E5", "#E98F75"]
STAR_COLOR = "#FF1F1F"


@dataclass
class GroupSpec:
    label: str
    path: Path
    color: str


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


def _as_objective_row(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 3:
            return None
        try:
            row = np.asarray(value[:3], dtype=float)
        except (TypeError, ValueError):
            return None
        if row.shape == (3,):
            return row
    return None


def _extract_objectives(payload: Any) -> np.ndarray:
    rows: List[np.ndarray] = []

    if isinstance(payload, Mapping):
        if "observations" in payload:
            for item in payload["observations"]:
                if isinstance(item, Mapping):
                    row = _as_objective_row(item.get("objectives"))
                    if row is not None:
                        rows.append(row)
        elif "pareto_front" in payload:
            rows.extend(_extract_objectives(payload["pareto_front"]))
        elif "points" in payload:
            rows.extend(_extract_objectives(payload["points"]))
        elif "objectives" in payload:
            row = _as_objective_row(payload.get("objectives"))
            if row is not None:
                rows.append(row)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            if isinstance(item, Mapping):
                row = _as_objective_row(item.get("objectives"))
                if row is None and "point" in item:
                    row = _as_objective_row(item.get("point"))
                if row is not None:
                    rows.append(row)
            else:
                row = _as_objective_row(item)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise ValueError("Could not find any objective vectors in the JSON payload.")
    return np.vstack(rows)


def dominates(lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-12) -> bool:
    return bool(np.all(lhs <= rhs + atol) and np.any(lhs < rhs - atol))


def pareto_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            if dominates(points[j], points[i]):
                mask[i] = False
                break
    return mask


def _unique_rows(points: np.ndarray, decimals: int = 10) -> np.ndarray:
    rounded = np.round(np.asarray(points, dtype=float), decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return np.sort(indices)


def _filter_pareto(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    unique_indices = _unique_rows(points)
    unique_points = points[unique_indices]
    mask = pareto_mask(unique_points)
    return unique_points[mask]


def _load_group_points(group: GroupSpec) -> Dict[str, Any]:
    raw_payload = _read_json(group.path)
    all_points = _extract_objectives(raw_payload)
    pareto_points = _filter_pareto(all_points)
    return {
        "label": group.label,
        "path": group.path,
        "color": group.color,
        "all_points": all_points,
        "pareto_points": pareto_points,
    }


def _auto_highlights(group_payloads: Sequence[Mapping[str, Any]], count: int = 6) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for payload in group_payloads:
        group_label = str(payload["label"])
        front = np.asarray(payload["pareto_points"], dtype=float)
        if len(front) == 0:
            continue
        front = front[np.argsort(front[:, 0])]
        candidate_indices = {0, len(front) - 1, len(front) // 2}
        for idx in sorted(candidate_indices):
            merged.append(
                {
                    "group": group_label,
                    "objectives": front[idx].tolist(),
                }
            )
    if not merged:
        return []
    merged.sort(key=lambda item: item["objectives"][0])
    if len(merged) <= count:
        chosen = merged
    else:
        target_positions = np.linspace(0, len(merged) - 1, count)
        chosen = []
        used: set[int] = set()
        for target in target_positions:
            idx = int(round(float(target)))
            while idx in used and idx + 1 < len(merged):
                idx += 1
            used.add(idx)
            chosen.append(merged[idx])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    output: List[Dict[str, Any]] = []
    for idx, item in enumerate(chosen):
        output.append(
            {
                "label": letters[idx],
                "group": item["group"],
                "objectives": item["objectives"],
            }
        )
    return output


def _resolve_highlights(
    config: Mapping[str, Any],
    config_dir: Path,
    group_payloads: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    highlight_cfg = config.get("highlight", {})
    if isinstance(highlight_cfg, Mapping) and highlight_cfg.get("path"):
        path = _resolve_path(str(highlight_cfg["path"]), config_dir)
        payload = _read_json(path)
        if not isinstance(payload, Sequence):
            raise ValueError("Highlight file must contain a JSON list.")
        output: List[Dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            row = _as_objective_row(item.get("objectives"))
            if row is None:
                continue
            output.append(
                {
                    "label": str(item.get("label", "")),
                    "group": str(item.get("group", "")),
                    "objectives": row.tolist(),
                }
            )
        if output:
            return output
    auto_count = int(highlight_cfg.get("auto_count", 6)) if isinstance(highlight_cfg, Mapping) else 6
    return _auto_highlights(group_payloads, count=max(1, auto_count))


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.28,
        }
    )


def _plot_3d(
    config: Mapping[str, Any],
    group_payloads: Sequence[Mapping[str, Any]],
    highlights: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    plot_cfg = config.get("plot", {})
    title = str(plot_cfg.get("title", "")).strip()
    x_label = str(plot_cfg.get("x_label", "Charging Time/s"))
    y_label = str(plot_cfg.get("y_label", "Temperature Rise/K"))
    z_label = str(plot_cfg.get("z_label", "Capacity loss/%"))
    x_scale = float(plot_cfg.get("x_scale", 1.0))
    y_scale = float(plot_cfg.get("y_scale", 1.0))
    z_scale = float(plot_cfg.get("z_scale", 1.0))
    elev = float(plot_cfg.get("elev", 18.0))
    azim = float(plot_cfg.get("azim", -122.0))

    _configure_plot_style()
    fig = plt.figure(figsize=(10.8, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    for payload in group_payloads:
        front = np.asarray(payload["pareto_points"], dtype=float)
        if len(front) == 0:
            continue
        point_color = str(payload["color"])
        ax.scatter(
            front[:, 0] * x_scale,
            front[:, 1] * y_scale,
            front[:, 2] * z_scale,
            s=22,
            facecolors=[to_rgba(point_color, alpha=0.18)],
            edgecolors=[to_rgba(point_color, alpha=0.72)],
            linewidths=1.1,
            label=str(payload["label"]),
        )

    for item in highlights:
        row = np.asarray(item["objectives"], dtype=float)
        x_val = row[0] * x_scale
        y_val = row[1] * y_scale
        z_val = row[2] * z_scale
        ax.scatter(
            [x_val],
            [y_val],
            [z_val],
            s=340,
            marker="*",
            c=STAR_COLOR,
            edgecolors=STAR_COLOR,
            linewidths=1.2,
            alpha=0.98,
        )
        label = str(item.get("label", "")).strip()
        if label:
            ax.text(x_val + 75, y_val + 0.1, z_val + 0.03, label, fontsize=16, color="black")

    ax.set_xlabel(x_label, labelpad=18)
    ax.set_ylabel(y_label, labelpad=22)
    ax.set_zlabel(z_label, labelpad=12)
    if title:
        ax.set_title(title, pad=18)
    ax.text2D(0.015, 0.72, z_label, transform=ax.transAxes, rotation=90, va="center")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#666666")
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "patero_soh_pareto_3d.png"
    pdf_path = output_dir / "patero_soh_pareto_3d.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def _build_demo_points(profile_shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows: List[List[float]] = []
    for band in range(6):
        x_vals = np.linspace(2200 + band * 120, 7200 - band * 220, 28)
        x_vals = x_vals + rng.normal(0.0, 45.0, size=x_vals.shape)
        base_temp = 1.55 + 4.7 * np.exp(-(x_vals - 2200.0) / 2100.0)
        base_loss = 0.24 + 0.95 * np.exp(-(x_vals - 2200.0) / 2200.0)
        y_vals = base_temp + 0.16 * band + 0.55 * profile_shift + rng.normal(0.0, 0.03, size=x_vals.shape)
        z_vals = base_loss + 0.06 * band + 0.28 * profile_shift + rng.normal(0.0, 0.015, size=x_vals.shape)
        for x_val, y_val, z_val in zip(x_vals, y_vals, z_vals):
            rows.append([round(float(x_val), 4), round(float(y_val), 4), round(float(z_val), 4)])
            dominated = [
                round(float(x_val + rng.uniform(120.0, 420.0)), 4),
                round(float(y_val + rng.uniform(0.10, 0.50)), 4),
                round(float(z_val + rng.uniform(0.03, 0.13)), 4),
            ]
            rows.append(dominated)
    return np.asarray(rows, dtype=float)


def _demo_observation_payload(label: str, points: np.ndarray) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []
    for idx, row in enumerate(points):
        observations.append(
            {
                "theta": [0.0, 0.0, 0.0, 0.0, 0.0],
                "objectives": row.tolist(),
                "feasible": True,
                "violation": None,
                "source": "demo",
                "iteration": idx // 2,
                "details": {"label": label},
            }
        )
    return {
        "label": label,
        "observations": observations,
    }


def _write_demo_assets() -> Path:
    DEFAULT_DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    groups = [
        ("LLMBO-MO", 1.00, DEFAULT_COLORS[0], 101, "llmbo_mo_database.json"),
        ("ParEGO", 0.72, DEFAULT_COLORS[1], 102, "parego_database.json"),
        ("NSGAII", 0.42, DEFAULT_COLORS[2], 103, "nsgaii_database.json"),
    ]

    highlight_candidates: Dict[str, np.ndarray] = {}
    config_groups: List[Dict[str, str]] = []

    for idx, (label, profile_shift, color, seed, file_name) in enumerate(groups):
        points = _build_demo_points(profile_shift=profile_shift, seed=seed)
        payload = _demo_observation_payload(label, points)
        file_path = DEFAULT_DEMO_DATA_DIR / file_name
        _write_json(file_path, payload)
        config_groups.append(
            {
                "label": label,
                "path": str(file_path.relative_to(PATERO_ROOT)).replace("\\", "/"),
                "color": color,
            }
        )
        highlight_candidates[label] = _filter_pareto(points)

    front_llmbo = highlight_candidates["LLMBO-MO"][np.argsort(highlight_candidates["LLMBO-MO"][:, 0])]
    front_parego = highlight_candidates["ParEGO"][np.argsort(highlight_candidates["ParEGO"][:, 0])]
    front_nsgaii = highlight_candidates["NSGAII"][np.argsort(highlight_candidates["NSGAII"][:, 0])]

    highlights = [
        {"label": "A", "group": "LLMBO-MO", "objectives": front_llmbo[0].tolist()},
        {"label": "B", "group": "LLMBO-MO", "objectives": front_llmbo[-1].tolist()},
        {"label": "C", "group": "ParEGO", "objectives": front_parego[len(front_parego) // 2].tolist()},
        {"label": "D", "group": "NSGAII", "objectives": front_nsgaii[0].tolist()},
        {"label": "E", "group": "NSGAII", "objectives": front_nsgaii[-1].tolist()},
        {"label": "F", "group": "NSGAII", "objectives": front_nsgaii[len(front_nsgaii) // 2].tolist()},
    ]
    highlights_path = DEFAULT_DEMO_DATA_DIR / "highlights.json"
    _write_json(highlights_path, highlights)

    config_payload = {
        "groups": config_groups,
        "plot": {
            "title": "",
            "x_label": "Charging Time/s",
            "y_label": "Temperature Rise/K",
            "z_label": "Capacity loss/%",
            "x_scale": 1.0,
            "y_scale": 1.0,
            "z_scale": 1.0,
            "elev": 18,
            "azim": -122,
        },
        "highlight": {
            "path": str(highlights_path.relative_to(PATERO_ROOT)).replace("\\", "/"),
            "auto_count": 6,
        },
    }
    _write_json(DEFAULT_CONFIG_PATH, config_payload)
    return DEFAULT_CONFIG_PATH


def _load_config(path: Path) -> Dict[str, Any]:
    config = _read_json(path)
    if not isinstance(config, Mapping):
        raise ValueError("Config file must be a JSON object.")
    groups = config.get("groups")
    if not isinstance(groups, Sequence) or not groups:
        raise ValueError("Config must contain a non-empty 'groups' list.")
    return dict(config)


def _build_group_specs(config: Mapping[str, Any], config_dir: Path) -> List[GroupSpec]:
    groups: List[GroupSpec] = []
    raw_groups = config.get("groups", [])
    for idx, item in enumerate(raw_groups):
        if not isinstance(item, Mapping):
            raise ValueError("Each group entry must be a JSON object.")
        label = str(item.get("label", f"group_{idx + 1}"))
        color = str(item.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]))
        if "path" not in item:
            raise ValueError(f"Group '{label}' is missing 'path'.")
        path = _resolve_path(str(item["path"]), config_dir)
        groups.append(GroupSpec(label=label, path=path, color=color))
    return groups


def run_from_config(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    config = _load_config(config_path)
    group_specs = _build_group_specs(config, config_dir=config_dir)
    group_payloads = [_load_group_points(group) for group in group_specs]
    highlights = _resolve_highlights(config, config_dir=config_dir, group_payloads=group_payloads)
    artifacts = _plot_3d(config, group_payloads, highlights, output_dir=output_dir)

    manifest = {
        "config_path": str(config_path),
        "groups": [
            {
                "label": payload["label"],
                "input_path": str(payload["path"]),
                "n_total_points": int(len(payload["all_points"])),
                "n_pareto_points": int(len(payload["pareto_points"])),
            }
            for payload in group_payloads
        ],
        "highlight_points": highlights,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a reference-style 3D Pareto scatter figure from Python JSON experiment results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a plotting config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--make-demo",
        action="store_true",
        help="Write demo JSON files and generate the demo figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if args.make_demo:
        config_path = _write_demo_assets()
    output_dir = args.output_dir.resolve()
    manifest = run_from_config(config_path=config_path, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
