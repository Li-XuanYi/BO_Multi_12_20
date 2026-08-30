from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Box_Fig.plot_hv_box import _extract_metric_values, run_from_config


def test_extract_metric_values_from_report_records() -> None:
    payload = {
        "records": [
            {"seed": 0, "variant": "alpha", "status": "ok", "display_hv": 0.741},
            {"seed": 1, "variant": "beta", "status": "ok", "display_hv": 0.733},
            {"seed": 2, "variant": "alpha", "status": "failed", "display_hv": 0.752},
        ]
    }
    result = _extract_metric_values(payload, metric_key="display_hv", variant="alpha")
    assert np.allclose(result, [0.741])


def test_extract_metric_values_from_values_list() -> None:
    payload = {"values": [0.701, 0.706, 0.709]}
    result = _extract_metric_values(payload, metric_key="display_hv")
    assert np.allclose(result, [0.701, 0.706, 0.709])


def test_run_from_config_writes_outputs(tmp_path: Path) -> None:
    eimo_path = tmp_path / "eimo.json"
    parego_path = tmp_path / "parego.json"
    eimo_path.write_text(json.dumps({"records": [{"display_hv": 0.742}, {"display_hv": 0.743}]}), encoding="utf-8")
    parego_path.write_text(json.dumps({"records": [{"display_hv": 0.739}, {"display_hv": 0.744}]}), encoding="utf-8")

    config_path = tmp_path / "config.json"
    config = {
        "plot": {"figure_size": [3.2, 3.2], "y_label": "HV"},
        "groups": [
            {"label": "EIMO", "path": str(eimo_path), "color": "#D45162"},
            {"label": "ParEGO", "path": str(parego_path), "color": "#2E8BC8"},
        ],
        "output": {"basename": "box_test_output"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outputs = run_from_config(config_path)
    assert Path(outputs["png"]).exists()
    assert Path(outputs["pdf"]).exists()
