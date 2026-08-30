from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Patero.plot_soh_pareto import _extract_objectives, _filter_pareto, pareto_mask


def test_pareto_mask_filters_dominated_points() -> None:
    points = np.array(
        [
            [2000.0, 2.0, 0.30],
            [2200.0, 2.5, 0.35],
            [2500.0, 3.0, 0.50],
            [2300.0, 2.4, 0.33],
        ]
    )
    mask = pareto_mask(points)
    assert mask.tolist() == [True, False, False, False]


def test_extract_objectives_supports_observation_db_payload() -> None:
    payload = {
        "observations": [
            {"objectives": [2100.0, 2.1, 0.31]},
            {"objectives": [2600.0, 2.8, 0.42]},
        ]
    }
    result = _extract_objectives(payload)
    assert result.shape == (2, 3)
    assert np.allclose(result[0], [2100.0, 2.1, 0.31])


def test_filter_pareto_keeps_two_tradeoff_points() -> None:
    points = np.array(
        [
            [2100.0, 4.8, 1.10],
            [2600.0, 3.1, 0.55],
            [2900.0, 3.6, 0.75],
            [3300.0, 2.2, 0.62],
        ]
    )
    result = _filter_pareto(points)
    assert result.shape == (3, 3)
    assert any(np.allclose(row, [2100.0, 4.8, 1.10]) for row in result)
    assert any(np.allclose(row, [2600.0, 3.1, 0.55]) for row in result)
    assert any(np.allclose(row, [3300.0, 2.2, 0.62]) for row in result)
