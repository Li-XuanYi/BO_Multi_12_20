from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataBase.database import ObservationDB
from llm.llm_interface import build_llm_interface
from tools.freeze_seed8409_and_plot_profiles import _select_balanced_protocol


def test_database_optimal_protocols_alias_pareto_front() -> None:
    db = ObservationDB()
    db.add_observation(
        theta=np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float),
        objectives=np.array([4200.0, 3.6, 0.80], dtype=float),
        feasible=True,
        source="a",
    )
    db.add_observation(
        theta=np.array([5.0, 4.0, 2.8, 0.20, 0.18], dtype=float),
        objectives=np.array([3600.0, 4.0, 0.70], dtype=float),
        feasible=True,
        source="b",
    )
    db.add_observation(
        theta=np.array([4.2, 3.6, 2.6, 0.22, 0.19], dtype=float),
        objectives=np.array([5000.0, 5.2, 0.95], dtype=float),
        feasible=True,
        source="c",
    )

    optimal = db.get_optimal_protocols()

    assert len(optimal) == db.pareto_size
    assert len(optimal) == 2
    assert db.export_optimal_protocols()[0]["theta"] == optimal[0].theta.tolist()


def test_select_balanced_protocol_prefers_closest_point_to_pareto_ideal() -> None:
    protocols = [
        {
            "theta": [6.0, 5.0, 3.0, 0.4, 0.3],
            "objectives": [2800.0, 7.0, 1.4],
            "source": "bo",
            "iteration": 10,
        },
        {
            "theta": [2.0, 2.0, 2.0, 0.1, 0.1],
            "objectives": [7200.0, 1.3, 0.66],
            "source": "bo",
            "iteration": 11,
        },
        {
            "theta": [3.5, 3.0, 2.4, 0.2, 0.2],
            "objectives": [4300.0, 3.6, 0.52],
            "source": "bo",
            "iteration": 12,
        },
    ]

    chosen = _select_balanced_protocol(protocols)

    assert chosen["theta"] == protocols[2]["theta"]
    assert chosen["selection_method"] == "pareto_balanced_distance_to_ideal"
    assert float(chosen["selection_score"]) >= 0.0


def test_region_preference_replay_reads_summary_telemetry(tmp_path: Path) -> None:
    replay_path = tmp_path / "summary.json"
    replay_path.write_text(
        json.dumps(
            {
                "region_lift_telemetry": [
                    {"preference": {"kind": "none", "parser_status": "inactive_window_skipped"}},
                    {
                        "preference": {
                            "kind": "point",
                            "coordinate_space": "raw",
                            "preference_direction": "promising",
                            "point": {"I1": 4.1, "I2": 3.3, "I3": 2.4, "dSOC1": 0.2, "dSOC2": 0.18},
                            "confidence": 0.8,
                            "preference_type": "balanced",
                            "reason": "replayed",
                            "risk_flags": [],
                            "parser_status": "ok",
                        }
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    llm = build_llm_interface(
        param_bounds={
            "I1": (2.0, 6.0),
            "I2": (2.0, 5.0),
            "I3": (2.0, 3.0),
            "dSOC1": (0.1, 0.4),
            "dSOC2": (0.1, 0.3),
        },
        backend="mock",
        region_pref_replay_path=str(replay_path),
    )

    pref0 = llm.query_region_preference({"iteration": 0})
    pref1 = llm.query_region_preference({"iteration": 1})
    pref2 = llm.query_region_preference({"iteration": 2})

    assert pref0.kind == "none"
    assert pref1.kind == "point"
    assert pref1.point is not None
    assert np.isclose(pref1.point["I1"], 4.1)
    assert pref2.kind == "none"
    assert pref2.parser_status == "replay_missing"
