from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_region_lift_v2_50iter import _aggregate_variant, _gate_conditions_met


def test_aggregate_variant_tracks_region_diagnostics() -> None:
    records = [
        {
            "seed": 0,
            "variant": "warmstart_region_lifted_gp",
            "status": "ok",
            "display_hv": 1.0,
            "canonical_hv": 0.40,
            "hypervolume_raw": 0.40,
            "hv_violations": 0,
            "effective_lift_accept_count": 0,
            "lift_accept_rate": 0.0,
            "effective_lift_accept_rate": 0.0,
            "plain_candidate_inside_region_count": 2,
            "diagnostic_override_candidate_count": 1,
            "zero_shift_accept_count": 0,
            "fallback_distribution": {"override_disabled": 20},
        },
        {
            "seed": 1,
            "variant": "warmstart_region_lifted_gp",
            "status": "ok",
            "display_hv": 1.1,
            "canonical_hv": 0.42,
            "hypervolume_raw": 0.42,
            "hv_violations": 0,
            "effective_lift_accept_count": 0,
            "lift_accept_rate": 0.0,
            "effective_lift_accept_rate": 0.0,
            "plain_candidate_inside_region_count": 0,
            "diagnostic_override_candidate_count": 2,
            "zero_shift_accept_count": 0,
            "fallback_distribution": {"override_disabled": 20},
        },
    ]

    summary = _aggregate_variant(records)

    assert summary["plain_candidate_inside_region_count_total"] == 2
    assert summary["diagnostic_override_candidate_count_total"] == 3


def test_gate_conditions_require_dual_hv_win_and_nonzero_region_signal() -> None:
    report = {
        "aggregates": {
            "warmstart_region_lifted_gp": {
                "plain_candidate_inside_region_count_total": 1,
                "diagnostic_override_candidate_count_total": 0,
            }
        },
        "comparisons": {
            "warmstart_region_lifted_gp_vs_strict_baseline": {"mean_canonical_hv_delta": 0.01},
            "warmstart_region_lifted_gp_vs_warmstart_plain_ei": {"mean_canonical_hv_delta": 0.00},
        },
    }

    gate = _gate_conditions_met(report)

    assert gate["passed"] is True

    report["comparisons"]["warmstart_region_lifted_gp_vs_strict_baseline"]["mean_canonical_hv_delta"] = -0.01
    gate = _gate_conditions_met(report)
    assert gate["passed"] is False
