from __future__ import annotations

import numpy as np

from llm.llm_interface import build_llm_interface
from llmbo.warmstart_selector import (
    WarmStartCandidate,
    WarmStartSelectionConfig,
    filter_warmstart_candidates,
    select_warmstart_portfolio,
)
from utils.constants import DEFAULT_BOUNDS


def _cfg(n_select: int = 3) -> WarmStartSelectionConfig:
    return WarmStartSelectionConfig(
        n_select=n_select,
        bounds={k: tuple(v) for k, v in DEFAULT_BOUNDS.items()},
        hard_dsoc_sum_max=0.70,
        soft_dsoc_sum_max=0.65,
        diversity_weight=0.5,
        soft_penalty_weight=0.8,
        monotone_bonus=0.1,
        boundary_probe_limit=1,
    )


def test_filter_rejects_hard_dsoc_violation() -> None:
    valid = np.array([4.0, 3.5, 2.5, 0.30, 0.20])
    invalid = np.array([4.0, 3.5, 2.5, 0.40, 0.30])

    filtered, summary = filter_warmstart_candidates([valid, invalid], _cfg())

    assert len(filtered) == 1
    assert np.allclose(filtered[0].theta, valid)
    assert summary["filtered"]["hard_dsoc"] == 1


def test_selector_prefers_diverse_portfolio_over_near_duplicates() -> None:
    candidates = [
        WarmStartCandidate(np.array([5.6, 4.6, 2.8, 0.20, 0.20]), confidence=0.9),
        WarmStartCandidate(np.array([5.59, 4.59, 2.79, 0.201, 0.199]), confidence=0.88),
        WarmStartCandidate(np.array([2.5, 2.4, 2.1, 0.35, 0.24]), confidence=0.7),
        WarmStartCandidate(np.array([4.0, 3.5, 2.5, 0.25, 0.20]), confidence=0.7),
    ]

    selected, summary = select_warmstart_portfolio(candidates, _cfg(n_select=3))
    selected_arr = np.vstack([item.theta for item in selected])

    assert summary["selected_count"] == 3
    assert np.min(np.linalg.norm(selected_arr[0] - selected_arr[1:], axis=1)) > 0.2


def test_boundary_probe_limit_is_enforced() -> None:
    candidates = [
        WarmStartCandidate(np.array([5.5, 4.5, 2.8, 0.36, 0.30]), confidence=0.95, style="boundary_probe"),
        WarmStartCandidate(np.array([4.8, 3.8, 2.6, 0.37, 0.29]), confidence=0.94, style="boundary_probe"),
        WarmStartCandidate(np.array([3.0, 2.7, 2.2, 0.25, 0.20]), confidence=0.70),
        WarmStartCandidate(np.array([4.0, 3.5, 2.5, 0.22, 0.22]), confidence=0.70),
    ]

    selected, summary = select_warmstart_portfolio(candidates, _cfg(n_select=3))
    boundary_count = sum(float(item.theta[3] + item.theta[4]) >= 0.665 for item in selected)

    assert boundary_count <= 1
    assert summary["boundary_selected"] <= 1


def test_monotone_profile_gets_soft_bonus_without_hard_filtering() -> None:
    monotone = WarmStartCandidate(np.array([4.8, 3.8, 2.6, 0.24, 0.20]), confidence=0.5)
    non_monotone = WarmStartCandidate(np.array([3.8, 4.8, 2.6, 0.24, 0.20]), confidence=0.5)

    selected, _ = select_warmstart_portfolio([non_monotone, monotone], _cfg(n_select=1))

    assert np.allclose(selected[0].theta, monotone.theta)


def test_warmstart_disk_cache_can_replay_selected_portfolio(tmp_path) -> None:
    cache_path = tmp_path / "warmstart_cache.json"
    selected = [
        [5.75, 4.15, 2.55, 0.185, 0.195],
        [2.10, 2.00, 2.00, 0.10, 0.10],
        [3.45, 2.95, 2.00, 0.235, 0.215],
    ]
    cache_path.write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "candidate_pool": [],
                "final_selected": selected,
                "summary": {"method": "test_cache"},
            }
        ),
        encoding="utf-8",
    )

    llm = build_llm_interface(
        DEFAULT_BOUNDS,
        backend="mock",
        warmstart_cache_path=str(cache_path),
        warmstart_cache_mode="read",
        warmstart_cache_use_selected=True,
    )
    points = llm.generate_warmstart_candidates(n=3, batch_size=8, max_attempts=1)
    summary = llm.get_warmstart_summary()

    assert np.allclose(np.vstack(points), np.asarray(selected, dtype=float))
    assert summary["disk_cache"] == "hit_selected"
