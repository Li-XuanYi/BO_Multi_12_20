from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.llm_interface import build_llm_interface
from llm.rerank_prompt import render_candidate_rerank_prompt
from llm.warmstart_prompt import PLACEHOLDER_PATTERN
from llmbo.acquisition import build_ei_candidate_pool, select_topm_for_rerank
from llmbo.rerank import CandidateInfo, RerankOutput, RerankState, compute_online_gate, rerank_topm_with_llm
from utils.constants import DEFAULT_BOUNDS


def _make_state() -> RerankState:
    return RerankState(
        iter_id=3,
        w_vec=[0.55, 0.30, 0.15],
        tau_t=0.210,
        scalar_best=0.182,
        hv_current=0.365,
        hv_gain_recent_mean=0.012,
        violation_rate_recent=0.10,
        llm_uncertainty_recent=0.15,
        safe_margin_summary={"near_safe": 2, "recent_failures": 1},
        history_summary=[{"iteration": 1, "canonical_hv": 0.31, "pareto_size": 4}],
    )


def _make_candidates() -> list[CandidateInfo]:
    return [
        CandidateInfo(idx=0, x=[4.9, 4.0, 2.8, 0.20, 0.16], mu_fw=0.19, sigma_fw=0.03, ei=0.12, rank_by_ei=1),
        CandidateInfo(idx=1, x=[4.0, 3.4, 2.4, 0.26, 0.20], mu_fw=0.24, sigma_fw=0.04, ei=0.08, rank_by_ei=2),
        CandidateInfo(idx=2, x=[3.2, 2.8, 2.1, 0.34, 0.26], mu_fw=0.33, sigma_fw=0.05, ei=0.01, rank_by_ei=3),
    ]


def test_candidate_rerank_prompt_resolves_placeholders() -> None:
    prompt = render_candidate_rerank_prompt(
        state=_make_state(),
        candidates=_make_candidates()[:2],
        param_bounds=DEFAULT_BOUNDS,
        scalarization_formula="f_w = max_i(w_i * gap_i) + 0.05 * sum_i(w_i * gap_i)",
    )

    assert "Candidate shortlist:" in prompt
    assert "q_good" in prompt
    assert "weight vector" in prompt
    assert not PLACEHOLDER_PATTERN.findall(prompt)


def test_build_ei_candidate_pool_and_topm() -> None:
    X_pool = np.array(
        [
            [4.9, 4.0, 2.8, 0.20, 0.16],
            [4.0, 3.4, 2.4, 0.26, 0.20],
            [3.2, 2.8, 2.1, 0.34, 0.26],
        ],
        dtype=float,
    )
    candidates = build_ei_candidate_pool(
        X_pool,
        mu=np.array([0.19, 0.24, 0.33], dtype=float),
        sigma=np.array([0.03, 0.04, 0.05], dtype=float),
        ei=np.array([0.12, 0.08, 0.01], dtype=float),
        theta_best=np.array([4.8, 3.9, 2.7, 0.21, 0.17], dtype=float),
    )

    topm = select_topm_for_rerank(candidates, top_m=2, min_ei=0.02)

    assert len(candidates) == 3
    assert candidates[0].rank_by_ei == 1
    assert len(topm) == 2
    assert [item.idx for item in topm] == [0, 1]


def test_compute_online_gate_penalizes_violation_and_uncertainty() -> None:
    high_gate = compute_online_gate(
        hv_gain_recent_mean=0.02,
        violation_rate_recent=0.0,
        llm_uncertainty_recent=0.1,
        lambda_max=0.6,
        a=4.0,
        b=3.0,
        c=2.0,
    )
    low_gate = compute_online_gate(
        hv_gain_recent_mean=-0.01,
        violation_rate_recent=0.4,
        llm_uncertainty_recent=0.9,
        lambda_max=0.6,
        a=4.0,
        b=3.0,
        c=2.0,
    )

    assert 0.0 <= low_gate.g_value <= 0.6
    assert 0.0 <= high_gate.g_value <= 0.6
    assert high_gate.g_value > low_gate.g_value


def test_rerank_prefers_high_q_good_when_gate_positive() -> None:
    rerank_result = rerank_topm_with_llm(
        topm_candidates=_make_candidates()[:2],
        llm_outputs=[
            RerankOutput(idx=0, q_good=0.20, confidence=0.7, rationale_short="too risky"),
            RerankOutput(idx=1, q_good=0.90, confidence=0.8, rationale_short="better tradeoff"),
        ],
        gate_value=0.6,
        score_mode="log_ei_plus_gate",
        n_select=1,
    )

    assert rerank_result["selected_indices"] == [1]
    assert rerank_result["entropy_mean"] is not None


def test_llm_interface_rerank_falls_back_under_mock() -> None:
    llm = build_llm_interface(
        DEFAULT_BOUNDS,
        backend="mock",
        n_samples=1,
        temperature=0.0,
    )

    outputs = llm.score_candidate_goodness(_make_state(), _make_candidates()[:2])

    assert len(outputs) == 2
    assert [item.idx for item in outputs] == [0, 1]
    assert outputs[0].q_good > outputs[1].q_good
