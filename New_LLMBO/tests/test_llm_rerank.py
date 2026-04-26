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
from llmbo.rerank import CandidateInfo, RerankOutput, RerankState, rerank_topm_with_llm
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
        CandidateInfo(
            idx=0,
            x=[4.9, 4.0, 2.8, 0.20, 0.16],
            mu_fw=0.19,
            sigma_fw=0.03,
            ei=0.12,
            rank_by_ei=1,
            log_ei=float(np.log(0.12)),
            log_ei_gap_to_best=0.0,
            dSOC_sum=0.36,
            margin_to_soft_limit=0.29,
            hard_violation_flag=False,
            monotone_flag=True,
        ),
        CandidateInfo(
            idx=1,
            x=[4.0, 3.4, 2.4, 0.26, 0.20],
            mu_fw=0.24,
            sigma_fw=0.04,
            ei=0.115,
            rank_by_ei=2,
            log_ei=float(np.log(0.115)),
            log_ei_gap_to_best=float(np.log(0.12) - np.log(0.115)),
            dSOC_sum=0.46,
            margin_to_soft_limit=0.19,
            hard_violation_flag=False,
            monotone_flag=True,
        ),
        CandidateInfo(
            idx=2,
            x=[3.2, 2.8, 2.1, 0.34, 0.26],
            mu_fw=0.33,
            sigma_fw=0.05,
            ei=0.01,
            rank_by_ei=3,
            log_ei=float(np.log(0.01)),
            log_ei_gap_to_best=float(np.log(0.12) - np.log(0.01)),
            dSOC_sum=0.60,
            margin_to_soft_limit=0.05,
            hard_violation_flag=False,
            monotone_flag=True,
        ),
    ]


def test_candidate_rerank_prompt_resolves_placeholders() -> None:
    prompt = render_candidate_rerank_prompt(
        state=_make_state(),
        candidates=_make_candidates()[:2],
        param_bounds=DEFAULT_BOUNDS,
        scalarization_formula="Lower scalarized objective is better under the current weight vector.",
    )

    assert "Candidate shortlist:" in prompt
    assert "candidate_id" in prompt
    assert "0.70 is hard feasibility" in prompt
    assert "0.65 is the soft safety margin" in prompt
    assert "log_ei_gap_to_best" in prompt
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
    assert candidates[0].log_ei is not None
    assert candidates[0].log_ei_gap_to_best == 0.0
    assert len(topm) == 2
    assert [item.idx for item in topm] == [0, 1]


def test_safe_tiebreak_only_selects_from_eligible_set() -> None:
    rerank_result = rerank_topm_with_llm(
        topm_candidates=_make_candidates(),
        llm_outputs=[
            RerankOutput(idx=0, q_good=0.00, confidence=1.0, rationale_short="plain best"),
            RerankOutput(idx=1, q_good=1.00, confidence=1.0, rationale_short="nearly tied better"),
            RerankOutput(idx=2, q_good=1.00, confidence=0.9, rationale_short="too far in EI"),
        ],
        mode="ei_preserving_tiebreak",
        gate=0.10,
        max_log_ei_gap=0.20,
        max_bonus=0.05,
        q_bad_threshold=0.60,
        min_confidence=0.50,
    )

    assert set(rerank_result["eligible_indices"]) == {0, 1}
    assert rerank_result["selected_indices"] == [1]
    assert 2 not in rerank_result["selected_indices"]


def test_eligible_size_le_one_returns_no_selection() -> None:
    rerank_result = rerank_topm_with_llm(
        topm_candidates=_make_candidates(),
        llm_outputs=[
            RerankOutput(idx=0, q_good=0.90, confidence=0.9, rationale_short="only eligible"),
            RerankOutput(idx=1, q_good=0.10, confidence=0.9, rationale_short="too far"),
            RerankOutput(idx=2, q_good=0.10, confidence=0.9, rationale_short="too far"),
        ],
        mode="ei_preserving_tiebreak",
        gate=0.10,
        max_log_ei_gap=0.01,
        max_bonus=0.05,
        q_bad_threshold=0.60,
        min_confidence=0.50,
    )

    assert rerank_result["selected_indices"] == []
    assert rerank_result["fallback_reason"] == "eligible_too_small"


def test_risk_veto_only_does_not_reward_low_ei_candidate() -> None:
    rerank_result = rerank_topm_with_llm(
        topm_candidates=_make_candidates()[:2],
        llm_outputs=[
            RerankOutput(idx=0, q_good=0.55, confidence=0.9, rationale_short="acceptable"),
            RerankOutput(idx=1, q_good=0.95, confidence=0.9, rationale_short="looks safe"),
        ],
        mode="risk_veto_only",
        gate=0.10,
        max_log_ei_gap=0.20,
        max_bonus=0.05,
        q_bad_threshold=0.60,
        min_confidence=0.50,
    )

    assert rerank_result["selected_indices"] == [0]


def test_low_confidence_is_neutralized() -> None:
    rerank_result = rerank_topm_with_llm(
        topm_candidates=_make_candidates()[:2],
        llm_outputs=[
            RerankOutput(idx=0, q_good=0.10, confidence=0.90, rationale_short="bad"),
            RerankOutput(idx=1, q_good=1.00, confidence=0.20, rationale_short="low confidence"),
        ],
        mode="ei_preserving_tiebreak",
        gate=0.10,
        max_log_ei_gap=0.20,
        max_bonus=0.05,
        q_bad_threshold=0.60,
        min_confidence=0.50,
    )

    row_by_idx = {int(row["idx"]): row for row in rerank_result["rows"]}
    assert row_by_idx[1]["q_effective"] == 0.5
    assert row_by_idx[1]["confidence_effective"] == 0.0
    assert rerank_result["selected_indices"] == [0]


def test_llm_interface_rerank_mock_returns_structured_scores() -> None:
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
