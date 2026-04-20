from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmbo.proposal import ProposalBlendConfig, ProposalTrainingRecord, WeightedGMMSampler
from utils.constants import DEFAULT_BOUNDS, LLM_SAFE_DSOC_SUM_MAX


def _make_records() -> list[ProposalTrainingRecord]:
    cluster_a = [
        np.array([5.6, 4.7, 2.8, 0.20, 0.18]),
        np.array([5.4, 4.5, 2.7, 0.22, 0.17]),
        np.array([5.5, 4.6, 2.9, 0.21, 0.19]),
        np.array([5.7, 4.8, 2.8, 0.19, 0.18]),
        np.array([5.3, 4.4, 2.7, 0.23, 0.16]),
    ]
    cluster_b = [
        np.array([3.2, 2.9, 2.2, 0.30, 0.22]),
        np.array([3.0, 2.8, 2.1, 0.32, 0.21]),
        np.array([3.1, 2.7, 2.0, 0.29, 0.23]),
        np.array([3.3, 2.9, 2.2, 0.31, 0.20]),
        np.array([3.4, 3.0, 2.3, 0.28, 0.24]),
    ]
    records: list[ProposalTrainingRecord] = []
    for idx, theta in enumerate(cluster_a + cluster_b):
        improvement = 0.12 if idx < len(cluster_a) else 0.02
        records.append(
            ProposalTrainingRecord(
                theta=theta,
                scalar_y=float(0.25 + 0.03 * idx),
                improvement=improvement,
                feasible=True,
                near_constraint_penalty=max(0.0, float(theta[3] + theta[4] - 0.65)),
                monotone_penalty=0.0,
                source="test",
                iteration=idx // 2,
                weight=float(1e-3 + improvement),
            )
        )
    return records


def test_weighted_gmm_sampler_fits_and_samples_within_bounds() -> None:
    sampler = WeightedGMMSampler(
        DEFAULT_BOUNDS,
        n_components=3,
        min_train_size=6,
        covariance_floor=1e-3,
        elite_fraction=0.4,
        blend=ProposalBlendConfig(n_proposal=12, proposal_mix_local=0.25, proposal_mix_global=0.75),
        safe_dsoc_sum_max=LLM_SAFE_DSOC_SUM_MAX,
    )
    summary = sampler.fit(_make_records())

    assert summary["ready"] is True
    samples = sampler.sample(
        n=12,
        rng=np.random.default_rng(7),
        center=np.array([5.5, 4.6, 2.8, 0.20, 0.18]),
    )

    assert samples.shape == (12, 5)
    lo = np.array([DEFAULT_BOUNDS[k][0] for k in ["I1", "I2", "I3", "dSOC1", "dSOC2"]], dtype=float)
    hi = np.array([DEFAULT_BOUNDS[k][1] for k in ["I1", "I2", "I3", "dSOC1", "dSOC2"]], dtype=float)
    assert np.all(samples >= lo - 1e-12)
    assert np.all(samples <= hi + 1e-12)
    assert np.all(samples[:, 3] + samples[:, 4] <= LLM_SAFE_DSOC_SUM_MAX + 1e-9)


def test_weighted_gmm_sampler_requires_enough_records() -> None:
    sampler = WeightedGMMSampler(DEFAULT_BOUNDS, min_train_size=8)
    summary = sampler.fit(_make_records()[:4])

    assert summary["ready"] is False
    assert sampler.is_ready() is False
    assert summary["reason"] == "insufficient_records"


def test_weighted_gmm_sampler_scores_elite_region_higher() -> None:
    sampler = WeightedGMMSampler(DEFAULT_BOUNDS, min_train_size=6, elite_fraction=0.4)
    sampler.fit(_make_records())

    elite_point = np.array([[5.5, 4.6, 2.8, 0.21, 0.18]], dtype=float)
    weak_point = np.array([[2.2, 2.1, 2.0, 0.38, 0.26]], dtype=float)

    elite_score = sampler.score(elite_point)[0]
    weak_score = sampler.score(weak_point)[0]

    assert elite_score > weak_score
