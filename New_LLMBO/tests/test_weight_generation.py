from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmbo.optimizer import generate_riesz_weight_set, is_usable_simplex_weight_set
from llmbo.riesz_cache import _make_cache_key, load_or_generate_riesz


def _unique_rows(W: np.ndarray) -> int:
    return int(np.unique(np.round(np.asarray(W, dtype=float), 8), axis=0).shape[0])


def test_riesz_weight_generation_does_not_collapse_to_vertices() -> None:
    W = generate_riesz_weight_set(
        n_obj=3,
        n_div=10,
        s=2.0,
        n_iter=300,
        lr=5e-3,
        seed=42,
    )

    assert W.shape == (66, 3)
    assert is_usable_simplex_weight_set(W, n_obj=3)
    assert _unique_rows(W) >= 33
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-8)


def test_riesz_cache_rejects_degenerate_legacy_cache(tmp_path: Path) -> None:
    n_obj = 3
    n_div = 10
    s = 2.0
    n_iter = 0
    lr = 5e-3
    seed = 42
    cache_key = _make_cache_key(n_obj, n_div, s, n_iter, lr, seed)
    cache_path = tmp_path / f"riesz_{cache_key}.pkl"

    bad = np.vstack(
        [
            np.tile([0.9803921568627451, 0.00980392156862745, 0.00980392156862745], (22, 1)),
            np.tile([0.00980392156862745, 0.9803921568627451, 0.00980392156862745], (22, 1)),
            np.tile([0.00980392156862745, 0.00980392156862745, 0.9803921568627451], (22, 1)),
        ]
    )
    with cache_path.open("wb") as f:
        pickle.dump(bad, f, protocol=pickle.HIGHEST_PROTOCOL)

    W = load_or_generate_riesz(
        n_obj=n_obj,
        n_div=n_div,
        s=s,
        n_iter=n_iter,
        lr=lr,
        seed=seed,
        cache_dir=str(tmp_path),
    )

    assert W.shape == (66, 3)
    assert is_usable_simplex_weight_set(W, n_obj=3)
    assert _unique_rows(W) >= 33
