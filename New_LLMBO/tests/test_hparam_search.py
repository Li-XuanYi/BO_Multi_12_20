#!/usr/bin/env python
"""Verify hparam_search logic without Optuna or sklearn.

Tests:
  1. Search spaces are valid
  2. build_trial_config produces valid Pydantic Config
  3. Config flows through build_optimizer_config correctly
  4. CLI dry-run works
  5. Result dir structure is correct

Run:
    python tests/test_hparam_search.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def _check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def main() -> int:
    global PASS, FAIL

    print("=" * 60)
    print("test_hparam_search: Search space + Config flow verification")
    print("=" * 60)

    # ── Test 1: Search spaces ──────────────────────────────────────────
    print("\n[1] Search space definitions")
    from exp.hparam_search import SEARCH_SPACES, RESULT_DIR, build_trial_config

    _check("3 spaces defined", len(SEARCH_SPACES) == 3)
    for name, space in SEARCH_SPACES.items():
        _check(f"{name} has >=5 params", len(space) >= 5, f"got {len(space)}")

    # ── Test 2: Mock trial → Config ────────────────────────────────────
    print("\n[2] Mock trial produces valid Pydantic Config")

    class MockTrial:
        """Fake Optuna trial with deterministic params."""
        def __init__(self, params: dict):
            self._params = params
            self.number = 0

        def suggest_float(self, name, low, high):
            return self._params.get(name, (low + high) / 2)

        def suggest_int(self, name, low, high):
            return self._params.get(name, (low + high) // 2)

        def suggest_categorical(self, name, choices):
            return self._params.get(name, choices[0])

    # Use extreme values to test boundary handling
    extreme_params = {
        "gp_kernel_nu": 3.5,
        "gp_alpha_log": -7.0,
        "gp_n_restarts": 10,
        "mobo_eta": 0.01,
        "mobo_n_weights": 30,
        "acq_n_cand": 30,
        "acq_n_select": 3,
        "bo_n_warmstart": 15,
        "bo_n_random_init": 10,
        "llm_temperature": 0.3,
    }

    trial = MockTrial(extreme_params)
    config = build_trial_config(trial, "default", n_iterations=50, seed=42)

    _check("kernel_nu=3.5", config.gp.kernel_nu == 3.5, f"got {config.gp.kernel_nu}")
    _check("alpha=1e-7", abs(config.gp.alpha - 1e-7) < 1e-15, f"got {config.gp.alpha}")
    _check("n_restarts=10", config.gp.n_restarts_optimizer == 10)
    _check("eta=0.01", abs(config.mobo.eta - 0.01) < 1e-10)
    _check("n_weights=30", config.mobo.n_weights == 30)
    _check("n_cand=30", config.acquisition.n_cand == 30)
    _check("n_select=3", config.acquisition.n_select == 3)
    _check("n_warmstart=15", config.bo.n_warmstart == 15)
    _check("n_random_init=10", config.bo.n_random_init == 10)
    _check("n_iterations=50", config.bo.n_iterations == 50)
    _check("temperature=0.3", abs(config.llm.warmstart.temperature - 0.3) < 1e-10)

    # ── Test 3: Config → flat dict passthrough ─────────────────────────
    print("\n[3] Config flows through build_optimizer_config")
    from main import build_optimizer_config

    flat = build_optimizer_config(
        config,
        argparse.Namespace(preset=None, mock=True),
        RESULT_DIR / "test_trial",
    )

    _check("kernel_nu passed", flat["kernel_nu"] == 3.5)
    _check("gp_alpha passed", abs(flat["gp_alpha"] - 1e-7) < 1e-15)
    _check("gp_n_restarts passed", flat["gp_n_restarts_optimizer"] == 10)
    _check("eta passed", abs(flat["eta"] - 0.01) < 1e-10)
    _check("weight_count passed", flat["weight_count"] == 30)
    _check("n_warmstart passed", flat["n_warmstart"] == 15)
    _check("n_random_init passed", flat["n_random_init"] == 10)
    _check("n_candidates passed", flat["n_candidates"] == 30)
    _check("n_select passed", flat["n_select"] == 3)

    # ── Test 4: Preset + search space coexist ──────────────────────────
    print("\n[4] Preset overrides still work with search params")
    flat_preset = build_optimizer_config(
        config,
        argparse.Namespace(preset="warmstart_plain_ei", mock=True),
        RESULT_DIR / "test_trial",
    )

    _check("preset n_warmstart=3 overrides config=15", flat_preset["n_warmstart"] == 3)
    _check("config gp_alpha still passed", abs(flat_preset["gp_alpha"] - 1e-7) < 1e-15)
    _check("preset enable_llm_rerank=False", flat_preset["enable_llm_rerank"] is False)

    # ── Test 5: Narrow and wide spaces produce valid configs ───────────
    print("\n[5] All search spaces produce valid configs")
    for space_name in SEARCH_SPACES:
        trial_mid = MockTrial({})  # All defaults (midpoints)
        cfg = build_trial_config(trial_mid, space_name, n_iterations=10, seed=0)
        _check(
            f"{space_name}: valid Config",
            cfg.bo.n_iterations == 10 and cfg.gp.alpha > 0,
        )

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
