#!/usr/bin/env python
"""Tests for main.py config passthrough.

Run:
    python tests/test_main_config.py

Three levels of testing:
  1. build_optimizer_config — Pydantic Config values land in the flat dict
  2. Preset overrides land correctly
  3. BayesOptimizer.cfg — final merged config (DEFAULT_CONFIG + preset + flat)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.schema import Config, create_minimal_config, GPConfig, BOConfig, MOBOConfig
from config.presets import EXPERIMENT_PRESETS
from main import build_optimizer_config

# Part 3 integration — only if sklearn is available
_HAS_SKLEARN = False
try:
    from llmbo.optimizer import BayesOptimizer, DEFAULT_CONFIG
    _HAS_SKLEARN = True
except ImportError:
    pass

PASS = 0
FAIL = 0
SKIP = 0


def _args(preset: str | None = None, mock: bool = True) -> argparse.Namespace:
    return argparse.Namespace(preset=preset, mock=mock)


def _check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def _skip(label: str, reason: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {label}  ({reason})")


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1: Config → flat dict passthrough
# ═══════════════════════════════════════════════════════════════════════════

def test_gp_fields_pass_through() -> None:
    print("\n[1] GP fields pass through")
    cfg = Config(gp=GPConfig(kernel_nu=1.5, alpha=1e-3, normalize_y=False, n_restarts_optimizer=10))
    flat = build_optimizer_config(cfg, _args(), Path("results"))

    _check("kernel_nu", flat["kernel_nu"] == 1.5, f"got {flat['kernel_nu']}")
    _check("gp_alpha", flat["gp_alpha"] == 1e-3, f"got {flat['gp_alpha']}")
    _check("gp_normalize_y", flat["gp_normalize_y"] is False, f"got {flat['gp_normalize_y']}")
    _check("gp_n_restarts", flat["gp_n_restarts_optimizer"] == 10, f"got {flat['gp_n_restarts_optimizer']}")


def test_mobo_fields_pass_through() -> None:
    print("\n[2] MOBO fields pass through")
    cfg = Config(mobo=MOBOConfig(eta=0.10, n_weights=25))
    flat = build_optimizer_config(cfg, _args(), Path("results"))

    _check("eta", flat["eta"] == 0.10, f"got {flat['eta']}")
    _check("weight_count", flat["weight_count"] == 25, f"got {flat['weight_count']}")


def test_bo_warmstart_batch_params_pass_through() -> None:
    print("\n[3] BO warmstart batch params pass through")
    cfg = Config(bo=BOConfig(warmstart_batch_size=30, warmstart_max_llm_attempts=8, warmstart_hv_log_interval=3))
    flat = build_optimizer_config(cfg, _args(), Path("results"))

    _check("batch_size", flat["warmstart_batch_size"] == 30, f"got {flat['warmstart_batch_size']}")
    _check("max_attempts", flat["warmstart_max_attempts"] == 8, f"got {flat['warmstart_max_attempts']}")
    _check("hv_log_interval", flat["warmstart_hv_log_interval"] == 3, f"got {flat['warmstart_hv_log_interval']}")


def test_bo_core_params_pass_through() -> None:
    print("\n[4] BO core params pass through")
    cfg = Config(bo=BOConfig(n_iterations=100, n_warmstart=20, n_random_init=5))
    flat = build_optimizer_config(cfg, _args(), Path("results"))

    _check("max_iterations", flat["max_iterations"] == 100, f"got {flat['max_iterations']}")
    _check("n_warmstart", flat["n_warmstart"] == 20, f"got {flat['n_warmstart']}")
    _check("n_random_init", flat["n_random_init"] == 5, f"got {flat['n_random_init']}")


def test_default_config_values_pass_through() -> None:
    print("\n[5] Default Pydantic Config values pass through")
    cfg = Config()
    flat = build_optimizer_config(cfg, _args(), Path("results"))

    _check("kernel_nu=2.5", flat["kernel_nu"] == 2.5)
    _check("gp_alpha=1e-5", flat["gp_alpha"] == 1e-5)
    _check("gp_normalize_y=True", flat["gp_normalize_y"] is True)
    _check("eta=0.05", flat["eta"] == 0.05)
    _check("weight_count=15", flat["weight_count"] == 15)
    _check("n_random_init=10", flat["n_random_init"] == 10)
    _check("warmstart_batch_size=20", flat["warmstart_batch_size"] == 20)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2: Preset overrides
# ═══════════════════════════════════════════════════════════════════════════

def test_preset_overrides_config_values() -> None:
    print("\n[6] Preset overrides Config values")
    cfg = Config(bo=BOConfig(n_warmstart=100))
    flat = build_optimizer_config(cfg, _args("warmstart_plain_ei"), Path("results"))

    _check("n_warmstart=3 (preset wins)", flat["n_warmstart"] == 3, f"got {flat['n_warmstart']}")


def test_warmstart_plain_ei_preset() -> None:
    print("\n[7] warmstart_plain_ei preset")
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    flat = build_optimizer_config(cfg, _args("warmstart_plain_ei"), Path("results"))

    _check("experiment_preset", flat["experiment_preset"] == "warmstart_plain_ei")
    _check("enable_iterative_guidance=False", flat["enable_iterative_guidance"] is False)
    _check("enable_gp_llm_coupling=False", flat["enable_gp_llm_coupling"] is False)
    _check("enable_acq_prior_coupling=False", flat["enable_acq_prior_coupling"] is False)
    _check("enable_proposal_sampler=False", flat["enable_proposal_sampler"] is False)
    _check("enable_llm_rerank=False", flat["enable_llm_rerank"] is False)
    _check("target_transform_mode=none", flat["target_transform_mode"] == "none")


def test_risk_veto_preset() -> None:
    print("\n[8] warmstart_risk_veto preset")
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    flat = build_optimizer_config(cfg, _args("warmstart_risk_veto"), Path("results"))

    _check("enable_llm_rerank=True", flat["enable_llm_rerank"] is True)
    _check("llm_rerank_mode=risk_veto_only", flat["llm_rerank_mode"] == "risk_veto_only")


def test_parego_baseline_preset() -> None:
    print("\n[9] parego_baseline preset")
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    flat = build_optimizer_config(cfg, _args("parego_baseline"), Path("results"))

    _check("n_warmstart=0", flat["n_warmstart"] == 0)
    _check("n_random_init=6", flat["n_random_init"] == 6)
    _check("weight_strategy", flat["weight_strategy"] == "parego_reference_cycle")
    _check("acquisition_strategy", flat["acquisition_strategy"] == "parego_lcb_de")


def test_region_lifted_gp_preset() -> None:
    print("\n[10] warmstart_region_lifted_gp preset")
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    flat = build_optimizer_config(cfg, _args("warmstart_region_lifted_gp"), Path("results"))

    _check("enable_region_lifted_gp=True", flat["enable_region_lifted_gp"] is True)
    _check("region_lift_external_influence_mode", flat["region_lift_external_influence_mode"] == "diagnostic_only")
    _check("region_lift_anchor_weighting", flat["region_lift_anchor_weighting"] == "ei_softmax")


def test_region_lifted_gp_force_pool_tuned_preset() -> None:
    print("\n[11] warmstart_region_lifted_gp_force_pool_tuned preset")
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    flat = build_optimizer_config(cfg, _args("warmstart_region_lifted_gp_force_pool_tuned"), Path("results"))

    _check("enable_region_lifted_gp=True", flat["enable_region_lifted_gp"] is True)
    _check("region_lift_external_influence_mode=force_pool", flat["region_lift_external_influence_mode"] == "force_pool")
    _check("region_lift_n_anchors=64", flat["region_lift_n_anchors"] == 64)
    _check("region_lift_candidate_oversample=16", flat["region_lift_candidate_oversample"] == 16)


def test_presets_consistency() -> None:
    print("\n[12] Presets in config/presets.py match main.py import")
    _check("EXPERIMENT_PRESETS imported OK", len(EXPERIMENT_PRESETS) >= 10, f"got {len(EXPERIMENT_PRESETS)}")
    _check("warmstart_plain_ei exists", "warmstart_plain_ei" in EXPERIMENT_PRESETS)
    _check("parego_baseline exists", "parego_baseline" in EXPERIMENT_PRESETS)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 3: Integration — BayesOptimizer.cfg  (requires sklearn)
# ═══════════════════════════════════════════════════════════════════════════

def test_custom_gp_params_reach_optimizer() -> None:
    print("\n[13] Custom GP params reach BayesOptimizer.cfg")
    if not _HAS_SKLEARN:
        _skip("all Part 3 tests", "sklearn not available")
        return

    cfg = Config(gp=GPConfig(kernel_nu=1.5, alpha=1e-3, n_restarts_optimizer=10))
    flat = build_optimizer_config(cfg, _args(), Path("/tmp/test_ckpt_main"))
    opt = BayesOptimizer(config=flat)

    _check("kernel_nu=1.5", opt.cfg["kernel_nu"] == 1.5, f"got {opt.cfg['kernel_nu']}")
    _check("gp_alpha=1e-3", opt.cfg["gp_alpha"] == 1e-3, f"got {opt.cfg['gp_alpha']}")
    _check("gp_n_restarts=10", opt.cfg["gp_n_restarts_optimizer"] == 10, f"got {opt.cfg['gp_n_restarts_optimizer']}")


def test_default_values_fill_from_default_config() -> None:
    print("\n[14] Keys not in flat dict use DEFAULT_CONFIG values")
    if not _HAS_SKLEARN:
        return

    cfg = Config()
    flat = build_optimizer_config(cfg, _args(), Path("/tmp/test_ckpt_main"))
    opt = BayesOptimizer(config=flat)

    _check(
        "enable_iterative_guidance from DEFAULT",
        opt.cfg["enable_iterative_guidance"] == DEFAULT_CONFIG["enable_iterative_guidance"],
    )
    _check(
        "enable_acq_prior_coupling from DEFAULT",
        opt.cfg["enable_acq_prior_coupling"] == DEFAULT_CONFIG["enable_acq_prior_coupling"],
    )
    _check(
        "acquisition_strategy from DEFAULT",
        opt.cfg["acquisition_strategy"] == DEFAULT_CONFIG["acquisition_strategy"],
    )


def test_preset_plus_custom_gp() -> None:
    print("\n[15] Preset flags + custom GP params coexist in BayesOptimizer.cfg")
    if not _HAS_SKLEARN:
        return

    cfg = Config(gp=GPConfig(kernel_nu=1.5))
    flat = build_optimizer_config(cfg, _args("warmstart_plain_ei"), Path("/tmp/test_ckpt_main"))
    opt = BayesOptimizer(config=flat)

    _check("preset: enable_iterative_guidance=False", opt.cfg["enable_iterative_guidance"] is False)
    _check("config: kernel_nu=1.5", opt.cfg["kernel_nu"] == 1.5)
    _check("default: llm_rerank_mode=none", opt.cfg["llm_rerank_mode"] == "none")


def test_no_preset_uses_default_config_flags() -> None:
    print("\n[16] No preset -> DEFAULT_CONFIG feature flags used (not hardcoded False)")
    if not _HAS_SKLEARN:
        return

    cfg = Config()
    flat = build_optimizer_config(cfg, _args(preset=None), Path("/tmp/test_ckpt_main"))
    opt = BayesOptimizer(config=flat)

    _check(
        "enable_acq_prior_coupling uses DEFAULT",
        opt.cfg["enable_acq_prior_coupling"] == DEFAULT_CONFIG["enable_acq_prior_coupling"],
        f"got {opt.cfg['enable_acq_prior_coupling']}, DEFAULT={DEFAULT_CONFIG['enable_acq_prior_coupling']}",
    )


# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    tests = [
        # Part 1: Config passthrough
        test_gp_fields_pass_through,
        test_mobo_fields_pass_through,
        test_bo_warmstart_batch_params_pass_through,
        test_bo_core_params_pass_through,
        test_default_config_values_pass_through,
        # Part 2: Presets
        test_preset_overrides_config_values,
        test_warmstart_plain_ei_preset,
        test_risk_veto_preset,
        test_parego_baseline_preset,
        test_region_lifted_gp_preset,
        test_region_lifted_gp_force_pool_tuned_preset,
        test_presets_consistency,
        # Part 3: Integration (skipped if no sklearn)
        test_custom_gp_params_reach_optimizer,
        test_default_values_fill_from_default_config,
        test_preset_plus_custom_gp,
        test_no_preset_uses_default_config_flags,
    ]

    print("=" * 60)
    print("test_main_config: Config passthrough verification")
    print("=" * 60)

    for t in tests:
        try:
            t()
        except Exception as exc:
            global FAIL
            FAIL += 1
            print(f"  [ERROR] {t.__name__}: {exc}")

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped, {PASS + FAIL + SKIP} total")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
