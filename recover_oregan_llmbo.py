"""Recovery: Run LLMBO-MO only for ORegan2022 (ParEGO already complete)."""
from __future__ import annotations
import json, logging, statistics, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llmbo.optimizer import BayesOptimizer
from DataBase.database import ObservationDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("oregan_recovery")

SEEDS = [8500, 8501, 8502, 8503, 8504]
N_WARMSTART = 3
N_RANDOM_INIT_LLMBO = 3
N_ITERATIONS = 50
PARAM_SET = "ORegan2022"

OUTPUT_BASE = Path("multi_param_exp")
OUT = OUTPUT_BASE / PARAM_SET / "llmbo"

LLM_CONFIG = {
    "llm_backend": "openai",
    "llm_model": "deepseek-v4-flash",
    "llm_api_base": "https://api.deepseek.com",
    "llm_api_key": "sk-9538336f41ce46ae8758f68fde5bebf2",
    "llm_n_samples": 1,
    "llm_temperature": 0.3,
    "warmstart_temperature": 0.3,
    "warmstart_max_retries": 3,
    "warmstart_max_tokens": 2500,
    "warmstart_context_level": "full",
    "region_preference_max_tokens": 4096,
    "region_preference_prompt_version": "default",
}

def build_config(seed):
    n_iter = 56 - N_WARMSTART - N_RANDOM_INIT_LLMBO
    return {
        "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
        "max_iterations": n_iter,
        "n_warmstart": N_WARMSTART,
        "n_random_init": N_RANDOM_INIT_LLMBO,
        "n_candidates": 15, "n_select": 1,
        **LLM_CONFIG,
        "battery_param_set": PARAM_SET,
        "w_sample_seed": seed, "init_seed": seed,
        "enable_iterative_guidance": True,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": True,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": True,
        "enable_warmstart_portfolio": True,
        "warmstart_pool_size": 16,
        "warmstart_cache_path": None,
        "warmstart_cache_mode": "read_write",
        "target_transform_mode": "none",
        "objective_preprocess_mode": "minmax",
        "weight_strategy": "riesz_relaxed_cycle",
        "weight_count": 30,
        "region_lift_mode": "heuristic_correlation",
        "region_lift_control_mode": "none",
        "region_lift_external_influence_mode": "force_pool",
        "region_lift_include_raw_candidates": False,
        "region_lift_lambda_max": 0.20,
        "region_lift_n_anchors": 64,
        "region_lift_active_until": 16,
        "region_lift_min_width": 0.03,
        "region_lift_max_width": 0.80,
        "region_lift_trust_init": 0.7,
        "region_lift_anchor_weighting": "ei_softmax",
        "region_lift_anchor_temperature": 0.35,
        "region_lift_require_inside": True,
        "region_lift_candidate_oversample": 16,
        "region_lift_point_current_probe_levels": 3,
        "region_lift_point_current_probe_keep": 2,
        "region_lift_dsoc_margin": 0.01,
        "ei_n_external_restarts": 32,
        "region_lift_lgbo_shift_source": "posterior_covariance",
        "region_lift_lgbo_min_variance": 1e-12,
        "checkpoint_dir": str(OUT / f"seed{seed}" / "checkpoints"),
        "checkpoint_every": 9999,
    }

def save_results(optimizer, out_dir, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = optimizer.database
    optimizer.save_results(str(out_dir))
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

results = []
for seed in SEEDS:
    logger.info(f"\n{'='*50}\nLLMBO-MO {PARAM_SET} seed={seed}\n{'='*50}")
    t0 = time.perf_counter()
    try:
        cfg = build_config(seed)
        optimizer = BayesOptimizer(config=cfg)
        optimizer.run()
        out_dir = OUT / f"seed{seed}"
        s = save_results(optimizer, out_dir, seed)
        s["method"] = "llmbo"
        s["param_set"] = PARAM_SET
        results.append(s)
        logger.info(f"DONE seed={seed}: HV={s.get('canonical_hv', '?'):.4f} ({time.perf_counter()-t0:.0f}s)")
    except Exception as e:
        logger.error(f"FAILED seed={seed}: {e}")
        traceback.print_exc()

# Summary
hvs = [r.get("canonical_hv", 0) for r in results]
if hvs:
    print(f"\nLLMBO-MO {PARAM_SET}: {statistics.mean(hvs):.4f} +- {statistics.stdev(hvs):.4f}")
    print(f"Per-seed: {[f'{h:.4f}' for h in hvs]}")
