"""ParEGO seeds 8410-8413 on Chen2020. Stop when any seed HV < LLMBO baseline."""
import json, logging, statistics, sys, time, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from DataBase.database import ObservationDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parego_seeds")

OUTPUT_ROOT = Path("parego_rerun_chen2020_seed8409")
SEEDS = [8410, 8411, 8412, 8413]
LLMBO_HV = 0.38482555920297173

def run_and_save(seed):
    out = OUTPUT_ROOT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment_preset": "parego_matlab_reference",
        "max_iterations": 50,
        "n_warmstart": 0,
        "n_random_init": 6,
        "n_candidates": 1,
        "n_select": 1,
        "llm_backend": "mock",
        "llm_model": "mock",
        "llm_api_base": "",
        "llm_api_key": "",
        "llm_n_samples": 1,
        "llm_temperature": 0.7,
        "battery_param_set": "Chen2020",
        "warmstart_context_level": "full",
        "warmstart_max_tokens": 2500,
        "warmstart_max_retries": 3,
        "warmstart_temperature": None,
        "w_sample_seed": seed,
        "init_seed": seed,
        "checkpoint_dir": str(out / "checkpoints"),
        "checkpoint_every": 9999,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "enable_warmstart_portfolio": False,
        "target_transform_mode": "none",
    }

    t0 = time.perf_counter()
    opt = BayesOptimizer(config=cfg)
    db = opt.run()

    display_hv = db.compute_hypervolume()
    raw_hv = db.compute_hypervolume_raw()
    canonical_hv = raw_hv / db.hv_max if db.hv_max > 1e-12 else 0.0

    summary = {
        "algorithm": "parego_matlab_reference",
        "seed": seed,
        "param_set": "Chen2020",
        "n_total": db.size,
        "n_feasible": db.n_feasible,
        "pareto_size": db.pareto_size,
        "canonical_hv": canonical_hv,
        "display_hv": display_hv,
        "hypervolume_raw": raw_hv,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    db.save(str(out / "database.json"))

    elapsed = time.perf_counter() - t0
    return canonical_hv, elapsed

results = []
for seed in SEEDS:
    logger.info(f"=== ParEGO seed={seed} START ===")
    try:
        hv, elapsed = run_and_save(seed)
        delta = hv - LLMBO_HV
        results.append({"seed": seed, "hv": hv, "delta": delta, "elapsed": elapsed})
        logger.info(f"seed={seed} DONE | HV={hv:.6f} | delta={delta:+.6f} | {elapsed:.0f}s")

        if delta < 0:
            print(f"\n{'='*60}")
            print(f"STOPPING: seed={seed} ParEGO HV={hv:.6f} < LLMBO-MO HV={LLMBO_HV:.6f}")
            print(f"ParEGO UNDERPERFORMS by {abs(delta):.4f} ({abs(delta)/LLMBO_HV*100:.1f}%)")
            print(f"{'='*60}")
            break
        else:
            print(f"seed={seed}: ParEGO HV={hv:.6f} >= LLMBO-MO, continuing...")
    except Exception as e:
        logger.error(f"seed={seed} FAILED: {e}")
        traceback.print_exc()
        results.append({"seed": seed, "hv": None, "delta": None, "error": str(e)})

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"  LLMBO-MO baseline: {LLMBO_HV:.6f}")
for r in results:
    if r["hv"] is not None:
        print(f"  ParEGO seed={r['seed']}: {r['hv']:.6f}  delta={r['delta']:+.6f}")
    else:
        print(f"  ParEGO seed={r['seed']}: FAILED")
if results:
    hvs = [r["hv"] for r in results if r["hv"] is not None]
    if hvs:
        print(f"  Mean of completed: {statistics.mean(hvs):.6f}")
print(f"{'='*60}")
