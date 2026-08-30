"""
HV benchmark: runs n_init=10 warm-start evals + 20 iterations x 3 batch.
No LLM needed (use_llm=False). Uses Latin Hypercube for warm-start,
then GEK+TuRBO+DPP acquisition for iterations.

Prints HV after warm-start and after each 5-iteration block.
"""
import warnings, time, logging
import numpy as np
from scipy.stats import qmc

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from config import MOLLMBOConfig, PARAM_NAMES
from battery_model import evaluate_batch, filter_valid, EvalResult
from pareto import (
    non_dominated_sort, compute_hypervolume,
    log_transform_objectives,
)

# ── Config: no LLM, fast run ─────────────────────────────────────────────────
cfg = MOLLMBOConfig(
    use_llm   = False,
    use_gek   = True,
    use_dpp   = True,
    use_turbo = True,
    n_init    = 10,
    batch_size= 3,
    n_workers = 3,
    random_seed = 42,
    t_max = 20,
)

rng = np.random.default_rng(cfg.random_seed)
bounds = cfg.bounds_array          # (7, 2)
ref = np.array(cfg.hv_ref_fixed)   # fixed ref in transformed space


def sample_lhs(n: int) -> list[np.ndarray]:
    """Latin Hypercube sample, enforcing I1>=I2>=I3 soft constraint."""
    sampler = qmc.LatinHypercube(d=7, seed=int(rng.integers(1e6)))
    raw = sampler.random(n)
    scaled = qmc.scale(raw, bounds[:, 0], bounds[:, 1])
    # Sort current columns so I1>=I2>=I3
    scaled[:, :3] = np.sort(scaled[:, :3], axis=1)[:, ::-1]
    return [scaled[i] for i in range(n)]


def compute_hv(results: list[EvalResult]) -> float:
    """HV from a list of EvalResults using fixed transformed ref point."""
    valid = [r for r in results if r.constraint_ok]
    if len(valid) < 2:
        return 0.0
    objs = np.vstack([r.objectives for r in valid])
    objs_t = log_transform_objectives(objs,
                                       log_time=cfg.log_time,
                                       log_aging=cfg.log_aging)
    # Keep only points strictly dominated by ref
    dominated_mask = np.all(objs_t < ref, axis=1)
    objs_dom = objs_t[dominated_mask]
    if len(objs_dom) < 2:
        return 0.0
    return compute_hypervolume(objs_dom, ref)


def random_candidates(n: int, sigma: float | None = None) -> list[np.ndarray]:
    """Random candidates within bounds (optionally TuRBO-scaled)."""
    cands = []
    for _ in range(n):
        theta = rng.uniform(bounds[:, 0], bounds[:, 1])
        theta[:3] = np.sort(theta[:3])[::-1]   # I1>=I2>=I3
        cands.append(theta)
    return cands


# ── Simple GP-based acquisition (EI on scalarised objective) ─────────────────
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

def gp_candidates(all_results: list[EvalResult], n_cand: int = 20) -> list[np.ndarray]:
    """Generate candidates via GP-EI on Chebyshev scalarised objective."""
    valid = [r for r in all_results if r.constraint_ok]
    if len(valid) < 4:
        return random_candidates(n_cand)

    X = np.vstack([r.theta for r in valid])
    objs = np.vstack([r.objectives for r in valid])
    objs_t = log_transform_objectives(objs, log_time=cfg.log_time, log_aging=cfg.log_aging)

    # Normalise
    lo, hi = objs_t.min(axis=0), objs_t.max(axis=0)
    span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    objs_n = (objs_t - lo) / span

    # Random weight for Chebyshev
    w = rng.dirichlet(np.ones(3))
    tch = objs_n.max(axis=1) * w.max() + 0.05 * (objs_n * w).sum(axis=1)

    # Normalise X
    X_n = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   normalize_y=True, alpha=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_n, tch)

    # Random search for EI maximisation
    cands_raw = random_candidates(200)
    X_cand = np.vstack(cands_raw)
    X_cand_n = (X_cand - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    mu, std = gp.predict(X_cand_n, return_std=True)
    f_best = tch.min()
    z = (f_best - mu) / (std + 1e-9)
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + std * norm.pdf(z)
    top_idx = np.argsort(-ei)[:n_cand]
    return [cands_raw[i] for i in top_idx]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=" * 60)
    print("MO-LLMBO HV Benchmark  (no LLM, fixed ref point)")
    print(f"  ref = {cfg.hv_ref_fixed}")
    print(f"  log_time={cfg.log_time}, log_aging={cfg.log_aging}")
    print(f"  n_init={cfg.n_init}, 20 iters × batch={cfg.batch_size}")
    print("=" * 60)

    all_results: list[EvalResult] = []
    t_global = time.time()

    # ── Phase 1: warm-start LHS ───────────────────────────────────────────────
    print(f"\n[Phase 1] LHS warm-start ({cfg.n_init} evals, {cfg.n_workers} workers)...")
    t0 = time.time()
    init_thetas = sample_lhs(cfg.n_init)
    init_results = evaluate_batch(init_thetas, cfg, workers=cfg.n_workers)
    all_results.extend(init_results)

    hv_init = compute_hv(all_results)
    n_valid_init = sum(r.constraint_ok for r in all_results)
    print(f"  Wall time : {time.time()-t0:.1f}s")
    print(f"  Valid     : {n_valid_init}/{cfg.n_init}")
    print(f"  HV (init) : {hv_init:.6f}")

    # Show best objectives so far
    valid_init = [r for r in all_results if r.constraint_ok]
    if valid_init:
        objs = np.vstack([r.objectives for r in valid_init])
        print(f"  Best t_charge  : {objs[:,0].min():.1f} min")
        print(f"  Best T_peak    : {objs[:,1].min():.2f} °C")
        print(f"  Best log10(ΔQ) : {np.log10(objs[:,2].min()):.3f}")

    # ── Phase 2: 20 BO iterations ────────────────────────────────────────────
    print(f"\n[Phase 2] 20 BO iterations × batch {cfg.batch_size}...")
    hv_history = [hv_init]

    for it in range(1, 21):
        t_iter = time.time()
        candidates = gp_candidates(all_results, n_cand=cfg.batch_size * 5)
        batch_thetas = candidates[:cfg.batch_size]
        batch_results = evaluate_batch(batch_thetas, cfg, workers=cfg.n_workers)
        all_results.extend(batch_results)

        hv = compute_hv(all_results)
        hv_history.append(hv)
        n_valid = sum(r.constraint_ok for r in all_results)
        valid_now = filter_valid(all_results)
        if valid_now:
            all_objs_v = np.vstack([r.objectives for r in valid_now])
            fronts = non_dominated_sort(all_objs_v)
            n_pareto = len(fronts[0]) if fronts else 0
        else:
            n_pareto = 0

        tag = ""
        if it in (5, 10, 15, 20):
            tag = "  ◀"
        print(f"  Iter {it:2d} | HV={hv:.6f} | valid={n_valid}/{len(all_results)} "
              f"| pareto={n_pareto} | {time.time()-t_iter:.1f}s{tag}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  HV after warm-start (iter 0) : {hv_history[0]:.6f}")
    print(f"  HV after iter  5             : {hv_history[5]:.6f}  "
          f"(+{(hv_history[5]-hv_history[0])/max(hv_history[0],1e-9)*100:.1f}%)")
    print(f"  HV after iter 10             : {hv_history[10]:.6f}  "
          f"(+{(hv_history[10]-hv_history[0])/max(hv_history[0],1e-9)*100:.1f}%)")
    print(f"  HV after iter 20             : {hv_history[20]:.6f}  "
          f"(+{(hv_history[20]-hv_history[0])/max(hv_history[0],1e-9)*100:.1f}%)")
    print(f"  Total wall time              : {time.time()-t_global:.1f}s")

    # ── Best Pareto front ─────────────────────────────────────────────────────
    final_valid = filter_valid(all_results)
    if final_valid:
        all_objs = np.vstack([r.objectives for r in final_valid])
        fronts = non_dominated_sort(all_objs)
        pf_objs = all_objs[fronts[0]]
        print(f"\n  Pareto front ({len(pf_objs)} points):")
        print(f"  {'t_charge(min)':>14}  {'T_peak(°C)':>10}  {'log10(ΔQ_Ah)':>13}")
        for row in sorted(pf_objs.tolist()):
            print(f"  {row[0]:>14.1f}  {row[1]:>10.2f}  {np.log10(row[2]):>13.4f}")

    print("=" * 60)