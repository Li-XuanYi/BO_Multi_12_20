# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLAMBO-MO**: LLM-Augmented Multi-Objective Bayesian Optimization for battery fast-charging protocol optimization. Optimizes a 3-stage constant-current (CC) charging protocol to minimize charging time, peak temperature, and capacity fade (3 objectives) using a physics-informed GP with LLM guidance at two touchpoints.

## Running the Code

```bash
# Quick demo (5 iterations, no LLM API needed)
python main.py --demo

# Generate a config template, then run
python main.py --generate-template --template-mode full
python main.py --config config.json --verbose

# CLI overrides on top of a config file
python main.py --config config.json --bo.n_iterations=100 --acquisition.n_cand=20
```

## Architecture

The optimization loop lives in `llmbo/optimizer.py` (`BayesOptimizer.run()`). Execution flow:

1. **LLM Touchpoint 1a** — `llm/llm_interface.py` asks the LLM to generate three coupling matrices (W_time, W_temp, W_aging) encoding physics relationships between decision variables.
2. **LLM Touchpoint 1b** — LLM generates `n_warmstart` initial candidates (replaces random initialization).
3. **PyBaMM Evaluation** — `pybamm_simulator.py` runs the SPMe battery model to get objectives: `[time_s, delta_temp_K, aging_%]` (all minimize).
4. **Main BO Loop** (T iterations):
   - Sample a Tchebycheff weight vector `w_vec` (Riesz s-energy from `llmbo/ParEGO.py`)
   - Update Tchebycheff context in `DataBase/database.py`
   - Fit `llmbo/gp_model.py` on scalarized observations
   - **LLM Touchpoint 2** — LLM generates candidates informed by `w_vec` + GP state
   - `llmbo/acquisition.py` evaluates EI × W_charge and selects top candidates
   - Evaluate candidates in PyBaMM, add to database, save checkpoint

### Module Responsibilities

| Module | Role |
|--------|------|
| `main.py` | CLI, config loading, async entry point |
| `config/schema.py` | Pydantic config (11 classes); `Config` is the root |
| `config/load.py` | Loads JSON → env vars → CLI overrides (later = higher priority) |
| `llmbo/optimizer.py` | Orchestrates the entire BO loop |
| `llmbo/gp_model.py` | Physics-informed GP kernel: `k = RBF + γ · ∇Ψᵀ W ∇Ψ` where Ψ is ohmic heat proxy |
| `llmbo/acquisition.py` | EI × W_charge; adaptive μ/σ tracking; stagnation expansion |
| `llmbo/ParEGO.py` | Riesz s-energy weight generation on 2-simplex |
| `llm/llm_interface.py` | Two LLM call functions; falls back to LHS on failure |
| `DataBase/database.py` | Stores observations, computes HV, tracks Pareto front and stagnation |
| `pybamm_simulator.py` | SPMe simulation; returns objectives + feasibility |
| `exp/` | Ablation (V0–V6) and baseline (ParEGO, NSGA-II, etc.) runners |
| `plot/` | Pareto front, HV curve, protocol visualization |

### Decision Variables (θ ∈ ℝ⁵)

| Variable | Range | Meaning |
|----------|-------|---------|
| I1 | [2.0, 6.0] A | Stage 1 current |
| I2 | [2.0, 5.0] A | Stage 2 current |
| I3 | [2.0, 3.0] A | Stage 3 current |
| dSOC1 | [0.10, 0.40] | Stage 1 SOC width |
| dSOC2 | [0.10, 0.30] | Stage 2 SOC width |

Constraint: `dSOC1 + dSOC2 ≤ 0.70`

### Tchebycheff Scalarization

Objectives are log-transformed before scalarization:
- `f̃₁ = log10(time_s)`, `f̃₂ = temp_K`, `f̃₃ = log10(aging_%)`
- `f_tch = max_i(w_i · f̄_i) + η · Σ(w_i · f̄_i)`, η = 0.05

### Interface Contracts

`llmbo/acquisition.py` depends on `GPProtocol` and `DatabaseProtocol` (duck-typed interfaces defined in those modules). Don't change `predict()` / `get_f_min()` signatures without updating both sides.

## Key Design Patterns

- **No global config imports**: `Config` object is passed explicitly through call chains.
- **Gamma annealing**: Physics kernel coupling strength decays over iterations (`llmbo/gp_model.py:GammaAnnealer`).
- **Coupling matrix blend**: `W^(t) = Σ w_i·W_i / Σ w_i` — per-iteration blend of LLM matrices weighted by current `w_vec`.
- **Stagnation expansion**: If no improvement for N iterations, acquisition σ is multiplied by `(1 + ρ)` to encourage exploration.
- **LLM fallback**: Any LLM call failure silently falls back to LHS + physics heuristics.

## Experiments

See `exp/README_Experiments.md` for the full ablation (V0–V6) and baseline comparison setup. Default budget: 5 warmstart + 10 random init + 50 BO iterations = 65 evaluations.
