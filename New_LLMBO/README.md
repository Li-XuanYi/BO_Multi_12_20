# LLAMBO-MO: LLM-Augmented Multi-Objective Bayesian Optimization

LLAMBO-MO is a physics-informed multi-objective Bayesian optimization framework that leverages Large Language Models (LLMs) to optimize battery fast-charging protocols. The system balances three competing objectives: minimizing charging time, peak temperature rise, and capacity fade.

## 🎯 Problem Formulation

### Decision Variables (5D)

| Variable | Range | Description |
|----------|--------|-------------|
| I1 | [2.0, 6.0] A | Stage 1 charging current |
| I2 | [2.0, 5.0] A | Stage 2 charging current |
| I3 | [2.0, 3.0] A | Stage 3 charging current |
| dSOC1 | [0.10, 0.40] | Stage 1 SOC interval width |
| dSOC2 | [0.10, 0.30] | Stage 2 SOC interval width |

**Constraint**: `dSOC1 + dSOC2 ≤ 0.70`

### Objectives (3D, all minimization)

- **f₁**: Charging time [s]
- **f₂**: Peak temperature rise [K]
- **f₃**: Capacity fade [%]

### Scalarization

Uses Tchebycheff scalarization with augmented term:

```
f̃₁ = log₁₀(time_s)
f̃₂ = temp_K
f̃₃ = log₁₀(aging_%)

f_tch = max_i(w_i · f̄_i) + η(0.05) · Σ(w_i · f̄_i)
```

## 🏗️ Architecture

### Directory Structure

```
New_LLMBO/
├── main.py                    # CLI entry point
├── pybamm_simulator.py        # Battery physics simulator
├── config/                    # Configuration system
│   ├── schema.py             # Pydantic schema definitions
│   └── load.py              # Config loader (JSON → env → CLI)
├── DataBase/                  # Observation database
│   └── database.py          # Storage, Pareto tracking, HV computation
├── llm/                       # LLM integration
│   ├── llm_interface.py     # Two-touchpoint LLM interface
│   ├── warmstart_prompt.py   # Warmstart prompt templates
│   └── templates/           # Prompt template files
├── llmbo/                     # Core optimization engine
│   ├── optimizer.py         # Main optimizer orchestration
│   ├── gp_model.py          # Gaussian Process model
│   ├── acquisition.py       # Expected Improvement acquisition
│   └── riesz_cache.py       # Riesz s-energy weight caching
├── utils/                     # Utilities
│   └── constants.py         # Global constants and bounds
├── results/                   # Output directory
└── .riesz_cache/              # Cached Riesz weight sets
```

### Core Components

| Component | Responsibility |
|-----------|----------------|
| `main.py` | CLI parsing, config loading, async entry point |
| `BayesOptimizer` | Orchestrates the entire optimization loop |
| `ObservationDB` | Stores observations, computes HV, tracks Pareto front |
| `PyBaMMSimulator` | SPMe battery model simulation |
| `MaternGPModel` | Physics-informed GP with coupling support |
| `AcquisitionFunction` | EI optimization with multi-start L-BFGS-B |
| `LLMInterface` | Two LLM touchpoints with fallback to LHS |

## 🔄 Workflow

### 1. Initialization Phase (`optimizer.setup()`)

```python
# Create core components
simulator = PyBaMMSimulator()
database = ObservationDB(param_bounds, ref_point, ideal_point)
llm = build_llm_interface(backend, model, api_key)
gp_model = build_gp_stack(param_bounds, kernel_nu)
acquisition = build_acquisition_function(gp_model, param_bounds)
weight_set = load_or_generate_riesz(n_obj=3, n_div=10)  # Cached
```

### 2. Warmstart Phase (`run_initialization()`)

```
For each warmstart point:
    1. Generate candidates via LLM Touchpoint 1b
       - LLM generates n_warmstart initial protocols
       - Fallback to Latin Hypercube Sampling (LHS) on failure

    2. Evaluate candidates via PyBaMM
       - Run SPMe simulation
       - Compute objectives: [time_s, temp_K, aging_%]

    3. Store in database
       - Update Pareto front
       - Compute Hypervolume
```

### 3. Main BO Loop (`run_optimization_loop()`)

For each iteration t = 1 to T:

```
┌─────────────────────────────────────────────────────────────┐
│ 3.1 Sample weight vector                                │
│     w_vec ~ Riesz s-energy weights (random permutation)    │
├─────────────────────────────────────────────────────────────┤
│ 3.2 Update Tchebycheff context                         │
│     - Dynamic y_min, y_max from feasible observations     │
│     - Update database with current w_vec                  │
├─────────────────────────────────────────────────────────────┤
│ 3.3 Fit GP model                                      │
│     - Training data: all feasible observations              │
│     - Target: Tchebycheff scalarized objectives           │
│     - Kernel: Matern 5/2 + WhiteKernel                 │
├─────────────────────────────────────────────────────────────┤
│ 3.4 Compute uncertainty hotspots                        │
│     - Probe N points via Sobol sequence                  │
│     - Select top K points with highest GP std            │
├─────────────────────────────────────────────────────────────┤
│ 3.5 LLM Touchpoint 2: Iteration guidance             │
│     Input:                                              │
│       - Current iteration t                                │
│       - Weight vector w_vec                               │
│       - Best solution θ_best                              │
│       - Uncertainty hotspots                              │
│       - Database summary (Pareto front, stats)          │
│                                                          │
│     Output:                                              │
│       - mode: "point" or "region"                       │
│       - confidence: [0, 1]                              │
│       - representative point OR region bounds              │
│     Fallback to LHS/physics heuristics on failure        │
├─────────────────────────────────────────────────────────────┤
│ 3.6 Build GP-LLM coupling                             │
│     - Generate sampling grid based on LLM guidance         │
│     - Compute posterior variance σ²_λλ = wᵀ Σ_λλ w      │
│     - Coupling strength: λ = confidence / σ_λλ           │
├─────────────────────────────────────────────────────────────┤
│ 3.7 Acquisition function optimization                    │
│     Score = EI(θ) × W_charge(θ)                        │
│                                                          │
│     where:                                               │
│       EI(θ) = (f_min - μ)Φ(z) + σφ(z)                │
│       z = (f_min - μ) / σ                              │
│       μ_coupeld = μ - λ · Σ_xgλ · w                     │
├─────────────────────────────────────────────────────────────┤
│ 3.8 Evaluate selected candidates                         │
│     - Run PyBaMM simulation                              │
│     - Store in database with GP predictions               │
├─────────────────────────────────────────────────────────────┤
│ 3.9 Update statistics                                  │
│     - Pareto front                                     │
│     - Hypervolume                                      │
│     - Stagnation detection                              │
│     - Save checkpoint (periodic)                         │
└─────────────────────────────────────────────────────────────┘
```

### 4. Results Save (`save_results()`)

```
results/
├── database.json          # Full observation database
├── db_final.json        # Final database snapshot
├── pareto_front.json    # Non-dominated solutions
└── summary.json         # Optimization summary
    - n_total, n_feasible
    - pareto_size, hypervolume
    - warmstart_trace, hv_trace
    - last_guidance, last_gp_llm_coupling
    - config (redacted)
```

## 🚀 Usage

### Quick Demo (5 iterations, no LLM API required)

```bash
python main.py --demo
```

### Generate Config Template

```bash
python main.py --generate-template --template-mode full
# Outputs: config_template.json
```

### Run with Configuration File

```bash
python main.py --config config.json --verbose
```

### CLI Overrides (higher priority than config file)

```bash
python main.py --config config.json \
    --bo.n_iterations=100 \
    --acquisition.n_cand=20 \
    --llm.model=gpt-4.1-mini
```

### Mock Mode (No LLM API calls)

```bash
python main.py --mock
```

## ⚙️ Configuration

Configuration priority (latter overrides former):
1. Default values in `config/schema.py`
2. JSON configuration file (`--config`)
3. Environment variables (`LLM_API_KEY`, `BO_N_ITERATIONS`, etc.)
4. CLI overrides (`--bo.n_iterations=100`)

### Key Configuration Parameters

| Section | Parameter | Default | Description |
|----------|------------|----------|-------------|
| `bo` | `n_iterations` | 20 | Main BO loop iterations |
| `bo` | `n_warmstart` | 10 | LLM warmstart candidates |
| `bo` | `gamma_max` | 1.0 | Max GP-LLM coupling strength |
| `llm` | `model` | gpt-4.1-mini | LLM model name |
| `llm` | `api_key` | - | OpenAI-compatible API key |
| `llm` | `base_url` | - | API base URL |
| `acquisition` | `n_cand` | 15 | Number of candidates per iteration |
| `acquisition` | `n_select` | 1 | Number of points to evaluate |
| `data` | `save_interval` | 5 | Checkpoint save frequency |

## 🔬 Key Algorithms

### Tchebycheff Scalarization

```python
# Log-transform objectives
Y_tilde = Y_raw.copy()
Y_tilde[:, 0] = log10(Y_raw[:, 0])  # time
Y_tilde[:, 2] = log10(Y_raw[:, 2])  # aging

# Dynamic normalization
Y_bar = (Y_tilde - y_min) / (y_max - y_min)

# Tchebycheff with tiebreaker
Wf = w_vec * Y_bar
F_tch = max(Wf) + eta * sum(Wf)  # eta = 0.05
```

### Riesz s-Energy Weight Generation

- Start from Das-Dennis grid (uniform on simplex)
- Apply projected gradient descent on Riesz energy
- Encourages well-distributed weight vectors

```python
# Riesz energy gradient at point i
grad[i] = Σ_j (s / |w_i - w_j|^{s+2}) * (w_j - w_i)
```

### GP-LLM Coupling

The GP posterior mean is adjusted based on LLM guidance:

```python
# Coupling strength
lambda = confidence / sqrt(variance_of_LLM_region)

# Adjusted mean
mu_coupled = mu - lambda * covariance(X, LLM_grid) @ LLM_weights
```

### Stagnation Detection

- Track HV improvement over sliding window (size=2)
- If no improvement for 2 consecutive iterations → stagnation
- Acquisition sigma scaled by `(1 + 0.2 * min(stagnation_count, 3))`

## 📊 Metrics

### Hypervolume (HV)

- Normalized HV ∈ [0, 1]
- Reference point: `[5400s, 318K, 0.1%]`
- Ideal point: `[2700s, 298K, 0.001%]`
- log₁₀ transform applied to time and aging objectives

### Pareto Front

- Incrementally updated (O(|PF|) per insertion)
- Dominance: a dominates b iff ∀f_a ≤ f_b and ∃f_a < f_b

## 🛠️ Dependencies

```bash
# Core
numpy
scipy
scikit-learn

# Battery simulation
pybamm

# LLM integration
openai  # or compatible API

# CLI (optional)
pydantic
```

## 📝 Output Format

### Observation Database (`database.json`)

```json
{
  "version": "2.0",
  "param_bounds": {...},
  "ref_point": [5400.0, 318.0, 0.1],
  "ideal_point": [2700.0, 298.0, 0.001],
  "observations": [
    {
      "theta": [I1, I2, I3, dSOC1, dSOC2],
      "objectives": [time_s, temp_K, aging_pct],
      "feasible": true,
      "source": "bo|llm_warmstart|random_init",
      "iteration": 1,
      "acq_value": 0.123,
      "acq_type": "EI_gp_llm_coupled",
      "gp_pred": {
        "mean_coupled": 100.0,
        "mean_base": 105.0,
        "std": 10.0,
        "coupling_lambda": 0.5
      },
      "llm_rationale": "...",
      "timestamp": "2026-04-14T..."
    }
  ],
  "pareto_indices": [0, 5, 12],
  "iteration_stats": [...]
}
```

## 🔍 Debugging

### Enable verbose logging

```bash
python main.py --config config.json --verbose
```

### Checkpoint inspection

```python
import json
from DataBase.database import ObservationDB

# Load checkpoint
db = ObservationDB.load("checkpoints/db_t0010.json")
print(db.summary())

# Access Pareto front
pareto = db.get_pareto_front()
for obs in pareto:
    print(f"θ={obs.theta}, f={obs.objectives}")
```

## 📚 References

- **ParEGO**: Deb et al., "Pareto-based Multi-objective Bayesian Optimization" (2016)
- **Riesz s-Energy**: Liu & Qin, "Regularized Simplex Lattice Design" (2022)
- **PyBaMM**: Mooney et al., "Battery Modelling in Python" (2021)
- **LG INR21700-M50**: Chen et al., "Comprehensive Parameterization of Lithium-Ion Battery" (2020)

## 📄 License

This project is research code. Please cite appropriately if used in academic work.

## 🤝 Contributing

This is a research prototype. Key extension points:
- New battery models (extend `PyBaMMSimulator`)
- New scalarization functions (modify `optimizer.py`)
- New acquisition functions (extend `acquisition.py`)
- Alternative LLM backends (extend `llm_interface.py`)
