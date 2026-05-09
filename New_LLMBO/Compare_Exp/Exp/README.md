# Compare_Exp/Exp

Baseline experiment runners for paper comparison.

## Contents

### NSGA-II (Multi-objective Genetic Algorithm)

- `nsga2_runner.py`: NSGA-II runner wrapping pymoo's implementation
- `run_nsga2_experiments.py`: Multi-seed experiment launcher

Usage:
```bash
python Compare_Exp/Exp/run_nsga2_experiments.py --seeds 0 1 2 3 4 --n-evals 56
```

### ParEGO (Pareto Efficient Global Optimization)

- `parego_runner.py`: ParEGO runner wrapping llmbo.BayesOptimizer with ParEGO preset
- `run_parego_experiments.py`: Multi-seed experiment launcher

Two variants supported:
- `baseline`: ParEGO with Riesz s-energy weights (weight_count=30)
- `matlab_reference`: MATLAB-style ParEGO with Das-Dennis weights (n_div=30)

Usage:
```bash
# Run MATLAB-reference ParEGO (recommended)
python Compare_Exp/Exp/run_parego_experiments.py --seeds 0 1 2 3 4 --n-evals 56 --variant matlab_reference

# Run baseline ParEGO
python Compare_Exp/Exp/run_parego_experiments.py --seeds 0 1 2 3 4 --n-evals 56 --variant baseline
```

### PlatEMO DISK / PIMD

- `platemo_runner.py`: Python adapter for the MATLAB PlatEMO DISK/PIMD algorithms
- `platemo_eval_helper.py`: Python-side PyBaMM evaluator called from MATLAB
- `platemo_bridge/`: MATLAB bridge files used by `platemo_runner.py`
- `run_platemo_experiments.py`: Multi-seed experiment launcher

The runner keeps the official MATLAB algorithm implementation and converts only
the interface layer. It writes the same `summary.json`, `database.json`, and
`pareto_front.json` files as the other baselines.

Usage:
```bash
# Uses PLATEMO_ROOT if set; otherwise tries the sibling PlatEMO path from this workspace.
python Compare_Exp/Exp/run_platemo_experiments.py --algorithm DISK --seeds 8409 --n-evals 56
python Compare_Exp/Exp/run_platemo_experiments.py --algorithm PIMD --seeds 8409 --n-evals 56

# Explicit PlatEMO root / MATLAB command
python Compare_Exp/Exp/run_platemo_experiments.py --algorithm DISK --platemo-root "D:/path/to/PlatEMO/PlatEMO" --matlab-command matlab
```

## Output Format

All runners produce compatible output:
- `summary.json`: Experiment summary with hypervolume trace
- `database.json`: All observations
- `pareto_front.json`: Final Pareto front

## Dependencies

These runners depend on the main project modules:
- `llmbo.optimizer`: BayesOptimizer and EXPERIMENT_PRESETS
- `DataBase.database`: ObservationDB
- `pybamm_simulator`: Battery simulation
- `utils.constants`: Parameter bounds and constants
- MATLAB + PlatEMO for `PlatEMORunner`
