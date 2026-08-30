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

### DISK / PIMD (Python Native Implementation)

- `disk_pimd_algorithms.py`: Python native implementation of DISK and PIMD algorithms
- `run_disk_python.py`: DISK experiment runner
- `run_pimd_python.py`: PIMD experiment runner

These are pure Python implementations of the DISK (Dynamic Island Single-objective Kriging)
and PIMD (Pareto-based Infilling with Maximum Diversity) algorithms, using Kriging surrogate
models and Tchebycheff scalarization. No MATLAB or PlatEMO required.

Usage:
```bash
# Run DISK experiments (5 seeds, 50 evals)
python Compare_Exp/run_disk_python.py --seeds 8409 8410 8411 8412 8413 --n-evals 50

# Run PIMD experiments
python Compare_Exp/run_pimd_python.py --seeds 8409 8410 8411 8412 8413 --n-evals 50

# Custom parameter set
python Compare_Exp/run_disk_python.py --param-set Ecker2015 --n-evals 56
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

## Additional Tools

### Plotting Scripts

- `plot_disk_pimd_hv.py`: HV convergence plot for DISK vs PIMD comparison
