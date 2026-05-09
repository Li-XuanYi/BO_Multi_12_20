"""
platemo_runner.py - Python adapter for PlatEMO DISK/PIMD baselines.

The algorithm implementation remains the official MATLAB PlatEMO code.  This
runner provides the same Python-facing interface as NSGA2Runner/ParEGORunner:

    runner = PlatEMORunner(algorithm="DISK", seed=8409, n_evals=56)
    runner.run()
    runner.save_results("output_dir")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataBase.database import OBJECTIVE_NAMES, ObservationDB, make_observation_db

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS = {"DISK", "PIMD"}


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"PlatEMO trace file not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _matlab_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _default_platemo_candidates() -> Iterable[Path]:
    env = os.getenv("PLATEMO_ROOT")
    if env:
        yield Path(env)

    sibling = (
        PROJECT_ROOT.parent
        / "192178cb2531720c24b3fb6dd2a3613d_97d1a25f628840c99192fb58ea9cae39_8"
        / "PlatEMO"
        / "PlatEMO"
    )
    yield sibling

    for found in PROJECT_ROOT.parent.glob("*/PlatEMO/PlatEMO"):
        yield found


def resolve_platemo_root(platemo_root: Optional[Path | str] = None) -> Path:
    candidates = [Path(platemo_root)] if platemo_root else list(_default_platemo_candidates())
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "platemo.m").exists():
            return root
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find PlatEMO root containing platemo.m. "
        "Pass platemo_root=... or set PLATEMO_ROOT. Checked: " + checked
    )


class PlatEMORunner:
    """Run PlatEMO DISK/PIMD on the New_LLMBO battery problem."""

    DEFAULT_PARAMETERS = {
        "DISK": [60, 5],
        "PIMD": [15, 5],
    }

    def __init__(
        self,
        algorithm: str = "DISK",
        seed: int = 0,
        n_evals: int = 56,
        population_size: int = 20,
        algorithm_parameters: Optional[List[float]] = None,
        platemo_root: Optional[Path | str] = None,
        matlab_command: str = "matlab",
        python_executable: Optional[Path | str] = None,
        work_dir: Optional[Path | str] = None,
    ):
        alg = algorithm.upper()
        if alg not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported PlatEMO algorithm '{algorithm}'. Choose from {sorted(SUPPORTED_ALGORITHMS)}")

        self.algorithm = alg
        self.seed = int(seed)
        self.n_evals = int(n_evals)
        self.population_size = min(int(population_size), self.n_evals)
        self.algorithm_parameters = list(algorithm_parameters or self.DEFAULT_PARAMETERS[alg])
        self.platemo_root = resolve_platemo_root(platemo_root)
        self.matlab_command = matlab_command
        self.python_executable = str(python_executable or sys.executable)
        self.bridge_dir = Path(__file__).resolve().parent / "platemo_bridge"
        self.work_dir = Path(work_dir) if work_dir else (
            PROJECT_ROOT
            / "checkpoints"
            / "platemo_bridge"
            / f"{self.algorithm.lower()}_seed{self.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.db: Optional[ObservationDB] = None
        self.hv_trace: List[Dict[str, Any]] = []
        self.trace_records: List[Dict[str, Any]] = []
        self._result_summary: Optional[Dict[str, Any]] = None
        self._matlab_stdout = ""
        self._matlab_stderr = ""

    def run(self) -> ObservationDB:
        """Run MATLAB PlatEMO and convert the JSONL trace into ObservationDB."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.work_dir / "evaluations.jsonl"
        config_path = self.work_dir / "platemo_config.json"
        final_population_path = self.work_dir / "final_population.json"

        config = {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "n_evals": self.n_evals,
            "population_size": self.population_size,
            "algorithm_parameters": self.algorithm_parameters,
            "project_root": str(PROJECT_ROOT),
            "platemo_root": str(self.platemo_root),
            "output_dir": str(self.work_dir),
            "trace_path": str(trace_path),
            "final_population_path": str(final_population_path),
            "python_executable": self.python_executable,
        }
        _json_dump(config_path, config)

        batch = f"addpath({_matlab_quote(self.bridge_dir)}); run_platemo_battery({_matlab_quote(config_path)})"
        cmd = [self.matlab_command, "-batch", batch]
        if shutil.which(self.matlab_command) is None and not Path(self.matlab_command).exists():
            raise FileNotFoundError(
                f"MATLAB command not found: {self.matlab_command!r}. "
                "Install MATLAB, add it to PATH, or pass --matlab-command with the full matlab.exe path."
            )
        logger.info(
            "PlatEMO %s: seed=%d, maxFE=%d, N=%d, root=%s",
            self.algorithm,
            self.seed,
            self.n_evals,
            self.population_size,
            self.platemo_root,
        )

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
        self._matlab_stdout = proc.stdout or ""
        self._matlab_stderr = proc.stderr or ""
        (self.work_dir / "matlab_stdout.log").write_text(self._matlab_stdout, encoding="utf-8")
        (self.work_dir / "matlab_stderr.log").write_text(self._matlab_stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"MATLAB PlatEMO run failed with exit code {proc.returncode}. "
                f"See {self.work_dir / 'matlab_stderr.log'}"
            )

        self.trace_records = _read_jsonl(trace_path)
        self.db = make_observation_db()
        self.hv_trace = []
        for i, rec in enumerate(self.trace_records, start=1):
            theta = np.asarray(rec["theta"], dtype=float)
            objectives = np.asarray(rec["objectives"], dtype=float)
            feasible = bool(rec.get("feasible", False)) and float(rec.get("constraint", 0.0)) <= 0.0
            source = f"{self.algorithm.lower()}_init" if i <= self.population_size else self.algorithm.lower()
            self.db.add_from_simulator(
                theta=theta,
                result={
                    "raw_objectives": objectives,
                    "feasible": feasible,
                    "violation": rec.get("violation"),
                    "details": {
                        "platemo_eval_index": rec.get("eval_index", i),
                        "platemo_constraint": rec.get("constraint"),
                        "platemo_elapsed_s": rec.get("elapsed_s"),
                    },
                },
                source=source,
                iteration=i,
            )
            display_hv = self.db.compute_hypervolume()
            raw_hv = self.db.compute_hypervolume_raw()
            canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0
            self.hv_trace.append(
                {
                    "eval_index": i,
                    "phase": "init" if i <= self.population_size else "platemo",
                    "iteration": i,
                    "source": source,
                    "theta": theta.tolist(),
                    "feasible": feasible,
                    "hypervolume": display_hv,
                    "display_hv": display_hv,
                    "canonical_hv": canonical_hv,
                    "hypervolume_raw": raw_hv,
                    "pareto_size": self.db.pareto_size,
                    "n_total": self.db.size,
                    "n_feasible": self.db.n_feasible,
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )

        logger.info(
            "PlatEMO %s done: %d evals, %d feasible, %d Pareto, HV=%.4f",
            self.algorithm,
            self.db.size,
            self.db.n_feasible,
            self.db.pareto_size,
            self.db.compute_hypervolume(),
        )
        return self.db

    def save_results(self, output_dir: str) -> Dict[str, Any]:
        if self.db is None:
            raise RuntimeError("Must call run() before save_results()")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        display_hv = self.db.compute_hypervolume()
        raw_hv = self.db.compute_hypervolume_raw()
        canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0

        summary = {
            "algorithm": f"platemo_{self.algorithm.lower()}",
            "seed": self.seed,
            "n_evals": self.n_evals,
            "n_total": self.db.size,
            "population_size": self.population_size,
            "algorithm_parameters": self.algorithm_parameters,
            "platemo_root": str(self.platemo_root),
            "matlab_command": self.matlab_command,
            "n_feasible": self.db.n_feasible,
            "pareto_size": self.db.pareto_size,
            "hypervolume": display_hv,
            "display_hv": display_hv,
            "canonical_hv": canonical_hv,
            "hypervolume_raw": raw_hv,
            "hv_trace": self.hv_trace,
            "best_per_objective": {},
            "timestamp": datetime.now().isoformat(),
        }

        best = self.db.get_best_per_objective()
        for name in OBJECTIVE_NAMES:
            if name in best:
                obs = best[name]
                idx = list(OBJECTIVE_NAMES).index(name)
                summary["best_per_objective"][name] = {
                    "value": float(obs.objectives[idx]),
                    "theta": obs.theta.tolist(),
                }

        _json_dump(out / "summary.json", summary)
        self.db.save(str(out / "database.json"))

        pareto_data = [
            {
                "theta": obs.theta.tolist(),
                "objectives": obs.objectives.tolist(),
            }
            for obs in self.db.get_pareto_front()
        ]
        _json_dump(out / "pareto_front.json", {"points": pareto_data})

        bridge_out = out / "platemo_bridge"
        bridge_out.mkdir(exist_ok=True)
        for name in ["platemo_config.json", "evaluations.jsonl", "final_population.json", "matlab_stdout.log", "matlab_stderr.log"]:
            src = self.work_dir / name
            if src.exists():
                shutil.copy2(src, bridge_out / name)

        self._result_summary = summary
        logger.info("PlatEMO results saved to %s", out)
        return summary

    def get_summary(self) -> Optional[Dict[str, Any]]:
        return self._result_summary
