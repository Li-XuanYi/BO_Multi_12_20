"""
Compare_Exp/Exp/__init__.py
===========================
Baseline experiment runners for paper comparison.

This module contains standalone experiment runners for:
- NSGA-II: Multi-objective genetic algorithm baseline
- ParEGO: Bayesian optimization with Tchebycheff scalarization baseline
- PlatEMO DISK/PIMD: MATLAB expensive MOEA baselines through a Python bridge
"""

from Compare_Exp.Exp.nsga2_runner import NSGA2Runner
from Compare_Exp.Exp.parego_runner import ParEGORunner
from Compare_Exp.Exp.platemo_runner import PlatEMORunner

__all__ = ["NSGA2Runner", "ParEGORunner", "PlatEMORunner"]
