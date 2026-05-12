"""
Compare_Exp/Exp/__init__.py
===========================
Baseline experiment runners for paper comparison.

This module contains standalone experiment runners for:
- NSGA-II: Multi-objective genetic algorithm baseline
- ParEGO: Bayesian optimization with Tchebycheff scalarization baseline
- DISK/PIMD: Python native implementation of expensive MOEA baselines
"""

from Compare_Exp.Exp.nsga2_runner import NSGA2Runner
from Compare_Exp.Exp.parego_runner import ParEGORunner
from Compare_Exp.Exp.disk_pimd_algorithms import DISKOptimizer, PIMDOptimizer

__all__ = ["NSGA2Runner", "ParEGORunner", "DISKOptimizer", "PIMDOptimizer"]
