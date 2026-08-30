"""Central configuration for MO-LLMBO.

All hyperparameters reference §10 of FrameWork.md.
Ablation switches (use_gek/use_llm/use_dpp) map directly to §12 Table.

Multi-objective strategy (§10.6):
    Chebyshev scalarisation with RISE weight sequences.
    Each iteration t uses λ_t = rise_sequence[t], a single GP learns
    F_tch(θ; λ_t) = max_k{λ_t[k]·ĝ_k(θ)} + tch_rho·Σ λ_t[k]·ĝ_k(θ)
    where ĝ_k = (f_k - z*_k) / (z^nad_k - z*_k + tch_eps).
    K = t_max weight vectors are pre-generated once via RISE and cycled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

Float = np.floating[Any]

# Canonical parameter order for θ = [I1, I2, I3, SOC_sw1, SOC_sw2, V_CV, I_cutoff]
PARAM_NAMES: list[str] = ["I1", "I2", "I3", "SOC_sw1", "SOC_sw2", "V_CV", "I_cutoff"]
OBJ_NAMES:   list[str] = ["t_charge", "T_peak", "delta_Q_aging"]


def _default_bounds() -> dict[str, tuple[float, float]]:
    return {
        "I1":       (1.5, 10.0),
        "I2":       (1.5, 10.0),
        "I3":       (1.5, 10.0),
        "SOC_sw1":  (0.2,  0.6),
        "SOC_sw2":  (0.5,  0.8),
        "V_CV":     (4.10, 4.20),
        "I_cutoff": (0.05, 0.5),
    }


@dataclass
class MOLLMBOConfig:
    # ── Battery (LG INR21700-M50) ──────────────────────────────────────────
    q_nom:    float = 5.0     # Ah
    dcir:     float = 0.030   # Ω, mean DCIR
    soc_start: float = 0.1
    soc_end:   float = 0.8
    t_ambient: float = 25.0   # °C

    # ── Search space ───────────────────────────────────────────────────────
    bounds: dict[str, tuple[float, float]] = field(default_factory=_default_bounds)

    # ── Hard constraints ───────────────────────────────────────────────────
    V_max:             float = 4.2
    T_max_conservative: float = 45.0  # °C
    T_max_aggressive:   float = 60.0  # °C

    # ── BO loop (§10.1) ────────────────────────────────────────────────────
    n_init:     int = 15   # LLM warm-start evals
    n_cand:     int = 15   # LLM candidates per iteration
    batch_size: int = 3    # q per iteration; DPP selects from GP+LLM candidates
    t_max:      int = 50   # iterations → 15 + 50×3 = 165 total evals
                           # NOTE: K (RISE weight count) = t_max; no separate field needed

    # ── RISE weight sequence (§10.6) ───────────────────────────────────────
    # Riesz s-Energy Sampling on the (m-1)-simplex.
    # rise_sequence() generates t_max weight vectors λ_0 … λ_{T-1} spread
    # uniformly over the simplex; iteration t uses λ_t to scalarise GP targets.
    rise_s: float = 1.0   # Riesz energy exponent s; s=1 recommended for m=3

    # ── Chebyshev scalarisation (§10.6) ────────────────────────────────────
    # F_tch^aug(θ; λ) = max_k{λ_k·ĝ_k(θ)} + tch_rho·Σ_k λ_k·ĝ_k(θ)
    #   ĝ_k = (f_k - z*_k) / (z^nad_k - z*_k + tch_eps)
    # tch_rho > 0 ensures strict Pareto-optimality of minimisers (augmented Tch).
    # z* and z^nad are updated every iteration from all valid observations.
    tch_rho: float = 0.05  # augmented Tch penalty weight ρ
    tch_eps: float = 1e-6  # numerical stability in normalised denominator

    # ── Physics proxy function Ψ (§3.1) ───────────────────────────────────────
    psi_mc_p:      float = 70.7    # LG M50 thermal mass m·c_p [J/K]
    psi_sei_alpha: float = 0.5     # SEI Butler-Volmer exponent [A⁻¹]
    psi_sei_A:     float = 2.7e-5  # SEI rate prefactor [Ah]

    # ── GEK surrogate (§10.2) ──────────────────────────────────────────────
    gek_grad_dims:  list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    gek_sigma_grad: float = 0.1   # gradient observation noise σ_g
    gek_min_n:      int   = 3     # min obs to activate GEK; falls back to sklearn GP
    gamma_0:     float = 0.1   # physics kernel initial coupling weight γ₀
    gp_alpha:    float = 1e-6  # GP nugget for numerical stability
    n_restarts:  int   = 5     # MLE restarts for GP hyperparameters
    eps_norm:    float = 1e-6  # objective normalisation stability ε
    max_n_mle:   int   = 100   # max obs used for MLE (cap Cholesky at ~200×200)

    # ── TuRBO trust region (§10.3) ─────────────────────────────────────────
    sigma_min_frac:   float = 0.01  # relative to parameter range
    sigma_max_frac:   float = 0.50
    turbo_success_tol: int  = 3
    turbo_failure_tol: int  = 3
    beta_grad_inflate: float = 0.2  # σ²_grad inflation factor β
    r_thresh:          float = 0.5  # gradient discrepancy threshold

    # ── DPP (§10.4) ────────────────────────────────────────────────────────
    dpp_length_scale: float = 0.5
    dpp_k:            int   = 3

    # ── LLM (§10.5) ────────────────────────────────────────────────────────
    llm_model:            str   = "gpt-4o"
    llm_temperature_init: float = 0.2   # Touchpoint 1a: deterministic matrix
    llm_temperature_cand: float = 0.7   # Touchpoint 1b/2: diverse candidates
    llm_top_p:            float = 0.9
    context_level:        str   = "partial"  # none | partial | full

    # ── Objective transform ────────────────────────────────────────────────
    log_aging: bool = True  # apply log10 to delta_Q_aging before GP fitting
    log_time:  bool = True  # apply log10 to t_charge before GP fitting

    # ── Physical penalty values (replace magic 999) ────────────────────────
    # Used to fill failed / constraint-violated results in Database.
    # Also serve as the upper bound for the fixed HV reference point.
    # Units: raw objective space (before any log transform)
    penalty_t_charge: float = 200.0   # min  (= 12000 s, worst-case all stages at 1.5A + full CV)
    penalty_T_peak:   float = 55.0    # °C   (= 328.15 K, above hard constraint)
    penalty_delta_Q:  float = 0.01    # Ah   (= 10^-2, severe aging)

    # ── Fixed HV reference point (transformed space) ───────────────────────
    # Precomputed in log-transformed space matching log_time=True, log_aging=True:
    #   log10(200) ≈ 2.301,  55.0,  log10(0.01) = -2.0
    # Fixed across all iterations → HV values are directly comparable.
    # Must strictly dominate every valid Pareto point in transformed space.
    hv_ref_fixed: tuple = (2.301, 55.0, -2.0)  # log10(200)=2.301

    # ── Ablation switches (§12 Table) ──────────────────────────────────────
    use_gek:   bool = True
    use_llm:   bool = True
    use_dpp:   bool = True
    use_turbo: bool = True

    # ── Misc ───────────────────────────────────────────────────────────────
    n_workers:    int  = 3     # parallel PyBaMM workers
    random_seed:  int  = 42
    # hv_ref_scale removed: replaced by hv_ref_fixed (physics-based fixed point)

    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> MOLLMBOConfig:
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    @property
    def dim(self) -> int:
        return len(self.bounds)

    @property
    def param_names(self) -> list[str]:
        return list(self.bounds.keys())

    @property
    def bounds_array(self) -> NDArray[Float]:
        """shape (d, 2)"""
        return np.array(list(self.bounds.values()), dtype=float)

    @property
    def param_ranges(self) -> NDArray[Float]:
        """shape (d,)  upper - lower for each dimension"""
        b = self.bounds_array
        return b[:, 1] - b[:, 0]

    def ablation_tag(self) -> str:
        """Experiment label matching §12 Table rows."""
        parts = [
            "GEK"   if self.use_gek   else "noGEK",
            "LLM"   if self.use_llm   else "noLLM",
            "DPP"   if self.use_dpp   else "noDPP",
            "TuRBO" if self.use_turbo else "noTuRBO",
        ]
        return "_".join(parts)

    @property
    def n_obj(self) -> int:
        """Number of objectives (always 3 for this problem)."""
        return len(OBJ_NAMES)

    def to_yaml(self, path: str) -> None:
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)