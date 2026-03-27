"""LLM integration layer for MO-LLMBO.

Three Touchpoints (§14):
    1a  LLMInterface.get_coupling_matrices()               → dict[str, np.ndarray]
    1b  LLMInterface.warm_start(n)                         → list[np.ndarray]  (7,) each
    2   LLMInterface.generate_candidates(summary, lambda_t, n) → list[np.ndarray]

Stateless design (§8.1):
    Every API call is independent. All "memory" is injected via DatabaseSummarizer.
    No conversation history is ever accumulated.

Chebyshev-RISE integration (§10.6):
    generate_candidates receives the current RISE weight vector λ_t (shape (3,))
    and injects an <optimization_direction> block into the prompt, directing the
    LLM to focus on the trade-off axis defined by λ_t for the current iteration.
    DatabaseSummarizer.generate_summary accepts the same lambda_t for inclusion
    in the context when context_level is "partial" or "full".

Dependencies:  openai>=1.0   (set OPENAI_API_KEY in environment)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config import MOLLMBOConfig, OBJ_NAMES, PARAM_NAMES

log = logging.getLogger(__name__)

# ── Utilities ──────────────────────────────────────────────────────────────────

def _project_to_psd(W: NDArray) -> NDArray:
    """Eigenvalue-clip to the PSD cone (§3.4).

    W_psd = Σ_{λ_i > 0} λ_i v_i v_iᵀ
    """
    W = (W + W.T) / 2.0           # symmetrise first
    eigvals, eigvecs = np.linalg.eigh(W)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs * eigvals) @ eigvecs.T


def _theta_dict_to_array(d: dict[str, Any], config: MOLLMBOConfig) -> NDArray | None:
    """Convert a {param: value} dict from LLM output to a θ array.

    Returns None if a required key is missing or value is non-numeric.
    Values are clipped to bounds rather than rejected, matching the
    LLAMBO strategy of accepting noisy proposals and letting BO refine them.
    """
    theta = np.empty(config.dim)
    bounds = config.bounds_array        # (d, 2)
    for i, name in enumerate(config.param_names):
        if name not in d:
            return None
        try:
            v = float(d[name])
        except (TypeError, ValueError):
            return None
        theta[i] = np.clip(v, bounds[i, 0], bounds[i, 1])
    return theta


def _parse_theta_list(text: str, config: MOLLMBOConfig) -> list[NDArray]:
    """Robust θ extraction from raw LLM text.

    Pipeline:
        1. Strip markdown code fences
        2. Try json.loads on the full text
        3. Fall back to regex: extract the outermost [...] array
        4. Convert each valid dict → θ array
    """
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")

    candidates: list[dict] = []

    # Attempt 1: full parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = [parsed]
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first [...] block
    if not candidates:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                candidates = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    # Attempt 3: extract individual {...} objects
    if not candidates:
        for m in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
            try:
                candidates.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue

    results: list[NDArray] = []
    for d in candidates:
        if not isinstance(d, dict):
            continue
        theta = _theta_dict_to_array(d, config)
        if theta is not None:
            results.append(theta)
    return results


# ── DatabaseSummarizer (§7) ────────────────────────────────────────────────────

class DatabaseSummarizer:
    """Converts optimizer state to a self-contained LLM prompt context.

    Stateless between iterations: each call to generate_summary is fully
    independent — all "memory" is injected via the Database object (§8.1).

    Token budget per context level (§7.1):
        none    ~80 tokens
        partial ~350 tokens   (+<optimization_direction> when lambda_t provided)
        full    ~650 tokens

    Args:
        config: MOLLMBOConfig — stored once at construction, used in every call.

    Args (generate_summary):
        db:          Database    — live observation store (provides results + Pareto)
        level:       "none" | "partial" | "full"   (keyword alias: context_level)
        lambda_t:    (m,) current RISE weight vector; None during warm-start.
        grad_psi:    (d,) ∇Ψ at current best θ; defaults to zeros if not provided.
        sigma_grad:  (d,) per-dim gradient noise σ; defaults to config.gek_sigma_grad.

    Returns:
        str — prompt context block ready for injection
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        self.config = config

    def generate_summary(
        self,
        db: Any,                          # Database (imported lazily to avoid circular)
        level: str = "partial",
        lambda_t: NDArray | None = None,
        grad_psi: NDArray | None = None,
        sigma_grad: NDArray | None = None,
    ) -> str:
        """Build a self-contained context string from the current Database.

        Extracts all_results, pareto_results, and iteration count directly
        from the Database object so the caller only needs to pass one argument.
        grad_psi and sigma_grad are optional; zero vectors are used as defaults
        during warm-start when no physics gradients are available yet.
        """
        config = self.config

        # ── Extract lists from Database ───────────────────────────────────────
        all_results: list = list(db._results)
        valid_results = [r for r, ok in zip(db._results, db._valid_mask) if ok]
        pf_idx = db.pareto_front_indices()
        pareto_results = [valid_results[i] for i in pf_idx] if pf_idx else []
        iteration = len(db)

        # ── Default gradient arrays ───────────────────────────────────────────
        if grad_psi is None:
            grad_psi = np.zeros(config.dim)
        if sigma_grad is None:
            sigma_grad = np.full(config.dim, config.gek_sigma_grad)

        parts = [self._problem_description(config)]
        if level in ("partial", "full"):
            parts.append(self._data_card(all_results, pareto_results, iteration, config))
            parts.append(self._gradient_sensitivity(grad_psi, sigma_grad, config))
            if lambda_t is not None:
                parts.append(self._optimization_direction(lambda_t, config))
        if level == "full":
            parts.append(self._pareto_table(pareto_results))
            parts.append(self._lessons_learned(all_results, pareto_results, config))
            parts.append(self._target_region(pareto_results, config))
        return "\n\n".join(parts)

    # ── private section builders ───────────────────────────────────────────────

    def _problem_description(self, config: MOLLMBOConfig) -> str:
        b = config.bounds
        return (
            f"<problem>\n"
            f"Battery: LG INR21700-M50, {config.q_nom} Ah NMC811/Graphite\n"
            f"Charging: SOC {config.soc_start} → {config.soc_end}, ambient {config.t_ambient}°C\n"
            f"Protocol: 3-stage CC-CV with parameters:\n"
            + "\n".join(
                f"  {n}: [{lo:.3g}, {hi:.3g}]"
                for n, (lo, hi) in b.items()
            )
            + f"\nConstraint: I1 ≥ I2 ≥ I3 (decreasing current profile)\n"
            f"Objectives (all minimize): t_charge [min], T_peak [°C], delta_Q_aging [Ah]\n"
            f"</problem>"
        )

    def _data_card(
        self,
        all_results: list,
        pareto_results: list,
        iteration: int,
        config: MOLLMBOConfig,
    ) -> str:
        n_total  = len(all_results)
        n_pareto = len(pareto_results)
        n_valid  = sum(1 for r in all_results if r.constraint_ok)

        fastest  = min(pareto_results, key=lambda r: r.t_charge, default=None)
        gentlest = min(pareto_results, key=lambda r: r.T_peak,   default=None)

        def _fmt_result(r: Any) -> str:
            return (
                f"θ=({', '.join(f'{v:.2f}' for v in r.theta)})"
                f" → t={r.t_charge:.1f}min, T={r.T_peak:.1f}°C, aging={r.delta_Q_aging:.4e}Ah"
            )

        lines = [
            "<data_card>",
            f"Iteration: {iteration}/{config.t_max}",
            f"Evaluated: {n_total} protocols ({n_valid} constraint-OK)",
            f"Pareto front size: {n_pareto} solutions",
        ]
        if fastest:
            lines.append(f"Fastest:  {_fmt_result(fastest)}")
        if gentlest and gentlest is not fastest:
            lines.append(f"Gentlest: {_fmt_result(gentlest)}")

        # Objective ranges over all valid results
        valid = [r for r in all_results if r.constraint_ok]
        if valid:
            t_vals = [r.t_charge    for r in valid]
            T_vals = [r.T_peak      for r in valid]
            a_vals = [r.delta_Q_aging for r in valid]
            lines += [
                "Objective ranges (valid only):",
                f"  t_charge: [{min(t_vals):.1f}, {max(t_vals):.1f}] min",
                f"  T_peak:   [{min(T_vals):.1f}, {max(T_vals):.1f}] °C",
                f"  aging:    [{min(a_vals):.2e}, {max(a_vals):.2e}] Ah",
            ]
        lines.append("</data_card>")
        return "\n".join(lines)

    def _gradient_sensitivity(
        self,
        grad_psi: NDArray,
        sigma_grad: NDArray,
        config: MOLLMBOConfig,
    ) -> str:
        """Show |∂Ψ/∂θ_i| and reliability (σ_grad) per parameter."""
        lines = ["<gradient_sensitivity>"]
        for name, g, s in zip(config.param_names, grad_psi, sigma_grad):
            reliability = "reliable" if s < 0.1 * abs(g) + 1e-8 else "uncertain"
            lines.append(f"  |∂Ψ/∂{name}| = {abs(g):.4f}   σ={s:.4f}  [{reliability}]")
        lines.append("</gradient_sensitivity>")
        return "\n".join(lines)

    def _optimization_direction(
        self,
        lambda_t: NDArray,
        config: MOLLMBOConfig,
    ) -> str:
        """Encode the current RISE weight vector as a plain-language trade-off directive.

        The Chebyshev scalarisation weight λ_t defines the GP's optimisation axis
        for this iteration.  We translate it into a human-readable sentence so the
        LLM can align its proposals without needing to understand the maths.

        Args:
            lambda_t: (m,) weight vector from RISE sequence; entries sum to 1.
            config:   MOLLMBOConfig (provides OBJ_NAMES order)
        Returns:
            <optimization_direction> XML block, ~60 tokens.
        """
        # Map objective index → human label
        obj_labels = ["charging speed", "thermal safety", "battery longevity"]

        # Sort objectives by weight descending to build a priority sentence
        order     = np.argsort(lambda_t)[::-1]          # highest weight first
        lam_strs  = [f"{OBJ_NAMES[k]}={lambda_t[k]:.2f}" for k in order]
        pri_strs  = [f"{obj_labels[k]} (λ={lambda_t[k]:.2f})" for k in order]

        # Interpret dominant weight
        dominant_idx = int(order[0])
        dominant_lam = lambda_t[dominant_idx]

        if dominant_lam >= 0.6:
            emphasis = f"strongly prioritise {obj_labels[dominant_idx]}"
        elif dominant_lam >= 0.4:
            emphasis = f"moderately prioritise {obj_labels[dominant_idx]}"
        else:
            emphasis = "balance all three objectives roughly equally"

        lines = [
            "<optimization_direction>",
            f"This iteration's Chebyshev weights: {', '.join(lam_strs)}",
            f"Priority order: {' > '.join(pri_strs)}",
            f"Instruction: {emphasis}. Candidates should reflect this trade-off.",
            "</optimization_direction>",
        ]
        return "\n".join(lines)

    def _pareto_table(self, pareto_results: list) -> str:
        """Top-8 Pareto solutions in ascending quality order (OPRO §5.3).

        Ascending order means best solutions appear last, nudging the LLM
        to propose improvements beyond the current best.
        """
        if not pareto_results:
            return "<pareto_front>No Pareto solutions yet.</pareto_front>"

        # Sort by t_charge descending → worst first (best last = OPRO ordering)
        sorted_results = sorted(pareto_results, key=lambda r: r.t_charge, reverse=True)[:8]

        header = "| Rank | " + " | ".join(PARAM_NAMES) + " | t_charge | T_peak | aging |"
        sep    = "|------|" + "--------|" * len(PARAM_NAMES) + "----------|--------|-------|"
        rows   = []
        for rank, r in enumerate(sorted_results, 1):
            params = " | ".join(f"{v:.3f}" for v in r.theta)
            rows.append(
                f"| {rank:<4} | {params} | {r.t_charge:8.1f} | {r.T_peak:6.1f} | {r.delta_Q_aging:.2e} |"
            )

        return "<pareto_front>\n" + "\n".join([header, sep] + rows) + "\n</pareto_front>"

    def _lessons_learned(
        self,
        all_results: list,
        pareto_results: list,
        config: MOLLMBOConfig,
    ) -> str:
        """Auto-extract up to 3 data-driven patterns (§7.3).

        All rules are purely statistical — no GP required.
        Kept to ≤ 5 lessons per §8.2 state.lessons max-size.
        """
        valid  = [r for r in all_results if r.constraint_ok]
        if len(valid) < 5:
            return "<lessons>Not enough data yet.</lessons>"

        lessons: list[str] = []
        I1_vals = np.array([r.theta[0] for r in valid])
        I2_vals = np.array([r.theta[1] for r in valid])
        T_vals  = np.array([r.T_peak   for r in valid])
        a_vals  = np.array([r.delta_Q_aging for r in valid])

        # Lesson 1: temperature violation zone (protocols within valid but near limit)
        hot_mask = T_vals > config.T_max_conservative * 0.9
        if hot_mask.sum() >= 2:
            hot_I1_mean = I1_vals[hot_mask].mean()
            lessons.append(
                f"Temperature hotzone: I1 > {hot_I1_mean:.1f}A tends to push T_peak"
                f" above {config.T_max_conservative * 0.9:.0f}°C"
                f" ({hot_mask.sum()} observations)."
            )

        # Lesson 2: aging acceleration knee (via Pearson correlation I2 vs aging)
        if len(I2_vals) >= 5:
            corr = float(np.corrcoef(I2_vals, a_vals)[0, 1])
            if abs(corr) > 0.4:
                direction = "increases" if corr > 0 else "decreases"
                lessons.append(
                    f"Aging {'strongly' if abs(corr)>0.7 else 'moderately'} "
                    f"{direction} with I2 (ρ={corr:.2f})."
                )

        # Lesson 3: under-explored parameter region
        # Find which bound-quarter of the I1 range has fewest evaluations
        lo, hi = config.bounds["I1"]
        mid    = (lo + hi) / 2.0
        n_low  = (I1_vals < mid).sum()
        n_high = (I1_vals >= mid).sum()
        if n_low < n_high / 3:
            lessons.append(
                f"Under-explored: low-current region I1 < {mid:.1f}A"
                f" has only {n_low} evaluations vs {n_high} at higher currents."
            )
        elif n_high < n_low / 3:
            lessons.append(
                f"Under-explored: high-current region I1 ≥ {mid:.1f}A"
                f" has only {n_high} evaluations vs {n_low} at lower currents."
            )

        if not lessons:
            lessons.append("No strong patterns detected yet.")

        return "<lessons>\n" + "\n".join(f"- {l}" for l in lessons) + "\n</lessons>"

    def _target_region(
        self,
        pareto_results: list,
        config: MOLLMBOConfig,
    ) -> str:
        """Describe the largest gap in Pareto coverage to direct LLM search (§7.3)."""
        if len(pareto_results) < 2:
            return "<target_region>Explore broadly — too few Pareto solutions.</target_region>"

        t_vals = np.array([r.t_charge for r in pareto_results])
        T_vals = np.array([r.T_peak   for r in pareto_results])
        a_vals = np.array([r.delta_Q_aging for r in pareto_results])

        # Check for gap: no solution simultaneously fast AND cool
        fast_thresh = np.percentile(t_vals, 25)   # fastest quartile
        cool_thresh = np.percentile(T_vals, 50)    # cooler half
        fast_and_cool = sum(
            1 for r in pareto_results
            if r.t_charge <= fast_thresh and r.T_peak <= cool_thresh
        )

        lines = ["<target_region>"]
        if fast_and_cool == 0:
            lines.append(
                f"Gap: no Pareto solution achieves both t < {fast_thresh:.1f}min"
                f" AND T < {cool_thresh:.1f}°C simultaneously."
            )
            lines.append(
                "Suggestion: explore moderate I1 (5–7A) with late SOC_sw1 (> 0.45)"
                " to balance speed and temperature."
            )
        else:
            # General guidance toward improving hypervolume
            best_t = t_vals.min()
            best_T = T_vals.min()
            lines.append(
                f"Current best: t={best_t:.1f}min, T_min={best_T:.1f}°C."
            )
            lines.append(
                "Target: improve aging without sacrificing more than 5% in charging time."
            )
        lines.append("</target_region>")
        return "\n".join(lines)


# ── LLMInterface (three Touchpoints) ──────────────────────────────────────────

class LLMInterface:
    """Manages all LLM API calls for MO-LLMBO.

    Args:
        config: MOLLMBOConfig — used for bounds, param names, LLM settings

    Touchpoint 1a  get_coupling_matrices() → {"W_time": (7,7), "W_temp": (7,7), "W_aging": (7,7)}
    Touchpoint 1b  warm_start(n=15)        → list[np.ndarray (7,)]
    Touchpoint 2   generate_candidates(summary, n=15) → list[np.ndarray (7,)]
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        import openai  # deferred: graceful failure if not installed
        self.config = config
        self._client = openai.OpenAI()   # reads OPENAI_API_KEY from env

    # ── Touchpoint 1a ─────────────────────────────────────────────────────────

    def get_coupling_matrices(self) -> dict[str, NDArray]:
        """Query the LLM once for physics coupling matrices W_time, W_temp, W_aging.

        Each W is a (d, d) PSD matrix encoding the gradient coupling structure
        of the physics proxy kernel (§3.2):
            k_physics(θ, θ') = γ · ∇Ψ(θ)ᵀ W ∇Ψ(θ')

        W_ij > 0 means parameters i and j have synergistic effects on the objective.
        Called once at initialization; low temperature for deterministic output.

        Returns:
            dict with keys "W_time", "W_temp", "W_aging", each shape (d, d).
        """
        d    = self.config.dim
        names = self.config.param_names

        prompt = (
            "You are an electrochemistry expert.\n\n"
            f"A {self.config.q_nom} Ah NMC811/Graphite cell is charged with a 3-stage CC-CV protocol.\n"
            f"The protocol parameters are: {names}\n\n"
            "Provide three symmetric positive-semidefinite coupling matrices W_time, W_temp, W_aging "
            f"of size {d}×{d}. "
            "W[i][j] encodes the joint sensitivity of the two parameters on the objective:\n"
            "  - W_time:  charging speed (dominated by current magnitude)\n"
            "  - W_temp:  peak temperature (dominated by I²R, so I1,I2,I3 coupling is high)\n"
            "  - W_aging: SEI capacity loss (driven by high current at high SOC)\n\n"
            "Physical intuition:\n"
            "  - I1, I2, I3 strongly couple to each other for heat/aging (I²R term)\n"
            "  - SOC_sw1 and SOC_sw2 couple with the adjacent currents (stage duration)\n"
            "  - V_CV and I_cutoff have mild coupling with other parameters\n"
            "  - Diagonal entries represent self-sensitivity (should be largest)\n\n"
            f"Return ONLY valid JSON with exactly these keys, each a {d}×{d} nested list "
            "(no NaN, no None, values in [0, 1], diagonal dominant):\n"
            '{"W_time": [[...]], "W_temp": [[...]], "W_aging": [[...]]}'
        )

        text = self._call(prompt, temperature=self.config.llm_temperature_init)
        return self._parse_coupling_matrices(text, d)

    def _parse_coupling_matrices(self, text: str, d: int) -> dict[str, NDArray]:
        """Parse and PSD-project the three coupling matrices."""
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
        fallback = {k: np.eye(d) * 0.5 for k in ("W_time", "W_temp", "W_aging")}

        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                log.warning("Coupling matrix parse failed; using identity fallback.")
                return fallback
            try:
                raw = json.loads(m.group(0))
            except json.JSONDecodeError:
                log.warning("Coupling matrix parse failed; using identity fallback.")
                return fallback

        result: dict[str, NDArray] = {}
        for key in ("W_time", "W_temp", "W_aging"):
            if key not in raw:
                log.warning("Missing %s in LLM response; using identity.", key)
                result[key] = np.eye(d) * 0.5
                continue
            W = np.array(raw[key], dtype=float)
            if W.shape != (d, d):
                log.warning("%s shape mismatch (%s vs %s); using identity.", key, W.shape, (d, d))
                result[key] = np.eye(d) * 0.5
                continue
            if not np.all(np.isfinite(W)):
                W = np.where(np.isfinite(W), W, 0.0)
            result[key] = _project_to_psd(W)
        return result

    # ── Touchpoint 1b ─────────────────────────────────────────────────────────

    def warm_start(self, n: int | None = None) -> list[NDArray]:
        """Generate initial protocol proposals before any experiments (§5.2).

        Follows the LLAMBO warmstarting pattern: ask for n diverse protocols
        covering the full Pareto trade-off space (fast / gentle / balanced).
        No experimental data is available at this stage.

        Args:
            n: number of proposals (default: config.n_init)
        Returns:
            list of (7,) θ arrays; may be shorter than n if parsing fails.
        """
        n = n or self.config.n_init
        b = self.config.bounds

        prompt = (
            "You are an expert in lithium-ion battery charging.\n\n"
            f"Generate {n} diverse charging protocols for a {self.config.q_nom} Ah NMC811/Graphite cell.\n"
            f"Target: SOC {self.config.soc_start} → {self.config.soc_end}, ambient {self.config.t_ambient}°C\n\n"
            "Parameter bounds:\n"
            + "\n".join(f"  {name}: [{lo:.3g}, {hi:.3g}]" for name, (lo, hi) in b.items())
            + "\n\n"
            "Three objectives to cover (all minimize):\n"
            "  1. Charging time [min] — faster = higher currents\n"
            "  2. Peak temperature [°C] — cooler = lower currents\n"
            "  3. Capacity degradation [Ah] — less aging = lower I at high SOC\n\n"
            f"Design {n} protocols with diversity across the trade-off space:\n"
            f"  - {n//3} speed-optimized: high I1/I2/I3, aggressive profile\n"
            f"  - {n//3} longevity-optimized: low currents, early switching\n"
            f"  - {n - 2*(n//3)} balanced: moderate currents\n\n"
            "CRITICAL: Maintain I1 ≥ I2 ≥ I3 (decreasing current profile).\n\n"
            f"Return ONLY a JSON array of exactly {n} dicts:\n"
            '[{"I1": ..., "I2": ..., "I3": ..., "SOC_sw1": ..., "SOC_sw2": ..., "V_CV": ..., "I_cutoff": ...}, ...]'
        )

        text = self._call(prompt, temperature=self.config.llm_temperature_cand)
        return _parse_theta_list(text, self.config)

    # ── Touchpoint 2 ──────────────────────────────────────────────────────────

    def generate_candidates(
        self,
        summary: str,
        lambda_t: NDArray | None = None,
        n: int | None = None,
    ) -> list[NDArray]:
        """Generate candidate protocols conditioned on current optimisation state.

        Called every iteration. The summary string from DatabaseSummarizer
        encodes all relevant history (Pareto front, gradients, lessons, target).
        lambda_t (the current RISE weight) is injected as an explicit trade-off
        instruction so the LLM biases proposals toward the axis the GP is exploring.

        Args:
            summary:   output of DatabaseSummarizer.generate_summary()
            lambda_t:  (m,) current RISE weight vector; None → omit direction block.
                       When provided, overrides any <optimization_direction> block
                       already present in summary (avoids duplication).
            n:         number of proposals (default: config.n_cand)
        Returns:
            list of (7,) θ arrays; may be shorter than n if parsing fails.
        """
        n = n or self.config.n_cand

        # Build optional direction instruction (only if not already in summary)
        direction_block = ""
        if lambda_t is not None and "<optimization_direction>" not in summary:
            summarizer = DatabaseSummarizer(self.config)
            direction_block = "\n\n" + summarizer._optimization_direction(
                lambda_t, self.config
            )

        prompt = (
            f"{summary}{direction_block}\n\n"
            f"Based on the optimisation state above, generate {n} new charging protocols.\n\n"
            "CRITICAL instructions:\n"
            "  1. Maintain I1 ≥ I2 ≥ I3 (decreasing current profile)\n"
            "  2. Ensure SOC_sw1 < SOC_sw2\n"
            "  3. Respect the OPTIMIZATION DIRECTION block — weight your proposals "
            "toward the stated trade-off axis\n"
            "  4. Focus on the TARGET REGION and LESSONS described above\n"
            "  5. Present candidates in ascending quality order (best estimated protocol LAST)\n"
            "  6. Be diverse — avoid clustering all proposals in one region\n\n"
            f"Return ONLY a JSON array of exactly {n} dicts:\n"
            '[{"I1": ..., "I2": ..., "I3": ..., "SOC_sw1": ..., "SOC_sw2": ..., "V_CV": ..., "I_cutoff": ...}, ...]'
        )

        text = self._call(prompt, temperature=self.config.llm_temperature_cand)
        return _parse_theta_list(text, self.config)

    # ── API plumbing ───────────────────────────────────────────────────────────

    def _call(
        self,
        prompt: str,
        temperature: float,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        """Single OpenAI chat completion with exponential-backoff retry.

        Args:
            prompt:       user message content
            temperature:  sampling temperature
            max_retries:  retry on RateLimitError / APIError
            retry_delay:  initial wait seconds (doubles each retry)
        Returns:
            raw response text string
        """
        import openai

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an electrochemistry expert assisting a multi-objective "
                    "Bayesian optimization loop. Always return only valid JSON as requested."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=self.config.llm_top_p,
                    max_tokens=2000,
                )
                return response.choices[0].message.content or ""
            except openai.RateLimitError:
                wait = retry_delay * (2 ** attempt)
                log.warning("Rate limit hit; retrying in %.0fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            except openai.APIError as exc:
                log.error("OpenAI APIError: %s", exc)
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

        return ""   # exhausted retries