# Equation Audit

**Scope:** equations compiled by `main.tex` and their correspondence to the
implementation or archived reporting pipeline.

| Topic | Revision and implementation correspondence | Class | Optimizer rerun |
|---|---|---:|---:|
| Protocol and stage time | Keeps the five implemented coordinates, the 0.80 target SOC, the hard SOC-sum condition, and reference-C-rate current scaling. | B | No |
| Thermal objective | States the implemented `1.10` multiplier as an alignment factor, not a calibration. | B | No |
| Degradation quantity | Replaces capacity-fade wording with the exact empirical computation and `D_chg` (a.u.); it remains an uncalibrated ranking proxy. | A/B | No |
| Log transformation | Makes the logarithms dimensionless with `t_ref=1 s` and `D_ref=1 a.u.` and preserves the code floors `max(t/t_ref,1)` and `max(D/D_ref,1e-12)`. | B | No |
| Dynamic normalization | States that the ideal and feasible historical ranges are recomputed before each GP fit; a range below 5% of the fixed reporting range is replaced by that full range. | B | No |
| Absolute ideal gap | Retains the absolute value because it is present in the code. It is redundant for current feasible observations but was not removed from the algorithm. | B | No |
| Tchebycheff target | Keeps the implemented augmented target and `eta=0.05`; the 30 Riesz directions and construction settings are reported once in the reproducibility table. | B | No |
| Matérn GP and EI | Corrects the ARD distance to `sum_d ((x_d-x'_d)/ell_d)^2`; standard kernel and EI definitions are unnumbered to reduce formula clutter. | A/B | No |
| Warm-start portfolio | Records the implemented clipped LLM score, SOC-boundary penalty, monotone-current bonus, and normalized-space diversity coefficient `0.45`. | B | No |
| Posterior-covariance lift | Uses uniform feasible Sobol anchors, posterior covariance in numerator and denominator, a `1e-12` variance floor, and a `0.025` mean-absolute anchor-shift budget. Only the acquisition mean changes. | B | No |
| Lift bound | Restricts the uncertainty bound to the posterior-covariance construction: `|delta(x)| <= s_hat sigma_z(x)`. It is not generalized to a prior-kernel numerator. | B | No |
| Reporting hypervolume | Separates fixed benchmark reporting boxes from dynamic optimizer normalization and states that unclipped sHV can exceed one and is comparable only within a parameterization. | B | No |

## Implementation evidence

- Current files inspected: `pybamm_simulator.py`, `llmbo/optimizer.py`,
  `llmbo/acquisition.py`, `llmbo/region_lifted_gp.py`,
  `llmbo/scalarization.py`, and `llmbo/warmstart_selector.py`.
- Main-run summaries record zero accepted/effective mean lifts and 16
  `force_pool` influence events per run; the main acquisition mean is plain.
- Same-batch summaries record 12 active lift events per Region/Full run.
  Historical commit `910ae9d` matches their telemetry and passes the LGBO
  coupling into acquisition before selection. The manifests do not record a
  commit or source hash, so this is reconstruction evidence rather than
  archive-recorded provenance.
- In that historical path, the adaptive score is
  `clip(max(0.35, 0.85*clip(s_raw)*a_width*a_repeat*a_late), 0, 1)`;
  it is a deterministic guidance score, not a probability. This code-level
  detail is kept here rather than expanded in the paper.

## Class-C equation changes not made

Changing the absolute gap, normalization policy, lift covariance source,
anchor weighting, score gates, shift budget, proxy equation, constraints, or
penalties would change the algorithm or objective. Such changes require new
matched optimization runs and are not part of this revision.
