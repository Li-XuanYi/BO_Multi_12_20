# Manuscript Revision Audit

**Audit date:** 2026-07-30
**Scope:** source graph compiled by `main.tex`, archived run summaries, and
implementation versions needed to interpret telemetry
**Status:** Phase 1 was completed before manuscript edits; this file now also
records the corrected execution interpretation used by the revision

## 1. Audit basis and evidence boundary

The active manuscript graph is:

- `main.tex`
- `fig_llambo_mo_framework.tex`
- `Section/introduction.tex`
- `Section/related_work.tex`
- `Section/problem.tex`
- `Section/method.tex`
- `Section/experiments.tex`
- `Section/conclusion.tex`
- `Section/experiment_values.tex`
- the figures and tables referenced by those files
- the hand-written bibliography embedded in `main.tex`

`IEEE_TTE_Paper_Structure.tex`, `IEEE-LaTeX/llambo_mo.tex`, and the two
`ecker2015_nhv_*_values.tex` files are legacy or inactive sources and are not
part of the current compilation graph. Their numerical conventions must not be
mixed into the active manuscript without an explicit provenance conversion.

This audit also checked the implementation and archived experiment summaries
needed to distinguish a textual correction from an algorithmic change. The
central evidence rule for the revision is:

> Only simulator outputs and archived run telemetry are treated as objective
> evidence. LLM output is a proposal or guidance signal, not an objective
> observation. The degradation quantity is an uncalibrated protocol-level
> proxy and must not be reported as measured capacity loss.

## 2. Revision classes

| Class | Meaning | Action in this revision |
|---|---|---|
| A | Wording, organization, notation, figure/table placement, citation, or formatting; no numerical or algorithmic change | Implement and compile |
| B | Mathematical clarification that exactly preserves the implementation and archived numerical results | Implement, document the implementation correspondence, and compile |
| C | Algorithm, objective, constraint, hyperparameter, data, or evaluation change that would alter an optimization run | Do not implement silently; record as requiring reruns or author input |

## 3. Pre-revision structure and required restructuring

The pre-revision manuscript had no dedicated Related Work section and separated
experimental setup from results:

1. Introduction
2. Battery Fast-Charging Problem and Evaluation Model
3. Proposed LLMBO-MO Framework
4. Experimental Setup
5. Results and Discussion
6. Conclusion

The revised manuscript should use the following six-section structure:

1. Introduction
2. Related Work
   - Battery Fast-Charging Optimization
   - Multiobjective Bayesian Optimization
   - LLM-Assisted Bayesian Optimization
   - Research Gap and Positioning
3. Fast-Charging Optimization Problem
   - Protocol Parameterization
   - Objectives and Constraints
   - Electrochemical--Thermal Evaluation Model
   - Degradation Proxy
4. LLM-Augmented Multiobjective Bayesian Optimization
   - Design Principles and Workflow
   - Objective Transformation and ParEGO Scalarization
   - LLM Warm Start
   - Weight-Conditioned Region Guidance
   - Bounded Posterior-Covariance Mean Lift
   - Complete Procedure
5. Experimental Validation
   - Models, LLM Configuration, and Protocol
   - Baselines and Metrics
   - Main Results
   - Same-Batch Ablation
   - Pareto and Protocol Analysis
   - Runtime, Reproducibility, and Evidence Limitations
6. Conclusion

Required structural actions:

- Merge the optimization problem, electrochemical model, thermal model, and
  degradation proxy into one coherent problem section.
- Remove the standalone black-box-interface narrative; describe the simulator
  interface only where it supports reproducibility.
- Compress standard GP, EI, and ParEGO background.
- Expand the implementation-faithful derivation of the posterior-covariance
  mean lift.
- Move the laboratory platform photograph to the evidence-limitations
  subsection, after the quantitative results.
- Combine setup and results under one Experimental Validation section.

These are Class A changes.

## 4. Evidence-critical findings

### 4.1 Main matched benchmark archives

The two matched five-seed benchmark archives use 56 evaluations per run
(6 initialization evaluations and 50 BO iterations). Their current reported
sample means and sample standard deviations are:

- Chen2020 parameterization: NSGA-II
  `0.3222 +/- 0.0237`, ParEGO `0.3853 +/- 0.0094`, and LLMBO-MO
  `0.3835 +/- 0.0079`; LLMBO-MO exceeds ParEGO in 2 of 5 seeds.
- Ecker2015 parameterization: ParEGO `1.5866 +/- 0.0130` and LLMBO-MO
  `1.8684 +/- 0.0027`; LLMBO-MO exceeds ParEGO in 5 of 5 seeds.

The archived telemetry for both main LLMBO-MO benchmark configurations records
50 region-lift attempts and **0 accepted region lifts** per run. The relevant
configuration fields are `region_lift_apply_override=False`,
`enable_gp_llm_coupling=False`, and `enable_acq_prior_coupling=False`.
Nevertheless, the `force_pool` option adds region-conditioned candidates to the
plain-EI pool during the first 16 BO iterations, and the archive records
`region_pool_influenced_acquisition_count=16` for every main run. Consequently:

- The main benchmark data do not demonstrate an effect of the
  posterior-covariance lift.
- The main runs evaluate screened LLM initialization plus early
  preference-conditioned candidate-pool expansion followed by plain EI; they do
  not evaluate an applied posterior-mean override.
- The Chen2020 result must not be described as a mean improvement over ParEGO.
- The Ecker2015 result supports an empirical advantage for the evaluated
  complete preset under a joint change of battery parameterization and LLM
  backend; it does not isolate transfer across cell parameterizations.
- Any statement attributing either main result specifically to the region-lift
  mechanism is unsupported and must be removed or qualified.

This is a claim correction (Class A), not an experiment change.

### 4.2 Same-batch ablation archive

The four ablation variants share the same five seeds, 56-evaluation budget, and
reporting pipeline:

- Baseline: `0.3836 +/- 0.0113`
- Warm start: `0.3902 +/- 0.0087`
- Region guidance: `0.3862 +/- 0.0125`
- Full: `0.3900 +/- 0.0067`

Warm start has the highest mean; Full has the smallest sample standard
deviation. The adaptive posterior-covariance variants record:

- 12 active posterior-mean lifts in every Region and Full run (the first 12
  BO iterations);
- 3--8 iterations in which lifted and plain EI choose different candidates,
  depending on variant and seed;
- zero events in the archived
  `region_pool_influenced_acquisition_count` field for all ten Region/Full
  runs.

Historical run-version code matching this telemetry passes the LGBO
posterior-covariance coupling into the acquisition function before candidate
selection. The lifted candidate is therefore evaluated whenever it differs
from the plain-EI candidate. The archived fields
`region_lift_apply_override=False` and
`region_lift_external_influence_mode=diagnostic_only` govern a separate
generic post-acquisition/region-pool path; they do not disable the
pre-acquisition LGBO coupling. The zero region-pool count is therefore
expected. Consequently:

- Region is an active posterior-mean-lift variant with six random initial
  points.
- Full combines six LLM-selected initial points with the same active lift.
- The four-way archive is a controlled same-batch component comparison, but
  its five-seed numerical differences should be reported descriptively rather
  than as a general causal or inferential result.

No significance or equivalence claim is justified with five seeds.

The run manifest does not record a git commit or source hash. Commit
`910ae9d` is a post-run historical version whose acquisition flow and
telemetry fields match the archive; it is supporting reconstruction evidence,
not an archive-recorded provenance identifier.

### 4.3 Physical platform

The optimization and quantitative validation are simulation-only. The
laboratory charging platform was not used to generate the benchmark objectives
or validate the reported protocols. The photograph may be retained only as
planned replay/deployment context. It must not be presented as experimental
validation of the numerical results.

### 4.4 Degradation quantity

The implementation evaluates

`D_chg = 100 * Q_eff / Cap(mean_SOC, mean_T, mean_I)`,

where the empirical `Cap` relation is evaluated from average trajectory
quantities. This is not a calibrated measured percentage of capacity loss.
The active manuscript and plots inconsistently label it as `Q_s (%)`,
`degradation proxy (%)`, and capacity fade. The revision must use
`D_chg (a.u.)` and “protocol-level degradation proxy” throughout.

Renaming and documenting the existing computation are Class A/B changes.
Replacing or recalibrating the formula is Class C.

## 5. Equation and notation audit

### 5.1 Equations to combine or compress

The following equations are standard or fragmented and should be shortened:

- Combine the three-objective problem and its encoded voltage, temperature,
  SOC, and protocol-bound constraints into one compact optimization statement.
- Combine stage-wise SOC updates and total charging time in one display.
- Move the Riesz-energy construction to an appendix or reduce it to a concise
  implementation statement; the finite gradient procedure only
  approximately minimizes the energy.
- Compress the standard GP posterior and EI definitions unless a term is needed
  for the covariance-lift derivation.
- Combine the EI definition and its standardized improvement variable into one
  display.

All are Class A if the implemented quantities are unchanged.

### 5.2 Objective transformation and scalarization

Current issues:

- `log10(t_c)` and `log10(Q_s)` take logarithms of dimensional quantities.
- The text describes a fixed nominal ideal point, whereas the implementation
  updates componentwise transformed minima and ranges from all feasible
  observations at every iteration.
- The implementation uses an absolute ideal gap.

Implementation-faithful clarification:

- Write dimensionless transforms
  `log10(t_c / t_ref)` and `log10(D_chg / D_ref)`, with
  `t_ref = 1 s` and `D_ref = 1 a.u.` so the numerical values are unchanged.
- State explicitly that normalization and the componentwise ideal point are
  recomputed from the currently feasible observations before each GP fit.
- Retain the absolute gap because the code uses it, while explaining that the
  dynamically updated componentwise minimum makes the feasible observed gaps
  nonnegative at the update time.

These are Class B. Removing the absolute value from the implementation or
switching to fixed normalization is Class C and requires rerunning all affected
experiments.

### 5.3 Matérn-5/2 kernel

The current display divides by a length scale twice: the distance already
contains ARD length scales and the kernel formula divides by another scalar
`ell`. Use

`r^2 = sum_d ((theta_d - theta'_d) / ell_d)^2`

and

`k = sigma_f^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)`.

This is a notation correction preserving the scikit-learn Matérn-5/2
implementation (Class B).

### 5.4 Warm-start selection score

The implemented deterministic portfolio selector uses:

- clipped LLM-reported score;
- a soft-limit overrun penalty with weight `0.65`;
- a monotone-current bonus of `0.08`;
- a diversity term with weight `0.45`;
- at most one boundary probe.

The manuscript formula currently omits the LLM-reported score and does not
define the normalization of every term. It should be rewritten as a
deterministic heuristic in normalized decision space, not as a calibrated
probability or confidence score. This is Class B if the displayed formula
matches the implementation exactly.

### 5.5 Posterior-covariance lift

The current derivation is incomplete and mixes prior-kernel and posterior-
covariance notation. For the evaluated adaptive posterior mode, the documented
construction must use the same posterior covariance matrix in numerator and
denominator:

- uniform Sobol anchors in the accepted region;
- posterior anchor covariance `Sigma_GG`;
- protected variance
  `V_eps = max(v^T Sigma_GG v, epsilon_v)`;
- LLM-reported guidance score after clipping/scaling;
- shift strength proportional to `score / sqrt(V_eps)`;
- posterior cross-covariance `Sigma_{theta,G}`;
- the implemented mean-absolute-shift budget; the generic
  `region_lift_max_shift_std` setting is not applied in the LGBO code path;
- covariance unchanged.

For the uncapped posterior construction, covariance Cauchy--Schwarz gives an
implementation-consistent bound of the form

`|Delta mu(theta)| <= score * sigma(theta)
 sqrt(V / max(V, epsilon_v)) <= score * sigma(theta)`,

before the additional implementation caps. A `sqrt(score)` bound would not
match the implemented multiplier and must not be asserted.

The manuscript must not state that the influence necessarily decreases
monotonically with iteration, because the covariance, score, clipping, and
acceptance gates change jointly. This documentation is Class B. Changing the
lift formula, anchor weighting, covariance source, or caps is Class C.

### 5.6 Budget and algorithm

The algorithm currently calls `T` the total budget but performs initialization
and then `T` additional iterations. It also mentions selecting `n_select`
points while the implementation evaluates one point per BO iteration.
Rewrite the procedure using:

`N = n_init + n_BO = 6 + 50 = 56`,

and one selected point per BO iteration. This is Class B.

### 5.7 Symbol conflicts

Resolve the following conflicts:

- replace `Q_s` with `D_chg`;
- reserve `s_LLM` for the LLM-reported guidance score;
- avoid using `c` for both that score and Gaussian centers;
- distinguish the ideal point from the EI standardized variable;
- distinguish GP standard deviation from local-region widths;
- use one consistent symbol for normalized decision vectors.

These are Class A/B notation changes.

## 6. Language and claim audit

### 6.1 Terms to replace

Use evidence-matched phrasing:

- “safe” / “safety-guaranteed” -> “within the encoded voltage,
  temperature, SOC, and protocol limits”
- “unsafe” -> “outside the encoded limits” or “rejected by deterministic
  checks”
- “physically valid” -> “successfully simulated and within the encoded
  constraints”
- “mechanistic reasoning” -> “textual rationale referring to supplied battery
  context”
- “LLM confidence” -> “LLM-reported guidance score”
- “capacity fade (%)” -> “protocol-level degradation proxy (a.u.)”
- “thermal stress” -> “peak temperature rise” or “thermal exposure”
- “Pareto-optimal” -> “nondominated in the evaluated archive” or
  “approximate Pareto set”

### 6.2 Repeated material

The manuscript repeats deterministic filtering, fail-open behavior,
simulation-only scope, and proxy limitations in the Abstract, Introduction,
Method, Experiments, figure caption, and Conclusion. Keep each idea in:

- one compact design statement in the Introduction;
- one operational description in the Method;
- one evidence-boundary statement in Experimental Validation;
- one concise limitation in the Conclusion.

### 6.3 Abstract

The abstract should be one paragraph and contain:

- the three-objective simulation problem;
- the LLM warm-start and optional posterior-mean guidance roles;
- the fixed 56-evaluation, five-seed comparisons;
- the actual Chen2020 and Ecker2015 findings without claiming universal
  dominance;
- the same-batch ablation result without claiming causality;
- an explicit “uncalibrated degradation proxy” statement;
- three concise contributions aligned with the evidence.

## 7. Figure and table audit

### 7.1 Framework figure

`fig_llambo_mo_framework.tex` is vector-based but too dense after scaling.
Required changes:

- simplify the number of boxes and raise final text size to approximately
  8--9 pt;
- replace “safety constraints,” “valid and feasible,” and “Pareto-optimal” with
  evidence-matched terms;
- show the dynamic normalization order before GP fitting;
- distinguish the optional posterior-mean lift from plain EI fallback;
- avoid implying that every main benchmark run accepted a lift;
- shorten the caption to state the workflow and evidence boundary.

### 7.2 Main convergence figure

`Section/figures/benchmark_hv.pdf` already contains matched five-seed curves
and may be retained. Its discussion must state that curves show means and
sample standard deviations under a fixed 56-evaluation budget.

### 7.3 Pareto projections

The three 2-D projections already exist. Their labels should use
“degradation proxy (a.u.)”, “Time--proxy”, and “Temperature--proxy”.
The source is a separate GPT-4.1-mini seed-8409 archive, whereas the main
benchmark table uses DeepSeek variants. The caption or surrounding text must
state this backend difference. Relabeling/replotting from the existing archive
is Class A and does not require rerunning BO.

### 7.4 Charging profiles

The profile figure comes from the same separate archive as the Pareto
projection. The low-temperature profile is an archived feasible point but not
in the final nondominated set. Do not call all displayed profiles Pareto
protocols. State their selection role precisely.

### 7.5 Ablation figure

The existing figure is a valid same-batch ablation. Move the initialization
boundary from evaluation `6.0` to `6.5`, increase label sizes, and state that
Warm start has the highest mean while Full has the smallest sample standard
deviation. Replotting from the unchanged archive is Class A.

### 7.6 Experimental platform

Retain the processed high-resolution photograph, place it at the end of the
runtime/evidence-limitations subsection, and caption it as a laboratory
platform available for future protocol replay. Do not call it validation data.

### 7.7 Float ordering

The current PDF places Fig. 2 and Tables III--IV before their first substantive
text references. Reorder source references/floats so each item is introduced
before it appears.

## 8. Citation and related-work audit

The current manuscript lacks a dedicated Related Work section. Existing
citations should be retained unless demonstrably irrelevant. The new section
must distinguish:

- battery fast-charging optimization and closed-loop protocol search;
- scalarized and hypervolume-based MOBO;
- LLM-assisted BO mechanisms;
- transfer BO and the paper's actual evidence gap.

High-priority additions supported by primary publication pages include:

- Gardner et al., constrained BO;
- Daulton et al., qEHVI and qNEHVI;
- Tighineanu et al., GP transfer for BO;
- recent battery fast-charging BO work by Dong et al., Zhu et al., Jiang et
  al., and Jeong et al.;
- LABO as a contrasting LLM-as-low-fidelity mechanism.

Adding citations and a transparent “not evaluated as a baseline” limitation is
Class A. Adding qEHVI/qNEHVI or other new numerical baselines is Class C.

The anonymous transfer-optimization manuscript supplied as a reference may
guide organization and terminology, but it must not be cited as a formally
published paper unless authoritative bibliographic metadata is available.

## 9. Missing reproducibility information

Add one compact reproducibility table. Values verified in archived
configurations include:

- total evaluations: 56 (`6 + 50`);
- seeds: 8409--8413;
- LLM models: `deepseek-v3` for Chen2020 and same-batch ablation,
  `deepseek-v3-thinking` for Ecker2015;
- API style: OpenAI-compatible endpoint;
- temperature: 0;
- LLM samples per call: 1;
- warm-start pool: 16;
- warm-start maximum tokens: 2500;
- maximum retries: 3;
- GP kernel: Matérn-5/2 with normalized targets;
- GP nugget/alpha: `1e-6`;
- GP optimizer restarts: 5;
- ParEGO weight count: 30;
- Riesz settings: `s=2`, 300 update steps, learning rate `0.005`,
  seed 42;
- same-batch adaptive region anchors: 64;
- configured generic minimum guidance score: 0.6, but the LGBO path does not
  enforce this threshold;
- adaptive effective-score floor in the same-batch posterior mode: 0.35;
- posterior variance floor: `1e-12`;
- generic maximum standardized shift configured as 0.25 but not used by the
  LGBO path;
- active LGBO mean-absolute-shift budget: 0.025;
- same-batch lift window: first 12 BO iterations.

Items not present in the archive must be reported as “not recorded,” not
inferred. API keys and secrets must never be reproduced.

## 10. Changes that require rerunning experiments

The following Class C actions are explicitly excluded from the text-only
revision:

- removing the absolute ideal-gap operation from code;
- changing fixed versus dynamic normalization;
- changing the posterior-covariance lift equation, covariance source, anchors,
  acceptance gates, or caps;
- replacing or calibrating the degradation proxy;
- changing voltage, temperature, SOC, current, or stage constraints;
- adding qEHVI, qNEHVI, Ecker2015 NSGA-II, or other new baselines;
- increasing the number of seeds or making inferential statistical claims;
- isolating battery-parameter transfer from the LLM-backend change;
- changing the LLM backend and rerunning matched transfer experiments;
- adding physical-platform measurements or long-term cycling validation;
- changing any hyperparameter that affects archived trajectories.

These items will be listed in `change_log.md` as deferred rather than silently
implemented.

## 11. Phase-1 decision

The manuscript can be substantially improved through Class A and
implementation-faithful Class B changes without rerunning the optimizer.
However, the revision must narrow its central empirical claim:

> The current archives support a warm-start contribution and, in the main
> benchmark preset, early preference-conditioned candidate-pool expansion under
> plain EI. The same-batch Region and Full variants actively apply the
> posterior-covariance lift during the first 12 BO iterations; Warm has the
> highest observed mean and Full has a nearly identical mean with the smallest
> sample deviation. These five-seed outcomes are descriptive, and the lift is
> not exercised in the matched main benchmarks. The archives do not provide
> physical or calibrated-degradation validation.

All subsequent revisions must preserve this boundary.

## 12. August 7 sensitivity-experiment addendum

Two previously deferred questions now have matched five-seed evidence:

- Objective preprocessing was rerun with dynamic min--max, z-score scaling,
  and no objective scaling. The three arms share initialization and weights;
  all retain decision-variable scaling and GP target standardization.
- The posterior-mean budget was swept with fixed initialization, weights, and
  archived region payloads, so only `B_mu` changes and no online LLM call is
  made.

The manuscript may report the resulting sample summaries descriptively. It
must not claim statistical significance, universal robustness, or a generally
optimal preprocessing or budget value.
