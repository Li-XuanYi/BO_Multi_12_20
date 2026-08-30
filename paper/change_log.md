# Revision Change Log

## Class A: wording, structure, citations, and presentation

- Reorganized the paper into six sections: Introduction, Related Work,
  Problem, Method, Experimental Validation, and Conclusion.
- Rewrote the abstract and discussion to lead with the supported results and
  removed repeated implementation and limitation text.
- Reduced the compiled manuscript from about 4,267 to 2,836 words (about 34%)
  while retaining the key results, equations, references, and evidence limits.
- Added a focused Related Work section covering battery fast-charging BO,
  ParEGO and hypervolume-based MOBO, LLM-assisted BO, and transfer BO.
- Standardized `D_chg` as an uncalibrated proxy in arbitrary units across
  text, tables, and plots.
- Relabeled and regenerated benchmark, Pareto, ablation, and charging-profile
  figures from existing archives; no optimization trajectory was rerun.
- Corrected the ablation boundary to 6.5 and increased plot text sizes.
- Moved the laboratory platform image to the evidence-limitations subsection.
  `figures/experiment_platform.png` is byte-identical to the user-specified
  5760x4320 source; the paper uses a 2400x1800, 600-dpi print derivative.
- Distinguished the separate GPT-4.1-mini qualitative archive from the
  DeepSeek main benchmark.

## Class B: implementation-faithful mathematical clarification

- Added dimensionless reference scales and the implemented log floors.
- Clarified dynamic ideal/range recomputation and retained the code's absolute
  ideal gap.
- Corrected the ARD Matérn distance and made standard formulas unnumbered.
- Replaced generic warm-start prose with the implemented portfolio score.
- Wrote the implemented empirical degradation proxy and narrowed its meaning.
- Replaced nonmatching lift formulas with the posterior-covariance
  construction, mean-absolute shift budget, unchanged covariance, and valid
  posterior-mode bound.
- Corrected the same-batch execution interpretation: the LGBO mean lift is
  active before acquisition selection for the first 12 BO iterations.

## Class C: rerun experiments added in the present revision

- Restored explicit routing of `minmax`, `zscore`, and `none` objective
  preprocessing and added a regression test.
- Ran 15 matched Chen2020 preprocessing experiments: five fixed seeds, a
  shared 3-LLM + 3-random initialization per seed, 50 plain-EI steps, and
  `deepseek-v4-flash` with thinking disabled.
- Ran 25 fixed-payload Chen2020 replays for
  `B_mu = {0.005, 0.0125, 0.025, 0.05, 0.1}`. Initialization, weights, and
  archived region payloads are fixed within each seed; no online LLM call is
  made in this sweep.
- Replaced the earlier three-seed short-budget plot with a two-panel,
  five-seed sensitivity figure using sample standard deviations.

## Class C: still deferred

- Any further change to objectives, proxy calibration, constraints, penalties,
  scalarization, lift equations, gates, caps, anchors, or acquisition settings.
- New qEHVI/qNEHVI or Ecker2015 NSGA-II comparisons.
- More seeds or inferential significance/equivalence claims.
- A matched Chen2020/Ecker2015 transfer study with one fixed LLM backend.
- Physical-platform replay, long-term cycling, or calibrated degradation
  validation.

The two sensitivity studies above were rerun; the remaining Class-C items were
not applied.
