# Paper Source Traceability

This file maps the current LLMBO-MO paper and engineering report claims to code or archived experiment evidence. Paths are relative to the `New_LLMBO/` repository root.

## Main Documents

- Paper draft: `paper/main.tex`
- Framework figure: `paper/fig_llambo_mo_framework.tex`
- Engineering report: `paper/LLMBO_experiment_report.md`

## Framework and Implementation Evidence

| Paper claim | Code evidence |
|---|---|
| Closed-loop optimizer orchestration | `llmbo/optimizer.py` (`BayesOptimizer.run`, `run_initialization`, `run_optimization_loop`) |
| LLM WarmStart generation and response parsing | `llm/llm_interface.py`, `llm/warmstart_prompt.py` |
| WarmStart repair, diversity, and portfolio selection | `llmbo/warmstart_selector.py` |
| Riesz weights and augmented Tchebycheff scalarization | `llmbo/scalarization.py`, `llmbo/riesz_cache.py`, `llmbo/optimizer.py` |
| GP surrogate and region-lifted prediction | `llmbo/gp_model.py`, `llmbo/region_lifted_gp.py` |
| EI acquisition, guidance bonus, and risk penalty | `llmbo/acquisition.py` |
| Observation database, Pareto set, and HV bookkeeping | `DataBase/database.py` |
| SPMe/thermal/aging simulator | `pybamm_simulator.py` |
| Variable bounds, reference points, and dSOC constraint | `utils/constants.py` |

## Retained Favorable Experiment Evidence

| Main-text use | Source path | Audited values | Reporting verdict |
|---|---|---|---|
| Chen2020 seed8409 advantage case | `Compare_Exp/reports/2026-05-12_llmbo_mo_advantage_report/evidence_manifest.json` | GPT-4.1-mini tuned LLMBO-MO HV `0.3848255592`; ParEGO reference HV `0.3523110937`; delta `+0.0325144655`; both `n_total=56` | Suitable as a representative same-budget Chen2020 case study. Do not generalize to all Chen2020 five-seed archives. |
| Chen2020 plotted HV archive | `Compare_Exp/experiment_records/（HV）05-03/manifest.json` | Confirms seed8409 plotted centerline and archive source mapping | Use only as the figure/source map for the retained case. |
| Ecker2015 five-seed comparison | `Compare_Exp/experiment_records/Ecker2015_HV05-12/curve_data/final_summary.json` | LLMBO-MO HV `1.8684172233 ± 0.0024423480`; ParEGO HV `1.5865626976 ± 0.0116208304`; seeds `8409--8413`; `max_evals=56` | Strongest multi-seed favorable result for the main paper. |
| Four-group ablation | `Ablation_Exp/Ablation523_4group/combined_4group_results.json` | Baseline `0.383635`; WarmStart `0.390242`; LLM_Region `0.386211`; Full LLMBO `0.393196` | Supports positive contributions from WarmStart and LLM-region guidance, with full LLMBO best on mean HV. |
| Runtime context | `Compare_Exp/experiment_records/computational_time_3algo_5seeds_50iter_2026_05_12/computational_time_report.json` | NSGA-II `194.3±11.8s`; ParEGO `252.4±17.7s`; LLMBO-MO `440.3±31.3s` | Report only as computational caveat because it uses DeepSeek-V3 and does not form the core favorable claim. |

## Representative Pareto Points

Source: `Compare_Exp/reports/2026-05-12_llmbo_mo_advantage_report/evidence_manifest.json`.

| Method | Fast point | Balanced point | Conservative point |
|---|---|---|---|
| LLMBO-MO GPT-4.1-mini tuned | `(2880.0, 7.5669641913, 1.2606511929)` | `(6112.0, 2.8574864297, 0.5713455871)` | `(7200.0, 1.5287231422, 0.6401361427)` |
| ParEGO reference | `(3304.0, 6.3148055686, 1.0239001603)` | `(5290.0, 3.2018496235, 0.6208047252)` | `(7109.0, 1.5372096262, 0.6514006577)` |

These points support qualitative Pareto-front discussion. The fast point is a speed-focused trade-off, while the balanced and conservative points show lower temperature or degradation for LLMBO-MO relative to the selected ParEGO representatives.

## Reporting Guardrails

- Chen2020 seed8409 is a representative case study, not a blanket five-seed superiority claim.
- Ecker2015 is the strongest five-seed favorable result in the current archive.
- ORegan2022-labeled `Box_Fig/demo_data` results must not be described as Chen2020 evidence.
- DISK/PIMD may be used only as external supporting context when evaluation-budget caveats are explicit.
- Old Chen2020 mean-HV wording using `0.3872` vs `0.3763` is intentionally removed from the main paper.
