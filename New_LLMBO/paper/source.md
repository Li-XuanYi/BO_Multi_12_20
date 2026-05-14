# Paper Draft Source Traceability

本文档记录 `paper_draft.md` 各章节内容对应的源文件（代码、实验数据、报告）。

路径均为相对于 `New_LLMBO/` 的相对路径。

---

## I. INTRODUCTION

- 文献综述部分：根据 `paper_draft.md` 原有内容，用户已写好，无代码来源
- 贡献点（Contributions）中的技术描述对应代码：
  - "two-touchpoint architecture" → `llm/llm_interface.py` (LLMInterface 类，§A–§G)
  - "GP-LLM coupling mechanism" → `llmbo/gp_model.py` (LLMPreferenceCoupling, predict_with_coupling, build_preference_coupling)
  - "stagnation-aware acquisition" → `llmbo/acquisition.py` (sigma_scale = 1.0 + 0.20 * min(stagnation_count, 3))

## II. PROBLEM FORMULATION

- 用户已写好，技术细节对应：
  - 决策变量定义和边界 → `utils/constants.py` (DEFAULT_BOUNDS, DSOC_SUM_MAX)
  - 约束 dSOC1 + dSOC2 ≤ 0.70 → `utils/constants.py` (dsoc_sum_violates_limit, project_dsoc_pair)
  - 3-CC protocol 参数表 → `README.md` Decision Variables 章节

## III. ELECTROCHEMICAL-THERMAL-AGING MODEL

- 用户已写好，技术细节对应：
  - SPMe 模型实现 → `pybamm_simulator.py` (PyBaMMSimulator 类)
  - 热模型耦合 → `pybamm_simulator.py` 中 SPMe + lumped thermal 的 PyBaMM 配置
  - 老化模型 → `pybamm_simulator.py` 中 empirical / SEI aging 模型
  - LG INR21700-M50 参数 → PyBaMM 内置 Chen2020 参数集

---

## IV. PROPOSED LLAMBO-MO

### A. Motivation and Framework Overview

- 整体架构描述 → `README.md` Architecture + Workflow 章节
- 优化主循环编排 → `llmbo/optimizer.py` (BayesOptimizer.run(), run_optimization_loop())
- 两阶段 Touchpoint 设计 → `llm/llm_interface.py` (generate_warmstart_candidates 为 Touchpoint 1b, query_iteration_guidance 为 Touchpoint 2)

### B. Riesz s-Energy Weight Generation and Augmented Tchebycheff Scalarization

**Eq. (9) Log-transform:**
- `llmbo/scalarization.py` → `log_transform_objectives()` (line 24–33)
  - f̃₁ = log₁₀(time), f̃₂ = temp_K, f̃₃ = log₁₀(aging)

**Eq. (10) MinMax normalization:**
- `llmbo/scalarization.py` → `compute_objective_preprocess_context()` (line 70–)
- `llmbo/optimizer.py` → `_update_dynamic_bounds()`, `_y_tilde_min`, `_y_tilde_max` 字段
- min_range_floor = 5% → `llmbo/scalarization.py` → `apply_min_range_floor()` (line 56–67)

**Eq. (11) Augmented Tchebycheff:**
- `llmbo/scalarization.py` → `compute_tchebycheff_from_raw_with_ideal()`
- η = 0.05 → `llmbo/optimizer.py` DEFAULT_CONFIG["eta"] = 0.05 (line 133)

**Eq. (12–13) Riesz s-energy:**
- `llmbo/riesz_cache.py` → `load_or_generate_riesz()` (完整缓存机制)
- 权重生成核心 → `llmbo/optimizer.py` → `generate_riesz_weight_set()` 函数
- Das-Dennis H=10, s=2.0 → DEFAULT_CONFIG: riesz_n_div=10, riesz_s=2.0 (line 113–114)
- 梯度下降参数 → riesz_n_iter=300, riesz_lr=5e-3 (line 115–116)
- 权重采样策略 → weight_strategy="riesz_relaxed_cycle", weight_sampling_mode="cycle_without_replacement" (line 118–122)
- 缓存目录 → `.riesz_cache/`

### C. Gaussian Process Surrogate Model

**Eq. (14) GP posterior:**
- `llmbo/gp_model.py` → `MaternGPModel.predict()` (line 291–299)

**Eq. (15) Matérn 5/2 kernel:**
- `llmbo/gp_model.py` → `MaternGPModel.fit()` (line 215–289)
  - kernel 构造: ConstantKernel * Matern(nu=2.5) + WhiteKernel (line 233–244)
  - ARD: length_scale=np.ones(5), length_scale_bounds=(1e-3, 1e3)
  - n_restarts_optimizer=5 → DEFAULT_CONFIG["gp_n_restarts_optimizer"] (line 108)
- 输入归一化 → `_normalize_X()` (line 535–537): (X - lo) / (hi - lo)
- target_transform_mode="none" → DEFAULT_CONFIG (line 109)

### D. LLM Integration: Two-Touchpoint Architecture

#### 1) Touchpoint 1: Warmstart Candidate Generation

- LLM 调用 → `llm/llm_interface.py` → `LLMInterface.generate_warmstart_candidates()` (line 1269–1423)
- Warmstart prompt 渲染 → `llm/warmstart_prompt.py` → `render_warmstart_prompt()`, `WarmStartPromptContextBuilder`
- prompt context level = "full" → DEFAULT_CONFIG["warmstart_context_level"] = "full" (line 101)
- 响应解析与验证 → `llm/llm_interface.py` → `ResponseParser.parse_candidates()` + `validate_candidate()` (line 527–554, 357–379)
- Portfolio selection → `llmbo/warmstart_selector.py` → `select_warmstart_portfolio()`
  - diversity_weight=0.45, soft_penalty_weight=0.65, monotone_bonus=0.08 → DEFAULT_CONFIG (line 85–87)
- 物理启发式回退 → `llm/llm_interface.py` → `PhysicsHeuristicFallback.physics_informed_warmstart()` (line 587–639)
  - 7 个先验点 + 8 个极端方向点 + LHS 补全
- LLM API 调用封装 → `llm/llm_interface.py` → `LLMCaller._openai_call()` (line 185–245)
  - system prompt: "You are an expert in lithium-ion battery fast charging optimization..."

#### 2) Touchpoint 2: Iteration-Level Guidance

- LLM guidance 查询 → `llm/llm_interface.py` → `LLMInterface.query_iteration_guidance()` (line 1075–1110)
- Guidance prompt 渲染 → `llm/iteration_prompt.py` → `render_iteration_guidance_prompt()`
- 内联 prompt 模板（legacy）→ `llm/llm_interface.py` → `_build_iteration_prompt()` (line 660–770)
  - 包含 w_vec 解读、few-shot 历史、Pareto 上下文
- Pareto 上下文构建 → `DataBase/database.py` → `ObservationDB.to_llm_context()`
- Uncertainty hotspots → `llmbo/optimizer.py` → `_compute_uncertainty_hotspots()` (Sobol 探测 + GP std 排序)
- Few-shot 上下文（top-3, worst-2）→ `_build_iteration_prompt()` 中 line 699–738
- 响应解析 → `ResponseParser.parse_guidance()` (line 420–431)
  - 输出 IterationGuidance dataclass: mode, confidence, point/lb/ub
- Guidance state 构建 → `llmbo/optimizer.py` → `_build_guidance_state()`
- Heuristic 回退 → `LLMInterface._fallback_iteration_guidance()` (line 975–1009)

### E. GP-LLM Coupling via Acquisition-Time Mean Shift

**Eq. (16) λ coupling strength:**
- `llmbo/gp_model.py` → `build_preference_coupling()` (line 406–476)
  - base_lambda = confidence / sqrt(posterior_variance) (line 447)
  - annealed_lambda = base_lambda * (decay_rate ** t) (line 448)
  - clamped to [lambda_min, lambda_max] (line 449)
- decay_rate=0.75, lambda_max=1.0 → DEFAULT_CONFIG (line 145, 143)

**Eq. (17) Coupled mean shift:**
- `llmbo/gp_model.py` → `predict_with_coupling()` (line 357–377)
  - sigma_xg_z = posterior_covariance_standardized(X_new, coupling.grid) (line 366)
  - base_z = sigma_xg_z @ coupling.weights (line 367)
  - shift_z = lambda * gate * mask * base_z (line 368–373)
  - shift_y = shift_z * y_std (line 375–376)
  - return mean - shift_y, std (line 377)

**Eq. (18) Spatial mask:**
- `llmbo/gp_model.py` → `_coupling_local_mask()` (line 593–625)
  - point mode: Gaussian kernel mask (line 599–607)
  - region mode: inside=1.0, outside=exp decay (line 609–623)

**Grid construction:**
- `llmbo/optimizer.py` → `_build_gp_llm_coupling_from_guidance()` 调用 coupling 构建
- guidance_grid_size=64, guidance_point_grid_size=25 → DEFAULT_CONFIG (line 137–138)

**Posterior covariance (不修改):**
- `llmbo/gp_model.py` → `posterior_covariance()` (line 379–404)
  - 使用 Cholesky 分解计算精确后验协方差
  - predict_with_coupling 中 std 保持不变

### F. Acquisition Function with Stagnation-Aware Exploration

**Eq. (19) EI:**
- `llmbo/acquisition.py` → `expected_improvement()` (line 638–645)
  - EI = (f_min - mean) * Φ(z) + std * φ(z)

**Eq. (20) Acquisition prior:**
- `llmbo/acquisition.py` → `AcquisitionPrior.bonus()` (line 123–135)
  - guidance_bonus: region mode 和 point mode 的距离衰减 (line 175–193)
- `llmbo/acquisition.py` → `AcquisitionPrior.risk()` (line 137–152)
  - safe_risk_weight, hard_risk_weight, monotone_risk_weight
- Score 计算 → `AcquisitionFunction.step()` (line 305–310):
  - score = normalize(log1p(EI)) + prior_bonus - risk_penalty

**Candidate pool:**
- `llmbo/acquisition.py` → `_build_candidate_pool()` (line 366–404)
  - external candidates (LLM guidance) + internal seeds + L-BFGS-B optimized + random
- L-BFGS-B 优化 → `_optimize_from_seed()` (line 406–431)
- n_restarts_optimizer=16, n_random_candidates=128 → DEFAULT_CONFIG["ei_n_restarts", "ei_n_random_samples"] (line 110–111)

**Eq. (21) Stagnation scaling:**
- `llmbo/acquisition.py` → `step()` (line 280):
  - sigma_scale = 1.0 + 0.20 * min(stagnation_count, 3)
- Stagnation detection → `DataBase/database.py` → 滑动窗口 HV 改进检测
- DEFAULT_CONFIG["enable_acq_prior_coupling"]=True (line 136)

### G. Overall Algorithm

- Algorithm 1 伪代码对应 → `llmbo/optimizer.py` BayesOptimizer:
  - Line 1: generate_riesz_weight_set → riesz_cache.py
  - Line 3–5: run_initialization() → optimizer.py (line 784–)
  - Line 6–20: run_optimization_loop() → optimizer.py (line 914–)
  - Line 7: _next_weight() → weight vector cycling
  - Line 8: _update_dynamic_bounds()
  - Line 9: database.update_tchebycheff_context() → scalarization.py
  - Line 10: gp.fit() → gp_model.py
  - Line 11: _compute_uncertainty_hotspots()
  - Line 13: llm.query_iteration_guidance() → llm_interface.py
  - Line 14: _build_gp_llm_coupling_from_guidance() → gp_model.py coupling
  - Line 15: _build_acquisition_prior() → acquisition.py AcquisitionPrior
  - Line 17: af.step() → acquisition.py
  - Line 18: simulator.evaluate() → pybamm_simulator.py
  - Line 19–20: database 更新 + stagnation 检测

---

## V. EXPERIMENTS

### A. Experimental Setup

- 两个电池参数集:
  - Chen2020 (LG INR21700-M50) → `pybamm_simulator.py` param_set="Chen2020"
  - Ecker2015 → `pybamm_simulator.py` param_set="Ecker2015" 支持
  - Ecker2015 ref/ideal point → `utils/constants.py` (ECKER2015_REF_POINT, ECKER2015_IDEAL_POINT)
- 评估预算 56 evals (6 init + 50 iter) → DEFAULT_CONFIG["max_iterations"]=50, "n_warmstart"=3, "n_random_init"=3
- 5 seeds (8409–8413) → 所有实验报告一致使用
- Ref/Ideal point → `utils/constants.py` (REF_POINT, IDEAL_POINT)
- LLM model → DEFAULT_CONFIG["llm_model"]="gpt-4.1-mini" (line 95)
- GP config → DEFAULT_CONFIG: kernel_nu=2.5, gp_alpha=1e-6, gp_n_restarts_optimizer=5

### B. Baseline Methods

- ParEGO 实现 → `llmbo/acquisition.py` → `SimpleParEGOAcquisitionFunction` (line 512–635)
  - LCB + differential_evolution
  - config: parego_lcb_variance_weight=0.5, de_population=30, de_maxiter=200
- NSGA-II → `Compare_Exp/` 目录下的 NSGA-II 实验脚本
- DISK → `Compare_Exp/run_disk_python.py` + PlatEMO bridge
- PIMD → `Compare_Exp/run_pimd_experiments.ps1` + PlatEMO bridge

### C. HV Convergence Comparison

**Table III (Chen2020, 5 seeds):**

| 数据来源文件 |
|---|

- LLAMBO-MO: `Box_Fig/demo_data/llmbo_mo_report.json` (5 seeds, canonical_hv)
  - seed 8409: 0.3783, seed 8410: 0.3826, seed 8411: 0.3939, seed 8412: 0.3886, seed 8413: 0.3742
  - mean=0.3872, std=0.0142

- ParEGO: `Box_Fig/demo_data/parego_report.json` (5 seeds, canonical_hv)
  - seed 8409: 0.3896, seed 8410: 0.3761, seed 8411: 0.3995, seed 8412: 0.3793, seed 8413: 0.3821
  - mean=0.3763, std=0.0041

- NSGA-II: `Box_Fig/demo_data/nsgaii_report.json` (5 seeds, canonical_hv)
  - mean=0.3273, std=0.0216

- DISK: `Compare_Exp/experiment_records/disk_python_Chen2020_5seeds_50evals_2026_05_11/report_5seeds.json`
  - aggregates.canonical_hv: mean=0.3091, std=0.0306

- PIMD: `Box_Fig/demo_data/pimd_report.json` (5 seeds, canonical_hv)
  - mean=0.2982, std=0.0128

**更详细的同期对比报告:**
- `Compare_Exp/experiment_records/computational_time_3algo_5seeds_50iter_2026_05_12/computational_time_report.json`
  - 包含 LLAMBO-MO / ParEGO / NSGA-II 三者的每 seed runtime + HV
  - LLAMBO-MO: mean runtime=440.3s, std=31.3s
  - ParEGO: mean runtime=252.4s, std=17.7s
  - NSGA-II: mean runtime=194.3s, std=11.8s
  - 注意: 此报告使用 deepseek-v3 模型, HV 值与 Box_Fig 略有差异（不同 LLM backend）

**优势展示报告:**
- `Compare_Exp/reports/2026-05-12_llmbo_mo_advantage_report/llmbo_mo_advantage_report.md`
  - seed8409 单 seed 详细对比（10 iter, 50 iter）
  - 代表点摘录（用于 Section V-D Pareto Front）

**Table IV (Ecker2015, 5 seeds):**

- `Compare_Exp/experiment_records/Ecker2015_HV05-12/curve_data/final_summary.json`
  - LLAMBO-MO: canonical_hv mean=1.8684, std=0.0024
  - ParEGO: canonical_hv mean=1.5866, std=0.0116
- Ecker2015 HV 收敛曲线数据:
  - `Compare_Exp/experiment_records/Ecker2015_HV05-12/curve_data/hv_convergence_parego_vs_llmbo.json`
- Ecker2015 原始实验:
  - ParEGO: `optimized_experiments/parego_ecker_5seeds_56evals_2026_05_11/`
  - LLMBO-MO: `Compare_Exp/experiment_records/Ecker2015_HV05-12/raw_sources/LLMBO-MO/`
- Ecker2015 图表:
  - `Compare_Exp/images/Ecker2015_HV05-12/ecker2015_hv_convergence_parego_vs_llmbo.png`
  - `Compare_Exp/images/Ecker2015_HV05-12/ecker2015_optimal_protocols_parego_vs_llmbo.pdf`

### D. Pareto Front Quality

- 代表点数据来源: `Compare_Exp/reports/2026-05-12_llmbo_mo_advantage_report/llmbo_mo_advantage_report.md`
  - "50 iter / LLMBO-MO / GPT-4.1-mini tuned": (2880, 7.567, 1.261), (6112, 2.857, 0.571), (7200, 1.529, 0.640)
  - "50 iter / ParEGO reference": (3304, 6.315, 1.024), (5290, 3.202, 0.621), (7109, 1.537, 0.651)
  - "NSGA-II / best single seed4": (3315, 6.714, 1.015), (4607, 3.193, 0.676), (6738, 1.831, 0.539)

- Pareto 图绘制脚本:
  - `Patero/plot_soh_pareto.py` (Chen2020 Pareto front 可视化)
  - `Compare_Exp/plot_ecker2015_optimal_protocols.py` (Ecker2015)
  - `Compare_Exp/plot_ecker2015_hv_convergence.py`

### E. Ablation Study

**Table V 数据来源:**
- `optimized_experiments/baseline_warmstart_llmgp_50iter_seed01234_2026_04_29_run1/report_5seeds.json`
  - records 中每条包含 variant, seed, canonical_hv
  - 三个 variant:
    - strict_baseline: mean=0.3700, std=0.0054
    - warmstart_plain_ei: mean=0.3751, std=0.0019
    - warmstart_region_lifted_gp: mean=0.3695, std=0.0057

- 消融实验说明: `Ablation_Exp/README.md`
  - LLMBP+WarmStart = warmstart_region_lifted_gp
  - WarmStart = warmstart_plain_ei
  - Baseline = strict_baseline

- 消融实验预设配置 → `llmbo/optimizer.py` → EXPERIMENT_PRESETS (line 231–)
  - "strict_baseline": n_warmstart=0, n_random_init=6, no LLM
  - "warmstart_plain_ei": n_warmstart=3, enable_iterative_guidance=False
  - "warmstart_region_lifted_gp_force_pool_tuned": full LLAMBO-MO

- 消融实验运行脚本:
  - `Ablation_Exp/Process/` 目录下的实验脚本
  - `tools/run_warmstart_vs_baseline.py`

### F. Computational Efficiency

- Runtime 数据 → 同 Table III 数据源:
  - `Compare_Exp/experiment_records/computational_time_3algo_5seeds_50iter_2026_05_12/computational_time_report.json`
    - aggregates.parego.runtime_s: mean=252.4s
    - aggregates.nsga2.runtime_s: mean=194.3s
    - aggregates.llmbo_mo.runtime_s: mean=440.3s
  - timing_scope: "wall time per seed, including optimizer.run and save_results"
  - 注意: 此实验使用 deepseek-v3 模型

- Runtime 对比绘图:
  - `Compare_Exp/run_computational_time_comparison.py`
  - `Compare_Exp/images/computational_time_3algo_5seeds_50iter_2026_05_12/`

---

## VI. CONCLUSION

- 纯文本总结，无独立数据源
- 结论中的数值声明对应 Section V 各表格数据

---

## 补充: 关键配置文件一览

| 文件 | 作用 |
|------|------|
| `llmbo/optimizer.py` DEFAULT_CONFIG (line 68–228) | 全部默认超参数定义 |
| `llmbo/optimizer.py` EXPERIMENT_PRESETS (line 231–) | 消融/对比实验预设 |
| `config/schema.py` | Pydantic 配置 schema |
| `utils/constants.py` | 决策变量边界、ref/ideal point |
| `.riesz_cache/` | 预计算的 Riesz 权重集合 |
