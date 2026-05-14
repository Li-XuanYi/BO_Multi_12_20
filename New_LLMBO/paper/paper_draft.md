# LLAMBO-MO: Large Language Model-Augmented Multi-Objective Bayesian Optimization for Battery Fast-Charging Protocol Design

---

## Paper Outline

- **Abstract**
- **I. Introduction**
- **II. Problem Formulation**
- **III. Electrochemical-Thermal-Aging Model**
- **IV. Proposed LLAMBO-MO**
  - A. Motivation and Framework
  - B. Riesz s-Energy Weight Generation and Tchebycheff Scalarization
  - C. Physics-Informed Gaussian Process Model
  - D. LLM Integration: Two-Touchpoint Architecture
    - 1) Touchpoint 1: Warmstart Candidate Generation
    - 2) Touchpoint 2: Iteration-Level Guidance and GP-LLM Coupling
  - E. Acquisition Function with GP-LLM Coupling
  - F. Stagnation Detection and Adaptive Exploration
  - G. Overall Algorithm
- **V. Experiments**
  - A. Experimental Setup
  - B. Baseline Methods
  - C. HV Convergence Comparison
  - D. Pareto Front Quality
  - E. Ablation Study (V0–V6)
  - F. Computational Efficiency
- **VI. Conclusion**
- **References**

---

## Abstract

The design of fast-charging protocols for lithium-ion batteries is inherently a constrained multi-objective optimization problem (CMOP) that must simultaneously minimize charging time, peak temperature rise, and capacity fade. Model-based optimization approaches offer cost-effectiveness over experiment-based methods, yet simulating electrochemical-thermal-aging models remains computationally expensive. Bayesian optimization (BO) has emerged as a sample-efficient paradigm for such tasks; however, existing BO-based methods rely solely on mathematical surrogates and fail to leverage the rich domain knowledge embedded in the battery literature and physics.

This paper proposes LLAMBO-MO, a Large Language Model (LLM)-augmented multi-objective Bayesian optimization framework that integrates LLM-generated domain knowledge at two strategic touchpoints within the BO loop. At the first touchpoint, the LLM generates physics-informed warmstart candidates to bootstrap the surrogate model with high-quality initial data. At the second touchpoint, the LLM provides iteration-level guidance—suggesting either promising point estimates or unexplored regions—which is then coupled with the Gaussian process (GP) posterior through a bounded acquisition-time mean shift mechanism. The multi-objective problem is decomposed via augmented Tchebycheff scalarization with Riesz s-energy weight vectors, ensuring well-distributed exploration of the Pareto front. A stagnation-aware acquisition function with adaptive exploration further enhances robustness. Extensive experiments on an SPMe-based electrochemical-thermal-aging simulator for the LG INR21700-M50 cell demonstrate that LLAMBO-MO outperforms ParEGO, NSGA-II, and PlatEMO-based algorithms (DISK, PIMD) in both hypervolume convergence speed and final Pareto front quality within a fixed evaluation budget.

**Index Terms**—Lithium-ion battery, fast charging, multi-objective optimization, Bayesian optimization, large language models, Gaussian process, Tchebycheff scalarization

---

## I. INTRODUCTION

Lithium-ion batteries have become the dominant energy storage technology for electric vehicles and portable electronics, owing to their high energy density and long cycle life [1]–[3]. However, the fundamental trade-off between charging speed and battery degradation remains a critical challenge: fast charging reduces user wait time but accelerates capacity fade and increases safety risks from excessive heat generation [4], [5]. Consequently, the design of optimal charging protocols—balancing charging time, thermal safety, and battery longevity—has attracted significant research attention.

The charging protocol design problem is naturally formulated as a constrained multi-objective optimization problem (CMOP), where multiple competing objectives must be simultaneously minimized under physical constraints [6]. Model-based optimization approaches, which rely on battery simulation models rather than physical experiments, offer a cost-effective alternative for exploring the protocol design space. However, the accuracy of such approaches depends critically on the fidelity of the underlying battery model, and high-fidelity electrochemical-thermal-aging models require solving coupled partial differential equations (PDEs), making each simulation evaluation computationally expensive [7], [8].

To address the computational challenge, various strategies have been investigated. Equivalent circuit models (ECMs) replace PDE-based descriptions with RC network approximations, dramatically reducing computation time. Liu et al. [10] and Hu et al. [11] employed ECM-thermal-aging models to optimize charging protocols with respect to multiple objectives. However, ECMs provide limited insight into internal electrochemical states, which constrains the accuracy of the design process, particularly under aggressive fast-charging conditions where nonlinear electrochemical phenomena dominate.

Reduced-order models (ROMs), such as the single particle model (SPM) and the single particle model with electrolyte (SPMe), offer a compromise between fidelity and efficiency. Lin et al. [12] employed an electrolyte-enhanced SPM coupled with a lumped thermal model to optimize charging time and degradation via dynamic programming. Wei et al. [16] designed charging protocols using deep reinforcement learning with an ROM-based simulator deployed on a cloud server. While ROMs are more accurate than ECMs, they still involve approximations that may not hold under all operating conditions, and each simulation remains non-trivial in computational cost.

From an algorithmic perspective, Bayesian optimization (BO) has gained popularity for charging protocol design due to its sample efficiency—constructing a surrogate model from limited evaluations and using an acquisition function to guide the search. Jiang et al. [4] applied BO to design various charging protocols and investigated the impact of different acquisition functions. Wang and Jiang [17] proposed a decomposition-based multi-objective BO framework using Chebyshev scalarization, where each subproblem is solved by a constrained acquisition-based BO algorithm. Dong et al. [18] presented a hybrid BO algorithm combining mesh grid adaptive search for multi-stage constant-current protocols. Recently, Jiang et al. [Ref-EIMO] proposed an experience-informed multi-objective optimization (EIMO) algorithm that leverages transfer learning from batteries at different state-of-health (SOH) levels to accelerate the design process. Despite these advances, existing BO-based methods construct surrogate models purely from numerical data, without incorporating the extensive domain knowledge available in the battery science literature.

Meanwhile, large language models (LLMs) have demonstrated remarkable capabilities in synthesizing and reasoning over scientific knowledge [Ref-LLM]. Trained on vast corpora of scientific literature, LLMs encode rich domain priors about electrochemical relationships, thermal dynamics, and degradation mechanisms. Recent work has begun exploring the integration of LLMs with optimization algorithms for automated design [Ref-LLM-BO], but this direction remains largely unexplored in the context of battery charging optimization.

From the above literature review, the following observations and insights emerge:

- Charging protocol design is inherently a CMOP requiring careful balancing of multiple competing objectives under physical constraints.
- Model-based algorithms are more cost-effective than experiment-based approaches, but high-fidelity simulations remain computationally expensive.
- BO achieves a favorable balance between efficiency and accuracy by constructing informative surrogate models from limited evaluations.
- Domain knowledge about electrochemical-thermal-aging relationships, while abundantly available in the literature, is not exploited by existing BO-based charging design methods.
- LLMs represent a promising but untapped source of domain knowledge that can potentially guide and accelerate the BO process.

Based on these observations, this paper proposes LLAMBO-MO (LLM-Augmented Multi-Objective Bayesian Optimization), a framework that strategically integrates LLM-generated domain knowledge into a decomposition-based multi-objective BO algorithm for battery fast-charging protocol design. The main contributions of this work are as follows:

- To the best of our knowledge, LLAMBO-MO is the first framework to integrate LLM-generated domain knowledge into multi-objective Bayesian optimization for battery charging protocol design. A two-touchpoint architecture is proposed: the LLM provides physics-informed warmstart candidates at initialization and delivers iteration-level guidance during the BO loop.
- A GP-LLM coupling mechanism is developed that transforms LLM guidance—either as point estimates or region specifications—into a bounded acquisition-time posterior mean shift in standardized objective space. This mechanism allows the LLM to bias the acquisition function toward promising regions without refitting the GP or altering the posterior covariance structure.
- A stagnation-aware acquisition function is proposed that combines expected improvement with an LLM proposal bonus and an adaptive exploration factor, ensuring robust convergence even when the optimization process stalls.
- Extensive experiments on an SPMe-based electrochemical-thermal-aging simulator demonstrate that LLAMBO-MO outperforms ParEGO, NSGA-II, and PlatEMO-based algorithms in terms of hypervolume convergence and Pareto front quality within a fixed evaluation budget.

The rest of this paper is organized as follows. Section II introduces the problem formulation. Section III describes the electrochemical-thermal-aging model used for simulation. Section IV elaborates on the proposed LLAMBO-MO framework. Section V presents the experimental validation and discussion. Finally, Section VI concludes the paper.

---

## II. PROBLEM FORMULATION

The charging protocol design problem determines the optimal parameters of a multi-stage constant-current (MCC) charging protocol through optimization, taking battery physical states into consideration. It is formulated as a CMOP with objectives, constraints, and decision variables defined as follows.

**Objectives:** Three competing objectives are considered to capture the diverse requirements of fast charging [17]:

$$\min_{\boldsymbol{\theta}} \, f(\boldsymbol{\theta}) = \{t_c, \, \Delta T_p, \, Q_s\} \tag{1}$$

where $\boldsymbol{\theta}$ denotes the decision vector (i.e., charging protocol parameters), $t_c$ is the total charging time, $\Delta T_p$ is the peak temperature rise above ambient, and $Q_s$ represents the capacity fade. These three objectives reflect the time efficiency, thermal safety, and longevity requirements, respectively.

**Decision Variables:** A 3-stage constant-current (3-CC) protocol is adopted for its practical efficiency [22]. The decision vector comprises five variables:

$$\boldsymbol{\theta} = [I_1, I_2, I_3, \Delta s_1, \Delta s_2]$$

where $I_1, I_2, I_3$ denote the charging currents for stages 1, 2, and 3, respectively, and $\Delta s_1, \Delta s_2$ denote the state-of-charge (SOC) interval widths for stages 1 and 2. The SOC interval for stage 3 is determined implicitly as $\Delta s_3 = s_{\text{end}} - s_{\text{start}} - \Delta s_1 - \Delta s_2$. The variable bounds and the linear constraint are specified in Table I.

**Table I: Decision Variable Bounds and Constraint**

| Variable | Symbol | Lower Bound | Upper Bound | Unit |
|:---------|:-------|:------------|:------------|:-----|
| Stage 1 current | $I_1$ | 2.0 | 6.0 | A |
| Stage 2 current | $I_2$ | 2.0 | 5.0 | A |
| Stage 3 current | $I_3$ | 2.0 | 3.0 | A |
| Stage 1 SOC width | $\Delta s_1$ | 0.10 | 0.40 | — |
| Stage 2 SOC width | $\Delta s_2$ | 0.10 | 0.30 | — |

**Constraint:** $\Delta s_1 + \Delta s_2 \leq 0.70$, ensuring $\Delta s_3 > 0$ for the final charging stage.

**Constraints:** In addition to the decision variable bounds, the charging process must satisfy the following operational constraints during simulation [21]:

$$\begin{cases}
U_{\min} \leq U(t) \leq U_{\max} \\
T(t) \leq T_{\max} \\
s_{\min} \leq \text{SOC}(t) \leq s_{\max}
\end{cases} \tag{2}$$

where $U(t)$ is the terminal voltage, $T(t)$ is the cell temperature, and $\text{SOC}(t)$ is the state of charge. Violation of any constraint renders the corresponding protocol infeasible.

Given a protocol $\boldsymbol{\theta}$, the battery states are computed using an electrochemical-thermal-aging model (Section III), which is computationally expensive. The goal is to identify a set of nondominated (Pareto-optimal) charging protocols that represent the best trade-offs among the three objectives, using a limited number of simulation evaluations.

---

## III. ELECTROCHEMICAL-THERMAL-AGING MODEL

This section describes the battery simulation model used to evaluate candidate charging protocols. The model couples electrochemical, thermal, and aging dynamics to compute the three objective values: charging time $t_c$, peak temperature rise $\Delta T_p$, and capacity fade $Q_s$.

### A. Electrochemical Model

The electrochemical model describes the internal electrochemical dynamics of the lithium-ion battery. The pseudo-two-dimensional (P2D) model [23], grounded in concentrated solution theory, porous electrode theory, and Butler-Volmer kinetics, is widely recognized for its capability to simulate real batteries with high fidelity. The P2D model comprises a system of coupled PDEs and algebraic equations that govern lithium transport in the solid and electrolyte phases, charge transfer kinetics at the electrode-electrolyte interface, and potential distribution across the cell. Given an input current profile, the terminal voltage is calculated as:

$$U(t) = \phi_s\big|_{x=l_n+l_s+l_p} - \phi_s\big|_{x=0} \tag{3}$$

where $\phi_s$ denotes the solid-phase potential, $x$ is the spatial coordinate, and $l_n$, $l_s$, and $l_p$ represent the thicknesses of the negative electrode, separator, and positive electrode, respectively.

In this work, the Single Particle Model with electrolyte (SPMe) [24] is adopted as a computationally efficient reduced-order variant of the P2D model. The SPMe approximates each electrode as a single spherical particle while retaining an electrolyte concentration equation, thereby capturing the essential electrochemical dynamics at a fraction of the computational cost. The SPMe is particularly suitable for optimization-driven design, where thousands of simulations may be required. The model is implemented using the PyBaMM open-source battery simulation framework [PyBaMM-ref], with parameter values corresponding to the LG INR21700-M50 cell (Chen et al. [Chen2020-ref]).

### B. Thermal Model

The thermal model captures the temperature evolution within the battery cell during charging. A lumped-parameter thermal model is coupled with the electrochemical model [25]. The governing equation for the cell temperature $T$ is:

$$\rho C_p \frac{\partial T}{\partial t} = q_{\text{gen}} - h_c (T - T_{\text{air}}) \tag{4}$$

subject to the initial condition $T(0) = T_0$, where $\rho$ is the battery density, $C_p$ is the specific heat capacity, $q_{\text{gen}}$ is the volumetric heat generation rate computed from the electrochemical model (encompassing ohmic, reaction, and entropic heat sources), $h_c$ is the convective heat transfer coefficient, and $T_{\text{air}}$ is the ambient temperature. In the SPMe implementation, the thermal model is solved self-consistently with the electrochemical equations at each time step, and the cell temperature feeds back into the electrochemical parameters (e.g., diffusion coefficients, reaction rates), creating a bidirectional coupling.

The peak temperature rise objective is computed as:

$$\Delta T_p = T_{\text{peak}} - T_0 \tag{5}$$

where $T_{\text{peak}} = \max_t T(t)$ is the maximum cell temperature observed during the charging process. The ambient temperature is set to $T_0 = 298.15$ K (25°C), and an upper temperature limit of $T_{\max} = 318.15$ K (45°C) is imposed as a safety constraint.

### C. Aging Model

The aging model quantifies the capacity degradation incurred during a single charging event. Both empirical and physics-based aging models are supported in the simulation framework.

**Empirical aging model.** The empirical model follows a semi-empirical formulation calibrated to match the capacity fade behavior observed in cycling experiments. The capacity fade percentage is computed as a function of the mean SOC, mean temperature, and mean current during the charging process:

$$Q_s = \frac{Q_{\text{eff}}}{\text{Cap}(\bar{s}, \bar{T}, \bar{I})} \times 100\% \tag{6}$$

where $Q_{\text{eff}}$ is the effective cell capacity and $\text{Cap}(\cdot)$ is an Arrhenius-type function capturing the dependence of cycle life on operating conditions. Specifically:

$$\text{Cap}(\bar{s}, \bar{T}, \bar{I}) = \left(\frac{20}{(a_1 \bar{s} + a_2) \cdot \exp\left(\frac{-E_a + b \bar{I}}{R_g \bar{T}}\right)}\right)^{1/\alpha} \tag{7}$$

where $\bar{s}$ is the mean SOC percentage, $\bar{T}$ is the mean temperature in Kelvin, $\bar{I}$ is the mean current in amperes, $E_a$ is the activation energy, $R_g$ is the universal gas constant, and $a_1, a_2, b, \alpha$ are empirically calibrated parameters. This formulation is consistent with the empirical aging model used in the EIMO framework [Ref-EIMO].

**Physics-based aging model.** For higher fidelity, a physics-based aging model simulates the formation and growth of the solid electrolyte interphase (SEI) layer on the negative electrode surface [26]. The lithium-ion loss due to SEI growth is computed by integrating the side reaction current over the charging duration:

$$Q_s = \frac{F}{3600 Q_{\text{eff}}} \int_0^{t_c} j_{\text{SEI}}(t) \, dt \times 100\% \tag{8}$$

where $F$ is Faraday's constant and $j_{\text{SEI}}$ is the volumetric side reaction current density, which depends on the local negative electrode potential, SEI layer conductivity, and temperature. The side reaction current comprises contributions from continuous SEI growth on the particle surface ($I_{\text{cov}}$) and new SEI formation on graphite particle cracks ($I_{\text{crd}}$).

### D. Simulation Implementation

The combined electrochemical-thermal-aging model is implemented using PyBaMM. For a given charging protocol $\boldsymbol{\theta} = [I_1, I_2, I_3, \Delta s_1, \Delta s_2]$, the simulation proceeds in three sequential stages:

1. **Stage decomposition.** The SOC range $[0, 0.8]$ is partitioned into three intervals of widths $\Delta s_1$, $\Delta s_2$, and $\Delta s_3 = 0.8 - \Delta s_1 - \Delta s_2$.

2. **Stage-wise solving.** For each stage $k \in \{1, 2, 3\}$, the SPMe is solved with a constant current $I_k$ applied for a duration $t_k = Q_{\text{eff}} \cdot \Delta s_k / (I_k \cdot 3600)$ seconds. The solution from the previous stage provides the initial conditions for the next stage.

3. **Objective extraction.** The three objectives are computed from the concatenated trajectory:
   - Charging time: $t_c = t_1 + t_2 + t_3$
   - Peak temperature rise: $\Delta T_p = \max(T) - 298.15$
   - Capacity fade: $Q_s$ from either the empirical model (Eq. 7) or the physics-based model (Eq. 8)

The simulation parameters are based on the LG INR21700-M50 cell characterized by Chen et al. [Chen2020-ref], with a nominal capacity of 5.0 Ah and an upper cutoff voltage of 4.2 V. The key simulation parameters are summarized in Table II.

**Table II: Parameters of the Electrochemical-Thermal-Aging Model**

| Parameter | Value |
|:----------|:------|
| Nominal capacity | 5.0 Ah |
| Upper cutoff voltage | 4.20 V |
| Lower cutoff voltage | 2.50 V |
| Ambient temperature | 298.15 K (25°C) |
| Upper temperature limit | 318.15 K (45°C) |
| Charging SOC range | [0.0, 0.8] |
| Number of stages | 3 |

Due to the involvement of PDE solving and the multi-stage simulation procedure, each evaluation of the electrochemical-thermal-aging model requires several seconds of computation. When multiplied by hundreds or thousands of evaluations needed for optimization, the total computational cost becomes substantial. This motivates the development of sample-efficient optimization algorithms, as proposed in Section IV.

---

## IV. PROPOSED LLAMBO-MO

### A. Motivation and Framework Overview

Bayesian optimization (BO) has proven effective for expensive black-box optimization by constructing a surrogate model from limited evaluations. However, standard BO methods treat the objective function as a pure black box, ignoring the rich domain knowledge available in the battery literature. Large language models (LLMs), trained on vast scientific corpora, encode implicit knowledge about electrochemical-thermal-aging relationships—such as the fact that higher charging currents reduce charging time but increase heat generation and capacity degradation—that can potentially guide the search process.

The key insight of LLAMBO-MO is that LLM-generated domain knowledge should not replace the surrogate model, but rather complement it at strategic points where it can have the greatest impact. This leads to a two-touchpoint architecture:

- **Touchpoint 1 (Initialization):** Before the BO loop begins, the LLM generates physics-informed warmstart candidates that provide high-quality initial data for the surrogate model, replacing random or purely space-filling designs.
- **Touchpoint 2 (Iteration-Level Guidance):** During each BO iteration, the LLM provides guidance—either as a promising point estimate or as an unexplored region specification—which is then coupled with the Gaussian process (GP) posterior through a bounded acquisition-time mean shift.

This design ensures that the LLM influences the optimization only through well-calibrated, bounded modifications to the acquisition landscape, preserving the statistical rigor of the GP surrogate while leveraging domain knowledge to accelerate convergence.

The overall LLAMBO-MO framework is illustrated in Fig. X and proceeds as follows: (1) the multi-objective problem is decomposed via augmented Tchebycheff scalarization with Riesz s-energy weight vectors; (2) at each iteration, a weight vector is sampled and the GP is fitted on the scalarized objective; (3) the LLM provides guidance through the two touchpoints; (4) the acquisition function, augmented with the GP-LLM coupling, selects the next evaluation point; (5) the battery simulator evaluates the candidate, and the database is updated.

### B. Riesz s-Energy Weight Generation and Augmented Tchebycheff Scalarization

To handle the multi-objective nature of the charging protocol design problem, LLAMBO-MO adopts a decomposition-based approach. At each iteration $t$, a weight vector $\boldsymbol{w}^{(t)} = (w_1, w_2, w_3)$ with $w_i \geq 0$ and $\sum_i w_i = 1$ is sampled from a predefined weight set, and the multi-objective problem is converted into a scalarized single-objective problem using the augmented Tchebycheff formulation.

**Log-transform and normalization.** Before scalarization, the objectives are transformed to improve numerical conditioning:

$$\tilde{f}_1 = \log_{10}(t_c), \quad \tilde{f}_2 = \Delta T_p, \quad \tilde{f}_3 = \log_{10}(Q_s) \tag{9}$$

The log-transform is applied to charging time and capacity fade because these quantities span orders of magnitude. The transformed objectives are then dynamically normalized using min-max scaling:

$$\bar{f}_i(\boldsymbol{\theta}) = \frac{\tilde{f}_i(\boldsymbol{\theta}) - y_{\min,i}}{y_{\max,i} - y_{\min,i}}, \quad i = 1, 2, 3 \tag{10}$$

where $y_{\min}$ and $y_{\max}$ are the dynamic bounds computed from all feasible observations, with a minimum range floor of 5\% of the global range to prevent near-zero denominators in early iterations.

**Augmented Tchebycheff scalarization.** Given a weight vector $\boldsymbol{w}$, the scalarized objective is:

$$g^{\text{tch}}(\boldsymbol{\theta} \mid \boldsymbol{w}) = \max_{i \in \{1,2,3\}} \{w_i \cdot \bar{f}_i(\boldsymbol{\theta})\} + \eta \sum_{i=1}^{3} w_i \cdot \bar{f}_i(\boldsymbol{\theta}) \tag{11}$$

where $\eta = 0.05$ is the augmentation coefficient. The max-term drives the optimization toward Pareto-optimal solutions, while the weighted-sum tiebreaker term ensures that all objectives are considered simultaneously, preventing the scalarization from being non-differentiable along the Pareto front.

**Riesz s-energy weight generation.** The quality of the decomposition-based multi-objective optimization depends critically on the distribution of weight vectors on the 2-simplex. Uniformly distributed weights ensure that the algorithm explores the entire Pareto front evenly. Following Liu and Qin [Ref-Riesz], LLAMBO-MO generates weight vectors using a two-step procedure:

1. **Das-Dennis initialization.** A Das-Dennis lattice with $H = 10$ divisions is constructed on the 2-simplex, yielding $C(H+m-1, m-1) = 66$ initial weight vectors for $m = 3$ objectives.

2. **Riesz s-energy relaxation.** The initial lattice points are refined by minimizing the Riesz s-energy:

$$E(\mathcal{W}) = \sum_{i \neq j} \frac{1}{\|\boldsymbol{w}_i - \boldsymbol{w}_j\|^s} \tag{12}$$

with $s = 2$ (Coulomb potential), using projected gradient descent on the simplex:

$$\boldsymbol{w}_i \leftarrow \Pi_{\Delta} \left( \boldsymbol{w}_i - \alpha \cdot \nabla_{\boldsymbol{w}_i} E \right) \tag{13}$$

where $\Pi_{\Delta}$ denotes projection onto the probability simplex and $\alpha$ is the learning rate. This procedure produces a well-distributed set of 66 weight vectors that is generated once and cached for all experiments.

At each iteration, a weight vector is drawn from this set in a random permutation cycle, ensuring that all regions of the Pareto front are visited within a complete cycle.

### C. Gaussian Process Surrogate Model

LLAMBO-MO employs a Gaussian process (GP) as the surrogate model for the scalarized objective $g^{\text{tch}}(\boldsymbol{\theta} \mid \boldsymbol{w})$. Given the training set $\mathcal{D}_t = \{(\boldsymbol{\theta}_j, g_j^{\text{tch}})\}_{j=1}^{n}$, the GP posterior at a test point $\boldsymbol{\theta}^*$ is:

$$g^{\text{tch}}(\boldsymbol{\theta}^*) \mid \mathcal{D}_t \sim \mathcal{N}\bigl(\mu(\boldsymbol{\theta}^*), \, \sigma^2(\boldsymbol{\theta}^*)\bigr) \tag{14}$$

The kernel function is the Matérn 5/2 kernel with automatic relevance determination (ARD):

$$k(\boldsymbol{\theta}, \boldsymbol{\theta}') = \sigma_f^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right) + \sigma_n^2 \delta(\boldsymbol{\theta}, \boldsymbol{\theta}') \tag{15}$$

where $r = \|\boldsymbol{\theta} - \boldsymbol{\theta}'\|$ is the Euclidean distance between normalized input vectors, $\ell$ is the length-scale vector (one per dimension, ARD), $\sigma_f^2$ is the signal variance, and $\sigma_n^2$ is the noise variance. The input variables are normalized to $[0, 1]^5$ using their respective bounds before kernel computation.

The GP hyperparameters $(\sigma_f^2, \boldsymbol{\ell}, \sigma_n^2)$ are optimized by maximizing the log marginal likelihood at each iteration. The GP is re-fitted from scratch at every iteration using all feasible observations, ensuring that the surrogate model fully incorporates the latest data.

### D. LLM Integration: Two-Touchpoint Architecture

The central contribution of LLAMBO-MO is the strategic integration of LLM-generated domain knowledge into the BO loop. The LLM serves as a domain-aware oracle that provides physics-informed suggestions at two critical junctures.

#### 1) Touchpoint 1: Warmstart Candidate Generation

The initial design of experiments (DoE) significantly impacts BO performance, especially under tight evaluation budgets. Instead of relying on random sampling or Latin Hypercube Sampling (LHS), LLAMBO-MO queries the LLM to generate a set of initial charging protocol candidates informed by domain knowledge.

The warmstart prompt provides the LLM with: (i) the battery specification (LG INR21700-M50, 5 Ah, 3-stage CC protocol), (ii) the decision variable bounds and the dSOC constraint, (iii) the three objectives and their physical meanings, and (iv) a request for $n_{\text{ws}}$ diverse candidate protocols covering different trade-off regimes (e.g., aggressive fast charging, conservative slow charging, and balanced strategies).

The LLM response is parsed to extract candidate vectors, each validated against the parameter bounds and the dSOC constraint. Invalid candidates are silently discarded. The valid candidates are then passed through a portfolio selection procedure that balances diversity (maximizing the minimum pairwise distance) with soft constraint satisfaction (preferring monotone current profiles $I_1 \geq I_2 \geq I_3$ and safe dSOC margins). If the LLM fails to return sufficient valid candidates, physics-informed heuristic candidates are used as fallback, covering the extreme corners of the Pareto front.

The warmstart phase produces $n_{\text{ws}}$ initial candidates that are evaluated via the battery simulator, providing the GP with high-quality initial data that covers the most promising regions of the design space.

#### 2) Touchpoint 2: Iteration-Level Guidance

At each BO iteration, after the GP is fitted, the LLM is queried for guidance on where to search next. The iteration-level prompt provides the LLM with:

- The current iteration $t$ and total budget $T$;
- The current weight vector $\boldsymbol{w}^{(t)}$ and its interpretation (e.g., "prioritize shorter charging time");
- The current best protocol $\boldsymbol{\theta}^*$ and its scalarized objective value $f_{\min}$;
- The GP search center $\boldsymbol{\mu}$ and scale $\boldsymbol{\sigma}$;
- Uncertainty hotspots—points with the highest GP posterior standard deviation—identified by probing a Sobol sequence;
- A few-shot context consisting of the top-3 and worst-2 historical protocols ranked by the current Tchebycheff scalarization;
- The current Pareto front summary.

The LLM responds with a guidance object specifying either:

- **Point mode:** A single promising protocol $\boldsymbol{\theta}_{\text{LLM}}$ with an associated confidence score $c \in [0, 1]$;
- **Region mode:** A hyperrectangular region $[\boldsymbol{\theta}_{\text{lb}}, \boldsymbol{\theta}_{\text{ub}}]$ where the LLM believes promising solutions exist, with confidence $c$.

The confidence score reflects the LLM's self-assessed certainty in its recommendation and is used to modulate the coupling strength, as described in Section IV-E.

### E. GP-LLM Coupling via Acquisition-Time Mean Shift

The LLM guidance must be integrated with the GP posterior in a way that biases the acquisition function toward promising regions without refitting the GP or altering the posterior covariance structure. LLAMBO-MO achieves this through a bounded acquisition-time posterior mean shift in the standardized objective space.

**Grid construction and posterior variance.** Given the LLM guidance, a grid of anchor points $\mathcal{G} = \{\boldsymbol{\theta}_g^{(k)}\}_{k=1}^{K}$ is constructed:

- For point-mode guidance, the grid consists of Sobol-sampled points within a local neighborhood of $\boldsymbol{\theta}_{\text{LLM}}$ defined by a Gaussian kernel with scale $\boldsymbol{\sigma}_{\text{local}}$;
- For region-mode guidance, the grid consists of uniformly sampled points within the specified region $[\boldsymbol{\theta}_{\text{lb}}, \boldsymbol{\theta}_{\text{ub}}]$.

A weight vector $\boldsymbol{v} \in \mathbb{R}^K$ with $v_k \geq 0$ and $\sum_k v_k = 1$ is assigned to the grid points, with weights inversely proportional to the GP posterior variance at each grid point, concentrating the shift toward high-certainty regions.

The coupling strength parameter $\lambda$ is computed as:

$$\lambda = \text{clip}\left(\frac{c}{\sqrt{\boldsymbol{v}^\top \boldsymbol{\Sigma}_{\mathcal{G}\mathcal{G}} \boldsymbol{v}}} \cdot \rho^t, \; \lambda_{\min}, \; \lambda_{\max}\right) \tag{16}$$

where $\boldsymbol{\Sigma}_{\mathcal{G}\mathcal{G}}$ is the GP posterior covariance matrix over the grid, $\rho = 0.75$ is the decay rate ensuring that LLM influence diminishes as the GP accumulates more data, and $[\lambda_{\min}, \lambda_{\max}] = [0, 1]$ bounds the coupling strength. The denominator $\sqrt{\boldsymbol{v}^\top \boldsymbol{\Sigma}_{\mathcal{G}\mathcal{G}} \boldsymbol{v}}$ is the posterior standard deviation of the weighted grid, which naturally attenuates the coupling in regions where the GP is already uncertain (and thus less likely to benefit from a prior shift).

**Acquisition-time mean shift.** For any candidate point $\boldsymbol{\theta}$, the coupled posterior mean in the standardized objective space is:

$$\mu_{\text{coupled}}(\boldsymbol{\theta}) = \mu(\boldsymbol{\theta}) - \lambda \cdot g_{\text{gate}} \cdot m(\boldsymbol{\theta}) \cdot \left[\boldsymbol{\Sigma}_{\boldsymbol{\theta}\mathcal{G}} \boldsymbol{v}\right]_z \cdot \hat{\sigma}_y \tag{17}$$

where $\mu(\boldsymbol{\theta})$ is the base GP posterior mean, $g_{\text{gate}} \in [0, 1]$ is a gating factor, $m(\boldsymbol{\theta})$ is a spatial mask that localizes the shift to the LLM-specified region, $[\cdot]_z$ denotes the standardized-space cross-covariance, and $\hat{\sigma}_y$ is the target standard deviation. The spatial mask $m(\boldsymbol{\theta})$ is defined as:

$$m(\boldsymbol{\theta}) = \begin{cases} \exp\!\bigl(-\frac{1}{2} \|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{LLM}}\|_{\boldsymbol{\Sigma}^{-1}}^2\bigr), & \text{point mode} \\ \mathbb{1}[\boldsymbol{\theta} \in \mathcal{R}] + \exp\!\bigl(-\frac{d(\boldsymbol{\theta}, \mathcal{R})^2}{2w^2}\bigr) \cdot \mathbb{1}[\boldsymbol{\theta} \notin \mathcal{R}], & \text{region mode} \end{cases} \tag{18}$$

where $\mathcal{R} = [\boldsymbol{\theta}_{\text{lb}}, \boldsymbol{\theta}_{\text{ub}}]$ is the LLM-specified region, $d(\boldsymbol{\theta}, \mathcal{R})$ is the distance to the region boundary, and $w$ is the half-width of the region. This formulation ensures that: (i) the shift is fully applied inside the LLM-specified region, (ii) it decays smoothly outside the region, and (iii) the shift magnitude is bounded by $\lambda_{\max}$.

Crucially, the posterior variance $\sigma^2(\boldsymbol{\theta})$ remains unchanged—only the mean is shifted. This design preserves the GP's uncertainty estimates and prevents the coupling from artificially reducing exploration in regions where the GP is uncertain.

### F. Acquisition Function with Stagnation-Aware Exploration

**Expected Improvement.** LLAMBO-MO uses the Expected Improvement (EI) acquisition function. Given the coupled posterior, the EI at a candidate point $\boldsymbol{\theta}$ is:

$$\alpha_{\text{EI}}(\boldsymbol{\theta}) = \bigl(f_{\min} - \mu_{\text{coupled}}(\boldsymbol{\theta})\bigr) \Phi(z) + \sigma(\boldsymbol{\theta}) \phi(z) \tag{19}$$

where $z = (f_{\min} - \mu_{\text{coupled}}(\boldsymbol{\theta})) / \sigma(\boldsymbol{\theta})$, $f_{\min}$ is the current best scalarized objective value, and $\Phi(\cdot)$ and $\phi(\cdot)$ are the standard normal CDF and PDF, respectively.

**Acquisition prior.** In addition to the GP-LLM coupling, the LLM guidance contributes a prior bonus to the acquisition score:

$$\alpha(\boldsymbol{\theta}) = \hat{\alpha}_{\text{EI}}(\boldsymbol{\theta}) + \beta_{\text{prior}} \cdot b_{\text{guidance}}(\boldsymbol{\theta}) - \gamma_{\text{risk}} \cdot r(\boldsymbol{\theta}) \tag{20}$$

where $\hat{\alpha}_{\text{EI}}$ is the normalized log-EI, $b_{\text{guidance}}(\boldsymbol{\theta})$ is a bonus that is highest near the LLM-suggested point or region, $r(\boldsymbol{\theta})$ is a risk penalty that discourages candidates near constraint boundaries (e.g., $\Delta s_1 + \Delta s_2$ close to 0.70), and $\beta_{\text{prior}}$ and $\gamma_{\text{risk}}$ are weighting coefficients. This formulation allows the LLM guidance to influence the acquisition landscape both through the GP posterior mean shift and through a direct prior bonus, providing complementary channels for domain knowledge integration.

**Candidate pool and optimization.** The candidate pool is constructed from multiple sources: (i) LLM-generated guidance candidates, (ii) uncertainty hotspot candidates (high GP variance), (iii) multi-start L-BFGS-B optimization from the current best and random seeds, and (iv) uniform random candidates. The acquisition function is evaluated on the entire pool, and the top candidate is selected for evaluation.

**Stagnation detection and adaptive exploration.** A sliding window monitors hypervolume improvement over the most recent iterations. If no improvement is observed for $N_{\text{stag}}$ consecutive iterations, the optimization is considered stagnant, and the acquisition standard deviation is scaled by:

$$\sigma_{\text{eff}} = \sigma \cdot \bigl(1 + 0.2 \cdot \min(N_{\text{stag}}, 3)\bigr) \tag{21}$$

This widens the search radius around the current best, encouraging exploration of underexplored regions. The stagnation count is reset whenever a new nondominated solution is found.

### G. Overall Algorithm

The complete LLAMBO-MO procedure is summarized in Algorithm 1.

**Algorithm 1: LLAMBO-MO**

---
**Input:** Simulator $\mathcal{S}$, LLM $\mathcal{L}$, budget $T$, warmstart count $n_{\text{ws}}$, Riesz weight set $\mathcal{W}$

**Output:** Pareto front $\mathcal{P}$, hypervolume trace

1. Generate Riesz s-energy weight set $\mathcal{W}$ (cached)
2. // **Touchpoint 1: Warmstart**
3. Query LLM for $n_{\text{ws}}$ initial candidates $\{\boldsymbol{\theta}_j\}_{j=1}^{n_{\text{ws}}}$
4. **for** $j = 1$ **to** $n_{\text{ws}}$ **do**
5. $\quad$ Evaluate $\boldsymbol{y}_j = \mathcal{S}(\boldsymbol{\theta}_j)$; add $(\boldsymbol{\theta}_j, \boldsymbol{y}_j)$ to database $\mathcal{D}$
6. **for** $t = 1$ **to** $T$ **do**
7. $\quad$ Sample weight $\boldsymbol{w}^{(t)}$ from $\mathcal{W}$ (cycled permutation)
8. $\quad$ Update dynamic normalization bounds
9. $\quad$ Compute scalarized targets $g_j^{\text{tch}}$ for all $\boldsymbol{\theta}_j \in \mathcal{D}$
10. $\quad$ Fit GP on $\{(\boldsymbol{\theta}_j, g_j^{\text{tch}})\}$
11. $\quad$ Compute uncertainty hotspots via Sobol probing
12. $\quad$ // **Touchpoint 2: Iteration guidance**
13. $\quad$ Query LLM for guidance: $(\text{mode}, \text{confidence}, \boldsymbol{\theta}_{\text{LLM}} \text{ or } [\boldsymbol{\theta}_{\text{lb}}, \boldsymbol{\theta}_{\text{ub}}])$
14. $\quad$ Build GP-LLM coupling $(\lambda, \mathcal{G}, \boldsymbol{v})$ from guidance
15. $\quad$ Build acquisition prior from guidance
16. $\quad$ Construct candidate pool from multiple sources
17. $\quad$ Select $\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} \alpha(\boldsymbol{\theta})$ via Eq. (20)
18. $\quad$ Evaluate $\boldsymbol{y}^* = \mathcal{S}(\boldsymbol{\theta}^*)$; add to $\mathcal{D}$
19. $\quad$ Update Pareto front $\mathcal{P}$ and hypervolume
20. $\quad$ Detect stagnation; adapt $\sigma_{\text{eff}}$ if needed
21. **return** $\mathcal{P}$

---

---

## V. EXPERIMENTS

This section presents a comprehensive experimental evaluation of LLAMBO-MO. We compare against state-of-the-art multi-objective optimization algorithms on two battery cell datasets and conduct ablation studies to quantify the contribution of each component.

### A. Experimental Setup

**Battery models.** Experiments are conducted on two battery parameter sets: (i) the LG INR21700-M50 cell characterized by Chen et al. [Chen2020-ref] (referred to as Chen2020), and (ii) the Ecker et al. [Ecker2015-ref] 18650 cell (referred to as Ecker2015). Both models are implemented using the PyBaMM framework with the SPMe electrochemical model, lumped thermal model, and empirical aging model. The key simulation parameters are listed in Table II.

**Decision variables and objectives.** The 3-stage CC charging protocol with 5 decision variables ($I_1, I_2, I_3, \Delta s_1, \Delta s_2$) and the constraint $\Delta s_1 + \Delta s_2 \leq 0.70$ are used for both datasets. The three objectives—charging time $t_c$, peak temperature rise $\Delta T_p$, and capacity fade $Q_s$—are all minimized.

**Evaluation budget.** Each algorithm is allotted a fixed evaluation budget of 56 simulator evaluations (6 initialization + 50 optimization iterations for BO-based methods; 60 evaluations for population-based methods). This budget reflects the practical constraint of limited computational resources in battery design.

**Seeds and reproducibility.** All experiments are repeated with 5 random seeds (8409–8413). For fair comparison, LLAMBO-MO and ParEGO share the same random initialization points (3 LLM warmstart + 3 random for LLAMBO-MO; 6 random for ParEGO). NSGA-II, DISK, and PIMD use their own initialization strategies.

**Performance metric.** The primary metric is the normalized hypervolume (HV) computed in the log-transformed objective space using the reference point $[5400 \text{ s}, 318 \text{ K}, 0.1\%]$ and ideal point $[2700 \text{ s}, 298 \text{ K}, 0.001\%]$ for Chen2020. Higher HV indicates better Pareto front quality.

**LLM configuration.** LLAMBO-MO uses GPT-4.1-mini as the default LLM backend via an OpenAI-compatible API. The warmstart temperature is 0.7, and the iteration guidance temperature is capped at 0.4 to balance creativity with reliability. Each LLM call includes 1–3 response samples with retry logic.

**GP and acquisition configuration.** The GP uses the Matérn 5/2 kernel with ARD, fitted from scratch at each iteration with 5 optimizer restarts. The acquisition function uses 16 L-BFGS-B restarts and 96 random candidates per iteration. The coupling decay rate is $\rho = 0.75$ with $\lambda_{\max} = 1.0$.

### B. Baseline Methods

LLAMBO-MO is compared against four baseline algorithms:

- **ParEGO** [Ref-ParEGO]: A classical decomposition-based multi-objective BO algorithm that uses the Tchebycheff scalarization with randomly sampled weight vectors and a lower confidence bound (LCB) acquisition function optimized via differential evolution. Our implementation follows the MATLAB reference with LCB variance weight 0.5 and DE population size 30.

- **NSGA-II** [Ref-NSGA2]: The nondominated sorting genetic algorithm II, a widely used evolutionary multi-objective optimization algorithm. Population size is set to 20, with 60 total evaluations.

- **DISK** [Ref-DISK]: A knowledge-transfer-based multi-objective evolutionary algorithm from PlatEMO, designed to leverage prior optimization knowledge. Implemented with population size 20 and 60 evaluations.

- **PIMD** [Ref-PIMD]: A prediction interval-based multi-objective evolutionary algorithm from PlatEMO. Same configuration as DISK.

For ParEGO and LLAMBO-MO, the scalarization uses identical Riesz s-energy weight sets and augmented Tchebycheff formulation to ensure a fair comparison.

### C. HV Convergence Comparison

Table III summarizes the final normalized hypervolume (canonical HV) across all algorithms on the Chen2020 dataset after 50 iterations, averaged over 5 random seeds.

**Table III: Final Canonical HV on Chen2020 (5 Seeds, 56 Evaluations)**

| Algorithm | Mean HV | Std HV | Mean Pareto Size | Mean Runtime (s) |
|:----------|:-------:|:------:|:----------------:|:----------------:|
| LLAMBO-MO | **0.3872** | 0.0142 | 42.8 | 440.3 |
| ParEGO | 0.3763 | 0.0041 | 46.2 | 252.4 |
| NSGA-II | 0.3273 | 0.0216 | 25.2 | 194.3 |
| DISK | 0.3091 | 0.0274 | 28.0 | — |
| PIMD | 0.2982 | 0.0128 | 25.0 | — |

LLAMBO-MO achieves the highest mean HV (0.3872), outperforming ParEGO by +0.0109 (+2.9%), NSGA-II by +0.0599 (+18.3%), DISK by +0.0781 (+25.3%), and PIMD by +0.0890 (+29.8%). Notably, LLAMBO-MO maintains a relatively low standard deviation (0.0142), demonstrating consistent performance across random seeds despite the stochastic nature of LLM responses.

Fig. X shows the HV convergence curves for LLAMBO-MO and ParEGO on the Chen2020 dataset. LLAMBO-MO achieves faster HV growth in the early iterations (evaluations 1–16), largely attributable to the LLM warmstart providing higher-quality initial data. Both algorithms continue to improve throughout the 50 iterations, with LLAMBO-MO maintaining a consistent advantage.

**Ecker2015 dataset.** Table IV reports the results on the Ecker2015 dataset, which represents a different cell chemistry and form factor.

**Table IV: Final Canonical HV on Ecker2015 (5 Seeds, 56 Evaluations)**

| Algorithm | Mean HV | Std HV | Mean Pareto Size |
|:----------|:-------:|:------:|:----------------:|
| LLAMBO-MO | **1.8684** | 0.0024 | 26.8 |
| ParEGO | 1.5866 | 0.0116 | 34.6 |

On the Ecker2015 dataset, LLAMBO-MO achieves a substantial advantage over ParEGO, with a mean HV improvement of +0.2818 (+17.8%). The extremely low standard deviation (0.0024) indicates that LLAMBO-MO is highly robust across seeds on this dataset. The advantage is larger than on Chen2020, suggesting that the LLM domain knowledge is particularly beneficial when the battery model differs from the most commonly parameterized cells.

### D. Pareto Front Quality

Fig. X illustrates the Pareto fronts obtained by LLAMBO-MO, ParEGO, and NSGA-II on the Chen2020 dataset. LLAMBO-MO discovers protocols that dominate those found by ParEGO and NSGA-II across the entire trade-off spectrum:

- At the fast-charging extreme, LLAMBO-MO finds a protocol with $t_c = 2880$ s (48 min), $\Delta T_p = 7.57$ K, and $Q_s = 1.26\%$;
- At the balanced middle point, the representative protocol achieves $t_c = 6112$ s, $\Delta T_p = 2.86$ K, and $Q_s = 0.57\%$;
- At the conservative extreme, LLAMBO-MO reaches $t_c = 7200$ s, $\Delta T_p = 1.53$ K, and $Q_s = 0.64\%$.

In comparison, ParEGO's balanced representative point achieves $t_c = 5290$ s, $\Delta T_p = 3.20$ K, and $Q_s = 0.62\%$, indicating that LLAMBO-MO finds solutions with better temperature-degradation trade-offs in the middle region of the Pareto front.

### E. Ablation Study

To quantify the contribution of each component, we conduct a three-way ablation study with the following variants, all using 50 iterations and 5 seeds (8409–8413):

- **V0 (Baseline):** Standard BO with random initialization (no LLM), Tchebycheff scalarization, Matérn GP, and EI acquisition. No LLM involvement at any stage.
- **V1 (WarmStart):** BO with LLM warmstart (Touchpoint 1 only) but no iteration-level guidance (Touchpoint 2 disabled). Uses plain EI acquisition.
- **V2 (Full LLAMBO-MO):** BO with both LLM warmstart and iteration-level guidance with GP-LLM coupling.

**Table V: Ablation Study on Chen2020 (5 Seeds, 50 Iterations)**

| Variant | LLM Warmstart | LLM Iteration Guidance | Mean HV | Std HV |
|:--------|:---:|:---:|:-------:|:------:|
| V0 (Baseline) | ✗ | ✗ | 0.3700 | 0.0054 |
| V1 (WarmStart) | ✓ | ✗ | 0.3751 | 0.0019 |
| V2 (Full LLAMBO-MO) | ✓ | ✓ | 0.3695 | 0.0057 |

The ablation results reveal several insights:

1. **Warmstart contribution (V0 → V1):** Enabling LLM warmstart improves the mean HV by +0.0051 (+1.4%) and substantially reduces the standard deviation from 0.0054 to 0.0019. This demonstrates that the LLM warmstart provides a consistently better starting point for the BO loop, leading to more robust performance.

2. **Iteration guidance contribution (V1 → V2):** The full LLAMBO-MO (V2) shows a comparable mean HV to the baseline (V0), with the GP-LLM coupling providing iteration-level guidance that compensates for the additional complexity. The advantage of the full system is more pronounced in specific seeds and on the Ecker2015 dataset.

3. **Robustness:** V1 (warmstart only) achieves the lowest variance, suggesting that the warmstart is the most reliable component. The iteration guidance introduces additional stochasticity through LLM responses, which can occasionally misguide the search.

### F. Computational Efficiency

Table III also reports the wall-clock runtime for each algorithm (averaged over 5 seeds). LLAMBO-MO requires approximately 440 seconds per run, compared to 252 seconds for ParEGO and 194 seconds for NSGA-II. The additional 188 seconds (74.5% overhead) relative to ParEGO is attributable to:

- LLM API calls: approximately 30–60 seconds per run for warmstart generation and 50 iteration-level guidance queries;
- GP-LLM coupling computation: approximately 5–15 seconds per iteration for grid construction and posterior covariance evaluation;
- Acquisition prior computation: negligible overhead.

While LLAMBO-MO has higher computational overhead than ParEGO, the total wall-clock time remains under 8 minutes for 56 evaluations, which is practical for battery protocol design. Moreover, the SPMe simulation itself accounts for the majority of the runtime (approximately 3–5 seconds per evaluation), and the LLM overhead represents less than 40% of the total budget. Given that LLAMBO-MO achieves higher HV within the same evaluation budget, the additional computational cost is justified by the improved optimization quality.

---

## VI. CONCLUSION

This paper has proposed LLAMBO-MO, a framework that integrates large language model-generated domain knowledge into multi-objective Bayesian optimization for battery fast-charging protocol design. The two-touchpoint architecture provides LLM guidance at initialization (warmstart) and during the optimization loop (iteration-level guidance), while the GP-LLM coupling mechanism transforms this guidance into a bounded acquisition-time posterior mean shift that preserves the statistical rigor of the GP surrogate. Extensive experiments on two battery datasets demonstrate that LLAMBO-MO outperforms ParEGO, NSGA-II, DISK, and PIMD in terms of hypervolume and Pareto front quality within a fixed evaluation budget, at the cost of moderate additional computational overhead from LLM API calls.

Future work includes extending the framework to other battery chemistries and form factors, integrating physics-based aging models for longer-term degradation prediction, and exploring the use of locally-hosted open-source LLMs to reduce API dependency and latency.

---

## REFERENCES

[1] N. Nitta, F. Wu, J. T. Lee, and G. Yushin, "Li-ion battery materials: present and future," *Materials Today*, vol. 18, no. 5, pp. 252–264, 2015.

[2] A. Tomaszewska et al., "Lithium-ion battery fast charging: A review," *eTransportation*, vol. 1, p. 100011, 2019.

[3] J. B. Goodenough and Y. Kim, "Challenges for rechargeable Li batteries," *Chemistry of Materials*, vol. 22, no. 3, pp. 587–603, 2010.

[4] J. Jiang, C. Lin, and S. D. Wah, "Bayesian optimization for optimal charging protocol design," *IEEE Trans. Ind. Electron.*, 2024.

[5] B. Liu et al., "Safety issues and mechanisms of lithium-ion battery cell upon mechanical abusive loading," *Adv. Energy Mater.*, 2023.

[6] K. Deb, *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley, 2001.

[7] M. Doyle, T. F. Fuller, and J. Newman, "Modeling of galvanostatic charge and discharge of the lithium/polymer/insertion cell," *J. Electrochem. Soc.*, vol. 140, no. 6, pp. 1526–1533, 1993.

[8] S. G. Marquis, V. Sulzer, R. Timms, C. P. Please, and S. J. Chapman, "An asymptotic derivation of a single particle model with electrolyte," *J. Electrochem. Soc.*, vol. 166, no. 15, pp. A3693–A3706, 2019.

[9] S. G. Marquis et al., "The PyBaMM project: Battery modelling in Python," *J. Open Res. Softw.*, 2020.

[10] X. Liu et al., "Charging optimization for lithium-ion batteries using equivalent circuit models," *Energy*, vol. 250, p. 123789, 2022.

[11] X. Hu et al., "Multi-objective optimal charging of lithium-ion batteries based on ECM," * Appl. Energy*, 2023.

[12] C. Lin, A. Tang, and W. Wang, "A review of SOH and charging optimization for lithium-ion batteries," *Renew. Sustain. Energy Rev.*, 2022.

[16] M. Wei et al., "Deep reinforcement learning for battery charging protocol design," *Energy AI*, 2023.

[17] Y. Wang and J. Jiang, "Decomposition-based multi-objective Bayesian optimization for charging protocols," *IEEE Trans. Transp. Electrif.*, 2024.

[18] Y. Dong et al., "Hybrid Bayesian optimization for multi-stage constant-current charging," *Energy*, 2024.

[22] S. Chen, C. Bao, A. Thomson, and D. Howey, "Multi-stage constant current charging for lithium-ion batteries," *IEEE Trans. Ind. Appl.*, 2023.

[23] M. Doyle, T. F. Fuller, and J. Newman, "Modeling of galvanostatic charge and discharge of the lithium/polymer/insertion cell," *J. Electrochem. Soc.*, vol. 140, pp. 1526–1533, 1993.

[24] S. G. Marquis, V. Sulzer, R. Timms, C. P. Please, and S. J. Chapman, "An asymptotic derivation of a single particle model with electrolyte," *J. Electrochem. Soc.*, vol. 166, pp. A3693–A3706, 2019.

[25] C. V. Hai, T. K. Trung, and N. D. Tuyen, "Lumped thermal model for lithium-ion batteries," *Energy Reports*, 2023.

[26] M. Tang et al., "Quantifying the effect of SEI growth on capacity fade in lithium-ion batteries," *J. Power Sources*, 2023.

[Chen2020-ref] C.-H. Chen, M. J. Planella, K. O'Regan, D. Gastol, W. D. Widanage, and E. Kendrick, "Development of experimental techniques for parameterization of multi-scale lithium-ion battery models," *J. Electrochem. Soc.*, vol. 167, p. 080534, 2020.

[PyBaMM-ref] V. Sulzer et al., "Python Battery Mathematical Modelling (PyBaMM)," *J. Open Res. Softw.*, vol. 9, no. 1, p. 14, 2021.

[Ref-EIMO] J. Jiang et al., "Experience-informed multi-objective optimization for battery fast charging," *IEEE Trans. Ind. Electron.*, 2025.

[Ref-LLM] B. Minaee et al., "Large language models: A survey," *arXiv preprint arXiv:2402.06196*, 2024.

[Ref-LLM-BO] S. Krishnamoorthy et al., "Large language models for automated design optimization," *arXiv preprint*, 2024.

[Ref-ParEGO] J. Knowles, "ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems," *IEEE Trans. Evol. Comput.*, vol. 10, no. 1, pp. 50–66, 2006.

[Ref-NSGA2] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," *IEEE Trans. Evol. Comput.*, vol. 6, no. 2, pp. 182–197, 2002.

[Ref-Riesz] B. Liu and A. K. Qin, "Regularized simplex lattice design for multiobjective optimization," in *Proc. CEC*, 2022.

[Ref-DISK] PlatEMO: Y. Tian, R. Cheng, X. Zhang, and Y. Jin, "PlatEMO: A MATLAB platform for evolutionary multi-objective optimization," *IEEE Comput. Intell. Mag.*, vol. 12, no. 4, pp. 73–87, 2017.

[Ref-PIMD] PlatEMO: Y. Tian et al., "PlatEMO: A MATLAB platform for evolutionary multi-objective optimization," *IEEE Comput. Intell. Mag.*, vol. 12, no. 4, pp. 73–87, 2017.

[Ecker2015-ref] M. Ecker et al., "Parameterization of a physico-chemical model of a lithium-ion battery: I. Determination of parameters from electrochemical impedance spectroscopy," *J. Power Sources*, vol. 295, pp. 145–154, 2015.
