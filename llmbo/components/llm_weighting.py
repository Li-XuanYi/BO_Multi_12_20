"""
LLM 增强权重模块（LLAMBO 风格 + PE-GenBO）
实现论文公式 (9)：W_LLM = ∏ [1/√(2πσ²) exp(-(θ-μ)²/(2σ²))]

统一 3D 参数空间：I1, SOC1, I2
"""

import numpy as np
import asyncio
import json
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI

# 导入 config
from config import LLM_CONFIG, MOBO_CONFIG, get_llm_param, PARAM_BOUNDS


class LLAMBOWeighting:
    """
    LLAMBO 风格的 LLM 权重函数 + 敏感度排序（3D 参数空间）

    核心功能：
    1. W_LLM 权重计算：W_LLM(θ) = ∏_{j=1}^d [N(θ_j | μ_j, σ_j²)]
    2. 敏感度排序推理：rank_t(j) ∈ {1,2,3}（LLM 分析物理影响）
    3. ARD 长度尺度驱动：ℓ_{t,j} = ℓ_base * (rank_t(j) / d)^α_ℓ
    """

    def __init__(
        self,
        param_bounds: Dict[str, tuple] = None,
        llm_api_key: str = None,
        base_url: str = None,
        model: str = None,
        sigma_scale: float = None,
        length_scale_base: float = None,
        length_scale_alpha: float = None,
        verbose: bool = False
    ):
        """
        初始化 LLM 权重模块（3D 参数空间）

        参数：
            param_bounds: 参数边界 {I1, SOC1, I2}
            llm_api_key: LLM API 密钥
            base_url: API 基础 URL
            model: LLM 模型
            sigma_scale: σ 缩放因子（控制焦点宽度）
            length_scale_base: ARD 长度尺度基础值 ℓ_base
            length_scale_alpha: 敏感度指数 α_ℓ
            verbose: 详细输出
        """
        # 使用默认 3D 参数边界
        self.param_bounds = param_bounds or {
            'I1': (3.0, 7.0),
            'SOC1': (0.1, 0.7),
            'I2': (1.0, 5.0)
        }

        # 从 config 读取默认值
        llm_api_key = llm_api_key or LLM_CONFIG.get('api_key')
        base_url = base_url or LLM_CONFIG.get('base_url')
        model = model or LLM_CONFIG.get('model')
        self.sigma_scale = sigma_scale if sigma_scale is not None else get_llm_param('weighting', 'sigma_scale', 0.15)
        self.length_scale_base = length_scale_base if length_scale_base is not None else get_llm_param('weighting', 'length_scale_base', 0.5)
        self.length_scale_alpha = length_scale_alpha if length_scale_alpha is not None else get_llm_param('weighting', 'length_scale_alpha', 0.8)
        self.verbose = verbose

        # 3D 参数空间
        self.param_names = ['I1', 'SOC1', 'I2']
        self.d = len(self.param_names)

        # LLM 客户端
        if llm_api_key is not None:
            self.client = AsyncOpenAI(base_url=base_url, api_key=llm_api_key)
            self.model = model
        else:
            self.client = None

        # 焦点参数（初始化为参数空间中心）
        self.mu_focus = np.array([
            (self.param_bounds[k][0] + self.param_bounds[k][1]) / 2
            for k in self.param_names
        ])

        # 焦点宽度（初始化为参数范围的 sigma_scale）
        param_ranges = np.array([
            self.param_bounds[k][1] - self.param_bounds[k][0]
            for k in self.param_names
        ])
        self.sigma_focus = self.sigma_scale * param_ranges

        # 敏感度排序（初始化为均等：[1..d]）
        self.sensitivity_ranking = np.arange(1, self.d + 1, dtype=int)

        # ARD 长度尺度（初始化并计算）
        self.ard_length_scales = np.full(self.d, self.length_scale_base)
        self._compute_length_scales()

        # 记录更新历史
        self.update_history = []
        self.sensitivity_history = []

    # ========================================
    # 敏感度排序推理（3D 版本）
    # ========================================
    async def infer_sensitivity_ranking(
        self,
        database: List[Dict],
        iteration: int
    ) -> np.ndarray:
        """
        从 LLM 推理参数敏感度排序（3D 版本）

        流程：
        1. 构建包含物理知识的 Prompt（Top-10 观测、失败样本、物理机制）
        2. LLM 推理各参数对目标的影响强度
        3. 输出排序 rank_t(j) ∈ {1,2,3}（1=最敏感，3=最不敏感）
        4. 验证排序有效性（必须是{1,2,3}的排列）
        5. 更新 self.sensitivity_ranking 并触发长度尺度计算

        参数：
            database: 评估历史
            iteration: 当前迭代轮次

        返回：
            ranking: (3,) 数组，ranking[j] = rank_t(j)
        """
        if self.client is None:
            if self.verbose:
                print("  [Sensitivity] LLM 未配置，使用默认排序")
            return self.sensitivity_ranking

        # 构建 Prompt
        prompt = self._build_sensitivity_prompt(database)

        if self.verbose:
            print(f"  [Sensitivity] 推理第{iteration}轮敏感度排序...")

        max_retries = get_llm_param('weighting', 'sensitivity_max_retries', 3)

        for attempt in range(max_retries):
            try:
                # 调用 LLM（JSON 模式）
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in battery fast-charging physics. "
                                "Analyze parameter sensitivity based on experimental data and domain knowledge. "
                                "Output strictly in JSON format."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=get_llm_param('weighting', 'sensitivity_temperature', 0.2),
                    max_tokens=get_llm_param('weighting', 'sensitivity_max_tokens', 400),
                    response_format={"type": "json_object"}
                )

                # 解析响应
                result = json.loads(response.choices[0].message.content)
                ranking = self._parse_sensitivity_response(result)

                # 验证成功
                if ranking is not None:
                    self.sensitivity_ranking = ranking

                    # 计算长度尺度
                    self._compute_length_scales()

                    # 记录历史
                    self.sensitivity_history.append({
                        'iteration': iteration,
                        'ranking': ranking.copy(),
                        'length_scales': self.ard_length_scales.copy()
                    })

                    if self.verbose:
                        print(f"    排序：{ranking} → 长度尺度：{self.ard_length_scales}")

                    return ranking

                else:
                    if self.verbose:
                        print(f"    [尝试{attempt+1}/{max_retries}] 解析失败，重试...")

            except Exception as e:
                if self.verbose:
                    print(f"    [尝试{attempt+1}/{max_retries}] LLM 调用失败：{e}")

        # 重试耗尽，返回默认排序
        if self.verbose:
            print(f"  [Sensitivity] 推理失败，使用默认排序")
        return self.sensitivity_ranking

    def _build_sensitivity_prompt(self, database: List[Dict]) -> str:
        """
        构建敏感度排序 Prompt（包含物理知识）- 3D 版本
        """
        # 1. Top-10 观测（按时间排序）
        valid_data = [r for r in database if r.get('valid', False)]

        obs_str = ""
        if len(valid_data) > 0:
            sorted_data_time = sorted(valid_data, key=lambda x: x.get('time', 1e9))
            top_10_time = sorted_data_time[:10]

            obs_str += "**Top-10 by Time (fastest charging):**\n"
            for i, r in enumerate(top_10_time, 1):
                p = r.get('params', {})
                # 支持 3D 参数
                i1 = p.get('I1', p.get('current1', 0))
                soc1 = p.get('SOC1', p.get('time1', p.get('switch_soc', 0)))
                i2 = p.get('I2', p.get('current2', 0))

                obs_str += (
                    f"  Exp {i}: I1={i1:.2f}A, SOC1={soc1:.3f}, I2={i2:.2f}A "
                    f"→ Time={r.get('time', 0):.1f}s, Temp={r.get('temp', 0):.1f}K, Aging={r.get('aging', 0):.5f}\n"
                )

            # 添加 Top-5 按温度
            sorted_data_temp = sorted(valid_data, key=lambda x: x.get('temp', 1e9))
            top_5_temp = sorted_data_temp[:5]

            obs_str += "\n**Top-5 by Temperature (coolest):**\n"
            for i, r in enumerate(top_5_temp, 1):
                p = r.get('params', {})
                i1 = p.get('I1', p.get('current1', 0))
                soc1 = p.get('SOC1', p.get('time1', p.get('switch_soc', 0)))
                i2 = p.get('I2', p.get('current2', 0))

                obs_str += (
                    f"  Exp {i}: I1={i1:.2f}A, SOC1={soc1:.3f}, I2={i2:.2f}A "
                    f"→ Time={r.get('time', 0):.1f}s, Temp={r.get('temp', 0):.1f}K, Aging={r.get('aging', 0):.5f}\n"
                )

            # 添加 Top-5 按老化
            sorted_data_aging = sorted(valid_data, key=lambda x: x.get('aging', 1e9))
            top_5_aging = sorted_data_aging[:5]

            obs_str += "\n**Top-5 by Aging (least degradation):**\n"
            for i, r in enumerate(top_5_aging, 1):
                p = r.get('params', {})
                i1 = p.get('I1', p.get('current1', 0))
                soc1 = p.get('SOC1', p.get('time1', p.get('switch_soc', 0)))
                i2 = p.get('I2', p.get('current2', 0))

                obs_str += (
                    f"  Exp {i}: I1={i1:.2f}A, SOC1={soc1:.3f}, I2={i2:.2f}A "
                    f"→ Time={r.get('time', 0):.1f}s, Temp={r.get('temp', 0):.1f}K, Aging={r.get('aging', 0):.5f}\n"
                )

            # 数据统计摘要
            params_list = []
            for r in valid_data:
                p = r.get('params', {})
                params_list.append([
                    p.get('I1', p.get('current1', 0)),
                    p.get('SOC1', p.get('time1', p.get('switch_soc', 0))),
                    p.get('I2', p.get('current2', 0))
                ])
            params_array = np.array(params_list)

            obs_str += "\n**Data Summary:**\n"
            obs_str += f"  I1 range: [{params_array[:, 0].min():.2f}, {params_array[:, 0].max():.2f}]A, mean={params_array[:, 0].mean():.2f}A\n"
            obs_str += f"  SOC1 range: [{params_array[:, 1].min():.3f}, {params_array[:, 1].max():.3f}], mean={params_array[:, 1].mean():.3f}\n"
            obs_str += f"  I2 range: [{params_array[:, 2].min():.2f}, {params_array[:, 2].max():.2f}]A, mean={params_array[:, 2].mean():.2f}A\n"
        else:
            obs_str = "  (No valid data yet)\n"

        # 2. 失败样本
        invalid_data = [r for r in database if not r.get('valid', False)]

        if len(invalid_data) > 0:
            fail_str = ""
            for i, r in enumerate(invalid_data[:5], 1):
                p = r.get('params', {})
                i1 = p.get('I1', p.get('current1', 0))
                soc1 = p.get('SOC1', p.get('time1', p.get('switch_soc', 0)))
                i2 = p.get('I2', p.get('current2', 0))
                fail_str += (
                    f"  Failed {i}: I1={i1:.2f}A, SOC1={soc1:.3f}, I2={i2:.2f}A\n"
                )
        else:
            fail_str = "  (No failures yet)\n"

        # 3. 物理机制（仅描述机制，不给出敏感度标签）
        physics_str = """
**Physical Mechanisms (3D Parameter Space: I1, SOC1, I2):**
1. **I1 (current1)**: Higher current accelerates charging but increases heat generation (Joule heating ∝ I²R).
   - Direct impact on power dissipation and thermal stress

2. **SOC1 (switch_soc)**: State-of-charge at phase transition point.
   - Controls duration of Phase 1 (higher SOC1 = longer high-current exposure)
   - Accumulated heat Q = ∫ I1²·R(SOC)·dt

3. **I2 (current2)**: Phase 2 replenishment current.
   - Determines charging rate during final stage
   - Lower values reduce thermal and electrical stress
   - Constraint: I2 <= I1
"""

        # 完整 Prompt
        prompt = f"""
You are analyzing a battery fast-charging optimization problem with 3 decision variables:
- **I1** (current1): Phase 1 current [3.0-7.0]A
- **SOC1** (switch_soc): Phase transition point [0.1-0.7]
- **I2** (current2): Phase 2 current [1.0-5.0]A

Objectives to MINIMIZE:
- Charging time
- Peak temperature
- Capacity fade (aging)

**Observed Experiments:**
{obs_str}

**Failed Experiments (constraints violated):**
{fail_str}

{physics_str}

**Task:**
Based on the experimental data above, rank the 3 parameters by their sensitivity to the objectives (time, temperature, aging).

Your ranking MUST be informed by the experimental data above. Analyze which parameters show the strongest correlation with objective variations in the data.

Output JSON:
{{
  "ranking": {{
    "I1": <rank 1-3, where 1=most sensitive>,
    "SOC1": <rank 1-3>,
    "I2": <rank 1-3>
  }},
  "reasoning": "<brief explanation based on observed data patterns>"
}}

**CRITICAL**: The 3 ranks must be a permutation of [1,2,3] (each used exactly once).
"""
        return prompt

    def _parse_sensitivity_response(self, result: Dict) -> Optional[np.ndarray]:
        """
        解析 LLM 响应，提取敏感度排序（3D 版本）
        """
        try:
            ranking_dict = result.get('ranking', {})

            # 提取排序
            ranks = []
            for param_name in self.param_names:
                rank = ranking_dict.get(param_name)

                if rank is None:
                    if self.verbose:
                        print(f"    [解析错误] 缺少参数：{param_name}")
                    return None

                rank = int(rank)

                if rank < 1 or rank > 3:
                    if self.verbose:
                        print(f"    [解析错误] {param_name}排序越界：{rank}")
                    return None

                ranks.append(rank)

            ranking = np.array(ranks, dtype=int)

            # 验证是否为排列
            if not np.array_equal(sorted(ranking), [1, 2, 3]):
                if self.verbose:
                    print(f"    [解析错误] 不是有效排列：{ranking}")
                return None

            return ranking

        except Exception as e:
            if self.verbose:
                print(f"    [解析错误] {e}")
            return None

    def _compute_length_scales(self):
        """
        计算 ARD 长度尺度（3D 版本）

        公式：ℓ_{t,j} = ℓ_base * (rank_t(j) / d)^α_ℓ
        """
        self.ard_length_scales = self.length_scale_base * \
            (self.sensitivity_ranking / self.d) ** self.length_scale_alpha

    def get_length_scales(self) -> np.ndarray:
        """获取当前 ARD 长度尺度（供 GP 模型使用）"""
        return self.ard_length_scales.copy()

    def get_sensitivity_ranking(self) -> np.ndarray:
        """获取当前敏感度排序"""
        return self.sensitivity_ranking.copy()

    # ========================================
    # Touchpoint 2: LLM 候选生成
    # ========================================
    async def generate_candidates(
        self,
        database: list,
        weights: np.ndarray,
        grad_psi: np.ndarray,
        iteration: int,
        total_iterations: int,
        n_candidates: int = 15,
    ) -> list:
        """
        Touchpoint 2：向 LLM 请求 n_candidates 个候选充电协议（3D 版本）

        3D 参数空间：I1, SOC1, I2

        返回 list of dict，每个 dict 包含 'I1', 'SOC1', 'I2' 键。
        失败时回退随机采样。
        """
        if self.client is None:
            return self._fallback_random(n_candidates)

        prompt = self._build_candidates_prompt(database, weights, grad_psi,
                                               iteration, total_iterations, n_candidates)
        max_retries = get_llm_param('acquisition', 'gen_max_retries', 3)
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in lithium-ion battery fast-charging optimization. "
                                "Suggest charging protocols as JSON arrays. "
                                "Each protocol has keys 'I1' (float), 'SOC1' (float), 'I2' (float). "
                                "Output ONLY valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=get_llm_param('acquisition', 'gen_max_tokens', 2000),
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                candidates = self._parse_candidates_response(raw, n_candidates)
                if candidates:
                    if self.verbose:
                        print(f"  [Touchpoint 2] 获取 {len(candidates)} 个 LLM 候选")
                    return candidates
            except Exception as e:
                if self.verbose:
                    print(f"  [Touchpoint 2] 尝试 {attempt+1}/{max_retries} 失败：{e}")
        return self._fallback_random(n_candidates)

    def _build_candidates_prompt(
        self, database, weights, grad_psi, iteration, total_iterations, n_candidates
    ) -> str:
        """构建 Touchpoint 2 的 Prompt（包含完整上下文）- 3D 版本"""
        I1_lo, I1_hi = PARAM_BOUNDS['I1']
        S1_lo, S1_hi = PARAM_BOUNDS['SOC1']
        I2_lo, I2_hi = PARAM_BOUNDS['I2']

        # 有效样本摘要（最多 8 个最优）
        valid = [r for r in database if r.get('valid', False)]
        obs_str = "No valid data yet.\n"
        if valid:
            top = sorted(valid, key=lambda r: r.get('time', 1e9))[:8]
            lines = []
            for r in top:
                p = r.get('params', r)
                i1 = p.get('I1', p.get('current1', 3.0))
                s1 = p.get('SOC1', p.get('time1', 0.4))
                i2 = p.get('I2', p.get('current2', 2.0))
                lines.append(
                    f"  I1={i1:.2f}A SOC1={s1:.3f} I2={i2:.2f}A"
                    f" → time={r.get('time', 0):.0f}s"
                    f" temp={r.get('temp', 0):.1f}K"
                    f" aging={r.get('aging', 0):.2e}"
                )
            obs_str = "\n".join(lines) + "\n"

        w_str = f"time={weights[0]:.3f}, temp={weights[1]:.3f}, aging={weights[2]:.3f}"
        g_str = f"[{grad_psi[0]:.3f}, {grad_psi[1]:.3f}, {grad_psi[2]:.3f}]" if grad_psi is not None else "N/A"

        prompt = f"""
Battery fast-charging optimization — Iteration {iteration}/{total_iterations}

**Parameter space (3D, 2-stage CC protocol):**
- I1  (Phase-1 current): [{I1_lo:.2f}, {I1_hi:.2f}] A
- SOC1 (switch SOC):     [{S1_lo:.2f}, {S1_hi:.2f}]
- I2  (Phase-2 current): [{I2_lo:.2f}, {I2_hi:.2f}] A

**Objectives to minimise:** charging time (s), peak temperature (K), capacity fade (aging).

**Current scalarisation weights:** {w_str}

**Physics proxy gradient ∇Ψ at best point:** {g_str}

**Best observed protocols (sorted by time):**
{obs_str}
**Physics hints:**
- Ψ(θ) = R̄₁·I₁²·t₁ + R̄₂·I₂²·t₂  (R̄ ≈ 0.01 Ω)
- High I1 → faster but hotter; high SOC1 → longer Phase-1
- I2 ≪ I1 reduces thermal stress near full charge

**Task:** Suggest {n_candidates} diverse charging protocols that are likely to improve on the current best, considering the given weights.

Output a JSON object with a single key "candidates" containing an array of {n_candidates} objects:
{{
  "candidates": [
    {{"I1": <float>, "SOC1": <float>, "I2": <float>}},
    ...
  ]
}}
All values must be within the stated bounds.
"""
        return prompt

    def _parse_candidates_response(self, raw: str, n_candidates: int) -> list:
        """解析 LLM JSON 响应，提取候选点列表。"""
        try:
            data = json.loads(raw)
            cands_raw = data.get("candidates", data) if isinstance(data, dict) else data
            if not isinstance(cands_raw, list):
                return []
            result = []
            for c in cands_raw:
                if not isinstance(c, dict):
                    continue
                try:
                    I1 = float(c.get("I1", c.get("i1", 3.0)))
                    S1 = float(c.get("SOC1", c.get("soc1", 0.4)))
                    I2 = float(c.get("I2", c.get("i2", 2.0)))
                    I1 = float(np.clip(I1, *PARAM_BOUNDS['I1']))
                    S1 = float(np.clip(S1, *PARAM_BOUNDS['SOC1']))
                    I2 = float(np.clip(I2, *PARAM_BOUNDS['I2']))
                    result.append({"I1": I1, "SOC1": S1, "I2": I2, "source": "llm_sample"})
                except (ValueError, TypeError):
                    continue
            return result[:n_candidates]
        except json.JSONDecodeError:
            return []

    def _fallback_random(self, n: int) -> list:
        """LLM 失败时的随机回退。"""
        return [
            {
                "I1": float(np.random.uniform(*PARAM_BOUNDS['I1'])),
                "SOC1": float(np.random.uniform(*PARAM_BOUNDS['SOC1'])),
                "I2": float(np.random.uniform(*PARAM_BOUNDS['I2'])),
                "source": "random_fallback",
            }
            for _ in range(n)
        ]

    def compute_weight(self, x: np.ndarray) -> float:
        """
        计算 LLAMBO 公式 (9) 的权重（3D 版本）

        W_LLM(θ) = ∏_{j=1}^3 [N(θ_j | μ_j, σ_j²)]

        其中 N(θ_j | μ_j, σ_j²) = 1/√(2πσ_j²) exp(-(θ_j - μ_j)²/(2σ_j²))

        参数：
            x: 候选点 (3,) [I1, SOC1, I2]

        返回：
            weight: [0, 1]，归一化权重
        """
        # 确保 x 是 1D 数组
        if x.ndim > 1:
            x = x.flatten()

        # 计算多元高斯密度（连乘）
        weight = 1.0

        for j in range(self.d):
            # 高斯密度：N(θ_j | μ_j, σ_j²)
            gaussian_term = (1.0 / np.sqrt(2 * np.pi * self.sigma_focus[j]**2)) * \
                            np.exp(-(x[j] - self.mu_focus[j])**2 / (2 * self.sigma_focus[j]**2))

            weight *= gaussian_term

        # 归一化到 [0, 1]
        max_weight = 1.0
        for j in range(self.d):
            max_weight *= (1.0 / np.sqrt(2 * np.pi * self.sigma_focus[j]**2))

        normalized_weight = weight / max_weight if max_weight > 1e-10 else 1.0

        # 限制在 [0, 1]
        normalized_weight = np.clip(normalized_weight, 0.0, 1.0)

        # 保留至少 10% 的探索能力
        normalized_weight = max(normalized_weight, 0.1)

        return normalized_weight

    def _extract_pareto(self, database: List[Dict]) -> List[Dict]:
        """提取 Pareto 前沿"""
        valid_data = [r for r in database if r.get('valid', False)]

        if len(valid_data) == 0:
            return []

        # 提取目标值
        objectives = np.array([[
            r.get('time', 0),
            r.get('temp', 0),
            r.get('aging', 0)
        ] for r in valid_data])

        # Pareto 支配
        is_pareto = np.ones(len(valid_data), dtype=bool)

        for i in range(len(valid_data)):
            for j in range(len(valid_data)):
                if i == j:
                    continue

                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_pareto[i] = False
                    break

        pareto_front = [valid_data[i] for i in range(len(valid_data)) if is_pareto[i]]

        return pareto_front

    async def update_focus_from_llm(
        self,
        database: List[Dict],
        gp_list: List = None,
        iteration: int = 0
    ):
        """
        从 LLM 推理更新焦点中心μ（3D 版本）
        """
        if self.client is None:
            if self.verbose:
                print("  [LLM Weighting] LLM 未配置，跳过焦点更新")
            return

        # 提取 Pareto 前沿
        pareto_front = self._extract_pareto(database)

        if len(pareto_front) < 3:
            if self.verbose:
                print(f"  [LLM Weighting] Pareto 点不足 ({len(pareto_front)})，跳过更新")
            return

        # 构建 Prompt
        prompt = self._build_focus_prompt(pareto_front)

        if self.verbose:
            print(f"  [LLM Weighting] 调用 LLM 推理焦点...")

        try:
            # 调用 LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in battery optimization. "
                            "Analyze Pareto-optimal points and suggest the most promising region for next exploration. "
                            "Output strictly in JSON format."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=get_llm_param('weighting', 'temperature', 0.3),
                max_tokens=get_llm_param('weighting', 'max_tokens', 500),
                response_format={"type": "json_object"}
            )

            # 解析响应
            result = json.loads(response.choices[0].message.content)

            # 提取焦点中心（3D）
            new_mu = np.array([
                float(result.get('focus_I1', self.mu_focus[0])),
                float(result.get('focus_SOC1', self.mu_focus[1])),
                float(result.get('focus_I2', self.mu_focus[2]))
            ])

            # 边界检查
            new_mu[0] = np.clip(new_mu[0], *self.param_bounds['I1'])
            new_mu[1] = np.clip(new_mu[1], *self.param_bounds['SOC1'])
            new_mu[2] = np.clip(new_mu[2], *self.param_bounds['I2'])

            # 更新焦点
            self.mu_focus = new_mu

            if self.verbose:
                print(f"    焦点已更新：μ=[{self.mu_focus[0]:.2f}, {self.mu_focus[1]:.3f}, {self.mu_focus[2]:.2f}]")

            # 从 GP 更新σ（如果提供）
            if gp_list is not None:
                self._update_sigma_from_gp(gp_list)

            # 记录历史
            self.update_history.append({
                'iteration': iteration,
                'mu': self.mu_focus.copy(),
                'sigma': self.sigma_focus.copy()
            })

        except Exception as e:
            if self.verbose:
                print(f"  [LLM Weighting] 焦点更新失败：{e}")

    def _build_focus_prompt(self, pareto_front: List[Dict]) -> str:
        """构建 LLM Prompt（3D 版本）"""
        # 序列化 Pareto 点
        points_str = ""
        for i, point in enumerate(pareto_front[:10], 1):
            p = point.get('params', {})
            i1 = p.get('I1', p.get('current1', 0))
            soc1 = p.get('SOC1', p.get('time1', 0))
            i2 = p.get('I2', p.get('current2', 0))

            points_str += (
                f"Point {i}: I1={i1:.2f}A, SOC1={soc1:.3f}, I2={i2:.2f}A "
                f"→ Time={point.get('time', 0):.1f}s, Temp={point.get('temp', 0):.1f}K, Aging={point.get('aging', 0):.5f}\n"
            )

        prompt = f"""
Given {len(pareto_front)} Pareto-optimal battery charging strategies:

{points_str}

**Physics Context:**
- High I1 → Fast charging but more heat
- High SOC1 → Extended Phase 1 duration, heat accumulation
- Low I2 → Slower Phase 2, less stress

**Trade-offs:**
- Time vs Temperature: Higher I1/higher SOC1 reduces time but increases temperature
- Temperature vs Aging: Higher temperature accelerates aging
- I1 and SOC1 are coupled: High I1 + High SOC1 = thermal runaway risk

**Task:**
Based on the Pareto points above, suggest the most promising parameter region for the NEXT exploration.

Output JSON:
{{
  "focus_I1": <value in [3.0, 7.0]>,
  "focus_SOC1": <value in [0.1, 0.7]>,
  "focus_I2": <value in [1.0, 5.0]>,
  "reasoning": "<brief explanation>"
}}
"""
        return prompt

    def _update_sigma_from_gp(self, gp_list: List):
        """从 GP 预测不确定性更新σ（3D 版本）"""
        try:
            # 在μ处预测
            X_mu = self.mu_focus.reshape(1, -1)

            # 收集每个 GP 在μ处的标准差
            std_list = []
            for gp in gp_list:
                _, std = gp.predict(X_mu, return_std=True)
                std_list.append(float(std[0]))

            # 参数范围
            param_ranges = np.array([
                self.param_bounds['I1'][1] - self.param_bounds['I1'][0],
                self.param_bounds['SOC1'][1] - self.param_bounds['SOC1'][0],
                self.param_bounds['I2'][1] - self.param_bounds['I2'][0]
            ])

            # 归一化 GP 不确定性到 [0,1]
            ref_ranges = np.array([
                MOBO_CONFIG.get('reference_point', {}).get('time', 3000) - MOBO_CONFIG.get('ideal_point', {}).get('time', 600),
                MOBO_CONFIG.get('reference_point', {}).get('temp', 320) - MOBO_CONFIG.get('ideal_point', {}).get('temp', 300),
                MOBO_CONFIG.get('reference_point', {}).get('aging', 0.01) - MOBO_CONFIG.get('ideal_point', {}).get('aging', 0)
            ])
            ref_ranges = np.maximum(ref_ranges, 1e-10)

            normalized_stds = np.array(std_list) / ref_ranges
            avg_normalized_std = float(np.mean(normalized_stds))
            avg_normalized_std = np.clip(avg_normalized_std, 0.0, 1.0)

            # σ = σ_base × (1 + α_uncertainty × normalized_std)
            sigma_base = self.sigma_scale * param_ranges
            uncertainty_amplification = 1.0 + 2.0 * avg_normalized_std

            self.sigma_focus = sigma_base * uncertainty_amplification

            if self.verbose:
                print(f"    σ已更新：{self.sigma_focus}")
                print(f"    (归一化 GP 不确定性：{avg_normalized_std:.4f}, 放大系数：{uncertainty_amplification:.2f})")

        except Exception as e:
            if self.verbose:
                print(f"    [警告] σ更新失败：{e}")


# ============================================================
# 快速测试（3D 版本）
# ============================================================
if __name__ == "__main__":
    import asyncio

    async def test():
        print("测试 LLAMBOWeighting（3D 参数空间）...")

        # 参数边界（3D）
        param_bounds = {
            'I1': (3.0, 7.0),
            'SOC1': (0.1, 0.7),
            'I2': (1.0, 5.0)
        }

        # 初始化（无 LLM）
        weighting = LLAMBOWeighting(
            param_bounds=param_bounds,
            llm_api_key=None,
            verbose=True
        )

        print(f"\n初始焦点：μ={weighting.mu_focus}")
        print(f"初始宽度：σ={weighting.sigma_focus}")
        print(f"初始排序：{weighting.sensitivity_ranking}")
        print(f"初始长度尺度：{weighting.ard_length_scales}")

        # 测试权重计算（3D）
        test_points = [
            np.array([5.0, 0.4, 3.0]),  # 中心点
            np.array([7.0, 0.7, 5.0]),  # 边界点
            np.array([3.0, 0.1, 1.0])   # 另一边界
        ]

        print("\n测试权重计算:")
        for i, x in enumerate(test_points, 1):
            weight = weighting.compute_weight(x)
            print(f"  点{i} {x} → 权重={weight:.4f}")

        # 测试长度尺度计算
        print("\n测试长度尺度计算:")
        test_rankings = [
            [1, 2, 3],  # I1 最敏感
            [2, 1, 3],  # SOC1 最敏感
            [3, 2, 1]   # I2 最敏感
        ]

        for ranking in test_rankings:
            weighting.sensitivity_ranking = np.array(ranking)
            weighting._compute_length_scales()
            print(f"  排序{ranking} → 长度尺度={weighting.ard_length_scales}")

        print("\n测试完成！")

    asyncio.run(test())
