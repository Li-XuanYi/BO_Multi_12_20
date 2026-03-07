"""
LLM增强权重模块（LLAMBO风格 + PE-GenBO）
实现论文公式(9)：W_LLM = ∏ [1/√(2πσ²) exp(-(θ-μ)²/(2σ²))]

Round 2新增功能（PE-GenBO §3.4）：
1. LLM驱动的敏感度排序推理 → ARD长度尺度
2. 4D参数空间支持 (current1, time1, current2, v_switch)
3. 动态长度尺度计算：ℓ_{t,j} = ℓ_base * (rank_t(j) / d)^α_ℓ
"""

import numpy as np
import asyncio
import json
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI

# 导入config
from config import LLM_CONFIG, MOBO_CONFIG, get_llm_param


class LLAMBOWeighting:
    """
    LLAMBO风格的LLM权重函数 + 敏感度排序（PE-GenBO）
    
    核心功能：
    1. W_LLM权重计算：W_LLM(θ) = ∏_{j=1}^d [N(θ_j | μ_j, σ_j²)]
    2. 敏感度排序推理：rank_t(j) ∈ {1,2,3,4}（LLM分析物理影响）
    3. ARD长度尺度驱动：ℓ_{t,j} = ℓ_base * (rank_t(j) / d)^α_ℓ
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, tuple],
        llm_api_key: str = None,
        base_url: str = None,
        model: str = None,
        sigma_scale: float = None,
        length_scale_base: float = None,
        length_scale_alpha: float = None,
        verbose: bool = False
    ):
        """
        初始化LLM权重模块（4D参数空间）
        
        参数：
            param_bounds: 参数边界 {current1, time1, current2, v_switch}
            llm_api_key: LLM API密钥
            base_url: API基础URL
            model: LLM模型
            sigma_scale: σ缩放因子（控制焦点宽度）
            length_scale_base: ARD长度尺度基础值 ℓ_base
            length_scale_alpha: 敏感度指数 α_ℓ
            verbose: 详细输出
        """
        self.param_bounds = param_bounds
        
        # 从config读取默认值
        llm_api_key = llm_api_key or LLM_CONFIG['api_key']
        base_url = base_url or LLM_CONFIG['base_url']
        model = model or LLM_CONFIG['model']
        self.sigma_scale = sigma_scale if sigma_scale is not None else get_llm_param('weighting', 'sigma_scale', 0.15)
        self.length_scale_base = length_scale_base if length_scale_base is not None else get_llm_param('weighting', 'length_scale_base', 0.5)
        self.length_scale_alpha = length_scale_alpha if length_scale_alpha is not None else get_llm_param('weighting', 'length_scale_alpha', 0.8)
        self.verbose = verbose
        
        # LLM客户端
        if llm_api_key is not None:
            self.client = AsyncOpenAI(base_url=base_url, api_key=llm_api_key)
            self.model = model
        else:
            self.client = None
        
        # 参数维度（4D）
        self.param_names = ['current1', 'time1', 'current2', 'v_switch']
        self.d = len(self.param_names)  # d=4
        
        # 焦点参数（初始化为参数空间中心）
        self.mu_focus = np.array([
            (param_bounds['current1'][0] + param_bounds['current1'][1]) / 2,
            (param_bounds['time1'][0] + param_bounds['time1'][1]) / 2,
            (param_bounds['current2'][0] + param_bounds['current2'][1]) / 2,
            (param_bounds['v_switch'][0] + param_bounds['v_switch'][1]) / 2
        ])
        
        # 焦点宽度（初始化为参数范围的sigma_scale）
        param_ranges = np.array([
            param_bounds['current1'][1] - param_bounds['current1'][0],
            param_bounds['time1'][1] - param_bounds['time1'][0],
            param_bounds['current2'][1] - param_bounds['current2'][0],
            param_bounds['v_switch'][1] - param_bounds['v_switch'][0]
        ])
        self.sigma_focus = self.sigma_scale * param_ranges
        
        # 敏感度排序（初始化为均等：[1,2,3,4]）
        self.sensitivity_ranking = np.array([1, 2, 3, 4], dtype=int)
        
        # ARD长度尺度（初始化并计算）
        self.ard_length_scales = np.full(self.d, self.length_scale_base)
        self._compute_length_scales()  # 根据初始排序计算
        
        # 记录更新历史
        self.update_history = []
        self.sensitivity_history = []
    
    # ========================================
    # Round 2新增：敏感度排序推理（§3.4）
    # ========================================
    async def infer_sensitivity_ranking(
        self,
        database: List[Dict],
        iteration: int
    ) -> np.ndarray:
        """
        从LLM推理参数敏感度排序（PE-GenBO §3.4）
        
        流程：
        1. 构建包含物理知识的Prompt（Top-10观测、失败样本、物理机制）
        2. LLM推理各参数对目标的影响强度
        3. 输出排序 rank_t(j) ∈ {1,2,3,4}（1=最敏感，4=最不敏感）
        4. 验证排序有效性（必须是{1,2,3,4}的排列）
        5. 更新self.sensitivity_ranking并触发长度尺度计算
        
        参数：
            database: 评估历史
            iteration: 当前迭代轮次
        
        返回：
            ranking: (4,) 数组，ranking[j] = rank_t(j)
        """
        if self.client is None:
            if self.verbose:
                print("  [Sensitivity] LLM未配置，使用默认排序")
            return self.sensitivity_ranking
        
        # 构建Prompt
        prompt = self._build_sensitivity_prompt(database)
        
        if self.verbose:
            print(f"  [Sensitivity] 推理第{iteration}轮敏感度排序...")
        
        max_retries = get_llm_param('weighting', 'sensitivity_max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                # 调用LLM（JSON模式）
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
                        print(f"    排序: {ranking} → 长度尺度: {self.ard_length_scales}")
                    
                    return ranking
                
                else:
                    if self.verbose:
                        print(f"    [尝试{attempt+1}/{max_retries}] 解析失败，重试...")
            
            except Exception as e:
                if self.verbose:
                    print(f"    [尝试{attempt+1}/{max_retries}] LLM调用失败: {e}")
        
        # 重试耗尽，返回默认排序
        if self.verbose:
            print(f"  [Sensitivity] 推理失败，使用默认排序")
        return self.sensitivity_ranking
    
    def _build_sensitivity_prompt(self, database: List[Dict]) -> str:
        """
        构建敏感度排序Prompt（包含物理知识）
        
        Prompt结构：
        1. Top-10观测（按时间排序）+ Top-5按温度 + Top-5按老化
        2. 失败样本（invalid）
        3. 数据统计摘要
        4. 物理机制说明（仅描述机制，不给出敏感度标签）
        5. 任务描述（排序输出）
        """
        # 1. Top-10观测（按时间排序）
        valid_data = [r for r in database if r['valid']]
        
        obs_str = ""
        if len(valid_data) > 0:
            sorted_data_time = sorted(valid_data, key=lambda x: x['time'])
            top_10_time = sorted_data_time[:10]
            
            obs_str += "**Top-10 by Time (fastest charging):**\n"
            for i, r in enumerate(top_10_time, 1):
                p = r['params']
                obs_str += (
                    f"  Exp {i}: I1={p['current1']:.2f}A, T1={p['time1']:.1f}min, "
                    f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V "
                    f"→ Time={r['time']:.1f}min, Temp={r['temp']:.1f}K, Aging={r['aging']:.5f}\n"
                )
            
            # 添加Top-5按温度
            sorted_data_temp = sorted(valid_data, key=lambda x: x['temp'])
            top_5_temp = sorted_data_temp[:5]
            
            obs_str += "\n**Top-5 by Temperature (coolest):**\n"
            for i, r in enumerate(top_5_temp, 1):
                p = r['params']
                obs_str += (
                    f"  Exp {i}: I1={p['current1']:.2f}A, T1={p['time1']:.1f}min, "
                    f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V "
                    f"→ Time={r['time']:.1f}min, Temp={r['temp']:.1f}K, Aging={r['aging']:.5f}\n"
                )
            
            # 添加Top-5按老化
            sorted_data_aging = sorted(valid_data, key=lambda x: x['aging'])
            top_5_aging = sorted_data_aging[:5]
            
            obs_str += "\n**Top-5 by Aging (least degradation):**\n"
            for i, r in enumerate(top_5_aging, 1):
                p = r['params']
                obs_str += (
                    f"  Exp {i}: I1={p['current1']:.2f}A, T1={p['time1']:.1f}min, "
                    f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V "
                    f"→ Time={r['time']:.1f}min, Temp={r['temp']:.1f}K, Aging={r['aging']:.5f}\n"
                )
            
            # 数据统计摘要
            params_array = np.array([[r['params']['current1'], r['params']['time1'], 
                                     r['params']['current2'], r['params']['v_switch']] 
                                    for r in valid_data])
            
            obs_str += "\n**Data Summary:**\n"
            obs_str += f"  current1 range: [{params_array[:, 0].min():.2f}, {params_array[:, 0].max():.2f}]A, mean={params_array[:, 0].mean():.2f}A\n"
            obs_str += f"  time1 range: [{params_array[:, 1].min():.1f}, {params_array[:, 1].max():.1f}]min, mean={params_array[:, 1].mean():.1f}min\n"
            obs_str += f"  current2 range: [{params_array[:, 2].min():.2f}, {params_array[:, 2].max():.2f}]A, mean={params_array[:, 2].mean():.2f}A\n"
            obs_str += f"  v_switch range: [{params_array[:, 3].min():.2f}, {params_array[:, 3].max():.2f}]V, mean={params_array[:, 3].mean():.2f}V\n"
        else:
            obs_str = "  (No valid data yet)\n"
        
        # 2. 失败样本
        invalid_data = [r for r in database if not r['valid']]
        
        if len(invalid_data) > 0:
            fail_str = ""
            for i, r in enumerate(invalid_data[:5], 1):
                p = r['params']
                fail_str += (
                    f"  Failed {i}: I1={p['current1']:.2f}A, T1={p['time1']:.1f}min, "
                    f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V\n"
                )
        else:
            fail_str = "  (No failures yet)\n"
        
        # 3. 物理机制（仅描述机制，不给出敏感度标签）
        physics_str = """
**Physical Mechanisms:**
1. **current1 (I1)**: Higher current accelerates charging but increases heat generation (Joule heating ∝ I²R). 
   - Direct impact on power dissipation and thermal stress
   
2. **time1 (T1)**: Duration of high-current Phase 1
   - Controls total energy input and cumulative heat accumulation
   - Longer duration extends exposure to high-power conditions
   
3. **current2 (I2)**: Phase 2 tapering current
   - Determines charging rate during final stage
   - Lower values reduce thermal and electrical stress
   
4. **v_switch (V_sw)**: Voltage threshold for phase switch
   - Controls when transition from constant current to constant voltage occurs
   - Higher thresholds extend high-current phase duration
"""
        
        # 完整Prompt
        prompt = f"""
You are analyzing a battery fast-charging optimization problem with 4 decision variables:
- **current1** (I1): Phase 1 current [3.0-6.0]A
- **time1** (T1): Phase 1 duration [2.0-40.0]min
- **current2** (I2): Phase 2 current [1.0-4.0]A  
- **v_switch** (V_sw): Voltage switch point [3.8-4.2]V

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
Based on the experimental data above, rank the 4 parameters by their sensitivity to the objectives (time, temperature, aging).

Your ranking MUST be informed by the experimental data above. Different iterations may have different rankings depending on the observed trade-offs. Analyze which parameters show the strongest correlation with objective variations in the data.

Output JSON:
{{
  "ranking": {{
    "current1": <rank 1-4, where 1=most sensitive>,
    "time1": <rank 1-4>,
    "current2": <rank 1-4>,
    "v_switch": <rank 1-4>
  }},
  "reasoning": "<brief explanation based on observed data patterns>"
}}

**CRITICAL**: The 4 ranks must be a permutation of [1,2,3,4] (each used exactly once).
"""
        return prompt
    
    def _parse_sensitivity_response(self, result: Dict) -> Optional[np.ndarray]:
        """
        解析LLM响应，提取敏感度排序
        
        验证：
        1. 必须包含4个参数的排序
        2. 排序必须是{1,2,3,4}的排列
        
        参数：
            result: LLM JSON响应
        
        返回：
            ranking: (4,) 数组 [rank_current1, rank_time1, rank_current2, rank_v_switch]
                    或 None（解析失败）
        """
        try:
            ranking_dict = result.get('ranking', {})
            
            # 提取排序
            ranks = []
            for param_name in self.param_names:
                rank = ranking_dict.get(param_name)
                
                if rank is None:
                    if self.verbose:
                        print(f"    [解析错误] 缺少参数: {param_name}")
                    return None
                
                rank = int(rank)
                
                if rank < 1 or rank > 4:
                    if self.verbose:
                        print(f"    [解析错误] {param_name}排序越界: {rank}")
                    return None
                
                ranks.append(rank)
            
            ranking = np.array(ranks, dtype=int)
            
            # 验证是否为排列
            if not np.array_equal(sorted(ranking), [1, 2, 3, 4]):
                if self.verbose:
                    print(f"    [解析错误] 不是有效排列: {ranking}")
                return None
            
            return ranking
        
        except Exception as e:
            if self.verbose:
                print(f"    [解析错误] {e}")
            return None
    
    def _compute_length_scales(self):
        """
        计算ARD长度尺度（PE-GenBO公式）
        
        公式：ℓ_{t,j} = ℓ_base * (rank_t(j) / d)^α_ℓ
        
        其中：
        - ℓ_base: 基础长度尺度（config中配置）
        - rank_t(j): 参数j的敏感度排序（1=最敏感，4=最不敏感）
        - d: 参数维度（4）
        - α_ℓ: 敏感度指数（config中配置，推荐0.8）
        
        效果：
        - rank=1（最敏感）→ ℓ小 → GP对该维度平滑度要求高
        - rank=4（最不敏感）→ ℓ大 → GP允许更大变化
        """
        self.ard_length_scales = self.length_scale_base * \
            (self.sensitivity_ranking / self.d) ** self.length_scale_alpha
    
    def get_length_scales(self) -> np.ndarray:
        """
        获取当前ARD长度尺度（供GP模型使用）
        
        返回：
            length_scales: (4,) 数组 [ℓ_current1, ℓ_time1, ℓ_current2, ℓ_v_switch]
        """
        return self.ard_length_scales.copy()
    
    def get_sensitivity_ranking(self) -> np.ndarray:
        """
        获取当前敏感度排序
        
        返回：
            ranking: (4,) 数组 [rank_current1, rank_time1, rank_current2, rank_v_switch]
        """
        return self.sensitivity_ranking.copy()
    
    # ========================================
    # 原有功能（升级到4D）
    # ========================================
    async def update_focus_from_llm(
        self,
        database: List[Dict],
        gp_list: List = None,
        iteration: int = 0
    ):
        """
        从LLM推理更新焦点中心μ（4D版本）
        
        策略：
        1. 提取当前Pareto前沿
        2. 向LLM询问：哪个参数区域最有潜力？
        3. 更新μ_focus（4D）
        4. 从GP更新σ_focus（可选）
        
        参数：
            database: 评估历史
            gp_list: GP模型列表（用于推导σ）
            iteration: 当前迭代轮次
        """
        if self.client is None:
            if self.verbose:
                print("  [LLM Weighting] LLM未配置，跳过焦点更新")
            return
        
        # 提取Pareto前沿
        pareto_front = self._extract_pareto(database)
        
        if len(pareto_front) < 3:
            if self.verbose:
                print(f"  [LLM Weighting] Pareto点不足({len(pareto_front)})，跳过更新")
            return
        
        # 构建Prompt
        prompt = self._build_focus_prompt(pareto_front)
        
        if self.verbose:
            print(f"  [LLM Weighting] 调用LLM推理焦点...")
        
        try:
            # 调用LLM
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
            
            # 提取焦点中心（4D）
            new_mu = np.array([
                float(result.get('focus_current1', self.mu_focus[0])),
                float(result.get('focus_time1', self.mu_focus[1])),
                float(result.get('focus_current2', self.mu_focus[2])),
                float(result.get('focus_v_switch', self.mu_focus[3]))
            ])
            
            # 边界检查
            new_mu[0] = np.clip(new_mu[0], *self.param_bounds['current1'])
            new_mu[1] = np.clip(new_mu[1], *self.param_bounds['time1'])
            new_mu[2] = np.clip(new_mu[2], *self.param_bounds['current2'])
            new_mu[3] = np.clip(new_mu[3], *self.param_bounds['v_switch'])
            
            # 更新焦点
            self.mu_focus = new_mu
            
            if self.verbose:
                print(f"    焦点已更新: μ=[{self.mu_focus[0]:.2f}, {self.mu_focus[1]:.1f}, {self.mu_focus[2]:.2f}, {self.mu_focus[3]:.2f}]")
            
            # 从GP更新σ（如果提供）
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
                print(f"  [LLM Weighting] 焦点更新失败: {e}")
    
    def compute_weight(self, x: np.ndarray) -> float:
        """
        计算LLAMBO公式(9)的权重（4D版本）
        
        W_LLM(θ) = ∏_{j=1}^4 [N(θ_j | μ_j, σ_j²)]
        
        其中 N(θ_j | μ_j, σ_j²) = 1/√(2πσ_j²) exp(-(θ_j - μ_j)²/(2σ_j²))
        
        参数：
            x: 候选点 (4,) [current1, time1, current2, v_switch]
        
        返回：
            weight: [0, 1]，归一化权重
        """
        # 确保x是1D数组
        if x.ndim > 1:
            x = x.flatten()
        
        # 计算多元高斯密度（连乘）
        weight = 1.0
        
        for j in range(self.d):
            # 高斯密度：N(θ_j | μ_j, σ_j²)
            gaussian_term = (1.0 / np.sqrt(2 * np.pi * self.sigma_focus[j]**2)) * \
                            np.exp(-(x[j] - self.mu_focus[j])**2 / (2 * self.sigma_focus[j]**2))
            
            weight *= gaussian_term
        
        # 归一化到[0, 1]
        # 理论最大值在μ处
        max_weight = 1.0
        for j in range(self.d):
            max_weight *= (1.0 / np.sqrt(2 * np.pi * self.sigma_focus[j]**2))
        
        normalized_weight = weight / max_weight if max_weight > 1e-10 else 1.0
        
        # 限制在[0, 1]
        normalized_weight = np.clip(normalized_weight, 0.0, 1.0)
        
        # 保留至少10%的探索能力
        normalized_weight = max(normalized_weight, 0.1)
        
        return normalized_weight
    
    def _extract_pareto(self, database: List[Dict]) -> List[Dict]:
        """提取Pareto前沿"""
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) == 0:
            return []
        
        # 提取目标值
        objectives = np.array([[
            r['time'],
            r['temp'],
            r['aging']
        ] for r in valid_data])
        
        # Pareto支配
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
    
    def _build_focus_prompt(self, pareto_front: List[Dict]) -> str:
        """构建LLM Prompt（4D版本）"""
        # 序列化Pareto点
        points_str = ""
        for i, point in enumerate(pareto_front[:10], 1):  # 最多10个点
            p = point['params']
            points_str += (
                f"Point {i}: I1={p['current1']:.2f}A, T1={p['time1']:.1f}min, "
                f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V "
                f"→ Time={point['time']:.1f}min, Temp={point['temp']:.1f}K, Aging={point['aging']:.5f}\n"
            )
        
        prompt = f"""
Given {len(pareto_front)} Pareto-optimal battery charging strategies:

{points_str}

**Physics Context:**
- High current1 (I1) → Fast charging but more heat
- Long time1 (T1) → Extended high-current phase, heat accumulation
- Low current2 (I2) → Slower Phase 2, less stress
- High v_switch (V_sw) → Later switch, more high-current exposure

**Trade-offs:**
- Time vs Temperature: Higher I1/longer T1 reduces time but increases temperature
- Temperature vs Aging: Higher temperature accelerates aging
- I1 and T1 are coupled: High I1 + Long T1 = thermal runaway risk

**Task:**
Based on the Pareto points above, suggest the most promising parameter region for the NEXT exploration.

Output JSON:
{{
  "focus_current1": <value in [3.0, 6.0]>,
  "focus_time1": <value in [2.0, 40.0]>,
  "focus_current2": <value in [1.0, 4.0]>,
  "focus_v_switch": <value in [3.8, 4.2]>,
  "reasoning": "<brief explanation>"
}}
"""
        return prompt
    
    def _update_sigma_from_gp(self, gp_list: List):
        """从GP预测不确定性更新σ（4D版本，修复量纲问题）"""
        try:
            # 在μ处预测
            X_mu = self.mu_focus.reshape(1, -1)
            
            # 收集每个GP在μ处的标准差
            std_list = []
            for gp in gp_list:
                _, std = gp.predict(X_mu, return_std=True)
                std_list.append(float(std[0]))
            
            # 参数范围
            param_ranges = np.array([
                self.param_bounds['current1'][1] - self.param_bounds['current1'][0],
                self.param_bounds['time1'][1] - self.param_bounds['time1'][0],
                self.param_bounds['current2'][1] - self.param_bounds['current2'][0],
                self.param_bounds['v_switch'][1] - self.param_bounds['v_switch'][0]
            ])
            
            # 归一化GP不确定性到[0,1]：除以参考点范围（消除量纲影响）
            # 三个GP分别对应 time, temp, aging，量级差异巨大
            # 使用归一化的平均不确定性（0=完全确定，1=最大不确定）
            ref_ranges = np.array([
                MOBO_CONFIG['reference_point']['time'] - MOBO_CONFIG['ideal_point']['time'],
                MOBO_CONFIG['reference_point']['temp'] - MOBO_CONFIG['ideal_point']['temp'],
                MOBO_CONFIG['reference_point']['aging'] - MOBO_CONFIG['ideal_point']['aging']
            ])
            ref_ranges = np.maximum(ref_ranges, 1e-10)  # 防止除零
            
            normalized_stds = np.array(std_list) / ref_ranges
            avg_normalized_std = float(np.mean(normalized_stds))
            avg_normalized_std = np.clip(avg_normalized_std, 0.0, 1.0)
            
            # σ = σ_base × (1 + α_uncertainty × normalized_std)
            # σ_base = sigma_scale × param_ranges（初始值）
            sigma_base = self.sigma_scale * param_ranges
            uncertainty_amplification = 1.0 + 2.0 * avg_normalized_std  # 最大放大3倍
            
            self.sigma_focus = sigma_base * uncertainty_amplification
            
            if self.verbose:
                print(f"    σ已更新: {self.sigma_focus}")
                print(f"    (归一化GP不确定性: {avg_normalized_std:.4f}, 放大系数: {uncertainty_amplification:.2f})")
        
        except Exception as e:
            if self.verbose:
                print(f"    [警告] σ更新失败: {e}")


# ============================================================
# 快速测试（4D版本）
# ============================================================
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("测试 LLAMBOWeighting（4D）...")
        
        # 参数边界（4D）
        param_bounds = {
            'current1': (3.0, 6.0),
            'time1': (2.0, 40.0),
            'current2': (1.0, 4.0),
            'v_switch': (3.8, 4.2)
        }
        
        # 初始化（无LLM）
        weighting = LLAMBOWeighting(
            param_bounds=param_bounds,
            llm_api_key=None,
            verbose=True
        )
        
        print(f"\n初始焦点: μ={weighting.mu_focus}")
        print(f"初始宽度: σ={weighting.sigma_focus}")
        print(f"初始排序: {weighting.sensitivity_ranking}")
        print(f"初始长度尺度: {weighting.ard_length_scales}")
        
        # 测试权重计算（4D）
        test_points = [
            np.array([4.5, 21.0, 2.5, 4.0]),  # 中心点
            np.array([6.0, 40.0, 1.0, 4.2]),  # 边界点
            np.array([3.0, 2.0, 4.0, 3.8])    # 另一边界
        ]
        
        print("\n测试权重计算:")
        for i, x in enumerate(test_points, 1):
            weight = weighting.compute_weight(x)
            print(f"  点{i} {x} → 权重={weight:.4f}")
        
        # 测试长度尺度计算
        print("\n测试长度尺度计算:")
        test_rankings = [
            [1, 2, 3, 4],  # current1最敏感
            [2, 1, 3, 4],  # time1最敏感
            [4, 3, 2, 1]   # v_switch最敏感
        ]
        
        for ranking in test_rankings:
            weighting.sensitivity_ranking = np.array(ranking)
            weighting._compute_length_scales()
            print(f"  排序{ranking} → 长度尺度={weighting.ard_length_scales}")
        
        print("\n测试完成！")
    
    asyncio.run(test())
