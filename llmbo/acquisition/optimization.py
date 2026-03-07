"""
采集函数优化模块（PE-GenBO §4）
Round 2新增：两段式生成式采集优化

核心创新：
1. 传统采集（经典方法）：MC-EI + 随机海选 + L-BFGS
2. 生成式采集（Round 2）：LLM生成候选 → GP-LCB×W_LLM选择
3. 阈值退火：动态平衡探索-开发
"""

import numpy as np
import json
import asyncio
from typing import Dict, List, Optional, Callable
from scipy.optimize import minimize
from scipy.stats import norm
from openai import AsyncOpenAI

from acquisition.scalarization import TchebycheffScalarizer
from config import MOBO_CONFIG, LLM_CONFIG, get_algorithm_param, get_llm_param


class AcquisitionOptimizer:
    """
    采集函数优化器（PE-GenBO）
    
    两种优化模式：
    1. optimize_classic(): 经典MC-EI（随机海选 + L-BFGS）
    2. optimize(): 生成式采集（LLM候选生成 + PE acquisition）
    
    PE acquisition：α^PE(θ) = [f̂(θ) - β·s(θ)] / max(W_LLM(θ), 0.01)
    """
    
    def __init__(
        self,
        param_bounds: Dict,
        scalarizer=None,
        n_candidates: int = None,
        n_mc_samples: int = None,
        enable_llm_weighting: bool = False,
        llm_api_key: str = None,
        base_url: str = None,
        model: str = None,
        verbose: bool = False
    ):
        """
        初始化采集优化器（4D参数空间）
        
        参数：
            param_bounds: 参数边界 {current1, time1, current2, v_switch}
            scalarizer: Tchebycheff标量化器（可选）
            n_candidates: 候选点数（阶段1，经典方法）
            n_mc_samples: MC采样数（经典方法）
            enable_llm_weighting: 是否启用LLM权重
            llm_api_key: LLM API密钥
            base_url: API基础URL
            model: LLM模型
            verbose: 详细输出
        """
        self.param_bounds = param_bounds
        
        # 从config读取默认值
        self.n_candidates = n_candidates if n_candidates is not None else get_algorithm_param('acquisition', 'n_candidates', 2000)
        self.n_mc_samples = n_mc_samples if n_mc_samples is not None else get_algorithm_param('acquisition', 'n_mc_samples', 128)
        self.enable_llm_weighting = enable_llm_weighting
        self.verbose = verbose
        
        # LLM客户端
        llm_api_key = llm_api_key or LLM_CONFIG['api_key']
        base_url = base_url or LLM_CONFIG['base_url']
        model = model or LLM_CONFIG['model']
        
        if llm_api_key is not None:
            self.client = AsyncOpenAI(base_url=base_url, api_key=llm_api_key)
            self.model = model
        else:
            self.client = None
        
        # 转换为列表形式（用于scipy）
        self.param_names = ['current1', 'time1', 'current2', 'v_switch']
        self.bounds_list = [
            param_bounds['current1'],
            param_bounds['time1'],
            param_bounds['current2'],
            param_bounds['v_switch']
        ]
        
        # 初始化标量化器（会在optimize()中动态更新）
        self.scalarizer = scalarizer
        
        # PE acquisition参数
        self.beta = get_llm_param('acquisition', 'beta', 2.0)
        self.n_batch = get_llm_param('acquisition', 'n_batch', 50)
        
        # 阈值退火参数
        self.gamma_0 = get_llm_param('acquisition', 'threshold_gamma_0', 50.0)
        self.gamma_T = get_llm_param('acquisition', 'threshold_gamma_T', 99.0)
        self.threshold_percentile = self.gamma_0  # 初始化为探索
        
        # 高方差区域参数
        self.B_ratio = get_llm_param('acquisition', 'B_explore_ratio_init', 0.1)
    
    # ========================================
    # Round 2新增：生成式采集优化（§4）
    # ========================================
    async def optimize(
        self,
        gp_list: List,
        weights: np.ndarray,
        database: List[Dict],
        llm_weight_func: Optional[Callable] = None,
        iteration: int = 0,
        total_iterations: int = 100,
        scalarizer: TchebycheffScalarizer = None,
        legacy_db: List[Dict] = None
    ) -> Dict:
        """
        生成式采集优化（PE-GenBO §4）
        
        两段式流程：
        1. LLM生成候选：基于Top-K观测、失败样本、高方差区域
        2. PE acquisition选择：α^PE = lcb_scalar / max(w_llm, 0.01)
        
        阈值退火：γ_t从50%（探索）退火到99%（开发）
        
        参数：
            gp_list: GP模型列表 [gp_time, gp_temp, gp_aging]
            weights: Dirichlet权重 (3,)
            database: 评估历史（transformed，用于GP预测/标量化）
            llm_weight_func: LLM权重函数（可选）
            iteration: 当前迭代轮次
            total_iterations: 总迭代数
            scalarizer: 外部标量化器（来自main loop Step F，空间一致）
            legacy_db: 原始空间数据库（用于LLM prompt，避免LLM看到log值）
        
        返回：
            best_params: {'current1', 'time1', 'current2', 'v_switch'}
        """
        # 回退：如果LLM未配置，使用经典方法
        if self.client is None:
            if self.verbose:
                print("  [Acquisition] LLM未配置，使用经典优化")
            return self.optimize_classic(gp_list, weights, database, llm_weight_func,
                                         scalarizer=scalarizer)
        
        # 更新阈值百分位（探索→开发）
        self._update_threshold_percentile(iteration, total_iterations)
        
        # 使用外部标量化器（空间一致）或回退到内部构建
        if scalarizer is not None:
            self.scalarizer = scalarizer
        else:
            self._update_scalarizer(database)
        
        # 识别高方差区域
        high_variance_regions = self._identify_high_variance_regions(gp_list, database)
        
        if self.verbose:
            print(f"  [Acquisition] 生成式优化 - 第{iteration}轮")
            print(f"    阈值百分位: {self.threshold_percentile:.1f}%")
            print(f"    高方差区域数: {len(high_variance_regions)}")
        
        # ===== 阶段1: LLM生成候选 =====
        # 使用 legacy_db（原始空间）构建 LLM prompt，避免 LLM 看到 log 值
        prompt_db = legacy_db if legacy_db is not None else database
        llm_candidates = await self._llm_generate_candidates(
            prompt_db, high_variance_regions, iteration
        )
        
        if len(llm_candidates) == 0:
            if self.verbose:
                print("    [警告] LLM生成失败，回退到经典方法")
            return self.optimize_classic(gp_list, weights, database, llm_weight_func)
        
        if self.verbose:
            print(f"    LLM生成候选数: {len(llm_candidates)}")
        
        # ===== 阶段2: PE acquisition选择 =====
        pe_scores = []
        
        for candidate in llm_candidates:
            x = np.array([
                candidate['current1'],
                candidate['time1'],
                candidate['current2'],
                candidate['v_switch']
            ])
            
            # 计算PE acquisition
            pe_score = self._compute_pe_acquisition(
                x, gp_list, weights, llm_weight_func
            )
            
            pe_scores.append(pe_score)
        
        pe_scores = np.array(pe_scores)
        
        # 选择最优（PE采集最小 = 最有潜力）
        best_idx = np.argmin(pe_scores)
        best_candidate = llm_candidates[best_idx]
        best_score = pe_scores[best_idx]
        
        if self.verbose:
            print(f"    PE采集范围: [{np.min(pe_scores):.4f}, {np.max(pe_scores):.4f}]")
            print(f"    最优候选: PE={best_score:.4f}")
        
        return {
            'current1': float(best_candidate['current1']),
            'time1': float(best_candidate['time1']),
            'current2': float(best_candidate['current2']),
            'v_switch': float(best_candidate['v_switch'])
        }
    
    async def _llm_generate_candidates(
        self,
        database: List[Dict],
        high_variance_regions: List[Dict],
        iteration: int
    ) -> List[Dict]:
        """
        LLM生成候选点（并行多温度）
        
        策略：
        - 在多个温度下生成：[0.6, 0.8, 1.0]
        - 每个温度生成n_batch个候选
        - 合并去重
        
        参数：
            database: 评估历史
            high_variance_regions: 高方差区域提示
            iteration: 当前迭代轮次
        
        返回：
            candidates: [{'current1', 'time1', 'current2', 'v_switch'}, ...]
        """
        prompt = self._build_generation_prompt(database, high_variance_regions, iteration)
        
        temperatures = get_llm_param('acquisition', 'gen_temperatures', [0.6, 0.8, 1.0])
        max_retries = get_llm_param('acquisition', 'gen_max_retries', 3)
        
        # 并行调用LLM
        tasks = []
        for temp in temperatures:
            tasks.append(self._call_llm_generation(prompt, temp, max_retries))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并候选
        all_candidates = []
        for result in results:
            if isinstance(result, list):
                all_candidates.extend(result)
        
        # 去重（四舍五入到2位小数）
        unique_candidates = []
        seen = set()
        
        for candidate in all_candidates:
            key = (
                round(candidate['current1'], 2),
                round(candidate['time1'], 1),
                round(candidate['current2'], 2),
                round(candidate['v_switch'], 2)
            )
            
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        
        if self.verbose:
            print(f"    LLM生成: {len(all_candidates)}个 → 去重后{len(unique_candidates)}个")
        
        return unique_candidates
    
    async def _call_llm_generation(
        self,
        prompt: str,
        temperature: float,
        max_retries: int
    ) -> List[Dict]:
        """
        单次LLM调用生成候选
        
        参数：
            prompt: 生成Prompt
            temperature: 温度参数
            max_retries: 最大重试次数
        
        返回：
            candidates: 候选列表
        """
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in battery fast-charging optimization. "
                                "Generate diverse promising parameter combinations based on historical data and physical constraints. "
                                "Output strictly in JSON format."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=get_llm_param('acquisition', 'gen_max_tokens', 1500),
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                candidates = self._parse_candidates(result)
                
                if candidates is not None and len(candidates) > 0:
                    return candidates
            
            except Exception as e:
                if self.verbose:
                    print(f"    [生成失败 T={temperature}, 尝试{attempt+1}] {e}")
        
        return []
    
    def _parse_candidates(self, result: Dict) -> Optional[List[Dict]]:
        """
        解析LLM生成的候选点
        
        验证：
        - 必须在边界内
        - 必须包含4个参数
        
        参数：
            result: LLM JSON响应
        
        返回：
            candidates: [{'current1', 'time1', 'current2', 'v_switch'}, ...]
                       或 None（解析失败）
        """
        try:
            raw_candidates = result.get('candidates', [])
            
            if not isinstance(raw_candidates, list):
                return None
            
            valid_candidates = []
            
            for c in raw_candidates:
                # 提取参数
                try:
                    candidate = {
                        'current1': float(c.get('current1')),
                        'time1': float(c.get('time1')),
                        'current2': float(c.get('current2')),
                        'v_switch': float(c.get('v_switch'))
                    }
                except (TypeError, ValueError, KeyError):
                    continue
                
                # 边界检查
                if not (self.param_bounds['current1'][0] <= candidate['current1'] <= self.param_bounds['current1'][1]):
                    continue
                if not (self.param_bounds['time1'][0] <= candidate['time1'] <= self.param_bounds['time1'][1]):
                    continue
                if not (self.param_bounds['current2'][0] <= candidate['current2'] <= self.param_bounds['current2'][1]):
                    continue
                if not (self.param_bounds['v_switch'][0] <= candidate['v_switch'] <= self.param_bounds['v_switch'][1]):
                    continue
                
                valid_candidates.append(candidate)
            
            return valid_candidates if len(valid_candidates) > 0 else None
        
        except Exception as e:
            if self.verbose:
                print(f"    [解析错误] {e}")
            return None
    
    def _build_generation_prompt(
        self,
        database: List[Dict],
        high_variance_regions: List[Dict],
        iteration: int
    ) -> str:
        """构建LLM候选生成Prompt（动态参数范围 + 正确单位）"""
        # 动态获取参数边界
        b = self.param_bounds

        # 1. Top-10观测
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) > 0:
            sorted_data = sorted(valid_data, key=lambda x: x['time'])
            top_10 = sorted_data[:10]
            
            obs_str = ""
            for i, r in enumerate(top_10, 1):
                p = r['params']
                obs_str += (
                    f"  #{i}: I1={p['current1']:.2f}A, T1={p['time1']:.0f}s, "
                    f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V "
                    f"→ Time={r['time']:.0f}s, Temp={r['temp']:.1f}K, Aging={r['aging']:.5f}\n"
                )
        else:
            obs_str = "  (No valid data yet)\n"
        
        # 2. 失败样本
        invalid_data = [r for r in database if not r['valid']]
        
        if len(invalid_data) > 0:
            fail_str = ""
            for i, r in enumerate(invalid_data[:5], 1):
                p = r['params']
                fail_str += (
                    f"  Failed #{i}: I1={p['current1']:.2f}A, T1={p['time1']:.0f}s, "
                    f"I2={p['current2']:.2f}A, V_sw={p['v_switch']:.2f}V "
                    f"(Reason: {r.get('violation', r.get('error', 'Unknown'))})\n"
                )
        else:
            fail_str = "  (No failures yet)\n"
        
        # 3. 高方差区域
        if len(high_variance_regions) > 0:
            variance_str = f"Model predicts HIGH UNCERTAINTY in these regions:\n"
            for i, region in enumerate(high_variance_regions[:3], 1):
                variance_str += (
                    f"  Region {i}: I1~{region['current1']:.2f}, T1~{region['time1']:.0f}s, "
                    f"I2~{region['current2']:.2f}, V_sw~{region['v_switch']:.2f} "
                    f"(σ={region['std']:.3f})\n"
                )
        else:
            variance_str = "Model is relatively confident across the space.\n"
        
        # 完整Prompt（使用动态边界）
        prompt = f"""
You are optimizing battery fast-charging with 4 decision variables:
- **current1** (I1): Phase 1 current [{b['current1'][0]}-{b['current1'][1]}] A
- **time1** (T1): Phase 1 duration [{b['time1'][0]:.0f}-{b['time1'][1]:.0f}] seconds
- **current2** (I2): Phase 2 current [{b['current2'][0]}-{b['current2'][1]}] A
- **v_switch** (V_sw): Voltage switch point [{b['v_switch'][0]}-{b['v_switch'][1]}] V

Objectives to MINIMIZE:
- Charging time (seconds)
- Peak temperature (K)
- Capacity fade (aging %)

**Best Experiments So Far (Top-10 by time):**
{obs_str}

**Failed Experiments (violated constraints):**
{fail_str}

**Regions to Explore (high model uncertainty):**
{variance_str}

**Physical Constraints:**
- I1 ↔ T1 coupling: High I1 + Long T1 → Thermal runaway risk
- Temperature limit: Must stay below 318.15K (45°C)
- Aging mechanism: High temp × time → irreversible SEI growth
- V_sw effect: Higher V_sw → Longer high-current phase

**Current Status:** Iteration {iteration}
- Early iterations → EXPLORE: Try diverse regions, especially high-uncertainty areas
- Later iterations → EXPLOIT: Refine around best-performing combinations

**Task:**
Generate {self.n_batch} diverse parameter combinations that are:
1. Likely to improve upon current best (minimize time/temp/aging trade-off)
2. Respect physical constraints (avoid extreme I1+T1 combinations)
3. Include some points in high-uncertainty regions (for exploration)
4. Diverse across the parameter space (avoid clustering)

Output JSON:
{{
  "candidates": [
    {{
            "current1": <value in [{b['current1'][0]}, {b['current1'][1]}]>,
            "time1": <value in [{b['time1'][0]:.0f}, {b['time1'][1]:.0f}]>,
            "current2": <value in [{b['current2'][0]}, {b['current2'][1]}]>,
            "v_switch": <value in [{b['v_switch'][0]}, {b['v_switch'][1]}]>
    }},
    ... ({self.n_batch} candidates total)
  ],
  "reasoning": "<brief strategy explanation>"
}}
"""
        return prompt
    
    def _compute_pe_acquisition(
        self,
        x: np.ndarray,
        gp_list: List,
        weights: np.ndarray,
        llm_weight_func: Optional[Callable] = None
    ) -> float:
        """
        计算PE acquisition（PE-GenBO §4.1）
        
        公式：α^PE(θ) = [f̂(θ) - β·s(θ)] / max(W_LLM(θ), 0.01)
                       = lcb_scalar / max(w_llm, 0.01)
        
        其中：
        - f̂(θ): GP均值预测（标量化后）
        - s(θ): GP标准差预测（标量化后）
        - β: 置信参数（控制探索程度）
        - W_LLM(θ): LLM权重（用于惩罚焦点外的点）
        
        参数：
            x: 候选点 (4,)
            gp_list: GP列表
            weights: Dirichlet权重
            llm_weight_func: LLM权重函数
        
        返回：
            pe_score: PE采集值（越小越好）
        """
        x = x.reshape(1, -1)
        
        # GP预测
        mu_list = []
        std_list = []
        
        for gp in gp_list:
            mu, std = gp.predict(x, return_std=True)
            mu_list.append(mu[0])
            std_list.append(std[0])
        
        mu_list = np.array(mu_list)
        std_list = np.array(std_list)
        
        # 避免除零
        std_list = np.maximum(std_list, 1e-10)
        
        # GP-LCB（分量级）
        lcb_list = mu_list - self.beta * std_list
        
        # 标量化
        lcb_scalar = self.scalarizer.scalarize(lcb_list, weights, normalize=True)
        
        # LLM权重（如果提供）
        if llm_weight_func is not None:
            try:
                w_llm = llm_weight_func(x[0])
                w_llm = max(w_llm, 0.01)  # 避免除零
            except:
                w_llm = 1.0
        else:
            w_llm = 1.0
        
        # PE acquisition：lcb / w_llm
        # w_llm大（焦点内）→ PE小 → 优先选择
        # w_llm小（焦点外）→ PE大 → 惩罚
        pe_score = lcb_scalar / w_llm
        
        return pe_score
    
    def _identify_high_variance_regions(
        self,
        gp_list: List,
        database: List[Dict]
    ) -> List[Dict]:
        """
        识别高方差区域（§4.2）
        
        策略：
        - 在参数空间随机采样100个点
        - 计算GP标准差
        - 返回top-B个高方差点（B = B_ratio × 100）
        
        参数：
            gp_list: GP列表
            database: 评估历史（未使用，预留）
        
        返回：
            high_var_regions: [{'current1', 'time1', 'current2', 'v_switch', 'std'}, ...]
        """
        n_probe = 100
        B = int(self.B_ratio * n_probe)
        
        # 随机采样
        probe_points = np.random.rand(n_probe, 4)
        probe_points[:, 0] = probe_points[:, 0] * (self.param_bounds['current1'][1] - self.param_bounds['current1'][0]) + self.param_bounds['current1'][0]
        probe_points[:, 1] = probe_points[:, 1] * (self.param_bounds['time1'][1] - self.param_bounds['time1'][0]) + self.param_bounds['time1'][0]
        probe_points[:, 2] = probe_points[:, 2] * (self.param_bounds['current2'][1] - self.param_bounds['current2'][0]) + self.param_bounds['current2'][0]
        probe_points[:, 3] = probe_points[:, 3] * (self.param_bounds['v_switch'][1] - self.param_bounds['v_switch'][0]) + self.param_bounds['v_switch'][0]
        
        # 计算平均标准差
        avg_stds = []
        
        for x in probe_points:
            x_reshaped = x.reshape(1, -1)
            
            stds = []
            for gp in gp_list:
                _, std = gp.predict(x_reshaped, return_std=True)
                stds.append(std[0])
            
            avg_std = np.mean(stds)
            avg_stds.append(avg_std)
        
        avg_stds = np.array(avg_stds)
        
        # Top-B高方差点
        top_indices = np.argsort(avg_stds)[-B:][::-1]
        
        high_var_regions = []
        for idx in top_indices:
            x = probe_points[idx]
            high_var_regions.append({
                'current1': x[0],
                'time1': x[1],
                'current2': x[2],
                'v_switch': x[3],
                'std': avg_stds[idx]
            })
        
        return high_var_regions
    
    def _compute_threshold(
        self,
        candidates: List[Dict],
        pe_scores: np.ndarray
    ) -> float:
        """
        计算阈值（用于筛选候选）
        
        公式：τ_t = percentile(PE_scores, γ_t)
        
        参数：
            candidates: 候选列表
            pe_scores: PE采集值
        
        返回：
            threshold: 阈值
        """
        if len(pe_scores) == 0:
            return 0.0
        
        threshold = np.percentile(pe_scores, self.threshold_percentile)
        return threshold
    
    def _update_threshold_percentile(
        self,
        iteration: int,
        total_iterations: int
    ):
        """
        更新阈值百分位（探索→开发）
        
        公式：γ_t = γ_0 + (γ_T - γ_0) * (t / T)
        
        参数：
            iteration: 当前迭代轮次
            total_iterations: 总迭代数
        """
        progress = iteration / max(total_iterations, 1)
        self.threshold_percentile = self.gamma_0 + (self.gamma_T - self.gamma_0) * progress
    
    def _update_scalarizer(self, database: List[Dict]):
        """
        更新标量化器（从database推导理想点和参考点）
        
        参数：
            database: 评估历史
        """
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) == 0:
            # 使用默认值
            ideal_point = np.zeros(3)
            reference_point = np.array([
                MOBO_CONFIG['reference_point']['time'],
                MOBO_CONFIG['reference_point']['temp'],
                MOBO_CONFIG['reference_point']['aging']
            ])
        else:
            # 提取目标值
            times = np.array([r['time'] for r in valid_data])
            temps = np.array([r['temp'] for r in valid_data])
            agings = np.array([r['aging'] for r in valid_data])
            
            # 动态理想点和参考点
            ideal_point = np.array([
                np.min(times),
                np.min(temps),
                np.min(agings)
            ])
            
            reference_point = np.array([
                MOBO_CONFIG['reference_point']['time'],
                MOBO_CONFIG['reference_point']['temp'],
                MOBO_CONFIG['reference_point']['aging']
            ])
        
        # 初始化标量化器
        self.scalarizer = TchebycheffScalarizer(
            ideal_point=ideal_point,
            reference_point=reference_point,
            eta=MOBO_CONFIG['eta']
        )
    
    # ========================================
    # 经典优化方法（Round 1保留）
    # ========================================
    def optimize_classic(
        self,
        gp_list: List,
        weights: np.ndarray,
        database: List[Dict],
        llm_weight_func: Optional[Callable] = None,
        scalarizer: TchebycheffScalarizer = None
    ) -> Dict:
        """
        经典采集优化（MC-EI + 随机海选 + L-BFGS）- 4D版本
        
        流程：
        1. 初始化标量化器
        2. 随机海选n_candidates个点
        3. 计算MC-EI（可选：应用LLM权重）
        4. 选择top-5，局部优化
        5. 返回最优点
        
        参数：
            gp_list: GP模型列表 [gp_time, gp_temp, gp_aging]
            weights: Dirichlet权重 (3,)
            database: 评估历史（transformed）
            llm_weight_func: LLM权重函数（可选）
            scalarizer: 外部标量化器（优先使用，空间一致）
        
        返回：
            best_params: {'current1', 'time1', 'current2', 'v_switch'}
        """
        # 使用外部标量化器或回退到内部构建
        if scalarizer is not None:
            self.scalarizer = scalarizer
        else:
            self._update_scalarizer(database)
        
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) == 0:
            # 无有效点，随机采样（4D）
            return {
                'current1': np.random.uniform(*self.param_bounds['current1']),
                'time1': np.random.uniform(*self.param_bounds['time1']),
                'current2': np.random.uniform(*self.param_bounds['current2']),
                'v_switch': np.random.uniform(*self.param_bounds['v_switch'])
            }
        
        # 计算当前最优（用于EI）
        current_best = self._compute_current_best(valid_data, weights)
        
        if self.verbose:
            print(f"    经典优化 - 当前最优标量值: {current_best:.4f}")
        
        # ===== 阶段1: 随机海选 =====
        candidates = []
        ei_values = []
        
        for _ in range(self.n_candidates):
            x = np.array([
                np.random.uniform(*self.param_bounds['current1']),
                np.random.uniform(*self.param_bounds['time1']),
                np.random.uniform(*self.param_bounds['current2']),
                np.random.uniform(*self.param_bounds['v_switch'])
            ])
            
            # 计算MC-EI
            ei = self._compute_mc_ei(
                x, gp_list, weights, current_best, llm_weight_func
            )
            
            candidates.append(x)
            ei_values.append(ei)
        
        # 选择top-5
        ei_values = np.array(ei_values)
        top_indices = np.argsort(ei_values)[-5:][::-1]
        
        if self.verbose:
            print(f"    阶段1完成: {self.n_candidates}个候选点")
            print(f"    Top-5 EI: {[ei_values[i] for i in top_indices]}")
        
        # ===== 阶段2: 局部优化 =====
        best_ei = -np.inf
        best_x = None
        
        for idx in top_indices:
            x0 = candidates[idx]
            
            # 定义目标函数（最大化EI = 最小化负EI）
            def objective(x):
                ei = self._compute_mc_ei(
                    x, gp_list, weights, current_best, llm_weight_func
                )
                return -ei  # 最小化负EI
            
            # L-BFGS-B优化
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=self.bounds_list,
                    options={
                        'maxiter': get_algorithm_param('acquisition', 'local_maxiter', 20),
                        'ftol': get_algorithm_param('acquisition', 'local_ftol', 1e-6)
                    }
                )
                
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            
            except Exception as e:
                if self.verbose:
                    print(f"    [警告] 局部优化失败: {e}")
                continue
        
        # 回退
        if best_x is None:
            best_idx = np.argmax(ei_values)
            best_x = candidates[best_idx]
            best_ei = ei_values[best_idx]
        
        if self.verbose:
            print(f"    阶段2完成: 最优EI={best_ei:.6f}")
        
        return {
            'current1': float(best_x[0]),
            'time1': float(best_x[1]),
            'current2': float(best_x[2]),
            'v_switch': float(best_x[3])
        }
    
    def _compute_current_best(
        self,
        valid_data: List[Dict],
        weights: np.ndarray
    ) -> float:
        """
        计算当前最优标量化值
        
        参数：
            valid_data: 有效数据点
            weights: Dirichlet权重
        
        返回：
            current_best: 当前最优标量值（越小越好）
        """
        scalars = []
        
        for r in valid_data:
            objectives = np.array([r['time'], r['temp'], r['aging']])
            scalar = self.scalarizer.scalarize(objectives, weights, normalize=True)
            scalars.append(scalar)
        
        return np.min(scalars)
    
    def _compute_mc_ei(
        self,
        x: np.ndarray,
        gp_list: List,
        weights: np.ndarray,
        current_best: float,
        llm_weight_func: Optional[Callable] = None
    ) -> float:
        """
        计算蒙特卡洛期望改进（MC-EI）
        
        核心修正：
        - Tchebycheff是非线性的，必须先采样再标量化
        - 不能直接算E[Tcheb(f)]
        
        公式：
            EI(x) = E[max(0, f_best - Tcheb(f_sample))]
            其中 f_sample ~ GP(x)
        
        参数：
            x: 候选点 (3,)
            gp_list: GP列表
            weights: Dirichlet权重
            current_best: 当前最优标量值
            llm_weight_func: LLM权重函数（可选）
        
        返回：
            ei: 期望改进值
        """
        x = x.reshape(1, -1)
        
        # GP预测（均值和标准差）
        mu_list = []
        std_list = []
        
        for gp in gp_list:
            mu, std = gp.predict(x, return_std=True)
            mu_list.append(mu[0])
            std_list.append(std[0])
        
        mu_list = np.array(mu_list)
        std_list = np.array(std_list)
        
        # 避免除零
        std_list = np.maximum(std_list, 1e-10)
        
        # 蒙特卡洛采样
        samples = []
        
        for _ in range(self.n_mc_samples):
            # 从GP后验采样
            sample = np.random.normal(mu_list, std_list)
            samples.append(sample)
        
        samples = np.array(samples)  # (n_mc_samples, 3)
        
        # 对每个样本计算Tchebycheff标量化
        scalars = self.scalarizer.scalarize_batch(
            samples, weights, normalize=True
        )
        
        # 计算改进量
        improvements = np.maximum(0, current_best - scalars)
        
        # 期望改进
        ei = np.mean(improvements)
        
        # 应用LLM权重（如果提供）
        if llm_weight_func is not None:
            try:
                llm_weight = llm_weight_func(x[0])
                ei *= llm_weight
            except Exception as e:
                if self.verbose:
                    print(f"    [警告] LLM权重计算失败: {e}")
        
        return ei


# ============================================================
# 快速测试（4D版本）
# ============================================================
if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
    import asyncio
    
    async def test():
        print("测试 AcquisitionOptimizer（4D）...")
        
        # 创建虚拟GP模型（4D）
        np.random.seed(42)
        X_train = np.random.rand(15, 4)
        X_train[:, 0] = X_train[:, 0] * 3.0 + 3.0     # current1 [3,6]
        X_train[:, 1] = X_train[:, 1] * 38.0 + 2.0    # time1 [2,40]
        X_train[:, 2] = X_train[:, 2] * 3.0 + 1.0     # current2 [1,4]
        X_train[:, 3] = X_train[:, 3] * 0.4 + 3.8     # v_switch [3.8,4.2]
        
        y_train_time = np.random.randint(30, 100, size=15)
        y_train_temp = np.random.uniform(300, 310, size=15)
        y_train_aging = np.random.uniform(0.001, 0.05, size=15)
        
        kernel = C(1.0) * Matern(nu=2.5)
        
        gp_time = GaussianProcessRegressor(kernel=kernel)
        gp_time.fit(X_train, y_train_time)
        
        gp_temp = GaussianProcessRegressor(kernel=kernel)
        gp_temp.fit(X_train, y_train_temp)
        
        gp_aging = GaussianProcessRegressor(kernel=kernel)
        gp_aging.fit(X_train, y_train_aging)
        
        gp_list = [gp_time, gp_temp, gp_aging]
        
        # 创建虚拟database（4D）
        fake_database = []
        for i in range(15):
            fake_database.append({
                'params': {
                    'current1': X_train[i, 0],
                    'time1': X_train[i, 1],
                    'current2': X_train[i, 2],
                    'v_switch': X_train[i, 3]
                },
                'time': y_train_time[i],
                'temp': y_train_temp[i],
                'aging': y_train_aging[i],
                'valid': True
            })
        
        # 初始化优化器
        param_bounds = {
            'current1': (3.0, 6.0),
            'time1': (2.0, 40.0),
            'current2': (1.0, 4.0),
            'v_switch': (3.8, 4.2)
        }
        
        optimizer = AcquisitionOptimizer(
            param_bounds=param_bounds,
            n_candidates=500,  # 减少以加速测试
            n_mc_samples=64,   # 减少以加速测试
            llm_api_key=None,   # 无LLM，将使用经典方法
            verbose=True
        )
        
        # 测试经典优化
        weights = np.array([0.4, 0.35, 0.25])
        
        next_params = await optimizer.optimize(
            gp_list=gp_list,
            weights=weights,
            database=fake_database,
            llm_weight_func=None,
            iteration=10,
            total_iterations=100
        )
        
        print(f"\n下一个查询点:")
        print(f"  current1: {next_params['current1']:.2f}")
        print(f"  time1: {next_params['time1']:.1f}")
        print(f"  current2: {next_params['current2']:.2f}")
        print(f"  v_switch: {next_params['v_switch']:.2f}")
        
        print("\n测试完成！")
    
    asyncio.run(test())
