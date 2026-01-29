"""
LLM增强权重模块（LLAMBO风格）
实现论文公式(9)：W_LLM = ∏ [1/√(2πσ²) exp(-(θ-μ)²/(2σ²))]

核心功能：
1. 从LLM推理获得焦点中心μ（promising region）
2. 从GP不确定性推导焦点宽度σ
3. 计算多元高斯密度权重
"""

import numpy as np
import asyncio
import json
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI

# 导入config
from config import LLM_CONFIG, get_llm_param


class LLAMBOWeighting:
    """
    LLAMBO风格的LLM权重函数
    
    公式：W_LLM(θ) = ∏_{j=1}^q [N(θ_j | μ_j, σ_j²)]
    
    其中：
    - μ_j: LLM推理的焦点中心（high-potential region）
    - σ_j: 从GP不确定性推导的焦点宽度
    - N: 高斯概率密度函数
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, tuple],
        llm_api_key: str = None,
        base_url: str = None,
        model: str = None,
        sigma_scale: float = None,
        verbose: bool = False
    ):
        """
        初始化LLM权重模块（参数从config读取）
        
        参数：
            param_bounds: 参数边界
            llm_api_key: LLM API密钥
            base_url: API基础URL
            model: LLM模型
            sigma_scale: σ缩放因子（控制焦点宽度）
            verbose: 详细输出
        """
        self.param_bounds = param_bounds
        
        # 从config读取默认值
        llm_api_key = llm_api_key or LLM_CONFIG['api_key']
        base_url = base_url or LLM_CONFIG['base_url']
        model = model or LLM_CONFIG['model']
        self.sigma_scale = sigma_scale if sigma_scale is not None else get_llm_param('weighting', 'sigma_scale', 0.15)
        self.verbose = verbose
        
        # LLM客户端
        if llm_api_key is not None:
            self.client = AsyncOpenAI(base_url=base_url, api_key=llm_api_key)
            self.model = model
        else:
            self.client = None
        
        # 焦点参数（初始化为参数空间中心）
        self.mu_focus = np.array([
            (param_bounds['current1'][0] + param_bounds['current1'][1]) / 2,
            (param_bounds['switch_soc'][0] + param_bounds['switch_soc'][1]) / 2,
            (param_bounds['current2'][0] + param_bounds['current2'][1]) / 2
        ])
        
        # 焦点宽度（初始化为参数范围的sigma_scale）
        param_ranges = np.array([
            param_bounds['current1'][1] - param_bounds['current1'][0],
            param_bounds['switch_soc'][1] - param_bounds['switch_soc'][0],
            param_bounds['current2'][1] - param_bounds['current2'][0]
        ])
        self.sigma_focus = self.sigma_scale * param_ranges  # ← 使用self.sigma_scale
        
        # 记录更新历史
        self.update_history = []
    
    async def update_focus_from_llm(
        self,
        database: List[Dict],
        gp_list: List = None,
        iteration: int = 0
    ):
        """
        从LLM推理更新焦点中心μ
        
        策略：
        1. 提取当前Pareto前沿
        2. 向LLM询问：哪个参数区域最有潜力？
        3. 更新μ_focus
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
            
            # 提取焦点中心
            new_mu = np.array([
                float(result.get('focus_current1', self.mu_focus[0])),
                float(result.get('focus_switch_soc', self.mu_focus[1])),
                float(result.get('focus_current2', self.mu_focus[2]))
            ])
            
            # 边界检查
            new_mu[0] = np.clip(new_mu[0], *self.param_bounds['current1'])
            new_mu[1] = np.clip(new_mu[1], *self.param_bounds['switch_soc'])
            new_mu[2] = np.clip(new_mu[2], *self.param_bounds['current2'])
            
            # 更新焦点
            self.mu_focus = new_mu
            
            if self.verbose:
                print(f"    焦点已更新: μ=[{self.mu_focus[0]:.2f}, {self.mu_focus[1]:.3f}, {self.mu_focus[2]:.2f}]")
            
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
        计算LLAMBO公式(9)的权重
        
        W_LLM(θ) = ∏_{j=1}^3 [N(θ_j | μ_j, σ_j²)]
        
        其中 N(θ_j | μ_j, σ_j²) = 1/√(2πσ_j²) exp(-(θ_j - μ_j)²/(2σ_j²))
        
        参数：
            x: 候选点 (3,) [current1, switch_soc, current2]
        
        返回：
            weight: [0, 1]，归一化权重
        """
        # 确保x是1D数组
        if x.ndim > 1:
            x = x.flatten()
        
        # 计算多元高斯密度（连乘）
        weight = 1.0
        
        for j in range(3):
            # 高斯密度：N(θ_j | μ_j, σ_j²)
            gaussian_term = (1.0 / np.sqrt(2 * np.pi * self.sigma_focus[j]**2)) * \
                            np.exp(-(x[j] - self.mu_focus[j])**2 / (2 * self.sigma_focus[j]**2))
            
            weight *= gaussian_term
        
        # 归一化到[0, 1]
        # 理论最大值在μ处
        max_weight = 1.0
        for j in range(3):
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
        """构建LLM Prompt"""
        # 序列化Pareto点
        points_str = ""
        for i, point in enumerate(pareto_front[:10], 1):  # 最多10个点
            p = point['params']
            points_str += (
                f"Point {i}: I1={p['current1']:.2f}A, SOC_sw={p['switch_soc']:.3f}, I2={p['current2']:.2f}A "
                f"→ Time={point['time']}, Temp={point['temp']:.1f}K, Aging={point['aging']:.5f}\n"
            )
        
        prompt = f"""
Given {len(pareto_front)} Pareto-optimal battery charging strategies:

{points_str}

**Physics Context:**
- High current (I1) → Fast charging but more heat
- High SOC cutoff → Longer Phase 1, more heat accumulation
- Low I2 → Slower Phase 2, less stress

**Trade-offs:**
- Time vs Temperature: Higher current reduces time but increases temperature
- Temperature vs Aging: Higher temperature accelerates aging
- I1 and SOC_sw are coupled: High I1 + High SOC = thermal runaway risk

**Task:**
Based on the Pareto points above, suggest the most promising parameter region for the NEXT exploration.

Output JSON:
{{
  "focus_current1": <value in [3.0, 6.0]>,
  "focus_switch_soc": <value in [0.3, 0.7]>,
  "focus_current2": <value in [1.0, 4.0]>,
  "reasoning": "<brief explanation>"
}}
"""
        return prompt
    
    def _update_sigma_from_gp(self, gp_list: List):
        """从GP预测不确定性更新σ"""
        try:
            # 在μ处预测
            X_mu = self.mu_focus.reshape(1, -1)
            
            std_list = []
            for gp in gp_list:
                _, std = gp.predict(X_mu, return_std=True)
                std_list.append(std[0])
            
            # 取平均不确定性
            avg_std = np.mean(std_list)
            
            # 更新σ（GP不确定性 + 基础宽度）
            param_ranges = np.array([
                self.param_bounds['current1'][1] - self.param_bounds['current1'][0],
                self.param_bounds['switch_soc'][1] - self.param_bounds['switch_soc'][0],
                self.param_bounds['current2'][1] - self.param_bounds['current2'][0]
            ])
            
            # σ = 0.1 × 参数范围 + 0.05 × GP不确定性
            self.sigma_focus = 0.1 * param_ranges + 0.05 * avg_std * param_ranges
            
            if self.verbose:
                print(f"    σ已更新: [{self.sigma_focus[0]:.3f}, {self.sigma_focus[1]:.4f}, {self.sigma_focus[2]:.3f}]")
        
        except Exception as e:
            if self.verbose:
                print(f"    [警告] σ更新失败: {e}")


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("测试 LLAMBOWeighting...")
        
        # 参数边界
        param_bounds = {
            'current1': (3.0, 6.0),
            'switch_soc': (0.3, 0.7),
            'current2': (1.0, 4.0)
        }
        
        # 初始化（无LLM）
        weighting = LLAMBOWeighting(
            param_bounds=param_bounds,
            llm_api_key=None,
            verbose=True
        )
        
        print(f"\n初始焦点: μ={weighting.mu_focus}")
        print(f"初始宽度: σ={weighting.sigma_focus}")
        
        # 测试权重计算
        test_points = [
            np.array([4.5, 0.5, 2.5]),  # 中心点
            np.array([6.0, 0.7, 1.0]),  # 边界点
            np.array([3.0, 0.3, 4.0])   # 另一边界
        ]
        
        print("\n测试权重计算:")
        for i, x in enumerate(test_points, 1):
            weight = weighting.compute_weight(x)
            print(f"  点{i} {x} → 权重={weight:.4f}")
        
        print("\n测试完成！")
    
    asyncio.run(test())
