"""
LLM Coupling Inference - Touchpoint 1a
从 LLM 获取参数耦合矩阵 W (3x3 for 3D parameter space)

统一 3D 参数空间：I1, SOC1, I2
"""

import numpy as np
import json
import re
import asyncio
from typing import Dict, List, Optional
from openai import AsyncOpenAI

# 导入 config
from config import LLM_CONFIG, get_llm_param
# 导入 PSD 保证函数
from models.kernels import ensure_psd


class LLMCouplingInference:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, n_dims: int = 3, verbose: bool = False):
        """初始化 LLM 耦合推理（参数从 config 读取）"""
        # 从 config 读取默认值
        self.api_key = api_key or LLM_CONFIG['api_key']
        self.base_url = base_url or LLM_CONFIG['base_url']
        self.model = model or LLM_CONFIG['model']
        self.n_dims = n_dims  # 3D 决策空间
        self.verbose = verbose

        # 创建客户端
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def infer_coupling_matrix(
        self,
        param_names: List[str],
        current_data: List[Dict] = None
    ) -> np.ndarray:
        """
        推理参数耦合矩阵 W (3x3) - 3D 参数空间
        """
        # 1. 构建基于 LLAMBO 逻辑的 Prompt (无诱导)
        prompt = self._build_inference_prompt(param_names, current_data)

        if self.verbose:
            print(f"  [LLM Coupling] Sending Evidence-Based Prompt (Length: {len(prompt)} chars)...")

        # ========== P1 修复：添加重试机制 ==========
        MAX_RETRIES = get_llm_param('coupling', 'max_retries', 3)
        for retry in range(MAX_RETRIES):
            try:
                # 调用 LLM
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a scientist specializing in Lithium-ion battery electrochemistry. "
                                "Your task is to identify parameter interactions based on provided experimental data and physical laws. "
                                "Output strictly in the requested format wrapped in '##'."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=get_llm_param('coupling', 'temperature', 0.2),
                    max_tokens=get_llm_param('coupling', 'max_tokens', 800),
                )

                content = response.choices[0].message.content

                # 防御性解析
                matrix = self._parse_matrix_from_content(content)
                return self._post_process_matrix(matrix)

            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    backoff_base = get_llm_param('coupling', 'retry_backoff_base', 2)
                    wait_time = backoff_base ** retry  # 指数退避
                    if self.verbose:
                        print(f"  [重试 {retry+1}/{MAX_RETRIES}] 错误：{e}")
                        print(f"  等待 {wait_time}s 后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    # 所有重试失败
                    if self.verbose:
                        print(f"  [LLM Error] 所有重试失败：{e}")
                        print(f"  回退到单位矩阵（无耦合假设）")
                    return np.eye(self.n_dims)
        # ==========================================

    def _build_inference_prompt(self, param_names: List[str], data_samples: List[Dict]) -> str:
        """
        构建 Prompt：包含参数语义定义 + 数据观测 (Few-Shot) + 物理机制提问
        3D 参数空间：I1, SOC1, I2
        """

        # Part A: 参数语义定义 (Semantic Definition) - 3D
        param_desc = """
**Parameter Definitions (3D Decision Space):**
1. `I1` (current1): Constant current during Phase 1 [A]. Range [3.0, 7.0].
   Physical driver: Joule heating power P = I²R(SOC), dominant heat source.

2. `SOC1` (switch_soc): State-of-charge at phase transition point. Range [0.1, 0.7].
   Physical driver: Controls duration of Phase 1. Higher SOC1 extends high-current exposure.
   Accumulated heat Q_total = ∫₀^{t(SOC1)} I1²R(t)dt.

3. `I2` (current2): Constant current during Phase 2 [A]. Range [1.0, 5.0].
   Physical driver: Replenishment current after high-power phase.
   Effectiveness depends on residual temperature from Phase 1.
   Constraint: I2 <= I1.

**Battery Specifications (Chen2020 LG M50):**
- Capacity: 5.0 Ah (1C = 5A)
- Chemistry: NMC cathode + Graphite anode
- Thermal: Heat transfer coefficient 10 W/(m²·K), thermal mass ~50 J/K
- Safety: Max temp 316K (43°C), Max voltage 4.2V
"""

        # Part B: 观测数据序列化 (Data Serialization - LLAMBO Style) - 3D
        data_section = ""
        if data_samples and len(data_samples) > 0:
            data_section = "**Experimental Observations (Pilot Data):**\n"
            valid_samples = [s for s in data_samples if s.get('valid', True)]
            for i, sample in enumerate(valid_samples[:10]):
                p = sample['params']
                time_s = sample.get('time', 0)
                t = sample.get('temp', 0)
                aging = sample.get('aging', 0)

                # 3D 参数序列化
                i1_val = p.get('I1', p.get('current1', 0))
                soc1_val = p.get('SOC1', p.get('time1', p.get('switch_soc', 0)))
                i2_val = p.get('I2', p.get('current2', 0))

                data_section += (
                    f"Trial {i+1}: "
                    f"I1={i1_val:.2f}A, SOC1={soc1_val:.3f}, I2={i2_val:.2f}A. "
                    f"Outcome: Time={time_s:.0f}s, Peak Temp={t:.1f}K, Aging={aging:.5f}.\n"
                )

            # 添加统计摘要
            if len(valid_samples) >= 3:
                data_section += "\n**Correlation Summary (Parameter vs Objectives):**\n"

                # 提取参数矩阵 (3D)
                params_list = []
                for s in valid_samples:
                    p = s['params']
                    params_list.append([
                        p.get('I1', p.get('current1', 0)),
                        p.get('SOC1', p.get('time1', p.get('switch_soc', 0))),
                        p.get('I2', p.get('current2', 0))
                    ])
                params_matrix = np.array(params_list)
                objectives_matrix = np.array([[s.get('time', 0), s.get('temp', 0), s.get('aging', 0)]
                                              for s in valid_samples])

                param_labels = ['I1', 'SOC1', 'I2']
                obj_labels = ['time', 'temp', 'aging']

                for i, param in enumerate(param_labels):
                    for j, obj in enumerate(obj_labels):
                        if params_matrix[:, i].std() > 1e-6 and objectives_matrix[:, j].std() > 1e-6:
                            corr = np.corrcoef(params_matrix[:, i], objectives_matrix[:, j])[0, 1]
                            if abs(corr) > 0.5:
                                strength = "strong" if abs(corr) > 0.7 else "moderate"
                                direction = "positive" if corr > 0 else "negative"
                                data_section += f"  {param} vs {obj}: {corr:.2f} ({strength} {direction})\n"

            data_section += "\n(Analyze the data above: Do you see patterns where specific parameter combinations lead to extreme outcomes?)\n"

        # Part C: 推理任务 (Inference Task) - 3D
        task_section = """
**Physics (Detailed):**
- Heat generation: Q_dot = I²R(SOC) where R increases at high SOC
- Heat accumulation: Delta_T = integral(Q_dot dt) / (m·Cp)
  → High I1 combined with high SOC1 leads to superlinear heating
- Thermal cascade: Phase1 heat affects Phase2 starting temperature
  → I1-I2 interaction through thermal inertia (~30s time constant)
- SOC1 controls Phase 1 duration: higher SOC1 = longer exposure to high current

**Task:**
Construct a 3x3 Interaction Matrix (Coupling Matrix) W where w_ij represents the interaction strength between parameter i and j.

**Reasoning Steps (Chain of Thought):**
1. **I1 ↔ SOC1 (Current × Duration)**:
   Heat accumulation is the integral of power over time: Q = I1² × R × t(SOC1).
   I1 controls Power, SOC1 controls Duration → their joint effect on temperature is multiplicative.

2. **I1 ↔ I2 (Thermal Cascade)**:
   Phase 1 residual heat affects Phase 2 starting temperature.
   High I1 creates elevated baseline for I2's heating effects.
   Coupling occurs through thermal inertia (time constant ~30s).

3. **SOC1 ↔ I2 (Duration × Phase2 Current)**:
   Higher SOC1 extends Phase 1, increasing accumulated heat.
   This affects I2's thermal contribution in Phase 2.

**Matrix Construction Rules:**
- Diagonal elements w_ii = 1.0 (self-coupling is maximum)
- Off-diagonal values in [0, 1] (symmetric)
- Higher values indicate stronger interaction between parameters

**Output Format:**
You must wrap the final matrix in '##' delimiters. Use this format:

##
[[1.0, <value>, <value>],
 [<value>, 1.0, <value>],
 [<value>, <value>, 1.0]]
##

where <value> represents the coupling strength you determine from the data and physical reasoning.
"""
        return param_desc + "\n" + data_section + "\n" + task_section

    def _parse_matrix_from_content(self, content: str) -> np.ndarray:
        """
        LLAMBO 风格的防御性解析
        """
        try:
            # 1. 尝试提取 ## 之间的内容 (最稳健)
            if "##" in content:
                json_str = content.split("##")[1].strip()
            else:
                # 2. 回退：寻找 JSON 数组结构
                match = re.search(r"\[\s*\[.*?\]\s*\]", content, re.DOTALL)
                if match:
                    json_str = match.group()
                else:
                    raise ValueError("No matrix pattern found in response")

            matrix = np.array(json.loads(json_str))
            return matrix
        except Exception as e:
            raise ValueError(f"Parsing failed: {e}")

    def _post_process_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        后处理：确保矩阵的数学性质 (对称，对角线为 1, 归一化，PSD)
        """
        expected_dim = self.n_dims

        # 尺寸检查
        if matrix.shape != (expected_dim, expected_dim):
            # 尝试截断或填充
            new_mat = np.eye(expected_dim)
            min_dim = min(matrix.shape[0], expected_dim)
            new_mat[:min_dim, :min_dim] = matrix[:min_dim, :min_dim]
            matrix = new_mat

        # 归一化 [0, 1]
        matrix = np.clip(matrix, 0.0, 1.0)

        # 强制对称
        matrix = (matrix + matrix.T) / 2.0

        # 强制对角线为 1
        np.fill_diagonal(matrix, 1.0)

        # PSD 修正
        eps_psd = get_llm_param('composite_kernel', 'eps_psd', 1e-5) or 1e-5
        matrix = ensure_psd(matrix, eps=eps_psd)

        # 再次强制对角线为 1（PSD 修正可能微调对角线）
        np.fill_diagonal(matrix, 1.0)

        return matrix

    async def get_objective_coupling_matrices(self) -> Dict[str, np.ndarray]:
        """
        Touchpoint 1a — 一次性获取三个目标各自的耦合矩阵。

        按 FrameWork §4.1 (Partial Context) 调用 LLM，分别推断：
          W_time  : 面向充电时间目标的耦合矩阵
          W_temp  : 面向峰值温度目标的耦合矩阵
          W_aging : 面向老化程度目标的耦合矩阵

        返回：
            {"W_time": np.ndarray(3,3), "W_temp": ..., "W_aging": ...}
        """
        param_names = ['I1', 'SOC1', 'I2']
        results = {}

        objective_prompts = {
            "W_time": "coupling for minimizing total charging time",
            "W_temp": "coupling for minimizing peak temperature",
            "W_aging": "coupling for minimizing aging (capacity fade via SEI growth)",
        }

        for matrix_key, objective_desc in objective_prompts.items():
            if self.verbose:
                print(f"  [1a] 推断 {matrix_key} ({objective_desc[:30]}...)")
            try:
                # 使用专用 prompt 进行推断
                W = await self._infer_single_objective_matrix(
                    param_names, objective_desc
                )
                results[matrix_key] = W
            except Exception as e:
                if self.verbose:
                    print(f"  [1a] {matrix_key} 失败：{e}，使用单位矩阵")
                results[matrix_key] = np.eye(len(param_names))

        return results

    async def _infer_single_objective_matrix(
        self,
        param_names: List[str],
        objective_description: str
    ) -> np.ndarray:
        """
        针对单个目标推断 3×3 耦合矩阵（FrameWork §4.1 Partial Context）
        """
        n = len(param_names)
        prompt = f"""As an electrochemistry expert, analyze the physical coupling
between parameters in a two-stage CC lithium-ion battery fast
charging protocol theta = (I1, SOC1, I2) for an NMC811/graphite
cell (5Ah). The battery is charged from SOC 0.1 to 0.8.

Provide a {n}x{n} symmetric coupling matrix W for the objective:
  {objective_description}

Each entry wij in [0,1] quantifies how strongly parameter i and
parameter j interact when optimizing for that specific objective.
Parameters: I1 (Phase-1 current [A]), SOC1 (switch SOC), I2 (Phase-2 current [A]).

Output JSON only:
{{"W": [[w11,w12,w13],[w12,w22,w23],[w13,w23,w33]],
  "rationale": {{"I1_SOC1": "...", "SOC1_I2": "...", "I1_I2": "..."}}}}"""

        MAX_RETRIES = get_llm_param('coupling', 'max_retries', 3)
        for retry in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a battery electrochemistry expert. "
                            "Output strictly valid JSON."
                        )},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=get_llm_param('coupling', 'temperature', 0.2),
                    max_tokens=get_llm_param('coupling', 'max_tokens', 800),
                )
                content = response.choices[0].message.content
                parsed = self._parse_single_matrix(content, n)
                return self._post_process_3x3(parsed, n)
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** retry)
                else:
                    raise e

        return np.eye(n)

    def _parse_single_matrix(self, content: str, n: int) -> np.ndarray:
        """从 LLM 输出解析单个 n×n 矩阵"""
        try:
            # 尝试直接解析 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                W = data.get("W", None)
                if W is not None:
                    return np.array(W, dtype=float)
        except Exception:
            pass
        # fallback
        return np.eye(n)

    def _post_process_3x3(self, matrix: np.ndarray, n: int) -> np.ndarray:
        """后处理：裁剪、对称、PSD 保证"""
        if matrix.shape != (n, n):
            new_mat = np.eye(n)
            m = min(matrix.shape[0], n)
            new_mat[:m, :m] = matrix[:m, :m]
            matrix = new_mat
        matrix = np.clip(matrix, 0.0, 1.0)
        matrix = (matrix + matrix.T) / 2.0
        np.fill_diagonal(matrix, 1.0)
        eps = get_llm_param('composite_kernel', 'eps_psd', 1e-5) or 1e-5
        matrix = ensure_psd(matrix, eps=eps)
        np.fill_diagonal(matrix, 1.0)
        return matrix


if __name__ == "__main__":
    import asyncio
    import os

    async def test():
        print("测试 LLAMBO 风格推理 (3D 参数空间)...")
        api_key = "sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7"
        if not api_key:
            print("请设置 LLM_API_KEY")
            return

        inferencer = LLMCouplingInference(api_key, verbose=True, n_dims=3)

        # 模拟 Pilot GP 数据 (3D 参数空间)
        mock_data = [
            {'params': {'I1': 6.0, 'SOC1': 0.5, 'I2': 2.0}, 'time': 600, 'temp': 305.0, 'aging': 0.005},
            {'params': {'I1': 3.0, 'SOC1': 0.4, 'I2': 2.0}, 'time': 2400, 'temp': 302.0, 'aging': 0.002},
            {'params': {'I1': 6.0, 'SOC1': 0.6, 'I2': 2.0}, 'time': 2100, 'temp': 320.0, 'aging': 0.030},
        ]

        W = await inferencer.infer_coupling_matrix(
            param_names=['I1', 'SOC1', 'I2'],
            current_data=mock_data
        )

        print("\nInferred Matrix (3x3):")
        print(W)

    asyncio.run(test())
