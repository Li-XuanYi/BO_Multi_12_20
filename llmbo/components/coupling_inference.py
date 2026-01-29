"""
LLM耦合矩阵推理模块 (ICLR 2024 LLAMBO 风格重构版)

核心改进：
1. 移除诱导性Prompt：不再告诉LLM"谁和谁耦合"，而是让它基于物理定义分析。
2. 语义序列化：将参数数值翻译为物理场景描述 (e.g., "High Power Phase").
3. 基于证据推理：利用 Pilot Data 让 LLM 自己从数据中发现协变规律。
4. 防御性解析：使用 ## 定界符确保 JSON 解析 100% 成功。
"""

import numpy as np
import json
import re
import asyncio
from typing import Dict, List, Optional
from openai import AsyncOpenAI

# 导入config
from config import LLM_CONFIG, get_llm_param

class LLMCouplingInference:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, verbose: bool = False):
        """初始化LLM耦合推理（参数从config读取）"""
        # 从config读取默认值
        self.api_key = api_key or LLM_CONFIG['api_key']
        self.base_url = base_url or LLM_CONFIG['base_url']
        self.model = model or LLM_CONFIG['model']
        self.verbose = verbose
        
        # 创建客户端
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def infer_coupling_matrix(
        self, 
        param_names: List[str], 
        current_data: List[Dict] = None
    ) -> np.ndarray:
        """
        推理参数耦合矩阵 W (3x3)
        """
        # 1. 构建基于 LLAMBO 逻辑的 Prompt (无诱导)
        prompt = self._build_inference_prompt(param_names, current_data)
        
        if self.verbose:
            print(f"  [LLM Coupling] Sending Evidence-Based Prompt (Length: {len(prompt)} chars)...")

        # ========== P1修复：添加重试机制 ==========
        MAX_RETRIES = get_llm_param('coupling', 'max_retries', 3)
        for retry in range(MAX_RETRIES):
            try:
                # 调用LLM
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
                        print(f"  [重试 {retry+1}/{MAX_RETRIES}] 错误: {e}")
                        print(f"  等待 {wait_time}s 后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    # 所有重试失败
                    if self.verbose:
                        print(f"  [LLM Error] 所有重试失败: {e}")
                        print(f"  回退到单位矩阵（无耦合假设）")
                    return np.eye(len(param_names))
        # ==========================================

    def _build_inference_prompt(self, param_names: List[str], data_samples: List[Dict]) -> str:
        """
        构建 Prompt：包含参数语义定义 + 数据观测 (Few-Shot) + 物理机制提问
        """
        
        # Part A: 参数语义定义 (Semantic Definition)
        # 仅仅定义物理意义，绝不暗示耦合关系
        param_desc = """
**Parameter Definitions:**
1. `current1` (I1): Constant current applied during the 1st phase. (Physical driver: Joule Heating Power $P = I^2 R$).
2. `switch_soc` (SOC_sw): The SOC threshold to stop Phase 1. (Physical driver: Duration of Phase 1).
3. `current2` (I2): Constant current for the 2nd phase (Replenishment phase).
"""

        # Part B: 观测数据序列化 (Data Serialization - LLAMBO Style)
        # 将 Pilot GP 阶段产生的少量数据转化为“实验记录”，让 LLM 自己看数据说话
        data_section = ""
        if data_samples and len(data_samples) > 0:
            data_section = "**Experimental Observations (Pilot Data):**\n"
            # 取前 10 个样本 (← P1修复: 5→10)
            for i, sample in enumerate(data_samples[:10]):
                p = sample['params']
                # 尝试获取目标值，处理可能缺失的情况
                t = sample.get('temp', 0)
                aging = sample.get('aging', 0)
                
                # 序列化：转化为自然语言描述
                data_section += (
                    f"Trial {i+1}: "
                    f"Applied I1 = {p['current1']:.1f}A until SOC reached {p['switch_soc']:.2f}. "
                    f"Outcome: Peak Temp = {t:.1f}K, Aging Impact = {aging:.5f}.\n"
                )
            
            data_section += "\n(Analyze the data above: Do you see patterns where specific parameter combinations lead to extreme outcomes?)\n"

        # Part C: 推理任务 (Inference Task)
        # 核心：CoT 引导，要求先分析机制，再量化矩阵
        task_section = f"""
**Task:**
**Battery Specifications (Chen2020 LG M50):**
- Capacity: 5.0 Ah (1C = 5A)
- Chemistry: NMC cathode + Graphite anode
- Thermal: Heat transfer coefficient 10 W/(m²·K), thermal mass ~50 J/K
- Safety: Max temp 313K (40°C), Max voltage 4.4V

**Physics (Detailed):**
- Heat generation: Q̇ = I²R(SOC) where R increases at high SOC
- Heat accumulation: ΔT = ∫Q̇ dt / (m·Cp) 
  → High I1 + Long phase1 (high switch_soc) = superlinear heating
- Thermal cascade: Phase1 heat → Phase2 starting temp ↑
  → I1-I2 coupling through thermal inertia (~30s time constant)
Construct a 3x3 Interaction Matrix (Coupling Matrix) $W$ where $w_{{ij}}$ represents the interaction strength between parameter $i$ and $j$.

**Reasoning Steps (Chain of Thought):**
1. **Analyze I1 <-> switch_soc**: 
   - From Physics: Heat is the integral of power over time. `current1` controls Power, `switch_soc` controls Time.
   - From Data: In the trials above, did high I1 *combined* with high switch_soc cause a non-linear spike in Temp?
   - If yes, they are coupled (Value 0.6-0.9). If independent, low value.
   
2. **Analyze I1 <-> I2**:
   - Does the value of `current1` physically alter how `current2` behaves?
   - Or are they separate temporal phases? (Likely lower coupling).

3. **Construct Matrix**:
   - Diagonal ($w_{{ii}}$) must be 1.0.
   - Matrix must be symmetric.
   - Values range [0, 1]. 0 = Independent, 1 = Strongly Coupled.

**Output Format:**
You must wrap the final JSON list of lists in '##' delimiters.
Example:
##
[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
##
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
                # 2. 回退：寻找 JSON 数组结构（P1修复：非贪婪正则）
                match = re.search(r"\[\s*\[.*?\]\s*\]", content, re.DOTALL)  # ← 加?变非贪婪
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
        后处理：确保矩阵的数学性质 (对称, 对角线为1, 归一化)
        """
        # 尺寸检查
        if matrix.shape != (3, 3):
            # 尝试截断或填充
            new_mat = np.eye(3)
            min_dim = min(matrix.shape[0], 3)
            new_mat[:min_dim, :min_dim] = matrix[:min_dim, :min_dim]
            matrix = new_mat

        # 归一化 [0, 1]
        matrix = np.clip(matrix, 0.0, 1.0)
        
        # 强制对称
        matrix = (matrix + matrix.T) / 2.0
        
        # 强制对角线为 1
        np.fill_diagonal(matrix, 1.0)
        
        return matrix

# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test():
        print("测试 LLAMBO 风格推理...")
        api_key = "sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7"
        if not api_key: 
            print("请设置 LLM_API_KEY")
            return

        inferencer = LLMCouplingInference(api_key, verbose=True)
        
        # 模拟几条 Pilot GP 跑出来的真实数据
        # 这种数据体现了：单有大电流不够，必须配合高SOC才会过热（体现耦合）
        mock_data = [
            {'params': {'current1': 6.0, 'switch_soc': 0.3, 'current2': 2.0}, 'temp': 305.0, 'aging': 0.005}, # I1大但时间短 -> 温升尚可
            {'params': {'current1': 3.0, 'switch_soc': 0.7, 'current2': 2.0}, 'temp': 302.0, 'aging': 0.002}, # I1小但时间长 -> 温升低
            {'params': {'current1': 6.0, 'switch_soc': 0.7, 'current2': 2.0}, 'temp': 320.0, 'aging': 0.030}, # I1大且时间长 -> 温升爆炸 (耦合证据!)
        ]
        
        W = await inferencer.infer_coupling_matrix(
            param_names=['current1', 'switch_soc', 'current2'],
            current_data=mock_data
        )
        
        print("\nInferred Matrix:")
        print(W)

    asyncio.run(test())