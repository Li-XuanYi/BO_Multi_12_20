
import numpy as np
import json
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional
from openai import AsyncOpenAI

# 导入config
from config import LLM_CONFIG, get_llm_param

# 导入辅助函数
from components.warmstart_utils import (
    validate_strategy,
    clean_strategy,
    generate_random_strategy,
    load_template,
    inject_template_values
)


class LLMWarmStart:
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        context_level: str = None,
        verbose: bool = False
    ):
        """
        初始化LLM热启动（参数从config读取）
        
        参数：
            api_key: LLM API密钥（从config或环境变量读取）
            base_url: API基础URL
            model: LLM模型名称
            context_level: 上下文级别 ('full', 'partial', 'none')
            verbose: 详细输出
        """
        # 从config读取默认值
        self.api_key = api_key or LLM_CONFIG['api_key']
        self.base_url = base_url or LLM_CONFIG['base_url']
        self.model = model or LLM_CONFIG['model']
        self.context_level = context_level if context_level is not None else get_llm_param('warmstart', 'context_level', 'full')
        self.verbose = verbose
        
        # 创建客户端
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # 加载对应的模板（带 fallback）
        try:
            self.template = load_template(self.context_level)
            if self.verbose:
                print(f"  [LLM WarmStart] 加载模板: {self.context_level}")
        except FileNotFoundError:
            # 模板文件缺失：使用内联默认模板
            if self.verbose:
                print(f"  [LLM WarmStart] 模板文件缺失，使用内联默认模板")
            self.template = self._default_template()
        except Exception as e:
            print(f"  [LLM WarmStart] 模板加载失败: {e}，使用内联默认模板")
            self.template = self._default_template()
    
    @staticmethod
    def _default_template() -> str:
        """内联默认模板（当模板文件缺失时使用）"""
        return """You are optimizing a lithium-ion battery CC-CC-CV fast-charging protocol.

Decision variables:
- current1 (I1): Phase 1 current [3.0-6.0] A
- time1 (T1): Phase 1 duration [120-3600] s
- current2 (I2): Phase 2 current [1.0-4.5] A
- v_switch (V_sw): CC-to-CV voltage [3.8-4.2] V

Objectives to MINIMIZE: charging time (s), peak temperature (K), capacity fade (aging %).

Physical constraints:
- High current → fast charging but more heat (Joule heating ∝ I²R)
- High temperature → accelerated SEI growth → more aging
- Temperature must stay below 318.15K (45°C)

Generate [N_STRATEGIES] diverse charging strategies as JSON.
IMPORTANT: Each strategy must explore a DIFFERENT region of the parameter space. 
Include a mix of: conservative (low I1 ~3-4A), moderate (I1 ~4-5A), and aggressive (I1 ~5-6A) approaches.
Avoid generating strategies with I1 > 6A as they risk thermal violation (>318K).
{
  "strategies": [
    {"current1": <value>, "time1": <value>, "current2": <value>, "v_switch": <value>, "rationale": "<brief>"},
    ...
  ]
}
"""
    
    async def generate_strategies(
        self,
        n_strategies: int = 5,
        param_bounds: Dict = None,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        生成初始充电策略（真正的并行多次生成版本）
        
        参数：
            n_strategies: 需要生成的策略数量
            param_bounds: 参数边界
            save_path: 保存路径（可选）
        
        返回：
            strategies: [{'current1': ..., 'switch_soc': ..., 'current2': ...}, ...]
        """
        # 默认参数边界（4D，time1 单位为秒）
        if param_bounds is None:
            param_bounds = {
                'current1': (3.0, 6.0),
                'time1': (120.0, 3600.0),
                'current2': (1.0, 4.5),
                'v_switch': (3.8, 4.2)
            }
        
        # 构建prompt
        prompt = inject_template_values(self.template, n_strategies)
        
        if self.verbose:
            print(f"  [LLM WarmStart] LLAMBO式单次生成 (context={self.context_level})...")
        
        # ========== LLAMBO式直接采信：单次调用 + 重试 ==========
        MAX_RETRIES = 3
        all_candidates = []
        
        for attempt in range(MAX_RETRIES):
            try:
                candidates = await self._single_generation(
                    prompt, temperature=0.7, param_bounds=param_bounds
                )
                if candidates:
                    all_candidates.extend(candidates)
                    if self.verbose:
                        print(f"    第{attempt+1}次调用: 获得{len(candidates)}个有效候选")
                
                # 够了就停
                if len(all_candidates) >= n_strategies:
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"    第{attempt+1}次调用失败: {e}")
        
        # 取前n_strategies个（保持LLM原始排序，不做MaxMin）
        final_strategies = all_candidates[:n_strategies]
        
        # 不足则随机补齐
        while len(final_strategies) < n_strategies:
            final_strategies.append(generate_random_strategy(param_bounds, seed=len(final_strategies)))
            if self.verbose:
                print(f"    补充随机策略: {len(final_strategies)}/{n_strategies}")
        
        # 记录多样性（仅日志，不用于选择）
        from components.warmstart_utils import compute_generalized_variance
        diversity_score = compute_generalized_variance(final_strategies)
        self.last_diversity_score = diversity_score
        
        if self.verbose:
            print(f"  [LLM WarmStart] 直接采信LLM生成的{len(final_strategies)}个策略")
            print(f"  [LLM WarmStart] Generalized Variance (多样性): {diversity_score:.6f}")
        # ==========================================
        
        # 保存（如果指定）
        if save_path is not None:
            self._save_strategies(final_strategies, save_path, param_bounds, diversity_score)
        
        return final_strategies
    
    async def _single_generation(
        self,
        prompt: str,
        temperature: float,
        param_bounds: Dict
    ) -> List[Dict]:
        """
        单次LLM调用（用于并行生成）
        
        参数：
            prompt: LLM提示词
            temperature: 温度参数
            param_bounds: 参数边界
        
        返回：
            有效策略列表（失败时返回空列表）
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in battery electrochemistry and fast-charging optimization. "
                                   "Always respond with valid JSON format wrapped in ## JSON ## markers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=get_llm_param('warmstart', 'max_tokens', 2500),
                response_format={"type": "json_object"}
            )
            
            # 解析响应
            content = response.choices[0].message.content
            strategies = self._parse_json_defensive(content)
            
            # 只保留在边界内的策略
            valid_strategies = [
                s for s in strategies
                if self._is_within_bounds(s, param_bounds)
            ]
            
            return valid_strategies
        
        except Exception as e:
            # 失败时返回空列表，由gather处理
            return []
    
    def _parse_json_defensive(self, content: str) -> List[Dict]:
        """
        防御性JSON解析（优先直接解析，## JSON ##作为fallback）
        
        由于使用了response_format={"type": "json_object"}，LLM输出必然是合法JSON，
        因此优先直接解析。只在失败时尝试## JSON ##标记提取。
        
        参数：
            content: LLM返回的原始字符串
        
        返回：
            策略列表
        """
        import json
        
        # 优先：直接解析（适配OpenAI JSON mode）
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Fallback：尝试提取 ## JSON ## 标记内容（兼容非OpenAI模型）
            if '## JSON ##' in content:
                try:
                    json_parts = content.split('## JSON ##')
                    json_str = json_parts[1].strip() if len(json_parts) >= 2 else json_parts[-1].strip()
                    result = json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    raise ValueError(f"JSON解析失败: {content[:200]}")
            else:
                raise ValueError(f"JSON解析失败且无## JSON ##标记: {content[:200]}")
        
        # 提取策略列表
        if 'strategies' in result:
            return result['strategies']
        elif isinstance(result, list):
            return result
        else:
            return [result]
    
    def _is_within_bounds(self, strategy: Dict, param_bounds: Dict) -> bool:
        """检查策略是否在参数边界内（动态检查所有键）"""
        try:
            return all(
                param_bounds[k][0] <= float(strategy[k]) <= param_bounds[k][1]
                for k in param_bounds.keys()
            )
        except (KeyError, ValueError, TypeError):
            return False
    
    def _save_strategies(
        self,
        strategies: List[Dict],
        save_path: str,
        param_bounds: Dict,
        diversity_score: float = None
    ):
        """
        保存生成的策略到JSON
        
        参数：
            strategies: 策略列表
            save_path: 保存路径
            param_bounds: 参数边界
            diversity_score: Generalized Variance多样性得分（可选）
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 构建保存内容
        save_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'context_level': self.context_level,
                'n_strategies': len(strategies),
                'param_bounds': param_bounds,
                'generalized_variance': diversity_score
            },
            'strategies': strategies
        }
        
        # 保存
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"  [LLM WarmStart] 策略已保存至: {save_path}")
    
    @staticmethod
    def load_strategies(load_path: str) -> List[Dict]:
        """
        从JSON加载策略
        
        参数：
            load_path: 文件路径
        
        返回：
            策略列表
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data['strategies']


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test():
        print("测试 LLMWarmStart (优化版)...")
        
        # 获取API key
        api_key = "sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7"
        
        if api_key is None:
            print("  警告: 未设置 LLM_API_KEY，跳过LLM测试")
            print("\n  测试回退机制（随机策略）:")
            
            # 测试随机回退（4D）
            param_bounds = {
                'current1': (3.0, 6.0),
                'time1': (2.0, 40.0),
                'current2': (1.0, 4.0),
                'v_switch': (3.8, 4.2)
            }
            
            for i in range(3):
                strategy = generate_random_strategy(param_bounds, seed=i)
                print(f"    策略{i+1}: I1={strategy['current1']:.1f}A, "
                      f"T1={strategy['time1']:.1f}min, "
                      f"I2={strategy['current2']:.1f}A, "
                      f"V_sw={strategy['v_switch']:.2f}V")
            
            return
        
        # 参数边界（4D）
        param_bounds = {
            'current1': (3.0, 6.0),
            'time1': (2.0, 40.0),
            'current2': (1.0, 4.0),
            'v_switch': (3.8, 4.2)
        }
        
        # 测试三种上下文级别
        for context_level in ['full', 'partial', 'none']:
            print(f"\n{'='*60}")
            print(f"测试上下文级别: {context_level}")
            print('='*60)
            
            # 初始化
            warmstart = LLMWarmStart(
                api_key=api_key,
                model='gpt-4o',
                context_level=context_level,
                verbose=True
            )
            
            # 生成策略
            strategies = await warmstart.generate_strategies(
                n_strategies=5,
                param_bounds=param_bounds,
                save_path=f'./llmbo/components/test_json/test_strategies_{context_level}.json'
            )
            
            # 显示结果（4D）
            print(f"\n生成的策略 ({context_level}):")
            for i, s in enumerate(strategies, 1):
                rationale = s.get('rationale', 'N/A')
                print(f"  策略{i}: I1={s['current1']:.1f}A, "
                      f"T1={s['time1']:.1f}min, "
                      f"I2={s['current2']:.1f}A, "
                      f"V_sw={s['v_switch']:.2f}V")
                if rationale != 'N/A':
                    print(f"          {rationale}")
        
        print("\n测试完成！")
    
    # 运行测试
    asyncio.run(test())