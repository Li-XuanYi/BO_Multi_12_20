"""
LLM热启动模块（优化版）
生成物理合理的初始充电策略，加速Pareto前沿的发现

核心改进：
1. 三级模板系统（Full/Partial/None上下文，支持消融实验）
2. 精确数值格式化（自动匹配边界精度）
3. 指数退避重试机制（提升稳定性）
4. 配置保存/加载（可复现性）
5. 修复switch_soc类型错误（float而非int）
"""

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
    """
    LLM热启动生成器
    
    功能：
    1. 生成物理合理的初始充电策略
    2. 快速覆盖Pareto前沿的不同区域
    3. 避免随机采样导致的低质量初始点
    4. 支持多级上下文（消融实验）
    """
    
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
        
        # 加载对应的模板
        try:
            self.template = load_template(self.context_level)  # ← 使用self.context_level
            if self.verbose:
                print(f"  [LLM WarmStart] 加载模板: {self.context_level}")
        except Exception as e:
            print(f"  [LLM WarmStart] 模板加载失败: {e}")
            raise
    
    async def generate_strategies(
        self,
        n_strategies: int = 5,
        param_bounds: Dict = None,
        max_retries: int = 3,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        生成初始充电策略
        
        参数：
            n_strategies: 需要生成的策略数量
            param_bounds: 参数边界
            max_retries: 最大重试次数
            save_path: 保存路径（可选）
        
        返回：
            strategies: [{'current1': ..., 'switch_soc': ..., 'current2': ...}, ...]
        """
        # 默认参数边界
        if param_bounds is None:
            param_bounds = {
                'current1': (3.0, 6.0),
                'switch_soc': (0.3, 0.7),
                'current2': (1.0, 4.0)
            }
        
        # 构建prompt
        prompt = inject_template_values(self.template, n_strategies)
        
        if self.verbose:
            print(f"  [LLM WarmStart] 调用 {self.model} (context={self.context_level})...")
        
        # 调用LLM（带重试）
        try:
            strategies = await self._call_with_retry(
                prompt, 
                n_strategies, 
                param_bounds, 
                max_retries
            )
            
            # 保存（如果指定）
            if save_path is not None:
                self._save_strategies(strategies, save_path, param_bounds)
            
            return strategies
        
        except Exception as e:
            if self.verbose:
                print(f"  [LLM WarmStart] 所有重试失败: {e}")
                print(f"  回退到随机策略")
            
            # 回退：生成随机策略
            return [generate_random_strategy(param_bounds, seed=i) for i in range(n_strategies)]
    
    async def _call_with_retry(
        self,
        prompt: str,
        n_strategies: int,
        param_bounds: Dict,
        max_retries: int
    ) -> List[Dict]:
        """

        参数：
            prompt: LLM提示词
            n_strategies: 目标策略数量
            param_bounds: 参数边界
            max_retries: 最大重试次数
        
        返回：
            策略列表
        """
        for retry in range(max_retries):
            try:
                # 调用LLM
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
                    temperature=get_llm_param('warmstart', 'temperature', 0.7),
                    max_tokens=get_llm_param('warmstart', 'max_tokens', 2500),
                    response_format={"type": "json_object"}
                )
                
                # 防御性解析
                raw_content = response.choices[0].message.content
                
                # 尝试提取 ## JSON ## 标记内容
                if '## JSON ##' in raw_content:
                    try:
                        json_parts = raw_content.split('## JSON ##')
                        json_str = json_parts[1].strip() if len(json_parts) >= 2 else json_parts[-1].strip()
                        result = json.loads(json_str)
                        
                        if self.verbose:
                            print(f"  [LLM WarmStart] 成功解析防御性JSON标记")
                    
                    except (json.JSONDecodeError, IndexError) as e:
                        if self.verbose:
                            print(f"  [LLM WarmStart] 防御性解析失败，尝试直接解析: {e}")
                        result = json.loads(raw_content)
                else:
                    # 没有标记，直接解析
                    result = json.loads(raw_content)
                
                # 提取策略
                strategies = self._extract_strategies(result, n_strategies, param_bounds)
                
                if self.verbose:
                    print(f"  [LLM WarmStart] 成功生成 {len(strategies)}/{n_strategies} 个有效策略")
                
                return strategies
            
            except Exception as e:
                if retry < max_retries - 1:
                    # 指数退避
                    backoff_base = get_llm_param('warmstart', 'retry_backoff_base', 2)
                    wait_time = backoff_base ** retry
                    if self.verbose:
                        print(f"  [LLM WarmStart] 重试 {retry+1}/{max_retries}，等待 {wait_time}s...")
                        print(f"  错误: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    # 最后一次重试失败，抛出异常
                    raise e
    
    def _extract_strategies(
        self,
        result: Dict,
        n_strategies: int,
        param_bounds: Dict
    ) -> List[Dict]:
        """
        从LLM响应中提取和验证策略
        
        参数：
            result: LLM返回的JSON
            n_strategies: 目标数量
            param_bounds: 参数边界
        
        返回：
            验证后的策略列表
        """
        # 提取策略
        if 'strategies' in result:
            raw_strategies = result['strategies']
        elif isinstance(result, list):
            raw_strategies = result
        else:
            raw_strategies = [result]
        
        # 验证和清理
        valid_strategies = []
        for s in raw_strategies:
            if validate_strategy(s, param_bounds, verbose=False):
                cleaned = clean_strategy(s)
                valid_strategies.append(cleaned)
        
        # 如果生成不足，补充随机策略
        while len(valid_strategies) < n_strategies:
            random_strategy = generate_random_strategy(param_bounds, seed=len(valid_strategies))
            valid_strategies.append(random_strategy)
            if self.verbose:
                print(f"  [LLM WarmStart] 补充随机策略 {len(valid_strategies)}/{n_strategies}")
        
        return valid_strategies[:n_strategies]
    
    def _save_strategies(
        self,
        strategies: List[Dict],
        save_path: str,
        param_bounds: Dict
    ):
        """
        保存生成的策略到JSON
        
        参数：
            strategies: 策略列表
            save_path: 保存路径
            param_bounds: 参数边界
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
                'param_bounds': param_bounds
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
            
            # 测试随机回退
            param_bounds = {
                'current1': (3.0, 6.0),
                'switch_soc': (0.3, 0.7),
                'current2': (1.0, 4.0)
            }
            
            for i in range(3):
                strategy = generate_random_strategy(param_bounds, seed=i)
                print(f"    策略{i+1}: I1={strategy['current1']:.1f}A, "
                      f"SOC_sw={strategy['switch_soc']:.2f}, "
                      f"I2={strategy['current2']:.1f}A")
            
            return
        
        # 参数边界
        param_bounds = {
            'current1': (3.0, 6.0),
            'switch_soc': (0.3, 0.7),
            'current2': (1.0, 4.0)
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
            
            # 显示结果
            print(f"\n生成的策略 ({context_level}):")
            for i, s in enumerate(strategies, 1):
                rationale = s.get('rationale', 'N/A')
                print(f"  策略{i}: I1={s['current1']:.1f}A, "
                      f"SOC_sw={s['switch_soc']:.2f}, "
                      f"I2={s['current2']:.1f}A")
                if rationale != 'N/A':
                    print(f"          {rationale}")
        
        print("\n测试完成！")
    
    # 运行测试
    asyncio.run(test())