"""
WarmStart辅助函数
提供数值格式化、策略验证、模板加载等功能
"""

import os
import numpy as np
from typing import Dict, List


def count_decimal_places(n: float) -> int:
    """
    计算浮点数的小数位数（参考LLAMBO）
    
    用途：确保生成的参数值与边界的精度匹配
    
    示例：
        count_decimal_places(3.0) → 0
        count_decimal_places(0.30) → 1
        count_decimal_places(0.001) → 3
    
    参数：
        n: 浮点数
    
    返回：
        小数位数
    """
    s = format(n, '.10f')
    if '.' not in s:
        return 0
    return len(s.split('.')[1].rstrip('0'))


def format_parameter(param_name: str, value: float, param_bounds: Dict) -> str:
    """
    格式化参数值（匹配边界的小数位数）
    
    参数：
        param_name: 参数名称
        value: 参数值
        param_bounds: 参数边界字典
    
    返回：
        格式化后的字符串
    """
    lower_bound = param_bounds[param_name][0]
    
    # 计算边界的小数位数
    n_dp = count_decimal_places(lower_bound)
    
    # 格式化
    if param_name == 'switch_soc':
        # SOC通常保留2位小数
        return f'{value:.2f}'
    elif param_name in ['current1', 'current2']:
        # 电流通常保留1位小数
        return f'{value:.1f}'
    else:
        # 其他参数根据边界自动决定
        return f'{value:.{n_dp}f}'


def validate_strategy(strategy: Dict, param_bounds: Dict, verbose: bool = False) -> bool:
    """
    验证策略是否满足约束
    
    检查项：
    1. 参数值在边界内
    2. 数值类型正确
    3. 物理合理性（可选）
    
    参数：
        strategy: 策略字典
        param_bounds: 参数边界
        verbose: 是否打印详细错误信息
    
    返回：
        True if valid, False otherwise
    """
    try:
        # 检查必需的key
        required_keys = ['current1', 'switch_soc', 'current2']
        for key in required_keys:
            if key not in strategy:
                if verbose:
                    print(f"  验证失败: 缺少参数 '{key}'")
                return False
        
        # 提取并转换为float
        current1 = float(strategy['current1'])
        switch_soc = float(strategy['switch_soc'])
        current2 = float(strategy['current2'])
        
        # 检查边界
        if not (param_bounds['current1'][0] <= current1 <= param_bounds['current1'][1]):
            if verbose:
                print(f"  验证失败: current1={current1:.2f} 超出边界 {param_bounds['current1']}")
            return False
        
        if not (param_bounds['switch_soc'][0] <= switch_soc <= param_bounds['switch_soc'][1]):
            if verbose:
                print(f"  验证失败: switch_soc={switch_soc:.2f} 超出边界 {param_bounds['switch_soc']}")
            return False
        
        if not (param_bounds['current2'][0] <= current2 <= param_bounds['current2'][1]):
            if verbose:
                print(f"  验证失败: current2={current2:.2f} 超出边界 {param_bounds['current2']}")
            return False
        
        # 物理合理性检查（更严格）
        issues = []
        
        # 检查1: current1应该 > current2
        if current1 <= current2:
            issues.append(f"current1({current1:.1f}) ≤ current2({current2:.1f})")
        
        # 检查2: 过于激进的电流
        if current1 > 6.0:
            if verbose:
                print(f"  拒绝: current1={current1:.1f}A 超过安全上限6.0A")
            return False
        
        # 检查3: SOC-电流耦合检查（参考LLMBO约束）
        if current1 > 5.5 and switch_soc > 0.60:
            issues.append(f"高电流({current1:.1f}A) + 高SOC切换({switch_soc:.2f}) 可能导致温度失控")
        
        # 检查4: 电流差异过小（缺乏两阶段意义）
        if abs(current1 - current2) < 0.5:
            issues.append(f"电流差异过小({abs(current1-current2):.1f}A)，两阶段意义不明显")
        
        # 打印警告但不拒绝（除非current1 > 6.0）
        if verbose and issues:
            print(f"  警告: {'; '.join(issues)}")
        
        return True
    
    except (KeyError, ValueError, TypeError) as e:
        if verbose:
            print(f"  验证失败: {e}")
        return False


def load_template(context_level: str) -> str:
    """
    加载指定上下文级别的模板
    
    参数：
        context_level: 'full', 'partial', 'none'
    
    返回：
        模板字符串
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模板文件路径
    template_files = {
        'full': os.path.join(current_dir, 'templates', 'battery_full.txt'),
        'partial': os.path.join(current_dir, 'templates', 'battery_partial.txt'),
        'none': os.path.join(current_dir, 'templates', 'battery_none.txt')
    }
    
    if context_level not in template_files:
        raise ValueError(f"Invalid context_level: {context_level}. Must be 'full', 'partial', or 'none'.")
    
    template_path = template_files[context_level]
    
    # 读取模板
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return template
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")


def inject_template_values(template: str, n_strategies: int) -> str:
    """
    向模板注入具体数值
    
    参数：
        template: 模板字符串
        n_strategies: 需要生成的策略数量
    
    返回：
        注入后的prompt字符串
    """
    # 替换占位符
    prompt = template.replace('[N_STRATEGIES]', str(n_strategies))
    
    return prompt


def clean_strategy(strategy: Dict) -> Dict:
    """
    清理和标准化策略字典
    
    功能：
    1. 转换数值类型
    2. 四舍五入到合适精度
    3. 移除多余字段
    
    参数：
        strategy: 原始策略字典
    
    返回：
        清理后的策略字典
    """
    cleaned = {
        'current1': round(float(strategy['current1']), 1),  # 保留1位小数
        'switch_soc': round(float(strategy['switch_soc']), 2),  # 保留2位小数
        'current2': round(float(strategy['current2']), 1)   # 保留1位小数
    }
    
    # 保留rationale（如果有）
    if 'rationale' in strategy:
        cleaned['rationale'] = strategy['rationale']
    
    return cleaned


def generate_random_strategy(param_bounds: Dict, seed: int = None) -> Dict:
    """
    生成随机策略（回退方案）
    
    参数：
        param_bounds: 参数边界
        seed: 随机种子（可选）
    
    返回：
        随机策略字典
    """
    if seed is not None:
        np.random.seed(seed)
    
    strategy = {
        'current1': round(np.random.uniform(*param_bounds['current1']), 1),
        'switch_soc': round(np.random.uniform(*param_bounds['switch_soc']), 2),
        'current2': round(np.random.uniform(*param_bounds['current2']), 1)
    }
    
    return strategy


def score_strategy_safety(strategy: Dict, param_bounds: Dict) -> float:
    """
    评估策略的安全性得分（参考LLMBO的参数相关性）
    
    得分越高 = 越安全（但可能越慢）
    得分越低 = 越激进（但可能违反约束）
    
    参数：
        strategy: 策略字典
        param_bounds: 参数边界（未使用，保留接口兼容）
    
    返回：
        safety_score: [0, 1]，1=极安全，0=极危险
    """
    current1 = strategy['current1']
    switch_soc = strategy['switch_soc']
    current2 = strategy['current2']
    
    score = 1.0
    
    # 惩罚1: 高电流
    if current1 > 5.5:
        score -= 0.3 * (current1 - 5.5) / 0.5  # 5.5-6.0范围惩罚0-0.3
    
    # 惩罚2: 高SOC + 高电流的耦合风险
    if current1 > 5.0 and switch_soc > 0.60:
        score -= 0.2
    
    # 惩罚3: 电流差异过小
    if abs(current1 - current2) < 1.0:
        score -= 0.1
    
    # 奖励: current1 > current2（物理合理）
    if current1 > current2:
        score += 0.1
    
    # 限制在[0, 1]
    return max(0.0, min(1.0, score))



# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("测试 warmstart_utils...")
    
    # 测试1: 小数位数计算
    print("\n[测试1] 小数位数计算:")
    test_values = [3.0, 0.30, 0.001, 1.234567]
    for val in test_values:
        n_dp = count_decimal_places(val)
        print(f"  {val} → {n_dp} 位小数")
    
    # 测试2: 参数格式化
    print("\n[测试2] 参数格式化:")
    param_bounds = {
        'current1': (3.0, 6.0),
        'switch_soc': (0.3, 0.7),
        'current2': (1.0, 4.0)
    }
    test_strategy = {'current1': 5.123, 'switch_soc': 0.456, 'current2': 2.789}
    for key, val in test_strategy.items():
        formatted = format_parameter(key, val, param_bounds)
        print(f"  {key}: {val} → {formatted}")
    
    # 测试3: 策略验证
    print("\n[测试3] 策略验证:")
    valid_strategy = {'current1': 5.0, 'switch_soc': 0.5, 'current2': 2.5}
    invalid_strategy = {'current1': 10.0, 'switch_soc': 0.5, 'current2': 2.5}
    
    print(f"  有效策略: {validate_strategy(valid_strategy, param_bounds, verbose=True)}")
    print(f"  无效策略: {validate_strategy(invalid_strategy, param_bounds, verbose=True)}")
    
    # 测试4: 模板加载
    print("\n[测试4] 模板加载:")
    for level in ['full', 'partial', 'none']:
        try:
            template = load_template(level)
            print(f"  {level}: 加载成功 ({len(template)} 字符)")
        except Exception as e:
            print(f"  {level}: 加载失败 - {e}")
    
    # 测试5: 模板注入
    print("\n[测试5] 模板注入:")
    template = load_template('full')
    prompt = inject_template_values(template, n_strategies=5)
    print(f"  注入后长度: {len(prompt)} 字符")
    print(f"  是否包含 '5 个策略': {'5 个策略' in prompt}")
    
    print("\n测试完成！")