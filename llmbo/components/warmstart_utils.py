"""
WarmStart辅助函数
提供数值格式化、策略验证、模板加载等功能
"""

import os
import numpy as np
from typing import Dict, List


def validate_strategy(strategy: Dict, param_bounds: Dict, verbose: bool = False) -> bool:
    """
    验证策略参数是否在边界内（动态从param_bounds获取keys）
    """
    try:
        # 动态获取required_keys
        required_keys = list(param_bounds.keys())
        
        for key in required_keys:
            if key not in strategy:
                if verbose:
                    print(f"  验证失败: 缺少参数 '{key}'")
                return False
        
        for key in required_keys:
            val = float(strategy[key])
            lo, hi = param_bounds[key]
            if not (lo <= val <= hi):
                if verbose:
                    print(f"  验证失败: {key}={val} 超出边界 [{lo}, {hi}]")
                return False
        
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


def clean_strategy(strategy: Dict, param_bounds: Dict) -> Dict:
    """
    清理和标准化策略字典（动态clip到边界内）
    
    参数：
        strategy: 原始策略字典
        param_bounds: 参数边界
    
    返回：
        清理后的策略（clip到边界内，保留rationale等额外字段）
    """
    cleaned = {}
    
    # 对每个参数进行clip
    for key in param_bounds.keys():
        if key in strategy:
            val = float(strategy[key])
            lo, hi = param_bounds[key]
            clipped = np.clip(val, lo, hi)
            
            # 根据参数类型确定精度
            if key == 'v_switch':
                cleaned[key] = round(clipped, 2)
            else:
                cleaned[key] = round(clipped, 1)
    
    # 保留额外字段
    for key in ['rationale', 'scenario', 'reasoning']:
        if key in strategy:
            cleaned[key] = strategy[key]
    
    return cleaned


def generate_random_strategy(param_bounds: Dict, seed: int = None) -> Dict:
    """
    生成随机策略（动态从param_bounds生成）
    
    参数：
        param_bounds: 参数边界字典
        seed: 随机种子（可选）
    
    返回：
        随机策略字典
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    strategy = {}
    for key, (lo, hi) in param_bounds.items():
        val = rng.uniform(lo, hi)
        # 根据参数类型确定精度
        if key == 'v_switch':
            strategy[key] = round(val, 2)
        else:
            strategy[key] = round(val, 1)
    
    return strategy




def select_diverse_maxmin(candidates: List[Dict], n_required: int) -> List[Dict]:
    """从候选池中选择最分散的 n 个策略（MaxMin 距离准则）"""
    if len(candidates) <= n_required:
        return candidates
    
    param_keys = ['current1', 'time1', 'current2', 'v_switch']
    X = np.array([[c[k] for k in param_keys] for c in candidates])
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    selected_indices = [0]
    for _ in range(n_required - 1):
        best_idx = max(
            (i for i in range(len(X_norm)) if i not in selected_indices),
            key=lambda i: min(
                np.linalg.norm(X_norm[i] - X_norm[j])
                for j in selected_indices
            )
        )
        selected_indices.append(best_idx)
    
    return [candidates[i] for i in selected_indices]


def compute_generalized_variance(
    strategies: List[Dict],
    param_keys: List[str] = None
) -> float:
    """
    计算策略集的 Generalized Variance（LLAMBO Section 4 diversity 度量）
    
    GV = det(Σ), Σ 为参数的协方差矩阵
    
    返回:
        det(Σ)，策略数 < 2 时返回 0.0
    """
    if param_keys is None:
        param_keys = ['current1', 'time1', 'current2', 'v_switch']
    
    if len(strategies) < 2:
        return 0.0
    
    try:
        X = np.array([[float(s[k]) for k in param_keys] for s in strategies])
        cov_matrix = np.cov(X.T)
        gv = np.linalg.det(cov_matrix)
        if np.isnan(gv) or np.isinf(gv):
            return 0.0
        return float(gv)
    except Exception:
        return 0.0


