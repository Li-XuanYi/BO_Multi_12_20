"""
WarmStart 辅助函数
提供数值格式化、策略验证、模板加载等功能

统一 3D 参数空间：I1, SOC1, I2
"""

import os
import numpy as np
from typing import Dict, List, Optional


def validate_strategy(strategy: Dict, param_bounds: Dict, verbose: bool = False) -> bool:
    """
    验证策略参数是否在边界内（动态从 param_bounds 获取 keys）

    参数：
        strategy: 策略字典
        param_bounds: 参数边界 {'I1': (3.0, 7.0), 'SOC1': (0.1, 0.7), 'I2': (1.0, 5.0)}
        verbose: 是否打印详细验证信息

    返回：
        bool: 验证是否通过
    """
    try:
        # 动态获取 required_keys
        required_keys = list(param_bounds.keys())

        for key in required_keys:
            if key not in strategy:
                if verbose:
                    print(f"  验证失败：缺少参数 '{key}'")
                return False

        for key in required_keys:
            val = float(strategy[key])
            lo, hi = param_bounds[key]
            if not (lo <= val <= hi):
                if verbose:
                    print(f"  验证失败：{key}={val} 超出边界 [{lo}, {hi}]")
                return False

        # 检查 I2 <= I1 约束
        if 'I1' in strategy and 'I2' in strategy:
            if float(strategy['I2']) > float(strategy['I1']) + 0.1:  # 允许 0.1 的容差
                if verbose:
                    print(f"  验证失败：I2={strategy['I2']} > I1={strategy['I1']}")
                return False

        return True

    except (KeyError, ValueError, TypeError) as e:
        if verbose:
            print(f"  验证失败：{e}")
        return False


def load_template(context_level: str, prompt_dir: str = "prompts") -> str:
    """
    加载指定上下文级别的模板（从 prompts 目录）

    参数：
        context_level: 'full', 'partial', 'none'
        prompt_dir: Prompt 目录名称（默认 'prompts'，相对于 components 目录）

    返回：
        模板字符串
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 模板文件路径（从 prompts/warmstart 目录加载）
    template_files = {
        'full': os.path.join(current_dir, prompt_dir, 'warmstart', 'battery_full.txt'),
        'partial': os.path.join(current_dir, prompt_dir, 'warmstart', 'battery_partial.txt'),
        'none': os.path.join(current_dir, prompt_dir, 'warmstart', 'battery_none.txt')
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


def inject_template_values(template: str, n_strategies: int, history_section: str = "") -> str:
    """
    向模板注入具体数值

    参数：
        template: 模板字符串
        n_strategies: 需要生成的策略数量
        history_section: 历史观测数据（可选）

    返回：
        注入后的 prompt 字符串
    """
    # 替换占位符
    prompt = template.replace('[N_STRATEGIES]', str(n_strategies))

    # 处理历史记录部分
    if history_section:
        prompt = prompt.replace('{HISTORY_SECTION}', history_section)
    else:
        prompt = prompt.replace('{HISTORY_SECTION}', '(No historical data available - this is the initial warm-start phase)')

    return prompt


def clean_strategy(strategy: Dict, param_bounds: Dict) -> Dict:
    """
    清理和标准化策略字典（动态 clip 到边界内）- 3D 参数空间

    参数：
        strategy: 原始策略字典
        param_bounds: 参数边界

    返回：
        清理后的策略（clip 到边界内，保留 rationale 等额外字段）
    """
    cleaned = {}

    # 对每个参数进行 clip
    for key in param_bounds.keys():
        if key in strategy:
            val = float(strategy[key])
            lo, hi = param_bounds[key]
            clipped = np.clip(val, lo, hi)

            # 根据参数类型确定精度
            if key == 'SOC1':
                cleaned[key] = round(clipped, 3)  # SOC 保留 3 位小数
            else:
                cleaned[key] = round(clipped, 2)  # 电流保留 2 位小数

    # 保留额外字段
    for key in ['rationale', 'scenario', 'reasoning']:
        if key in strategy:
            cleaned[key] = strategy[key]

    # 确保 I2 <= I1 约束（软性调整）
    if 'I1' in cleaned and 'I2' in cleaned:
        if cleaned['I2'] > cleaned['I1']:
            cleaned['I2'] = cleaned['I1']

    return cleaned


def generate_random_strategy(param_bounds: Dict, seed: int = None) -> Dict:
    """
    生成随机策略（动态从 param_bounds 生成）- 3D 参数空间

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
        if key == 'SOC1':
            strategy[key] = round(val, 3)  # SOC 保留 3 位小数
        else:
            strategy[key] = round(val, 2)  # 电流保留 2 位小数

    # 确保 I2 <= I1
    if 'I1' in strategy and 'I2' in strategy:
        if strategy['I2'] > strategy['I1']:
            strategy['I2'] = strategy['I1']

    return strategy


def validate_and_clip_candidate(candidate: Dict, param_bounds: Dict) -> Optional[Dict]:
    """
    验证并裁剪候选点到边界内（用于 LLM 输出后处理）

    参数：
        candidate: LLM 生成的原始候选点
        param_bounds: 参数边界

    返回：
        验证并通过的候选点，如果验证失败返回 None
    """
    try:
        validated = {}

        # 检查必需参数
        for key in param_bounds.keys():
            if key not in candidate:
                return None
            val = float(candidate[key])
            lo, hi = param_bounds[key]
            validated[key] = np.clip(val, lo, hi)

        # 精度处理
        for key in param_bounds.keys():
            if key == 'SOC1':
                validated[key] = round(validated[key], 3)
            else:
                validated[key] = round(validated[key], 2)

        # I2 <= I1 约束（软约束，允许轻微违反后调整）
        if validated.get('I2', 0) > validated.get('I1', 0):
            validated['I2'] = validated['I1']

        # 保留额外字段
        for key in ['rationale', 'reasoning', 'source']:
            if key in candidate:
                validated[key] = candidate[key]

        return validated

    except (KeyError, ValueError, TypeError):
        return None


def select_diverse_maxmin(candidates: List[Dict], n_required: int) -> List[Dict]:
    """从候选池中选择最分散的 n 个策略（MaxMin 距离准则）- 3D 参数空间"""
    if len(candidates) <= n_required:
        return candidates

    # 3D 参数空间
    param_keys = ['I1', 'SOC1', 'I2']
    X = np.array([[c[k] for k in param_keys] for c in candidates if all(k in c for k in param_keys)])

    if len(X) < n_required:
        return candidates[:n_required] if candidates else []

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
        param_keys = ['I1', 'SOC1', 'I2']  # 3D 参数空间

    if len(strategies) < 2:
        return 0.0

    try:
        X = np.array([[float(s[k]) for k in param_keys] for s in strategies if all(k in s for k in param_keys)])
        if len(X) < 2:
            return 0.0
        cov_matrix = np.cov(X.T)
        gv = np.linalg.det(cov_matrix)
        if np.isnan(gv) or np.isinf(gv):
            return 0.0
        return float(gv)
    except Exception:
        return 0.0


# ============================================================
# 新增：DatabaseSummarizer - 历史数据摘要生成器
# ============================================================
class DatabaseSummarizer:
    """
    数据库摘要生成器 - 为 LLM 提供历史观测的结构化摘要

    参考 FrameWork.md §7: DatabaseSummarizer & Prompt Template
    """

    def __init__(self, database: List[Dict], context_level: str = 'partial'):
        """
        参数：
            database: 实验记录列表，格式为 [{'params': {...}, 'time': ..., 'temp': ..., 'aging': ..., 'valid': bool}, ...]
            context_level: 'full', 'partial', 'none'
        """
        self.database = database or []
        self.context_level = context_level
        self.valid_experiments = [exp for exp in self.database if exp.get('valid', False)]

    def generate_summary(
        self,
        weights: np.ndarray = None,
        grad_psi: np.ndarray = None,
        iteration: int = None,
        total_iterations: int = None,
        n_top: int = 5
    ) -> str:
        """
        生成结构化摘要

        参数：
            weights: Tchebycheff 权重向量
            grad_psi: Ψ 梯度
            iteration: 当前迭代
            total_iterations: 总迭代数
            n_top: 返回 Top-K 观测

        返回：
            摘要字符串
        """
        if not self.valid_experiments:
            return self._empty_summary(iteration, total_iterations)

        sections = []

        # 1. 进度信息
        if iteration is not None:
            sections.append(f"**Optimization Progress:** Iteration {iteration}/{total_iterations}")

        # 2. Top 观测（按标量化值排序）
        if weights is not None:
            top_section = self._format_top_observations(weights, n_top)
            sections.append(top_section)

        # 3. 统计摘要
        stats_section = self._format_statistics()
        sections.append(stats_section)

        # 4. 梯度信息（Full context 专用）
        if self.context_level == 'full' and grad_psi is not None:
            grad_section = self._format_gradient_info(grad_psi)
            sections.append(grad_section)

        return "\n\n".join(sections)

    def _empty_summary(self, iteration: int = None, total_iterations: int = None) -> str:
        """空数据库摘要"""
        if iteration is not None:
            return f"**Optimization Progress:** Iteration {iteration}/{total_iterations}\n\n**No historical data available - this is the initial warm-start phase.**"
        return "**No historical data available.**"

    def _format_top_observations(self, weights: np.ndarray, n_top: int) -> str:
        """格式化 Top-K 观测"""
        # 计算标量化值
        scored = []
        for exp in self.valid_experiments:
            obj = np.array([exp['time'], exp['temp'], exp['aging']])
            # Tchebycheff scalarization (simplified)
            score = np.max(weights * np.abs(obj - np.min([e['time'] for e in self.valid_experiments])))
            scored.append((exp, score))

        # 排序
        scored.sort(key=lambda x: x[1])
        top_n = scored[:n_top]

        lines = ["**Top Observations (by scalarized objective):**"]
        for i, (exp, score) in enumerate(top_n, 1):
            params = exp.get('params', {})
            # 支持 3D 和 4D 格式
            if 'I1' in params:
                param_str = f"I1={params['I1']:.2f}A, SOC1={params['SOC1']:.3f}, I2={params['I2']:.2f}A"
            else:
                param_str = f"I1={params.get('current1', 'N/A')}, SOC1={params.get('time1', 'N/A')}, I2={params.get('current2', 'N/A')}"

            lines.append(f"  {i}. [{param_str}] → time={exp['time']:.0f}s, temp={exp['temp']:.1f}K, aging={exp['aging']:.2e}")

        return "\n".join(lines)

    def _format_statistics(self) -> str:
        """格式化统计摘要"""
        if not self.valid_experiments:
            return "**Statistics:** No valid experiments"

        times = [e['time'] for e in self.valid_experiments]
        temps = [e['temp'] for e in self.valid_experiments]
        agings = [e['aging'] for e in self.valid_experiments]

        lines = [
            "**Statistics:**",
            f"  Total experiments: {len(self.database)} (valid: {len(self.valid_experiments)})",
            f"  Time: min={min(times):.0f}s, max={max(times):.0f}s, mean={np.mean(times):.0f}s",
            f"  Temp: min={min(temps):.1f}K, max={max(temps):.1f}K, mean={np.mean(temps):.1f}K",
            f"  Aging: min={min(agings):.2e}, max={max(agings):.2e}, mean={np.mean(agings):.2e}"
        ]
        return "\n".join(lines)

    def _format_gradient_info(self, grad_psi: np.ndarray) -> str:
        """格式化梯度信息"""
        param_names = ['I1', 'SOC1', 'I2']
        lines = ["**Sensitivity Metrics (Ψ gradient):**"]
        for name, grad in zip(param_names, grad_psi):
            lines.append(f"  |∂Ψ/∂{name}| = {abs(grad):.4f}")
        return "\n".join(lines)


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 warmstart_utils (3D 参数空间)")
    print("=" * 60)

    # 测试 1: 参数边界验证
    print("\n[测试 1] 参数边界验证")
    param_bounds = {'I1': (3.0, 7.0), 'SOC1': (0.1, 0.7), 'I2': (1.0, 5.0)}
    strategy = {'I1': 5.0, 'SOC1': 0.4, 'I2': 3.0}
    assert validate_strategy(strategy, param_bounds, verbose=True)
    print("  验证通过")

    # 测试 2: 随机策略生成
    print("\n[测试 2] 随机策略生成")
    for i in range(3):
        s = generate_random_strategy(param_bounds, seed=i)
        print(f"  策略{i+1}: I1={s['I1']:.2f}A, SOC1={s['SOC1']:.3f}, I2={s['I2']:.2f}A")
        assert validate_strategy(s, param_bounds)

    # 测试 3: 策略清理
    print("\n[测试 3] 策略清理（clip 到边界）")
    dirty = {'I1': 10.0, 'SOC1': 0.9, 'I2': 2.0, 'rationale': 'test'}
    cleaned = clean_strategy(dirty, param_bounds)
    print(f"  原始：I1=10.0, SOC1=0.9, I2=2.0")
    print(f"  清理后：I1={cleaned['I1']:.2f}, SOC1={cleaned['SOC1']:.3f}, I2={cleaned['I2']:.2f}")
    assert cleaned['I1'] == 7.0  # clip 到上限
    assert cleaned['SOC1'] == 0.7  # clip 到上限

    # 测试 4: DatabaseSummarizer
    print("\n[测试 4] DatabaseSummarizer")
    mock_db = [
        {'params': {'I1': 4.0, 'SOC1': 0.3, 'I2': 2.5}, 'time': 1800, 'temp': 310, 'aging': 0.002, 'valid': True},
        {'params': {'I1': 5.0, 'SOC1': 0.4, 'I2': 3.0}, 'time': 1500, 'temp': 315, 'aging': 0.003, 'valid': True},
        {'params': {'I1': 6.0, 'SOC1': 0.5, 'I2': 3.5}, 'time': 1300, 'temp': 320, 'aging': 0.004, 'valid': True},
    ]
    summarizer = DatabaseSummarizer(mock_db, context_level='full')
    summary = summarizer.generate_summary(
        weights=np.array([0.4, 0.3, 0.3]),
        grad_psi=np.array([0.1, 0.05, 0.08]),
        iteration=5,
        total_iterations=50
    )
    print(summary)

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
