"""
optimizer.py  —  差分进化（DE）优化采集函数
==============================================
完全对标 MATLAB ParEGO_Source.m 中的 DE 子函数。

MATLAB 原版参数：
    种群大小: 30
    迭代次数: 200（Source）/ 20（TrOpt）
    F = 0.5（缩放因子）
    CR = 0.9（交叉概率）
    策略: DE/rand/1/bin
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple


def de_optimize(
    obj_func: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    pop_size: int = 30,
    max_iter: int = 200,
    F: float = 0.5,
    CR: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """
    差分进化优化器，对标 MATLAB DE 子函数。

    Parameters
    ----------
    obj_func : 目标函数（标量输入，标量输出），DE 做最小化
               对标 MATLAB obj()，传入 neg_ei_scalar
    lb       : (dim,) 搜索下界（归一化空间）
    ub       : (dim,) 搜索上界（归一化空间）
    pop_size : 种群大小（对标 MATLAB 30）
    max_iter : 迭代次数（对标 MATLAB 200）
    F        : 缩放因子（对标 MATLAB 0.5）
    CR       : 交叉概率（对标 MATLAB 0.9）
    rng      : numpy 随机数生成器

    Returns
    -------
    x_best : (dim,) 最优候选点
    y_best : float  最优目标值
    """
    if rng is None:
        rng = np.random.default_rng()

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = len(lb)

    # ── 初始化种群（对标 MATLAB: x = (up-dn).*rand([30,d])+dn）──────
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)

    # ── 初始评估 ────────────────────────────────────────────────────
    fitness = np.array([obj_func(pop[i]) for i in range(pop_size)])

    # ── 主循环 ──────────────────────────────────────────────────────
    for _ in range(max_iter):
        for i in range(pop_size):
            # DE/rand/1 变异（对标 MATLAB: v = x(rs(1),:) + 0.5*(x(rs(2),:)-x(rs(3),:))）
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(candidates, 3, replace=False)
            v = pop[r1] + F * (pop[r2] - pop[r3])

            # 二项交叉（对标 MATLAB: u = v.*(rj<0.9) + x.*(rj>=0.9)）
            mask = rng.random(dim) < CR
            if not mask.any():
                mask[rng.integers(dim)] = True   # 保证至少一维来自变异向量
            u = np.where(mask, v, pop[i])

            # 边界修复（对标 MATLAB 逐元素 clip）
            u = np.clip(u, lb, ub)

            # 贪婪选择
            f_u = obj_func(u)
            if f_u <= fitness[i]:
                pop[i]     = u
                fitness[i] = f_u

    # ── 返回最优解 ───────────────────────────────────────────────────
    best_idx = int(np.argmin(fitness))
    return pop[best_idx].copy(), float(fitness[best_idx])
