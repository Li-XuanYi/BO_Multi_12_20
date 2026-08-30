"""
acquisition.py  —  EI 采集函数
================================
标准期望改进（Expected Improvement），对标 MATLAB Infill_Standard_GP_EI.m。

MATLAB 原版注意事项：
    - MATLAB 代码用的是 mse（方差），直接除 s=mse（未开方）→ 这是 Bug
    - 本实现修正为标准 EI：s = std（标准差）
    - 返回值为负 EI（因为 DE 做的是最小化）
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mu: np.ndarray,
    std: np.ndarray,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    标准期望改进采集函数（最小化方向）。

    EI(θ) = (f_best - μ(θ) - ξ) · Φ(Z) + σ(θ) · φ(Z)
    其中 Z = (f_best - μ(θ) - ξ) / σ(θ)

    Parameters
    ----------
    mu     : (M,) GP 后验均值
    std    : (M,) GP 后验标准差
    f_best : 当前已知最优标量化值（min of y_agg_norm in DB）
    xi     : exploration-exploitation 权衡系数（默认 0.01）

    Returns
    -------
    ei : (M,) EI 值，越大越好
    """
    mu  = np.asarray(mu).reshape(-1)
    std = np.asarray(std).reshape(-1)

    improvement = f_best - mu - xi
    Z  = improvement / np.where(std > 1e-12, std, 1e-12)
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei = np.where(std > 1e-12, ei, 0.0)
    return ei


def neg_ei_scalar(
    x: np.ndarray,
    gp_model,
    f_best: float,
    xi: float = 0.01,
) -> float:
    """
    单点负 EI（供 DE 优化器作为目标函数）。

    对标 MATLAB obj() 函数，返回 -EI（DE 做最小化）。

    Parameters
    ----------
    x        : (dim,) 单个候选点（归一化）
    gp_model : GPModel 实例
    f_best   : 当前最优标量化值
    xi       : exploration 系数

    Returns
    -------
    neg_ei : float，值越小表示 EI 越大
    """
    x = np.atleast_2d(x)
    mu, std = gp_model.predict(x)
    ei = expected_improvement(mu, std, f_best, xi)
    return float(-ei[0])
