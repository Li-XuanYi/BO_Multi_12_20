"""
scalarization.py  —  权重向量生成 + 增广 Chebyshev 标量化
============================================================
完全对标 MATLAB 的 EqualWeight(30, 3) + 增广 Chebyshev 聚合函数。

EqualWeight(H, M) 生成规则：
    所有满足 sum(w) = H 的非负整数向量 w ∈ Z^M，
    归一化后得到均匀分布在单纯形上的权重向量。
    EqualWeight(30, 3) → C(32,2) = 496 个权重向量。
"""

from __future__ import annotations

import numpy as np
from itertools import combinations_with_replacement
from typing import Tuple


# ---------------------------------------------------------------------------
#  权重向量生成（对标 MATLAB EqualWeight）
# ---------------------------------------------------------------------------

def equal_weight(H: int, M: int) -> np.ndarray:
    """
    生成均匀单纯形权重向量，完全对标 MATLAB EqualWeight(H, M)。

    Parameters
    ----------
    H : 划分粒度（MATLAB 用 30）
    M : 目标数（3）

    Returns
    -------
    W : (n_w, M) 权重矩阵，每行之和为 1，所有元素 > 0
        行数 = C(H+M-1, M-1)

    示例
    ----
    equal_weight(30, 3) → shape (496, 3)
    """
    # 生成所有满足 sum = H 的非负整数向量
    indices = list(combinations_with_replacement(range(M), H))
    # 统计每个维度出现次数
    W_int = np.zeros((len(indices), M), dtype=int)
    for i, idx in enumerate(indices):
        for j in idx:
            W_int[i, j] += 1

    W = W_int / H                    # 归一化到 [0,1]，和为 1
    W = W + 1e-6                     # 防止出现 0（对标 MATLAB: W = W + 0.000001）
    W = W / W.sum(axis=1, keepdims=True)  # 重新归一化
    return W.astype(np.float64)


def sample_weight(W: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    从权重矩阵 W 中随机选取一行，对标 MATLAB：
        idx_w_sel = ceil(n_w * rand());
        w_sel = W(idx_w_sel, :);

    Parameters
    ----------
    W   : (n_w, M) 权重矩阵
    rng : numpy 随机数生成器（None 则使用全局 rng）

    Returns
    -------
    w : (M,) 权重向量
    """
    if rng is None:
        idx = np.random.randint(0, len(W))
    else:
        idx = rng.integers(0, len(W))
    return W[idx].copy()


# ---------------------------------------------------------------------------
#  增广 Chebyshev 标量化（对标 MATLAB ParEGO_Source.m）
# ---------------------------------------------------------------------------

def augmented_chebyshev(
    y_norm: np.ndarray,
    w: np.ndarray,
    rho: float = 0.05,
) -> np.ndarray:
    """
    增广 Chebyshev 标量化，完全对标 MATLAB：
        w_inv  = 1 / w_sel
        y_agg  = max(w_inv .* y_norm, [], 2) + rho * sum(w_inv .* y_norm, 2)

    Parameters
    ----------
    y_norm : (N, M) 归一化目标值（已做 min-max 归一化）
    w      : (M,)  权重向量（来自 sample_weight）
    rho    : 增广项系数（对标 MATLAB 的 0.05）

    Returns
    -------
    y_agg : (N,) 标量化聚合值，越小越好
    """
    w = np.asarray(w, dtype=np.float64).reshape(1, -1)   # (1, M)
    w_inv = 1.0 / w                                        # 对标 MATLAB: w_sel = 1./w_sel

    weighted = w_inv * y_norm                              # (N, M)
    y_agg = weighted.max(axis=1) + rho * weighted.sum(axis=1)
    return y_agg


def scalarize(
    y: np.ndarray,
    w: np.ndarray,
    rho: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    完整标量化流程（归一化 + 增广 Chebyshev + 标准化），
    对标 MATLAB ParEGO_Source.m 中：
        y_norm     = (DB.y - min(DB.y)) ./ (max(DB.y) - min(DB.y))
        y_agg      = max(w_inv .* y_norm)' + 0.05 * sum(w_inv .* y_norm, 2)
        y_agg_norm = (y_agg - mean(y_agg)) ./ std(y_agg)

    Parameters
    ----------
    y   : (N, M) 原始目标值
    w   : (M,)  权重向量
    rho : 增广项系数

    Returns
    -------
    y_agg_norm : (N,) 标准化后的聚合值（输入 GP 训练）
    y_agg      : (N,) 标准化前的聚合值（用于计算 f_best）
    """
    # Step 1: min-max 归一化
    lo    = y.min(axis=0)
    hi    = y.max(axis=0)
    denom = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    y_norm = (y - lo) / denom                             # (N, M)

    # Step 2: 增广 Chebyshev
    y_agg = augmented_chebyshev(y_norm, w, rho)           # (N,)

    # Step 3: 标准化（对标 MATLAB: (y_agg - mean) / std）
    mu    = y_agg.mean()
    sigma = y_agg.std()
    if sigma < 1e-12:
        sigma = 1.0
    y_agg_norm = (y_agg - mu) / sigma                     # (N,)

    return y_agg_norm, y_agg
