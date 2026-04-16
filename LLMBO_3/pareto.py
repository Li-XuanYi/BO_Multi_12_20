"""
pareto.py  —  Pareto 工具函数
==============================
非支配排序（对标 MATLAB NonDominateSort.m）
超体积计算（对标 MATLAB Hypervolume.m）

修复记录
--------
Bug 1 [non_dominate_sort]: 按第一目标排序后，仅检查 i→j 方向的支配关系，
      当 y_sorted[i,0] == y_sorted[j,0] 时，j 支配 i 的情况被漏检，
      导致被支配点错误地进入前沿1。
      修复：直接在原始 y 空间对未分配点做完整双向支配检查。

Bug 2 [_hv_3d_wfg]: z-轴切片厚度计算错误，对非顶层使用了
      (ref[2] - y[i,2]) 而非正确的 (y_sorted[i+1,2] - y_sorted[i,2])，
      导致所有非顶层切片被重复计算（double-counting）。
      修复：逆序扫描时使用相邻层 z 值之差作为切片厚度。
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
#  非支配排序
# ---------------------------------------------------------------------------

def non_dominate_sort(
    y: np.ndarray,
    first_front_only: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    非支配排序，对标 MATLAB NonDominateSort.m（均 minimize）。

    Parameters
    ----------
    y                : (N, M) 目标值矩阵，均 minimize
    first_front_only : True → 只找第一前沿后立即返回

    Returns
    -------
    front_rank : (N,) 每个个体的前沿编号（从 1 开始）
    max_front  : 最大前沿编号

    修复说明
    --------
    原版按第一目标排序后仅做 i→j 单向检查，在第一目标相同时
    漏掉 j 支配 i 的情况。修复后直接在原始 y 空间进行双向支配检查，
    不依赖排序前提。
    """
    N = len(y)
    front_rank = np.full(N, np.inf)
    max_front  = 0
    assigned   = np.zeros(N, dtype=bool)

    while not assigned.all():
        max_front   += 1
        current_front = []
        unassigned    = np.where(~assigned)[0]

        for i in unassigned:
            # 检查 i 是否被任意其他未分配点支配（完整双向检查）
            dominated_by_any = False
            for j in unassigned:
                if j == i:
                    continue
                diff = y[j] - y[i]               # j - i；若全 <= 且有 < 则 j 支配 i
                if (diff <= 1e-10).all() and (diff < -1e-10).any():
                    dominated_by_any = True
                    break
            if not dominated_by_any:
                current_front.append(i)

        for i in current_front:
            front_rank[i] = max_front
            assigned[i]   = True

        if first_front_only:
            break

    return front_rank, max_front


def pareto_front(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取第一 Pareto 前沿（非支配解集）。

    Parameters
    ----------
    y : (N, M) 目标值，均 minimize

    Returns
    -------
    pf_y   : (K, M) Pareto 前沿目标值
    pf_idx : (K,)   在原始 y 中的索引
    """
    N = len(y)
    is_dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = y[j] - y[i]
            if (diff <= 1e-10).all() and (diff < -1e-10).any():
                is_dominated[i] = True
                break
    pf_idx = np.where(~is_dominated)[0]
    return y[pf_idx], pf_idx


# ---------------------------------------------------------------------------
#  超体积计算（精确计算 M<=3，MC 估计 M>=4）
# ---------------------------------------------------------------------------

def hypervolume(y: np.ndarray, ref_point: np.ndarray) -> float:
    """
    计算超体积（HV），对标 MATLAB Hypervolume.m。

    Parameters
    ----------
    y         : (N, M) 目标值，均 minimize
    ref_point : (M,)   参考点（应支配所有 Pareto 点）

    Returns
    -------
    hv : float 超体积值
    """
    y         = np.asarray(y, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)

    mask = (y < ref_point).all(axis=1)
    y    = y[mask]
    if len(y) == 0:
        return 0.0

    M = y.shape[1]
    if M == 2:
        return _hv_2d(y, ref_point)
    elif M == 3:
        return _hv_3d_wfg(y, ref_point)
    else:
        return _hv_mc(y, ref_point, n_samples=int(1e5))


def _hv_2d(y: np.ndarray, ref: np.ndarray) -> float:
    """二维精确超体积（扫描线）。"""
    idx = np.argsort(y[:, 0])
    y   = y[idx]
    hv  = 0.0
    prev_y1 = ref[1]
    for row in y:
        if row[1] < prev_y1:
            hv += (ref[0] - row[0]) * (prev_y1 - row[1])
            prev_y1 = row[1]
    return hv


def _hv_3d_wfg(y: np.ndarray, ref: np.ndarray) -> float:
    """
    三维精确超体积，扫描线（sweep）算法。

    对标 MATLAB Hypervolume.m 的精确计算分支（M == 3）。

    原版 Bug（已修复）：
        对第 i 层（非顶层）错误地使用 (ref[2] - y[i,2]) 作为切片厚度，
        正确值应为 (y_sorted[i+1,2] - y_sorted[i,2])。
        这导致所有非顶层切片的贡献被重复累加（double-counting）。

    修复后算法（z 方向升序排列，逆序扫描）：
        thickness[i] = z_{i+1} - z_i   (i < n-1)
        thickness[n-1] = ref_z - z_{n-1}
        HV = sum_i  HV_2D(points[:i+1, :2]) * thickness[i]
    """
    mask = (y < ref).all(axis=1)
    y    = y[mask]
    if len(y) == 0:
        return 0.0

    idx      = np.argsort(y[:, 2])
    y_sorted = y[idx]
    n        = len(y_sorted)
    hv       = 0.0

    for i in range(n - 1, -1, -1):
        # ── 切片厚度（Bug 修复核心）─────────────────────────────────
        # 原版对所有 i 均使用 ref[2] - y_sorted[i,2]，造成 double-counting。
        # 正确：仅顶层使用 ref_z - z_top，其余使用相邻层差值。
        if i == n - 1:
            dz = ref[2] - y_sorted[i, 2]
        else:
            dz = y_sorted[i + 1, 2] - y_sorted[i, 2]  # ← 修复点

        if dz <= 0:
            continue

        hv += _hv_2d(y_sorted[:i + 1, :2], ref[:2]) * dz

    return hv


def _hv_mc(y: np.ndarray, ref: np.ndarray, n_samples: int = 100000) -> float:
    """Monte Carlo 超体积估计（M >= 4）。"""
    lo        = y.min(axis=0)
    samples   = np.random.uniform(lo, ref, size=(n_samples, y.shape[1]))
    dominated = np.zeros(n_samples, dtype=bool)
    for point in y:
        dominated |= (point <= samples).all(axis=1)
    volume = np.prod(ref - lo)
    return volume * dominated.mean()


# ---------------------------------------------------------------------------
#  归一化超体积（对标 MATLAB Cal_DB_HV.m）
# ---------------------------------------------------------------------------

def normalized_hypervolume(
    y: np.ndarray,
    ref_lo: np.ndarray,
    ref_hi: np.ndarray,
    ref_point: np.ndarray | None = None,
) -> float:
    """
    归一化后计算超体积，对标 MATLAB Cal_DB_HV.m：
        y_cal_hv = (y - ref_lo) / (ref_hi - ref_lo)
        HV = Hypervolume(y_cal_hv, [1,1,1])

    Parameters
    ----------
    y         : (N, M) 原始目标值
    ref_lo    : (M,)   归一化下界（理想点）
    ref_hi    : (M,)   归一化上界（天底点）
    ref_point : (M,)   超体积参考点（默认 [1,1,1]）
    """
    if ref_point is None:
        ref_point = np.ones(y.shape[1])
    denom  = np.where(ref_hi - ref_lo < 1e-12, 1.0, ref_hi - ref_lo)
    y_norm = (y - ref_lo) / denom
    return hypervolume(y_norm, ref_point)