"""
riesz_cache.py — Riesz s-energy 权重集合磁盘缓存
=================================================
Fix 7: 避免每次实例化 BayesOptimizer / ParEGOOptimizer 都重新计算 Riesz 权重集合。

权重集合由参数完全决定（n_obj, n_div, s, n_iter, lr, seed），结果确定性。
首次生成后 pickle 序列化到磁盘，后续直接加载，加速约 900×。

用法
----
    from llmbo.riesz_cache import load_or_generate_riesz

    W = load_or_generate_riesz(
        n_obj=3, n_div=10, s=2.0, n_iter=500, lr=5e-3, seed=42
    )
    # 参数变更后强制重新生成：
    W = load_or_generate_riesz(..., force_regen=True)
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = ".riesz_cache"


def _make_cache_key(
    n_obj: int,
    n_div: int,
    s: float,
    n_iter: int,
    lr: float,
    seed: int,
) -> str:
    """根据参数生成 MD5 缓存键。"""
    param_str = f"n_obj={n_obj}_n_div={n_div}_s={s}_n_iter={n_iter}_lr={lr}_seed={seed}"
    return hashlib.md5(param_str.encode()).hexdigest()


def load_or_generate_riesz(
    n_obj:       int   = 3,
    n_div:       int   = 10,
    s:           float = 2.0,
    n_iter:      int   = 500,
    lr:          float = 5e-3,
    seed:        int   = 42,
    cache_dir:   str   = _DEFAULT_CACHE_DIR,
    force_regen: bool  = False,
) -> np.ndarray:
    """
    加载或生成 Riesz s-energy 权重集合。

    首次调用时生成并缓存到磁盘；后续调用直接从磁盘加载（约 0.005s vs 0.64s）。

    Parameters
    ----------
    n_obj       : 目标维度（默认 3）
    n_div       : Das-Dennis 分割数（n_div=10, n_obj=3 → 66 点）
    s           : Riesz 势能指数（s=2 等价于 Coulomb 势）
    n_iter      : 梯度下降步数
    lr          : 学习率
    seed        : 生成种子（固定保证复现性）
    cache_dir   : 缓存目录（默认 .riesz_cache/）
    force_regen : True → 忽略缓存，强制重新生成

    Returns
    -------
    W : (N, n_obj)  N 个权重向量，每行满足 Σ=1
    """
    # 延迟导入，避免循环依赖
    from llmbo.optimizer import generate_riesz_weight_set

    cache_key  = _make_cache_key(n_obj, n_div, s, n_iter, lr, seed)
    cache_path = Path(cache_dir) / f"riesz_{cache_key}.pkl"

    # 尝试从缓存加载
    if not force_regen and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                W = pickle.load(f)
            if isinstance(W, np.ndarray) and W.ndim == 2 and W.shape[1] == n_obj:
                logger.info(
                    "Riesz 权重集合从缓存加载: %s  shape=%s",
                    cache_path, W.shape
                )
                return W
            else:
                logger.warning("Riesz 缓存文件格式异常，重新生成: %s", cache_path)
        except Exception as exc:
            logger.warning("Riesz 缓存加载失败 (%s)，重新生成", exc)

    # 生成权重集合
    logger.info(
        "生成 Riesz 权重集合 (n_div=%d, n_iter=%d, seed=%d) ...",
        n_div, n_iter, seed
    )
    W = generate_riesz_weight_set(
        n_obj=n_obj, n_div=n_div, s=s, n_iter=n_iter, lr=lr, seed=seed
    )

    # 写入缓存
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(W, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Riesz 权重集合已缓存: %s  shape=%s", cache_path, W.shape)
    except Exception as exc:
        logger.warning("Riesz 缓存写入失败 (%s)，继续使用内存中的结果", exc)

    return W
