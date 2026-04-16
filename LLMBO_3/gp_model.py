"""
gp_model.py  —  GPy 高斯过程代理模型封装
==========================================
使用 GPy ARD Matérn 5/2 核，支持用 LLM 提供的
length_scales 热启动超参，预留 Phase 2 复合核接口。


"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib
    if not hasattr(matplotlib, "numpy"):
        matplotlib.numpy = np
    import GPy
    GPY_AVAILABLE = True
except Exception as e:
    GPY_AVAILABLE = False
    raise ImportError(
        f"无法导入 GPy；请确认当前解释器安装了可用的 GPy，并检查 GPy/Matplotlib 兼容性。原始错误: {e}"
    )
    raise ImportError("请安装 GPy: pip install GPy")

logger = logging.getLogger(__name__)


class GPModel:
    """
    ARD Matérn 5/2 高斯过程代理模型。

    Parameters
    ----------
    dim            : 决策空间维度（5）
    length_scales  : LLM 提供的初始长度尺度 (dim,)，None 则使用默认值 1.0
    noise_var      : 初始噪声方差
    n_restarts     : 超参优化重启次数（越多越准但越慢）
    """

    def __init__(
        self,
        dim: int = 5,
        length_scales: Optional[np.ndarray] = None,
        noise_var: float = 1e-3,
        n_restarts: int = 3,
    ) -> None:
        assert GPY_AVAILABLE, "GPy 未安装"
        self.dim           = dim
        self.noise_var     = noise_var
        self.n_restarts    = n_restarts
        self.model: Optional[GPy.models.GPRegression] = None

        # LLM 提供的初始长度尺度（Phase 2 复合核也从这里读取）
        if length_scales is not None:
            self.length_scales = np.asarray(length_scales, dtype=float).reshape(-1)
            assert len(self.length_scales) == dim, \
                f"length_scales 维度 {len(self.length_scales)} != dim {dim}"
        else:
            self.length_scales = np.ones(dim)

    # ------------------------------------------------------------------
    #  训练
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合 GP 模型。

        Parameters
        ----------
        X : (N, dim) 归一化决策向量
        y : (N,)     标量化目标值（来自 scalarize 的 y_agg_norm）
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).reshape(-1, 1)
        assert X.shape[1] == self.dim, f"X 维度错误: {X.shape[1]} != {self.dim}"

        # ── 构建核函数（ARD Matérn 5/2）──────────────────────────────
        # 对标 MATLAB fitrgp 默认使用 ARD 核
        # Phase 2 在此处替换为复合核：k_base + gamma_t * coupling_term
        kernel = GPy.kern.Matern52(
            input_dim=self.dim,
            ARD=True,                        # 每维独立长度尺度
            lengthscale=self.length_scales,  # LLM 热启动
        )

        # ── 构建 GP 回归模型 ──────────────────────────────────────────
        self.model = GPy.models.GPRegression(
            X, y,
            kernel=kernel,
            noise_var=self.noise_var,
        )

        # 约束超参范围，防止优化发散
        self.model.kern.lengthscale.constrain_bounded(1e-3, 10.0, warning=False)
        self.model.kern.variance.constrain_bounded(1e-3, 100.0, warning=False)
        self.model.likelihood.variance.constrain_bounded(1e-6, 1.0, warning=False)

        # ── 超参优化 ─────────────────────────────────────────────────
        try:
            self.model.optimize_restarts(
                num_restarts=self.n_restarts,
                verbose=False,
                robust=True,       # 忽略优化中的数值错误
                messages=False,
            )
        except Exception as e:
            logger.warning(f"GP 超参优化失败，使用当前值: {e}")

        logger.debug(
            f"GP 拟合完成 | N={X.shape[0]} | "
            f"lengthscale={self.model.kern.lengthscale.values} | "
            f"noise={self.model.likelihood.variance.values[0]:.2e}"
        )

    # ------------------------------------------------------------------
    #  预测
    # ------------------------------------------------------------------

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        后验预测。

        Parameters
        ----------
        X_new : (M, dim) 候选点

        Returns
        -------
        mu  : (M,) 后验均值
        std : (M,) 后验标准差（注意：是 std，不是 var）
        """
        assert self.model is not None, "请先调用 fit()"
        X_new = np.atleast_2d(X_new)
        mu, var = self.model.predict(X_new)
        std = np.sqrt(np.maximum(var, 1e-12))
        return mu.reshape(-1), std.reshape(-1)

    # ------------------------------------------------------------------
    #  Phase 2 接口预留
    # ------------------------------------------------------------------

    def set_composite_kernel(
        self,
        coupling_matrix: np.ndarray,
        gamma_t: float,
    ) -> None:
        """
        Phase 2：注入 LLM 物理先验的复合核。
        当前版本为占位符，后续替换 self.model 的 kern。

        k(θ,θ') = k_base(θ,θ'; L) + γ_t · Σ_ij W_ij ∇_i f ∇_j f'
        """
        raise NotImplementedError(
            "复合核为 Phase 2 功能，当前使用标准 ARD Matérn 5/2"
        )
