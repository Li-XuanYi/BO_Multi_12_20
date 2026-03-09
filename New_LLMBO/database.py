"""
database.py — LLAMBO-MO 观测数据库
====================================
核心数据存储层, 为 GP / LLM-GP 耦合 / EI 采集函数 提供统一接口.

功能概览
--------
1. 存储每轮评估数据: θ=(I₁,SOC₁,I₂), f(θ)=(time,temp,aging), 约束, 来源
2. 维护 Pareto 前沿 & 超体积指标 (Hypervolume)
3. 为 GP 提供标准化训练数据 (X, Y)
4. 为 LLM 生成结构化上下文 (历史摘要 / top-k / Pareto)
5. 记录 EI / 采集函数值, 支持回溯分析
6. JSON 持久化 (save / load)

数据流
------
    Simulator ──evaluate──▶ database.add_observation(...)
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
          GP Training      LLM Context      EI History
        get_train_XY()   to_llm_context()  get_acq_history()
                │               │               │
                ▼               ▼               ▼
           GP.fit(X,Y)    LLM.generate()   Visualization
                │               │
                ▼               ▼
          acq(EI/EHVI)    candidate θ
                │               │
                └───────┬───────┘
                        ▼
              database.add_observation(...)  [下一轮]
"""

import json
import copy
import logging
import os
from datetime import datetime
from typing import (
    Dict, List, Optional, Tuple, Any, Union,
)

import numpy as np

logger = logging.getLogger(__name__)


# ================================================================
#  常量
# ================================================================
OBJECTIVE_NAMES  = ("time_s", "temp_K", "aging_pct")
OBJECTIVE_LABELS = ("充电时间 [s]", "峰值温度 [K]", "老化程度 [%]")
PARAM_NAMES      = ("I1", "SOC1", "I2")
NUM_OBJECTIVES   = 3
NUM_PARAMS       = 3

# 默认参考点 (用于超体积计算, 均 minimize → 参考点取各目标上界)
DEFAULT_REF_POINT = np.array([7200.0, 338.0, 0.5])

# 默认参数边界
DEFAULT_BOUNDS = {
    "I1":   (3.0, 7.0),
    "SOC1": (0.10, 0.70),
    "I2":   (1.0, 5.0),
}


# ================================================================
#  单条观测记录
# ================================================================
class Observation:
    """
    单条评估记录

    Attributes
    ----------
    theta        : np.ndarray, shape (3,)  [I₁, SOC₁, I₂]
    objectives   : np.ndarray, shape (3,)  [time_s, temp_K, aging_pct]
    feasible     : bool                    是否满足约束
    violation    : str | None              违规描述
    source       : str                     来源标识
                     "init"     — 初始采样 (LHS / Sobol)
                     "gp_ei"   — GP 采集函数 (EI / EHVI / ParEGO)
                     "llm"     — LLM 直接生成
                     "llm_gp"  — LLM-GP 耦合 (LLM 作为 GP 均值函数)
                     "random"  — 随机探索
                     "manual"  — 手动指定
    iteration    : int                     BO 迭代轮次 (0-indexed)
    acq_value    : float | None            采集函数值 (EI / EHVI / scalarized)
    acq_type     : str | None              采集函数类型
    gp_pred      : dict | None             GP 预测 {mean, std} per objective
    llm_rationale: str | None              LLM 生成理由
    details      : dict | None             仿真器返回的额外信息
    timestamp    : str                     ISO 时间戳
    """

    __slots__ = (
        "theta", "objectives", "feasible", "violation",
        "source", "iteration", "acq_value", "acq_type",
        "gp_pred", "llm_rationale", "details", "timestamp",
    )

    def __init__(
        self,
        theta:         np.ndarray,
        objectives:    np.ndarray,
        feasible:      bool          = True,
        violation:     Optional[str] = None,
        source:        str           = "init",
        iteration:     int           = 0,
        acq_value:     Optional[float] = None,
        acq_type:      Optional[str]   = None,
        gp_pred:       Optional[Dict]  = None,
        llm_rationale: Optional[str]   = None,
        details:       Optional[Dict]  = None,
        timestamp:     Optional[str]   = None,
    ):
        self.theta         = np.asarray(theta, dtype=float)
        self.objectives    = np.asarray(objectives, dtype=float)
        self.feasible      = feasible
        self.violation     = violation
        self.source        = source
        self.iteration     = iteration
        self.acq_value     = acq_value
        self.acq_type      = acq_type
        self.gp_pred       = gp_pred
        self.llm_rationale = llm_rationale
        self.details       = details
        self.timestamp     = timestamp or datetime.now().isoformat()

    # ---- 序列化 ----
    def to_dict(self) -> Dict:
        return {
            "theta":         self.theta.tolist(),
            "objectives":    self.objectives.tolist(),
            "feasible":      self.feasible,
            "violation":     self.violation,
            "source":        self.source,
            "iteration":     self.iteration,
            "acq_value":     self.acq_value,
            "acq_type":      self.acq_type,
            "gp_pred":       self.gp_pred,
            "llm_rationale": self.llm_rationale,
            "details":       self.details,
            "timestamp":     self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Observation":
        return cls(
            theta         = np.array(d["theta"]),
            objectives    = np.array(d["objectives"]),
            feasible      = d.get("feasible", True),
            violation     = d.get("violation"),
            source        = d.get("source", "init"),
            iteration     = d.get("iteration", 0),
            acq_value     = d.get("acq_value"),
            acq_type      = d.get("acq_type"),
            gp_pred       = d.get("gp_pred"),
            llm_rationale = d.get("llm_rationale"),
            details       = d.get("details"),
            timestamp     = d.get("timestamp"),
        )

    def __repr__(self) -> str:
        status = "✓" if self.feasible else "✗"
        return (
            f"Obs({status} iter={self.iteration} src={self.source} "
            f"θ=[{self.theta[0]:.2f},{self.theta[1]:.3f},{self.theta[2]:.2f}] "
            f"f=[{self.objectives[0]:.0f}s,{self.objectives[1]:.1f}K,"
            f"{self.objectives[2]:.4f}%])"
        )


# ================================================================
#  观测数据库
# ================================================================
class ObservationDB:
    """
    LLAMBO-MO 核心数据库

    负责管理所有评估记录, 并提供:
      - GP 训练数据接口
      - LLM 上下文生成接口
      - Pareto 前沿维护
      - 超体积 (HV) 追踪
      - EI / 采集函数历史

    Parameters
    ----------
    param_bounds : dict          参数边界 {name: (lo, hi)}
    ref_point    : np.ndarray    超体积参考点 (3,)
    normalize    : bool          GP 训练数据是否归一化
    """

    def __init__(
        self,
        param_bounds: Optional[Dict] = None,
        ref_point:    Optional[np.ndarray] = None,
        normalize:    bool = True,
    ):
        self._observations: List[Observation] = []
        self._pareto_indices: List[int] = []  # 非支配解在 _observations 中的下标

        self.param_bounds = param_bounds or copy.deepcopy(DEFAULT_BOUNDS)
        self.ref_point    = np.asarray(ref_point) if ref_point is not None \
                            else DEFAULT_REF_POINT.copy()
        self.normalize    = normalize

        # 迭代级别统计 (每轮 BO 的 HV / best objectives)
        self._iteration_stats: List[Dict] = []

        # ── Tchebycheff 上下文（由 optimizer 每迭代调用 update_tchebycheff_context 注入）
        self._w_vec: np.ndarray = np.array([1.0/3, 1.0/3, 1.0/3])
        self._z_star: np.ndarray = np.zeros(NUM_OBJECTIVES)   # 理想点（各目标历史最小值）
        self._prev_f_min: float = float("inf")                 # 上一轮最优 Tchebycheff 值
        self._f_min: float = float("inf")                      # 当前最优 Tchebycheff 值
        self._theta_best: Optional[np.ndarray] = None          # 当前最优 θ
        self._stagnation_count: int = 0                        # 连续未改进轮次

        logger.info(
            f"ObservationDB 初始化: bounds={self.param_bounds}, "
            f"ref_point={self.ref_point.tolist()}"
        )

    # ============================================================
    #  添加 / 查询观测
    # ============================================================
    def add_observation(
        self,
        theta:         np.ndarray,
        objectives:    np.ndarray,
        feasible:      bool          = True,
        violation:     Optional[str] = None,
        source:        str           = "init",
        iteration:     Optional[int] = None,
        acq_value:     Optional[float] = None,
        acq_type:      Optional[str]   = None,
        gp_pred:       Optional[Dict]  = None,
        llm_rationale: Optional[str]   = None,
        details:       Optional[Dict]  = None,
    ) -> int:
        """
        添加一条观测记录

        Returns
        -------
        int : 记录在数据库中的索引
        """
        if iteration is None:
            iteration = self.current_iteration

        obs = Observation(
            theta=theta, objectives=objectives,
            feasible=feasible, violation=violation,
            source=source, iteration=iteration,
            acq_value=acq_value, acq_type=acq_type,
            gp_pred=gp_pred, llm_rationale=llm_rationale,
            details=details,
        )
        idx = len(self._observations)
        self._observations.append(obs)

        # 更新 Pareto 前沿
        if feasible:
            self._update_pareto()

        logger.debug(f"添加观测 #{idx}: {obs}")

        # 新增：每次添加可行观测后同步更新最优值
        if feasible:
            self._recompute_best()

        return idx

    def add_from_simulator(
        self,
        theta:   np.ndarray,
        result:  Dict,
        source:  str           = "init",
        iteration: Optional[int] = None,
        acq_value: Optional[float] = None,
        acq_type:  Optional[str]   = None,
        gp_pred:   Optional[Dict]  = None,
        llm_rationale: Optional[str] = None,
    ) -> int:
        """
        从 PyBaMMSimulator.evaluate() 返回值直接添加

        Parameters
        ----------
        theta  : array-like (3,)
        result : dict  来自 simulator.evaluate(theta)
        """
        return self.add_observation(
            theta=np.asarray(theta),
            objectives=result["raw_objectives"],
            feasible=result["feasible"],
            violation=result.get("violation"),
            source=source,
            iteration=iteration,
            acq_value=acq_value,
            acq_type=acq_type,
            gp_pred=gp_pred,
            llm_rationale=llm_rationale,
            details=result.get("details"),
        )

    @property
    def size(self) -> int:
        return len(self._observations)

    @property
    def n_feasible(self) -> int:
        return sum(1 for o in self._observations if o.feasible)

    @property
    def current_iteration(self) -> int:
        if not self._observations:
            return 0
        return max(o.iteration for o in self._observations)

    def get_observation(self, idx: int) -> Observation:
        return self._observations[idx]

    def get_all(self) -> List[Observation]:
        return list(self._observations)

    def get_feasible(self) -> List[Observation]:
        return [o for o in self._observations if o.feasible]

    def get_by_iteration(self, iteration: int) -> List[Observation]:
        return [o for o in self._observations if o.iteration == iteration]

    def get_by_source(self, source: str) -> List[Observation]:
        return [o for o in self._observations if o.source == source]

    # ============================================================
    #  GP 训练数据接口
    # ============================================================
    def get_train_XY(
        self,
        feasible_only: bool = True,
        normalize_X:   bool = None,
        normalize_Y:   bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 GP 训练数据

        Parameters
        ----------
        feasible_only : bool   是否只用可行解
        normalize_X   : bool   是否对 X 归一化到 [0,1] (默认跟随 self.normalize)
        normalize_Y   : bool   是否对 Y 标准化 (zero-mean, unit-var)

        Returns
        -------
        X : np.ndarray, shape (n, 3)  参数矩阵
        Y : np.ndarray, shape (n, 3)  目标矩阵
        """
        if normalize_X is None:
            normalize_X = self.normalize

        obs = self.get_feasible() if feasible_only else self._observations
        if not obs:
            return np.empty((0, NUM_PARAMS)), np.empty((0, NUM_OBJECTIVES))

        X = np.array([o.theta for o in obs])       # (n, 3)
        Y = np.array([o.objectives for o in obs])   # (n, 3)

        if normalize_X:
            X = self._normalize_X(X)
        if normalize_Y:
            Y = self._standardize_Y(Y)

        return X, Y

    def get_train_XY_single(
        self,
        obj_index:     int,
        feasible_only: bool = True,
        normalize_X:   bool = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回单目标 GP 训练数据

        Parameters
        ----------
        obj_index : int  目标索引 (0=time, 1=temp, 2=aging)

        Returns
        -------
        X : (n, 3)
        y : (n, 1)
        """
        X, Y = self.get_train_XY(feasible_only, normalize_X, normalize_Y=False)
        if len(Y) == 0:
            return X, np.empty((0, 1))
        return X, Y[:, obj_index:obj_index+1]

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        """X 归一化到 [0, 1] (基于 param_bounds)"""
        bounds = self.param_bounds
        lo = np.array([bounds[p][0] for p in PARAM_NAMES])
        hi = np.array([bounds[p][1] for p in PARAM_NAMES])
        return (X - lo) / (hi - lo + 1e-12)

    def denormalize_X(self, X_norm: np.ndarray) -> np.ndarray:
        """[0, 1] → 原始尺度"""
        bounds = self.param_bounds
        lo = np.array([bounds[p][0] for p in PARAM_NAMES])
        hi = np.array([bounds[p][1] for p in PARAM_NAMES])
        return X_norm * (hi - lo) + lo

    def _standardize_Y(self, Y: np.ndarray) -> np.ndarray:
        """Y 标准化 (zero-mean, unit-var, per objective)"""
        if len(Y) < 2:
            return Y
        mu  = Y.mean(axis=0, keepdims=True)
        std = Y.std(axis=0, keepdims=True) + 1e-8
        return (Y - mu) / std

    def get_Y_stats(
        self, feasible_only: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        返回目标统计量 (用于 GP 标准化还原 / LLM 上下文)

        Returns
        -------
        dict with keys: mean, std, min, max, each shape (3,)
        """
        obs = self.get_feasible() if feasible_only else self._observations
        if not obs:
            return {
                "mean": np.zeros(NUM_OBJECTIVES),
                "std":  np.ones(NUM_OBJECTIVES),
                "min":  np.zeros(NUM_OBJECTIVES),
                "max":  np.ones(NUM_OBJECTIVES),
            }
        Y = np.array([o.objectives for o in obs])
        return {
            "mean": Y.mean(axis=0),
            "std":  Y.std(axis=0) + 1e-8,
            "min":  Y.min(axis=0),
            "max":  Y.max(axis=0),
        }

    # ============================================================
    #  Pareto 前沿
    # ============================================================
    def _update_pareto(self) -> None:
        """
        重新计算 Pareto 前沿 (仅可行解)

        使用非支配排序: 对于 minimize, 点 a 支配 b 当且仅当
        ∀i: a_i ≤ b_i  且  ∃j: a_j < b_j
        """
        feasible = [(i, o) for i, o in enumerate(self._observations) if o.feasible]
        if not feasible:
            self._pareto_indices = []
            return

        indices = [i for i, _ in feasible]
        objs    = np.array([o.objectives for _, o in feasible])
        n = len(objs)

        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # j 支配 i?
                if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    is_dominated[i] = True
                    break

        self._pareto_indices = [
            indices[i] for i in range(n) if not is_dominated[i]
        ]

    def get_pareto_front(self) -> List[Observation]:
        """返回当前 Pareto 前沿观测"""
        return [self._observations[i] for i in self._pareto_indices]

    def get_pareto_XY(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 Pareto 前沿的 (X, Y)"""
        pf = self.get_pareto_front()
        if not pf:
            return np.empty((0, NUM_PARAMS)), np.empty((0, NUM_OBJECTIVES))
        X = np.array([o.theta for o in pf])
        Y = np.array([o.objectives for o in pf])
        return X, Y

    @property
    def pareto_size(self) -> int:
        return len(self._pareto_indices)

    # ============================================================
    #  超体积 (Hypervolume)
    # ============================================================
    def compute_hypervolume(
        self,
        ref_point: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算当前 Pareto 前沿的超体积指标

        使用精确算法 (3 目标可在 O(n log n) 完成)
        此处用通用的 inclusion-exclusion / 切片法

        Parameters
        ----------
        ref_point : (3,)  参考点, 默认 self.ref_point

        Returns
        -------
        float : HV 值 (越大越好)
        """
        ref = ref_point if ref_point is not None else self.ref_point
        _, Y_pf = self.get_pareto_XY()

        if len(Y_pf) == 0:
            return 0.0

        # 过滤: 只保留被参考点支配的点
        mask = np.all(Y_pf < ref, axis=1)
        Y_pf = Y_pf[mask]
        if len(Y_pf) == 0:
            return 0.0

        # 3D 超体积: 按第一目标排序, 逐层切片
        return self._hv_3d(Y_pf, ref)

    @staticmethod
    def _hv_3d(points: np.ndarray, ref: np.ndarray) -> float:
        """
        3D 精确超体积 (HSO 切片法)

        对第一维排序后, 逐点在 2D 平面上做增量超体积
        时间复杂度 O(n² log n), 对 BO 规模足够高效
        """
        n = len(points)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - points[0]))

        # 按 obj[0] 降序排列 (从参考点方向开始切)
        sorted_idx = np.argsort(-points[:, 0])
        pts = points[sorted_idx]

        hv = 0.0
        # 维护 2D front (obj[1], obj[2])
        front_2d = []  # 按 obj[1] 升序, obj[2] 非支配

        for i in range(n):
            if i == n - 1:
                x_width = ref[0] - pts[i, 0]
            else:
                x_width = pts[i + 1, 0] - pts[i, 0]

            # 将当前点 (obj[1], obj[2]) 插入 2D front
            new_pt = (pts[i, 1], pts[i, 2])
            front_2d = _insert_2d_front(front_2d, new_pt)

            # 2D HV
            hv_2d = _compute_2d_hv(front_2d, ref[1], ref[2])
            hv += x_width * hv_2d

        return float(hv)

    # ============================================================
    #  EI / 采集函数历史
    # ============================================================
    def get_acq_history(self) -> List[Dict]:
        """
        返回所有带采集函数值的记录

        Returns
        -------
        list of dict, each: {
            iteration, theta, objectives, acq_value, acq_type, source
        }
        """
        records = []
        for o in self._observations:
            if o.acq_value is not None:
                records.append({
                    "iteration":  o.iteration,
                    "theta":      o.theta.tolist(),
                    "objectives": o.objectives.tolist(),
                    "acq_value":  o.acq_value,
                    "acq_type":   o.acq_type,
                    "source":     o.source,
                    "feasible":   o.feasible,
                })
        return records

    def get_acq_values_by_iteration(self) -> Dict[int, List[float]]:
        """按迭代轮次汇总采集函数值"""
        result: Dict[int, List[float]] = {}
        for o in self._observations:
            if o.acq_value is not None:
                result.setdefault(o.iteration, []).append(o.acq_value)
        return result

    # ============================================================
    #  迭代统计 (每轮 BO 结束后调用)
    # ============================================================
    def record_iteration_stats(self, extra: Optional[Dict] = None) -> Dict:
        """
        记录当前迭代的统计快照

        在每轮 BO 迭代末尾调用, 追踪优化进程

        Returns
        -------
        dict : 本轮统计
        """
        stats = {
            "iteration":    self.current_iteration,
            "n_total":      self.size,
            "n_feasible":   self.n_feasible,
            "pareto_size":  self.pareto_size,
            "hypervolume":  self.compute_hypervolume(),
        }

        # 各目标当前最优
        y_stats = self.get_Y_stats(feasible_only=True)
        for i, name in enumerate(OBJECTIVE_NAMES):
            stats[f"best_{name}"] = float(y_stats["min"][i])

        # 来源分布
        sources = {}
        for o in self.get_by_iteration(self.current_iteration):
            sources[o.source] = sources.get(o.source, 0) + 1
        stats["source_counts"] = sources

        if extra:
            stats.update(extra)

        self._iteration_stats.append(stats)
        logger.info(
            f"Iter {stats['iteration']}: "
            f"HV={stats['hypervolume']:.4f}  "
            f"|PF|={stats['pareto_size']}  "
            f"n={stats['n_total']}"
        )
        return stats

    def get_iteration_stats(self) -> List[Dict]:
        return list(self._iteration_stats)

    def get_hv_trace(self) -> np.ndarray:
        """返回超体积随迭代的变化, shape (n_iters,)"""
        return np.array([s["hypervolume"] for s in self._iteration_stats])

    # ============================================================
    #  LLM 上下文生成
    # ============================================================
    def to_llm_context(
        self,
        max_observations: int  = 20,
        include_pareto:   bool = True,
        include_top_k:    int  = 5,
        include_stats:    bool = True,
        include_recent:   int  = 5,
    ) -> str:
        """
        生成 LLM 提示上下文: 将历史数据序列化为结构化文本

        Parameters
        ----------
        max_observations : 最多包含的历史记录数
        include_pareto   : 是否包含 Pareto 前沿
        include_top_k    : 每个目标的 top-k 最佳点
        include_stats    : 是否包含统计摘要
        include_recent   : 最近 N 条记录

        Returns
        -------
        str : 可直接嵌入 LLM prompt 的上下文文本
        """
        lines = []
        lines.append("=== 充电协议优化历史 ===")
        lines.append(f"已评估: {self.size} 条 (可行: {self.n_feasible})")
        lines.append(f"决策变量: I₁∈[{self.param_bounds['I1'][0]},{self.param_bounds['I1'][1]}]A, "
                      f"SOC₁∈[{self.param_bounds['SOC1'][0]},{self.param_bounds['SOC1'][1]}], "
                      f"I₂∈[{self.param_bounds['I2'][0]},{self.param_bounds['I2'][1]}]A")
        lines.append(f"目标 (均 minimize): 充电时间[s], 峰值温度[K], 老化程度[%]")
        lines.append("")

        # ── 统计摘要 ──
        if include_stats and self.n_feasible > 0:
            stats = self.get_Y_stats(feasible_only=True)
            lines.append("--- 目标统计 ---")
            for i, (name, label) in enumerate(zip(OBJECTIVE_NAMES, OBJECTIVE_LABELS)):
                lines.append(
                    f"  {label}: min={stats['min'][i]:.4f}  "
                    f"max={stats['max'][i]:.4f}  mean={stats['mean'][i]:.4f}"
                )
            lines.append(f"  超体积 (HV): {self.compute_hypervolume():.6f}")
            lines.append("")

        # ── Pareto 前沿 ──
        if include_pareto:
            pf = self.get_pareto_front()
            lines.append(f"--- Pareto 前沿 ({len(pf)} 个非支配解) ---")
            for j, o in enumerate(pf):
                lines.append(
                    f"  PF[{j}]: I₁={o.theta[0]:.2f}A  SOC₁={o.theta[1]:.3f}  "
                    f"I₂={o.theta[2]:.2f}A  →  "
                    f"time={o.objectives[0]:.0f}s  temp={o.objectives[1]:.2f}K  "
                    f"aging={o.objectives[2]:.6f}%"
                )
            lines.append("")

        # ── 各目标 Top-K ──
        if include_top_k > 0 and self.n_feasible > 0:
            feasible = self.get_feasible()
            for i, (name, label) in enumerate(zip(OBJECTIVE_NAMES, OBJECTIVE_LABELS)):
                sorted_obs = sorted(feasible, key=lambda o: o.objectives[i])
                top = sorted_obs[:include_top_k]
                lines.append(f"--- {label} Top-{len(top)} ---")
                for j, o in enumerate(top):
                    lines.append(
                        f"  [{j+1}] I₁={o.theta[0]:.2f}  SOC₁={o.theta[1]:.3f}  "
                        f"I₂={o.theta[2]:.2f}  →  {name}={o.objectives[i]:.4f}  "
                        f"(src={o.source})"
                    )
            lines.append("")

        # ── 最近记录 ──
        if include_recent > 0:
            recent = self._observations[-include_recent:]
            lines.append(f"--- 最近 {len(recent)} 条记录 ---")
            for o in recent:
                status = "✓" if o.feasible else f"✗({o.violation})"
                acq_str = f"  acq={o.acq_value:.4f}" if o.acq_value is not None else ""
                lines.append(
                    f"  iter={o.iteration} src={o.source}  "
                    f"I₁={o.theta[0]:.2f}  SOC₁={o.theta[1]:.3f}  I₂={o.theta[2]:.2f}  →  "
                    f"[{o.objectives[0]:.0f}s, {o.objectives[1]:.2f}K, "
                    f"{o.objectives[2]:.6f}%]  {status}{acq_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_llm_candidates_prompt(
        self,
        n_candidates: int = 5,
        strategy:     str = "explore",
    ) -> str:
        """
        生成要求 LLM 提出新候选点的 prompt 片段

        Parameters
        ----------
        n_candidates : 要求 LLM 生成的候选数量
        strategy     : "explore" | "exploit" | "balanced"

        Returns
        -------
        str : prompt 片段
        """
        context = self.to_llm_context()

        strategy_desc = {
            "explore":  "侧重探索未知区域, 尝试与已有观测差异较大的参数组合",
            "exploit":  "侧重开发已知优良区域, 在 Pareto 前沿附近微调",
            "balanced": "平衡探索与开发, 兼顾未知区域和已知优良区域",
        }

        prompt = f"""{context}

=== 任务 ===
基于以上充电协议优化历史, 请提出 {n_candidates} 个新的候选充电协议参数.

策略: {strategy_desc.get(strategy, strategy)}

要求:
1. 每个候选给出 (I₁, SOC₁, I₂) 三个值
2. 参数须在约束范围内: I₁∈[{self.param_bounds['I1'][0]},{self.param_bounds['I1'][1]}], \
SOC₁∈[{self.param_bounds['SOC1'][0]},{self.param_bounds['SOC1'][1]}], \
I₂∈[{self.param_bounds['I2'][0]},{self.param_bounds['I2'][1]}]
3. 简要说明每个候选的设计理由 (基于物理直觉或数据趋势)
4. 以 JSON 格式返回:
   [{{"I1": ..., "SOC1": ..., "I2": ..., "rationale": "..."}}]
"""
        return prompt

    def to_gp_mean_prompt(
        self,
        theta_query: np.ndarray,
    ) -> str:
        """
        生成 LLM-as-GP-mean 的 prompt (Ψ(θ) 物理代理函数)

        让 LLM 基于物理直觉预测给定 θ 的目标值,
        作为 GP 的非零均值函数 m(x)

        Parameters
        ----------
        theta_query : (3,) or (n, 3)  待预测参数

        Returns
        -------
        str : prompt
        """
        theta_query = np.atleast_2d(theta_query)
        context = self.to_llm_context(
            max_observations=15,
            include_top_k=3,
            include_recent=3,
        )

        query_str = "\n".join(
            f"  θ[{i}]: I₁={t[0]:.3f}A, SOC₁={t[1]:.4f}, I₂={t[2]:.3f}A"
            for i, t in enumerate(theta_query)
        )

        prompt = f"""{context}

=== GP 均值函数预测任务 ===
请根据电化学物理知识和以上历史数据, 预测以下参数的三目标值:

{query_str}

物理提示:
- 充电时间 ≈ ΔSOC × Q_nom / I, 电流越大时间越短
- 温度升高 ∝ I²R (焦耳热), 高电流 → 高温
- 老化: SEI 主要时间依赖; 锂沉积在高电流/低温时加速
- SOC₁ 决定两阶段的分配比例

请以 JSON 格式返回预测:
[{{"time_s": ..., "temp_K": ..., "aging_pct": ...}}]
(每个查询点一个 dict, 顺序对应)
"""
        return prompt

    # ============================================================
    #  持久化
    # ============================================================
    def save(self, path: str) -> None:
        """保存到 JSON 文件"""
        data = {
            "version":         "1.0",
            "param_bounds":    self.param_bounds,
            "ref_point":       self.ref_point.tolist(),
            "normalize":       self.normalize,
            "observations":    [o.to_dict() for o in self._observations],
            "pareto_indices":  self._pareto_indices,
            "iteration_stats": self._iteration_stats,
            "saved_at":        datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"数据库已保存: {path} ({self.size} 条记录)")

    @classmethod
    def load(cls, path: str) -> "ObservationDB":
        """从 JSON 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        db = cls(
            param_bounds = data.get("param_bounds"),
            ref_point    = np.array(data["ref_point"]) if "ref_point" in data else None,
            normalize    = data.get("normalize", True),
        )
        for od in data.get("observations", []):
            db._observations.append(Observation.from_dict(od))
        db._pareto_indices  = data.get("pareto_indices", [])
        db._iteration_stats = data.get("iteration_stats", [])

        # 重新校验 Pareto
        db._update_pareto()
        logger.info(f"数据库已加载: {path} ({db.size} 条记录, |PF|={db.pareto_size})")
        return db

    # ============================================================
    #  便捷查询
    # ============================================================
    def get_best_per_objective(self) -> Dict[str, Observation]:
        """返回每个目标的最优可行解"""
        feasible = self.get_feasible()
        if not feasible:
            return {}
        result = {}
        for i, name in enumerate(OBJECTIVE_NAMES):
            best = min(feasible, key=lambda o: o.objectives[i])
            result[name] = best
        return result

    def get_gp_pred_errors(self) -> Optional[np.ndarray]:
        """
        返回有 GP 预测的点的预测误差 (actual - predicted)

        Returns
        -------
        np.ndarray, shape (n, 3) 或 None
        """
        errors = []
        for o in self._observations:
            if o.gp_pred is not None and "mean" in o.gp_pred:
                pred = np.array(o.gp_pred["mean"])
                err  = o.objectives - pred
                errors.append(err)
        return np.array(errors) if errors else None

    def summary(self) -> str:
        """打印友好的数据库摘要"""
        lines = [
            "=" * 50,
            "LLAMBO-MO ObservationDB Summary",
            "=" * 50,
            f"  总记录数:       {self.size}",
            f"  可行解:         {self.n_feasible}",
            f"  当前迭代:       {self.current_iteration}",
            f"  Pareto 前沿:    {self.pareto_size} 个非支配解",
            f"  超体积 (HV):    {self.compute_hypervolume():.6f}",
        ]

        # 来源分布
        src_counts: Dict[str, int] = {}
        for o in self._observations:
            src_counts[o.source] = src_counts.get(o.source, 0) + 1
        lines.append(f"  来源分布:       {src_counts}")

        # 各目标最优
        if self.n_feasible > 0:
            best = self.get_best_per_objective()
            lines.append("  目标最优:")
            for name, label in zip(OBJECTIVE_NAMES, OBJECTIVE_LABELS):
                o = best[name]
                lines.append(
                    f"    {label}: {o.objectives[list(OBJECTIVE_NAMES).index(name)]:.4f}"
                    f"  @ θ=[{o.theta[0]:.2f}, {o.theta[1]:.3f}, {o.theta[2]:.2f}]"
                )

        lines.append("=" * 50)
        return "\n".join(lines)

    # ============================================================
    #  DatabaseProtocol 接口（供 acquisition.py 使用）
    # ============================================================

    def update_tchebycheff_context(
        self,
        w_vec: np.ndarray,
        z_star: Optional[np.ndarray] = None,
    ) -> None:
        """
        每迭代由 optimizer.py 调用，注入当前 Tchebycheff 权重和理想点。

        必须在 af.step() 之前调用，使 get_f_min() / get_theta_best() 语义正确。
        """
        self._w_vec = np.asarray(w_vec, dtype=float).ravel()

        # 更新理想点：各目标历史最小值（可行解）
        if z_star is not None:
            self._z_star = np.asarray(z_star, dtype=float).ravel()
        else:
            feasible = self.get_feasible()
            if feasible:
                Y = np.array([o.objectives for o in feasible])
                self._z_star = Y.min(axis=0)

        # 重新计算当前最优
        self._recompute_best()

    def _recompute_best(self) -> None:
        """
        根据当前 _w_vec 和 _z_star 重新计算 f_min 和 theta_best。
        在 update_tchebycheff_context 和 add_observation 后触发。
        """
        feasible = self.get_feasible()
        if not feasible:
            return

        best_tch = float("inf")
        best_theta = None

        for obs in feasible:
            diff = np.abs(obs.objectives - self._z_star)
            tch = float(np.max(self._w_vec * diff))
            if tch < best_tch:
                best_tch = tch
                best_theta = obs.theta.copy()

        self._prev_f_min = self._f_min
        self._f_min = best_tch
        self._theta_best = best_theta

        # 停滞判断：f_min 改进超过 1e-8 才算有效改进
        if self._prev_f_min - self._f_min > 1e-8:
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

    def get_f_min(self) -> float:
        """返回当前 Tchebycheff 标量最优值，供 acquisition.py EI 计算使用。"""
        return self._f_min

    def get_theta_best(self) -> np.ndarray:
        """返回当前最优 Tchebycheff 标量对应的决策向量 θ。"""
        if self._theta_best is None:
            lo = np.array([v[0] for v in self.param_bounds.values()])
            hi = np.array([v[1] for v in self.param_bounds.values()])
            return (lo + hi) / 2.0
        return self._theta_best.copy()

    def has_improved(self) -> bool:
        """若上次 update_tchebycheff_context 后 f_min 有改进则返回 True。"""
        return self._stagnation_count == 0

    def get_stagnation_count(self) -> int:
        """返回连续未改进的迭代次数。"""
        return self._stagnation_count

    def __repr__(self) -> str:
        return (
            f"ObservationDB(n={self.size}, feasible={self.n_feasible}, "
            f"|PF|={self.pareto_size}, HV={self.compute_hypervolume():.4f})"
        )


# ================================================================
#  辅助: 2D 超体积工具函数
# ================================================================
def _insert_2d_front(
    front: List[Tuple[float, float]],
    pt:    Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    向 2D 非支配集插入一个点, 移除被支配的旧点

    front 按 obj[0] 升序排列, 且保证非支配
    """
    new_front = []
    inserted = False
    dominated = False

    for fp in front:
        # fp 支配 pt?
        if fp[0] <= pt[0] and fp[1] <= pt[1]:
            if fp[0] < pt[0] or fp[1] < pt[1]:
                dominated = True
                new_front.append(fp)
                continue
        # pt 支配 fp? → 跳过 fp
        if pt[0] <= fp[0] and pt[1] <= fp[1]:
            if pt[0] < fp[0] or pt[1] < fp[1]:
                continue
        # 无支配关系
        if not inserted and pt[0] <= fp[0]:
            new_front.append(pt)
            inserted = True
        new_front.append(fp)

    if dominated:
        return front  # pt 被支配, 不加入
    if not inserted:
        new_front.append(pt)
    return new_front


def _compute_2d_hv(
    front: List[Tuple[float, float]],
    ref_y: float,
    ref_z: float,
) -> float:
    """
    计算 2D 非支配集相对于 (ref_y, ref_z) 的超面积

    front 按 obj[0] 升序排列
    """
    if not front:
        return 0.0

    # 按 y 排序
    pts = sorted(front, key=lambda p: p[0])
    hv = 0.0
    prev_z = ref_z

    for y, z in pts:
        if y >= ref_y or z >= ref_z:
            continue
        width = ref_y - y  # 但需要用阶梯法
        # 阶梯法: 从 ref_z 方向切
        pass

    # 用标准阶梯法重算
    pts_valid = [(y, z) for y, z in pts if y < ref_y and z < ref_z]
    if not pts_valid:
        return 0.0

    # 按 y 升序 → z 应该降序 (非支配)
    pts_valid.sort(key=lambda p: p[0])
    hv = 0.0
    prev_z_bound = ref_z
    for i, (y, z) in enumerate(pts_valid):
        if z < prev_z_bound:
            if i + 1 < len(pts_valid):
                y_width = pts_valid[i + 1][0] - y
            else:
                y_width = ref_y - y
            hv += y_width * (prev_z_bound - z)
            prev_z_bound = z

    return hv


# ================================================================
#  命令行测试
# ================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("LLAMBO-MO ObservationDB 单元测试")
    print("=" * 60)

    db = ObservationDB()

    # 模拟一些观测数据
    test_data = [
        # theta=(I1, SOC1, I2),   objectives=(time, temp, aging),  source
        ([3.5, 0.40, 2.0],  [2800, 305.0, 0.0012],  "init"),
        ([5.0, 0.35, 2.5],  [2100, 312.0, 0.0035],  "init"),
        ([7.0, 0.20, 4.0],  [1200, 322.0, 0.0150],  "init"),
        ([6.0, 0.30, 3.0],  [1600, 318.0, 0.0080],  "gp_ei"),
        ([4.5, 0.45, 2.0],  [2500, 308.0, 0.0020],  "llm"),
        ([6.5, 0.25, 3.5],  [1400, 320.0, 0.0100],  "llm_gp"),
        # 一条不可行解
        ([7.0, 0.15, 5.0],  [7200, 338.0, 0.5000],  "gp_ei"),
    ]

    for i, (theta, obj, src) in enumerate(test_data):
        feasible = (obj[0] < 7200)
        db.add_observation(
            theta=np.array(theta),
            objectives=np.array(obj),
            feasible=feasible,
            violation=None if feasible else "penalty",
            source=src,
            iteration=i // 3,
            acq_value=np.random.rand() if src in ("gp_ei", "llm_gp") else None,
            acq_type="EHVI" if src == "gp_ei" else ("LLM-EI" if src == "llm_gp" else None),
        )

    # 摘要
    print(db.summary())

    # GP 训练数据
    X, Y = db.get_train_XY(feasible_only=True, normalize_X=True)
    print(f"\nGP 训练数据: X.shape={X.shape}, Y.shape={Y.shape}")
    print(f"  X[0] (归一化): {X[0]}")
    print(f"  Y[0]: {Y[0]}")

    # Pareto 前沿
    pf = db.get_pareto_front()
    print(f"\nPareto 前沿 ({len(pf)} 个):")
    for o in pf:
        print(f"  {o}")

    # HV
    hv = db.compute_hypervolume()
    print(f"\n超体积: {hv:.6f}")

    # LLM 上下文
    ctx = db.to_llm_context()
    print(f"\nLLM 上下文 ({len(ctx)} 字符):")
    print(ctx[:500] + "..." if len(ctx) > 500 else ctx)

    # 迭代统计
    db.record_iteration_stats()

    # 保存 / 加载
    db.save("/tmp/test_llambo_db.json")
    db2 = ObservationDB.load("/tmp/test_llambo_db.json")
    print(f"\n重载后: {db2}")

    # GP mean prompt
    prompt = db.to_gp_mean_prompt(np.array([5.5, 0.30, 3.0]))
    print(f"\nGP Mean Prompt ({len(prompt)} 字符):")
    print(prompt[:300] + "...")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
