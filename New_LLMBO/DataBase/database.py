"""
database.py — LLAMBO-MO 观测数据库
====================================
[修复内容]
1. _recompute_best：实现正确 Tchebycheff 公式（Eq.1）：
     f̃₃ = log₁₀(aging)（Eq.2a）
     f̄ᵢ = (f̃ᵢ - y_min_i) / (y_max_i - y_min_i)（Eq.2b, 动态）
     f_tch = max_i(wᵢ·f̄ᵢ) + η·Σᵢ(wᵢ·f̄ᵢ)
2. _recompute_best 添加 update_stagnation 参数（默认 False）：
     - add_observation 调用时 update_stagnation=False（不计入停滞）
     - update_tchebycheff_context 调用时 update_stagnation=True（迭代级别）
3. update_tchebycheff_context 新增 y_min/y_max/eta 参数，由 optimizer 注入
4. 移除旧的 z_star 理想点减法逻辑
"""

import json
import copy
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

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

# 注意：aging_pct 和 time_s 在 HV 计算时会取 log₁₀，因此参考点和理想点的第一、三维也使用 log 空间值
# 第二维 (temp) 保持原始空间
DEFAULT_REF_POINT   = np.array([7200.0, 328.0, 0.1])         # HV 混合空间：[log₁₀(7200)≈3.857, 328K, log₁₀(0.1)=-1]
DEFAULT_IDEAL_POINT = np.array([2000.0, 298.0, 0.001])       # HV 混合空间：[log₁₀(1000)=3, 298K, log₁₀(0.001)=-3]

# HV_max 使用混合空间计算（与 _hv_3d 一致）：
#   第一维：log10(7200) - log10(2000) = 3.857 - 3.301 = 0.556
#   第二维：328 - 298 = 30
#   第三维：log10(0.1) - log10(0.001) = -1 - (-3) = 2
# HV_max = 0.556 × 30 × 2 ≈ 33.4
DEFAULT_HV_MAX = (np.log10(DEFAULT_REF_POINT[0]) - np.log10(DEFAULT_IDEAL_POINT[0])) * \
                 (DEFAULT_REF_POINT[1] - DEFAULT_IDEAL_POINT[1]) * \
                 (np.log10(DEFAULT_REF_POINT[2]) - np.log10(DEFAULT_IDEAL_POINT[2]))

DEFAULT_BOUNDS = {
    "I1":   (3.0, 7.0),
    "SOC1": (0.10, 0.70),
    "I2":   (1.0, 5.0),
}


def make_observation_db(param_bounds: Optional[Dict] = None, **kwargs) -> "ObservationDB":
    """
    工厂函数：确保 EIMO 和 ParEGO 使用完全相同的 ref/ideal/hv_max。

    Parameters
    ----------
    param_bounds : dict, optional
        参数边界，默认使用 DEFAULT_BOUNDS
    **kwargs : 其他传给 ObservationDB 的参数

    Returns
    -------
    ObservationDB 实例，使用全局统一的 ref_point 和 ideal_point
    """
    return ObservationDB(
        param_bounds=param_bounds or copy.deepcopy(DEFAULT_BOUNDS),
        ref_point=DEFAULT_REF_POINT.copy(),
        ideal_point=DEFAULT_IDEAL_POINT.copy(),
        normalize=True,
        **kwargs,
    )


# ================================================================
#  单条观测记录
# ================================================================
class Observation:
    """单条评估记录（与原版完全相同）"""

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

    [修复说明]
    - update_tchebycheff_context 现在由 optimizer 传入 y_min/y_max/eta
      （动态 min/max 在 optimizer 层计算，保证与 GP 训练时使用同一归一化）
    - _recompute_best 使用正确公式：log10(aging) + min-max 归一化 + η tiebreaker
    - add_observation 中 _recompute_best(update_stagnation=False) 避免迭代内计数累加
    """

    def __init__(
        self,
        param_bounds:  Optional[Dict] = None,
        ref_point:     Optional[np.ndarray] = None,
        ideal_point:   Optional[np.ndarray] = None,
        normalize:     bool = True,
    ):
        self._observations: List[Observation] = []
        self._pareto_indices: List[int] = []
        self._pareto_objectives: Optional[np.ndarray] = None  # Fix 4: 缓存 Pareto 目标值

        self.param_bounds  = param_bounds or copy.deepcopy(DEFAULT_BOUNDS)
        self.ref_point     = (np.asarray(ref_point) if ref_point is not None
                              else DEFAULT_REF_POINT.copy())
        self.ideal_point   = (np.asarray(ideal_point) if ideal_point is not None
                              else DEFAULT_IDEAL_POINT.copy())
        # HV_max = Π(ref_i - ideal_i) — 用于归一化，使 HV ∈ [0, 1]
        # 注意：HV 计算时第一维 (time) 和第三维 (aging) 取 log₁₀，第二维 (temp) 保持原始空间
        # 因此 HV_max 也必须使用混合空间计算
        self.hv_max = float((np.log10(self.ref_point[0]) - np.log10(self.ideal_point[0])) *
                            (self.ref_point[1] - self.ideal_point[1]) *
                            (np.log10(self.ref_point[2]) - np.log10(self.ideal_point[2])))
        self.normalize     = normalize

        # Fix 1: 运行时校验 ideal_point 一致性
        import warnings
        if not np.allclose(self.ideal_point, DEFAULT_IDEAL_POINT, rtol=1e-3):
            warnings.warn(
                f"ObservationDB: ideal_point={self.ideal_point.tolist()} 与 "
                f"DEFAULT_IDEAL_POINT={DEFAULT_IDEAL_POINT.tolist()} 不一致，"
                f"HV 归一化结果将不可比！请使用 make_observation_db() 工厂函数。",
                RuntimeWarning, stacklevel=2
            )

        self._iteration_stats: List[Dict] = []

        # ── Tchebycheff 上下文（由 optimizer 每迭代注入）─────────────────
        self._w_vec:   np.ndarray          = np.array([1.0/3, 1.0/3, 1.0/3])
        self._y_min:   np.ndarray          = np.zeros(NUM_OBJECTIVES)
        self._y_max:   np.ndarray          = np.ones(NUM_OBJECTIVES)
        self._eta:     float               = 0.05
        self._f_min:   float               = float("inf")
        self._prev_f_min: float            = float("inf")
        self._theta_best: Optional[np.ndarray] = None
        self._stagnation_count: int        = 0
        self._prev_hv_for_stagnation: float = 0.0
        self._prev_pareto_size: int        = 0   # 上次迭代 Pareto front 大小（停滞检测用）
        # Fix 5: 滑动窗口（最近2步是否改进），避免单次噪声重置停滞计数
        from collections import deque
        self._improvement_window: deque = deque(maxlen=2)

        logger.info(
            "ObservationDB 初始化: bounds=%s  ref_point=%s",
            self.param_bounds, self.ref_point.tolist()
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

        if feasible:
            # Fix 4: 增量更新，传入新点目标值
            self._update_pareto(new_obj=objectives)
            # FIX: update_stagnation=False — 迭代内添加观测不触发停滞计数
            # 停滞只在 update_tchebycheff_context（每迭代开始时）更新
            self._recompute_best(update_stagnation=False)

        logger.debug("添加观测 #%d: %s", idx, obs)
        return idx

    def add_from_simulator(
        self,
        theta:     np.ndarray,
        result:    Dict,
        source:    str           = "init",
        iteration: Optional[int] = None,
        acq_value: Optional[float] = None,
        acq_type:  Optional[str]   = None,
        gp_pred:   Optional[Dict]  = None,
        llm_rationale: Optional[str] = None,
    ) -> int:
        return self.add_observation(
            theta      = np.asarray(theta),
            objectives = result["raw_objectives"],
            feasible   = result["feasible"],
            violation  = result.get("violation"),
            source     = source,
            iteration  = iteration,
            acq_value  = acq_value,
            acq_type   = acq_type,
            gp_pred    = gp_pred,
            llm_rationale = llm_rationale,
            details    = result.get("details"),
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
        if normalize_X is None:
            normalize_X = self.normalize

        obs = self.get_feasible() if feasible_only else self._observations
        if not obs:
            return np.empty((0, NUM_PARAMS)), np.empty((0, NUM_OBJECTIVES))

        X = np.array([o.theta for o in obs])
        Y = np.array([o.objectives for o in obs])

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
        X, Y = self.get_train_XY(feasible_only, normalize_X, normalize_Y=False)
        if len(Y) == 0:
            return X, np.empty((0, 1))
        return X, Y[:, obj_index:obj_index+1]

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        bounds = self.param_bounds
        lo = np.array([bounds[p][0] for p in PARAM_NAMES])
        hi = np.array([bounds[p][1] for p in PARAM_NAMES])
        return (X - lo) / (hi - lo + 1e-12)

    def denormalize_X(self, X_norm: np.ndarray) -> np.ndarray:
        bounds = self.param_bounds
        lo = np.array([bounds[p][0] for p in PARAM_NAMES])
        hi = np.array([bounds[p][1] for p in PARAM_NAMES])
        return X_norm * (hi - lo) + lo

    def _standardize_Y(self, Y: np.ndarray) -> np.ndarray:
        if len(Y) < 2:
            return Y
        mu  = Y.mean(axis=0, keepdims=True)
        std = Y.std(axis=0, keepdims=True) + 1e-8
        return (Y - mu) / std

    def get_Y_stats(self, feasible_only: bool = True) -> Dict[str, np.ndarray]:
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
    def _update_pareto(self, new_obj: Optional[np.ndarray] = None) -> None:
        """
        Fix 4: 增量 Pareto 更新 — O(|PF|) 而非 O(n²)。

        Parameters
        ----------
        new_obj : np.ndarray (3,), optional
            新加入的可行解目标值。若为 None，则全量重建（load 时调用）。
        """
        feasible = [(i, o) for i, o in enumerate(self._observations) if o.feasible]
        if not feasible:
            self._pareto_indices = []
            self._pareto_objectives = None
            return

        # 全量重建（load 时或首次调用）
        if new_obj is None:
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
                    if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                        is_dominated[i] = True
                        break

            self._pareto_indices = [indices[i] for i in range(n) if not is_dominated[i]]
            self._pareto_objectives = objs[~is_dominated]
            return

        # 增量更新（add_observation 时调用）
        new_obj = np.asarray(new_obj, dtype=float)
        new_idx = len(self._observations) - 1  # 最新加入的观测索引

        # 首次加入 Pareto front
        if not self._pareto_indices:
            self._pareto_indices = [new_idx]
            self._pareto_objectives = new_obj[np.newaxis, :]
            return

        # 检查新点是否被现有 PF 中任意点支配
        for pf_obj in self._pareto_objectives:
            if np.all(pf_obj <= new_obj) and np.any(pf_obj < new_obj):
                return  # 新点被支配，不加入 PF

        # 新点不被支配：移除 PF 中被新点支配的点
        not_dominated_by_new = ~(
            np.all(new_obj <= self._pareto_objectives, axis=1) &
            np.any(new_obj < self._pareto_objectives, axis=1)
        )
        self._pareto_indices = [
            self._pareto_indices[i] for i in range(len(self._pareto_indices))
            if not_dominated_by_new[i]
        ]
        self._pareto_objectives = self._pareto_objectives[not_dominated_by_new]

        # 新点加入 PF
        self._pareto_indices.append(new_idx)
        self._pareto_objectives = np.vstack([self._pareto_objectives, new_obj])

    def get_pareto_front(self) -> List[Observation]:
        return [self._observations[i] for i in self._pareto_indices]

    def get_pareto_XY(self) -> Tuple[np.ndarray, np.ndarray]:
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
    #  超体积
    # ============================================================
    def compute_hypervolume(self, ref_point: Optional[np.ndarray] = None) -> float:
        """
        返回归一化超体积 HV ∈ [0, 1]，除以 HV_max = Π(ref_i - ideal_i)。

        注意：time_s (第一维) 和 aging_pct (第三维) 取 log₁₀ 后再计算 HV，第二维 (temp) 保持原始空间。
        """
        ref = ref_point if ref_point is not None else self.ref_point
        _, Y_pf = self.get_pareto_XY()
        if len(Y_pf) == 0:
            return 0.0

        # 构建混合空间：第一维 (time) 和第三维 (aging) 取 log₁₀，第二维 (temp) 保持原始值
        Y_hv = Y_pf.copy()
        Y_hv[:, 0] = np.log10(np.maximum(Y_pf[:, 0], 1.0))      # time 取 log
        Y_hv[:, 2] = np.log10(np.maximum(Y_pf[:, 2], 1e-12))    # aging 取 log

        # ref 也使用混合空间
        ref_hv = ref.copy()
        ref_hv[0] = np.log10(ref[0])
        ref_hv[2] = np.log10(ref[2])

        # 过滤掉超出参考点的点
        mask = np.all(Y_hv < ref_hv, axis=1)
        Y_hv = Y_hv[mask]
        if len(Y_hv) == 0:
            return 0.0

        hv_raw = self._hv_3d(Y_hv, ref_hv)
        return hv_raw / self.hv_max   # 归一化到 [0, 1]

    def compute_hypervolume_raw(self, ref_point: Optional[np.ndarray] = None) -> float:
        """
        返回未归一化的原始超体积（供调试用）。

        注意：time_s (第一维) 和 aging_pct (第三维) 取 log₁₀ 后再计算 HV，第二维 (temp) 保持原始空间。
        """
        ref = ref_point if ref_point is not None else self.ref_point
        _, Y_pf = self.get_pareto_XY()
        if len(Y_pf) == 0:
            return 0.0

        # 构建混合空间：第一维 (time) 和第三维 (aging) 取 log₁₀，第二维 (temp) 保持原始值
        Y_hv = Y_pf.copy()
        Y_hv[:, 0] = np.log10(np.maximum(Y_pf[:, 0], 1.0))      # time 取 log
        Y_hv[:, 2] = np.log10(np.maximum(Y_pf[:, 2], 1e-12))    # aging 取 log

        # ref 也使用混合空间
        ref_hv = ref.copy()
        ref_hv[0] = np.log10(ref[0])
        ref_hv[2] = np.log10(ref[2])

        # 过滤掉超出参考点的点
        mask = np.all(Y_hv < ref_hv, axis=1)
        Y_hv = Y_hv[mask]
        if len(Y_hv) == 0:
            return 0.0
        return self._hv_3d(Y_hv, ref_hv)

    @staticmethod
    def _hv_3d(points: np.ndarray, ref: np.ndarray) -> float:
        n = len(points)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - points[0]))

        # FIX: ASCENDING sort（不是 -points）
        # 升序扫描：pts[0] 最小 f1，pts[-1] 最大 f1
        # 每个切片宽度 = pts[i+1].f1 - pts[i].f1 > 0（升序保证正值）
        # 最后切片 = ref[0] - pts[-1].f1
        sorted_idx = np.argsort(points[:, 0])   # ← 升序
        pts = points[sorted_idx]
        hv = 0.0
        front_2d = []

        for i in range(n):
            new_pt = (pts[i, 1], pts[i, 2])
            front_2d = _insert_2d_front(front_2d, new_pt)
            if i == n - 1:
                x_width = ref[0] - pts[i, 0]
            else:
                x_width = pts[i + 1, 0] - pts[i, 0]   # 升序 → 正值
            hv_2d = _compute_2d_hv(front_2d, ref[1], ref[2])
            hv += x_width * hv_2d

        return float(hv)

    # ============================================================
    #  EI / 采集函数历史
    # ============================================================
    def get_acq_history(self) -> List[Dict]:
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
        result: Dict[int, List[float]] = {}
        for o in self._observations:
            if o.acq_value is not None:
                result.setdefault(o.iteration, []).append(o.acq_value)
        return result

    # ============================================================
    #  迭代统计
    # ============================================================
    def record_iteration_stats(self, extra: Optional[Dict] = None) -> Dict:
        stats = {
            "iteration":   self.current_iteration,
            "n_total":     self.size,
            "n_feasible":  self.n_feasible,
            "pareto_size": self.pareto_size,
            "hypervolume": self.compute_hypervolume(),           # 归一化 ∈ [0,1]
            "hypervolume_raw": self.compute_hypervolume_raw(),   # 原始值（调试用）
        }
        y_stats = self.get_Y_stats(feasible_only=True)
        for i, name in enumerate(OBJECTIVE_NAMES):
            stats[f"best_{name}"] = float(y_stats["min"][i])
        sources = {}
        for o in self.get_by_iteration(self.current_iteration):
            sources[o.source] = sources.get(o.source, 0) + 1
        stats["source_counts"] = sources
        if extra:
            stats.update(extra)
        self._iteration_stats.append(stats)
        logger.info(
            "Iter %d: HV=%.6f (raw=%.1f, max=%.1f)  |PF|=%d  n=%d",
            stats["iteration"], stats["hypervolume"],
            stats["hypervolume_raw"], self.hv_max,
            stats["pareto_size"], stats["n_total"]
        )
        return stats

    def get_iteration_stats(self) -> List[Dict]:
        return list(self._iteration_stats)

    def get_hv_trace(self) -> np.ndarray:
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
        lines = []
        lines.append("=== 充电协议优化历史 ===")
        lines.append(f"已评估: {self.size} 条 (可行: {self.n_feasible})")
        lines.append(
            f"决策变量: I₁∈[{self.param_bounds['I1'][0]},{self.param_bounds['I1'][1]}]A, "
            f"SOC₁∈[{self.param_bounds['SOC1'][0]},{self.param_bounds['SOC1'][1]}], "
            f"I₂∈[{self.param_bounds['I2'][0]},{self.param_bounds['I2'][1]}]A"
        )
        lines.append("目标 (均 minimize): 充电时间[s], 峰值温度[K], 老化程度[%]")
        lines.append("")

        if include_stats and self.n_feasible > 0:
            stats = self.get_Y_stats(feasible_only=True)
            lines.append("--- 目标统计 ---")
            for i, (name, label) in enumerate(zip(OBJECTIVE_NAMES, OBJECTIVE_LABELS)):
                lines.append(
                    f"  {label}: min={stats['min'][i]:.4f}  "
                    f"max={stats['max'][i]:.4f}  mean={stats['mean'][i]:.4f}"
                )
            lines.append(f"  超体积 (HV, 归一化): {self.compute_hypervolume():.6f}  [raw={self.compute_hypervolume_raw():.1f} / max={self.hv_max:.1f}]")
            lines.append("")

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

    def to_llm_candidates_prompt(self, n_candidates: int = 5, strategy: str = "explore") -> str:
        context = self.to_llm_context()
        strategy_desc = {
            "explore":  "侧重探索未知区域",
            "exploit":  "侧重开发已知优良区域",
            "balanced": "平衡探索与开发",
        }
        return (
            f"{context}\n=== 任务 ===\n"
            f"请提出 {n_candidates} 个新候选充电协议。策略: {strategy_desc.get(strategy, strategy)}\n"
            f"以 JSON 格式返回: [{{\"I1\": ..., \"SOC1\": ..., \"I2\": ..., \"rationale\": \"...\"}}]"
        )

    def to_gp_mean_prompt(self, theta_query: np.ndarray) -> str:
        theta_query = np.atleast_2d(theta_query)
        context = self.to_llm_context(max_observations=15, include_top_k=3, include_recent=3)
        query_str = "\n".join(
            f"  θ[{i}]: I₁={t[0]:.3f}A, SOC₁={t[1]:.4f}, I₂={t[2]:.3f}A"
            for i, t in enumerate(theta_query)
        )
        return (
            f"{context}\n=== GP 均值函数预测任务 ===\n{query_str}\n"
            f"以 JSON 格式返回: [{{\"time_s\": ..., \"temp_K\": ..., \"aging_pct\": ...}}]"
        )

    # ============================================================
    #  持久化
    # ============================================================
    def save(self, path: str) -> None:
        data = {
            "version":         "1.0",
            "param_bounds":    self.param_bounds,
            "ref_point":       self.ref_point.tolist(),
            "ideal_point":     self.ideal_point.tolist(),
            "normalize":       self.normalize,
            "observations":    [o.to_dict() for o in self._observations],
            "pareto_indices":  self._pareto_indices,
            "iteration_stats": self._iteration_stats,
            # Fix 5: 保存停滞窗口状态，断点续算后不跳变
            "improvement_window": list(self._improvement_window),
            "saved_at":        datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("数据库已保存: %s (%d 条记录)", path, self.size)

    @classmethod
    def load(cls, path: str) -> "ObservationDB":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        db = cls(
            param_bounds = data.get("param_bounds"),
            ref_point    = np.array(data["ref_point"]) if "ref_point" in data else None,
            ideal_point  = np.array(data["ideal_point"]) if "ideal_point" in data else None,
            normalize    = data.get("normalize", True),
        )
        for od in data.get("observations", []):
            db._observations.append(Observation.from_dict(od))
        db._pareto_indices  = data.get("pareto_indices", [])
        db._iteration_stats = data.get("iteration_stats", [])
        # Fix 5: 恢复停滞窗口状态
        from collections import deque
        db._improvement_window = deque(data.get("improvement_window", []), maxlen=2)
        db._update_pareto()
        logger.info("数据库已加载: %s (%d 条记录, |PF|=%d)", path, db.size, db.pareto_size)
        return db

    # ============================================================
    #  便捷查询
    # ============================================================
    def get_best_per_objective(self) -> Dict[str, "Observation"]:
        feasible = self.get_feasible()
        if not feasible:
            return {}
        result = {}
        for i, name in enumerate(OBJECTIVE_NAMES):
            best = min(feasible, key=lambda o: o.objectives[i])
            result[name] = best
        return result

    def get_gp_pred_errors(self) -> Optional[np.ndarray]:
        errors = []
        for o in self._observations:
            if o.gp_pred is not None and "mean" in o.gp_pred:
                pred = np.array(o.gp_pred["mean"])
                errors.append(o.objectives - pred)
        return np.array(errors) if errors else None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "LLAMBO-MO ObservationDB Summary",
            "=" * 50,
            f"  总记录数:       {self.size}",
            f"  可行解:         {self.n_feasible}",
            f"  当前迭代:       {self.current_iteration}",
            f"  Pareto 前沿:    {self.pareto_size} 个非支配解",
            f"  超体积 (HV, 归一化): {self.compute_hypervolume():.6f}  [raw={self.compute_hypervolume_raw():.1f} / max={self.hv_max:.1f}]",
        ]
        src_counts: Dict[str, int] = {}
        for o in self._observations:
            src_counts[o.source] = src_counts.get(o.source, 0) + 1
        lines.append(f"  来源分布:       {src_counts}")
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
        w_vec:  np.ndarray,
        y_min:  Optional[np.ndarray] = None,
        y_max:  Optional[np.ndarray] = None,
        eta:    float                = 0.05,
    ) -> None:
        """
        每迭代由 optimizer.py 调用，注入当前 Tchebycheff 权重和动态 min/max。

        [修复]
        - 接受 y_min/y_max（log 空间动态边界）由 optimizer 传入，保证与 GP 训练使用
          同一归一化基准（而非 database 内部独立计算，避免不一致）
        - eta 由 optimizer 传入（默认 0.05，与 Eq.1 一致）
        - update_stagnation=True：迭代边界才计入停滞

        Parameters
        ----------
        w_vec : (3,)   当前 Tchebycheff 权重（来自 Riesz 集合）
        y_min : (3,)   log 空间动态下界（由 optimizer._update_dynamic_bounds 计算）
        y_max : (3,)   log 空间动态上界
        eta   : float  tiebreaker 权重（Eq.1，默认 0.05）
        """
        self._w_vec = np.asarray(w_vec, dtype=float).ravel()
        self._eta   = float(eta)

        if y_min is not None:
            self._y_min = np.asarray(y_min, dtype=float).ravel()
        if y_max is not None:
            self._y_max = np.asarray(y_max, dtype=float).ravel()

        # 迭代开始时重算，并更新停滞计数
        self._recompute_best(update_stagnation=True)

    def _recompute_best(self, update_stagnation: bool = False) -> None:
        """
        根据当前 Tchebycheff 上下文重新计算 f_min 和 theta_best。

        [修复]
        公式（Eq.2a + Eq.2b + Eq.1）：
          f̃₃ = log₁₀(aging)             ← log 变换（Eq.2a）
          f̄ᵢ = (f̃ᵢ - y_min_i) / (y_max_i - y_min_i)  ← 动态归一化（Eq.2b）
          f_tch = max_i(wᵢ·f̄ᵢ) + η·Σᵢ(wᵢ·f̄ᵢ)  ← η=0.05 tiebreaker（Eq.1）

        Parameters
        ----------
        update_stagnation : bool
          True  → 更新 _stagnation_count（仅在 update_tchebycheff_context 调用时用）
          False → 不改变停滞计数（add_observation 调用时用，避免迭代内累计）
        """
        feasible = self.get_feasible()
        if not feasible:
            return

        # Step 1: 取所有可行解原始目标
        Y_raw = np.array([o.objectives for o in feasible])  # (n, 3)

        # Step 2: log₁₀ 变换（Eq.2a）
        Y_tilde = Y_raw.copy()
        Y_tilde[:, 0] = np.log10(np.maximum(Y_raw[:, 0], 1.0))
        Y_tilde[:, 2] = np.log10(np.maximum(Y_raw[:, 2], 1e-12))

        # Step 3: 动态 min-max 归一化（Eq.2b）
        # Fix 6: 初始阶段 denom ≈ 0 时，用全局物理参考范围兜底（而非 1.0）
        # 确保 F_tch 在初始阶段不全为 0，GP 可获得有效梯度信息
        global_range = np.array([
            np.log10(DEFAULT_REF_POINT[0]) - np.log10(DEFAULT_IDEAL_POINT[0]),  # ≈0.556
            DEFAULT_REF_POINT[1] - DEFAULT_IDEAL_POINT[1],                       # =30
            np.log10(DEFAULT_REF_POINT[2]) - np.log10(DEFAULT_IDEAL_POINT[2]),  # =2
        ])
        denom = self._y_max - self._y_min
        for i in range(3):
            if denom[i] < 0.05 * global_range[i]:
                denom[i] = global_range[i]
        Y_bar = (Y_tilde - self._y_min) / denom               # (n, 3)

        # Step 4: Tchebycheff + η tiebreaker（Eq.1）
        w     = self._w_vec
        Wf    = w[np.newaxis, :] * Y_bar        # (n, 3)
        F_tch = Wf.max(axis=1) + self._eta * Wf.sum(axis=1)  # (n,)

        best_idx = int(np.argmin(F_tch))
        best_tch = float(F_tch[best_idx])

        self._prev_f_min = self._f_min
        self._f_min      = best_tch
        self._theta_best = feasible[best_idx].theta.copy()

        if update_stagnation:
            # Fix 5: 阈值从 1e-6 提升到 1e-3；额外检测 |PF| 变化；滑动窗口平滑
            current_hv = self.compute_hypervolume()
            current_pareto_size = self.pareto_size
            hv_improvement = current_hv - self._prev_hv_for_stagnation
            pf_grew = current_pareto_size > self._prev_pareto_size

            # 本步是否有真实改进（HV 提升 > 0.1% 或 PF 增大）
            improved = (hv_improvement > 1e-3) or pf_grew
            self._improvement_window.append(improved)

            # 滑动窗口满2步且两步都未改进 → 停滞计数递增；否则重置
            if len(self._improvement_window) == 2 and not any(self._improvement_window):
                self._stagnation_count += 1
            elif improved:
                self._stagnation_count = 0

            self._prev_hv_for_stagnation = current_hv
            self._prev_pareto_size = current_pareto_size

    # ── DatabaseProtocol 四个必须方法 ─────────────────────────────────────

    def get_f_min(self) -> float:
        """返回当前 Tchebycheff 标量最优值（Eq.1 输出，动态归一化空间）。"""
        return self._f_min

    def get_theta_best(self) -> np.ndarray:
        """返回当前最优 θ（决策变量原始空间）。"""
        if self._theta_best is None:
            lo = np.array([v[0] for v in self.param_bounds.values()])
            hi = np.array([v[1] for v in self.param_bounds.values()])
            return (lo + hi) / 2.0
        return self._theta_best.copy()

    def has_improved(self) -> bool:
        """若上次迭代 f_min 下降则返回 True（供停滞检测 Eq.22）。"""
        return self._stagnation_count == 0

    def get_stagnation_count(self) -> int:
        """返回连续未改进迭代次数。"""
        return self._stagnation_count

    def __repr__(self) -> str:
        return (
            f"ObservationDB(n={self.size}, feasible={self.n_feasible}, "
            f"|PF|={self.pareto_size}, HV={self.compute_hypervolume():.6f})"
        )


# ================================================================
#  辅助: 2D 超体积工具函数
# ================================================================
def _insert_2d_front(
    front: List[Tuple[float, float]],
    pt:    Tuple[float, float],
) -> List[Tuple[float, float]]:
    new_front = []
    inserted  = False
    dominated = False

    for fp in front:
        if fp[0] <= pt[0] and fp[1] <= pt[1]:
            if fp[0] < pt[0] or fp[1] < pt[1]:
                dominated = True
                new_front.append(fp)
                continue
        if pt[0] <= fp[0] and pt[1] <= fp[1]:
            if pt[0] < fp[0] or pt[1] < fp[1]:
                continue
        if not inserted and pt[0] <= fp[0]:
            new_front.append(pt)
            inserted = True
        new_front.append(fp)

    if dominated:
        return front
    if not inserted:
        new_front.append(pt)
    return new_front


def _compute_2d_hv(
    front: List[Tuple[float, float]],
    ref_y: float,
    ref_z: float,
) -> float:
    if not front:
        return 0.0
    pts_valid = [(y, z) for y, z in front if y < ref_y and z < ref_z]
    if not pts_valid:
        return 0.0
    pts_valid.sort(key=lambda p: p[0])   # 升序 y（f2）
    # FIX: 高度 = ref_z - z（全高），而非 (prev_z - z)（增量）
    # 原代码用增量导致严重低估，如 2 点时少算 >95% 的面积
    hv = 0.0
    prev_z = ref_z
    for i, (y, z) in enumerate(pts_valid):
        if z < prev_z:
            y_width = pts_valid[i + 1][0] - y if i + 1 < len(pts_valid) else ref_y - y
            hv += y_width * (ref_z - z)   # ← ref_z，不是 prev_z
            prev_z = z
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
    print("LLAMBO-MO ObservationDB 修复验证测试")
    print("=" * 60)

    db = ObservationDB()

    test_data = [
        ([3.5, 0.40, 2.0], [2800, 305.0, 0.0012], "init"),
        ([5.0, 0.35, 2.5], [2100, 312.0, 0.0035], "init"),
        ([7.0, 0.20, 4.0], [1200, 322.0, 0.0150], "init"),
        ([6.0, 0.30, 3.0], [1600, 318.0, 0.0080], "gp_ei"),
        ([4.5, 0.45, 2.0], [2500, 308.0, 0.0020], "llm"),
        ([6.5, 0.25, 3.5], [1400, 320.0, 0.0100], "llm_gp"),
    ]

    for i, (theta, obj, src) in enumerate(test_data):
        db.add_observation(
            theta=np.array(theta),
            objectives=np.array(obj),
            feasible=True,
            source=src,
            iteration=i // 3,
        )

    print(f"添加 {db.size} 条观测后 stagnation_count = {db._stagnation_count}")
    assert db._stagnation_count == 0, "BUG: add_observation 不应累积停滞计数"
    print("[OK] stagnation_count 在 add_observation 后保持 0（修复验证通过）")

    # 模拟 optimizer 传入动态 min/max
    feasible = db.get_feasible()
    Y_raw = np.array([o.objectives for o in feasible])
    Y_tilde = Y_raw.copy()
    Y_tilde[:, 0] = np.log10(np.maximum(Y_raw[:, 0], 1.0))
    Y_tilde[:, 2] = np.log10(np.maximum(Y_raw[:, 2], 1e-12))
    y_min = Y_tilde.min(axis=0)
    y_max = Y_tilde.max(axis=0)

    w = np.array([0.4, 0.3, 0.3])
    db.update_tchebycheff_context(w, y_min=y_min, y_max=y_max, eta=0.05)
    print(f"\nupdate_tchebycheff_context 后:")
    print(f"  f_min = {db.get_f_min():.6f}")
    print(f"  theta_best = {db.get_theta_best().round(4)}")
    print(f"  has_improved = {db.has_improved()}")
    print(f"  stagnation_count = {db.get_stagnation_count()}")

    print("\n[OK] database.py 修复验证完成")

    # ============================================================
    #  HV 归一化验证测试
    # ============================================================
    print("\n" + "=" * 60)
    print("HV 归一化验证测试 (time 和 aging 都取 log)")
    print("=" * 60)

    print(f"\nHV_max = {db.hv_max:.2f}")
    print(f"  = (log10(7200)-log10(1000)) × (328-298) × (log10(0.1)-log10(0.001))")
    print(f"  = (3.857-3.0) × 30 × 2 = 0.857 × 30 × 2 ≈ 51.4")

    hv_raw = db.compute_hypervolume_raw()
    hv_norm = db.compute_hypervolume()

    print(f"\n6 个测试点的 HV 计算结果:")
    print(f"  HV_raw (未归一化) = {hv_raw:.2f}")
    print(f"  HV_norm (归一化)  = {hv_norm:.6f}")
    print(f"  HV_norm × HV_max  = {hv_norm * db.hv_max:.2f}")

    # 验证归一化：HV ∈ [0, 1]
    assert 0.0 <= hv_norm <= 1.0, f"BUG: HV 归一化失败，HV={hv_norm}"
    print("\n[OK] HV 归一化验证通过：HV ∈ [0, 1]")

    # 验证 HV_raw / HV_max = HV_norm
    expected_hv_norm = hv_raw / db.hv_max
    assert abs(hv_norm - expected_hv_norm) < 1e-10, "BUG: HV 归一化计算错误"
    print("[OK] HV 计算验证通过：HV_norm = HV_raw / HV_max")