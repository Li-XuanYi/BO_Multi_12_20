"""
database.py — LLAMBO-MO 观测数据库
====================================
决策变量（5维）：θ = (I1, I2, I3, dSOC1, dSOC2)
目标（3维，均最小化）：[time_s, delta_temp_K, aging_pct]

主要变更（相对旧版）：
  - 参数从 3D (I1, SOC1, I2) 升级为 5D (I1, I2, I3, dSOC1, dSOC2)
  - 删除 to_gp_mean_prompt() / to_llm_candidates_prompt()（由 llm_interface.py 接管）
  - 保留全部核心逻辑：Tchebycheff上下文、HV计算、Pareto追踪、停滞检测
"""

import json
import copy
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from llmbo.scalarization import (
    canonical_hv_from_raw,
    canonicalize_objective_preprocess_mode,
    compute_parego_reference_from_raw,
    compute_tchebycheff_from_raw,
    compute_tchebycheff_from_raw_with_ideal,
)
from utils.constants import (
    DEFAULT_BOUNDS as CANONICAL_DEFAULT_BOUNDS,
    DSOC_SUM_MAX as CANONICAL_DSOC_SUM_MAX,
    IDEAL_POINT as CANONICAL_IDEAL_POINT,
    LLM_SAFE_DSOC_SUM_MAX,
    PARAM_NAMES as CANONICAL_PARAM_NAMES,
    REF_POINT as CANONICAL_REF_POINT,
    ECKER2015_REF_POINT,
    ECKER2015_IDEAL_POINT,
)

logger = logging.getLogger(__name__)

# Try to use pymoo for robust HV calculation.
try:
    from pymoo.indicators.hv import Hypervolume as PymooHV
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False
    PymooHV = None

# ================================================================
#  常量
# ================================================================
OBJECTIVE_NAMES  = ("time_s", "temp_K", "aging_pct")
OBJECTIVE_LABELS = ("充电时间 [s]", "峰值温升 [K]", "老化程度 [%]")

# 决策变量（顺序与 pybamm_simulator.evaluate() 的 theta 完全对齐）
PARAM_NAMES = CANONICAL_PARAM_NAMES
NUM_OBJECTIVES = 3
NUM_PARAMS     = 5

# HV 计算：time 和 aging 取 log₁₀，temp 保持原始空间
DEFAULT_REF_POINT = CANONICAL_REF_POINT.copy()
DEFAULT_IDEAL_POINT = CANONICAL_IDEAL_POINT.copy()

DEFAULT_HV_MAX = (
    (np.log10(DEFAULT_REF_POINT[0]) - np.log10(DEFAULT_IDEAL_POINT[0])) *
    (DEFAULT_REF_POINT[1] - DEFAULT_IDEAL_POINT[1]) *
    (np.log10(DEFAULT_REF_POINT[2]) - np.log10(DEFAULT_IDEAL_POINT[2]))
)
PARETO_DUPLICATE_ATOL = 1e-9
HV_DUPLICATE_DECIMALS = 12
DSOC_SUM_MAX = CANONICAL_DSOC_SUM_MAX
HV_DISPLAY_DIVISOR = 0.4

# 决策变量边界（与 pybamm_simulator._run() 的换算逻辑对齐）
# I1/I2/I3: 协议电流参数（仿真器内部换算 I_A = I * Q_eff / 5）
# dSOC1/dSOC2: SOC 区间宽度（直接传入仿真器）
DEFAULT_BOUNDS = copy.deepcopy(CANONICAL_DEFAULT_BOUNDS)


def make_observation_db(param_bounds: Optional[Dict] = None, param_set: str = "Chen2020", **kwargs) -> "ObservationDB":
    """工厂函数：创建使用全局统一 ref/ideal/hv_max 的 ObservationDB 实例。

    Args:
        param_bounds: 参数边界，默认使用DEFAULT_BOUNDS
        param_set: 参数集名称，"Chen2020"或"Ecker2015"，影响reference points
        **kwargs: 其他传递给ObservationDB的参数
    """
    if param_set == "Ecker2015":
        ref_point = ECKER2015_REF_POINT.copy()
        ideal_point = ECKER2015_IDEAL_POINT.copy()
    else:
        ref_point = DEFAULT_REF_POINT.copy()
        ideal_point = DEFAULT_IDEAL_POINT.copy()

    return ObservationDB(
        param_bounds=param_bounds or copy.deepcopy(DEFAULT_BOUNDS),
        ref_point=ref_point,
        ideal_point=ideal_point,
        normalize=True,
        **kwargs,
    )


# ================================================================
#  单条观测记录
# ================================================================
class Observation:
    """单条评估记录。"""

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
        status = "OK" if self.feasible else "FAIL"
        t = self.theta
        f = self.objectives
        return (
            f"Obs({status} iter={self.iteration} src={self.source} "
            f"θ=[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f},{t[3]:.3f},{t[4]:.3f}] "
            f"f=[{f[0]:.0f}s,{f[1]:.1f}K,{f[2]:.4f}%])"
        )


# ================================================================
#  观测数据库
# ================================================================
class ObservationDB:
    """
    LLAMBO-MO 核心数据库（5维决策变量版本）。

    决策变量顺序：(I1, I2, I3, dSOC1, dSOC2)
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
        self._pareto_objectives: Optional[np.ndarray] = None

        self.param_bounds = param_bounds or copy.deepcopy(DEFAULT_BOUNDS)
        self.ref_point    = np.asarray(ref_point) if ref_point is not None else DEFAULT_REF_POINT.copy()
        self.ideal_point  = np.asarray(ideal_point) if ideal_point is not None else DEFAULT_IDEAL_POINT.copy()

        self.hv_max = float(
            (np.log10(self.ref_point[0]) - np.log10(self.ideal_point[0])) *
            (self.ref_point[1] - self.ideal_point[1]) *
            (np.log10(self.ref_point[2]) - np.log10(self.ideal_point[2]))
        )
        self.normalize = normalize

        self._iteration_stats: List[Dict] = []

        # Tchebycheff 上下文（由 optimizer 每迭代注入）
        self._w_vec:   np.ndarray = np.array([1.0/3, 1.0/3, 1.0/3])
        self._y_min:   np.ndarray = np.zeros(NUM_OBJECTIVES)
        self._y_max:   np.ndarray = np.ones(NUM_OBJECTIVES)
        self._ideal_point_raw: Optional[np.ndarray] = None
        self._eta:     float      = 0.05
        self._scalarization_mode: str = "log_ideal_gap"
        self._objective_preprocess_mode: str = "minmax"
        self._parego_invert_weights: bool = False
        self._f_min:   float      = float("inf")
        self._prev_f_min: float   = float("inf")
        self._theta_best: Optional[np.ndarray] = None
        self._stagnation_count: int = 0
        self._prev_hv_for_stagnation: float = 0.0
        self._prev_pareto_size: int = 0

        from collections import deque
        self._improvement_window = deque(maxlen=2)

        logger.info("ObservationDB 初始化: bounds=%s", self.param_bounds)

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
            self._update_pareto(new_obj=objectives)
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
        lo = np.array([self.param_bounds[p][0] for p in PARAM_NAMES])
        hi = np.array([self.param_bounds[p][1] for p in PARAM_NAMES])
        return (X - lo) / (hi - lo + 1e-12)

    def denormalize_X(self, X_norm: np.ndarray) -> np.ndarray:
        lo = np.array([self.param_bounds[p][0] for p in PARAM_NAMES])
        hi = np.array([self.param_bounds[p][1] for p in PARAM_NAMES])
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
        """增量 Pareto 更新 O(|PF|)，首次或 load 时全量重建。"""
        feasible = [(i, o) for i, o in enumerate(self._observations) if o.feasible]
        if not feasible:
            self._pareto_indices = []
            self._pareto_objectives = None
            return

        if new_obj is None:
            # 全量重建
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
            self._pareto_indices    = [indices[i] for i in range(n) if not is_dominated[i]]
            self._pareto_objectives = objs[~is_dominated]
            return

        # 增量更新
        new_obj = np.asarray(new_obj, dtype=float)
        new_idx = len(self._observations) - 1
        new_theta = np.asarray(self._observations[new_idx].theta, dtype=float)

        if not self._pareto_indices:
            self._pareto_indices    = [new_idx]
            self._pareto_objectives = new_obj[np.newaxis, :]
            return

        # Skip exact/near-exact duplicates to keep the Pareto front stable.
        if np.any(np.all(np.isclose(self._pareto_objectives, new_obj, atol=PARETO_DUPLICATE_ATOL), axis=1)):
            logger.debug(
                "Duplicate Pareto objective detected: %s. Skipping.",
                np.round(new_obj, 10),
            )
            return

        pf_thetas = np.array([self._observations[i].theta for i in self._pareto_indices], dtype=float)
        duplicate_theta_mask = np.all(np.isclose(pf_thetas, new_theta, atol=PARETO_DUPLICATE_ATOL), axis=1)
        if np.any(duplicate_theta_mask):
            duplicate_idx = int(np.flatnonzero(duplicate_theta_mask)[0])
            existing_obj = self._pareto_objectives[duplicate_idx]
            if np.all(new_obj <= existing_obj) and np.any(new_obj < existing_obj):
                self._pareto_indices.pop(duplicate_idx)
                self._pareto_objectives = np.delete(self._pareto_objectives, duplicate_idx, axis=0)
                logger.debug(
                    "Replacing duplicate Pareto theta with dominating objective: theta=%s",
                    np.round(new_theta, 6),
                )
                if not self._pareto_indices:
                    self._pareto_indices = [new_idx]
                    self._pareto_objectives = new_obj[np.newaxis, :]
                    return
            else:
                logger.debug(
                    "Duplicate Pareto theta detected: theta=%s. Skipping.",
                    np.round(new_theta, 6),
                )
                return

        # 检查新点是否被支配
        for pf_obj in self._pareto_objectives:
            if np.all(pf_obj <= new_obj) and np.any(pf_obj < new_obj):
                return

        # 移除被新点支配的旧点
        not_dominated_by_new = ~(
            np.all(new_obj <= self._pareto_objectives, axis=1) &
            np.any(new_obj < self._pareto_objectives, axis=1)
        )
        self._pareto_indices    = [self._pareto_indices[i] for i in range(len(self._pareto_indices)) if not_dominated_by_new[i]]
        self._pareto_objectives = self._pareto_objectives[not_dominated_by_new]
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
        """归一化超体积 HV ∈ [0, 1]，time 和 aging 取 log₁₀。"""
        hv_raw = self._compute_hypervolume_value(ref_point=ref_point)
        hv_max = self.hv_max if self.hv_max > 1e-12 else 1.0
        return (hv_raw / hv_max) / HV_DISPLAY_DIVISOR

    def compute_hypervolume_raw(self, ref_point: Optional[np.ndarray] = None) -> float:
        """未归一化超体积（供调试）。"""
        return self._compute_hypervolume_value(ref_point=ref_point)

    def compute_hypervolume_canonical(self, ref_point: Optional[np.ndarray] = None) -> float:
        """Canonical normalized HV used for benchmark and ablation comparisons."""
        return canonical_hv_from_raw(self.compute_hypervolume_raw(ref_point=ref_point), self.hv_max)

    def _compute_hypervolume_value(self, ref_point: Optional[np.ndarray] = None) -> float:
        ref = np.asarray(ref_point if ref_point is not None else self.ref_point, dtype=float).copy()
        _, Y_pf = self.get_pareto_XY()
        if len(Y_pf) == 0:
            return 0.0

        Y_hv = Y_pf.copy()
        Y_hv[:, 0] = np.log10(np.maximum(Y_pf[:, 0], 1.0))
        Y_hv[:, 2] = np.log10(np.maximum(Y_pf[:, 2], 1e-12))

        ref_hv = ref.copy()
        ref_hv[0] = np.log10(np.maximum(ref[0], 1.0))
        ref_hv[2] = np.log10(np.maximum(ref[2], 1e-12))

        mask = np.all(Y_hv < ref_hv, axis=1)
        Y_hv = Y_hv[mask]
        if len(Y_hv) == 0:
            return 0.0

        Y_hv = self._deduplicate_hv_points(Y_hv)
        if len(Y_hv) == 0:
            return 0.0

        if HAS_PYMOO:
            hv_calculator = PymooHV(ref_point=ref_hv)
            return float(hv_calculator.do(Y_hv))
        return self._hv_3d(Y_hv, ref_hv)

    @staticmethod
    def _deduplicate_hv_points(points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if len(pts) <= 1:
            return pts
        rounded = np.round(pts, decimals=HV_DUPLICATE_DECIMALS)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        return pts[np.sort(unique_idx)]

    @staticmethod
    def _hv_3d(points: np.ndarray, ref: np.ndarray) -> float:
        pts = ObservationDB._deduplicate_hv_points(points)
        if len(pts) == 0:
            return 0.0
        if len(pts) == 1:
            return float(np.prod(np.maximum(ref - pts[0], 0.0)))

        x_coords = np.unique(np.concatenate([pts[:, 0], ref[0:1]]))
        y_coords = np.unique(np.concatenate([pts[:, 1], ref[1:2]]))
        z_coords = np.unique(np.concatenate([pts[:, 2], ref[2:3]]))
        hv = 0.0
        for i in range(len(x_coords) - 1):
            x0, x1 = x_coords[i], x_coords[i + 1]
            if x1 <= x0:
                continue
            for j in range(len(y_coords) - 1):
                y0, y1 = y_coords[j], y_coords[j + 1]
                if y1 <= y0:
                    continue
                for k in range(len(z_coords) - 1):
                    z0, z1 = z_coords[k], z_coords[k + 1]
                    if z1 <= z0:
                        continue
                    corner = np.array([x0, y0, z0], dtype=float)
                    if np.any(np.all(pts <= corner + 1e-12, axis=1)):
                        hv += (x1 - x0) * (y1 - y0) * (z1 - z0)
        return float(hv)

    # ============================================================
    #  迭代统计
    # ============================================================
    def record_iteration_stats(self, extra: Optional[Dict] = None) -> Dict:
        stats = {
            "iteration":       self.current_iteration,
            "n_total":         self.size,
            "n_feasible":      self.n_feasible,
            "pareto_size":     self.pareto_size,
            "hypervolume":     self.compute_hypervolume(),
            "display_hv":      self.compute_hypervolume(),
            "canonical_hv":    self.compute_hypervolume_canonical(),
            "hypervolume_canonical": self.compute_hypervolume_canonical(),
            "hypervolume_raw": self.compute_hypervolume_raw(),
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
            "Iter %d: HV=%.6f  |PF|=%d  n=%d",
            stats["iteration"], stats["hypervolume"],
            stats["pareto_size"], stats["n_total"]
        )
        return stats

    def get_iteration_stats(self) -> List[Dict]:
        return list(self._iteration_stats)

    def get_hv_trace(self) -> np.ndarray:
        return np.array([s["hypervolume"] for s in self._iteration_stats])

    def get_canonical_hv_trace(self) -> np.ndarray:
        return np.array([
            s.get("canonical_hv", canonical_hv_from_raw(s.get("hypervolume_raw", 0.0), self.hv_max))
            for s in self._iteration_stats
        ])

    def get_hv_feedback_summary(self, window: int = 3) -> Dict[str, Any]:
        current_display_hv = float(self.compute_hypervolume())
        current_canonical_hv = float(self.compute_hypervolume_canonical())
        stats = self.get_iteration_stats()
        window = max(int(window), 1)
        if not stats:
            return {
                "current_hv": current_display_hv,
                "current_display_hv": current_display_hv,
                "current_canonical_hv": current_canonical_hv,
                "hv_delta_last_k": 0.0,
                "canonical_hv_delta_last_k": 0.0,
                "pareto_delta_last_k": 0,
                "window": window,
                "summary": "HV history unavailable in this iteration.",
            }

        recent = stats[-window:]
        start_hv = float(recent[0].get("hypervolume", current_display_hv))
        start_canonical = float(recent[0].get("canonical_hv", current_canonical_hv))
        delta = current_display_hv - start_hv
        canonical_delta = current_canonical_hv - start_canonical
        pareto_delta = int(self.pareto_size) - int(recent[0].get("pareto_size", self.pareto_size))
        return {
            "current_hv": current_display_hv,
            "current_display_hv": current_display_hv,
            "current_canonical_hv": current_canonical_hv,
            "hv_delta_last_k": float(delta),
            "canonical_hv_delta_last_k": float(canonical_delta),
            "pareto_delta_last_k": int(pareto_delta),
            "window": window,
            "summary": (
                f"display_hv={current_display_hv:.6f}, canonical_hv={current_canonical_hv:.6f}, "
                f"canonical_hv_delta_last_{window}={canonical_delta:.6f}, "
                f"pareto_size={self.pareto_size}, pareto_delta_last_{window}={pareto_delta}"
            ),
        }

    def get_similar_weight_guidance_stats(
        self,
        w_vec: np.ndarray,
        *,
        similarity_threshold: float = 0.85,
        fallback_score: float = 0.75,
        hv_gain_threshold: float = 1e-3,
    ) -> Dict[str, Any]:
        stats = self.get_iteration_stats()
        if len(stats) < 2:
            return {
                "similar_count": 0,
                "success_rate": float(fallback_score),
                "threshold": float(similarity_threshold),
                "summary": (
                    f"similar_weight_guidance: matches=0, score={float(fallback_score):.2f} "
                    "(fallback)"
                ),
            }

        w_now = np.asarray(w_vec, dtype=float).ravel()
        denom_now = max(float(np.linalg.norm(w_now)), 1e-12)
        weighted_success = 0.0
        total_weight = 0.0
        matches = 0

        for idx, stat in enumerate(stats[:-1]):
            hist_w = stat.get("w_vec")
            if hist_w is None or stat.get("llm_guidance") is None:
                continue
            hist_w_arr = np.asarray(hist_w, dtype=float).ravel()
            if hist_w_arr.size != w_now.size:
                continue

            denom_hist = max(float(np.linalg.norm(hist_w_arr)), 1e-12)
            similarity = float(np.dot(w_now, hist_w_arr) / (denom_now * denom_hist))
            if similarity < float(similarity_threshold):
                continue

            next_stat = stats[idx + 1]
            hv_gain = float(next_stat.get("canonical_hv", 0.0)) - float(stat.get("canonical_hv", 0.0))
            pf_gain = int(next_stat.get("pareto_size", 0)) - int(stat.get("pareto_size", 0))
            success = 1.0 if (hv_gain > float(hv_gain_threshold) or pf_gain > 0) else 0.0
            weighted_success += similarity * success
            total_weight += similarity
            matches += 1

        if total_weight <= 1e-12:
            return {
                "similar_count": 0,
                "success_rate": float(fallback_score),
                "threshold": float(similarity_threshold),
                "summary": (
                    f"similar_weight_guidance: matches=0, score={float(fallback_score):.2f} "
                    "(fallback)"
                ),
            }

        success_rate = float(np.clip(weighted_success / total_weight, 0.0, 1.0))
        return {
            "similar_count": int(matches),
            "success_rate": success_rate,
            "threshold": float(similarity_threshold),
            "summary": (
                f"similar_weight_guidance: matches={matches}, weighted_success={success_rate:.2f}"
            ),
        }

    def get_boundary_failure_stats(
        self,
        *,
        safe_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX,
        hard_dsoc_sum_max: float = DSOC_SUM_MAX,
        recent_window: int = 10,
        near_hard_margin: float = 0.02,
    ) -> Dict[str, Any]:
        all_obs = self.get_all()
        feasible = self.get_feasible()
        recent_window = max(int(recent_window), 1)
        if not all_obs:
            return {
                "recent_failures": 0,
                "recent_monotone": 0,
                "near_safe": 0,
                "near_hard": 0,
                "n_feasible": 0,
                "recent_window": recent_window,
                "summary": "none",
            }

        feasible_sums = (
            np.array([obs.theta[3] + obs.theta[4] for obs in feasible], dtype=float)
            if feasible else np.empty((0,), dtype=float)
        )
        recent = all_obs[-recent_window:]
        recent_failures = sum(1 for obs in recent if not obs.feasible)
        recent_monotone = sum(
            1
            for obs in recent
            if (obs.theta[1] > obs.theta[0] + 1e-9) or (obs.theta[2] > obs.theta[1] + 1e-9)
        )
        near_safe = int(np.sum(feasible_sums >= float(safe_dsoc_sum_max) - 1e-9)) if feasible_sums.size else 0
        near_hard = int(
            np.sum(feasible_sums >= float(hard_dsoc_sum_max) - float(near_hard_margin))
        ) if feasible_sums.size else 0
        n_feasible = int(len(feasible_sums))
        return {
            "recent_failures": int(recent_failures),
            "recent_monotone": int(recent_monotone),
            "near_safe": int(near_safe),
            "near_hard": int(near_hard),
            "n_feasible": n_feasible,
            "recent_window": recent_window,
            "summary": (
                f"recent_failures={recent_failures}/{recent_window}, "
                f"recent_monotone={recent_monotone}/{recent_window}, "
                f"near_safe={near_safe}/{n_feasible}, near_hard={near_hard}/{n_feasible}"
            ),
        }

    # ============================================================
    #  LLM 上下文生成（供 Touchpoint 2 Prompt 使用）
    # ============================================================
    def to_llm_context(
        self,
        max_observations: int  = 20,
        include_pareto:   bool = True,
        include_top_k:    int  = 5,
        include_stats:    bool = True,
        include_recent:   int  = 5,
    ) -> str:
        """生成结构化的 Pareto 上下文摘要，注入 LLM Prompt。"""
        lines = []
        lines.append("=== 充电协议优化历史 ===")
        lines.append(f"已评估: {self.size} 条 (可行: {self.n_feasible})")
        lines.append(
            f"决策变量: "
            f"I₁∈[{self.param_bounds['I1'][0]},{self.param_bounds['I1'][1]}]A, "
            f"I₂∈[{self.param_bounds['I2'][0]},{self.param_bounds['I2'][1]}]A, "
            f"I₃∈[{self.param_bounds['I3'][0]},{self.param_bounds['I3'][1]}]A, "
            f"δSOC₁∈[{self.param_bounds['dSOC1'][0]},{self.param_bounds['dSOC1'][1]}], "
            f"δSOC₂∈[{self.param_bounds['dSOC2'][0]},{self.param_bounds['dSOC2'][1]}]"
        )
        lines.append("目标 (均 minimize): 充电时间[s], 峰值温升[K], 老化程度[%]")
        lines.append("")

        if include_stats and self.n_feasible > 0:
            stats = self.get_Y_stats(feasible_only=True)
            lines.append("--- 目标统计 ---")
            for i, (name, label) in enumerate(zip(OBJECTIVE_NAMES, OBJECTIVE_LABELS)):
                lines.append(
                    f"  {label}: min={stats['min'][i]:.4f}  "
                    f"max={stats['max'][i]:.4f}  mean={stats['mean'][i]:.4f}"
                )
            lines.append(f"  超体积 HV (归一化): {self.compute_hypervolume():.6f}")
            lines.append("")

        if include_pareto:
            pf = self.get_pareto_front()
            lines.append(f"--- Pareto 前沿 ({len(pf)} 个非支配解) ---")
            for j, o in enumerate(pf):
                t = o.theta
                f = o.objectives
                lines.append(
                    f"  PF[{j}]: I₁={t[0]:.2f}A I₂={t[1]:.2f}A I₃={t[2]:.2f}A "
                    f"δSOC₁={t[3]:.3f} δSOC₂={t[4]:.3f}  →  "
                    f"time={f[0]:.0f}s  temp={f[1]:.2f}K  aging={f[2]:.6f}%"
                )
            lines.append("")

        if include_top_k > 0 and self.n_feasible > 0:
            feasible = self.get_feasible()
            for i, (name, label) in enumerate(zip(OBJECTIVE_NAMES, OBJECTIVE_LABELS)):
                sorted_obs = sorted(feasible, key=lambda o: o.objectives[i])
                top = sorted_obs[:include_top_k]
                lines.append(f"--- {label} Top-{len(top)} ---")
                for j, o in enumerate(top):
                    t = o.theta
                    lines.append(
                        f"  [{j+1}] I₁={t[0]:.2f} I₂={t[1]:.2f} I₃={t[2]:.2f} "
                        f"δSOC₁={t[3]:.3f} δSOC₂={t[4]:.3f}  →  "
                        f"{name}={o.objectives[i]:.4f}  (src={o.source})"
                    )
            lines.append("")

        if include_recent > 0:
            recent = self._observations[-include_recent:]
            lines.append(f"--- 最近 {len(recent)} 条记录 ---")
            for o in recent:
                status = "✓" if o.feasible else f"✗({o.violation})"
                t = o.theta
                f = o.objectives
                acq_str = f"  acq={o.acq_value:.4f}" if o.acq_value is not None else ""
                lines.append(
                    f"  iter={o.iteration} src={o.source}  "
                    f"I₁={t[0]:.2f} I₂={t[1]:.2f} I₃={t[2]:.2f} "
                    f"δSOC₁={t[3]:.3f} δSOC₂={t[4]:.3f}  →  "
                    f"[{f[0]:.0f}s, {f[1]:.2f}K, {f[2]:.6f}%]  {status}{acq_str}"
                )
            lines.append("")

        return "\n".join(lines)

    # ============================================================
    #  持久化
    # ============================================================
    def save(self, path: str) -> None:
        data = {
            "version":         "2.0",
            "param_bounds":    self.param_bounds,
            "ref_point":       self.ref_point.tolist(),
            "ideal_point":     self.ideal_point.tolist(),
            "normalize":       self.normalize,
            "observations":    [o.to_dict() for o in self._observations],
            "pareto_indices":  self._pareto_indices,
            "iteration_stats": self._iteration_stats,
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
        return {
            name: min(feasible, key=lambda o: o.objectives[i])
            for i, name in enumerate(OBJECTIVE_NAMES)
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "LLAMBO-MO ObservationDB Summary",
            "=" * 50,
            f"  总记录数:       {self.size}",
            f"  可行解:         {self.n_feasible}",
            f"  当前迭代:       {self.current_iteration}",
            f"  Pareto 前沿:    {self.pareto_size} 个非支配解",
            f"  超体积 (HV):    {self.compute_hypervolume():.6f}  [raw={self.compute_hypervolume_raw():.1f}]",
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
                idx = list(OBJECTIVE_NAMES).index(name)
                lines.append(
                    f"    {label}: {o.objectives[idx]:.4f}  "
                    f"@ θ=[{o.theta[0]:.2f},{o.theta[1]:.2f},{o.theta[2]:.2f},"
                    f"{o.theta[3]:.3f},{o.theta[4]:.3f}]"
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
        ideal_point_raw: Optional[np.ndarray] = None,
        eta:    float                = 0.05,
        scalarization_mode: str      = "log_ideal_gap",
        objective_preprocess_mode: str = "minmax",
        parego_invert_weights: bool  = False,
    ) -> None:
        """每迭代由 optimizer.py 调用，注入当前 Tchebycheff 权重和动态 min/max。"""
        self._w_vec = np.asarray(w_vec, dtype=float).ravel()
        self._eta   = float(eta)
        self._scalarization_mode = str(scalarization_mode or "log_ideal_gap").lower()
        self._objective_preprocess_mode = canonicalize_objective_preprocess_mode(objective_preprocess_mode)
        self._parego_invert_weights = bool(parego_invert_weights)
        if y_min is not None:
            self._y_min = np.asarray(y_min, dtype=float).ravel()
        if y_max is not None:
            self._y_max = np.asarray(y_max, dtype=float).ravel()
        self._ideal_point_raw = (
            None if ideal_point_raw is None
            else np.asarray(ideal_point_raw, dtype=float).ravel()
        )
        self._recompute_best(update_stagnation=True)

    def _recompute_best(self, update_stagnation: bool = False) -> None:
        """根据当前 Tchebycheff 上下文重新计算 f_min 和 theta_best。"""
        feasible = self.get_feasible()
        if not feasible:
            return

        Y_raw = np.array([o.objectives for o in feasible], dtype=float)

        if self._scalarization_mode == "parego_reference":
            F_tch = compute_parego_reference_from_raw(
                Y_raw,
                self._w_vec,
                eta=self._eta,
                eps_min=1e-6,
                invert_weights=self._parego_invert_weights,
            )
            best_idx = int(np.argmin(F_tch))
            self._prev_f_min = self._f_min
            self._f_min = float(F_tch[best_idx])
            self._theta_best = feasible[best_idx].theta.copy()

            if update_stagnation:
                current_hv = self.compute_hypervolume()
                current_pareto_size = self.pareto_size
                hv_improvement = current_hv - self._prev_hv_for_stagnation
                pf_grew = current_pareto_size > self._prev_pareto_size
                improved = (hv_improvement > 1e-3) or pf_grew
                self._improvement_window.append(improved)

                if len(self._improvement_window) == 2 and not any(self._improvement_window):
                    self._stagnation_count += 1
                elif improved:
                    self._stagnation_count = 0

                self._prev_hv_for_stagnation = current_hv
                self._prev_pareto_size = current_pareto_size
            return

        # Final scalarization is delegated to the shared module so optimizer,
        # database, and prompts use the same context-dependent f_w semantics.
        if self._ideal_point_raw is not None:
            F_tch = compute_tchebycheff_from_raw_with_ideal(
                Y_raw,
                self._w_vec,
                self._ideal_point_raw,
                self._y_min,
                self._y_max,
                eta=self._eta,
                preprocess_mode=self._objective_preprocess_mode,
            )
        else:
            F_tch = compute_tchebycheff_from_raw(
                Y_raw,
                self._w_vec,
                self._y_min,
                self._y_max,
                eta=self._eta,
                preprocess_mode=self._objective_preprocess_mode,
            )

        best_idx = int(np.argmin(F_tch))
        self._prev_f_min = self._f_min
        self._f_min      = float(F_tch[best_idx])
        self._theta_best = feasible[best_idx].theta.copy()

        if update_stagnation:
            current_hv = self.compute_hypervolume()
            current_pareto_size = self.pareto_size
            hv_improvement  = current_hv - self._prev_hv_for_stagnation
            pf_grew         = current_pareto_size > self._prev_pareto_size
            improved        = (hv_improvement > 1e-3) or pf_grew
            self._improvement_window.append(improved)

            if len(self._improvement_window) == 2 and not any(self._improvement_window):
                self._stagnation_count += 1
            elif improved:
                self._stagnation_count = 0

            self._prev_hv_for_stagnation = current_hv
            self._prev_pareto_size       = current_pareto_size

    def get_f_min(self) -> float:
        return self._f_min

    def get_theta_best(self) -> np.ndarray:
        if self._theta_best is None:
            lo = np.array([v[0] for v in self.param_bounds.values()])
            hi = np.array([v[1] for v in self.param_bounds.values()])
            return (lo + hi) / 2.0
        return self._theta_best.copy()

    def has_improved(self) -> bool:
        return self._stagnation_count == 0

    def get_stagnation_count(self) -> int:
        return self._stagnation_count

    def get_acq_history(self) -> List[Dict]:
        return [
            {
                "iteration":  o.iteration,
                "theta":      o.theta.tolist(),
                "objectives": o.objectives.tolist(),
                "acq_value":  o.acq_value,
                "acq_type":   o.acq_type,
                "source":     o.source,
                "feasible":   o.feasible,
            }
            for o in self._observations if o.acq_value is not None
        ]

    def __repr__(self) -> str:
        return (
            f"ObservationDB(n={self.size}, feasible={self.n_feasible}, "
            f"|PF|={self.pareto_size}, HV={self.compute_hypervolume():.6f})"
        )


# ================================================================
#  辅助: 2D 超体积工具函数
# ================================================================
def _insert_2d_front(front, pt):
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


def _compute_2d_hv(front, ref_y, ref_z):
    if not front:
        return 0.0
    pts_valid = [(y, z) for y, z in front if y < ref_y and z < ref_z]
    if not pts_valid:
        return 0.0
    pts_valid.sort(key=lambda p: p[0])
    hv = 0.0
    prev_z = ref_z
    for i, (y, z) in enumerate(pts_valid):
        if z < prev_z:
            y_width = pts_valid[i + 1][0] - y if i + 1 < len(pts_valid) else ref_y - y
            hv += y_width * (ref_z - z)
            prev_z = z
    return hv
