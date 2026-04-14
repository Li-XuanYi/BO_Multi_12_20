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
from utils.constants import (
    DEFAULT_BOUNDS as CANONICAL_DEFAULT_BOUNDS,
    IDEAL_POINT as CANONICAL_IDEAL_POINT,
    PARAM_NAMES as CANONICAL_PARAM_NAMES,
    REF_POINT as CANONICAL_REF_POINT,
)

logger = logging.getLogger(__name__)

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

# 决策变量边界（与 pybamm_simulator._run() 的换算逻辑对齐）
# I1/I2/I3: 协议电流参数（仿真器内部换算 I_A = I * Q_eff / 5）
# dSOC1/dSOC2: SOC 区间宽度（直接传入仿真器）
DEFAULT_BOUNDS = copy.deepcopy(CANONICAL_DEFAULT_BOUNDS)


def make_observation_db(param_bounds: Optional[Dict] = None, **kwargs) -> "ObservationDB":
    """工厂函数：创建使用全局统一 ref/ideal/hv_max 的 ObservationDB 实例。"""
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

        if not self._pareto_indices:
            self._pareto_indices    = [new_idx]
            self._pareto_objectives = new_obj[np.newaxis, :]
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
        ref = ref_point if ref_point is not None else self.ref_point
        _, Y_pf = self.get_pareto_XY()
        if len(Y_pf) == 0:
            return 0.0

        Y_hv = Y_pf.copy()
        Y_hv[:, 0] = np.log10(np.maximum(Y_pf[:, 0], 1.0))
        Y_hv[:, 2] = np.log10(np.maximum(Y_pf[:, 2], 1e-12))

        ref_hv = ref.copy()
        ref_hv[0] = np.log10(ref[0])
        ref_hv[2] = np.log10(ref[2])

        mask = np.all(Y_hv < ref_hv, axis=1)
        Y_hv = Y_hv[mask]
        if len(Y_hv) == 0:
            return 0.0

        hv_raw = self._hv_3d(Y_hv, ref_hv)
        return hv_raw / self.hv_max

    def compute_hypervolume_raw(self, ref_point: Optional[np.ndarray] = None) -> float:
        """未归一化超体积（供调试）。"""
        ref = ref_point if ref_point is not None else self.ref_point
        _, Y_pf = self.get_pareto_XY()
        if len(Y_pf) == 0:
            return 0.0

        Y_hv = Y_pf.copy()
        Y_hv[:, 0] = np.log10(np.maximum(Y_pf[:, 0], 1.0))
        Y_hv[:, 2] = np.log10(np.maximum(Y_pf[:, 2], 1e-12))

        ref_hv = ref.copy()
        ref_hv[0] = np.log10(ref[0])
        ref_hv[2] = np.log10(ref[2])

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

        sorted_idx = np.argsort(points[:, 0])
        pts = points[sorted_idx]
        hv = 0.0
        front_2d = []

        for i in range(n):
            new_pt = (pts[i, 1], pts[i, 2])
            front_2d = _insert_2d_front(front_2d, new_pt)
            x_width = pts[i + 1, 0] - pts[i, 0] if i < n - 1 else ref[0] - pts[i, 0]
            hv_2d = _compute_2d_hv(front_2d, ref[1], ref[2])
            hv += x_width * hv_2d

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
    ) -> None:
        """每迭代由 optimizer.py 调用，注入当前 Tchebycheff 权重和动态 min/max。"""
        self._w_vec = np.asarray(w_vec, dtype=float).ravel()
        self._eta   = float(eta)
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

        Y_raw = np.array([o.objectives for o in feasible])

        # log₁₀ 变换
        Y_tilde = Y_raw.copy()
        Y_tilde[:, 0] = np.log10(np.maximum(Y_raw[:, 0], 1.0))
        Y_tilde[:, 2] = np.log10(np.maximum(Y_raw[:, 2], 1e-12))

        # 动态归一化（分母过小时用全局范围兜底）
        global_range = np.array([
            np.log10(DEFAULT_REF_POINT[0]) - np.log10(DEFAULT_IDEAL_POINT[0]),
            DEFAULT_REF_POINT[1] - DEFAULT_IDEAL_POINT[1],
            np.log10(DEFAULT_REF_POINT[2]) - np.log10(DEFAULT_IDEAL_POINT[2]),
        ])
        denom = self._y_max - self._y_min
        for i in range(3):
            if denom[i] < 0.05 * global_range[i]:
                denom[i] = global_range[i]
        if self._ideal_point_raw is not None:
            ideal_tilde = self._ideal_point_raw[np.newaxis, :].copy()
            ideal_tilde[:, 0] = np.log10(np.maximum(ideal_tilde[:, 0], 1.0))
            ideal_tilde[:, 2] = np.log10(np.maximum(ideal_tilde[:, 2], 1e-12))
            Y_bar = np.abs(Y_tilde - ideal_tilde[0]) / denom
        else:
            Y_bar = (Y_tilde - self._y_min) / denom

        # Tchebycheff + η tiebreaker
        w  = self._w_vec
        Wf = w[np.newaxis, :] * Y_bar
        F_tch = Wf.max(axis=1) + self._eta * Wf.sum(axis=1)

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
