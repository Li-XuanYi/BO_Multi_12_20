"""
database.py  —  评估历史数据库
================================
管理所有已评估的 (θ, y) 对，支持增量追加与持久化存档。

字段约定（与 pybamm_simulator.py 完全对齐）：
    x : np.ndarray (N, 5)  归一化决策向量 [I1,I2,I3,dSOC1,dSOC2] ∈ [0,1]
    y : np.ndarray (N, 3)  目标值 [time_s, delta_temp_K, aging_%]，均 minimize
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


class Database:
    """轻量级评估历史数据库，纯 numpy 实现。"""

    def __init__(self, dim: int = 5, n_obj: int = 3):
        self.dim   = dim
        self.n_obj = n_obj
        self.x: np.ndarray = np.empty((0, dim))
        self.y: np.ndarray = np.empty((0, n_obj))
        self._timestamps: list[float] = []

    # ------------------------------------------------------------------
    #  读写接口
    # ------------------------------------------------------------------

    def add(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        追加一条或多条评估记录。

        Parameters
        ----------
        x_new : (dim,) 或 (M, dim)
        y_new : (n_obj,) 或 (M, n_obj)
        """
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_2d(y_new)
        assert x_new.shape[1] == self.dim,   f"x 维度错误: {x_new.shape[1]} != {self.dim}"
        assert y_new.shape[1] == self.n_obj, f"y 维度错误: {y_new.shape[1]} != {self.n_obj}"
        assert x_new.shape[0] == y_new.shape[0], "x 与 y 行数不一致"

        self.x = np.vstack([self.x, x_new])
        self.y = np.vstack([self.y, y_new])
        self._timestamps.extend([time.time()] * x_new.shape[0])

    def __len__(self) -> int:
        return len(self.x)

    @property
    def n(self) -> int:
        return len(self.x)

    # ------------------------------------------------------------------
    #  持久化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """保存为 JSON（便于跨语言读取）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x":          self.x.tolist(),
            "y":          self.y.tolist(),
            "timestamps": self._timestamps,
            "dim":        self.dim,
            "n_obj":      self.n_obj,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Database":
        """从 JSON 恢复。"""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        db = cls(dim=payload["dim"], n_obj=payload["n_obj"])
        db.x = np.array(payload["x"])
        db.y = np.array(payload["y"])
        db._timestamps = payload.get("timestamps", [])
        return db

    def save_npz(self, path: str | Path) -> None:
        """保存为 npz（速度快，适合大数据量）。"""
        np.savez(path, x=self.x, y=self.y)

    @classmethod
    def load_npz(cls, path: str | Path) -> "Database":
        data = np.load(path)
        db = cls(dim=data["x"].shape[1], n_obj=data["y"].shape[1])
        db.x = data["x"]
        db.y = data["y"]
        return db

    # ------------------------------------------------------------------
    #  统计工具
    # ------------------------------------------------------------------

    def y_min(self) -> np.ndarray:
        """各目标历史最小值，shape=(n_obj,)"""
        return self.y.min(axis=0)

    def y_max(self) -> np.ndarray:
        """各目标历史最大值，shape=(n_obj,)"""
        return self.y.max(axis=0)

    def y_norm(self) -> np.ndarray:
        """目标归一化：(y - min) / (max - min)，shape=(N, n_obj)"""
        lo = self.y_min()
        hi = self.y_max()
        denom = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        return (self.y - lo) / denom

    def summary(self) -> str:
        lines = [
            f"Database | n={self.n} | dim={self.dim} | n_obj={self.n_obj}",
            f"  y min : {self.y_min()}",
            f"  y max : {self.y_max()}",
        ]
        return "\n".join(lines)
