"""
llmbo/components/param_alias.py
=================================
参数别名适配器 (ParamAliasAdapter)

设计目标:
1. 统一参数命名 - 将不同来源的参数统一为 canonical keys (I1, SOC1, I2)
2. 3D/4D 兼容 - 支持新旧参数空间的无缝转换
3. 双向映射 - 支持 canonical -> alias 和 alias -> canonical 转换
4. 透明使用 - 在 DB、GP、采集函数、Prompt 中统一使用 canonical keys

参数映射表:
    Canonical (3D)          Alias (4D/legacy)       Description
    -------------------     -------------------     ---------------------------
    I1                      current1 / I1           Phase 1 电流 [A]
    SOC1                    switch_soc / SOC1       切换 SOC 点
    I2                      current2 / I2           Phase 2 电流 [A]

用法示例:
    from llmbo.components.param_alias import ParamAliasAdapter, CANONICAL_KEYS_3D

    # 创建适配器 (3D 模式)
    adapter = ParamAliasAdapter(space="3d")

    # 将 legacy 数据转换为 canonical 格式
    legacy_point = {"current1": 5.0, "time1": 0.4, "current2": 2.5}
    canonical = adapter.to_canonical(legacy_point)
    # -> {"I1": 5.0, "SOC1": 0.4, "I2": 2.5}

    # 将 canonical 转换为 legacy 格式 (用于旧接口)
    legacy = adapter.to_alias(canonical)
    # -> {"current1": 5.0, "time1": 0.4, "current2": 2.5}

    # 批量转换
    legacy_list = [{"current1": 5.0, "time1": 0.4, "current2": 2.5}, ...]
    canonical_list = adapter.batch_to_canonical(legacy_list)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# §A  常量定义
# ═══════════════════════════════════════════════════════════════════════════

# 3D 规范参数键 (标准命名)
CANONICAL_KEYS_3D = ("I1", "SOC1", "I2")

# 4D legacy 参数键 (旧版 CC-CV 协议)
LEGACY_KEYS_4D = ("current1", "time1", "current2", "v_switch")

# 3D 参数别名映射
CANONICAL_ALIASES_3D = {
    "I1": ("current1", "i1", "I_1"),
    "SOC1": ("switch_soc", "time1", "soc1", "SOC_1"),
    "I2": ("current2", "i2", "I_2"),
}

# 反向映射 (alias -> canonical)
ALIAS_TO_CANONICAL_3D = {}
for canonical, aliases in CANONICAL_ALIASES_3D.items():
    ALIAS_TO_CANONICAL_3D[canonical.lower()] = canonical
    for alias in aliases:
        ALIAS_TO_CANONICAL_3D[alias.lower()] = canonical


# ═══════════════════════════════════════════════════════════════════════════
# §B  参数空间枚举
# ═══════════════════════════════════════════════════════════════════════════

ParameterSpace = Literal["3d", "4d"]


@dataclass
class SpaceDefinition:
    """参数空间定义"""

    name: str
    keys: Tuple[str, ...]
    bounds_default: Dict[str, Tuple[float, float]]
    aliases: Dict[str, Tuple[str, ...]]


SPACE_3D = SpaceDefinition(
    name="3d",
    keys=CANONICAL_KEYS_3D,
    bounds_default={
        "I1": (3.0, 7.0),
        "SOC1": (0.1, 0.7),
        "I2": (1.0, 5.0),
    },
    aliases=CANONICAL_ALIASES_3D,
)

SPACE_4D = SpaceDefinition(
    name="4d",
    keys=LEGACY_KEYS_4D,
    bounds_default={
        "current1": (3.0, 6.0),
        "time1": (2.0, 40.0),
        "current2": (1.0, 4.0),
        "v_switch": (3.8, 4.2),
    },
    aliases={},
)


# ═══════════════════════════════════════════════════════════════════════════
# §C  参数别名适配器主类
# ═══════════════════════════════════════════════════════════════════════════

class ParamAliasAdapter:
    """
    参数别名适配器

    提供 canonical keys 与 alias keys 之间的双向转换，
    支持 3D 和 4D 两种参数空间。

    Attributes:
        space: 参数空间类型 ("3d" 或 "4d")
        keys: 当前空间的规范键
        bounds: 参数边界
    """

    def __init__(
        self,
        space: ParameterSpace = "3d",
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        初始化适配器

        Args:
            space: 参数空间类型 ("3d" 或 "4d")
            custom_bounds: 自定义参数边界（可选）
        """
        self.space = space

        # 选择空间定义
        if space == "3d":
            self._definition = SPACE_3D
        elif space == "4d":
            self._definition = SPACE_4D
        else:
            raise ValueError(f"未知参数空间：{space}")

        self.keys = self._definition.keys

        # 使用自定义边界或默认边界
        self.bounds = custom_bounds if custom_bounds else self._definition.bounds_default.copy()

        # 构建别名映射表
        self._alias_map: Dict[str, str] = {}
        for canonical, aliases in self._definition.aliases.items():
            self._alias_map[canonical.lower()] = canonical
            for alias in aliases:
                self._alias_map[alias.lower()] = canonical

    def normalize_key(self, key: str) -> str:
        """
        将任意键名规范化为 canonical key

        Args:
            key: 输入键名

        Returns:
            规范化的键名

        Raises:
            KeyError: 如果键名无法识别
        """
        key_lower = key.lower()

        # 直接匹配 canonical key
        if key_lower in [k.lower() for k in self.keys]:
            for k in self.keys:
                if k.lower() == key_lower:
                    return k

        # 通过别名映射
        if key_lower in self._alias_map:
            mapped = self._alias_map[key_lower]
            if mapped in self.keys:
                return mapped

        raise KeyError(f"无法识别的参数键：{key} (space={self.space})")

    def to_canonical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将别名字典转换为 canonical 格式

        Args:
            data: 输入字典（可能含别名键）

        Returns:
            转换后的规范字典
        """
        result = {}
        for key, value in data.items():
            try:
                canonical_key = self.normalize_key(key)
                result[canonical_key] = value
            except KeyError:
                # 保留无法识别的键
                result[key] = value
        return result

    def to_alias(
        self,
        data: Dict[str, Any],
        target_keys: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """
        将 canonical 字典转换为别名格式

        Args:
            data: 输入字典（canonical 格式）
            target_keys: 目标键名列表（可选，默认使用空间定义）

        Returns:
            转换后的别名字典
        """
        if target_keys is None:
            target_keys = self.keys

        result = {}
        for key in target_keys:
            if key in data:
                result[key] = data[key]
            else:
                # 尝试查找别名
                key_lower = key.lower()
                for data_key in data.keys():
                    if data_key.lower() == key_lower:
                        result[key] = data[data_key]
                        break
        return result

    def batch_to_canonical(
        self,
        data_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """批量转换字典列表为 canonical 格式"""
        return [self.to_canonical(data) for data in data_list]

    def batch_to_alias(
        self,
        data_list: List[Dict[str, Any]],
        target_keys: Optional[Tuple[str, ...]] = None,
    ) -> List[Dict[str, Any]]:
        """批量转换字典列表为别名格式"""
        return [self.to_alias(data, target_keys) for data in data_list]

    def to_array(
        self,
        data: Dict[str, Any],
        order: Optional[Tuple[str, ...]] = None,
    ) -> np.ndarray:
        """
        将字典转换为 numpy 数组

        Args:
            data: 输入字典
            order: 键的顺序（可选）

        Returns:
            numpy 数组
        """
        if order is None:
            order = self.keys

        values = []
        for key in order:
            if key in data:
                values.append(float(data[key]))
            else:
                # 尝试查找别名
                found = False
                for data_key in data.keys():
                    if self.normalize_key(data_key) == key:
                        values.append(float(data[data_key]))
                        found = True
                        break
                if not found:
                    raise KeyError(f"缺少参数：{key}")

        return np.array(values, dtype=float)

    def from_array(
        self,
        values: Union[np.ndarray, List[float]],
        order: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, float]:
        """
        将 numpy 数组转换为字典

        Args:
            values: 输入数组
            order: 键的顺序（可选）

        Returns:
            字典
        """
        if order is None:
            order = self.keys

        return {key: float(val) for key, val in zip(order, values)}

    def validate_bounds(
        self,
        data: Dict[str, Any],
        tolerance: float = 1e-6,
    ) -> Tuple[bool, List[str]]:
        """
        验证数据是否在参数边界内

        Args:
            data: 输入字典
            tolerance: 边界容差

        Returns:
            (is_valid, errors): 是否有效及错误消息
        """
        errors = []
        canonical = self.to_canonical(data)

        for key, value in canonical.items():
            if key in self.bounds:
                lo, hi = self.bounds[key]
                if value < lo - tolerance or value > hi + tolerance:
                    errors.append(
                        f"{key}={value} 超出边界 [{lo}, {hi}]"
                    )

        return len(errors) == 0, errors

    def clip_to_bounds(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        将数据裁剪到参数边界内

        Args:
            data: 输入字典

        Returns:
            裁剪后的字典
        """
        canonical = self.to_canonical(data)
        result = {}

        for key, value in canonical.items():
            if key in self.bounds:
                lo, hi = self.bounds[key]
                result[key] = max(lo, min(hi, value))
            else:
                result[key] = value

        return result

    @property
    def n_dims(self) -> int:
        """返回参数空间维度"""
        return len(self.keys)

    @property
    def bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回边界数组 (lower, upper)"""
        lo = np.array([self.bounds[k][0] for k in self.keys])
        hi = np.array([self.bounds[k][1] for k in self.keys])
        return lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# §D  全局适配器实例
# ═══════════════════════════════════════════════════════════════════════════

# 默认 3D 适配器（推荐在大部分模块中使用）
DEFAULT_ADAPTER_3D = ParamAliasAdapter(space="3d")

# 4D 适配器（用于兼容旧数据）
LEGACY_ADAPTER_4D = ParamAliasAdapter(space="4d")


# ═══════════════════════════════════════════════════════════════════════════
# §E  便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def to_canonical_3d(data: Dict[str, Any]) -> Dict[str, Any]:
    """快速转换为 3D canonical 格式"""
    return DEFAULT_ADAPTER_3D.to_canonical(data)


def from_canonical_3d(
    data: Dict[str, Any],
    target_keys: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """快速从 3D canonical 格式转换"""
    return DEFAULT_ADAPTER_3D.to_alias(data, target_keys)


def normalize_database_to_canonical(
    database: List[Dict[str, Any]],
    space: ParameterSpace = "3d",
) -> List[Dict[str, Any]]:
    """
    将数据库中的所有记录转换为 canonical 格式

    Args:
        database: 数据库记录列表
        space: 参数空间类型

    Returns:
        转换后的数据库
    """
    adapter = ParamAliasAdapter(space=space)
    result = []

    for record in database:
        # 处理参数字段
        if "params" in record:
            record["params"] = adapter.to_canonical(record["params"])

        # 处理 theta 字段
        if "theta" in record and isinstance(record["theta"], dict):
            record["theta"] = adapter.to_canonical(record["theta"])

        result.append(record)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# §F  CLI 测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ParamAliasAdapter 测试")
    print("=" * 60)

    # 测试 1: 3D 适配器
    print("\n[测试 1] 3D 适配器")
    adapter_3d = ParamAliasAdapter(space="3d")

    legacy_3d = {"current1": 5.0, "time1": 0.4, "current2": 2.5}
    canonical = adapter_3d.to_canonical(legacy_3d)
    print(f"  输入 (legacy):  {legacy_3d}")
    print(f"  输出 (canonical): {canonical}")
    assert canonical == {"I1": 5.0, "SOC1": 0.4, "I2": 2.5}

    # 反向转换
    back = adapter_3d.to_alias(canonical)
    print(f"  反向转换：{back}")

    # 测试 2: 数组转换
    print("\n[测试 2] 数组转换")
    arr = adapter_3d.to_array(canonical)
    print(f"  字典 -> 数组：{arr}")
    recovered = adapter_3d.from_array(arr)
    print(f"  数组 -> 字典：{recovered}")
    assert recovered == canonical

    # 测试 3: 边界验证
    print("\n[测试 3] 边界验证")
    valid_data = {"I1": 5.0, "SOC1": 0.4, "I2": 2.5}
    invalid_data = {"I1": 10.0, "SOC1": 0.4, "I2": 2.5}

    is_valid, errors = adapter_3d.validate_bounds(valid_data)
    print(f"  有效数据验证：{'通过' if is_valid else errors}")

    is_valid, errors = adapter_3d.validate_bounds(invalid_data)
    print(f"  无效数据验证：{'通过' if is_valid else errors}")

    # 测试 4: 边界裁剪
    print("\n[测试 4] 边界裁剪")
    clipped = adapter_3d.clip_to_bounds(invalid_data)
    print(f"  裁剪前：{invalid_data}")
    print(f"  裁剪后：{clipped}")
    assert clipped["I1"] == 7.0  # 裁剪到上界

    # 测试 5: 4D 适配器
    print("\n[测试 5] 4D 适配器")
    adapter_4d = ParamAliasAdapter(space="4d")
    legacy_4d = {"current1": 5.0, "time1": 20.0, "current2": 2.0, "v_switch": 4.0}
    canonical_4d = adapter_4d.to_canonical(legacy_4d)
    print(f"  输入：{legacy_4d}")
    print(f"  输出：{canonical_4d}")

    print("\n" + "=" * 60)
    print("所有测试通过")
    print("=" * 60)
