"""
llmbo/config/schema.py
========================
LLM-MOBO 配置数据模型（Pydantic）

设计目标:
1. 类型安全 - 使用 Pydantic 进行运行时类型校验
2. 层级清晰 - 按功能分组（电池/优化/算法/LLM/数据）
3. 可验证 - 提供 validate() 方法检查配置一致性
4. 可序列化 - 支持 to_dict() / from_dict() 用于保存/加载

用法示例:
    from llmbo.config.schema import Config
    cfg = Config()
    cfg.validate()  # 抛出 ValidationError 若配置非法
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo


# ═══════════════════════════════════════════════════════════════════════════
# §A  电池物理参数
# ═══════════════════════════════════════════════════════════════════════════

class BatteryConfig(BaseModel):
    """电池仿真物理参数"""

    param_set: str = Field(
        default="Chen2020",
        description="PyBaMM 模型参数集"
    )

    init_voltage: float = Field(
        default=3.0,
        description="初始电压 [V]",
        ge=0.0, le=5.0
    )

    init_temp: float = Field(
        default=298.15,
        description="初始温度 [K]",
        gt=0.0
    )

    sample_time: float = Field(
        default=90.0,
        description="每步时长 [s]",
        gt=0.0
    )

    voltage_max: float = Field(
        default=4.2,
        description="最大电压 [V]",
        gt=0.0
    )

    temp_max: float = Field(
        default=323.15,
        description="最大温度 [K] (= 45°C)",
        gt=0.0
    )

    soc_target: float = Field(
        default=0.8,
        description="充电目标 SOC",
        ge=0.0, le=1.0
    )

    @model_validator(mode='after')
    def validate_temps(self) -> 'BatteryConfig':
        """验证温度约束：temp_max > init_temp"""
        if self.temp_max <= self.init_temp:
            raise ValueError(
                f"temp_max ({self.temp_max}K) 必须大于 init_temp ({self.init_temp}K)"
            )
        return self


class ParamBounds(BaseModel):
    """决策变量参数边界"""

    I1: Tuple[float, float] = Field(
        default=(3.0, 7.0),
        description="第一阶段电流 [A]"
    )

    SOC1: Tuple[float, float] = Field(
        default=(0.1, 0.7),
        description="切换 SOC 点"
    )

    I2: Tuple[float, float] = Field(
        default=(1.0, 5.0),
        description="第二阶段电流 [A]"
    )

    @field_validator('I1', 'SOC1', 'I2')
    @classmethod
    def validate_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """验证边界有效性"""
        if len(v) != 2:
            raise ValueError("边界必须是 (min, max) 二元组")
        if v[0] >= v[1]:
            raise ValueError(f"下界 {v[0]} 必须小于上界 {v[1]}")
        return v

    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式"""
        return {
            'I1': self.I1,
            'SOC1': self.SOC1,
            'I2': self.I2,
        }


class ChargingRange(BaseModel):
    """充电范围参数"""

    soc0: float = Field(
        default=0.1,
        description="初始 SOC",
        ge=0.0, le=1.0
    )

    soc_end: float = Field(
        default=0.8,
        description="充电终止 SOC",
        ge=0.0, le=1.0
    )

    q_nom: float = Field(
        default=18000,
        description="标称容量 [C]",
        gt=0.0
    )

    @model_validator(mode='after')
    def validate_soc(self) -> 'ChargingRange':
        """验证 SOC 约束"""
        if self.soc_end <= self.soc0:
            raise ValueError(
                f"soc_end ({self.soc_end}) 必须大于 soc0 ({self.soc0})"
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# §B  贝叶斯优化参数
# ═══════════════════════════════════════════════════════════════════════════

class BOConfig(BaseModel):
    """贝叶斯优化参数"""

    n_warmstart: int = Field(
        default=5,
        description="LLM 热启动样本数",
        ge=1
    )

    n_random_init: int = Field(
        default=10,
        description="额外随机初始化样本数",
        ge=0
    )

    n_iterations: int = Field(
        default=50,
        description="BO 迭代次数",
        ge=1
    )

    # C-6 核心超参数
    gamma_init: float = Field(
        default=0.1,
        description="初始耦合强度 gamma",
        gt=0.0
    )

    gamma_min: float = Field(
        default=0.001,
        description="gamma 下限",
        gt=0.0
    )

    gamma_max: float = Field(
        default=2.0,
        description="gamma 上限",
        gt=0.0
    )

    gamma_update_rate: float = Field(
        default=0.1,
        description="gamma 更新率 rho",
        ge=0.0, le=1.0
    )

    # 采集函数超参数
    alpha_max: float = Field(
        default=0.7,
        description="Alpha 上限",
        ge=0.0
    )

    alpha_min: float = Field(
        default=0.05,
        description="Alpha 下限",
        ge=0.0
    )

    t_decay_alpha: int = Field(
        default=60,
        description="Alpha 衰减时间常数",
        ge=1
    )

    kappa: float = Field(
        default=0.20,
        description="Kappa 参数",
        ge=0.0
    )

    eps_sigma: float = Field(
        default=0.001,
        description="EPS_SIGMA 参数",
        gt=0.0
    )

    rho: float = Field(
        default=0.1,
        description="停滞扩展率 rho",
        ge=0.0, le=1.0
    )

    @model_validator(mode='after')
    def validate_gamma(self) -> 'BOConfig':
        """验证 gamma 参数"""
        if self.gamma_min >= self.gamma_max:
            raise ValueError(
                f"gamma_min ({self.gamma_min}) 必须小于 gamma_max ({self.gamma_max})"
            )
        if not (self.gamma_min <= self.gamma_init <= self.gamma_max):
            raise ValueError(
                f"gamma_init ({self.gamma_init}) 必须在 [{self.gamma_min}, {self.gamma_max}] 内"
            )
        return self

    @model_validator(mode='after')
    def validate_alpha(self) -> 'BOConfig':
        """验证 alpha 参数"""
        if self.alpha_min >= self.alpha_max:
            raise ValueError(
                f"alpha_min ({self.alpha_min}) 必须小于 alpha_max ({self.alpha_max})"
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# §C  算法参数（GP/采集函数/核函数）
# ═══════════════════════════════════════════════════════════════════════════

class GPConfig(BaseModel):
    """GP 超参数配置"""

    kernel_type: str = Field(
        default="matern",
        description="核函数类型"
    )

    kernel_nu: float = Field(
        default=2.5,
        description="Matern 核的 nu 参数",
        gt=0.0
    )

    kernel_length_scale: float = Field(
        default=1.0,
        description="长度尺度初始值",
        gt=0.0
    )

    kernel_length_scale_bounds: Tuple[float, float] = Field(
        default=(1e-2, 1e2),
        description="长度尺度边界"
    )

    constant_value: float = Field(
        default=1.0,
        description="常数核系数",
        gt=0.0
    )

    constant_value_bounds: Tuple[float, float] = Field(
        default=(1e-3, 1e3),
        description="常数核边界"
    )

    alpha: float = Field(
        default=1e-5,
        description="噪声水平",
        gt=0.0
    )

    n_restarts_optimizer: int = Field(
        default=5,
        description="超参数优化重启次数",
        ge=0
    )

    normalize_y: bool = Field(
        default=True,
        description="是否归一化目标值"
    )


class CompositeKernelConfig(BaseModel):
    """复合核参数"""

    gamma_init: float = Field(
        default=0.1,
        description="初始耦合强度",
        gt=0.0
    )

    gamma_bounds: Tuple[float, float] = Field(
        default=(0.001, 2.0),
        description="gamma 边界"
    )

    use_coupling: bool = Field(
        default=True,
        description="是否使用耦合核"
    )

    coupling_matrix_alpha: float = Field(
        default=0.95,
        description="W 融合权重 (data vs LLM)",
        ge=0.0, le=1.0
    )

    phi_length_scale_init: float = Field(
        default=1.0,
        description="phi_j 内部长度尺度初始值",
        gt=0.0
    )

    eps_psd: float = Field(
        default=1e-5,
        description="PSD 特征值下限",
        gt=0.0
    )


class AcquisitionConfig(BaseModel):
    """采集函数参数"""

    n_cand: int = Field(
        default=15,
        description="LLM 生成候选数 (N_cand)",
        ge=1
    )

    n_select: int = Field(
        default=3,
        description="选择评估数 (N_select)",
        ge=1
    )

    n_candidates: int = Field(
        default=2000,
        description="随机海选点数",
        ge=1
    )

    n_mc_samples: int = Field(
        default=128,
        description="MC-EI 采样数",
        ge=1
    )

    n_top_local: int = Field(
        default=5,
        description="局部优化的 top 点数",
        ge=1
    )

    local_maxiter: int = Field(
        default=20,
        description="局部优化最大迭代",
        ge=1
    )

    local_ftol: float = Field(
        default=1e-6,
        description="局部优化收敛阈值",
        gt=0.0
    )

    @model_validator(mode='after')
    def validate_n_select(self) -> 'AcquisitionConfig':
        """验证 n_select <= n_cand"""
        if self.n_select > self.n_cand:
            raise ValueError(
                f"n_select ({self.n_select}) 必须 <= n_cand ({self.n_cand})"
            )
        return self


class GradientConfig(BaseModel):
    """梯度估计参数"""

    epsilon: float = Field(
        default=1e-4,
        description="数值梯度步长",
        gt=0.0
    )

    n_samples: int = Field(
        default=10,
        description="耦合矩阵估计样本数",
        ge=1
    )

    method: str = Field(
        default="outer_product",
        description="估计方法"
    )


# ═══════════════════════════════════════════════════════════════════════════
# §D  多目标优化参数
# ═══════════════════════════════════════════════════════════════════════════

class MOBOConfig(BaseModel):
    """多目标优化参数"""

    n_weights: int = Field(
        default=15,
        description="Riesz s-energy 权重集合大小",
        ge=1
    )

    dirichlet_alpha: List[float] = Field(
        default=[1.0, 1.0, 1.0],
        description="Dirichlet 分布参数"
    )

    eta: float = Field(
        default=0.05,
        description="Tchebycheff 增强项系数",
        ge=0.0
    )

    reference_point: Dict[str, float] = Field(
        default_factory=lambda: {
            'time': 7200.0,
            'temp': 328.15,
            'aging': 0.01,
        },
        description="参考点（最坏情况）"
    )

    ideal_point: Dict[str, float] = Field(
        default_factory=lambda: {
            'time': 2500.0,
            'temp': 298.15,
            'aging': 1e-4,
        },
        description="理想点（最好情况）"
    )

    @field_validator('dirichlet_alpha')
    @classmethod
    def validate_dirichlet(cls, v: List[float]) -> List[float]:
        """验证 Dirichlet 参数"""
        if len(v) != 3:
            raise ValueError("dirichlet_alpha 必须是 3 元列表")
        if any(a <= 0 for a in v):
            raise ValueError("dirichlet_alpha 所有元素必须 > 0")
        return v

    @model_validator(mode='after')
    def validate_points(self) -> 'MOBOConfig':
        """验证参考点和理想点"""
        for key in ['time', 'temp', 'aging']:
            if self.ideal_point[key] >= self.reference_point[key]:
                raise ValueError(
                    f"ideal_point[{key}] ({self.ideal_point[key]}) "
                    f"必须 < reference_point[{key}] ({self.reference_point[key]})"
                )
        if self.ideal_point['aging'] <= 0:
            raise ValueError("ideal_point['aging'] 必须 > 0 (for log10)")
        return self


# ═══════════════════════════════════════════════════════════════════════════
# §E  LLM 参数
# ═══════════════════════════════════════════════════════════════════════════

class LLMWarmStartConfig(BaseModel):
    """LLM WarmStart 专用配置"""

    temperature: float = Field(
        default=0.7,
        description="采样温度",
        ge=0.0, le=2.0
    )

    max_tokens: int = Field(
        default=2500,
        description="最大 token 数",
        ge=1
    )

    context_level: str = Field(
        default="full",
        description="上下文级别 (full/partial/none)",
        pattern="^(full|partial|none)$"
    )

    max_retries: int = Field(
        default=3,
        description="最大重试次数",
        ge=0
    )

    retry_backoff_base: float = Field(
        default=2.0,
        description="指数退避基数",
        gt=0.0
    )


class LLMCouplingConfig(BaseModel):
    """LLM Coupling Inference 专用配置"""

    temperature: float = Field(
        default=0.2,
        description="采样温度",
        ge=0.0, le=2.0
    )

    max_tokens: int = Field(
        default=800,
        description="最大 token 数",
        ge=1
    )

    max_retries: int = Field(
        default=3,
        description="最大重试次数",
        ge=0
    )

    retry_backoff_base: float = Field(
        default=2.0,
        description="指数退避基数",
        gt=0.0
    )


class LLMWeightingConfig(BaseModel):
    """LLM Weighting 专用配置"""

    temperature: float = Field(
        default=0.3,
        description="采样温度",
        ge=0.0, le=2.0
    )

    max_tokens: int = Field(
        default=500,
        description="最大 token 数",
        ge=1
    )

    sigma_scale: float = Field(
        default=0.15,
        description="sigma 缩放因子",
        gt=0.0
    )

    update_interval: int = Field(
        default=5,
        description="每 N 轮更新一次焦点",
        ge=1
    )

    min_pareto_points: int = Field(
        default=3,
        description="最少 Pareto 点数",
        ge=1
    )

    # ARD 长度尺度驱动
    length_scale_base: float = Field(
        default=0.5,
        description="基础长度尺度 l_base",
        gt=0.0
    )

    length_scale_alpha: float = Field(
        default=0.8,
        description="敏感度指数 alpha_l",
        ge=0.0, le=1.0
    )

    # 敏感度排序推理
    sensitivity_temperature: float = Field(
        default=0.2,
        description="敏感度推理温度",
        ge=0.0, le=2.0
    )

    sensitivity_max_tokens: int = Field(
        default=400,
        description="敏感度推理最大 token 数",
        ge=1
    )

    sensitivity_max_retries: int = Field(
        default=3,
        description="敏感度推理最大重试次数",
        ge=0
    )


class LLMAcquisitionConfig(BaseModel):
    """LLM Acquisition 专用配置"""

    n_cand: int = Field(
        default=15,
        description="LLM 生成候选数",
        ge=1
    )

    n_select: int = Field(
        default=3,
        description="选择评估数",
        ge=1
    )

    beta: float = Field(
        default=2.0,
        description="GP-LCB 置信参数",
        gt=0.0
    )

    n_batch: int = Field(
        default=20,
        description="单次 LLM 生成的候选点数",
        ge=1
    )

    gen_temperatures: List[float] = Field(
        default=[0.6, 0.8, 1.0],
        description="多温度生成以增加多样性"
    )

    gen_max_tokens: int = Field(
        default=4000,
        description="候选生成最大 tokens",
        ge=1
    )

    gen_max_retries: int = Field(
        default=3,
        description="生成重试次数",
        ge=0
    )

    # 阈值退火参数
    threshold_gamma_0: float = Field(
        default=50.0,
        description="初始百分位 gamma_0",
        ge=0.0, le=100.0
    )

    threshold_gamma_t: float = Field(
        default=99.0,
        description="最终百分位 gamma_T",
        ge=0.0, le=100.0
    )

    b_explore_ratio_init: float = Field(
        default=0.1,
        description="初始高方差区域比例",
        ge=0.0, le=1.0
    )

    @model_validator(mode='after')
    def validate_thresholds(self) -> 'LLMAcquisitionConfig':
        """验证阈值"""
        if self.threshold_gamma_0 >= self.threshold_gamma_t:
            raise ValueError(
                f"threshold_gamma_0 ({self.threshold_gamma_0}) "
                f"必须 < threshold_gamma_t ({self.threshold_gamma_t})"
            )
        return self


class LLMConfig(BaseModel):
    """LLM 通用配置"""

    # API 配置
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            'LLM_API_KEY',
            'sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK'
        ),
        description="API 密钥"
    )

    base_url: str = Field(
        default_factory=lambda: os.getenv(
            'LLM_BASE_URL',
            'https://api.nuwaapi.com/v1'
        ),
        description="API 基础 URL"
    )

    model: str = Field(
        default_factory=lambda: os.getenv('LLM_MODEL', 'gpt-4o'),
        description="模型名称"
    )

    # 功能开关
    enable_warmstart: bool = Field(
        default=True,
        description="启用 WarmStart"
    )

    enable_coupling_inference: bool = Field(
        default=True,
        description="启用耦合推理"
    )

    enable_llm_weighting: bool = Field(
        default=True,
        description="启用 LLM 加权"
    )

    # 速率限制
    rate_limit_tokens: int = Field(
        default=100000,
        description="每分钟 token 数",
        ge=1
    )

    rate_limit_requests: int = Field(
        default=500,
        description="每分钟请求数",
        ge=1
    )

    # 子模块配置
    warmstart: LLMWarmStartConfig = Field(
        default_factory=LLMWarmStartConfig
    )

    coupling: LLMCouplingConfig = Field(
        default_factory=LLMCouplingConfig
    )

    weighting: LLMWeightingConfig = Field(
        default_factory=LLMWeightingConfig
    )

    acquisition: LLMAcquisitionConfig = Field(
        default_factory=LLMAcquisitionConfig
    )


# ═══════════════════════════════════════════════════════════════════════════
# §F  数据配置
# ═══════════════════════════════════════════════════════════════════════════

class DataConfig(BaseModel):
    """数据存储配置"""

    save_dir: str = Field(
        default="./results",
        description="保存目录"
    )

    save_interval: int = Field(
        default=5,
        description="每 N 轮保存检查点",
        ge=1
    )

    plot_interval: int = Field(
        default=10,
        description="每 N 轮绘图",
        ge=1
    )

    enable_log_aging: bool = Field(
        default=True,
        description="是否对 aging 进行 Log10 变换"
    )

    log_transform_min: float = Field(
        default=1e-6,
        description="Log 变换的最小值 (避免 log(0))",
        gt=0.0
    )


# ═══════════════════════════════════════════════════════════════════════════
# §G  全局常量
# ═══════════════════════════════════════════════════════════════════════════

class GlobalConstants(BaseModel):
    """全局常量（约束 C-6）"""

    psi_r1: float = Field(
        default=0.01,
        description="[Ohm] 焦耳热代理函数 R1_bar"
    )

    psi_r2: float = Field(
        default=0.01,
        description="[Ohm] 焦耳热代理函数 R2_bar"
    )

    gamma_init: float = Field(
        default=0.1,
        description="初始耦合强度"
    )


# ═══════════════════════════════════════════════════════════════════════════
# §H  主配置类
# ═══════════════════════════════════════════════════════════════════════════

class Config(BaseModel):
    """
    LLM-MOBO 主配置类

    所有配置项的顶层容器，提供统一的验证和序列化接口。
    """

    battery: BatteryConfig = Field(
        default_factory=BatteryConfig,
        description="电池物理参数"
    )

    param_bounds: ParamBounds = Field(
        default_factory=ParamBounds,
        description="决策变量参数边界"
    )

    charging_range: ChargingRange = Field(
        default_factory=ChargingRange,
        description="充电范围参数"
    )

    bo: BOConfig = Field(
        default_factory=BOConfig,
        description="贝叶斯优化参数"
    )

    gp: GPConfig = Field(
        default_factory=GPConfig,
        description="GP 超参数"
    )

    composite_kernel: CompositeKernelConfig = Field(
        default_factory=CompositeKernelConfig,
        description="复合核参数"
    )

    acquisition: AcquisitionConfig = Field(
        default_factory=AcquisitionConfig,
        description="采集函数参数"
    )

    gradient: GradientConfig = Field(
        default_factory=GradientConfig,
        description="梯度估计参数"
    )

    mobo: MOBOConfig = Field(
        default_factory=MOBOConfig,
        description="多目标优化参数"
    )

    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM 配置"
    )

    data: DataConfig = Field(
        default_factory=DataConfig,
        description="数据配置"
    )

    constants: GlobalConstants = Field(
        default_factory=GlobalConstants,
        description="全局常量"
    )

    def validate(self) -> List[str]:
        """
        验证配置一致性

        Returns:
            错误消息列表（空列表表示验证通过）
        """
        errors = []

        # 检查 1: temp_max 一致性
        temp_max_battery = self.battery.temp_max
        temp_max_ref = self.mobo.reference_point['temp']
        if temp_max_battery > temp_max_ref:
            errors.append(
                f"Warning: battery temp_max ({temp_max_battery}K) > "
                f"reference temp ({temp_max_ref}K)"
            )

        # 检查 2: gamma 一致性
        gamma_bo = self.bo.gamma_init
        gamma_kernel = self.composite_kernel.gamma_init
        if abs(gamma_bo - gamma_kernel) > 1e-6:
            errors.append(
                f"Warning: gamma_init 不一致："
                f"BO_CONFIG={gamma_bo}, Kernel={gamma_kernel}"
            )

        # 检查 3: n_cand 一致性
        n_cand_acq = self.acquisition.n_cand
        n_cand_llm = self.llm.acquisition.n_cand
        if n_cand_acq != n_cand_llm:
            errors.append(
                f"Warning: n_cand 不一致："
                f"Acquisition={n_cand_acq}, LLM={n_cand_llm}"
            )

        # 检查 4: n_select 一致性
        n_select_acq = self.acquisition.n_select
        n_select_llm = self.llm.acquisition.n_select
        if n_select_acq != n_select_llm:
            errors.append(
                f"Warning: n_select 不一致："
                f"Acquisition={n_select_acq}, LLM={n_select_llm}"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """从字典创建"""
        return cls.model_validate(data)

    def to_json(self, path: str) -> None:
        """保存到 JSON 文件"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """从 JSON 文件加载"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def model_post_init(self, __context: Any) -> None:
        """Post-init 验证"""
        errors = self.validate()
        if errors:
            import warnings
            for error in errors:
                warnings.warn(error, UserWarning)


# ═══════════════════════════════════════════════════════════════════════════
# §I  便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def create_minimal_config(
    n_iterations: int = 10,
    n_warmstart: int = 3,
    n_candidates: int = 5,
) -> Config:
    """
    创建最小化配置（用于快速测试）

    Args:
        n_iterations: 迭代次数
        n_warmstart: warmstart 样本数
        n_candidates: 候选点数

    Returns:
        最小化配置对象
    """
    return Config(
        bo=BOConfig(
            n_iterations=n_iterations,
            n_warmstart=n_warmstart,
            n_random_init=5,
        ),
        acquisition=AcquisitionConfig(
            n_cand=n_candidates,
            n_select=2,
        ),
        llm=LLMConfig(
            acquisition=LLMAcquisitionConfig(
                n_cand=n_candidates,
                n_select=2,
                n_batch=n_candidates,
            ),
        ),
    )
