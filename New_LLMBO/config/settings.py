"""
config/settings.py - 统一配置文件
====================================
所有实验参数只在这里修改！
其他文件通过 `from config.settings import Settings` 导入

═══════════════════════════════════════════════════════════════
  快速开始
═══════════════════════════════════════════════════════════════

1. 修改 LLM 配置（API Key、模型等）:
   ─────────────────────────────────
   打开本文件，找到第 31-40 行的 LLM 类:

   @dataclass(frozen=True)
   class LLM:
       BACKEND = "openai"      # 后端：openai / ollama / anthropic
       MODEL = "gpt-4o"        # 模型名称
       API_KEY = "sk-xxx"      # 你的 API 密钥
       API_BASE = "https://..."  # API 地址

2. 修改优化器配置:
   ─────────────────────────────────
   BO 类（第 85 行起）:
       N_ITERATIONS = 50       # 迭代次数
       N_WARMSTART = 10        # Warmstart 点数

   ParEGO 类（第 165 行起）:
       N_ITERATIONS = 300      # ParEGO 迭代次数

3. 修改 GP/采集函数配置:
   ─────────────────────────────────
   GP 类（第 95 行起）、Acquisition 类（第 115 行起）

═══════════════════════════════════════════════════════════════

用法示例:
    from config.settings import Settings

    # 访问 LLM 配置
    model = Settings.LLM.MODEL
    api_key = Settings.LLM.API_KEY

    # 访问优化配置
    n_iter = Settings.BO.N_ITERATIONS
    n_cands = Settings.ACQUISITION.N_CANDIDATES

    # 访问参数边界
    bounds = Settings.PARAM_BOUNDS.to_dict()

    # 在代码中使用
    llm = LLMInterface(
        backend=Settings.LLM.BACKEND,
        model=Settings.LLM.MODEL,
        api_key=Settings.LLM.API_KEY,
    )
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# 一、LLM 配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class LLM:
    """LLM API 配置"""
    BACKEND: str = "openai"
    MODEL: str = "gpt-3.5-turbo"         # Gemini 企业线路可用模型
    API_KEY: str = "sk-HmCBUaZaKtzEFmFSmGBZb9hIcALBDZFAyhGbyNU5VLB7FMyb"
    API_BASE: str = "https://api.nuwaapi.com/v1"
    TEMPERATURE: float = 0.7
    N_SAMPLES: int = 5
    TIMEOUT: int = 120


# ═══════════════════════════════════════════════════════════════
# 二、PyBaMM 电池仿真配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PyBaMM:
    """PyBaMM 电池仿真配置"""
    BATTERY_MODEL: str = "LG M50 (Chen2020)"
    # PARAM_SET: str = "Chen2020"
    # INIT_VOLTAGE: float = 3.0           # V
    # INIT_TEMP: float = 298.15           # K
    # SAMPLE_TIME: float = 90.0           # s
    # VOLTAGE_MAX: float = 4.2            # V
    # TEMP_MAX: float = 328.15            # K (= 50°C)
    # SOC_TARGET: float = 0.8


# ═══════════════════════════════════════════════════════════════
# 三、参数边界（决策变量）
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ParamBounds:
    """决策变量参数边界"""
    I1: Tuple[float, float] = (3.0, 7.0)
    SOC1: Tuple[float, float] = (0.1, 0.7)
    I2: Tuple[float, float] = (1.0, 5.0)

    def to_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            'I1': self.I1,
            'SOC1': self.SOC1,
            'I2': self.I2,
        }


# ═══════════════════════════════════════════════════════════════
# 四、贝叶斯优化配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BO:
    """贝叶斯优化配置"""
    N_ITERATIONS: int = 20
    N_WARMSTART: int = 10
    N_RANDOM_INIT: int = 10


# ═══════════════════════════════════════════════════════════════
# 五、高斯过程（GP）配置
# ═══════════════════════════════════════════════════════════════
# @dataclass(frozen=True)
# class GP:
#     """高斯过程配置"""
    # KERNEL_TYPE: str = "matern"
    # KERNEL_NU: float = 2.5
    # LENGTH_SCALE: float = 1.0
    # LENGTH_SCALE_BOUNDS: Tuple[float, float] = (1e-2, 1e2)
    # CONSTANT_VALUE: float = 1.0
    # CONSTANT_VALUE_BOUNDS: Tuple[float, float] = (1e-3, 1e3)
    # ALPHA: float = 1e-5
    # N_RESTARTS: int = 5
    # NORMALIZE_Y: bool = True


# ═══════════════════════════════════════════════════════════════
# 六、采集函数（EI）配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Acquisition:
    """采集函数配置"""
    N_CANDIDATES: int = 15          # LLM 生成候选数
    N_SELECT: int = 1               # 选择评估数
    N_MC_SAMPLES: int = 128         # MC-EI 采样数
    KAPPA: float = 0.20             # GP-LCB 置信参数
    EPS_SIGMA: float = 0.001
    RHO: float = 0.1                # 停滞扩展率
    XI: float = 0.05                 # EI 探索奖励


# ═══════════════════════════════════════════════════════════════
# 七、Coupling 配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Coupling:
    """耦合矩阵配置"""
    # GAMMA_INIT: float = 0.1
    GAMMA_MIN: float = 0.001
    GAMMA_MAX: float = 0.3
    GAMMA_T_DECAY: float = 20.0
    ALPHA_MAX: float = 0.7
    ALPHA_MIN: float = 0.05
    T_DECAY_ALPHA: float = 60.0


# ═══════════════════════════════════════════════════════════════
# 八、多目标优化配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MOBO:
    """多目标优化配置"""
    REFERENCE_POINT: Dict[str, float] = None
    IDEAL_POINT: Dict[str, float] = None
    ETA: float = 0.05               # Tchebycheff 增强项系数
    N_WEIGHTS: int = 66             # Riesz 权重集合大小

    def __post_init__(self):
        if self.REFERENCE_POINT is None:
            object.__setattr__(self, 'REFERENCE_POINT', {
                'time': 7200.0, 'temp': 328.15, 'aging': 0.01
            })
        if self.IDEAL_POINT is None:
            object.__setattr__(self, 'IDEAL_POINT', {
                'time': 2500.0, 'temp': 298.15, 'aging': 1e-4
            })


# ═══════════════════════════════════════════════════════════════
# 九、Riesz s-energy 权重配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Riesz:
    """Riesz s-energy 权重配置"""
    N_DIV: int = 10
    S: float = 2.0
    N_ITER: int = 500
    LR: float = 0.005
    SEED: int = 42


# ═══════════════════════════════════════════════════════════════
# 十、检查点与输出配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Output:
    """输出配置"""
    SAVE_DIR: str = "./results"
    CHECKPOINT_EVERY: int = 5
    PLOT_INTERVAL: int = 10


# ═══════════════════════════════════════════════════════════════
# 十一、ParEGO 专用配置
# ═══════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ParEGO:
    """ParEGO 配置"""
    N_ITERATIONS: int = 300
    N_WARMSTART: int = 15
    N_RANDOM_CANDS: int = 500
    GP_N_RESTARTS: int = 5


# ═══════════════════════════════════════════════════════════════
# 全局配置实例（推荐使用方式）
# ═══════════════════════════════════════════════════════════════
class Settings:
    """
    全局配置容器

    用法:
        from config.settings import Settings

        # 访问 LLM 配置
        model = Settings.LLM.MODEL
        api_key = Settings.LLM.API_KEY

        # 访问优化配置
        n_iter = Settings.BO.N_ITERATIONS

        # 访问参数边界
        bounds = Settings.PARAM_BOUNDS.to_dict()
    """
    LLM = LLM()
    PYBAMM = PyBaMM()
    PARAM_BOUNDS = ParamBounds()
    BO = BO()
    # GP = GP()
    ACQUISITION = Acquisition()
    COUPLING = Coupling()
    MOBO = MOBO()
    RIESZ = Riesz()
    OUTPUT = Output()
    PAREGO = ParEGO()


# ═══════════════════════════════════════════════════════════════
# 便捷导入（支持直接导入单个配置类）
# ═══════════════════════════════════════════════════════════════
__all__ = [
    'Settings',
    'LLM',
    'PyBaMM',
    'ParamBounds',
    'BO',
    'GP',
    'Acquisition',
    'Coupling',
    'MOBO',
    'Riesz',
    'Output',
    'ParEGO',
]
