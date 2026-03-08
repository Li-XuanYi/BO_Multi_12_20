"""
PE-GenBO 实验配置（声明式）
对照实验方案文档中的消融矩阵和Baseline定义
"""

# ============================================================
# 公共常量
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_BUDGET = {"n_warmstart": 5, "n_random_init": 10, "n_iterations": 50}
# 总评估次数 = n_warmstart + n_random_init + n_iterations = 65

REFERENCE_POINT = {"time": 5400.0, "temp": 318.0, "aging": 0.1}

# ============================================================
# 消融变体配置（V0–V6）
# ============================================================
# 每个字典直接传给 LLMMOBO.__init__（加上 llm_api_key 和 budget）
#
# 实验矩阵：
#                WarmStart  PhysicsKernel  GenAcq  LLMWeight  HVFeedback
# V0 (Full)        ✓            ✓           ✓        ✓          ✓
# V1 (No WS)       ✗            ✓           ✓        ✓          ✓
# V2 (No PK)       ✓            ✗           ✓        ✓          ✓
# V3 (No GA)       ✓            ✓           ✗        ✓          ✓
# V4 (No LW)       ✓            ✓           ✓        ✗          ✓
# V5 (No HV)       ✓            ✓           ✓        ✓          ✗
# V6 (Vanilla)     ✗            ✗           ✗        ✗          ✗

ABLATION_CONFIGS = {
    # 约束 C-6: gamma_init 统一为 0.1；V2/V6 无物理核时 gamma_init=0.0 合理
    "V0_Full": {
        "use_warmstart": True,
        "use_coupling": True,
        "use_adaptive_W": True,
        "use_llm_sampling": True,
        "gamma_adaptive": True,
        "gamma_init": 0.1,        # C-6
    },
    "V1_NoWarmStart": {
        "use_warmstart": False,
        "use_coupling": True,
        "use_adaptive_W": True,
        "use_llm_sampling": True,
        "gamma_adaptive": True,
        "gamma_init": 0.1,
    },
    "V2_NoPhysicsKernel": {
        "use_warmstart": True,
        "use_coupling": False,    # 无物理核
        "use_adaptive_W": False,  # 无核则无自适应矩阵
        "use_llm_sampling": True,
        "gamma_adaptive": False,  # γ≡0 无意义
        "gamma_init": 0.0,
    },
    "V3_NoGenAcq": {
        "use_warmstart": True,
        "use_coupling": True,
        "use_adaptive_W": True,
        "use_llm_sampling": False, # 无 LLM 采样，回退随机候选
        "gamma_adaptive": True,
        "gamma_init": 0.1,
    },
    "V4_NoAdaptiveW": {
        "use_warmstart": True,
        "use_coupling": True,
        "use_adaptive_W": False,  # W^(t) 固定为单位矩阵
        "use_llm_sampling": True,
        "gamma_adaptive": True,
        "gamma_init": 0.1,
    },
    "V5_NoHVFeedback": {
        "use_warmstart": True,
        "use_coupling": True,
        "use_adaptive_W": True,
        "use_llm_sampling": True,
        "gamma_adaptive": False,  # γ 固定，无 HV 反馈
        "gamma_init": 0.1,
    },
    "V6_VanillaBO": {
        "use_warmstart": False,
        "use_coupling": False,
        "use_adaptive_W": False,
        "use_llm_sampling": False,
        "gamma_adaptive": False,
        "gamma_init": 0.0,
    },
}

# ============================================================
# Baseline配置
# ============================================================
# type字段决定走哪条执行路径
BASELINE_CONFIGS = {
    "RandomSearch": {
        "type": "random",
        "n_eval": 65,
    },
    "SobolGP": {
        "type": "sobol_gp",
        "n_init": 15,
        "n_iterations": 50,
    },
    "ParEGO": {
        "type": "parego",
        "n_init": 15,
        "n_iterations": 50,
    },
    "NSGA2": {
        "type": "nsga2",
        "pop_size": 10,
        "n_gen": 5,
        "n_init": 15,       # Sobol初始化后再跑进化
    },
    "MOEAD": {
        "type": "moead",
        "pop_size": 10,
        "n_gen": 5,
        "n_init": 15,
    },
    # LLAMBO原始暂缓（Phase 4）
    # "LLAMBO_Original": {
    #     "type": "llambo_original",
    #     "n_init": 15,
    #     "n_iterations": 50,
    # },
}

# ============================================================
# 附加实验配置（Phase 4）
# ============================================================
CONTEXT_LEVELS = ["full", "partial", "none"]

BUDGET_SENSITIVITY = [25, 45, 65, 100, 200]

LLM_MODELS = [
    {"model": "gpt-4o", "base_url": "https://api.nuwaapi.com/v1"},
    {"model": "gpt-3.5-turbo", "base_url": "https://api.nuwaapi.com/v1"},
    {"model": "deepseek-reasoner", "base_url": "https://api.deepseek.com/v1"},
]
