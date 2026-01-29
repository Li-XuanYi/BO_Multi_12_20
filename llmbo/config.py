"""
LLM-MOBO 全局配置文件（重构版）

配置层级：
1. 物理参数 - 电池仿真相关
2. 优化参数 - BO算法参数
3. 算法参数 - GP/采集/核函数参数
4. LLM参数 - 分模块的LLM配置
5. 数据参数 - 保存/可视化
"""

import os

# ============================================================
# 1. 物理参数（BATTERY_CONFIG）
# ============================================================
BATTERY_CONFIG = {
    # PyBaMM模型
    'param_set': 'Chen2020',
    
    # 初始状态
    'init_voltage': 3.0,      # 初始电压 [V]
    'init_temp': 298.15,      # 初始温度 [K] = 25°C
    'sample_time': 90.0,      # 每步时长 [s] (原来30*3=90)
    
    # 安全约束（全局统一）
    'voltage_max': 4.2,       # 最大电压 [V]
    'temp_max': 316.0,        # 最大温度 [K] = 43°C
    'soc_target': 0.8,        # 充电目标SOC
}

# ============================================================
# 2. 充电策略参数空间（PARAM_BOUNDS）
# ============================================================
PARAM_BOUNDS = {
    'current1': (3.0, 6.0),           # 第一阶段电流 [A]
    'switch_soc': (0.3, 0.7),         # SOC切换阈值 [0-1]
    'current2': (1.0, 4.0),           # 第二阶段电流 [A]
}

# ============================================================
# 3. 老化计算参数（AGING_CONFIG）
# ============================================================
AGING_CONFIG = {
    'k_sei': 5e-5,           # SEI生长速率常数 [%/(A·h)]
    'E_a': 30000,            # 活化能 [J/mol]
    'R': 8.314,              # 气体常数 [J/(mol·K)]
    'T_ref': 298.15,         # 参考温度 [K]
}

# ============================================================
# 4. 贝叶斯优化参数（BO_CONFIG）
# ============================================================
BO_CONFIG = {
    # 初始化策略
    'n_warmstart': 5,         # LLM热启动样本数
    'n_random_init': 10,      # 额外随机初始化
    'n_iterations': 50,       # BO迭代次数
    
    # 初始耦合强度
    'gamma_init': 0.5,        # 初始gamma值
    'gamma_min': 0.1,         # gamma下限
    'gamma_max': 2.0,         # gamma上限
    'gamma_update_rate': 0.1, # gamma自适应调整率
}

# ============================================================
# 5. 算法参数（ALGORITHM_CONFIG）
# ============================================================
ALGORITHM_CONFIG = {
    # 5.1 GP核函数参数
    'gp': {
        'kernel_type': 'matern',           # 核类型
        'kernel_nu': 2.5,                  # Matern核的nu参数
        'kernel_length_scale': 1.0,        # 长度尺度初始值
        'kernel_length_scale_bounds': (1e-2, 1e2),
        'constant_value': 1.0,             # 常数核系数
        'constant_value_bounds': (1e-3, 1e3),
        'alpha': 1e-6,                     # 噪声水平
        'n_restarts_optimizer': 5,         # 超参数优化重启次数
        'normalize_y': True,               # 是否归一化y
    },
    
    # 5.2 复合核参数
    'composite_kernel': {
        'gamma_init': 0.5,                 # 耦合强度
        'gamma_bounds': (0.1, 2.0),        # gamma调整范围
        'use_coupling': True,              # 是否使用耦合核
        'coupling_matrix_alpha': 0.5,      # W融合权重（data vs LLM）
    },
    
    # 5.3 采集函数参数
    'acquisition': {
        'n_candidates': 2000,              # 随机海选点数
        'n_mc_samples': 128,               # MC-EI采样数
        'n_top_local': 5,                  # 局部优化的top点数
        'local_maxiter': 20,               # 局部优化最大迭代
        'local_ftol': 1e-6,                # 局部优化收敛阈值
    },
    
    # 5.4 梯度估计参数
    'gradient': {
        'epsilon': 1e-4,                   # 数值梯度步长
        'n_samples': 10,                   # 耦合矩阵估计样本数
        'method': 'outer_product',         # 估计方法
    },
}
# ============================================================
# 6. 多目标优化参数（MOBO_CONFIG）
# ============================================================
MOBO_CONFIG = {
    # Dirichlet分布参数（均匀采样）
    'dirichlet_alpha': [1.0, 1.0, 1.0],
    
    # Tchebycheff标量化参数
    'eta': 0.05,  # 增强项系数
    
    # 参考点（最坏情况，用于归一化）
    'reference_point': {
        'time': 150,      # 最多150步
        'temp': 318.0,    # 最高45°C
        'aging': 0.1,     # 最大老化10%
    },
    
    # 理想点（最好情况）
    'ideal_point': {
        'time': 10,       # 最快10步
        'temp': 298.15,   # 室温25°C
        'aging': 1e-6,    # 接近无老化（不能为0，log10问题）
    }
}

# ============================================================
# 7. LLM 配置（LLM_CONFIG）
# ============================================================
LLM_CONFIG = {
    # 7.1 通用配置
    'api_key': os.getenv('LLM_API_KEY', None),  # 从环境变量读取
    'base_url': 'https://api.nuwaapi.com/v1',
    'model': 'gpt-4o',
    
    # 功能开关
    'enable_warmstart': True,
    'enable_coupling_inference': True,
    'enable_llm_weighting': True,
    
    # 速率限制
    'rate_limit_tokens': 100000,   # 每分钟token数
    'rate_limit_requests': 500,    # 每分钟请求数
    
    # 7.2 WarmStart专用配置
    'warmstart': {
        'temperature': 0.7,            # 较高温度，增加多样性
        'max_tokens': 2500,            # 需要生成多个策略
        'context_level': 'full',       # 上下文级别 (full/partial/none)
        'max_retries': 3,              # 最大重试次数
        'retry_backoff_base': 2,       # 指数退避基数
    },
    
    # 7.3 Coupling Inference专用配置
    'coupling': {
        'temperature': 0.2,            # 低温度，确保稳定输出
        'max_tokens': 800,             # 只需输出矩阵
        'max_retries': 3,
        'retry_backoff_base': 2,
    },
    
    # 7.4 LLM Weighting专用配置
    'weighting': {
        'temperature': 0.3,            # 中等温度
        'max_tokens': 500,             # 输出焦点参数
        'sigma_scale': 0.15,           # σ缩放因子
        'update_interval': 5,          # 每N轮更新一次焦点
        'min_pareto_points': 3,        # 最少Pareto点数
    },
}

# ============================================================
# 8. 数据配置（DATA_CONFIG）
# ============================================================
DATA_CONFIG = {
    'save_dir': './results',
    'save_interval': 5,      # 每5轮保存检查点
    'plot_interval': 10,     # 每10轮绘图
    
    # 数据变换
    'enable_log_aging': True,     # 是否对aging进行Log10变换
    'log_transform_min': 1e-6,    # Log变换的最小值（避免log(0)）
}

# ============================================================
# 9. 辅助函数
# ============================================================
def get_algorithm_param(module: str, param: str, default=None):
    """
    安全获取算法参数
    
    使用示例：
        n_candidates = get_algorithm_param('acquisition', 'n_candidates', 2000)
    """
    try:
        return ALGORITHM_CONFIG[module][param]
    except KeyError:
        return default

def get_llm_param(module: str, param: str, default=None):
    """
    安全获取LLM参数
    
    使用示例：
        temp = get_llm_param('warmstart', 'temperature', 0.7)
    """
    try:
        return LLM_CONFIG[module][param]
    except KeyError:
        return default

def validate_config():
    """
    验证配置的一致性
    
    返回：
        (is_valid, errors)
    """
    errors = []
    
    # 检查1: temp_max一致性
    temp_max_battery = BATTERY_CONFIG['temp_max']
    temp_max_ref = MOBO_CONFIG['reference_point']['temp']
    if temp_max_battery > temp_max_ref:
        errors.append(f"Warning: battery temp_max ({temp_max_battery}K) > reference temp ({temp_max_ref}K)")
    
    # 检查2: aging最小值
    if MOBO_CONFIG['ideal_point']['aging'] <= 0:
        errors.append(f"Error: ideal aging must be > 0 (for log10 transform)")
    
    # 检查3: API密钥
    if LLM_CONFIG['api_key'] is None:
        errors.append(f"Warning: LLM_API_KEY not set, LLM features will be disabled")
    
    # 检查4: gamma一致性
    gamma_bo = BO_CONFIG['gamma_init']
    gamma_kernel = ALGORITHM_CONFIG['composite_kernel']['gamma_init']
    if gamma_bo != gamma_kernel:
        errors.append(f"Warning: gamma_init inconsistent: BO={gamma_bo}, Kernel={gamma_kernel}")
    
    return len(errors) == 0, errors