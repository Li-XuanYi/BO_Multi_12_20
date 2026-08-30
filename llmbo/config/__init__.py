"""
llmbo/config/__init__.py
==========================
LLM-MOBO 配置模块

导出:
    - Config: 主配置类（Pydantic 模型）
    - load_config: 配置加载函数
    - save_config: 配置保存函数
    - validate_config: 配置验证函数
    - generate_config_template: 配置模板生成
    - parse_cli_overrides: CLI 参数解析

用法示例:
    from llmbo.config import Config, load_config

    # 方式 1: 使用默认配置
    cfg = Config()

    # 方式 2: 从文件加载
    cfg = load_config("config.json")

    # 方式 3: 显式覆盖
    cfg = load_config(overrides={
        "bo": {"n_iterations": 100},
        "acquisition": {"n_cand": 20},
    })
"""

from .schema import (
    Config,
    BatteryConfig,
    ParamBounds,
    ChargingRange,
    BOConfig,
    GPConfig,
    CompositeKernelConfig,
    AcquisitionConfig,
    GradientConfig,
    MOBOConfig,
    LLMConfig,
    LLMWarmStartConfig,
    LLMCouplingConfig,
    LLMWeightingConfig,
    LLMAcquisitionConfig,
    DataConfig,
    GlobalConstants,
    get_default_config,
    create_minimal_config,
)

from .load import (
    load_config,
    save_config,
    validate_config,
    generate_config_template,
    parse_cli_overrides,
)

__all__ = [
    # 主配置类
    "Config",
    # 子配置类
    "BatteryConfig",
    "ParamBounds",
    "ChargingRange",
    "BOConfig",
    "GPConfig",
    "CompositeKernelConfig",
    "AcquisitionConfig",
    "GradientConfig",
    "MOBOConfig",
    "LLMConfig",
    "LLMWarmStartConfig",
    "LLMCouplingConfig",
    "LLMWeightingConfig",
    "LLMAcquisitionConfig",
    "DataConfig",
    "GlobalConstants",
    # 便捷函数
    "get_default_config",
    "create_minimal_config",
    # 加载/保存
    "load_config",
    "save_config",
    "validate_config",
    "generate_config_template",
    "parse_cli_overrides",
]
