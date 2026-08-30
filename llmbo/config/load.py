"""
llmbo/config/load.py
======================
LLM-MOBO 配置加载器

设计目标:
1. 灵活加载 - 支持从 JSON 文件 / 字典 / 环境变量加载配置
2. 合并覆盖 - 支持多层级配置覆盖（文件 < 字典 < 命令行）
3. 严格校验 - 使用 Pydantic 进行运行时类型和值校验
4. 显式注入 - 返回 Config 对象供 Optimizer 显式使用

用法示例:
    from llmbo.config.load import load_config, Config

    # 从 JSON 文件加载
    cfg = load_config("config.json")

    # 从字典覆盖
    cfg = load_config(overrides={"bo": {"n_iterations": 100}})

    # 命令行覆盖
    cfg = load_config(
        config_path="config.json",
        overrides={"n_iterations": 100},
    )

    # 显式注入 Optimizer
    from llmbo.optimizer import Optimizer
    opt = Optimizer(cfg)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
    DataConfig,
    GlobalConstants,
)


# ═══════════════════════════════════════════════════════════════════════════
# §A  配置加载主函数
# ═══════════════════════════════════════════════════════════════════════════

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> Config:
    """
    加载配置

    加载优先级（后者覆盖前者）:
        1. 默认配置（schema.py 中的 default_factory）
        2. JSON 文件（如果指定 config_path）
        3. 环境变量（LLM_API_KEY, LLM_MODEL 等）
        4. 字典覆盖（如果指定 overrides）

    Args:
        config_path: JSON 配置文件路径（可选）
        overrides: 字典形式的配置覆盖（可选）
        strict: 严格模式，若为 True 则校验失败时抛出异常

    Returns:
        Config: 配置对象

    Raises:
        ValidationError: 当 strict=True 且配置校验失败时
        FileNotFoundError: 当 config_path 指定但文件不存在时
    """
    # Step 1: 从默认值开始
    config_data: Dict[str, Any] = {}

    # Step 2: 从 JSON 文件加载（如果指定）
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        config_data = _deep_merge(config_data, file_data)

    # Step 3: 从环境变量加载 LLM 相关配置
    env_data = _load_from_env()
    config_data = _deep_merge(config_data, env_data)

    # Step 4: 应用字典覆盖（如果指定）
    if overrides is not None:
        config_data = _deep_merge(config_data, overrides)

    # Step 5: 创建并验证 Config 对象
    try:
        config = Config.from_dict(config_data)
    except Exception as e:
        if strict:
            raise
        # 非严格模式：记录警告并返回默认配置
        import warnings
        warnings.warn(f"配置加载失败：{e}，使用默认配置")
        config = Config()

    return config


# ═══════════════════════════════════════════════════════════════════════════
# §B  环境变量加载
# ═══════════════════════════════════════════════════════════════════════════

def _load_from_env() -> Dict[str, Any]:
    """
    从环境变量加载配置

    支持的环境变量:
        LLM_API_KEY         - API 密钥
        LLM_BASE_URL        - API 基础 URL
        LLM_MODEL           - 模型名称
        LLM_TEMPERATURE     - 默认温度
        LLM_RATE_LIMIT_TOKENS    - token 速率限制
        LLM_RATE_LIMIT_REQUESTS  - 请求速率限制

    Returns:
        包含环境变量配置的字典
    """
    env_config: Dict[str, Any] = {}

    # LLM 通用配置
    llm_config: Dict[str, Any] = {}

    if api_key := os.getenv('LLM_API_KEY'):
        llm_config['api_key'] = api_key

    if base_url := os.getenv('LLM_BASE_URL'):
        llm_config['base_url'] = base_url

    if model := os.getenv('LLM_MODEL'):
        llm_config['model'] = model

    if temperature := os.getenv('LLM_TEMPERATURE'):
        try:
            llm_config['warmstart'] = {'temperature': float(temperature)}
        except ValueError:
            pass

    if rate_tokens := os.getenv('LLM_RATE_LIMIT_TOKENS'):
        try:
            llm_config['rate_limit_tokens'] = int(rate_tokens)
        except ValueError:
            pass

    if rate_requests := os.getenv('LLM_RATE_LIMIT_REQUESTS'):
        try:
            llm_config['rate_limit_requests'] = int(rate_requests)
        except ValueError:
            pass

    if llm_config:
        env_config['llm'] = llm_config

    # 优化参数
    if n_iter := os.getenv('BO_N_ITERATIONS'):
        try:
            env_config['bo'] = {'n_iterations': int(n_iter)}
        except ValueError:
            pass

    if n_warm := os.getenv('BO_N_WARMSTART'):
        try:
            if 'bo' not in env_config:
                env_config['bo'] = {}
            env_config['bo']['n_warmstart'] = int(n_warm)
        except ValueError:
            pass

    return env_config


# ═══════════════════════════════════════════════════════════════════════════
# §C  工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典

    Args:
        base: 基础字典
        override: 覆盖字典

    Returns:
        合并后的字典
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    保存配置到 JSON 文件

    Args:
        config: 配置对象
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config.to_json(str(path))


def validate_config(config: Config) -> tuple[bool, list[str]]:
    """
    验证配置一致性

    Args:
        config: 配置对象

    Returns:
        (is_valid, errors): 是否有效及错误消息列表
    """
    errors = config.validate()
    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
# §D  配置模板生成
# ═══════════════════════════════════════════════════════════════════════════

def generate_config_template(
    output_path: Optional[Union[str, Path]] = None,
    mode: str = "full",
) -> str:
    """
    生成配置模板

    Args:
        output_path: 输出路径（可选），若指定则保存到文件
        mode: 模板模式
            - "full": 完整配置
            - "minimal": 最小配置（仅必要项）

    Returns:
        JSON 格式的模板字符串
    """
    if mode == "minimal":
        config = Config(
            bo=BOConfig(n_iterations=10, n_warmstart=3),
            acquisition=AcquisitionConfig(n_cand=5, n_select=2),
        )
    else:
        config = Config()

    template = json.dumps(config.to_dict(), indent=2, ensure_ascii=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)

    return template


# ═══════════════════════════════════════════════════════════════════════════
# §E  命令行解析器
# ═══════════════════════════════════════════════════════════════════════════

def parse_cli_overrides(args: list[str]) -> Dict[str, Any]:
    """
    解析命令行覆盖参数

    支持格式:
        --bo.n_iterations=100
        --llm.model=gpt-4o
        --data.save_dir=./results

    Args:
        args: 命令行参数列表

    Returns:
        嵌套字典形式的覆盖配置
    """
    overrides: Dict[str, Any] = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith('--'):
            i += 1
            continue

        # 解析 --key=value 或 --key value 格式
        if '=' in arg:
            key, value = arg[2:].split('=', 1)
        else:
            key = arg[2:]
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                value = args[i + 1]
                i += 1
            else:
                value = 'true'

        # 类型转换
        value = _parse_value(value)

        # 构建嵌套字典
        keys = key.split('.')
        current = overrides
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

        i += 1

    return overrides


def _parse_value(value: str) -> Any:
    """
    将字符串值转换为适当的 Python 类型

    Args:
        value: 字符串值

    Returns:
        转换后的值
    """
    # 布尔值
    if value.lower() in ('true', 'yes', 'on'):
        return True
    if value.lower() in ('false', 'no', 'off'):
        return False

    # 整数
    try:
        return int(value)
    except ValueError:
        pass

    # 浮点数
    try:
        return float(value)
    except ValueError:
        pass

    # JSON 数组/对象
    if value.startswith('[') or value.startswith('{'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # 默认返回字符串
    return value


# ═══════════════════════════════════════════════════════════════════════════
# §F  CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LLM-MOBO 配置加载器测试")
    print("=" * 60)

    # 测试 1: 默认配置
    print("\n[测试 1] 默认配置")
    cfg = load_config()
    print(f"  n_iterations: {cfg.bo.n_iterations}")
    print(f"  n_warmstart: {cfg.bo.n_warmstart}")
    print(f"  n_candidates: {cfg.acquisition.n_cand}")
    print(f"  LLM model: {cfg.llm.model}")

    # 测试 2: 字典覆盖
    print("\n[测试 2] 字典覆盖")
    cfg = load_config(overrides={
        'bo': {'n_iterations': 100, 'n_warmstart': 10},
        'acquisition': {'n_cand': 20},
    })
    print(f"  n_iterations: {cfg.bo.n_iterations}")
    print(f"  n_warmstart: {cfg.bo.n_warmstart}")
    print(f"  n_candidates: {cfg.acquisition.n_cand}")

    # 测试 3: 配置验证
    print("\n[测试 3] 配置验证")
    is_valid, errors = validate_config(cfg)
    print(f"  验证结果：{'通过' if is_valid else '失败'}")
    if errors:
        for error in errors:
            print(f"    - {error}")

    # 测试 4: 生成模板
    print("\n[测试 4] 生成配置模板")
    template = generate_config_template(mode="minimal")
    print(f"  模板长度：{len(template)} 字符")
    print(f"  预览：{template[:200]}...")

    # 测试 5: CLI 解析
    print("\n[测试 5] CLI 参数解析")
    test_args = [
        '--bo.n_iterations=50',
        '--llm.model=claude-sonnet-4-20250514',
        '--data.save_dir=./my_results',
        '--acquisition.n_cand', '15',
    ]
    overrides = parse_cli_overrides(test_args)
    print(f"  解析结果：{overrides}")

    print("\n" + "=" * 60)
    print("配置加载器测试完成")
    print("=" * 60)
