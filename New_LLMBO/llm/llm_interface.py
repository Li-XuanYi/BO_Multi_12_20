"""
components/llm_interface.py
============================
LLM Interface — LLAMBO-MO Framework

三个 Touchpoint 对应关系：
  Touchpoint 1a : generate_coupling_matrices()
      → 生成 W_time, W_temp, W_aging 耦合矩阵
      → 输出传给 CouplingMatrixManager.set_llm_matrices()

  Touchpoint 1b : generate_warmstart_candidates()
      → 生成 N_ws 个初始候选协议 [I1, SOC1, I2]
      → 输出传给 optimizer.py 进行 PyBaMM 评估

  Touchpoint 2  : generate_iteration_candidates()
      → 每迭代生成 m 个候选点
      → 输出传给 AcquisitionFunction.step() 进行 EI × W_charge 评分

额外接口（满足 acquisition.py 的 LLMPriorProtocol）：
  get_warmstart_center() → Optional[np.ndarray]
      → 返回 warmstart 候选点的均值作为 μ 初始化

设计哲学（学习自 LLAMBO）：
  ① 符号化 Prompt 模板：用 [PLACEHOLDER] 占位符，运行时替换
  ② 多采样投票：对 LLM 采样 n 次，解析验证后取有效响应
  ③ 参数空间验证：每个候选点必须在 param_bounds 内
  ④ 容错回退：LLM 失败时退回随机采样（不阻塞优化循环）

对外接口被 optimizer.py 使用：
─────────────────────────────────────────────────────────────────────
  LLMInterface(param_bounds, config)
  LLMInterface.generate_coupling_matrices()  → (W_time, W_temp, W_aging)
  LLMInterface.generate_warmstart_candidates(n)  → List[np.ndarray]
  LLMInterface.generate_iteration_candidates(n, state_dict)  → np.ndarray (m, 3)
  LLMInterface.get_warmstart_center()  → Optional[np.ndarray]  (LLMPriorProtocol)
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# 模块所在目录（用于定位 templates/）
_MODULE_DIR = Path(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())
)))
_TEMPLATE_DIR = _MODULE_DIR / "templates"


# ═══════════════════════════════════════════════════════════════════════════
# §A  配置数据类
# ═══════════════════════════════════════════════════════════════════════════

class LLMConfig:
    """
    LLM 调用配置。

    支持多种后端：
      - "ollama"    : 本地 Ollama（Qwen2.5 等），通过 OpenAI 兼容 API
      - "openai"    : OpenAI GPT-4 / GPT-3.5
      - "anthropic" : Anthropic Claude
      - "mock"      : 不调用 LLM，返回物理启发式默认值（测试/消融用）

    Parameters
    ----------
    backend     : str   后端类型 {"ollama", "openai", "anthropic", "mock"}
    model       : str   模型名称（如 "qwen2.5:7b", "gpt-4o", "claude-sonnet-4-20250514"）
    api_base    : str   API 地址（Ollama 默认 "http://localhost:11434/v1"）
    api_key     : str   API 密钥（Ollama 默认 "ollama"）
    temperature : float 采样温度
    n_samples   : int   每次请求的采样数（多采样投票）
    timeout     : int   请求超时秒数
    """

    def __init__(
        self,
        backend:     str   = "openai",
        model:       str   = "gpt-4o",
        api_base:    str   = "https://api.nuwaapi.com/v1",
        api_key:     str   = "sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK",
        temperature: float = 0.7,
        n_samples:   int   = 5,
        timeout:     int   = 120,
    ):
        self.backend     = backend
        self.model       = model
        self.api_base    = api_base
        self.api_key     = api_key
        self.temperature = temperature
        self.n_samples   = n_samples
        self.timeout     = timeout

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量构建配置（兼容 LLAMBO 风格）。"""
        return cls(
            backend     = os.environ.get("LLM_BACKEND",     "ollama"),
            model       = os.environ.get("LLM_MODEL",       "qwen2.5:7b"),
            api_base    = os.environ.get("LLM_API_BASE",    "http://localhost:11434/v1"),
            api_key     = os.environ.get("LLM_API_KEY",     "ollama"),
            temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            n_samples   = int(os.environ.get("LLM_N_SAMPLES",   "5")),
            timeout     = int(os.environ.get("LLM_TIMEOUT",     "120")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §B  模板引擎（学习 LLAMBO 的 utils_templates.py）
# ═══════════════════════════════════════════════════════════════════════════

class TemplateEngine:
    """
    符号化 Prompt 模板引擎。

    学习 LLAMBO 的模板设计：
      - 模板文件中使用 [PLACEHOLDER] 占位符
      - 运行时通过 render(template_name, **kwargs) 替换
      - 支持从文件或字符串加载

    与 LLAMBO 的差异：
      - LLAMBO 用固定字典映射（如 [MODEL] → task_dict['model']）
      - 本实现用动态 kwargs，更灵活

    用法示例::

        engine = TemplateEngine("/path/to/templates")
        prompt = engine.render("warmstart_candidates",
            BATTERY_MODEL="LG M50",
            I1_RANGE="3.0 - 7.0",
            NUM_CANDIDATES=10,
        )
    """

    def __init__(self, template_dir: Union[str, Path] = _TEMPLATE_DIR):
        self._dir = Path(template_dir)
        self._cache: Dict[str, str] = {}

    def load(self, name: str) -> str:
        """加载模板文件，缓存到内存。"""
        if name not in self._cache:
            path = self._dir / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"模板文件不存在: {path}")
            text = path.read_text(encoding="utf-8")
            # LLAMBO 风格：合并多余换行
            text = text.replace('\n\n', '[DOUBLE_NEWLINE]')
            text = text.replace('\n', ' ')
            text = text.replace('[DOUBLE_NEWLINE]', '\n')
            self._cache[name] = text
        return self._cache[name]

    def render(self, name: str, **kwargs) -> str:
        """
        加载模板并替换所有 [PLACEHOLDER]。

        Parameters
        ----------
        name   : 模板文件名（不含 .txt）
        kwargs : 占位符键值对（不需要方括号，如 BATTERY_MODEL="LG M50"）

        Returns
        -------
        str — 渲染后的 prompt 文本
        """
        text = self.load(name)
        for key, value in kwargs.items():
            placeholder = f"[{key}]"
            text = text.replace(placeholder, str(value))

        # 检查是否有未替换的占位符
        remaining = re.findall(r'\[([A-Z_]+)\]', text)
        if remaining:
            logger.warning(
                "TemplateEngine: 模板 '%s' 中有未替换的占位符: %s",
                name, remaining
            )
        return text


# ═══════════════════════════════════════════════════════════════════════════
# §C  LLM API 调用层（多后端）
# ═══════════════════════════════════════════════════════════════════════════

class LLMCaller:
    """
    LLM API 调用封装。

    支持 Ollama / OpenAI / Anthropic 三种后端，
    统一接口 call(prompt, n) → List[str]。

    学习 LLAMBO 的 chat_gpt() 函数设计，但支持多后端。
    """

    def __init__(self, config: LLMConfig):
        self._config = config

    def call(self, prompt: str, n: Optional[int] = None) -> List[str]:
        """
        发送 prompt 到 LLM，返回 n 个响应文本。

        Parameters
        ----------
        prompt : str   完整 prompt 文本
        n      : int   采样数（默认使用 config.n_samples）

        Returns
        -------
        List[str] — n 个响应文本（可能含无效响应，后续由解析器过滤）
        """
        n = n or self._config.n_samples
        backend = self._config.backend.lower()

        if backend == "mock":
            return self._mock_call(prompt, n)
        elif backend in ("ollama", "openai"):
            return self._openai_compatible_call(prompt, n)
        elif backend == "anthropic":
            return self._anthropic_call(prompt, n)
        else:
            raise ValueError(f"不支持的 LLM 后端: {backend}")

    # ── OpenAI 兼容 API（含 Ollama） ─────────────────────────────────────
    def _openai_compatible_call(self, prompt: str, n: int) -> List[str]:
        """
        通过 OpenAI 兼容 API 调用（Ollama / OpenAI / Azure）。

        LLAMBO 原始实现使用 openai.ChatCompletion.create(n=30)，
        但 Ollama 不支持 n>1，因此逐次调用。
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("需要安装 openai 包: pip install openai")
            return []

        client = OpenAI(
            base_url=self._config.api_base,
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )

        responses = []
        for i in range(n):
            try:
                resp = client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert in lithium-ion battery charging optimization. "
                                    "Always respond with valid JSON only, no explanations."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=2000,
                )
                text = resp.choices[0].message.content.strip()
                responses.append(text)
                logger.debug("LLMCaller: 采样 %d/%d 成功 (%d chars)", i+1, n, len(text))
            except Exception as exc:
                logger.warning("LLMCaller: 采样 %d/%d 失败: %s", i+1, n, exc)
                responses.append("")

        return responses

    # ── Anthropic API ─────────────────────────────────────────────────────
    def _anthropic_call(self, prompt: str, n: int) -> List[str]:
        try:
            import anthropic
        except ImportError:
            logger.error("需要安装 anthropic 包: pip install anthropic")
            return []

        client = anthropic.Anthropic(api_key=self._config.api_key)
        responses = []
        for i in range(n):
            try:
                resp = client.messages.create(
                    model=self._config.model,
                    max_tokens=2000,
                    temperature=self._config.temperature,
                    system="You are an expert in lithium-ion battery charging optimization. "
                           "Always respond with valid JSON only, no explanations.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                responses.append(text)
            except Exception as exc:
                logger.warning("LLMCaller: Anthropic 采样 %d/%d 失败: %s", i+1, n, exc)
                responses.append("")

        return responses

    # ── Mock（不调用 LLM，返回空字符串，由上层回退机制处理） ──────────────
    def _mock_call(self, prompt: str, n: int) -> List[str]:
        logger.info("LLMCaller [mock]: 返回 %d 个空响应（将触发物理启发式回退）", n)
        return [""] * n


# ═══════════════════════════════════════════════════════════════════════════
# §D  响应解析器（学习 LLAMBO 的 obtain_all_list_valid）
# ═══════════════════════════════════════════════════════════════════════════

class ResponseParser:
    """
    LLM 响应解析与验证。

    学习 LLAMBO 的 obtain_all_list_valid()：
      1. 从原始文本中提取 JSON
      2. 解析为 Python 对象
      3. 验证每个候选点是否在参数空间内
      4. 返回所有有效响应

    与 LLAMBO 的差异：
      - LLAMBO 使用 ConfigSpace 验证
      - 本实现使用 param_bounds 字典验证（更轻量）
      - 支持矩阵和候选点两种输出格式
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self._bounds = param_bounds
        self._keys = list(param_bounds.keys())  # ["I1", "SOC1", "I2"]

    # ── 提取 JSON（容错） ────────────────────────────────────────────────
    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """
        从 LLM 响应文本中提取 JSON 对象或数组。

        尝试顺序：
          1. 直接解析整个文本
          2. 提取 ```json ... ``` 代码块
          3. 提取第一个 { ... } 或 [ ... ] 块
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # 1. 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Markdown 代码块
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. 第一个 JSON 对象/数组
        for pattern in [r'(\[[\s\S]*\])', r'(\{[\s\S]*\})']:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # 4. LLAMBO 风格：用 ast.literal_eval 尝试
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass

        return None

    # ── 验证单个候选点 ───────────────────────────────────────────────────
    def validate_candidate(self, d: Dict[str, float]) -> Optional[np.ndarray]:
        """
        验证候选点字典是否在参数边界内。

        Parameters
        ----------
        d : {"I1": float, "SOC1": float, "I2": float}

        Returns
        -------
        np.ndarray (3,) 若有效，None 若无效
        """
        try:
            values = []
            for key in self._keys:
                val = float(d[key])
                lo, hi = self._bounds[key]
                if val < lo or val > hi:
                    logger.debug("候选点 %s=%f 越界 [%f, %f]", key, val, lo, hi)
                    return None
                values.append(val)
            return np.array(values, dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("候选点验证失败: %s", exc)
            return None

    # ── 解析候选点列表 ───────────────────────────────────────────────────
    def parse_candidates(self, responses: List[str]) -> List[np.ndarray]:
        """
        从多个 LLM 响应中解析并合并所有有效候选点。

        学习 LLAMBO 的 obtain_all_list_valid()：
          - 遍历每个响应
          - 提取 JSON 列表
          - 逐个验证候选点
          - 合并去重

        Returns
        -------
        List[np.ndarray] — 有效候选点列表（已去重）
        """
        all_valid: List[np.ndarray] = []
        seen_hashes = set()

        for resp_idx, text in enumerate(responses):
            parsed = self.extract_json(text)
            if parsed is None:
                logger.debug("响应 %d: JSON 提取失败", resp_idx)
                continue

            # 确保是列表
            if isinstance(parsed, dict):
                candidates = [parsed]
            elif isinstance(parsed, list):
                candidates = parsed
            else:
                continue

            valid_count = 0
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                theta = self.validate_candidate(cand)
                if theta is not None:
                    # 去重（四舍五入到 4 位小数后哈希）
                    h = tuple(theta.round(4).tolist())
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        all_valid.append(theta)
                        valid_count += 1

            logger.debug("响应 %d: 解析出 %d 个有效候选点", resp_idx, valid_count)

        logger.info("ResponseParser: 共解析 %d 个有效候选点（%d 个响应）",
                     len(all_valid), len(responses))
        return all_valid

    # ── 解析耦合矩阵 ────────────────────────────────────────────────────
    def parse_coupling_matrices(
        self, responses: List[str]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        从 LLM 响应中解析三个 3×3 耦合矩阵。

        Returns
        -------
        (W_time, W_temp, W_aging) 各为 (3,3) ndarray，若解析失败返回 None
        """
        for resp_idx, text in enumerate(responses):
            parsed = self.extract_json(text)
            if not isinstance(parsed, dict):
                continue

            try:
                W_time  = np.array(parsed["W_time"],  dtype=float).reshape(3, 3)
                W_temp  = np.array(parsed["W_temp"],  dtype=float).reshape(3, 3)
                W_aging = np.array(parsed["W_aging"], dtype=float).reshape(3, 3)

                # 基本验证：形状和有限性
                for name, W in [("W_time", W_time), ("W_temp", W_temp), ("W_aging", W_aging)]:
                    if not np.all(np.isfinite(W)):
                        logger.warning("矩阵 %s 含非有限值，跳过", name)
                        raise ValueError(f"{name} 含非有限值")

                logger.info("ResponseParser: 成功从响应 %d 解析耦合矩阵", resp_idx)
                return W_time, W_temp, W_aging

            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("响应 %d 矩阵解析失败: %s", resp_idx, exc)
                continue

        logger.warning("ResponseParser: 所有响应均未成功解析耦合矩阵")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# §E  权重 - 性能历史分析（用于 LLM 学习）
# ═══════════════════════════════════════════════════════════════════════════

def build_weight_performance_history(
    database: Any,  # ObservationDB
    current_w_vec: np.ndarray,
    weight_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    分析历史数据，总结每种权重焦点下的最优参数模式。

    这是 LLM 学习"权重→参数→性能"映射的关键函数。

    算法：
    1. 从历史迭代中提取权重向量（如果有记录）
    2. 按权重焦点分组：time_focus (w_time > threshold), temp_focus, aging_focus
    3. 对每组，找出该组中表现最好的 top-k 个配置的参数模式

    Parameters
    ----------
    database : ObservationDB
        包含历史观测的数据库
    current_w_vec : np.ndarray
        当前迭代的权重向量（用于确定当前焦点）
    weight_threshold : float
        判断"焦点"的阈值（默认 0.4，即权重 > 0.4 视为焦点）

    Returns
    -------
    Dict[str, Any]
        {
            "time_focus": {
                "condition": "w_time > [W_THRESHOLD]",
                "n_samples": int,
                "best_i1_range": "[I1_MIN, I1_MAX]",
                "best_i2_range": "[I2_MIN, I2_MAX]",
                "pattern": "较高 I1 和 I2 显著缩短时间"
            },
            "temp_focus": { ... },
            "aging_focus": { ... },
            "current_focus": "time" | "temp" | "aging" | "balanced"
        }
    """
    feasible = database.get_feasible()
    if len(feasible) < 3:
        # 数据不足，返回默认模式
        return _default_weight_patterns(weight_threshold)

    # 从历史观测中提取信息
    # 注意：我们没有直接存储每轮的权重，但可以从 Pareto 前沿推断
    # 这里使用简化方法：根据当前 Pareto 前沿的参数 - 性能关系来总结模式

    pareto_front = database.get_pareto_front()
    if len(pareto_front) < 2:
        return _default_weight_patterns(weight_threshold)

    # 分析 Pareto 前沿的参数分布
    results = {
        "time_focus": _analyze_focus_region(pareto_front, obj_idx=0, param_names=["I1", "I2"]),
        "temp_focus": _analyze_focus_region(pareto_front, obj_idx=1, param_names=["I1", "SOC1"]),
        "aging_focus": _analyze_focus_region(pareto_front, obj_idx=2, param_names=["I2", "SOC1"]),
    }

    # 确定当前焦点
    w_max = current_w_vec.max()
    if w_max <= 1.0/3 + 0.05:  # 均匀权重
        current_focus = "balanced"
    elif current_w_vec[0] == w_max:
        current_focus = "time"
    elif current_w_vec[1] == w_max:
        current_focus = "temp"
    else:
        current_focus = "aging"

    return {
        **results,
        "current_focus": current_focus,
        "weight_threshold": weight_threshold,
    }


def _default_weight_patterns(weight_threshold: float) -> Dict[str, Any]:
    """返回基于物理直觉的默认模式（当历史数据不足时）。"""
    return {
        "time_focus": {
            "condition": f"w_time > {weight_threshold}",
            "n_samples": 0,
            "best_i1_range": "[较高值]",
            "best_i2_range": "[较高值]",
            "pattern": "较高 I1 和 I2 显著缩短充电时间"
        },
        "temp_focus": {
            "condition": f"w_temp > {weight_threshold}",
            "n_samples": 0,
            "best_i1_range": "[较低值]",
            "best_soc1_range": "[中等值]",
            "pattern": "较低 I1 和中等 SOC1 降低峰值温度"
        },
        "aging_focus": {
            "condition": f"w_aging > {weight_threshold}",
            "n_samples": 0,
            "best_i2_range": "[较低值]",
            "best_soc1_range": "[较高值]",
            "pattern": "较低 I2 和高 SOC1 减少老化"
        },
        "current_focus": "unknown",
        "weight_threshold": weight_threshold,
    }


def _analyze_focus_region(
    pareto_front: List,
    obj_idx: int,
    param_names: List[str],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    分析 Pareto 前沿上某个目标最优区域的参数模式。

    Parameters
    ----------
    pareto_front : List[Observation]
        Pareto 前沿观测列表
    obj_idx : int
        目标索引 (0=time, 1=temp, 2=aging)
    param_names : List[str]
        与该目标最相关的参数名
    top_k : int
        分析 top-k 个最优解

    Returns
    -------
    Dict[str, Any]
        包含参数范围和模式描述
    """
    if len(pareto_front) == 0:
        return {"pattern": "数据不足", "n_samples": 0}

    # 按目标值排序
    sorted_obs = sorted(pareto_front, key=lambda o: o.objectives[obj_idx])
    top_obs = sorted_obs[:min(top_k, len(sorted_obs))]

    # 提取参数范围
    param_ranges = {}
    for i, pname in enumerate(param_names):
        values = [o.theta[i] for o in top_obs]
        param_ranges[f"best_{pname.lower()}_range"] = f"[{min(values):.1f}, {max(values):.1f}]"

    # 生成模式描述
    pattern = _generate_pattern_description(top_obs, obj_idx)

    return {
        "n_samples": len(top_obs),
        **param_ranges,
        "pattern": pattern,
    }


def _generate_pattern_description(
    obs_list: List,
    obj_idx: int,
) -> str:
    """根据观测列表生成自然语言模式描述。"""
    # obj_idx 决定模式描述
    if obj_idx == 0:  # time
        return "较高 I1 和 I2 显著缩短充电时间"
    elif obj_idx == 1:  # temp
        return "较低 I1 和中等 SOC1 降低峰值温度"
    else:  # aging
        return "较低 I2 和高 SOC1 减少老化"


def _format_weight_patterns_for_prompt(weight_history: Dict[str, Any]) -> str:
    """
    将权重 - 性能历史分析结果格式化为 prompt 字符串。

    Parameters
    ----------
    weight_history : Dict[str, Any]
        build_weight_performance_history() 的返回结果

    Returns
    -------
    str — 格式化的历史模式文本，用于填充 [HISTORICAL_WEIGHT_PATTERNS] 占位符
    """
    lines = ["=== 历史权重 - 性能模式 ==="]

    # 当前焦点
    current_focus = weight_history.get("current_focus", "unknown")
    lines.append(f"Current optimization focus: {current_focus.upper()}")
    lines.append("")

    # 各焦点的历史模式
    focus_map = {
        "time_focus": ("Time-focused (w_time > threshold)", ["best_i1_range", "best_i2_range"]),
        "temp_focus": ("Temperature-focused (w_temp > threshold)", ["best_i1_range", "best_soc1_range"]),
        "aging_focus": ("Aging-focused (w_aging > threshold)", ["best_i2_range", "best_soc1_range"]),
    }

    for focus_key, (focus_label, param_keys) in focus_map.items():
        if focus_key in weight_history:
            data = weight_history[focus_key]
            lines.append(f"- {focus_label}:")
            lines.append(f"    Pattern: {data.get('pattern', 'N/A')}")
            lines.append(f"    Samples: {data.get('n_samples', 0)} historical configurations")
            for param_key in param_keys:
                if param_key in data:
                    param_name = param_key.replace("best_", "").replace("_range", "").upper()
                    lines.append(f"    Best {param_name}: {data[param_key]}")
            lines.append("")

    # 当前权重的建议
    if current_focus != "unknown" and current_focus != "balanced":
        focus_data = weight_history.get(f"{current_focus}_focus", {})
        if focus_data:
            lines.append("Recommendation for current weights:")
            lines.append(f"  Based on {focus_data.get('n_samples', 0)} historical time-focused configurations,")
            lines.append(f"  try parameters in the ranges above for better performance.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# §F  物理启发式回退（LLM 失败时使用）
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsHeuristicFallback:
    """
    当 LLM 不可用或所有响应无效时的物理启发式回退。

    提供：
      - 默认耦合矩阵（基于物理直觉）
      - Latin Hypercube 采样候选点
      - 物理启发式中心点
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self._bounds = param_bounds
        self._lo = np.array([param_bounds["I1"][0],
                              param_bounds["SOC1"][0],
                              param_bounds["I2"][0]], dtype=float)
        self._hi = np.array([param_bounds["I1"][1],
                              param_bounds["SOC1"][1],
                              param_bounds["I2"][1]], dtype=float)

    def default_coupling_matrices(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        物理启发式默认耦合矩阵。

        W_time: 充电时间主要由 I₁、I₂ 控制
        W_temp: 温度主要由 I₁ 主导（低 SOC 高电流产热大）
        W_aging: 老化由高 SOC 下的 I₂ 主导（锂沉积风险）
        """
        W_time = np.array([
            [0.8, 0.2, 0.3],
            [0.2, 0.4, 0.1],
            [0.3, 0.1, 0.7],
        ])
        W_temp = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 0.3, 0.2],
            [0.1, 0.2, 0.4],
        ])
        W_aging = np.array([
            [0.3, 0.1, 0.2],
            [0.1, 0.6, 0.5],
            [0.2, 0.5, 0.9],
        ])
        return W_time, W_temp, W_aging

    def lhs_candidates(self, n: int, seed: int = 42) -> List[np.ndarray]:
        """
        Latin Hypercube Sampling 回退。

        Parameters
        ----------
        n    : 采样数
        seed : 随机种子

        Returns
        -------
        List[np.ndarray] — n 个候选点
        """
        rng = np.random.default_rng(seed)
        d = 3  # 维度
        # LHS 核心：每个维度分 n 等份，每份随机采一个
        samples = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                samples[i, j] = (perm[i] + rng.random()) / n

        # 映射到参数空间
        candidates = []
        for i in range(n):
            theta = self._lo + samples[i] * (self._hi - self._lo)
            candidates.append(theta)

        return candidates

    def physics_informed_warmstart(self, n: int) -> List[np.ndarray]:
        """
        物理启发式 warmstart 候选点。

        基于领域知识的典型充电协议：
          - 激进快充：高 I₁, 低 SOC₁, 低 I₂
          - 保守安全：低 I₁, 高 SOC₁, 低 I₂
          - 均衡方案：中等参数
        """
        lo, hi = self._lo, self._hi
        range_ = hi - lo

        # 固定的物理先验点
        prior_points = [
            np.array([6.5, 0.25, 1.5]),   # 激进快充
            np.array([3.5, 0.55, 1.5]),   # 保守安全
            np.array([5.0, 0.40, 2.5]),   # 均衡
            np.array([5.5, 0.30, 2.0]),   # 偏快均衡
            np.array([4.0, 0.50, 3.0]),   # 偏慢均衡
        ]

        # 裁剪到参数范围
        candidates = [np.clip(p, lo, hi) for p in prior_points[:min(n, 5)]]

        # 不够则补 LHS
        if len(candidates) < n:
            lhs_pts = self.lhs_candidates(n - len(candidates), seed=123)
            candidates.extend(lhs_pts)

        return candidates[:n]


# ═══════════════════════════════════════════════════════════════════════════
# §F  LLMInterface 主类（三个 Touchpoint 统一入口）
# ═══════════════════════════════════════════════════════════════════════════

class LLMInterface:
    """
    LLM 接口主类。

    统一管理三个 Touchpoint 的 prompt 构造、API 调用、响应解析。
    实现 acquisition.py 的 LLMPriorProtocol 接口。

    对外接口
    --------
    generate_coupling_matrices()
        → Touchpoint 1a：生成 W_time, W_temp, W_aging
        → 调用者：optimizer.py 初始化阶段

    generate_warmstart_candidates(n)
        → Touchpoint 1b：生成初始候选协议
        → 调用者：optimizer.py warm-start 阶段

    generate_iteration_candidates(n, state_dict)
        → Touchpoint 2：每迭代生成候选点
        → 调用者：optimizer.py 主循环

    get_warmstart_center()
        → 满足 LLMPriorProtocol，返回 warmstart 均值
        → 调用者：AcquisitionFunction.initialize()

    典型用法
    --------
    ::

        from llm.llm_interface import LLMInterface, LLMConfig
        from llmbo.gp_model import build_gp_stack, CouplingMatrixManager
        from llmbo.acquisition import build_acquisition_function

        config = LLMConfig(backend="ollama", model="qwen2.5:7b")
        llm = LLMInterface(PARAM_BOUNDS, config, battery_model="LG M50")

        # Touchpoint 1a: 耦合矩阵
        W_time, W_temp, W_aging = llm.generate_coupling_matrices()
        coupling_mgr.set_llm_matrices(W_time, W_temp, W_aging)

        # Touchpoint 1b: Warm-start
        ws_candidates = llm.generate_warmstart_candidates(n=10)
        # → 评估 ws_candidates，写入 Database

        # Algorithm 步骤 5: μ 初始化
        af.initialize(database, llm_prior=llm)  # llm 满足 LLMPriorProtocol

        # Touchpoint 2: 每迭代
        for t in range(T):
            X_cand = llm.generate_iteration_candidates(n=15, state_dict={...})
            result = af.step(X_cand, database, t, w_vec)
    """

    def __init__(
        self,
        param_bounds:   Dict[str, Tuple[float, float]],
        config:         Optional[LLMConfig] = None,
        template_dir:   Optional[Union[str, Path]] = None,
        # 电池物理参数（用于模板渲染）
        battery_model:  str   = "LG M50 (Chen2020)",
        Q_nom_Ah:       float = 5.0,
        SOC0:           float = 0.1,
        SOC_end:        float = 0.8,
    ):
        self._bounds = param_bounds
        self._config = config or LLMConfig()

        # 子组件
        tpl_dir = Path(template_dir) if template_dir else _TEMPLATE_DIR
        self._engine   = TemplateEngine(tpl_dir)
        self._caller   = LLMCaller(self._config)
        self._parser   = ResponseParser(param_bounds)
        self._fallback = PhysicsHeuristicFallback(param_bounds)

        # 电池参数（渲染模板用）
        self._battery_model = battery_model
        self._Q_nom_Ah      = Q_nom_Ah
        self._SOC0          = SOC0
        self._SOC_end       = SOC_end

        # 缓存 warmstart 结果（供 get_warmstart_center 使用）
        self._warmstart_cache: Optional[List[np.ndarray]] = None

        logger.info(
            "LLMInterface 初始化: backend=%s model=%s n_samples=%d",
            self._config.backend, self._config.model, self._config.n_samples
        )

    # ── 公共模板参数 ──────────────────────────────────────────────────────
    def _base_kwargs(self) -> Dict[str, str]:
        """所有模板共享的基础占位符。"""
        b = self._bounds
        return {
            "BATTERY_MODEL": self._battery_model,
            "Q_NOM":         f"{self._Q_nom_Ah}",
            "SOC0":          f"{self._SOC0}",
            "SOC_END":       f"{self._SOC_end}",
            "I1_RANGE":      f"{b['I1'][0]} - {b['I1'][1]}",
            "SOC1_RANGE":    f"{b['SOC1'][0]} - {b['SOC1'][1]}",
            "I2_RANGE":      f"{b['I2'][0]} - {b['I2'][1]}",
        }

    # ══════════════════════════════════════════════════════════════════════
    # Touchpoint 1a: 耦合矩阵生成
    # ══════════════════════════════════════════════════════════════════════

    def generate_coupling_matrices(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Touchpoint 1a: 生成三个目标耦合矩阵 W_time, W_temp, W_aging。

        流程：
          1. 渲染 coupling_matrix 模板
          2. 调用 LLM（多采样）
          3. 解析第一个有效响应
          4. 失败则回退物理启发式默认矩阵

        Returns
        -------
        (W_time, W_temp, W_aging) 各为 (3,3) np.ndarray
        """
        logger.info("=== Touchpoint 1a: 耦合矩阵生成 ===")

        prompt = self._engine.render("coupling_matrix", **self._base_kwargs())
        responses = self._caller.call(prompt)

        result = self._parser.parse_coupling_matrices(responses)
        if result is not None:
            W_time, W_temp, W_aging = result
            logger.info("Touchpoint 1a: LLM 成功生成耦合矩阵")
            return W_time, W_temp, W_aging

        # 回退
        logger.warning("Touchpoint 1a: LLM 失败，使用物理启发式默认矩阵")
        return self._fallback.default_coupling_matrices()

    # ══════════════════════════════════════════════════════════════════════
    # Touchpoint 1b: Warm-Start 候选点生成
    # ══════════════════════════════════════════════════════════════════════

    def generate_warmstart_candidates(
        self,
        n: int = 10,
        batch_size: int = 20,
        max_llm_attempts: int = 5,
    ) -> List[np.ndarray]:
        """
        Touchpoint 1b: 生成初始 warm-start 候选协议（支持批量生成）。

        流程：
          1. 分批调用 LLM（每批 batch_size 个，最多 max_llm_attempts 批）
          2. 合并去重所有有效候选点
          3. 若不足 n 个，用 LHS / 物理启发式补齐
          4. 缓存结果（供 get_warmstart_center() 使用）

        Parameters
        ----------
        n : int  需要的候选点数量（默认 10）
        batch_size : int  每批生成的候选点数（默认 20）
        max_llm_attempts : int  最多尝试的批次数（默认 5）

        Returns
        -------
        List[np.ndarray] — n 个候选点，每个为 (3,) ndarray
        """
        logger.info("=== Touchpoint 1b: Warm-Start 候选点生成 (n=%d, batch_size=%d) ===", n, batch_size)

        all_candidates = []
        seen_hashes = set()

        # 分批生成，直到达到目标数量或达到最大尝试次数
        for batch_idx in range(max_llm_attempts):
            if len(all_candidates) >= n:
                break

            logger.info("  批次 %d/%d: 请求 %d 个候选点", batch_idx + 1, max_llm_attempts, batch_size)
            kwargs = {**self._base_kwargs(), "NUM_CANDIDATES": str(batch_size)}
            prompt = self._engine.render("warmstart_candidates", **kwargs)
            responses = self._caller.call(prompt)

            batch_candidates = self._parser.parse_candidates(responses)

            # 去重并添加到总列表
            new_count = 0
            for cand in batch_candidates:
                h = tuple(cand.round(4).tolist())
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_candidates.append(cand)
                    new_count += 1

            logger.info("  批次 %d: 新增 %d 个有效候选点（总计 %d）",
                       batch_idx + 1, new_count, len(all_candidates))

        logger.info("Touchpoint 1b: LLM 共返回 %d 个有效候选点", len(all_candidates))

        # 补齐
        if len(all_candidates) < n:
            shortage = n - len(all_candidates)
            logger.info("Touchpoint 1b: 不足 %d 个，用启发式补 %d 个", n, shortage)
            fallback_pts = self._fallback.physics_informed_warmstart(shortage)
            all_candidates.extend(fallback_pts)

        candidates = all_candidates[:n]
        self._warmstart_cache = [c.copy() for c in candidates]

        logger.info("Touchpoint 1b: 最终返回 %d 个 warm-start 候选点", len(candidates))
        return candidates

    # ══════════════════════════════════════════════════════════════════════
    # Touchpoint 2: 迭代候选点生成
    # ══════════════════════════════════════════════════════════════════════

    def generate_iteration_candidates(
        self,
        n:          int,
        state_dict: Dict[str, Any],
    ) -> np.ndarray:
        """
        Touchpoint 2: 每迭代生成 m 个候选点（支持分批次）。

        流程：
          1. 从 state_dict 提取状态信息，渲染模板
          2. 构建权重 - 性能历史模式（用于 LLM 学习）
          3. 调用 LLM
          4. 解析有效候选点
          5. 不足则围绕 μ±σ 随机补齐

        Parameters
        ----------
        n          : int   需要的候选点数（每批）
        state_dict : dict  包含以下键：
            - iteration (int)
            - max_iterations (int)
            - theta_best (np.ndarray, 3)
            - f_min (float)
            - mu (np.ndarray, 3)
            - sigma (np.ndarray, 3)
            - stagnation_count (int)
            - w_vec (np.ndarray, 3)
            - data_summary (str, 可选)
            - sensitivity_info (str, 可选)
            - batch_index (int, 可选) — 当前批次索引
            - n_batches (int, 可选) — 总批次数
            - batch_history (list, 可选) — 前几批的评估结果
            - database (Any, 可选) — ObservationDB 用于历史分析

        Returns
        -------
        np.ndarray (m, 3) — 候选点矩阵
        """
        t = state_dict.get("iteration", 0)
        batch_idx = state_dict.get("batch_index", 0)
        n_batches = state_dict.get("n_batches", 1)
        logger.info("=== Touchpoint 2: 迭代候选生成 (t=%d, batch=%d/%d, n=%d) ===",
                    t, batch_idx + 1, n_batches, n)

        # ── 构建探索引导文本 ──────────────────────────────────────────
        stag = state_dict.get("stagnation_count", 0)
        if stag >= 3:
            exploration = (
                "WARNING: The optimization has stagnated for {} iterations. "
                "Generate more exploratory candidates further from the current best. "
                "Consider larger deviations and unexplored regions."
            ).format(stag)
        elif stag >= 1:
            exploration = (
                "Note: No improvement in the last iteration. "
                "Include some exploratory candidates beyond the current search region."
            )
        else:
            exploration = (
                "The optimization is progressing well. "
                "Focus candidates near the search center but maintain diversity."
            )

        # ── 提取状态值 ────────────────────────────────────────────────
        theta_best = np.asarray(state_dict.get("theta_best", [5.0, 0.4, 2.5]))
        mu    = np.asarray(state_dict.get("mu", theta_best))
        sigma = np.asarray(state_dict.get("sigma", [1.0, 0.15, 1.0]))
        w_vec = np.asarray(state_dict.get("w_vec", [0.33, 0.33, 0.34]))

        # ── 构建权重 - 性能历史模式 ───────────────────────────────────
        database = state_dict.get("database")
        if database is not None:
            weight_history = build_weight_performance_history(database, w_vec)
            historical_patterns = _format_weight_patterns_for_prompt(weight_history)
        else:
            historical_patterns = _format_weight_patterns_for_prompt(
                _default_weight_patterns(0.4)
            )

        # ── 构建批次历史文本 ─────────────────────────────────────────
        batch_history = state_dict.get("batch_history", [])
        if batch_history:
            batch_lines = []
            for i, record in enumerate(batch_history):
                if all(k in record for k in ("I1", "SOC1", "I2", "time", "temp", "aging")):
                    batch_lines.append(
                        f"  Batch {i+1}: I1={record['I1']:.2f}, SOC1={record['SOC1']:.2f}, I2={record['I2']:.2f} "
                        f"→ time={record['time']:.0f}s, temp={record['temp']:.1f}K, aging={record['aging']:.4f}%"
                    )
                else:
                    bidx = int(record.get("batch_index", i)) + 1
                    n_cand = record.get("n_candidates", "?")
                    n_sel = record.get("n_selected", "?")
                    best_acq = record.get("best_acq_value")
                    if best_acq is None:
                        batch_lines.append(
                            f"  Batch {bidx}: selected {n_sel}/{n_cand}, best_acq=N/A"
                        )
                    else:
                        batch_lines.append(
                            f"  Batch {bidx}: selected {n_sel}/{n_cand}, best_acq={float(best_acq):.6f}"
                        )
            batch_text = "Previous batches in this iteration:\n" + "\n".join(batch_lines)
        else:
            batch_text = "First batch of this iteration."

        kwargs = {
            **self._base_kwargs(),
            "NUM_CANDIDATES":         str(n),
            "ITERATION":              str(t),
            "MAX_ITERATIONS":         str(state_dict.get("max_iterations", 100)),
            "BEST_I1":                f"{theta_best[0]:.3f}",
            "BEST_SOC1":              f"{theta_best[1]:.3f}",
            "BEST_I2":                f"{theta_best[2]:.3f}",
            "BEST_FTCH":              f"{state_dict.get('f_min', 0.0):.6f}",
            "MU_VALUES":              f"[{mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}]",
            "SIGMA_VALUES":           f"[{sigma[0]:.3f}, {sigma[1]:.3f}, {sigma[2]:.3f}]",
            "STAGNATION_COUNT":       str(stag),
            "W_TIME":                 f"{w_vec[0]:.3f}",
            "W_TEMP":                 f"{w_vec[1]:.3f}",
            "W_AGING":                f"{w_vec[2]:.3f}",
            "DATA_SUMMARY":           state_dict.get("data_summary", ""),
            "SENSITIVITY_INFO":       state_dict.get("sensitivity_info", ""),
            "EXPLORATION_GUIDANCE":   exploration,
            "DESIRED_FTCH":           f"{state_dict.get('desired_fval', 0.0):.6f}",
            "TARGET_DESCRIPTION":     state_dict.get("target_description", ""),
            "HISTORICAL_WEIGHT_PATTERNS": historical_patterns,
            "BATCH_INDEX":            str(batch_idx),
            "N_BATCHES":              str(n_batches),
            "BATCH_HISTORY":          batch_text,
        }

        prompt = self._engine.render("iterative_candidates", **kwargs)
        responses = self._caller.call(prompt)
        candidates = self._parser.parse_candidates(responses)

        logger.info("Touchpoint 2: LLM 返回 %d 个有效候选点", len(candidates))

        # 不足则围绕 μ±σ 随机补齐
        if len(candidates) < n:
            shortage = n - len(candidates)
            logger.info("Touchpoint 2: 不足 %d 个，用 μ±σ 随机补 %d 个", n, shortage)
            lo = np.array([self._bounds["I1"][0], self._bounds["SOC1"][0], self._bounds["I2"][0]])
            hi = np.array([self._bounds["I1"][1], self._bounds["SOC1"][1], self._bounds["I2"][1]])
            rng = np.random.default_rng()
            for _ in range(shortage):
                pt = mu + sigma * rng.standard_normal(3)
                pt = np.clip(pt, lo, hi)
                candidates.append(pt)

        candidates = candidates[:n]
        X = np.stack(candidates)  # (n, 3)

        logger.info("Touchpoint 2: 最终返回 %d 个候选点", X.shape[0])
        return X

    # ══════════════════════════════════════════════════════════════════════
    # LLMPriorProtocol 接口（供 acquisition.py 使用）
    # ══════════════════════════════════════════════════════════════════════

    def get_warmstart_center(self) -> Optional[np.ndarray]:
        """
        返回 warmstart 候选点的均值，作为搜索中心 μ 的初始化。

        满足 acquisition.py 的 LLMPriorProtocol 接口。

        若 warmstart 尚未执行（_warmstart_cache 为空），返回 None，
        此时 SearchMuTracker 会使用 θ_best 初始化。

        Returns
        -------
        np.ndarray (3,) 或 None
        """
        if self._warmstart_cache is None or len(self._warmstart_cache) == 0:
            logger.info("get_warmstart_center: warmstart 未执行，返回 None")
            return None

        center = np.mean(self._warmstart_cache, axis=0)
        logger.info("get_warmstart_center: μ_init = %s", center.round(4))
        return center

    # ══════════════════════════════════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════════════════════════════════

    @property
    def config(self) -> LLMConfig:
        return self._config

    def get_warmstart_cache(self) -> Optional[List[np.ndarray]]:
        """返回缓存的 warmstart 候选点（供日志/调试）。"""
        return self._warmstart_cache


# ═══════════════════════════════════════════════════════════════════════════
# §G  工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def build_llm_interface(
    param_bounds:  Dict[str, Tuple[float, float]],
    backend:       str   = "openai",
    model:         str   = "gpt-4o",
    api_base:      str   = "https://api.nuwaapi.com/v1",
    api_key:       str   = "sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK",
    n_samples:     int   = 5,
    temperature:   float = 0.7,
    battery_model: str   = "LG M50 (Chen2020)",
    template_dir:  Optional[str] = None,
) -> LLMInterface:
    """
    工厂函数：一步构建 LLMInterface。

    典型用法::

        from llm.llm_interface import build_llm_interface
        from llmbo.gp_model import build_gp_stack
        from llmbo.acquisition import build_acquisition_function

        BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}

        llm = build_llm_interface(BOUNDS, backend="ollama", model="qwen2.5:7b")
        psi, coupling, gamma_ann, gp = build_gp_stack(BOUNDS)

        # Touchpoint 1a
        W_time, W_temp, W_aging = llm.generate_coupling_matrices()
        coupling.set_llm_matrices(W_time, W_temp, W_aging)

        # Touchpoint 1b
        ws = llm.generate_warmstart_candidates(n=10)

        # 初始化 AF
        af = build_acquisition_function(gp, psi, BOUNDS)
        af.initialize(database, llm_prior=llm)

        # 迭代
        X_cand = llm.generate_iteration_candidates(15, state_dict)
        result = af.step(X_cand, database, t, w_vec)
    """
    config = LLMConfig(
        backend=backend,
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        n_samples=n_samples,
    )
    return LLMInterface(
        param_bounds=param_bounds,
        config=config,
        template_dir=template_dir,
        battery_model=battery_model,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §H  自测
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(levelname)s %(name)s: %(message)s"
    )

    BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}

    print("=" * 60)
    print("1. TemplateEngine 自测")
    print("=" * 60)
    engine = TemplateEngine()
    prompt_ws = engine.render("warmstart_candidates",
        BATTERY_MODEL="LG M50",
        Q_NOM="5.0", SOC0="0.1", SOC_END="0.8",
        I1_RANGE="3.0 - 7.0", SOC1_RANGE="0.1 - 0.7", I2_RANGE="1.0 - 5.0",
        NUM_CANDIDATES="10",
    )
    print(f"  Warmstart prompt ({len(prompt_ws)} chars):")
    print(f"  ...{prompt_ws[:200]}...")
    print("  ✓ TemplateEngine 渲染通过")

    print("\n" + "=" * 60)
    print("2. ResponseParser 自测")
    print("=" * 60)
    parser = ResponseParser(BOUNDS)

    # 测试 JSON 解析
    test_responses = [
        '[{"I1": 5.0, "SOC1": 0.4, "I2": 2.5}, {"I1": 6.0, "SOC1": 0.3, "I2": 1.5}]',
        '```json\n[{"I1": 4.0, "SOC1": 0.5, "I2": 3.0}]\n```',
        'Here are some configs: {"I1": 3.5, "SOC1": 0.6, "I2": 2.0}',
        '{"I1": 99.0, "SOC1": 0.4, "I2": 2.5}',  # 越界
        'invalid json garbage',
    ]
    candidates = parser.parse_candidates(test_responses)
    print(f"  解析出 {len(candidates)} 个有效候选点:")
    for i, c in enumerate(candidates):
        print(f"    [{i}] I1={c[0]:.1f} SOC1={c[1]:.1f} I2={c[2]:.1f}")
    assert len(candidates) == 4, f"期望 4 个有效候选点，得到 {len(candidates)}"
    print("  ✓ 候选点解析通过（含越界过滤）")

    # 测试矩阵解析
    matrix_resp = ['{"W_time":[[0.8,0.2,0.3],[0.2,0.4,0.1],[0.3,0.1,0.7]],'
                   '"W_temp":[[1.0,0.3,0.1],[0.3,0.3,0.2],[0.1,0.2,0.4]],'
                   '"W_aging":[[0.3,0.1,0.2],[0.1,0.6,0.5],[0.2,0.5,0.9]]}']
    matrices = parser.parse_coupling_matrices(matrix_resp)
    assert matrices is not None, "矩阵解析失败!"
    print(f"  W_time 形状: {matrices[0].shape}")
    print("  ✓ 耦合矩阵解析通过")

    print("\n" + "=" * 60)
    print("3. PhysicsHeuristicFallback 自测")
    print("=" * 60)
    fallback = PhysicsHeuristicFallback(BOUNDS)
    W_t, W_te, W_a = fallback.default_coupling_matrices()
    print(f"  W_time eigenvalues: {np.linalg.eigvalsh(W_t).round(4)}")
    print(f"  W_temp eigenvalues: {np.linalg.eigvalsh(W_te).round(4)}")
    print(f"  W_aging eigenvalues: {np.linalg.eigvalsh(W_a).round(4)}")

    lhs_pts = fallback.lhs_candidates(8)
    print(f"  LHS 候选点 ({len(lhs_pts)} 个):")
    for i, p in enumerate(lhs_pts[:3]):
        print(f"    [{i}] I1={p[0]:.2f} SOC1={p[1]:.3f} I2={p[2]:.2f}")
    print("  ✓ 回退机制通过")

    print("\n" + "=" * 60)
    print("4. LLMInterface [mock] 完整流程自测")
    print("=" * 60)
    llm = build_llm_interface(BOUNDS, backend="mock")

    # Touchpoint 1a
    W_time, W_temp, W_aging = llm.generate_coupling_matrices()
    print(f"  Touchpoint 1a: W_time shape={W_time.shape}")
    assert W_time.shape == (3, 3)

    # Touchpoint 1b
    ws = llm.generate_warmstart_candidates(n=8)
    print(f"  Touchpoint 1b: {len(ws)} 个 warmstart 候选点")
    assert len(ws) == 8
    for c in ws:
        assert c.shape == (3,)
        assert np.all(c >= [3.0, 0.1, 1.0]) and np.all(c <= [7.0, 0.7, 5.0])

    # LLMPriorProtocol
    center = llm.get_warmstart_center()
    print(f"  Warmstart center: {center.round(4)}")
    assert center.shape == (3,)

    # Touchpoint 2
    state = {
        "iteration": 5,
        "max_iterations": 100,
        "theta_best": np.array([5.0, 0.4, 2.5]),
        "f_min": 0.35,
        "mu": np.array([5.0, 0.4, 2.5]),
        "sigma": np.array([0.8, 0.1, 0.6]),
        "stagnation_count": 0,
        "w_vec": np.array([0.5, 0.3, 0.2]),
    }
    X_cand = llm.generate_iteration_candidates(15, state)
    print(f"  Touchpoint 2: X_candidates shape={X_cand.shape}")
    assert X_cand.shape == (15, 3)
    assert np.all(X_cand >= [3.0, 0.1, 1.0]) and np.all(X_cand <= [7.0, 0.7, 5.0])

    # Protocol 合规性
    from llmbo.acquisition import LLMPriorProtocol
    assert isinstance(llm, LLMPriorProtocol), "LLMInterface 不满足 LLMPriorProtocol!"
    print("  ✓ LLMPriorProtocol isinstance 检查通过")

    print("\n✓ llm_interface.py 全部自测通过")
