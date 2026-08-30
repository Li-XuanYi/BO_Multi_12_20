"""
llm_warmstart.py  —  LLM 热启动模块
======================================
通过 LLM 零样本提示生成初始充电协议、GP 超参先验和搜索边界缩减建议。

设计参考：
  - LLAMBO (Liu et al., ICLR 2024) — 零样本热启动 + ICL 提示结构
  - LLMBO for battery (Kuai et al., 2025) — 电化学物理约束编码 + 复合核先验

返回契约（与 main.py warm_start() 完全对齐）：
    llm_data["protocols_normalized"]       : list[list[float]]  shape (n_llm, 5)
    llm_data["length_scales"]              : list[float]        shape (5,)
    llm_data["coupling_matrix"]            : list[list[float]]  shape (5, 5)
    llm_data["search_bounds_normalized"]   : {"lb": list, "ub": list}  各 shape (5,)

占位符约定：
    提示词模板中所有大写花括号占位符均由 PromptConfig 填充，
    例如 {SOH}、{N_PROTOCOLS}、{I1_LB}，在运行时替换为实际配置值。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  物理边界（与 main.py 完全对齐）
# ---------------------------------------------------------------------------
PHYS_LB = np.array([2.0, 2.0, 2.0, 0.10, 0.10])
PHYS_UB = np.array([6.0, 5.0, 3.0, 0.40, 0.30])
DIM_NAMES = ["I1_C", "I2_C", "I3_C", "dSOC1", "dSOC2"]


# ---------------------------------------------------------------------------
#  提示词配置（所有占位符在此集中定义）
# ---------------------------------------------------------------------------

@dataclass
class PromptConfig:
    """
    集中管理所有提示词占位符。
    修改物理设置时只需改这里，三条提示词自动同步。
    """
    # ── 问题描述 ────────────────────────────────────────────────
    soh: float = 0.7            # 电池健康状态（State of Health），0~1
    n_protocols: int = 10       # 需要生成的协议数量

    # ── 决策变量物理边界 ─────────────────────────────────────────
    i1_lb: float = 2.0          # 第一段电流下界 (C-rate)
    i1_ub: float = 6.0          # 第一段电流上界 (C-rate)
    i2_lb: float = 2.0          # 第二段电流下界 (C-rate)
    i2_ub: float = 5.0          # 第二段电流上界 (C-rate)
    i3_lb: float = 2.0          # 第三段电流下界 (C-rate)
    i3_ub: float = 3.0          # 第三段电流上界 (C-rate)
    dsoc1_lb: float = 0.10      # 第一段 SOC 跨度下界
    dsoc1_ub: float = 0.40      # 第一段 SOC 跨度上界
    dsoc2_lb: float = 0.10      # 第二段 SOC 跨度下界
    dsoc2_ub: float = 0.30      # 第二段 SOC 跨度上界
    dsoc_sum_max: float = 0.70  # dSOC1 + dSOC2 总和上限

    # ── 目标描述 ─────────────────────────────────────────────────
    obj_time_unit: str = "seconds"      # 充电时间单位
    obj_temp_unit: str = "Kelvin"       # 温升单位
    obj_aging_unit: str = "percent"     # 老化单位

    def as_dict(self) -> dict:
        return {
            "SOH":          f"{self.soh:.2f}",
            "N_PROTOCOLS":  str(self.n_protocols),
            "I1_LB":        f"{self.i1_lb:.1f}",
            "I1_UB":        f"{self.i1_ub:.1f}",
            "I2_LB":        f"{self.i2_lb:.1f}",
            "I2_UB":        f"{self.i2_ub:.1f}",
            "I3_LB":        f"{self.i3_lb:.1f}",
            "I3_UB":        f"{self.i3_ub:.1f}",
            "DSOC1_LB":     f"{self.dsoc1_lb:.2f}",
            "DSOC1_UB":     f"{self.dsoc1_ub:.2f}",
            "DSOC2_LB":     f"{self.dsoc2_lb:.2f}",
            "DSOC2_UB":     f"{self.dsoc2_ub:.2f}",
            "DSOC_SUM_MAX": f"{self.dsoc_sum_max:.2f}",
            "OBJ_TIME":     self.obj_time_unit,
            "OBJ_TEMP":     self.obj_temp_unit,
            "OBJ_AGING":    self.obj_aging_unit,
        }


# ---------------------------------------------------------------------------
#  提示词模板（参考 LLAMBO Fig.9-11 + 电池 LLMBO Fig.2-4）
#  所有大写占位符由 PromptConfig.as_dict() 填充
# ---------------------------------------------------------------------------

# ── 模板 1：热启动协议推荐（对应 LLAMBO 零样本热启动 + 电池 LLMBO Fig.2）──────
_WARMSTART_PROMPT_TEMPLATE = """\
You are an expert in lithium-ion battery fast charging and electrochemical engineering.

<PROBLEM DESCRIPTION>
Task: Recommend {N_PROTOCOLS} diverse and physically valid three-stage constant-current \
charging protocols for a lithium-ion battery.
Battery state of health (SOH): {SOH} (1.0 = new, 0.0 = fully degraded).
A lower SOH means the battery is more aged and requires gentler charging to avoid \
accelerated degradation.

<DECISION VARIABLES>
Each protocol is a vector of 5 values:
  1. I1 (C-rate, stage 1 current): range [{I1_LB}, {I1_UB}]
  2. I2 (C-rate, stage 2 current): range [{I2_LB}, {I2_UB}]
  3. I3 (C-rate, stage 3 current): range [{I3_LB}, {I3_UB}]
  4. dSOC1 (SOC fraction of stage 1, dimensionless): range [{DSOC1_LB}, {DSOC1_UB}]
  5. dSOC2 (SOC fraction of stage 2, dimensionless): range [{DSOC2_LB}, {DSOC2_UB}]

IMPORTANT: dSOC1 and dSOC2 are dimensionless fractions between 0 and 1, NOT percentages.
For example, dSOC1 = 0.25 means stage 1 covers 25% of the SOC range.

<PHYSICAL CONSTRAINTS — MUST BE SATISFIED>
  - Monotone current taper: I1 >= I2 >= I3 (each stage must not exceed the previous).
  - Total SOC span: dSOC1 + dSOC2 <= {DSOC_SUM_MAX} (remaining SOC goes to CV tail).
  - All values must remain within their stated ranges.

<OBJECTIVES (all minimise)>
  - Charging time ({OBJ_TIME}): faster is better.
  - Temperature rise ({OBJ_TEMP}): lower thermal stress is better.
  - Capacity fade ({OBJ_AGING}): lower aging is better.
  Note: for aged batteries (SOH < 0.8), temperature and aging objectives deserve \
higher priority.

<INSTRUCTIONS>
  - Generate {N_PROTOCOLS} protocols that explore the Pareto trade-off among the \
three objectives.
  - Do NOT generate values at the exact minimum or maximum of any range.
  - Do NOT generate duplicate protocols.
  - Incorporate prior electrochemical knowledge: high C-rates reduce charging time \
but increase heat and aging; for low SOH batteries, bias toward gentler protocols.
  - Each value must have at least 2 decimal places of precision.

<OUTPUT FORMAT>
Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown fences.
The object must have exactly one key "protocols", whose value is a list of {N_PROTOCOLS} \
lists, each inner list containing exactly 5 numbers in the order [I1, I2, I3, dSOC1, dSOC2].

Example structure (do not copy these values):
{{"protocols": [[4.50, 3.20, 2.10, 0.25, 0.18], [3.10, 2.80, 2.05, 0.30, 0.15]]}}
"""

# GP prior prompt is reserved for future use (Phase 2 with LLM-informed kernel).
# Currently not called — GP uses standard ARD Matérn 5/2 without LLM priors.


# ---------------------------------------------------------------------------
#  提示词渲染
# ---------------------------------------------------------------------------

def render(template: str, config: PromptConfig) -> str:
    """将模板中的占位符替换为 PromptConfig 中的值。"""
    text = template
    for key, value in config.as_dict().items():
        text = text.replace(f"{{{key}}}", value)
    return text


# ---------------------------------------------------------------------------
#  LLM 调用（Anthropic claude-sonnet-4-20250514）
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """
    调用 Anthropic Messages API，返回模型的纯文本回复。
    需要环境变量 ANTHROPIC_API_KEY。
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("请安装 anthropic: pip install anthropic --break-system-packages")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
#  JSON 解析工具（健壮版，容忍模型偶尔输出 markdown 围栏）
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """
    从 LLM 输出中提取 JSON。按优先级尝试：
      1. 直接 json.loads
      2. 剥离 ```json ... ``` 或 ``` ... ``` 围栏后 json.loads
      3. 提取第一个 {...} 块后 json.loads
    """
    raw = raw.strip()

    # 尝试 1：直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试 2：剥离 markdown 代码围栏
    stripped = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 尝试 3：提取第一个完整 {...} 块
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 输出中提取合法 JSON:\n{raw[:500]}")


# ---------------------------------------------------------------------------
#  归一化工具
# ---------------------------------------------------------------------------

def _call_llm_v2(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Preferred LLM caller with OpenAI-compatible backend support."""
    backend = os.environ.get("LLM_WARMSTART_BACKEND", "openai").strip().lower()

    if backend == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("NUWA_API_KEY")
        )
        if not api_key:
            raise RuntimeError("缺少 OPENAI_API_KEY（LLM_API_KEY / NUWA_API_KEY 也可）")

        base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_API_BASE")
            or os.environ.get("NUWA_API_BASE")
            or "https://api.nuwaapi.com/v1"
        )
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o"

        client = OpenAI(api_key=api_key, base_url=base_url)
        message = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in lithium-ion battery fast charging optimization. "
                        "Always respond with valid JSON only, no explanations or markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return message.choices[0].message.content

    if backend == "anthropic":
        return _call_llm(prompt, max_tokens=max_tokens, temperature=temperature)

    raise ValueError(f"Unsupported LLM_WARMSTART_BACKEND: {backend}")


def protocols_to_normalized(
    protocols_phys: list[list[float]],
    phys_lb: np.ndarray = PHYS_LB,
    phys_ub: np.ndarray = PHYS_UB,
) -> list[list[float]]:
    """
    物理空间协议 → 归一化空间 [0, 1]^5。
    供 main.py 导入使用。
    """
    lb = np.asarray(phys_lb)
    ub = np.asarray(phys_ub)
    result = []
    for p in protocols_phys:
        x = (np.asarray(p) - lb) / (ub - lb)
        result.append(np.clip(x, 0.0, 1.0).tolist())
    return result


def _phys_to_norm_bounds(
    lb_phys: np.ndarray,
    ub_phys: np.ndarray,
) -> tuple[list[float], list[float]]:
    """物理边界 → 归一化边界（相对于全局 PHYS_LB/UB）。"""
    lb_norm = ((lb_phys - PHYS_LB) / (PHYS_UB - PHYS_LB)).clip(0, 1)
    ub_norm = ((ub_phys - PHYS_LB) / (PHYS_UB - PHYS_LB)).clip(0, 1)
    return lb_norm.tolist(), ub_norm.tolist()


# ---------------------------------------------------------------------------
#  响应验证与修复
# ---------------------------------------------------------------------------

def _validate_protocols(
    protocols: list,
    config: PromptConfig,
) -> list[list[float]]:
    """
    检查协议列表格式，丢弃维度不对的条目，
    返回合法的 (n, 5) 物理值列表。
    """
    valid = []
    for p in protocols:
        if not isinstance(p, (list, tuple)) or len(p) != 5:
            logger.warning(f"  跳过非法协议（维度不对）: {p}")
            continue
        try:
            vals = [float(v) for v in p]
            bounds_ok = (
                config.i1_lb <= vals[0] <= config.i1_ub and
                config.i2_lb <= vals[1] <= config.i2_ub and
                config.i3_lb <= vals[2] <= config.i3_ub and
                config.dsoc1_lb <= vals[3] <= config.dsoc1_ub and
                config.dsoc2_lb <= vals[4] <= config.dsoc2_ub
            )
            monotone_ok = vals[0] >= vals[1] >= vals[2]
            dsoc_ok = (vals[3] + vals[4]) <= config.dsoc_sum_max
            if not bounds_ok or not monotone_ok or not dsoc_ok:
                logger.warning(f"  跳过非法协议（越界或约束不满足）: {vals}")
                continue
        except (ValueError, TypeError):
            logger.warning(f"  跳过非法协议（包含非数字）: {p}")
            continue
        valid.append(vals)
    return valid


# ---------------------------------------------------------------------------
#  主函数（main.py 的调用入口）
# ---------------------------------------------------------------------------

def main(
    n_llm: int,
    soh: float,
    output_path: str,
    model_temperature: float = 0.7,
) -> dict:
    """
    执行 LLM 热启动，仅返回 LLM 推荐的充电协议（归一化）。

    GP 不使用 LLM 先验（标准 ARD Matérn 5/2，超参由数据驱动）。
    EI/DE 使用全局搜索空间 [0,1]^5，不依赖 LLM 建议边界。

    Parameters
    ----------
    n_llm            : 需要 LLM 推荐的协议数量
    soh              : 电池健康状态 (0~1)
    output_path      : 原始 LLM 回复的保存路径（JSON）
    model_temperature: LLM 采样温度

    Returns
    -------
    dict with keys:
        protocols_normalized  : list[list[float]]  shape (n_valid, 5)
        _protocols_physical   : list[list[float]]  shape (n_valid, 5)  [调试用]
        _warmstart_raw        : str                                      [调试用]
    """
    config = PromptConfig(soh=soh, n_protocols=n_llm)

    # ── 调用 LLM 生成充电协议 ──────────────────────────────────────
    logger.info(f"[LLM] 调用 warmstart 提示词（SOH={soh:.2f}, n={n_llm}）...")
    prompt_ws = render(_WARMSTART_PROMPT_TEMPLATE, config)
    t0 = time.time()
    raw_ws = _call_llm_v2(prompt_ws, max_tokens=2048, temperature=model_temperature)
    logger.info(f"[LLM] warmstart 回复耗时 {time.time()-t0:.1f}s")

    try:
        ws_data = _parse_json(raw_ws)
        protocols_phys = _validate_protocols(
            ws_data.get("protocols", []), config
        )
    except Exception as e:
        logger.warning(f"[LLM] warmstart 解析失败: {e}，使用空协议列表")
        protocols_phys = []

    if len(protocols_phys) == 0:
        logger.warning("[LLM] 未获得合法协议，热启动退化为空（调用方将补充 LHS）")

    # ── 归一化协议到 [0,1]^5 ─────────────────────────────────────
    protocols_norm = protocols_to_normalized(protocols_phys)

    result = {
        "protocols_normalized": protocols_norm,
        "_protocols_physical":  protocols_phys,
        "_warmstart_raw":       raw_ws,
    }

    # ── 持久化（便于复盘）────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"[LLM] 结果已保存至 {out_path} | 合法协议: {len(protocols_norm)} 条")

    return result


# ---------------------------------------------------------------------------
#  命令行独立调用（调试用）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="LLM WarmStart 独立测试")
    parser.add_argument("--soh",    type=float, default=0.7)
    parser.add_argument("--n_llm", type=int,   default=5)
    parser.add_argument("--output", type=str,   default="./llm_init_debug.json")
    args = parser.parse_args()

    result = main(n_llm=args.n_llm, soh=args.soh, output_path=args.output)
    print("\n=== 返回摘要 ===")
    print(f"协议数量: {len(result['protocols_normalized'])}")
    print(f"length_scales: {result['length_scales']}")
    print(f"搜索边界 lb:   {result['search_bounds_normalized']['lb']}")
    print(f"搜索边界 ub:   {result['search_bounds_normalized']['ub']}")
