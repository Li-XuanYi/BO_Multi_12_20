"""
llm_interface.py — LLAMBO-MO LLM 接口
========================================
两个 Touchpoint：
  Touchpoint 1b : generate_warmstart_candidates(n)
      → 生成 N_ws 个初始充电协议（5D）

  Touchpoint 2  : generate_iteration_candidates(n, state_dict)
      → 每迭代生成候选点（5D），通过 w_vec 与 Tchebycheff-EI 耦合

决策变量（5维）：θ = (I1, I2, I3, dSOC1, dSOC2)
  - I1 ∈ [2.0, 6.0] A
  - I2 ∈ [2.0, 5.0] A
  - I3 ∈ [2.0, 3.0] A
  - dSOC1 ∈ [0.10, 0.40]   (第一段 SOC 区间宽度)
  - dSOC2 ∈ [0.10, 0.30]   (第二段 SOC 区间宽度)
  - dSOC3 = 0.8 - dSOC1 - dSOC2  (自动推导，不作为决策变量)

约束：dSOC1 + dSOC2 ≤ 0.70（由边界范围自然满足，验证层额外检查）

设计原则：
  - LLM 失败时静默回退 LHS/物理启发式，不阻塞优化主循环
  - WarmStart Prompt 使用外部模板 + 占位符渲染；Iteration Prompt 保持现有内联结构
  - w_vec 通过 Prompt 传入 LLM，实现 LLM 与采集函数的耦合
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from utils.constants import (
    DEFAULT_BOUNDS as CANONICAL_DEFAULT_BOUNDS,
    IDEAL_POINT as CANONICAL_IDEAL_POINT,
)

try:
    from llm.warmstart_prompt import (
        DEFAULT_DSOC_SUM_MAX,
        PLACEHOLDER_PATTERN,
        WarmStartPromptContextBuilder,
        render_warmstart_prompt,
    )
except ModuleNotFoundError:  # pragma: no cover - allows direct script execution
    from warmstart_prompt import (
        DEFAULT_DSOC_SUM_MAX,
        PLACEHOLDER_PATTERN,
        WarmStartPromptContextBuilder,
        render_warmstart_prompt,
    )

logger = logging.getLogger(__name__)

# 与 database.py 对齐的边界常量
DEFAULT_BOUNDS = {k: tuple(v) for k, v in CANONICAL_DEFAULT_BOUNDS.items()}
PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]
_DSOC_SUM_MAX = DEFAULT_DSOC_SUM_MAX


@dataclasses.dataclass
class IterationGuidance:
    mode: str
    confidence: float
    point: Optional[np.ndarray] = None
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    note: str = ""
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": float(self.confidence),
            "point": None if self.point is None else np.asarray(self.point, dtype=float).tolist(),
            "lb": None if self.lb is None else np.asarray(self.lb, dtype=float).tolist(),
            "ub": None if self.ub is None else np.asarray(self.ub, dtype=float).tolist(),
            "note": self.note,
            "raw_text": self.raw_text,
        }

    def representative_point(self) -> np.ndarray:
        if self.point is not None:
            return np.asarray(self.point, dtype=float).ravel()
        if self.lb is None or self.ub is None:
            raise ValueError("Guidance does not contain a point or region bounds")
        return (np.asarray(self.lb, dtype=float) + np.asarray(self.ub, dtype=float)) / 2.0

# dSOC1 + dSOC2 的最大值（由各自上界决定：0.40 + 0.30 = 0.70 < 0.80）
_DSOC_SUM_MAX = DEFAULT_DSOC_SUM_MAX


# ════════════════════════════════════════════════════════════════
# §A  LLM 配置
# ════════════════════════════════════════════════════════════════

class LLMConfig:
    """LLM 后端配置，支持 openai / anthropic / mock。"""

    def __init__(
        self,
        backend:     str   = "openai",
        model:       str   = "gpt-4.1-mini",
        api_base:    str   = "https://api.nuwaapi.com/v1",
        api_key:     str   = "",
        temperature: float = 0.7,
        n_samples:   int   = 3,
        timeout:     int   = 120,
    ):
        self.backend     = backend
        self.model       = model
        self.api_base    = api_base
        self.api_key     = api_key
        self.temperature = temperature
        self.n_samples   = n_samples
        self.timeout     = timeout


# ════════════════════════════════════════════════════════════════
# §B  LLM 调用器
# ════════════════════════════════════════════════════════════════

class LLMCaller:
    """统一的 LLM API 调用封装，返回 n 个响应文本列表。"""

    def __init__(self, config: LLMConfig):
        self._cfg = config

    def call(
        self,
        prompt: str,
        n: Optional[int] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        n = n or self._cfg.n_samples
        backend = self._cfg.backend.lower()

        if backend == "mock":
            return [""] * n
        elif backend in ("openai", "ollama"):
            return self._openai_call(
                prompt,
                n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif backend == "anthropic":
            return self._anthropic_call(
                prompt,
                n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            logger.warning("不支持的后端 %s，退回 mock", backend)
            return [""] * n

    def _openai_call(
        self,
        prompt: str,
        n: int,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("请安装 openai: pip install openai")
            return [""] * n

        client = OpenAI(
            base_url=self._cfg.api_base,
            api_key=self._cfg.api_key,
            timeout=self._cfg.timeout,
        )
        responses = []
        for i in range(n):
            try:
                resp = client.chat.completions.create(
                    model=self._cfg.model,
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
                    temperature=self._cfg.temperature if temperature is None else temperature,
                    max_tokens=2000 if max_tokens is None else int(max_tokens),
                )
                responses.append(resp.choices[0].message.content.strip())
            except Exception as e:
                logger.warning("LLM 调用 %d/%d 失败: %s", i + 1, n, e)
                responses.append("")
        return responses

    def _anthropic_call(
        self,
        prompt: str,
        n: int,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        try:
            import anthropic
        except ImportError:
            logger.error("请安装 anthropic: pip install anthropic")
            return [""] * n

        client = anthropic.Anthropic(api_key=self._cfg.api_key)
        responses = []
        for i in range(n):
            try:
                resp = client.messages.create(
                    model=self._cfg.model,
                    max_tokens=2000 if max_tokens is None else int(max_tokens),
                    temperature=self._cfg.temperature if temperature is None else temperature,
                    system=(
                        "You are an expert in lithium-ion battery fast charging optimization. "
                        "Always respond with valid JSON only, no explanations or markdown."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )
                responses.append(resp.content[0].text.strip())
            except Exception as e:
                logger.warning("Anthropic 调用 %d/%d 失败: %s", i + 1, n, e)
                responses.append("")
        return responses


# ════════════════════════════════════════════════════════════════
# §C  响应解析器
# ════════════════════════════════════════════════════════════════

class ResponseParser:
    """
    解析 LLM 响应，提取并验证 5D 候选点。
    验证规则：
      1. 每个参数在各自边界内
      2. dSOC1 + dSOC2 ≤ 0.70（防止 dSOC3 ≤ 0）
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        dsoc_sum_max: float = _DSOC_SUM_MAX,
    ):
        self._bounds = param_bounds
        self._dsoc_sum_max = float(dsoc_sum_max)

    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """从 LLM 响应文本中提取 JSON，容错处理。"""
        if not text or not text.strip():
            return None
        text = text.strip()

        # 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # markdown 代码块
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 提取第一个 JSON 数组或对象
        for pattern in [r'(\[[\s\S]*\])', r'(\{[\s\S]*\})']:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        return None

    def validate_candidate(self, d: Dict) -> Optional[np.ndarray]:
        """验证单个候选字典，返回 5D ndarray 或 None。"""
        try:
            values = []
            for key in PARAM_KEYS:
                val = float(d[key])
                lo, hi = self._bounds[key]
                if val < lo or val > hi:
                    logger.debug("候选点 %s=%.4f 越界 [%.2f, %.2f]", key, val, lo, hi)
                    return None
                values.append(val)

            # 额外检查 dSOC 约束
            dSOC_sum = values[3] + values[4]  # dSOC1 + dSOC2
            if dSOC_sum > self._dsoc_sum_max:
                logger.debug("dSOC1+dSOC2=%.3f > %.2f，候选无效", dSOC_sum, self._dsoc_sum_max)
                return None

            return np.array(values, dtype=float)

        except (KeyError, TypeError, ValueError) as e:
            logger.debug("候选点验证失败: %s", e)
            return None

    def repair_theta(self, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel().copy()
        if x.size != len(PARAM_KEYS):
            raise ValueError(f"Expected {len(PARAM_KEYS)} parameters, got {x.size}")

        for idx, key in enumerate(PARAM_KEYS):
            lo, hi = self._bounds[key]
            x[idx] = float(np.clip(x[idx], lo, hi))

        if x[3] + x[4] > self._dsoc_sum_max:
            scale = (self._dsoc_sum_max * 0.995) / max(x[3] + x[4], 1e-12)
            x[3] *= scale
            x[4] *= scale
            x[3] = float(np.clip(x[3], self._bounds["dSOC1"][0], self._bounds["dSOC1"][1]))
            x[4] = float(np.clip(x[4], self._bounds["dSOC2"][0], self._bounds["dSOC2"][1]))
        return x

    def repair_region_bounds(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([self._bounds[key][0] for key in PARAM_KEYS], dtype=float)
        hi = np.array([self._bounds[key][1] for key in PARAM_KEYS], dtype=float)

        lb = np.asarray(lb, dtype=float).ravel()
        ub = np.asarray(ub, dtype=float).ravel()
        if lb.size != len(PARAM_KEYS) or ub.size != len(PARAM_KEYS):
            raise ValueError(f"Expected {len(PARAM_KEYS)}-D bounds, got {lb.size} and {ub.size}")

        lower = np.clip(np.minimum(lb, ub), lo, hi)
        upper = np.clip(np.maximum(lb, ub), lo, hi)

        if upper[3] + upper[4] > self._dsoc_sum_max:
            scale = (self._dsoc_sum_max * 0.995) / max(upper[3] + upper[4], 1e-12)
            upper[3] *= scale
            upper[4] *= scale

        lower = np.minimum(lower, upper)
        return lower, upper

    def parse_guidance(self, responses: List[str]) -> Optional[IterationGuidance]:
        best: Optional[IterationGuidance] = None
        for text in responses:
            parsed = self.extract_json(text)
            if parsed is None:
                continue
            guidance = self._parse_single_guidance(parsed, raw_text=text)
            if guidance is None:
                continue
            if best is None or guidance.confidence > best.confidence:
                best = guidance
        return best

    @staticmethod
    def _coerce_confidence_value(value: Any, default: float = 0.5) -> float:
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except Exception:
                return float(default)
        if isinstance(value, (list, tuple)):
            for item in value:
                try:
                    return ResponseParser._coerce_confidence_value(item, default=default)
                except Exception:
                    continue
        return float(default)

    def _parse_single_guidance(
        self,
        payload: Any,
        *,
        raw_text: str,
    ) -> Optional[IterationGuidance]:
        note = ""
        mode: Optional[str] = None
        confidence: float = 0.5
        point: Optional[np.ndarray] = None
        lb: Optional[np.ndarray] = None
        ub: Optional[np.ndarray] = None

        if isinstance(payload, dict):
            if all(key in payload for key in PARAM_KEYS):
                candidate = self.validate_candidate(payload)
                if candidate is None:
                    return None
                return IterationGuidance(
                    mode="point",
                    confidence=0.35,
                    point=candidate,
                    note="candidate-fallback",
                    raw_text=raw_text,
                )

            mode = str(payload.get("mode", payload.get("type", payload.get("kind", "")))).lower()
            confidence = self._coerce_confidence_value(payload.get("confidence", payload.get("c", 0.5)))
            note = str(payload.get("note", payload.get("reason", payload.get("rationale", ""))))

            if mode == "point":
                raw_point = payload.get("point", payload.get("theta", payload.get("x")))
                if raw_point is None:
                    return None
                point = self.repair_theta(np.asarray(raw_point, dtype=float))
            elif mode == "region":
                raw_region = payload.get("region")
                if (
                    raw_region is not None
                    and isinstance(raw_region, (list, tuple))
                    and len(raw_region) == 2
                ):
                    lb, ub = self.repair_region_bounds(raw_region[0], raw_region[1])
                else:
                    raw_lb = payload.get("lb", payload.get("lower"))
                    raw_ub = payload.get("ub", payload.get("upper"))
                    if raw_lb is None or raw_ub is None:
                        return None
                    lb, ub = self.repair_region_bounds(raw_lb, raw_ub)
            else:
                return None

        elif isinstance(payload, list) and len(payload) >= 3 and isinstance(payload[0], str):
            mode = payload[0].strip().lower()
            confidence = self._coerce_confidence_value(payload[2])
            if mode == "point":
                point = self.repair_theta(np.asarray(payload[1], dtype=float))
            elif mode == "region":
                if not isinstance(payload[1], (list, tuple)) or len(payload[1]) != 2:
                    return None
                lb, ub = self.repair_region_bounds(payload[1][0], payload[1][1])
            else:
                return None
        else:
            return None

        confidence = float(np.clip(confidence, 0.0, 1.0))
        return IterationGuidance(
            mode=mode or "point",
            confidence=confidence,
            point=point,
            lb=lb,
            ub=ub,
            note=note,
            raw_text=raw_text,
        )

    def parse_candidates(self, responses: List[str]) -> List[np.ndarray]:
        """从多个 LLM 响应中解析并合并所有有效候选点（已去重）。"""
        all_valid: List[np.ndarray] = []
        seen = set()

        for resp_idx, text in enumerate(responses):
            parsed = self.extract_json(text)
            if parsed is None:
                continue

            candidates = [parsed] if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

            cnt = 0
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                theta = self.validate_candidate(cand)
                if theta is not None:
                    h = tuple(theta.round(4).tolist())
                    if h not in seen:
                        seen.add(h)
                        all_valid.append(theta)
                        cnt += 1

            logger.debug("响应 %d: 解析出 %d 个有效候选点", resp_idx, cnt)

        logger.info("ResponseParser: 共 %d 个有效候选点（%d 个响应）", len(all_valid), len(responses))
        return all_valid


# ════════════════════════════════════════════════════════════════
# §D  物理启发式回退
# ════════════════════════════════════════════════════════════════

class PhysicsHeuristicFallback:
    """LLM 不可用或响应无效时的回退采样策略。"""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        dsoc_sum_max: float = _DSOC_SUM_MAX,
    ):
        self._lo = np.array([param_bounds[k][0] for k in PARAM_KEYS])
        self._hi = np.array([param_bounds[k][1] for k in PARAM_KEYS])
        self._dsoc_sum_max = float(dsoc_sum_max)

    def physics_informed_warmstart(self, n: int) -> List[np.ndarray]:
        """
        基于领域知识的先验候选点（覆盖 Pareto 极端方向）。
        超出 n=15 的部分由 LHS 补全。
        """
        # 格式：[I1, I2, I3, dSOC1, dSOC2]
        prior_points = [
            # 激进快充：高电流，小 SOC 区间
            np.array([5.5, 4.5, 2.8, 0.20, 0.20]),
            # 保守安全：低电流，大 SOC 区间
            np.array([2.5, 2.5, 2.0, 0.35, 0.25]),
            # 均衡折衷
            np.array([4.0, 3.5, 2.5, 0.25, 0.20]),
            # 偏快，温度控制（I3 低）
            np.array([5.0, 4.0, 2.2, 0.20, 0.25]),
            # 低老化（I2/I3 低，高 SOC 区间小电流）
            np.array([3.5, 3.0, 2.0, 0.30, 0.28]),
            # 大 I1 快速启动，后段保守
            np.array([5.8, 3.0, 2.0, 0.18, 0.22]),
            # 平衡温度和老化
            np.array([3.0, 2.8, 2.2, 0.38, 0.28]),
        ]

        # 新增：8 个极端方向点（覆盖更多 Pareto 区域）
        extreme_points = [
            # 极端时间优先：最大电流，最小 SOC 区间
            np.array([6.0, 5.0, 3.0, 0.15, 0.15]),
            # 极端温度优先：最小电流，最大 SOC 区间
            np.array([2.0, 2.0, 2.0, 0.40, 0.30]),
            # 极端老化优先：渐进电流，大最终 SOC 区间
            np.array([3.5, 3.0, 2.5, 0.35, 0.30]),
            # 时间-温度权衡：高 I1，低 I2/I3
            np.array([5.8, 3.5, 2.2, 0.18, 0.22]),
            # 时间-老化权衡：大 I1，小 I3，大 dSOC2
            np.array([5.5, 4.0, 2.0, 0.20, 0.35]),
            # 温度-老化权衡：低电流，大 SOC 区间
            np.array([2.8, 2.5, 2.2, 0.38, 0.30]),
            # 均衡策略 2
            np.array([4.2, 3.8, 2.6, 0.22, 0.24]),
            # 均衡策略 3
            np.array([3.8, 3.2, 2.4, 0.28, 0.26]),
        ]

        # 合并所有策略点
        all_prior = prior_points + extreme_points

        # 裁剪到 n 个
        candidates = [np.clip(p, self._lo, self._hi) for p in all_prior[:min(n, len(all_prior))]]

        if len(candidates) < n:
            candidates.extend(self.lhs_candidates(n - len(candidates), seed=42))

        return candidates[:n]

    def lhs_candidates(self, n: int, seed: int = 0) -> List[np.ndarray]:
        """Latin Hypercube Sampling，生成边界内均匀分布候选点。"""
        if n <= 0:
            return []
        rng = np.random.default_rng(seed)
        d = len(PARAM_KEYS)
        samples = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            samples[:, j] = (perm + rng.random(n)) / n

        candidates = []
        for i in range(n):
            theta = self._lo + samples[i] * (self._hi - self._lo)
            # 检查 dSOC 约束，违反则微调
            if theta[3] + theta[4] > self._dsoc_sum_max:
                scale = self._dsoc_sum_max / (theta[3] + theta[4]) * 0.99
                theta[3] *= scale
                theta[4] *= scale
            candidates.append(np.clip(theta, self._lo, self._hi))

        return candidates


def _build_iteration_prompt(
    n: int,
    state_dict: Dict,
    param_bounds: Dict,
    pareto_context: str,
) -> str:
    b         = param_bounds
    t         = state_dict.get("iteration", 0)
    T         = state_dict.get("max_iterations", 50)
    w         = np.asarray(state_dict.get("w_vec", [1/3, 1/3, 1/3]))
    best      = state_dict.get("theta_best", np.array([4.0, 3.5, 2.5, 0.25, 0.20]))
    f_min     = state_dict.get("f_min", 0.5)
    mu        = state_dict.get("mu", best)
    sigma     = state_dict.get("sigma", np.array([0.8, 0.6, 0.3, 0.08, 0.05]))
    stag      = state_dict.get("stagnation_count", 0)

    # 解读权重向量，告知 LLM 优化方向
    focus_idx = int(np.argmax(w))
    focus_map = {
        0: f"PRIORITIZE shorter charging time (w_time={w[0]:.2f} is dominant). Try higher I1/I2.",
        1: f"PRIORITIZE lower peak temperature (w_temp={w[1]:.2f} is dominant). Try lower I1, moderate dSOC1.",
        2: f"PRIORITIZE less aging (w_aging={w[2]:.2f} is dominant). Try lower I3, moderate dSOC2.",
    }
    focus_desc = focus_map[focus_idx] if max(w) > 0.45 else "Explore balanced trade-off region."

    stag_guidance = ""
    if stag >= 3:
        stag_guidance = f"\nWARNING: Stagnated for {stag} iterations. Generate more exploratory candidates far from current best."
    elif stag >= 1:
        stag_guidance = "\nHint: Recent iterations showed no improvement. Include some explorative candidates."

    best_str = f"I1={best[0]:.2f}A, I2={best[1]:.2f}A, I3={best[2]:.2f}A, dSOC1={best[3]:.3f}, dSOC2={best[4]:.3f}"
    mu_str   = f"[{mu[0]:.2f}, {mu[1]:.2f}, {mu[2]:.2f}, {mu[3]:.3f}, {mu[4]:.3f}]"
    sig_str  = f"[{sigma[0]:.2f}, {sigma[1]:.2f}, {sigma[2]:.2f}, {sigma[3]:.3f}, {sigma[4]:.3f}]"

    return f"""You are an expert in battery fast charging optimization assisting a Bayesian Optimization loop.

Battery: LG INR21700-M50, 5Ah, 3-stage CC protocol (SOC 0%→80%).
Parameter bounds: I1∈[{b['I1'][0]},{b['I1'][1]}]A, I2∈[{b['I2'][0]},{b['I2'][1]}]A, I3∈[{b['I3'][0]},{b['I3'][1]}]A, dSOC1∈[{b['dSOC1'][0]},{b['dSOC1'][1]}], dSOC2∈[{b['dSOC2'][0]},{b['dSOC2'][1]}]
Constraint: dSOC1 + dSOC2 <= 0.70, and I1 >= I2 >= I3 recommended.

=== Optimization State (iteration {t}/{T}) ===
Current weight vector: time={w[0]:.2f}, temp={w[1]:.2f}, aging={w[2]:.2f}
→ {focus_desc}{stag_guidance}

Current best protocol: {best_str}
Best scalarized objective: {f_min:.6f}
Search center μ: {mu_str}
Search range σ: {sig_str}

{pareto_context}

=== Task ===
Generate {n} candidate protocols. Guidelines:
  1. Focus on the dominant objective as indicated above.
  2. Candidates should be diverse — do not cluster around one point.
  3. Mix exploitation (near μ ± σ) with exploration (boundary regions).
  4. Strictly respect ALL parameter bounds and the dSOC constraint.

Respond with ONLY a JSON array, no other text:
[{{"I1": value, "I2": value, "I3": value, "dSOC1": value, "dSOC2": value}}, ...]"""


def _build_guidance_prompt(
    state_dict: Dict[str, Any],
    param_bounds: Dict[str, Tuple[float, float]],
    pareto_context: str,
) -> str:
    b = param_bounds
    t = int(state_dict.get("iteration", 0))
    T = int(state_dict.get("max_iterations", 50))
    w = np.asarray(state_dict.get("w_vec", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
    best = np.asarray(state_dict.get("theta_best", [4.0, 3.5, 2.5, 0.25, 0.20]), dtype=float)
    f_min = float(state_dict.get("f_min", 0.5))
    stagnation = int(state_dict.get("stagnation_count", 0))
    ideal = np.asarray(state_dict.get("ideal_point", CANONICAL_IDEAL_POINT.tolist()), dtype=float)
    hotspots = state_dict.get("uncertainty_hotspots", [])
    previous_guidance = state_dict.get("previous_guidance")

    hotspots_lines: List[str] = []
    for idx, hotspot in enumerate(hotspots[:5]):
        theta = np.asarray(hotspot.get("theta", []), dtype=float).ravel()
        if theta.size != len(PARAM_KEYS):
            continue
        hotspots_lines.append(
            f"  hotspot[{idx}] std={float(hotspot.get('std', 0.0)):.4f} "
            f"theta=[{theta[0]:.2f}, {theta[1]:.2f}, {theta[2]:.2f}, {theta[3]:.3f}, {theta[4]:.3f}]"
        )
    hotspots_block = "\n".join(hotspots_lines) if hotspots_lines else "  none"

    prev_block = "none"
    if isinstance(previous_guidance, dict) and previous_guidance.get("mode"):
        prev_block = json.dumps(previous_guidance, ensure_ascii=False)

    focus_idx = int(np.argmax(w))
    focus_text = {
        0: "Prioritize faster charging time while respecting thermal and aging constraints.",
        1: "Prioritize lower peak temperature even if charging time becomes longer.",
        2: "Prioritize lower aging and gentler late-stage charging.",
    }[focus_idx]

    return f"""You are guiding a battery charging Bayesian optimization loop.

Battery: LG INR21700-M50, 5Ah, 3-stage CC charging from 0% to 80% SOC.
Decision variables:
  I1 in [{b['I1'][0]}, {b['I1'][1]}]
  I2 in [{b['I2'][0]}, {b['I2'][1]}]
  I3 in [{b['I3'][0]}, {b['I3'][1]}]
  dSOC1 in [{b['dSOC1'][0]}, {b['dSOC1'][1]}]
  dSOC2 in [{b['dSOC2'][0]}, {b['dSOC2'][1]}]
Constraint: dSOC1 + dSOC2 <= 0.70.
Objectives to minimize: charging time [s], temperature rise [K], aging [%].

Iteration {t}/{T}
Weight vector: [time={w[0]:.3f}, temp={w[1]:.3f}, aging={w[2]:.3f}]
Current focus: {focus_text}
Current scalarized best value: {f_min:.6f}
Current best protocol: [{best[0]:.2f}, {best[1]:.2f}, {best[2]:.2f}, {best[3]:.3f}, {best[4]:.3f}]
Current ideal objective estimate: [time={ideal[0]:.2f}, temp={ideal[1]:.2f}, aging={ideal[2]:.6f}]
Stagnation count: {stagnation}
Previous guidance: {prev_block}

High-uncertainty hotspots from the current GP:
{hotspots_block}

Observed optimization history:
{pareto_context}

Task:
Return exactly one JSON value in one of these formats:
["region", [[lb1, lb2, lb3, lb4, lb5], [ub1, ub2, ub3, ub4, ub5]], confidence]
["point", [I1, I2, I3, dSOC1, dSOC2], confidence]

Rules:
1. confidence must be in [0, 1].
2. Use "region" when broad exploration is better, and "point" when a precise promising protocol is clear.
3. Respect all bounds and the dSOC1 + dSOC2 <= 0.70 constraint.
4. Output JSON only, with no markdown, prose, or explanation."""


# ════════════════════════════════════════════════════════════════
# §F  LLMInterface 主类
# ════════════════════════════════════════════════════════════════

class LLMInterface:
    """
    LLM 接口主类，管理 Touchpoint 1b 和 Touchpoint 2。

    满足 acquisition.py 的 LLMPriorProtocol（提供 get_warmstart_center()）。
    """

    def __init__(
        self,
        param_bounds:  Dict[str, Tuple[float, float]],
        config:        Optional[LLMConfig] = None,
        battery_model: Optional[str] = None,
        battery_param_set: str = "Chen2020",
        warmstart_context_level: str = "full",
        warmstart_max_tokens: int = 2500,
        warmstart_max_retries: int = 3,
        warmstart_temperature: Optional[float] = None,
        soc_start: float = 0.0,
        soc_end: float = 0.8,
        dsoc_sum_max: float = _DSOC_SUM_MAX,
    ):
        self._bounds   = param_bounds or DEFAULT_BOUNDS
        self._config   = config or LLMConfig()
        self._battery  = battery_model
        self._battery_param_set = battery_param_set
        self._warmstart_context_level = warmstart_context_level
        self._warmstart_max_tokens = int(warmstart_max_tokens)
        self._warmstart_max_retries = int(warmstart_max_retries)
        self._warmstart_temperature = (
            self._config.temperature if warmstart_temperature is None
            else float(warmstart_temperature)
        )
        self._soc_start = float(soc_start)
        self._soc_end = float(soc_end)
        self._dsoc_sum_max = float(dsoc_sum_max)

        self._caller   = LLMCaller(self._config)
        self._parser   = ResponseParser(self._bounds, dsoc_sum_max=self._dsoc_sum_max)
        self._fallback = PhysicsHeuristicFallback(self._bounds, dsoc_sum_max=self._dsoc_sum_max)
        self._warmstart_context_builder = WarmStartPromptContextBuilder(
            param_bounds=self._bounds,
            battery_name=self._battery,
            param_set=self._battery_param_set,
            soc_start=self._soc_start,
            soc_end=self._soc_end,
            dsoc_sum_max=self._dsoc_sum_max,
            few_shot_examples=None,
        )

        self._warmstart_cache: Optional[List[np.ndarray]] = None

        logger.info(
            "LLMInterface 初始化: backend=%s model=%s warmstart_level=%s param_set=%s",
            self._config.backend,
            self._config.model,
            self._warmstart_context_level,
            self._battery_param_set,
        )

    def _render_warmstart_prompt(self, num_recommendation: int) -> str:
        context = self._warmstart_context_builder.build(num_recommendation=num_recommendation)
        return render_warmstart_prompt(self._warmstart_context_level, context)

    def _build_pareto_context(
        self,
        state_dict: Dict[str, Any],
        *,
        max_observations: int,
        include_top_k: int,
        include_recent: int,
    ) -> str:
        database = state_dict.get("database")
        if database is not None:
            try:
                return database.to_llm_context(
                    max_observations=max_observations,
                    include_pareto=True,
                    include_top_k=include_top_k,
                    include_stats=True,
                    include_recent=include_recent,
                )
            except Exception:
                return ""
        return str(state_dict.get("data_summary", ""))

    def _fallback_iteration_guidance(
        self,
        state_dict: Dict[str, Any],
    ) -> IterationGuidance:
        theta_best = self._parser.repair_theta(
            np.asarray(state_dict.get("theta_best", [4.0, 3.5, 2.5, 0.25, 0.20]), dtype=float)
        )
        w = np.asarray(state_dict.get("w_vec", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
        stagnation = int(state_dict.get("stagnation_count", 0))

        lo = np.array([self._bounds[key][0] for key in PARAM_KEYS], dtype=float)
        hi = np.array([self._bounds[key][1] for key in PARAM_KEYS], dtype=float)
        span = hi - lo

        center = theta_best.copy()
        focus_idx = int(np.argmax(w))
        if focus_idx == 0:
            center += np.array([0.18, 0.14, 0.04, -0.04, -0.03]) * span
        elif focus_idx == 1:
            center += np.array([-0.16, -0.12, -0.05, 0.06, 0.03]) * span
        else:
            center += np.array([-0.10, -0.14, -0.08, 0.04, 0.06]) * span

        center = self._parser.repair_theta(center)
        width_scale = 0.10 + 0.03 * min(stagnation, 3)
        half_width = np.maximum(span * width_scale, np.array([0.15, 0.12, 0.05, 0.02, 0.02]))
        lb, ub = self._parser.repair_region_bounds(center - half_width, center + half_width)

        return IterationGuidance(
            mode="region",
            confidence=0.25,
            lb=lb,
            ub=ub,
            note="heuristic-fallback",
        )

    def query_iteration_guidance(
        self,
        state_dict: Dict[str, Any],
    ) -> IterationGuidance:
        t = int(state_dict.get("iteration", 0))
        logger.info("=== Touchpoint 2b: iterative guidance query (t=%d) ===", t)

        pareto_context = self._build_pareto_context(
            state_dict,
            max_observations=10,
            include_top_k=2,
            include_recent=10,
        )
        prompt = _build_guidance_prompt(state_dict, self._bounds, pareto_context)
        responses = self._caller.call(
            prompt,
            n=max(1, int(self._config.n_samples)),
            temperature=min(float(self._config.temperature), 0.4),
            max_tokens=1200,
        )
        guidance = self._parser.parse_guidance(responses)
        if guidance is None:
            guidance = self._fallback_iteration_guidance(state_dict)

        logger.info(
            "Touchpoint 2b complete: mode=%s confidence=%.3f note=%s",
            guidance.mode,
            guidance.confidence,
            guidance.note or "-",
        )
        return guidance

    # ──────────────────────────────────────────────────────────────
    # Touchpoint 1b: Warm-Start 候选点生成
    # ──────────────────────────────────────────────────────────────
    def generate_warmstart_candidates(
        self,
        n:            int = 15,
        batch_size:   int = 20,
        max_attempts: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        生成 n 个初始充电协议用于 warm-start。

        流程：
          1. 分批调用 LLM（每批 batch_size，最多 max_attempts 批）
          2. 不足 n 个时用物理启发式补全
          3. 缓存结果供 get_warmstart_center() 使用
        """
        logger.info("=== Touchpoint 1b: Warm-Start (n=%d) ===", n)
        if max_attempts is None:
            max_attempts = 4

        all_candidates: List[np.ndarray] = []
        seen = set()

        for batch_idx in range(max_attempts):
            if len(all_candidates) >= n:
                break

            prompt = self._render_warmstart_prompt(batch_size)

            batch: List[np.ndarray] = []
            for retry_idx in range(self._warmstart_max_retries + 1):
                responses = self._caller.call(
                    prompt,
                    temperature=self._warmstart_temperature,
                    max_tokens=self._warmstart_max_tokens,
                )
                batch = self._parser.parse_candidates(responses)
                if batch:
                    break
                logger.info(
                    "  WarmStart 批次 %d/%d 第 %d/%d 次调用未产出有效候选点",
                    batch_idx + 1,
                    max_attempts,
                    retry_idx + 1,
                    self._warmstart_max_retries + 1,
                )

            new_cnt = 0
            for cand in batch:
                h = tuple(cand.round(4).tolist())
                if h not in seen:
                    seen.add(h)
                    all_candidates.append(cand)
                    new_cnt += 1

            logger.info(
                "  批次 %d/%d: 新增 %d 个有效候选点（总计 %d/%d）",
                batch_idx + 1, max_attempts, new_cnt, len(all_candidates), n
            )

        # 不足则用物理启发式补全
        if len(all_candidates) < n:
            shortage = n - len(all_candidates)
            logger.info("  LLM 候选不足，补充 %d 个物理启发式候选点", shortage)
            all_candidates.extend(self._fallback.physics_informed_warmstart(shortage))

        candidates = all_candidates[:n]
        self._warmstart_cache = [c.copy() for c in candidates]
        logger.info("Touchpoint 1b 完成: 返回 %d 个候选点", len(candidates))
        return candidates

    # ──────────────────────────────────────────────────────────────
    # Touchpoint 2: 迭代候选点生成
    # ──────────────────────────────────────────────────────────────
    def generate_iteration_candidates(
        self,
        n:          int,
        state_dict: Dict[str, Any],
    ) -> np.ndarray:
        """
        每迭代生成 n 个候选点。

        state_dict 必须包含的键：
          - iteration (int)
          - max_iterations (int)
          - theta_best (np.ndarray, 5D)
          - f_min (float)
          - mu (np.ndarray, 5D)
          - sigma (np.ndarray, 5D)
          - stagnation_count (int)
          - w_vec (np.ndarray, 3D) ← LLM-AF 耦合的核心接口
          - database (ObservationDB, 可选) ← 用于生成 Pareto 上下文

        Returns
        -------
        np.ndarray (n, 5)
        """
        t = state_dict.get("iteration", 0)
        logger.info("=== Touchpoint 2: 迭代候选生成 (t=%d, n=%d) ===", t, n)

        # 生成 Pareto 上下文
        database = state_dict.get("database")
        if database is not None:
            try:
                pareto_context = database.to_llm_context(
                    max_observations=15,
                    include_pareto=True,
                    include_top_k=3,
                    include_stats=True,
                    include_recent=3,
                )
            except Exception:
                pareto_context = ""
        else:
            pareto_context = state_dict.get("data_summary", "")

        prompt = _build_iteration_prompt(n, state_dict, self._bounds, pareto_context)
        responses = self._caller.call(prompt)
        candidates = self._parser.parse_candidates(responses)

        logger.info("Touchpoint 2: LLM 返回 %d 个有效候选点", len(candidates))

        # 不足则围绕 μ±σ 随机补全
        if len(candidates) < n:
            shortage = n - len(candidates)
            mu    = np.asarray(state_dict.get("mu",    [4.0, 3.5, 2.5, 0.25, 0.20]))
            sigma = np.asarray(state_dict.get("sigma", [0.8, 0.6, 0.3, 0.08, 0.05]))
            lo    = np.array([self._bounds[k][0] for k in PARAM_KEYS])
            hi    = np.array([self._bounds[k][1] for k in PARAM_KEYS])
            rng   = np.random.default_rng()

            logger.info("  不足 %d 个，用 μ±σ 随机补充 %d 个", n, shortage)
            for _ in range(shortage * 5):  # 多次尝试，处理 dSOC 约束
                if len(candidates) >= n:
                    break
                pt = mu + sigma * rng.standard_normal(5)
                pt = np.clip(pt, lo, hi)
                # 处理 dSOC 约束
                if pt[3] + pt[4] > _DSOC_SUM_MAX:
                    scale = _DSOC_SUM_MAX / (pt[3] + pt[4]) * 0.98
                    pt[3] *= scale
                    pt[4] *= scale
                candidates.append(pt)

            # 仍不足则用 LHS 补
            if len(candidates) < n:
                candidates.extend(
                    self._fallback.lhs_candidates(n - len(candidates), seed=t)
                )

        candidates = candidates[:n]
        X = np.stack(candidates)
        logger.info("Touchpoint 2 完成: 返回 %d 个候选点", X.shape[0])
        return X

    # ──────────────────────────────────────────────────────────────
    # LLMPriorProtocol 接口
    # ──────────────────────────────────────────────────────────────
    def get_warmstart_center(self) -> Optional[np.ndarray]:
        """
        返回 warmstart 候选点的均值，用于初始化搜索中心 μ。
        满足 acquisition.py 的 LLMPriorProtocol 接口。
        """
        if not self._warmstart_cache:
            return None
        center = np.mean(self._warmstart_cache, axis=0)
        logger.info("get_warmstart_center: μ_init = %s", center.round(4))
        return center

    @property
    def config(self) -> LLMConfig:
        return self._config


# ════════════════════════════════════════════════════════════════
# §G  工厂函数
# ════════════════════════════════════════════════════════════════

def build_llm_interface(
    param_bounds:  Dict[str, Tuple[float, float]],
    backend:       str   = "openai",
    model:         str   = "gpt-4.1-mini",
    api_base:      str   = "https://api.nuwaapi.com/v1",
    api_key:       str   = "",
    n_samples:     int   = 3,
    temperature:   float = 0.7,
    battery_model: Optional[str] = None,
    battery_param_set: str = "Chen2020",
    warmstart_context_level: str = "full",
    warmstart_max_tokens: int = 2500,
    warmstart_max_retries: int = 3,
    warmstart_temperature: Optional[float] = None,
    soc_start: float = 0.0,
    soc_end: float = 0.8,
    dsoc_sum_max: float = _DSOC_SUM_MAX,
) -> LLMInterface:
    """
    工厂函数：一步构建 LLMInterface。

    用法::

        from llm.llm_interface import build_llm_interface, DEFAULT_BOUNDS

        llm = build_llm_interface(DEFAULT_BOUNDS, backend="openai", model="gpt-4.1-mini")

        # Touchpoint 1b
        ws = llm.generate_warmstart_candidates(n=15)

        # 每迭代 Touchpoint 2
        X_cand = llm.generate_iteration_candidates(15, state_dict={
            "iteration": t,
            "max_iterations": T,
            "theta_best": theta_best,
            "f_min": f_min,
            "mu": mu,
            "sigma": sigma,
            "stagnation_count": stag,
            "w_vec": w_vec,
            "database": db,
        })
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
        battery_model=battery_model,
        battery_param_set=battery_param_set,
        warmstart_context_level=warmstart_context_level,
        warmstart_max_tokens=warmstart_max_tokens,
        warmstart_max_retries=warmstart_max_retries,
        warmstart_temperature=warmstart_temperature,
        soc_start=soc_start,
        soc_end=soc_end,
        dsoc_sum_max=dsoc_sum_max,
    )


# ════════════════════════════════════════════════════════════════
# §H  自测
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(levelname)s %(name)s: %(message)s"
    )

    print("=" * 60)
    print("1. WarmStart Prompt 模板渲染自测")
    print("=" * 60)
    prompt_builder = WarmStartPromptContextBuilder(
        param_bounds=DEFAULT_BOUNDS,
        battery_name=None,
        param_set="Chen2020",
        soc_start=0.0,
        soc_end=0.8,
        dsoc_sum_max=_DSOC_SUM_MAX,
    )
    prompt_context = prompt_builder.build(num_recommendation=6)
    for level in ("none", "partial", "full"):
        rendered = render_warmstart_prompt(level, prompt_context)
        leftovers = PLACEHOLDER_PATTERN.findall(rendered)
        assert not leftovers, f"{level} 模板仍有占位符: {leftovers}"
        print(f"  [{level}] {len(rendered)} chars")
    print("  PASS: 三档 WarmStart Prompt 模板渲染通过")

    print("\n" + "=" * 60)
    print("2. ResponseParser 自测（5D）")
    print("=" * 60)
    parser = ResponseParser(DEFAULT_BOUNDS)

    test_responses = [
        '[{"I1":5.0,"I2":4.0,"I3":2.5,"dSOC1":0.25,"dSOC2":0.20},'
        ' {"I1":2.5,"I2":2.0,"I3":2.0,"dSOC1":0.35,"dSOC2":0.28}]',
        '{"I1":3.5,"I2":3.0,"I3":2.2,"dSOC1":0.30,"dSOC2":0.25}',
        '{"I1":9.0,"I2":4.0,"I3":2.5,"dSOC1":0.25,"dSOC2":0.20}',   # I1 越界
        '{"I1":5.0,"I2":4.0,"I3":2.5,"dSOC1":0.45,"dSOC2":0.30}',   # dSOC1 越界
        '{"I1":5.0,"I2":4.0,"I3":2.5,"dSOC1":0.38,"dSOC2":0.35}',   # dSOC sum > 0.70
        'invalid json',
    ]
    candidates = parser.parse_candidates(test_responses)
    print(f"  解析出 {len(candidates)} 个有效候选点（期望 3）:")
    for i, c in enumerate(candidates):
        print(f"    [{i}] I1={c[0]:.2f} I2={c[1]:.2f} I3={c[2]:.2f} dSOC1={c[3]:.3f} dSOC2={c[4]:.3f}  sum={c[3]+c[4]:.3f}")
    assert len(candidates) == 3, f"期望 3 个，得到 {len(candidates)}"
    print("  PASS: ResponseParser 通过（包含越界和 dSOC 约束过滤）")

    print("\n" + "=" * 60)
    print("3. PhysicsHeuristicFallback 自测")
    print("=" * 60)
    fallback = PhysicsHeuristicFallback(DEFAULT_BOUNDS)
    ws_pts = fallback.physics_informed_warmstart(10)
    print(f"  物理先验候选点 ({len(ws_pts)} 个):")
    lo = np.array([DEFAULT_BOUNDS[k][0] for k in PARAM_KEYS])
    hi = np.array([DEFAULT_BOUNDS[k][1] for k in PARAM_KEYS])
    for i, p in enumerate(ws_pts):
        dSOC_sum = p[3] + p[4]
        in_bounds = np.all(p >= lo) and np.all(p <= hi) and dSOC_sum <= _DSOC_SUM_MAX
        print(
            f"    [{i}] {p.round(3).tolist()}  sum={dSOC_sum:.3f}  PASS"
            if in_bounds else f"    [{i}] FAIL 越界"
        )
        assert in_bounds, f"候选点 {i} 越界!"
    print("  PASS: 所有物理先验候选点在边界内")

    lhs_pts = fallback.lhs_candidates(8, seed=0)
    for p in lhs_pts:
        assert np.all(p >= lo) and np.all(p <= hi)
        assert p[3] + p[4] <= _DSOC_SUM_MAX + 1e-6
    print(f"  PASS: LHS {len(lhs_pts)} 个候选点全部合法")

    print("\n" + "=" * 60)
    print("4. LLMInterface [mock] 完整流程自测")
    print("=" * 60)
    llm = build_llm_interface(
        DEFAULT_BOUNDS,
        backend="mock",
        battery_param_set="Chen2020",
        warmstart_context_level="full",
        warmstart_max_tokens=2500,
        warmstart_max_retries=1,
        soc_start=0.0,
        soc_end=0.8,
        dsoc_sum_max=_DSOC_SUM_MAX,
    )

    # Touchpoint 1b（mock 模式 → 触发物理先验回退）
    ws = llm.generate_warmstart_candidates(n=10)
    print(f"  Touchpoint 1b: {len(ws)} 个 warmstart 候选点")
    assert len(ws) == 10
    for c in ws:
        assert c.shape == (5,)
        assert np.all(c >= lo) and np.all(c <= hi)
        assert c[3] + c[4] <= _DSOC_SUM_MAX + 1e-6

    # LLMPriorProtocol
    center = llm.get_warmstart_center()
    assert center is not None and center.shape == (5,)
    print(f"  Warmstart center: {center.round(3)}")

    # Touchpoint 2
    state = {
        "iteration":        5,
        "max_iterations":   50,
        "theta_best":       np.array([4.0, 3.5, 2.5, 0.25, 0.20]),
        "f_min":            0.35,
        "mu":               np.array([4.0, 3.5, 2.5, 0.25, 0.20]),
        "sigma":            np.array([0.8, 0.6, 0.3, 0.08, 0.05]),
        "stagnation_count": 0,
        "w_vec":            np.array([0.6, 0.2, 0.2]),   # time 导向
    }
    X_cand = llm.generate_iteration_candidates(15, state)
    print(f"  Touchpoint 2: X_cand shape={X_cand.shape}")
    assert X_cand.shape == (15, 5)
    assert np.all(X_cand >= lo) and np.all(X_cand <= hi)
    assert np.all(X_cand[:, 3] + X_cand[:, 4] <= _DSOC_SUM_MAX + 1e-6)

    print("\nPASS: llm_interface.py 全部自测通过")
