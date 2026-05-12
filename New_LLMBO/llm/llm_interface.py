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
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from llmbo.rerank import CandidateInfo, RerankOutput, RerankState
from llmbo.region_lifted_gp import LLMRegionPreference, parse_region_preference_payload
from llmbo.scalarization import compute_tchebycheff_from_raw_with_ideal
from llmbo.warmstart_selector import (
    WarmStartCandidate,
    WarmStartSelectionConfig,
    select_warmstart_portfolio,
)
from utils.constants import (
    DEFAULT_BOUNDS as CANONICAL_DEFAULT_BOUNDS,
    LLM_SAFE_DSOC_SUM_MAX,
    dsoc_sum_violates_limit,
    project_dsoc_pair,
)

try:
    from llm.iteration_prompt import render_iteration_guidance_prompt
    from llm.region_prompt import render_region_preference_prompt
    from llm.rerank_prompt import render_candidate_rerank_prompt
    from llm.warmstart_prompt import (
        DEFAULT_DSOC_SUM_MAX,
        PLACEHOLDER_PATTERN,
        WarmStartPromptContextBuilder,
        render_warmstart_prompt,
    )
except ModuleNotFoundError:  # pragma: no cover - allows direct script execution
    from iteration_prompt import render_iteration_guidance_prompt
    from region_prompt import render_region_preference_prompt
    from rerank_prompt import render_candidate_rerank_prompt
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
# ════════════════════════════════════════════════════════════════════════
# §A  LLM 配置（从环境变量 / .env 加载）
# ════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_default_llm_api_key() -> str:
    """从环境变量或 .env 文件加载默认 API key。"""
    try:
        from dotenv import load_dotenv
        _dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        if _dotenv_path.exists():
            load_dotenv(_dotenv_path)
        else:
            load_dotenv()
    except ImportError:
        pass
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


@lru_cache(maxsize=1)
def _get_default_llm_api_base() -> str:
    """从环境变量或 .env 文件加载默认 API base URL。"""
    try:
        from dotenv import load_dotenv
        _dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        if _dotenv_path.exists():
            load_dotenv(_dotenv_path)
        else:
            load_dotenv()
    except ImportError:
        pass
    return os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "https://api.minimax.chat/v1"


@lru_cache(maxsize=1)
def _get_default_llm_model() -> str:
    """从环境变量或 .env 文件加载默认模型名。"""
    try:
        from dotenv import load_dotenv
        _dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        if _dotenv_path.exists():
            load_dotenv(_dotenv_path)
        else:
            load_dotenv()
    except ImportError:
        pass
    return os.environ.get("LLM_MODEL") or "minimax-4o-mini"



class LLMConfig:
    """LLM 后端配置，支持 openai / anthropic / mock。"""

    def __init__(
        self,
        backend:     str   = "openai",
        model:       str   = _get_default_llm_model(),
        api_base:    str   = _get_default_llm_api_base(),
        api_key:     str   = _get_default_llm_api_key(),
        temperature: float = 0.7,
        n_samples:   int   = 3,
        timeout:     int   = 120,
        request_retries: int = 2,
        retry_backoff_s: float = 1.5,
    ):
        self.backend     = backend
        self.model       = model
        self.api_base    = api_base
        self.api_key     = api_key
        self.temperature = temperature
        self.n_samples   = n_samples
        self.timeout     = timeout
        self.request_retries = int(request_retries)
        self.retry_backoff_s = float(retry_backoff_s)


# ════════════════════════════════════════════════════════════════
# §B  LLM 调用器
# ════════════════════════════════════════════════════════════════

class LLMCaller:
    """统一的 LLM API 调用封装，返回 n 个响应文本列表。"""

    def __init__(self, config: LLMConfig):
        self._cfg = config
        self.last_call_diagnostics: List[Dict[str, Any]] = []

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
        self.last_call_diagnostics = []

        if backend == "mock":
            self.last_call_diagnostics = [
                {"backend": "mock", "model": self._cfg.model, "content_length": 0}
                for _ in range(n)
            ]
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

    @staticmethod
    def _coerce_text_part(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            pieces: List[str] = []
            for key in ("text", "content", "value"):
                if key in value:
                    text = LLMCaller._coerce_text_part(value.get(key))
                    if text:
                        pieces.append(text)
            return "\n".join(pieces)
        if isinstance(value, (list, tuple)):
            return "\n".join(
                text for text in (LLMCaller._coerce_text_part(item) for item in value) if text
            )
        text_attr = getattr(value, "text", None)
        if text_attr is not None:
            return LLMCaller._coerce_text_part(text_attr)
        content_attr = getattr(value, "content", None)
        if content_attr is not None and content_attr is not value:
            return LLMCaller._coerce_text_part(content_attr)
        return ""

    @staticmethod
    def _message_get(message: Any, key: str) -> Any:
        if isinstance(message, Mapping):
            return message.get(key)
        value = getattr(message, key, None)
        if value is not None:
            return value
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, Mapping):
            return extra.get(key)
        return None

    @staticmethod
    def _message_field_names(message: Any) -> List[str]:
        names: List[str] = []
        if isinstance(message, Mapping):
            names.extend(str(key) for key in message.keys())
        else:
            try:
                names.extend(str(key) for key in vars(message).keys())
            except TypeError:
                pass
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, Mapping):
                names.extend(str(key) for key in extra.keys())
        return sorted(set(names))

    @classmethod
    def _extract_message_text(cls, message: Any) -> Tuple[str, str]:
        for key in ("content", "output_text"):
            text = cls._coerce_text_part(cls._message_get(message, key))
            if text.strip():
                return text.strip(), key
        return "", "empty"

    @classmethod
    def _build_openai_diagnostic(
        cls,
        resp: Any,
        choice: Any,
        message: Any,
        text: str,
        source: str,
    ) -> Dict[str, Any]:
        content_text = cls._coerce_text_part(cls._message_get(message, "content"))
        reasoning_text = cls._coerce_text_part(cls._message_get(message, "reasoning_content"))
        usage = getattr(resp, "usage", None)
        usage_dict = None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif isinstance(usage, Mapping):
                usage_dict = dict(usage)
        return {
            "backend": "openai",
            "model": getattr(resp, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "message_field_names": cls._message_field_names(message),
            "content_length": len(content_text),
            "reasoning_content_length": len(reasoning_text),
            "extracted_text_source": source,
            "extracted_text_length": len(text),
            "usage": usage_dict,
        }

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
            success = False
            for attempt in range(self._cfg.request_retries + 1):
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
                    choice = resp.choices[0]
                    message = choice.message
                    content, source = self._extract_message_text(message)
                    self.last_call_diagnostics.append(
                        self._build_openai_diagnostic(resp, choice, message, content, source)
                    )
                    responses.append(content.strip())
                    success = True
                    break
                except Exception as e:
                    self.last_call_diagnostics.append(
                        {
                            "backend": "openai",
                            "model": self._cfg.model,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )
                    if attempt >= self._cfg.request_retries:
                        logger.warning("LLM 调用 %d/%d 失败: %s", i + 1, n, e)
                        break
                    delay = self._cfg.retry_backoff_s * (2 ** attempt)
                    logger.warning(
                        "LLM 调用 %d/%d 第 %d/%d 次失败: %s；%.1fs 后重试",
                        i + 1,
                        n,
                        attempt + 1,
                        self._cfg.request_retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
            if not success:
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
            success = False
            for attempt in range(self._cfg.request_retries + 1):
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
                    success = True
                    break
                except Exception as e:
                    if attempt >= self._cfg.request_retries:
                        logger.warning("Anthropic 调用 %d/%d 失败: %s", i + 1, n, e)
                        break
                    delay = self._cfg.retry_backoff_s * (2 ** attempt)
                    logger.warning(
                        "Anthropic 调用 %d/%d 第 %d/%d 次失败: %s；%.1fs 后重试",
                        i + 1,
                        n,
                        attempt + 1,
                        self._cfg.request_retries + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
            if not success:
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
        soft_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
    ):
        self._bounds = param_bounds
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._soft_dsoc_sum_max = (
            min(self._dsoc_sum_max, float(soft_dsoc_sum_max))
            if soft_dsoc_sum_max is not None else None
        )

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """去除 <think>...</think> 推理标签（如 MiniMax-M2.7 等思考模型会输出）。"""
        return re.sub(r"<think[\s\S]*?</think", "", text).strip()

    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """从 LLM 响应文本中提取 JSON，容错处理。

        支持:
          - 裸 JSON
          - markdown ```json ... ``` 包裹
          - <think>...</think> 推理标签包裹的 JSON（思考型模型）
          - 混有推理标签的 markdown JSON 块
        """
        if not text or not text.strip():
            return None

        # 步骤 1: 去除 <think>...</think> 推理标签（支持思考型模型）
        text = re.sub(r"<think[\s\S]*?</think", "", text).strip()
        if not text:
            return None

        # 步骤 2: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 步骤 3: markdown 代码块（json 或无语言标识）
        for pattern in [r"```(?:json)?\s*([\s\S]*?)\s*```", r"```([\s\S]*?)```"]:
            m = re.search(pattern, text)
            if m:
                inner = m.group(1).strip()
                # 去除代码块内的推理标签
                inner = re.sub(r"<think[\s\S]*?</think", "", inner).strip()
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass

        # 步骤 4: 提取第一个 JSON 数组或对象
        for pattern in [r"(\{[\s\S]*?\})", r"(\[[\s\S]*?\])"]:
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
            if dsoc_sum_violates_limit(values[3], values[4], dsoc_sum_max=self._dsoc_sum_max):
                logger.debug("dSOC1+dSOC2=%.3f > %.2f，候选无效", dSOC_sum, self._dsoc_sum_max)
                return None

            return self.repair_theta(np.array(values, dtype=float))

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

        repair_limit = self._soft_dsoc_sum_max or self._dsoc_sum_max
        if dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=repair_limit):
            x[3], x[4] = project_dsoc_pair(x[3], x[4], dsoc_sum_max=repair_limit)
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

        repair_limit = self._soft_dsoc_sum_max or self._dsoc_sum_max
        if dsoc_sum_violates_limit(upper[3], upper[4], dsoc_sum_max=repair_limit):
            upper[3], upper[4] = project_dsoc_pair(upper[3], upper[4], dsoc_sum_max=repair_limit)

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
        soft_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
    ):
        self._lo = np.array([param_bounds[k][0] for k in PARAM_KEYS])
        self._hi = np.array([param_bounds[k][1] for k in PARAM_KEYS])
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._soft_dsoc_sum_max = (
            min(self._dsoc_sum_max, float(soft_dsoc_sum_max))
            if soft_dsoc_sum_max is not None else None
        )

    def _repair_theta(self, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel().copy()
        x = np.clip(x, self._lo, self._hi)
        repair_limit = self._soft_dsoc_sum_max or self._dsoc_sum_max
        if dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=repair_limit):
            x[3], x[4] = project_dsoc_pair(x[3], x[4], dsoc_sum_max=repair_limit)
            x = np.clip(x, self._lo, self._hi)
        return x

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
        candidates = [self._repair_theta(p) for p in all_prior[:min(n, len(all_prior))]]

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
            candidates.append(self._repair_theta(theta))

        return candidates


def _build_iteration_prompt(
    n: int,
    state_dict: Dict,
    param_bounds: Dict,
    pareto_context: str,
    include_fewshot: bool = True,
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

    # ──────────────────────────────────────────────────────
    # Few-Shot 历史示例
    # ──────────────────────────────────────────────────────
    fewshot_block = ""
    if include_fewshot:
        database = state_dict.get("database")
        if database is not None:
            try:
                feasible = database.get_feasible()
                if len(feasible) >= 5:
                    y_min = state_dict.get("y_min", getattr(database, "_y_min", None))
                    y_max = state_dict.get("y_max", getattr(database, "_y_max", None))
                    ideal = state_dict.get("ideal_point_raw", getattr(database, "_ideal_point_raw", None))
                    eta = float(state_dict.get("eta", getattr(database, "_eta", 0.05)))
                    if y_min is None or y_max is None or ideal is None:
                        scored = []
                    else:
                        Y_raw = np.array([obs.objectives for obs in feasible], dtype=float)
                        scores = compute_tchebycheff_from_raw_with_ideal(
                            Y_raw,
                            w,
                            np.asarray(ideal, dtype=float),
                            np.asarray(y_min, dtype=float),
                            np.asarray(y_max, dtype=float),
                            eta=eta,
                        )
                        scored = list(zip(scores, feasible))
                    if not scored:
                        raise ValueError("scalarization context unavailable for few-shot prompt")
                    scored.sort(key=lambda x: x[0])

                    top_3 = scored[:3]
                    worst_2 = scored[-2:]

                    examples = []
                    for score, obs in top_3:
                        examples.append(f"Protocol: I1={obs.theta[0]:.2f}, I2={obs.theta[1]:.2f}, I3={obs.theta[2]:.2f}, dSOC1={obs.theta[3]:.3f}, dSOC2={obs.theta[4]:.3f} -> Score: {score:.6f} * (excellent)")
                    for score, obs in worst_2:
                        examples.append(f"Protocol: I1={obs.theta[0]:.2f}, I2={obs.theta[1]:.2f}, I3={obs.theta[2]:.2f}, dSOC1={obs.theta[3]:.3f}, dSOC2={obs.theta[4]:.3f} -> Score: {score:.6f} X (poor)")

                    fewshot_block = "\n".join(examples)
            except Exception:
                fewshot_block = ""

    return f"""You are an expert in battery fast charging optimization assisting a Bayesian Optimization loop.

Battery: LG INR21700-M50, 5Ah, 3-stage CC protocol (SOC 0%→80%).
Parameter bounds: I1∈[{b['I1'][0]},{b['I1'][1]}]A, I2∈[{b['I2'][0]},{b['I2'][1]}]A, I3∈[{b['I3'][0]},{b['I3'][1]}]A, dSOC1∈[{b['dSOC1'][0]},{b['dSOC1'][1]}], dSOC2∈[{b['dSOC2'][0]},{b['dSOC2'][1]}]
Practical hard limit: keep dSOC1 + dSOC2 strictly below 0.70 because points at exactly 0.70 can be rejected by the simulator.
Safety margin: keep dSOC1 + dSOC2 <= 0.65 whenever possible.
Current profile: I1 >= I2 >= I3 is recommended.

=== Historical Performance Examples ===
{fewshot_block}

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
  4. Strictly respect ALL parameter bounds, stay below 0.70 on dSOC1 + dSOC2, and avoid points above 0.65 unless there is a strong trade-off reason.

Respond with ONLY a JSON array, no other text:
[{{"I1": value, "I2": value, "I3": value, "dSOC1": value, "dSOC2": value}}, ...]"""


def _build_guidance_prompt(
    state_dict: Dict[str, Any],
    param_bounds: Dict[str, Tuple[float, float]],
    pareto_context: str,
    battery_model: Optional[str] = None,
) -> str:
    return render_iteration_guidance_prompt(
        state_dict=state_dict,
        param_bounds=param_bounds,
        pareto_context=pareto_context,
        battery_name=battery_model,
        safe_dsoc_sum_max=float(state_dict.get("safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
        hard_dsoc_sum_max=float(state_dict.get("hard_dsoc_sum_max", _DSOC_SUM_MAX)),
    )


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
        warmstart_prompt_version: Optional[str] = None,
        enable_iteration_fewshot: bool = True,
        warmstart_max_tokens: int = 2500,
        region_preference_max_tokens: int = 4096,
        region_preference_prompt_version: str = "default",
        warmstart_max_retries: int = 3,
        warmstart_temperature: Optional[float] = None,
        soc_start: float = 0.0,
        soc_end: float = 0.8,
        dsoc_sum_max: float = _DSOC_SUM_MAX,
        safe_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
        enable_warmstart_portfolio: bool = True,
        warmstart_pool_size: int = 16,
        warmstart_diversity_weight: float = 0.45,
        warmstart_soft_penalty_weight: float = 0.65,
        warmstart_monotone_bonus: float = 0.08,
        warmstart_archive_bonus_weight: float = 0.0,
        warmstart_boundary_probe_limit: int = 1,
        warmstart_cache_path: Optional[str] = None,
        warmstart_cache_mode: str = "read_write",
        warmstart_cache_use_selected: bool = False,
    ):
        self._bounds   = param_bounds or DEFAULT_BOUNDS
        self._config   = config or LLMConfig()
        self._battery  = battery_model
        self._battery_param_set = battery_param_set
        self._warmstart_context_level = warmstart_context_level
        self._warmstart_prompt_version = warmstart_prompt_version
        self._enable_iteration_fewshot = enable_iteration_fewshot
        self._warmstart_max_tokens = int(warmstart_max_tokens)
        self._region_preference_max_tokens = int(region_preference_max_tokens)
        self._region_preference_prompt_version = str(region_preference_prompt_version or "default")
        self._warmstart_max_retries = int(warmstart_max_retries)
        self._warmstart_temperature = (
            self._config.temperature if warmstart_temperature is None
            else float(warmstart_temperature)
        )
        self._soc_start = float(soc_start)
        self._soc_end = float(soc_end)
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._safe_dsoc_sum_max = (
            min(self._dsoc_sum_max, float(safe_dsoc_sum_max))
            if safe_dsoc_sum_max is not None else None
        )
        self._enable_warmstart_portfolio = bool(enable_warmstart_portfolio)
        self._warmstart_pool_size = max(1, int(warmstart_pool_size))
        self._warmstart_diversity_weight = float(warmstart_diversity_weight)
        self._warmstart_soft_penalty_weight = float(warmstart_soft_penalty_weight)
        self._warmstart_monotone_bonus = float(warmstart_monotone_bonus)
        self._warmstart_archive_bonus_weight = float(warmstart_archive_bonus_weight)
        self._warmstart_boundary_probe_limit = max(0, int(warmstart_boundary_probe_limit))
        self._warmstart_cache_path = Path(str(warmstart_cache_path)) if warmstart_cache_path else None
        self._warmstart_cache_mode = str(warmstart_cache_mode or "read_write").lower()
        self._warmstart_cache_use_selected = bool(warmstart_cache_use_selected)

        self._caller   = LLMCaller(self._config)
        self._parser   = ResponseParser(
            self._bounds,
            dsoc_sum_max=self._dsoc_sum_max,
            soft_dsoc_sum_max=self._safe_dsoc_sum_max,
        )
        self._fallback = PhysicsHeuristicFallback(
            self._bounds,
            dsoc_sum_max=self._dsoc_sum_max,
            soft_dsoc_sum_max=self._safe_dsoc_sum_max,
        )
        self._warmstart_context_builder = WarmStartPromptContextBuilder(
            param_bounds=self._bounds,
            battery_name=self._battery,
            param_set=self._battery_param_set,
            soc_start=self._soc_start,
            soc_end=self._soc_end,
            dsoc_sum_max=self._dsoc_sum_max,
            safe_dsoc_sum_max=self._safe_dsoc_sum_max,
            few_shot_examples=None,
        )

        self._warmstart_cache: Optional[List[np.ndarray]] = None
        self._warmstart_summary: Dict[str, Any] = {}

        logger.info(
            "LLMInterface 初始化: backend=%s model=%s warmstart_level=%s param_set=%s",
            self._config.backend,
            self._config.model,
            self._warmstart_context_level,
            self._battery_param_set,
        )

    def _render_warmstart_prompt(self, num_recommendation: int) -> str:
        context = self._warmstart_context_builder.build(num_recommendation=num_recommendation)
        # Use explicit prompt_version if set, otherwise fall back to context_level
        level = self._warmstart_prompt_version or self._warmstart_context_level
        return render_warmstart_prompt(level, context)

    @staticmethod
    def _coerce_theta_list(rows: Any) -> List[np.ndarray]:
        if not isinstance(rows, list):
            return []
        points: List[np.ndarray] = []
        for row in rows:
            try:
                theta = np.asarray(row, dtype=float).ravel()
            except Exception:
                continue
            if theta.size == len(PARAM_KEYS) and np.all(np.isfinite(theta)):
                points.append(theta)
        return points

    def _load_warmstart_disk_cache(self) -> Optional[Dict[str, Any]]:
        if self._warmstart_cache_path is None:
            return None
        if self._warmstart_cache_mode not in {"read", "read_write"}:
            return None
        if not self._warmstart_cache_path.exists():
            return None
        try:
            with open(self._warmstart_cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read warmstart cache %s: %s", self._warmstart_cache_path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_warmstart_disk_cache(
        self,
        *,
        candidate_pool: List[np.ndarray],
        selected: List[np.ndarray],
        summary: Dict[str, Any],
        target_pool: int,
    ) -> None:
        if self._warmstart_cache_path is None:
            return
        if self._warmstart_cache_mode not in {"write", "read_write"}:
            return
        payload = {
            "version": 1,
            "backend": str(self._config.backend),
            "model": str(self._config.model),
            "temperature": float(self._warmstart_temperature),
            "target_pool": int(target_pool),
            "candidate_pool": [np.asarray(theta, dtype=float).ravel().tolist() for theta in candidate_pool],
            "final_selected": [np.asarray(theta, dtype=float).ravel().tolist() for theta in selected],
            "summary": summary,
        }
        try:
            self._warmstart_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._warmstart_cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to save warmstart cache %s: %s", self._warmstart_cache_path, exc)

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

    def _fallback_region_preference(
        self,
        state_dict: Dict[str, Any],
    ) -> LLMRegionPreference:
        rows = state_dict.get("top_scalar_points") or []
        top_points: List[np.ndarray] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            theta = row.get("theta")
            if theta is None:
                continue
            try:
                point = self._parser.repair_theta(np.asarray(theta, dtype=float).ravel())
            except Exception:
                continue
            if point.size == len(PARAM_KEYS):
                top_points.append(point)

        if not top_points:
            recent = state_dict.get("recent_observations") or []
            for row in recent:
                if not isinstance(row, dict):
                    continue
                theta = row.get("theta")
                if theta is None:
                    continue
                try:
                    point = self._parser.repair_theta(np.asarray(theta, dtype=float).ravel())
                except Exception:
                    continue
                if point.size == len(PARAM_KEYS):
                    top_points.append(point)
                if len(top_points) >= 3:
                    break

        if not top_points:
            return LLMRegionPreference.none("mock_no_candidates")

        count = min(len(top_points), 3)
        weights = np.linspace(float(count), 1.0, count, dtype=float)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        center = np.sum(np.vstack(top_points[:count]) * weights[:, None], axis=0)
        center = self._parser.repair_theta(center)
        point_dict = {
            key: float(center[idx])
            for idx, key in enumerate(PARAM_KEYS)
        }
        payload = {
            "kind": "point",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "point": point_dict,
            "confidence": 0.72,
            "preference_type": "balanced",
            "reason": "mock fallback: weighted center of top scalarized historical points",
            "mechanistic_thinking": "Recent good scalarized points suggest a balanced current schedule with moderate SOC splits remains plausible.",
            "risk_flags": [],
        }
        pref = parse_region_preference_payload(payload)
        pref.raw_response = dict(payload)
        pref.raw_response_hash = str(abs(hash(json.dumps(payload, sort_keys=True))))
        pref.parser_status = "ok"
        return pref

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
        prompt = _build_guidance_prompt(
            state_dict,
            self._bounds,
            pareto_context,
            battery_model=self._battery,
        )
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

    def _extract_json_flexible(self, text: str) -> Optional[Any]:
        """
        Extract JSON from LLM response with multiple fallback strategies.
        More tolerant than the standard extract_json.
        """
        import json
        import re

        if not text or not text.strip():
            return None
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        patterns = [
            r'```(?:json)?\s*([\s\S]*?)\s*```',
            r'`({[^`]+})`',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Strategy 3: Find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try to fix common JSON errors
        fixed = re.sub(r',(\s*[}\]])', r'\1', text)
        fixed = re.sub(r"(?<!\\)'", '"', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None

    def query_region_preference(
        self,
        state_dict: Dict[str, Any],
    ) -> LLMRegionPreference:
        """
        Region-Lifted GP preference query with improved parsing tolerance.

        This path uses flexible JSON extraction and detailed logging to diagnose
        parsing issues. Invalid responses become kind="none" so the optimizer
        can fail-open to plain EI.
        """
        t = int(state_dict.get("iteration", 0))
        logger.info("=== Region-Lifted GP preference query (t=%d) ===", t)
        if str(self._config.backend).lower() == "mock":
            return self._fallback_region_preference(state_dict)

        prompt = render_region_preference_prompt(
            state=state_dict,
            param_bounds=self._bounds,
            prompt_version=self._region_preference_prompt_version,
        )
        logger.debug("Region preference prompt length: %d chars", len(prompt))

        responses = self._caller.call(
            prompt,
            n=1,
            temperature=min(float(self._config.temperature), 0.3),
            max_tokens=self._region_preference_max_tokens,
        )
        call_diagnostics = list(getattr(self._caller, "last_call_diagnostics", []) or [])

        for idx, text in enumerate(responses):
            logger.debug("Processing response %d/%d (length: %d)", idx + 1, len(responses), len(text))

            # Try flexible JSON extraction first
            parsed = self._extract_json_flexible(text)
            if parsed is None:
                logger.warning("Response %d: JSON extraction failed", idx + 1)
                logger.debug("Response preview: %s", self._safe_text_preview(text, 200))
                continue

            logger.debug("Response %d: JSON extracted successfully, type=%s", idx + 1, type(parsed).__name__)

            # Parse the region preference
            pref = parse_region_preference_payload(parsed, log_level=logging.INFO)

            # If parsing succeeded, enrich and return
            if pref.parser_status == "ok":
                if pref.raw_response_hash is None:
                    pref.raw_response_hash = str(abs(hash(text)))
                pref.raw_text_preview = self._safe_text_preview(json.dumps(parsed, ensure_ascii=True, sort_keys=True))
                pref.llm_call_diagnostics = call_diagnostics
                logger.info("Region preference parsed successfully: kind=%s, confidence=%.2f", pref.kind, pref.confidence)
                return pref
            else:
                logger.warning("Response %d: parse_region_preference_payload failed with status '%s'", idx + 1, pref.parser_status)

        # All responses failed - determine failure reason
        parser_status: Optional[str] = None
        if call_diagnostics:
            error_types = {
                str(item.get("error_type", "")).strip()
                for item in call_diagnostics
                if isinstance(item, dict) and item.get("error_type")
            }
            if "PermissionDeniedError" in error_types:
                parser_status = "query_permission_denied"
            elif error_types:
                parser_status = "query_exception"
        if parser_status is None and not responses:
            parser_status = "no_responses"
        elif parser_status is None:
            parser_status = "parse_fail"

        logger.error("All region preference responses failed, status=%s", parser_status)
        pref = LLMRegionPreference.none(parser_status)
        pref.raw_text_preview = self._safe_text_preview("\n".join(responses))
        pref.llm_call_diagnostics = call_diagnostics
        return pref

    @staticmethod
    def _safe_text_preview(text: str, limit: int = 600) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "...<truncated>"

    # ──────────────────────────────────────────────────────────────
    # Touchpoint 1b: Warm-Start 候选点生成
    # ──────────────────────────────────────────────────────────────
    def _fallback_candidate_goodness(
        self,
        state: RerankState,
        candidates: List[CandidateInfo],
    ) -> List[RerankOutput]:
        outputs: List[RerankOutput] = []
        for candidate in candidates:
            scale = max(float(candidate.sigma_fw), 1e-6)
            z = float(np.clip((float(state.tau_t) - float(candidate.mu_fw)) / scale, -8.0, 8.0))
            q_good = float(1.0 / (1.0 + np.exp(-z)))
            confidence = float(np.clip(0.35 + 0.15 * min(abs(z), 3.0), 0.35, 0.9))
            rationale = (
                "gp fallback: likely below threshold"
                if q_good >= 0.5 else
                "gp fallback: likely above threshold"
            )
            outputs.append(
                RerankOutput(
                    idx=int(candidate.idx),
                    q_good=q_good,
                    confidence=confidence,
                    rationale_short=rationale,
                    risk_flags=[],
                )
            )
        return outputs

    def score_candidate_goodness(
        self,
        state: RerankState,
        candidates: List[CandidateInfo],
    ) -> List[RerankOutput]:
        if not candidates:
            return []
        if str(self._config.backend).lower() == "mock":
            return self._fallback_candidate_goodness(state, candidates)

        prompt = render_candidate_rerank_prompt(
            state=state,
            candidates=candidates,
            param_bounds=self._bounds,
            scalarization_formula=(
                "Lower scalarized objective is better under the current weight vector."
            ),
            safe_dsoc_sum_max=float(self._safe_dsoc_sum_max or self._dsoc_sum_max),
            hard_dsoc_sum_max=float(self._dsoc_sum_max),
        )

        responses = self._caller.call(
            prompt,
            n=max(1, int(self._config.n_samples)),
            temperature=min(float(self._config.temperature), 0.2),
            max_tokens=1200,
        )

        candidate_ids = {int(candidate.idx) for candidate in candidates}
        aggregated: Dict[int, List[Dict[str, Any]]] = {}

        for text in responses:
            payload = ResponseParser.extract_json(text)
            if not isinstance(payload, dict):
                continue
            rows = payload.get("candidates")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    idx = int(row.get("candidate_id", row.get("idx")))
                except Exception:
                    continue
                if idx not in candidate_ids:
                    continue
                try:
                    q_good = float(np.clip(float(row.get("q_good", 0.5)), 0.0, 1.0))
                except Exception:
                    q_good = 0.5
                try:
                    confidence = float(np.clip(float(row.get("confidence", 0.5)), 0.0, 1.0))
                except Exception:
                    confidence = 0.5
                risk_flags = row.get("risk_flags", [])
                if not isinstance(risk_flags, list):
                    risk_flags = []
                aggregated.setdefault(idx, []).append(
                    {
                        "q_good": q_good,
                        "confidence": confidence,
                        "rationale_short": str(row.get("rationale_short", row.get("note", "")))[:120],
                        "risk_flags": [str(flag)[:64] for flag in risk_flags[:5]],
                    }
                )

        if not aggregated:
            return []

        outputs: List[RerankOutput] = []
        for candidate in candidates:
            rows = aggregated.get(int(candidate.idx))
            if not rows:
                continue
            q_good = float(np.mean([float(row["q_good"]) for row in rows]))
            confidence = float(np.mean([float(row["confidence"]) for row in rows]))
            best_row = max(rows, key=lambda row: float(row["confidence"]))
            outputs.append(
                RerankOutput(
                    idx=int(candidate.idx),
                    q_good=float(np.clip(q_good, 0.0, 1.0)),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    rationale_short=str(best_row.get("rationale_short", "")),
                    risk_flags=list(best_row.get("risk_flags", [])),
                )
            )
        return outputs

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
        target_pool = max(
            int(n),
            int(batch_size),
            int(self._warmstart_pool_size),
        )
        disk_cache = self._load_warmstart_disk_cache()
        cache_hit = False
        if disk_cache is not None:
            cached_selected = self._coerce_theta_list(disk_cache.get("final_selected"))
            cached_pool = self._coerce_theta_list(disk_cache.get("candidate_pool"))
            if self._warmstart_cache_use_selected and len(cached_selected) >= int(n):
                candidates = [self._parser.repair_theta(theta) for theta in cached_selected[: int(n)]]
                self._warmstart_summary = dict(disk_cache.get("summary") or {})
                self._warmstart_summary.update(
                    {
                        "disk_cache": "hit_selected",
                        "cache_path": str(self._warmstart_cache_path),
                        "final_selected_count": int(len(candidates)),
                        "final_selected": [np.asarray(theta, dtype=float).ravel().tolist() for theta in candidates],
                    }
                )
                self._warmstart_cache = [c.copy() for c in candidates]
                logger.info("Touchpoint 1b using selected warmstart cache: %s", self._warmstart_cache_path)
                return candidates
            if cached_pool:
                all_candidates = [self._parser.repair_theta(theta) for theta in cached_pool]
                cache_hit = True
                logger.info(
                    "Touchpoint 1b loaded %d cached warmstart pool points from %s",
                    len(all_candidates),
                    self._warmstart_cache_path,
                )

        for batch_idx in range(max_attempts):
            if cache_hit:
                break
            if len(all_candidates) >= target_pool:
                break

            request_size = max(int(batch_size), min(target_pool - len(all_candidates), target_pool))
            prompt = self._render_warmstart_prompt(request_size)

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
                batch_idx + 1, max_attempts, new_cnt, len(all_candidates), target_pool
            )

        # 不足则用物理启发式补全
        if len(all_candidates) < target_pool:
            shortage = target_pool - len(all_candidates)
            logger.info("  LLM 候选不足，补充 %d 个物理启发式候选点", shortage)
            all_candidates.extend(self._fallback.physics_informed_warmstart(shortage))

        if self._enable_warmstart_portfolio:
            wrapped = [
                WarmStartCandidate(theta=np.asarray(theta, dtype=float), source="llm_pool", raw_index=i)
                for i, theta in enumerate(all_candidates)
            ]
            cfg = WarmStartSelectionConfig(
                n_select=int(n),
                bounds=self._bounds,
                hard_dsoc_sum_max=float(self._dsoc_sum_max),
                soft_dsoc_sum_max=float(self._safe_dsoc_sum_max or self._dsoc_sum_max),
                diversity_weight=float(self._warmstart_diversity_weight),
                soft_penalty_weight=float(self._warmstart_soft_penalty_weight),
                monotone_bonus=float(self._warmstart_monotone_bonus),
                archive_bonus_weight=float(self._warmstart_archive_bonus_weight),
                boundary_probe_limit=int(self._warmstart_boundary_probe_limit),
            )
            selected, summary = select_warmstart_portfolio(wrapped, cfg)
            candidates = [np.asarray(item.theta, dtype=float).copy() for item in selected]
            if len(candidates) < n:
                shortage = int(n) - len(candidates)
                logger.info("  Portfolio selector returned %d/%d; appending fallback points", len(candidates), n)
                candidates.extend(self._fallback.physics_informed_warmstart(shortage))
            self._warmstart_summary = {
                **summary,
                "enabled": True,
                "pool_target": int(target_pool),
                "pool_collected": int(len(all_candidates)),
                "disk_cache": "hit_pool" if cache_hit else "miss",
                "cache_path": None if self._warmstart_cache_path is None else str(self._warmstart_cache_path),
            }
        else:
            candidates = all_candidates[:n]
            self._warmstart_summary = {
                "enabled": False,
                "method": "first_n",
                "requested": int(n),
                "pool_target": int(target_pool),
                "pool_collected": int(len(all_candidates)),
                "selected_count": int(len(candidates)),
                "selected": [np.asarray(theta, dtype=float).ravel().tolist() for theta in candidates],
                "disk_cache": "hit_pool" if cache_hit else "miss",
                "cache_path": None if self._warmstart_cache_path is None else str(self._warmstart_cache_path),
            }
        self._warmstart_summary["final_selected_count"] = int(len(candidates))
        self._warmstart_summary["final_selected"] = [
            np.asarray(theta, dtype=float).ravel().tolist() for theta in candidates
        ]
        self._save_warmstart_disk_cache(
            candidate_pool=all_candidates,
            selected=candidates,
            summary=self._warmstart_summary,
            target_pool=target_pool,
        )
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
        logger.info(
            "generate_iteration_candidates is a legacy path; optimizer mainline uses guidance/rerank interfaces"
        )
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

        prompt = _build_iteration_prompt(
            n, state_dict, self._bounds, pareto_context,
            include_fewshot=self._enable_iteration_fewshot
        )
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
                candidates.append(self._parser.repair_theta(np.clip(pt, lo, hi)))

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

    def get_warmstart_summary(self) -> Dict[str, Any]:
        return dict(self._warmstart_summary)

    @property
    def config(self) -> LLMConfig:
        return self._config


# ════════════════════════════════════════════════════════════════
# §G  工厂函数
# ════════════════════════════════════════════════════════════════

def build_llm_interface(
    param_bounds:  Dict[str, Tuple[float, float]],
    backend:       str   = "openai",
    model:         str   = _get_default_llm_model(),
    api_base:      str   = _get_default_llm_api_base(),
    api_key:       str   = _get_default_llm_api_key(),
    n_samples:     int   = 3,
    temperature:   float = 0.7,
    battery_model: Optional[str] = None,
    battery_param_set: str = "Chen2020",
    warmstart_context_level: str = "full",
    warmstart_prompt_version: Optional[str] = None,
    enable_iteration_fewshot: bool = True,
    warmstart_max_tokens: int = 2500,
    region_preference_max_tokens: int = 4096,
    region_preference_prompt_version: str = "default",
    warmstart_max_retries: int = 3,
    warmstart_temperature: Optional[float] = None,
    soc_start: float = 0.0,
    soc_end: float = 0.8,
    dsoc_sum_max: float = _DSOC_SUM_MAX,
    safe_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
    enable_warmstart_portfolio: bool = True,
    warmstart_pool_size: int = 16,
    warmstart_diversity_weight: float = 0.45,
    warmstart_soft_penalty_weight: float = 0.65,
    warmstart_monotone_bonus: float = 0.08,
    warmstart_archive_bonus_weight: float = 0.0,
    warmstart_boundary_probe_limit: int = 1,
    warmstart_cache_path: Optional[str] = None,
    warmstart_cache_mode: str = "read_write",
    warmstart_cache_use_selected: bool = False,
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
        warmstart_prompt_version=warmstart_prompt_version,
        enable_iteration_fewshot=enable_iteration_fewshot,
        warmstart_max_tokens=warmstart_max_tokens,
        region_preference_max_tokens=region_preference_max_tokens,
        region_preference_prompt_version=region_preference_prompt_version,
        warmstart_max_retries=warmstart_max_retries,
        warmstart_temperature=warmstart_temperature,
        soc_start=soc_start,
        soc_end=soc_end,
        dsoc_sum_max=dsoc_sum_max,
        safe_dsoc_sum_max=safe_dsoc_sum_max,
        enable_warmstart_portfolio=enable_warmstart_portfolio,
        warmstart_pool_size=warmstart_pool_size,
        warmstart_diversity_weight=warmstart_diversity_weight,
        warmstart_soft_penalty_weight=warmstart_soft_penalty_weight,
        warmstart_monotone_bonus=warmstart_monotone_bonus,
        warmstart_archive_bonus_weight=warmstart_archive_bonus_weight,
        warmstart_boundary_probe_limit=warmstart_boundary_probe_limit,
        warmstart_cache_path=warmstart_cache_path,
        warmstart_cache_mode=warmstart_cache_mode,
        warmstart_cache_use_selected=warmstart_cache_use_selected,
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
