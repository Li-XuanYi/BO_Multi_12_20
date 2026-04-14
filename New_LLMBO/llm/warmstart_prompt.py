"""
WarmStart prompt templates and rendering utilities.

This module keeps warm-start prompts out of llm_interface.py and organizes them
into three layers:
  1. Text templates on disk
  2. A context builder that resolves runtime placeholders
  3. A small battery metadata registry that centralizes domain wording
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from utils.constants import DEFAULT_BOUNDS as CANONICAL_DEFAULT_BOUNDS, DSOC_SUM_MAX as CANONICAL_DSOC_SUM_MAX

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates" / "warmstart"
WARMSTART_TEMPLATE_MAP = {
    "none": "basic",
    "partial": "problem",
    "full": "detailed",
}
PLACEHOLDER_PATTERN = re.compile(r"\[([A-Z][A-Z0-9_]{1,})\]")
DEFAULT_DSOC_SUM_MAX = CANONICAL_DSOC_SUM_MAX


@dataclass(frozen=True)
class BatteryPromptMetadata:
    param_set: str
    battery_name: str
    chemistry: str
    nominal_capacity_ah: float
    param_set_display: str
    expert_knowledge: Tuple[str, ...]


BATTERY_METADATA_REGISTRY: Dict[str, BatteryPromptMetadata] = {
    "Chen2020": BatteryPromptMetadata(
        param_set="Chen2020",
        battery_name="LG INR21700-M50",
        chemistry="NMC811/Graphite",
        nominal_capacity_ah=5.0,
        param_set_display="Chen2020 parameter set",
        expert_knowledge=(
            "Increasing I1 and I2 usually shortens charging time but raises peak temperature and aging risk.",
            "A larger dSOC1 keeps the cell at high current for longer, which is usually fast but thermally aggressive.",
            "Lower I3 and a meaningful final-stage SOC window help protect the cell in the high-SOC region.",
            "Balanced protocols usually combine a strong first stage with progressively safer later stages instead of using uniformly high current.",
        ),
    ),
}


def resolve_battery_metadata(
    param_set: str,
    battery_name: Optional[str] = None,
) -> BatteryPromptMetadata:
    meta = BATTERY_METADATA_REGISTRY.get(param_set)
    if meta is None:
        resolved_name = battery_name or "custom lithium-ion cell"
        return BatteryPromptMetadata(
            param_set=param_set,
            battery_name=resolved_name,
            chemistry="lithium-ion",
            nominal_capacity_ah=5.0,
            param_set_display=f"{param_set} parameter set",
            expert_knowledge=(
                "Higher charging currents usually reduce time but increase thermal and aging stress.",
                "A progressively decreasing current profile is often safer than holding high current into the high-SOC region.",
            ),
        )

    if battery_name is None or battery_name.strip() == "":
        return meta

    return BatteryPromptMetadata(
        param_set=meta.param_set,
        battery_name=battery_name,
        chemistry=meta.chemistry,
        nominal_capacity_ah=meta.nominal_capacity_ah,
        param_set_display=meta.param_set_display,
        expert_knowledge=meta.expert_knowledge,
    )


class WarmStartTemplateRenderer:
    """Load and render text templates that use [PLACEHOLDER] tokens."""

    def __init__(self, template_dir: Optional[Path] = None):
        self._template_dir = Path(template_dir or DEFAULT_TEMPLATE_DIR)
        self._cache: Dict[str, str] = {}

    def load(self, template_name: str) -> str:
        if template_name not in self._cache:
            template_path = self._template_dir / f"{template_name}.txt"
            if not template_path.exists():
                raise FileNotFoundError(f"WarmStart template not found: {template_path}")
            self._cache[template_name] = template_path.read_text(encoding="utf-8")
        return self._cache[template_name]

    def render(self, template_name: str, context: Mapping[str, str]) -> str:
        rendered = self.load(template_name)
        for key, value in context.items():
            rendered = rendered.replace(f"[{key}]", str(value))

        leftovers = PLACEHOLDER_PATTERN.findall(rendered)
        if leftovers:
            missing = ", ".join(sorted(set(leftovers)))
            raise ValueError(
                f"Unresolved placeholders remain in warmstart template '{template_name}': {missing}"
            )
        return rendered


def format_few_shot_examples(
    examples: Optional[Sequence[Mapping[str, object]]],
) -> str:
    if not examples:
        return ""

    lines = ["Few-shot examples from D_S:"]
    for idx, item in enumerate(examples, start=1):
        theta = item.get("theta")
        f_theta = item.get("f_theta", item.get("objectives"))
        if theta is None or f_theta is None:
            continue
        lines.append(f"Example {idx}: theta={theta}, f(theta)={f_theta}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


class WarmStartPromptContextBuilder:
    """Build placeholder values for the warm-start prompt templates."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        battery_name: Optional[str],
        param_set: str,
        soc_start: float,
        soc_end: float,
        dsoc_sum_max: float = DEFAULT_DSOC_SUM_MAX,
        few_shot_examples: Optional[Sequence[Mapping[str, object]]] = None,
    ):
        self._bounds = param_bounds
        self._meta = resolve_battery_metadata(param_set, battery_name=battery_name)
        self._soc_start = float(soc_start)
        self._soc_end = float(soc_end)
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._few_shot_examples = few_shot_examples

    def build(self, num_recommendation: int) -> Dict[str, str]:
        battery_name = self._meta.battery_name
        param_set_display = self._meta.param_set_display
        task_brief = (
            f"Recommend diverse 3-stage constant-current charging protocols for {battery_name} "
            f"from SOC {self._format_soc(self._soc_start)} to SOC {self._format_soc(self._soc_end)}."
        )
        objective_summary = "\n".join(
            [
                "- Charging time [s]: minimize total time to reach the target SOC.",
                "- Peak temperature rise [K]: minimize thermal stress during charging.",
                "- Aging degree [%]: minimize electrochemical degradation risk.",
            ]
        )
        problem_detail = (
            f"This is a multi-objective warm-start task for Bayesian optimization. "
            f"The simulator uses a 3-stage CC charging protocol parameterized by I1, I2, I3, "
            f"dSOC1, and dSOC2, while dSOC3 is implied by the remaining SOC window. "
            f"The underlying cell is modeled as {battery_name} ({self._meta.chemistry}, "
            f"{self._meta.nominal_capacity_ah:.1f} Ah) using the {param_set_display}."
        )
        expert_knowledge = "\n".join(f"- {line}" for line in self._meta.expert_knowledge)

        return {
            "NUM_RECOMMENDATION": str(int(num_recommendation)),
            "BATTERY_NAME": battery_name,
            "PARAM_SET_DISPLAY": param_set_display,
            "SOC_START": self._format_soc(self._soc_start),
            "SOC_END": self._format_soc(self._soc_end),
            "I1_RANGE": self._format_range("I1", unit="A"),
            "I2_RANGE": self._format_range("I2", unit="A"),
            "I3_RANGE": self._format_range("I3", unit="A"),
            "DSOC1_RANGE": self._format_range("dSOC1"),
            "DSOC2_RANGE": self._format_range("dSOC2"),
            "DSOC_SUM_MAX": f"{self._dsoc_sum_max:.2f}",
            "TASK_BRIEF": task_brief,
            "OBJECTIVE_SUMMARY": objective_summary,
            "PROBLEM_DETAIL": problem_detail,
            "EXPERT_KNOWLEDGE": expert_knowledge,
            "FEW_SHOT_BLOCK": format_few_shot_examples(self._few_shot_examples),
            "OUTPUT_SCHEMA": (
                '[{"I1": value, "I2": value, "I3": value, '
                '"dSOC1": value, "dSOC2": value}, ...]'
            ),
        }

    def _format_range(self, key: str, unit: str = "") -> str:
        lo, hi = self._bounds[key]
        suffix = f" {unit}" if unit else ""
        return f"[{lo:.2f}, {hi:.2f}]{suffix}"

    @staticmethod
    def _format_soc(value: float) -> str:
        return f"{value * 100:.0f}%"


def render_warmstart_prompt(
    level: str,
    context: Mapping[str, str],
    template_dir: Optional[Path] = None,
) -> str:
    if level not in WARMSTART_TEMPLATE_MAP:
        valid = ", ".join(sorted(WARMSTART_TEMPLATE_MAP))
        raise ValueError(f"Unsupported warmstart context level '{level}'. Expected one of: {valid}")

    template_name = WARMSTART_TEMPLATE_MAP[level]
    renderer = WarmStartTemplateRenderer(template_dir=template_dir)
    prompt = renderer.render(template_name, context)
    logger.debug(
        "WarmStart prompt rendered using template '%s' (%d chars)",
        template_name,
        len(prompt),
    )
    return prompt


if __name__ == "__main__":
    sample_bounds = {k: tuple(v) for k, v in CANONICAL_DEFAULT_BOUNDS.items()}
    builder = WarmStartPromptContextBuilder(
        param_bounds=sample_bounds,
        battery_name=None,
        param_set="Chen2020",
        soc_start=0.0,
        soc_end=0.8,
        dsoc_sum_max=DEFAULT_DSOC_SUM_MAX,
    )
    ctx = builder.build(num_recommendation=6)
    for level_name in ("none", "partial", "full"):
        text = render_warmstart_prompt(level_name, ctx)
        assert not PLACEHOLDER_PATTERN.findall(text)
        print(f"[{level_name}] {len(text)} chars")
