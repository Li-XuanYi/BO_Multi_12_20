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
    "experimental": "experimental",
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


@dataclass(frozen=True)
class PromptProtocolExample:
    label: str
    bucket: str
    theta: Tuple[float, float, float, float, float]
    note: str
    objectives: Optional[Tuple[float, float, float]] = None


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

DEFAULT_TRADEOFF_BUCKETS: Tuple[Tuple[str, str], ...] = (
    ("fast_charge", "time-first; use stronger early current while keeping the late stage controlled"),
    ("thermal_safe", "temperature-first; prefer cooler interior points and lower late-stage stress"),
    ("aging_safe", "aging-first; taper meaningfully into the high-SOC region"),
    ("balanced", "balanced trade-off; avoid extreme current or SOC-span choices"),
    ("front_loaded_fast", "aggressive early acceleration followed by a clearly safer tail"),
    ("high_margin_safe", "leave obvious dSOC safety margin and stay well inside the feasible region"),
)

DEFAULT_ANTI_COLLAPSE_RULES: Tuple[str, ...] = (
    "Cover as many distinct trade-off buckets as possible before giving a second protocol to any single bucket.",
    "Do not place multiple candidates on the same edge or corner of the search space.",
    "Avoid returning near-identical protocols that differ only in the 3rd or 4th decimal place.",
    "Prefer interior points unless a bucket explicitly calls for edge-seeking behavior.",
)

DEFAULT_ANTI_PATTERNS: Tuple[str, ...] = (
    "Do not make every protocol uniformly aggressive with both large currents and large dSOC spans.",
    "Do not keep high current deep into the high-SOC region for more than one deliberately aggressive prototype.",
    "Do not collapse the whole set into nearly flat current profiles unless a bucket explicitly calls for conservative behavior.",
    "Do not maximize dSOC1 + dSOC2 for most of the set; reserve near-margin behavior for a small minority of candidates.",
)

DEFAULT_FEW_SHOT_EXAMPLES: Dict[str, Tuple[PromptProtocolExample, ...]] = {
    "Chen2020": (
        PromptProtocolExample(
            label="Balanced interior starter",
            bucket="balanced",
            theta=(3.80, 3.20, 2.10, 0.30, 0.20),
            note="Good center-of-mass anchor with room to explore both faster and safer directions.",
        ),
        PromptProtocolExample(
            label="Front-loaded fast prototype",
            bucket="front_loaded_fast",
            theta=(5.70, 4.20, 2.50, 0.18, 0.20),
            note="Aggressive first two stages, but the final stage is clearly softer to reduce high-SOC stress.",
        ),
        PromptProtocolExample(
            label="Aging-aware taper",
            bucket="aging_safe",
            theta=(3.40, 2.90, 2.00, 0.24, 0.22),
            note="Moderate early current and a safer tail for the high-SOC region.",
        ),
    ),
}

DEFAULT_NEGATIVE_EXAMPLES: Dict[str, Tuple[PromptProtocolExample, ...]] = {
    "Chen2020": (
        PromptProtocolExample(
            label="Edge-hugging aggressive corner",
            bucket="avoid",
            theta=(6.00, 5.00, 3.00, 0.40, 0.29),
            note="Too many decisions are pushed to the edge at once, which risks poor feasibility margin and set collapse.",
        ),
        PromptProtocolExample(
            label="Mid-cluster duplicate pattern",
            bucket="avoid",
            theta=(4.20, 3.60, 2.40, 0.26, 0.22),
            note="Reasonable by itself, but returning many slight variants of this interior point weakens Pareto coverage.",
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


def format_protocol_examples(
    examples: Optional[Sequence[Mapping[str, object] | PromptProtocolExample]],
    *,
    title: str,
) -> str:
    if not examples:
        return ""

    lines = [title]
    for idx, item in enumerate(examples, start=1):
        if isinstance(item, PromptProtocolExample):
            theta = item.theta
            bucket = item.bucket
            label = item.label
            note = item.note
            objectives = item.objectives
        else:
            theta = item.get("theta")
            bucket = str(item.get("bucket", "unspecified"))
            label = str(item.get("label", f"Example {idx}"))
            note = str(item.get("note", item.get("why", "")))
            objectives = item.get("f_theta", item.get("objectives"))

        if theta is None:
            continue

        theta_text = "[" + ", ".join(
            [
                f"{float(theta[0]):.2f}",
                f"{float(theta[1]):.2f}",
                f"{float(theta[2]):.2f}",
                f"{float(theta[3]):.3f}",
                f"{float(theta[4]):.3f}",
            ]
        ) + "]"
        line = f"- {label} [{bucket}]: theta={theta_text}"
        if objectives is not None:
            line += f", f(theta)={objectives}"
        if note:
            line += f", note={note}"
        lines.append(line)
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
        safe_dsoc_sum_max: Optional[float] = None,
        few_shot_examples: Optional[Sequence[Mapping[str, object]]] = None,
    ):
        self._bounds = param_bounds
        self._meta = resolve_battery_metadata(param_set, battery_name=battery_name)
        self._param_set = param_set
        self._soc_start = float(soc_start)
        self._soc_end = float(soc_end)
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._safe_dsoc_sum_max = (
            min(float(safe_dsoc_sum_max), self._dsoc_sum_max)
            if safe_dsoc_sum_max is not None else min(0.65, self._dsoc_sum_max)
        )
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
        tradeoff_buckets = "\n".join(
            f"- {name}: {desc}"
            for name, desc in DEFAULT_TRADEOFF_BUCKETS
        )
        anti_collapse_rules = "\n".join(
            f"- {line}" for line in self._anti_collapse_rules(num_recommendation=num_recommendation)
        )
        anti_patterns = "\n".join(f"- {line}" for line in DEFAULT_ANTI_PATTERNS)
        few_shot_examples = self._few_shot_examples or DEFAULT_FEW_SHOT_EXAMPLES.get(self._param_set, ())
        negative_examples = DEFAULT_NEGATIVE_EXAMPLES.get(self._param_set, ())

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
            "SAFE_DSOC_SUM_MAX": f"{self._safe_dsoc_sum_max:.2f}",
            "TASK_BRIEF": task_brief,
            "OBJECTIVE_SUMMARY": objective_summary,
            "COLLECTION_OBJECTIVE": (
                "Maximize initial Pareto coverage and early feasible hypervolume by spreading the set across "
                "multiple trade-off directions instead of clustering near one solution family."
            ),
            "PROBLEM_DETAIL": problem_detail,
            "EXPERT_KNOWLEDGE": expert_knowledge,
            "SAFE_MARGIN_RISK_TEXT": (
                f"Pushing beyond the {self._safe_dsoc_sum_max:.2f} safety margin increases lithium plating risk, "
                "internal polarization, and thermal stress in the high-SOC region."
            ),
            "TRADEOFF_BUCKETS": tradeoff_buckets,
            "PER_BUCKET_MIN_COUNT": "1",
            "ANTI_COLLAPSE_RULES": anti_collapse_rules,
            "ANTI_PATTERNS": anti_patterns,
            "FEW_SHOT_BLOCK": format_protocol_examples(
                few_shot_examples,
                title="Curated pattern examples (treat as directional anchors, not hard constraints):",
            ),
            "NEGATIVE_EXAMPLE_BLOCK": format_protocol_examples(
                negative_examples,
                title="Patterns to avoid repeating across the warm-start set:",
            ),
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

    @staticmethod
    def _anti_collapse_rules(num_recommendation: int) -> List[str]:
        n_req = max(int(num_recommendation), 1)
        n_buckets = len(DEFAULT_TRADEOFF_BUCKETS)
        if n_req >= n_buckets:
            coverage_rule = (
                "Assign at least 1 protocol to each trade-off bucket before giving any bucket a second protocol."
            )
        else:
            coverage_rule = (
                "Cover as many distinct trade-off buckets as possible; do not let a single bucket dominate the set."
            )
        return [coverage_rule, *DEFAULT_ANTI_COLLAPSE_RULES]


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
        safe_dsoc_sum_max=min(0.65, DEFAULT_DSOC_SUM_MAX),
    )
    ctx = builder.build(num_recommendation=6)
    for level_name in ("none", "partial", "full"):
        text = render_warmstart_prompt(level_name, ctx)
        assert not PLACEHOLDER_PATTERN.findall(text)
        print(f"[{level_name}] {len(text)} chars")
