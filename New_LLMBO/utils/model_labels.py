from __future__ import annotations


MODEL_LABEL_ALIASES = {
    "deepseek-v4-flash": "Deepseek-V4",
    "deepseek-v4-pro": "Deepseek-V4-Pro",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gpt-5.4": "GPT-5.4",
}


def canonical_model_label(model_name: str | None) -> str:
    if not model_name:
        return ""
    return MODEL_LABEL_ALIASES.get(str(model_name), str(model_name))
