from utils.model_labels import canonical_model_label


def test_canonical_model_label_maps_deepseek_flash() -> None:
    assert canonical_model_label("deepseek-v4-flash") == "Deepseek-V4"


def test_canonical_model_label_maps_gpt41_nano() -> None:
    assert canonical_model_label("gpt-4.1-nano") == "GPT-4.1-nano"


def test_canonical_model_label_leaves_unknown_models_unchanged() -> None:
    assert canonical_model_label("custom-model-x") == "custom-model-x"
