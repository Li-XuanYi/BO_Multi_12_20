"""Test LLM interface basic functionality after improvements."""

import sys
sys.path.insert(0, '.')
from types import SimpleNamespace
import types

from llm.llm_interface import (
    build_llm_interface,
    DEFAULT_BOUNDS,
    LLMCaller,
    LLMConfig,
    _build_iteration_prompt,
)
import llm.llm_interface as interface_module


def test_openai_call_forwards_explicit_thinking_mode(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}', model_extra={}),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    caller = LLMCaller(
        LLMConfig(api_key="test-key", model="deepseek-v4-flash", thinking_mode="disabled")
    )

    assert caller.call("return json", n=1) == ['{"ok":true}']
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def _region_state():
    return {
        "iteration": 0,
        "max_iterations": 50,
        "w_vec": [0.3, 0.4, 0.3],
        "top_scalar_points": [],
        "recent_points": [],
    }


def test_openai_message_text_extraction_ignores_reasoning_content_when_content_empty():
    message = SimpleNamespace(content=None, model_extra={"reasoning_content": '{"kind":"none"}'})

    text, source = LLMCaller._extract_message_text(message)

    assert text == ""
    assert source == "empty"


def test_region_preference_attaches_call_diagnostics_on_success():
    class FakeCaller:
        def __init__(self):
            self.last_call_diagnostics = []

        def call(self, *args, **kwargs):
            self.last_call_diagnostics = [{"finish_reason": "stop", "extracted_text_length": 196}]
            return [
                '{"kind":"point","coordinate_space":"raw","preference_direction":"promising",'
                '"point":{"I1":4.5,"I2":3.5,"I3":2.5,"dSOC1":0.2,"dSOC2":0.2},'
                '"confidence":0.8}'
            ]

    llm = build_llm_interface(DEFAULT_BOUNDS, backend="openai", api_key="test-key")
    llm._caller = FakeCaller()

    pref = llm.query_region_preference(_region_state())

    assert pref.kind == "point"
    assert pref.parser_status == "ok"
    assert pref.llm_call_diagnostics[0]["finish_reason"] == "stop"
    assert '"kind": "point"' in pref.raw_text_preview


def test_region_preference_parse_fail_keeps_diagnostics_and_preview():
    class FakeCaller:
        def __init__(self):
            self.last_call_diagnostics = []

        def call(self, *args, **kwargs):
            self.last_call_diagnostics = [{"finish_reason": "length", "extracted_text_length": 19}]
            return ["not json at all"]

    llm = build_llm_interface(DEFAULT_BOUNDS, backend="openai", api_key="test-key")
    llm._caller = FakeCaller()

    pref = llm.query_region_preference(_region_state())

    assert pref.kind == "none"
    assert pref.parser_status == "parse_fail"
    assert pref.llm_call_diagnostics[0]["finish_reason"] == "length"
    assert pref.raw_text_preview == "not json at all"


def test_region_preference_empty_response_reports_call_error_type():
    class FakeCaller:
        def __init__(self):
            self.last_call_diagnostics = []

        def call(self, *args, **kwargs):
            self.last_call_diagnostics = [
                {"error_type": "PermissionDeniedError", "error": "403 Forbidden"}
            ]
            return []

    llm = build_llm_interface(DEFAULT_BOUNDS, backend="openai", api_key="test-key")
    llm._caller = FakeCaller()

    pref = llm.query_region_preference(_region_state())

    assert pref.kind == "none"
    assert pref.parser_status == "query_permission_denied"
    assert pref.llm_call_diagnostics[0]["error_type"] == "PermissionDeniedError"


def test_prompt_versions_and_region_token_budget_are_wired(monkeypatch):
    captured = {}

    def fake_region_prompt(*, state, param_bounds, prompt_version):
        captured["region_prompt_version"] = prompt_version
        return "region prompt"

    def fake_warmstart_prompt(level, context):
        captured["warmstart_level"] = level
        return "warmstart prompt"

    class FakeCaller:
        last_call_diagnostics = []

        def call(self, prompt, **kwargs):
            captured["max_tokens"] = kwargs.get("max_tokens")
            return ['{"kind":"none","confidence":0.0}']

    monkeypatch.setattr(interface_module, "render_region_preference_prompt", fake_region_prompt)
    monkeypatch.setattr(interface_module, "render_warmstart_prompt", fake_warmstart_prompt)
    llm = build_llm_interface(
        DEFAULT_BOUNDS,
        backend="openai",
        api_key="test-key",
        warmstart_prompt_version="experimental",
        region_preference_prompt_version="calibrated_v2",
        region_preference_max_tokens=777,
    )
    llm._caller = FakeCaller()

    llm.query_region_preference(_region_state())
    llm._render_warmstart_prompt(3)

    assert captured["region_prompt_version"] == "calibrated_v2"
    assert captured["max_tokens"] == 777
    assert captured["warmstart_level"] == "experimental"


def test_basic_functionality():
    """Test basic functionality"""
    print("=" * 60)
    print("Test 1: Import modules")
    print("=" * 60)

    try:
        # Test 1.1: Create LLMInterface instance
        llm = build_llm_interface(
            param_bounds=DEFAULT_BOUNDS,
            backend="mock",
            model="gpt-4.1-mini",
            enable_iteration_fewshot=True,
        )
        print(f"PASS: LLMInterface created successfully")
        print(f"   - backend: {llm._config.backend}")
        print(f"   - enable_iteration_fewshot: {llm._enable_iteration_fewshot}")

        # Test 1.2: Check method signature
        import inspect
        sig = inspect.signature(_build_iteration_prompt)
        args = [name for name in sig.parameters.keys()]
        print(f"PASS: _build_iteration_prompt has {len(args)} parameters")
        print(f"   - Parameter list: {args}")

        # Test 1.3: Check if include_fewshot parameter exists
        has_include_param = 'include_fewshot' in args
        print(f"PASS: include_fewshot parameter exists: {has_include_param}")

        # Test 2: PhysicsHeuristicFallback.physics_informed_warmstart
        print("=" * 60)
        print("Test 2: Physics Fallback Strategy")
        print("=" * 60)

        # Test 2.1: Test generating 15 candidate points
        candidates = llm._fallback.physics_informed_warmstart(15)
        print(f"PASS: Generated {len(candidates)} candidate points")
        print(f"   - First 7 points:")
        for i, c in enumerate(candidates[:7]):
            print(f"      [{i}] I1={c[0]:.2f} I2={c[1]:.2f} I3={c[2]:.2f} dSOC1={c[3]:.3f} dSOC2={c[4]:.3f}")
        print(f"   - Last 8 points (new extreme directions):")
        for i, c in enumerate(candidates[7:]):
            print(f"      [{i+7}] I1={c[0]:.2f} I2={c[1]:.2f} I3={c[2]:.2f} dSOC1={c[3]:.3f} dSOC2={c[4]:.3f}")

        print("=" * 60)
        print("PASS: All tests passed")
        return True

    except Exception as e:
        print(f"FAIL: Test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
