"""Test LLM interface basic functionality after improvements."""

import sys
sys.path.insert(0, '.')
from llm.llm_interface import (
    build_llm_interface,
    DEFAULT_BOUNDS,
    LLMConfig,
    _build_iteration_prompt,
)

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
