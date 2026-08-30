"""
Test script for LLM_Region parsing improvements.

This script tests the improved parsing logic for Region-Lifted GP
to verify that parse_fail issues are resolved.
"""

import json
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def test_coerce_param_dict():
    """Test improved _coerce_param_dict with partial dictionaries."""
    from llmbo.region_lifted_gp import _coerce_param_dict

    print("=" * 60)
    print("Test 1: _coerce_param_dict improvements")
    print("=" * 60)

    tests = [
        # (input, expected_success, description)
        ({"I1": 5.0, "I2": 4.0, "I3": 3.0}, True, "Partial dict (3/5 keys)"),
        ({"I1": 5.0, "I2": 4.0}, False, "Too partial (2/5 keys)"),
        ({"I1": "5.5", "I2": "4.5", "I3": "3.0", "dSOC1": "0.30", "dSOC2": "0.25"}, True, "String numbers"),
        ({"I1": 5.0, "I2": 4.0, "I3": 3.0, "dSOC1": 0.30, "dSOC2": 0.25}, True, "Complete dict"),
        ([5.0, 4.0, 3.0, 0.30, 0.25], True, "Array format"),
        ({"I1": None, "I2": 4.0, "I3": 3.0}, False, "None value"),
        ({}, False, "Empty dict"),
        (None, False, "None input"),
    ]

    passed = 0
    for input_val, expected, desc in tests:
        result = _coerce_param_dict(input_val)
        success = (result is not None) == expected
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {desc}: got {result is not None}, expected {expected}")
        if success:
            passed += 1

    print(f"\nPassed {passed}/{len(tests)} tests")
    return passed == len(tests)


def test_parse_region_preference_payload():
    """Test parse_region_preference_payload with various inputs."""
    from llmbo.region_lifted_gp import parse_region_preference_payload

    print("\n" + "=" * 60)
    print("Test 2: parse_region_preference_payload")
    print("=" * 60)

    tests = [
        # Valid point
        ({"kind": "point",
          "point": {"I1": 4.0, "I2": 3.5, "I3": 2.5, "dSOC1": 0.25, "dSOC2": 0.20},
          "confidence": 0.7},
         ("point", "ok"), "Valid point"),

        # Valid region
        ({"kind": "region",
          "lb": {"I1": 3.8, "I2": 3.0, "I3": 2.2, "dSOC1": 0.20, "dSOC2": 0.18},
          "ub": {"I1": 4.5, "I2": 4.0, "I3": 2.8, "dSOC1": 0.30, "dSOC2": 0.25},
          "confidence": 0.6},
         ("region", "ok"), "Valid region"),

        # Partial point (should be filled)
        ({"kind": "point",
          "point": {"I1": 5.0, "I2": 4.0, "I3": 3.0},
          "confidence": 0.7},
         ("point", "ok"), "Partial point (auto-fill)"),

        # Kind aliases
        ({"kind": "box",
          "lb": {"I1": 3.8, "I2": 3.0, "I3": 2.2, "dSOC1": 0.20, "dSOC2": 0.18},
          "ub": {"I1": 4.5, "I2": 4.0, "I3": 2.8, "dSOC1": 0.30, "dSOC2": 0.25}},
         ("region", "ok"), "Kind alias 'box'"),

        # None preference
        ({"kind": "none", "confidence": 0.0},
         ("none", "ok"), "Explicit none"),

        # Invalid kind
        ({"kind": "invalid", "confidence": 0.5},
         ("none", "invalid_kind"), "Invalid kind"),

        # Region missing bounds
        ({"kind": "region", "confidence": 0.5},
         ("none", "invalid_region_bounds"), "Region missing bounds"),
    ]

    passed = 0
    for payload, expected, desc in tests:
        result = parse_region_preference_payload(payload, log_level=logging.DEBUG)
        kind_ok = result.kind == expected[0]
        status_ok = result.parser_status == expected[1]
        success = kind_ok and status_ok
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {desc}: kind={result.kind}, status={result.parser_status}")
        if success:
            passed += 1

    print(f"\nPassed {passed}/{len(tests)} tests")
    return passed == len(tests)


def test_flexible_json_extraction():
    """Test flexible JSON extraction from various formats."""
    print("\n" + "=" * 60)
    print("Test 3: Flexible JSON extraction")
    print("=" * 60)

    from llm.llm_interface import LLMInterface

    interface = LLMInterface.__new__(LLMInterface)

    test_cases = [
        # Standard JSON
        ('{"kind": "point", "confidence": 0.8}', True, "Standard JSON"),

        # Markdown code block
        ('```json\n{"kind": "point", "confidence": 0.8}\n```', True, "Markdown json block"),

        # Inline code
        ('`{"kind": "point"}`', True, "Inline code"),

        # With extra text
        ('Here is the result: {"kind": "none", "confidence": 0}', True, "With extra text"),

        # Trailing comma (common LLM error)
        ('{"kind": "point", "confidence": 0.8,}', True, "Trailing comma"),

        # Single quotes (common LLM error)
        ("{'kind': 'point', 'confidence': 0.8}", True, "Single quotes"),

        # Empty
        ('', False, "Empty string"),

        # Invalid
        ('not json', False, "Invalid JSON"),
    ]

    passed = 0
    for text, expected, desc in test_cases:
        result = interface._extract_json_flexible(text)
        success = (result is not None) == expected
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {desc}: extracted={result is not None}")
        if success:
            passed += 1

    print(f"\nPassed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_end_to_end():
    """End-to-end test simulating actual LLM responses."""
    print("\n" + "=" * 60)
    print("Test 4: End-to-end parsing")
    print("=" * 60)

    from llmbo.region_lifted_gp import parse_region_preference_payload
    from llm.llm_interface import LLMInterface

    interface = LLMInterface.__new__(LLMInterface)

    # Simulate various LLM response formats
    simulated_responses = [
        # Format 1: Clean JSON
        '{"kind": "point", "point": {"I1": 4.5, "I2": 3.8, "I3": 2.5, "dSOC1": 0.25, "dSOC2": 0.20}, "confidence": 0.75, "reason": "test"}',

        # Format 2: Markdown wrapped
        '```json\n{"kind": "region", "lb": {"I1": 3.5, "I2": 3.0, "I3": 2.0, "dSOC1": 0.20, "dSOC2": 0.15}, "ub": {"I1": 5.0, "I2": 4.5, "I3": 3.0, "dSOC1": 0.35, "dSOC2": 0.28}, "confidence": 0.6}\n```',

        # Format 3: With extra text
        'Based on the data, I suggest this region: {"kind": "point", "point": {"I1": 5.0, "I2": 4.0, "I3": 2.8, "dSOC1": 0.22, "dSOC2": 0.18}, "confidence": 0.8}',

        # Format 4: Trailing comma
        '{"kind": "none", "confidence": 0.0, "reason": "insufficient data",}',
    ]

    passed = 0
    for i, response in enumerate(simulated_responses, 1):
        # Extract JSON
        parsed = interface._extract_json_flexible(response)
        if parsed is None:
            print(f"  [FAIL] Response {i}: JSON extraction failed")
            continue

        # Parse preference
        result = parse_region_preference_payload(parsed)

        if result.parser_status == "ok":
            print(f"  [PASS] Response {i}: kind={result.kind}, confidence={result.confidence:.2f}")
            passed += 1
        else:
            print(f"  [FAIL] Response {i}: status={result.parser_status}")

    print(f"\nPassed {passed}/{len(simulated_responses)} tests")
    return passed == len(simulated_responses)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLM_Region Parsing Fix - Test Suite")
    print("=" * 60)

    results = []
    results.append(("_coerce_param_dict", test_coerce_param_dict()))
    results.append(("parse_region_preference_payload", test_parse_region_preference_payload()))
    results.append(("flexible_json_extraction", test_flexible_json_extraction()))
    results.append(("end_to_end", test_end_to_end()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total_passed = sum(passed for _, passed in results)
    print(f"\nTotal: {total_passed}/{len(results)} test suites passed")

    return 0 if total_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
