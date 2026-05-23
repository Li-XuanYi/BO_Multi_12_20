"""
Live test for LLM_Region fix with real API.
Tests 3 iterations to verify parse_fail is resolved.
"""
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from llm.llm_interface import LLMInterface, LLMConfig

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_region_preference_parsing():
    """Test region preference query with live API."""
    print("=" * 70)
    print("Live Test: LLM_Region Preference Parsing")
    print("=" * 70)

    # API Configuration
    api_base = "https://api.chat.csu.edu.cn/v1"
    api_key = "sk-7MaTMMMYCQtdisiY69eeoJF6oadNCJiF6JZz9bDif5Jacxc6"
    model = "deepseek-v3"

    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print()

    # Create LLM interface
    from utils.constants import DEFAULT_BOUNDS

    bounds = {k: tuple(v) for k, v in DEFAULT_BOUNDS.items()}

    config = LLMConfig(
        backend="openai",
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.3,
        n_samples=1,
        timeout=120,
    )

    interface = LLMInterface(
        param_bounds=bounds,
        config=config,
        battery_model="LG INR21700-M50",
        battery_param_set="Chen2020",
        region_preference_max_tokens=4096,
    )

    # Test iterations
    results = []

    for t in range(1, 4):  # Test 3 iterations
        print(f"\n--- Iteration {t}/3 ---")

        # Simulate state (numpy arrays must be converted to lists for JSON serialization)
        state_dict = {
            "iteration": t,
            "max_iterations": 50,
            "w_vec": [0.6, 0.2, 0.2],  # Time-focused
            "f_min": 0.35 - t * 0.02,  # Improving
            "theta_best": [4.5, 3.8, 2.5, 0.25, 0.20],
            "stagnation_count": 0,
            "top_scalar_points": [
                {"theta": [5.0, 4.0, 2.8, 0.20, 0.18], "objectives": [1800, 5.5, 0.02]},
                {"theta": [4.0, 3.5, 2.5, 0.25, 0.20], "objectives": [2100, 4.8, 0.015]},
            ],
        }

        # Query region preference
        pref = interface.query_region_preference(state_dict)

        print(f"  Kind: {pref.kind}")
        print(f"  Confidence: {pref.confidence:.2f}")
        print(f"  Parser Status: {pref.parser_status}")
        print(f"  Reason: {pref.reason or 'N/A'}")

        if pref.kind == "point" and pref.point:
            print(f"  Point: I1={pref.point.get('I1'):.2f}, I2={pref.point.get('I2'):.2f}, ...")
        elif pref.kind == "region" and pref.lb and pref.ub:
            print(f"  Region: lb=[{pref.lb.get('I1'):.2f}, ...], ub=[{pref.ub.get('I1'):.2f}, ...]")

        results.append({
            "iteration": t,
            "kind": pref.kind,
            "status": pref.parser_status,
            "confidence": pref.confidence,
        })

        if pref.parser_status != "ok":
            print(f"  Raw Preview: {pref.raw_text_preview[:100]}...")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Total calls: {len(results)}")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    success_rate = status_counts.get("ok", 0) / len(results) * 100
    print(f"\nSuccess rate: {success_rate:.0f}%")

    if success_rate > 0:
        print("\n[OK] FIX VERIFIED: LLM_Region is now parsing successfully!")
    else:
        print("\n[FAIL] STILL FAILING: All calls resulted in parse_fail")

    return success_rate > 0


if __name__ == "__main__":
    try:
        success = test_region_preference_parsing()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
