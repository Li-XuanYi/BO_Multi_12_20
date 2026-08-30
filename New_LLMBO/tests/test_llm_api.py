#!/usr/bin/env python
"""Test LLM API connectivity.

Setup:
  1. Copy .env.example to .env and fill in your API key and base URL
  2. Run: python tests/test_llm_api.py

What it tests:
  - .env file loading works
  - API key and base URL are correctly read
  - OpenAI-compatible API responds to a simple chat completion
  - Touchpoint 1b warmstart candidate generation works
  - Touchpoint 2 iteration guidance works
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def _check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def main() -> int:
    global PASS, FAIL

    print("=" * 60)
    print("test_llm_api: LLM API connectivity test")
    print("=" * 60)

    # ── 1. .env loading ───────────────────────────────────────────────
    print("\n[1] .env file loading")
    env_path = ROOT / ".env"
    env_example_path = ROOT / ".env.example"

    _check(".env.example exists", env_example_path.exists())
    if env_path.exists():
        print(f"  NOTE: .env found at {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key = line.split("=", 1)[0]
                    if key in ("LLM_API_KEY", "OPENAI_API_KEY"):
                        val = line.split("=", 1)[1]
                        _check(f"{key} is set", len(val) > 0, f"value length: {len(val)}")
                        if val == "your_api_key_here" or not val:
                            print(f"  WARN: {key} has placeholder value!")
    else:
        print(f"  INFO: .env not found at {env_path}")
        print(f"  Copy {env_example_path} to {env_path} and fill in your credentials")

    # ── 2. Helper function values ────────────────────────────────────
    print("\n[2] Environment variable / helper values")
    from llm.llm_interface import (
        _get_default_llm_api_key,
        _get_default_llm_api_base,
        _get_default_llm_model,
    )

    api_key = _get_default_llm_api_key()
    api_base = _get_default_llm_api_base()
    model = _get_default_llm_model()

    _check(f"api_key not empty", len(api_key) > 0, f"got {len(api_key)} chars")
    _check(f"api_base not empty", len(api_base) > 0, f"got: {api_base}")
    _check(f"model not empty", len(model) > 0, f"got: {model}")

    # ── 3. OpenAI client instantiation ────────────────────────────────
    print("\n[3] OpenAI-compatible client")
    if not api_key:
        print("  SKIP: No API key, cannot test client")
    else:
        from openai import OpenAI
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            _check("OpenAI client created", True)

            # ── 4. Simple chat completion ──────────────────────────────
            print("\n[4] Simple chat completion")
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
                max_tokens=10,
                temperature=0.0,
            )
            elapsed = time.time() - t0

            text = resp.choices[0].message.content.strip()
            _check(f"response in <10s ({elapsed:.1f}s)", elapsed < 10)
            _check(f"response non-empty", len(text) > 0, f"got: {text!r}")
            _check(f"response 'OK'", text.upper() == "OK", f"got: {text!r}")
            print(f"  Response: {text!r}  ({elapsed:.1f}s)")
        except Exception as exc:
            FAIL += 1
            print(f"  [FAIL] API call failed: {exc}")
            return 1

    # ── 5. Touchpoint 1b warmstart candidates ────────────────────────
    print("\n[5] Touchpoint 1b: warmstart candidate generation")
    if not api_key:
        print("  SKIP: No API key")
    else:
        from llm.llm_interface import build_llm_interface, DEFAULT_BOUNDS
        from utils.constants import DSOC_SUM_MAX

        try:
            llm = build_llm_interface(
                DEFAULT_BOUNDS,
                backend="openai",
                model=model,
                api_base=api_base,
                api_key=api_key,
                warmstart_context_level="full",
                warmstart_max_tokens=2500,
                warmstart_max_retries=2,
                soc_start=0.0,
                soc_end=0.8,
                dsoc_sum_max=DSOC_SUM_MAX,
            )
            _check("LLMInterface built", True)

            t0 = time.time()
            candidates = llm.generate_warmstart_candidates(n=3)
            elapsed = time.time() - t0

            _check(f"got 3 candidates ({elapsed:.1f}s)", len(candidates) == 3, f"got {len(candidates)}")
            lo = [b[0] for b in DEFAULT_BOUNDS.values()]
            hi = [b[1] for b in DEFAULT_BOUNDS.values()]
            for i, c in enumerate(candidates):
                in_bounds = all(lo[j] <= c[j] <= hi[j] for j in range(5))
                dsoc_ok = c[3] + c[4] <= DSOC_SUM_MAX + 1e-6
                _check(f"candidate[{i}] in bounds", in_bounds, f"{c}")
                _check(f"candidate[{i}] dSOC valid", dsoc_ok, f"dSOC1+dSOC2={c[3]+c[4]:.3f}")
            print(f"  Candidates:\n" + "\n".join(f"    {c.round(3)}" for c in candidates))
        except Exception as exc:
            FAIL += 1
            print(f"  [FAIL] Touchpoint 1b failed: {exc}")

    # ── 6. Touchpoint 2 iteration guidance ───────────────────────────
    print("\n[6] Touchpoint 2: iteration guidance")
    if not api_key:
        print("  SKIP: No API key")
    else:
        import numpy as np
        from llm.llm_interface import build_llm_interface, DEFAULT_BOUNDS
        from utils.constants import DSOC_SUM_MAX

        try:
            llm = build_llm_interface(
                DEFAULT_BOUNDS,
                backend="openai",
                model=model,
                api_base=api_base,
                api_key=api_key,
                warmstart_context_level="full",
                warmstart_max_tokens=2500,
                warmstart_max_retries=2,
                soc_start=0.0,
                soc_end=0.8,
                dsoc_sum_max=DSOC_SUM_MAX,
            )
            state = {
                "iteration": 5,
                "max_iterations": 20,
                "theta_best": np.array([4.0, 3.5, 2.5, 0.3, 0.25]),
                "f_min": 0.15,
                "w_vec": np.array([0.4, 0.3, 0.3]),
                "database": None,
                "stagnation_count": 0,
            }
            t0 = time.time()
            result = llm.generate_iteration_candidates(n=3, state_dict=state)
            elapsed = time.time() - t0

            _check(f"got result ({elapsed:.1f}s)", result is not None)
            print(f"  Result type: {type(result).__name__}")
            print(f"  Elapsed: {elapsed:.1f}s")
        except Exception as exc:
            FAIL += 1
            print(f"  [FAIL] Touchpoint 2 failed: {exc}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL == 0 and api_key:
        print("\nAll API tests passed! LLM integration is working.")
    elif FAIL > 0:
        print("\nSome tests failed. Check the errors above.")
    else:
        print("\nNo API key set. Fill in .env and run again.")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
