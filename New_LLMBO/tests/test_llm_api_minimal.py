#!/usr/bin/env python
"""Minimal LLM API test — no project imports, just the API.

Run:
    pixi run python tests/test_llm_api_minimal.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

import os
from openai import OpenAI

PASS = 0
FAIL = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def extract_text(raw: str) -> str:
    """Strip <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()


def main() -> int:
    global PASS, FAIL

    print("=" * 60)
    print("test_llm_api_minimal: Direct API connectivity test")
    print("=" * 60)

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL", "https://api.minimax.chat/v1")
    model = os.environ.get("LLM_MODEL", "MiniMax-M2.7")

    print(f"\n  api_key: {api_key[:6]}... ({len(api_key)} chars)")
    print(f"  api_base: {api_base}")
    print(f"  model: {model}")

    check("api_key not empty", len(api_key) > 0)
    check("api_base not empty", len(api_base) > 0)
    check("model not empty", len(model) > 0)

    if not api_key:
        print("\n  SKIP: No API key. Fill in .env and run again.")
        return 1

    client = OpenAI(api_key=api_key, base_url=api_base)

    # ── Test 1: Simple chat ───────────────────────────────────────────
    print("\n[1] Simple chat completion")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
            max_tokens=2000,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        raw = resp.choices[0].message.content or ""
        text = extract_text(raw)
        check(f"response in <15s ({elapsed:.1f}s)", elapsed < 15)
        check(f"response non-empty after strip", len(text) > 0, f"raw: {raw!r}")
        check(f"response is 'OK'", text.upper() == "OK", f"got: {text!r}")
        print(f"  Raw: {raw!r}")
        print(f"  Stripped: {text!r}  ({elapsed:.1f}s)")
    except Exception as exc:
        FAIL += 1
        print(f"  [FAIL] API call failed: {exc}")

    # ── Test 2: Math ───────────────────────────────────────────────────
    print("\n[2] Math question")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer with just the number."},
            ],
            max_tokens=1000,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        raw = resp.choices[0].message.content or ""
        text = extract_text(raw)
        check(f"response in <15s ({elapsed:.1f}s)", elapsed < 15)
        check(f"response is '4'", text == "4", f"got: {text!r}")
        print(f"  Raw: {raw!r}")
        print(f"  Stripped: {text!r}  ({elapsed:.1f}s)")
    except Exception as exc:
        FAIL += 1
        print(f"  [FAIL] API call failed: {exc}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL == 0:
        print("\nAPI is working correctly!")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
