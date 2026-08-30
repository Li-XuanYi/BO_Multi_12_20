"""
Quick 10-iteration ablation test to verify LLM_Region fix.
Tests seeds 8409-8410 with 4 configurations.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["LLM_BACKEND"] = "openai"
os.environ["LLM_MODEL"] = "deepseek-v3"
os.environ["LLM_API_BASE"] = "https://api.chat.csu.edu.cn/v1"
os.environ["LLM_API_KEY"] = "sk-7MaTMMMYCQtdisiY69eeoJF6oadNCJiF6JZz9bDif5Jacxc6"

from Ablation_Exp.Process.tools.run_ablation_8409_8413_exp_prompt import main, VariantSpec

def run_quick_test():
    """Run 10-iteration test with 2 seeds."""
    import argparse

    # Override arguments (API key via environment)
    sys.argv = [
        "run_ablation_8409_8413_exp_prompt.py",
        "--iterations", "10",
        "--seeds", "8409", "8410",
        "--api-base", "https://api.chat.csu.edu.cn/v1",
        "--model", "deepseek-v3",
        "--output-root", "Ablation_Exp/experiment_records/quick_10iter_test",
    ]

    return main()

if __name__ == "__main__":
    print("=" * 70)
    print("Quick 10-iteration Test for LLM_Region Fix")
    print("=" * 70)
    print("Seeds: 8409, 8410")
    print("Iterations: 10")
    print("Model: deepseek-v3")
    print("=" * 70)

    import traceback
    try:
        run_quick_test()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
