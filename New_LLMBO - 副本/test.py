#!/usr/bin/env python3
"""
快速补丁脚本: 修复 llm_interface.py 中的 api_base /chat 后缀问题

运行方式: python3 patch_llm_interface.py
"""
import re
from pathlib import Path

target = Path("llm_interface.py")
if not target.exists():
    print("ERROR: llm_interface.py not found in current directory")
    exit(1)

content = target.read_text(encoding="utf-8")

# Fix 1: LLMConfig default api_base
# Fix 2: build_llm_interface default api_base
OLD = "https://api.nuwaapi.com/v1/chat"
NEW = "https://api.nuwaapi.com/v1"

n_replaced = content.count(OLD)
if n_replaced == 0:
    print("INFO: api_base already correct (no /chat suffix found)")
else:
    content = content.replace(OLD, NEW)
    target.write_text(content, encoding="utf-8")
    print(f"✓ Fixed {n_replaced} occurrence(s) of api_base: removed /chat suffix")
    print(f"  Before: {OLD}")
    print(f"  After:  {NEW}")