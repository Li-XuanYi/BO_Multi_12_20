from llmbo.utils.transforms import DataTransformer
import numpy as np

transformer = DataTransformer(enable_log_aging=True, verbose=True)

# 创建有变化的数据
fake_db = [
    {'aging': 0.0002, 'time': 50, 'temp': 305, 'valid': True},
    {'aging': 0.0003, 'time': 45, 'temp': 308, 'valid': True},
    {'aging': 0.0004, 'time': 40, 'temp': 310, 'valid': True},
]

transformed = transformer.fit_transform_database(fake_db)

print("\n检查变换后的aging:")
for i, r in enumerate(transformed):
    print(f"  样本{i+1}: {r['aging_raw']:.6f} → {r['aging']:.2f} (log10)")
# ```

# **预期结果**：
# ```
# 样本1: 0.000200 → -3.70 (log10)  ✓
# 样本2: 0.000300 → -3.52 (log10)  ✓
# 样本3: 0.000400 → -3.40 (log10)  ✓
# ```

# **如果实际结果是**：
# ```
# 样本1: 0.000200 → -6.00 (log10)  ✗
# 样本2: 0.000300 → -6.00 (log10)  ✗
# 样本3: 0.000400 → -6.00 (log10)  ✗