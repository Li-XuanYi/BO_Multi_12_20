"""
plot_optimal_count.py — 最优充电协议数量收敛曲线
=================================================
输入: Excel 文件，格式与 plot_hv.py 类似
输出: 最优协议数量 vs 评估次数曲线图 (PNG/PDF)

Excel 格式说明
--------------
与 plot_hv.py 完全相同:
  - 多 Sheet → 多子图
  - 第 1 列: "Evaluation"
  - 后续列: "<方法名>_run<N>" 格式，值为累计最优协议数量

用法:
  python plot_optimal_count.py input.xlsx [output.png]
"""

import sys
from plot_hv import plot_hv_convergence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_optimal_count.py <input.xlsx> [output.png]")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "optimal_count.png"

    plot_hv_convergence(
        in_path, out_path,
        ylabel="Number of optimal charging protocols",
        ylim=None,  # 自动范围
    )
