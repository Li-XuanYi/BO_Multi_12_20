import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 模拟数据生成 (保持趋势一致) ---
def generate_curve(soh, n=600):
    t = np.linspace(500, 5500, n)
    temp = 300 + 25 * np.exp(-t/2000) + np.random.normal(0, 0.4, n)
    loss = (55 / (t/550)) * soh + np.random.normal(0, 0.8, n)
    return t, temp, loss

fig = plt.figure(figsize=(12, 9), dpi=120)
ax = fig.add_subplot(111, projection='3d')

# --- 2. 还原原始色系 (十六进制精确匹配原图) ---
# 分别对应：淡紫色、淡青色、淡橙红色、嫩绿色
original_colors = ['#9D91DA', '#A5E9E1', '#F4A792', '#C6F1A8']
soh_levels = [1.0, 0.9, 0.8, 0.7]

# --- 3. 绘制散点：极淡阴影处理 ---
for i, soh in enumerate(soh_levels):
    x, y, z = generate_curve(soh)
    
    # 使用 edgecolor 控制外圈，使用 facecolor + 极低 alpha 控制内部阴影
    ax.scatter(x, y, z, 
               s=25, 
               edgecolors=original_colors[i], 
               facecolors=original_colors[i], 
               alpha=0.3,          # 整体（包含边框）的透明度
               linewidths=0.7, 
               depthshade=False,   # 关闭自带的3D深度阴影，防止颜色变脏
               label=f'SOH={soh}')
    
    # 核心技巧：为了让内部阴影“再淡一些”，我们覆盖一层更淡的填充
    # Matplotlib scatter 无法直接对 edge 和 face 设不同的透明度，
    # 这里的 0.3 综合效果已经非常接近原图。

# --- 4. 进一步淡化网格线 ---
# 将网格线透明度降到 0.2，颜色设为极浅灰
grid_params = {"color": (0.85, 0.85, 0.85, 0.2), "linestyle": "-"}
ax.xaxis._axinfo["grid"].update(grid_params)
ax.yaxis._axinfo["grid"].update(grid_params)
ax.zaxis._axinfo["grid"].update(grid_params)

# 彻底透明化背景面板
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# --- 5. 图例强化 (解决颜色太淡辨识难) ---
lgnd = ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=11)
for handle in lgnd.legend_handles:
    handle.set_alpha(1.0)      # 图例图标完全不透明，确保色系清晰
    handle.set_linewidth(1.5)  # 稍微加粗图例圈

# --- 6. 关键点 A-F 标注 ---
# 坐标根据模拟数据微调
stars = {'A': (600, 326, 58), 'B': (1600, 312, 16), 'C': (5500, 308, 12), 
         'D': (5600, 303, 6), 'E': (4200, 305, 10), 'F': (4800, 302, 5)}
for label, pos in stars.items():
    ax.scatter(pos[0], pos[1], pos[2], marker='*', c='red', s=180, zorder=100)
    ax.text(pos[0]-150, pos[1], pos[2]+2, label, fontsize=12, fontweight='bold', zorder=101)

# --- 7. 视角与标签 ---
ax.view_init(elev=22, azim=230)
ax.set_xlabel('Charging Time/s', labelpad=12, fontsize=11)
ax.set_ylabel('Temperature/K', labelpad=12, fontsize=11)
ax.set_zlabel('Lithium-ion loss/C', labelpad=10, fontsize=11)

# 刻度微调
ax.set_zlim(0, 60)
ax.set_ylim(300, 330)

plt.tight_layout()
plt.show()
