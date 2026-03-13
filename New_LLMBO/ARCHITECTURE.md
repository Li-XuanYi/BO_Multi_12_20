# New_LLMBO 体系架构说明 (Architecture Documentation)

本项目是一个结合了大型语言模型（LLM）与贝叶斯优化（BO）的多目标优化系统。为方便管理与扩展，整体代码与数据已按模块功能进行解耦和重组。

## 目录结构 (Directory Structure)

```text
New_LLMBO/
│
├── llmbo/                  # 贝叶斯优化核心模块 (Bayesian Optimization Core)
│   ├── gp_model.py         # 高斯过程替代模型 (Gaussian Process Surrogate)
│   ├── acquisition.py      # 采集函数 (如 Expected Improvement - EI 等)
│   ├── optimizer.py        # BO 优化器逻辑封装
│   └── ParEGO.py           # ParEGO 多目标优化算法实现
│
├── llm/                    # 语言模型交互模块 (LLM Interface & Prompts)
│   ├── llm_interface.py    # 与大语言模型(Claude/GPT等)通信的核心接口
│   ├── templates/          # 给 LLM 提供的提示词模板 (Prompt Templates)
│   └── extract_claude_conversation.py # 处理、解析 LLM 会话及输出结果的辅助工具
│
├── plot/                   # 可视化与制图模块 (Plotting & Visualization)
│   ├── plot_comparison.py  # 绘制对比实验图像
│   ├── plot_hv.py          # 绘制超体积(Hypervolume)收敛曲线
│   ├── plot_optimal_count.py # 绘制寻优数量变化曲线
│   ├── plot_pareto3d.py    # 绘制 3D 帕累托前沿散点图
│   └── plot_protocol.py    # 绘制电池充电协议等结果展示图
│
├── fig/                    # 生成图像存放目录 (Generated Figures)
│   └── *.png               # 脚本(plot)运行后在此处输出各类图像文件
│
├── DataBase/               # 数据存储与读写模块 (Data & State Management)
│   ├── database.py         # 数据库操作接口逻辑
│   ├── export_to_xlsx.py   # 将实验结果导出为 Excel 文件的封装
│   ├── xlsx_io.py          # Excel 文件读写与解析层
│   └── *.xlsx              # 实验状态、帕累托前沿结果等 Excel 落地文件
│
├── checkpoints/            # 训练与优化检查点 (Optimization Checkpoints)
│   └── *.json              # 记录每次 BO 迭代数据的 JSON 文件 (不作移动)
│
├── exp/                    # 实验运行与对比模块 (Experiments & Baselines)
│   ├── unified_runner.py   # 统一实验启动器
│   ├── parego_runner.py    # ParEGO 基准实验运行器
│   ├── README_Experiments.md # 实验参数与具体说明
│   └── results_parego_300/ # 常规基准实验输出目录
│
├── pybamm_simulator.py     # PyBaMM 电池物理仿真模拟器 (Simulator)
├── main.py                 # 主程序的入口
└── test.py                 # 单元测试与测试脚手架
```

## 体系运行流程 (System Workflow)

1. **环境与仿真**: `main.py` 启动优化流程，系统通过 `pybamm_simulator.py` 评估电池的实际物理表现并返回多维反馈信号（如循环寿命损耗，有效容量等）。
2. **状态记录**: 交互结果统一序列化并存储在 `DataBase` 目录下(Excel格式)或保存在 `checkpoints` 中(JSON)。
3. **贝叶斯更新 (`llmbo` 模块)**: `gp_model.py` 拟合数据，结合 `acquisition.py` 给出下一个有利探索点的数学分布建议。
4. **语言模型推理 (`llm` 模块)**: 搜集当前帕累托前沿和 BO 给出的分析结果，结合 `templates` 传导给 `llm_interface.py`，LLM据此理解优化动向并推荐下一个测试参数(Proposal)。
5. **结果评估 (`plot` 模块)**: 实验结束后，通过 `exp` 里的脚本汇总各个 Baseline （如 ParEGO），使用 `plot` 下的各绘图脚本分析收敛性与超体积，图片产出归入 `fig` 中供论文或报告使用。