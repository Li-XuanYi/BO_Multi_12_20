e7a9ed2 1.箱型图（Ecker2015）以及Patero图已经填入真实数据 2.准备消融实验 3.在ORegan2022上进行进一步对比实验
597d2a7 新增超参搜索 llm的api配置将从.env读取
633559f 	modified:   New_LLMBO/.claude/scheduled_tasks.json
976d3ef 	new file:   192178cb2531720c24b3fb6dd2a3613d_97d1a25f628840c99192fb58ea9cae39_8/PlatEMO 	deleted:    New_LLMBO/figs/hv_convergence_3way.pdf 	deleted:    New_LLMBO/figs/hv_convergence_3way.png 	deleted:    New_LLMBO/figs/optimal_protocols_3way.pdf 	deleted:    New_LLMBO/figs/optimal_protocols_3way.png 	deleted:    New_LLMBO/figs/pareto_front_2d_3way.pdf 	deleted:    New_LLMBO/figs/pareto_front_2d_3way.png 	deleted:    New_LLMBO/figs/pareto_front_3d_3way.pdf 	deleted:    New_LLMBO/figs/pareto_front_3d_3way.png 	deleted:    New_LLMBO/figs/seed8409_comparison/hv_convergence_3way.pdf 	deleted:    New_LLMBO/figs/seed8409_comparison/hv_convergence_3way.png 	deleted:    New_LLMBO/figs/seed8409_comparison/optimal_protocols_3way.pdf 	deleted:    New_LLMBO/figs/seed8409_comparison/optimal_protocols_3way.png 	deleted:    New_LLMBO/figs/seed8409_comparison/pareto_front_2d_3way.pdf 	deleted:    New_LLMBO/figs/seed8409_comparison/pareto_front_2d_3way.png 	deleted:    New_LLMBO/figs/seed8409_comparison/pareto_front_3d_3way.pdf 	deleted:    New_LLMBO/figs/seed8409_comparison/pareto_front_3d_3way.png 	deleted:    New_LLMBO/test_api.py
874cde5 1.新添DISK PIMD新型算法(PlatEMO&Compar_Exp/) 2.针对决策变量{t_c T_p Q_a}的预处理做了对比实验(scalatization_Exp/) 3.Patero图重新绘制(Patero/) 4.创建箱型图(Box_Fig/)
db6e2ba 修了 main.py 配置传递断裂和死代码问题，新建 config/presets.py 消除循环依赖，测试 43 pass / 0 fail。下一步建议在有 sklearn 的环境跑 Part 3 集成测试验证 BayesOptimizer 最终 cfg 合并正确。 (disable recaps in /config)
a740692 Add experiment progress monitor script
acdd76b Add PowerShell launcher for scalarization experiments
07a6f9d Add scalarization experiment run guide
ffd655d Add scalarization_Exp framework for objective preprocessing comparison
18320bf Merge branch 'main' of https://github.com/Li-XuanYi/BO_Multi_12_20
0be649f 整理文件夹 并且新添锂电池数据集Ecker Oregan 新添了DISK PIMD 同时权重选取存在一定问题
ce9a897 找回seed8409的plot
3196e56 增加了遗传算法的实验，绘制了hv对比折线图。
7ecc7f0 文件夹整理
8d6fdce 准备清理文件
1c1a67b llmbo_vs_parego_optimal_protocols_5seeds_2026_05_06 的Optimal_Protocol 以及 D:\Users\aa133\Desktop\BO_Multi_12_20\New_LLMBO\analysis_runs\llmbo_vs_parego_seed8409_figures_2026_05_06_reference_localstd 下的对比实验
8961ee8 实验还不错
12b70c4 不错 做了对比实验和消融实验
5fd6b8b chore: snapshot warmstart plain GP plain EI before Codex changes
0dad378 seed=1
4e7b1af seed=2效果不错（50轮）其中从 eval 10，以及 eval 14 ~ 28LLMEI > WarmStart > Baseline
7a40c5b 优化EI准备
c550e4d Please enter the commit message for your changes. Lines starting with '#' will be ignored, and an empty message aborts the commit.
02d3626 	renamed:    New_LLMBO/.claude/command/codeestimate.md -> New_LLMBO/.claude/commands/codeestimate.md 	new file:   New_LLMBO/analysis_runs/current_like/database.json 	new file:   New_LLMBO/analysis_runs/current_like/db_final.json 	new file:   New_LLMBO/analysis_runs/current_like/pareto_front.json 	new file:   New_LLMBO/analysis_runs/current_like/summary.json 	new file:   New_LLMBO/analysis_runs/no_llm_strict/database.json 	new file:   New_LLMBO/analysis_runs/no_llm_strict/db_final.json 	new file:   New_LLMBO/analysis_runs/no_llm_strict/pareto_front.json 	new file:   New_LLMBO/analysis_runs/no_llm_strict/summary.json 	new file:   New_LLMBO/analysis_runs/no_warmstart_guidance_on/database.json 	new file:   New_LLMBO/analysis_runs/no_warmstart_guidance_on/db_final.json 	new file:   New_LLMBO/analysis_runs/no_warmstart_guidance_on/pareto_front.json 	new file:   New_LLMBO/analysis_runs/no_warmstart_guidance_on/summary.json 	modified:   New_LLMBO/llm/__pycache__/llm_interface.cpython-310.pyc 	modified:   New_LLMBO/llm/__pycache__/llm_interface.cpython-38.pyc 	modified:   New_LLMBO/llm/llm_interface.py 	new file:   New_LLMBO/llm/llm_interface_backup.py 	new file:   New_LLMBO/test_llm_interface.py 	new file:   "New_LLMBO/\345\267\245\344\275\234\346\265\201\345\210\206\346\236\220.md"
