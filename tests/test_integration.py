"""
集成测试 - 验证修改后的 PyBaMM 仿真器与整个系统兼容
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_compatibility():
    """测试配置与仿真器的兼容性"""
    print("=" * 60)
    print("测试 1: 配置兼容性")
    print("=" * 60)

    from config.schema import Config

    cfg = Config()

    # 检查电池参数
    print(f"  param_set: {cfg.battery.param_set}")
    print(f"  init_voltage: {cfg.battery.init_voltage} V")
    print(f"  init_temp: {cfg.battery.init_temp} K")
    print(f"  temp_max: {cfg.battery.temp_max} K")
    print(f"  voltage_max: {cfg.battery.voltage_max} V")
    print(f"  soc_target: {cfg.battery.soc_target}")

    # 检查决策变量边界
    print(f"\n  决策变量边界:")
    for key in ["I1", "I2", "I3", "dSOC1", "dSOC2"]:
        bounds = cfg.param_bounds.to_dict()[key]
        print(f"    {key}: {bounds}")

    print("\n✓ 配置兼容性测试通过\n")


def test_optimizer_import():
    """测试优化器导入"""
    print("=" * 60)
    print("测试 2: 优化器导入")
    print("=" * 60)

    try:
        from llmbo.optimizer import BayesOptimizer
        print("  ✓ BayesOptimizer 导入成功")
    except ImportError as e:
        print(f"  ✗ BayesOptimizer 导入失败：{e}")
        raise

    try:
        from pybamm_simulator import PyBaMMSimulator
        print("  ✓ PyBaMMSimulator 导入成功")
    except ImportError as e:
        print(f"  ✗ PyBaMMSimulator 导入失败：{e}")
        raise

    print("\n✓ 优化器导入测试通过\n")


def test_simulator_with_config():
    """测试使用配置创建仿真器"""
    print("=" * 60)
    print("测试 3: 仿真器与配置集成")
    print("=" * 60)

    from config.schema import Config
    from pybamm_simulator import PyBaMMSimulator

    cfg = Config()

    # 创建仿真器
    sim = PyBaMMSimulator(
        Q_nom=5.0,  # Chen2020 default
        SOH=1.0,
        T_init=cfg.battery.init_temp,
        V_init=cfg.battery.init_voltage,
        T_max=cfg.battery.temp_max,
        V_max=cfg.battery.voltage_max,
        use_crate=False,
        aging_weight=10.0,
    )

    print(f"  仿真器参数:")
    print(f"    battery_name: {sim.battery_name}")
    print(f"    Q_eff: {sim.Q_eff} Ah")
    print(f"    T_init: {sim.T_init} K")
    print(f"    V_init: {sim.V_init} V")
    print(f"    T_max: {sim.T_max} K")
    print(f"    V_max: {sim.V_max} V")

    # 运行一次评估
    theta = np.array([4.0, 3.5, 2.5, 0.25, 0.25])
    result = sim.evaluate(theta)

    print(f"\n  评估结果:")
    print(f"    feasible: {result['feasible']}")
    if result['feasible']:
        obj = result['raw_objectives']
        print(f"    objectives: [{obj[0]/60:.1f}min, {obj[1]:.2f}K, {obj[2]:.4f}%]")

    print("\n✓ 仿真器与配置集成测试通过\n")


def test_batch_evaluation():
    """测试批量评估"""
    print("=" * 60)
    print("测试 4: 批量评估")
    print("=" * 60)

    from pybamm_simulator import PyBaMMSimulator

    sim = PyBaMMSimulator()

    # 生成一些测试样本
    np.random.seed(42)
    thetas = []
    for _ in range(5):
        I1 = np.random.uniform(2.5, 5.0)
        I2 = np.random.uniform(2.5, 4.5)
        I3 = np.random.uniform(2.0, 3.5)
        dSOC1 = np.random.uniform(0.15, 0.35)
        dSOC2 = np.random.uniform(0.15, 0.30)
        thetas.append([I1, I2, I3, dSOC1, dSOC2])

    thetas = np.array(thetas)
    print(f"  评估 {len(thetas)} 个样本...")

    results = sim.evaluate_batch(thetas)

    feasible_count = sum(1 for r in results if r['feasible'])
    print(f"  可行解数量：{feasible_count}/{len(thetas)}")

    if feasible_count > 0:
        feasible_results = [r for r in results if r['feasible']]
        avg_time = np.mean([r['raw_objectives'][0]/60 for r in feasible_results])
        avg_temp = np.mean([r['raw_objectives'][1] for r in feasible_results])
        avg_aging = np.mean([r['raw_objectives'][2] for r in feasible_results])
        print(f"  平均目标值：[{avg_time:.1f}min, {avg_temp:.2f}K, {avg_aging:.4f}%]")

    print("\n✓ 批量评估测试通过\n")


def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("=" * 60)
    print("测试 5: 参数敏感性")
    print("=" * 60)

    from pybamm_simulator import PyBaMMSimulator

    sim = PyBaMMSimulator()

    # 测试不同的电流组合
    test_cases = [
        ("低电流", [2.5, 2.5, 2.0, 0.30, 0.30]),
        ("中电流", [4.0, 3.5, 2.5, 0.25, 0.25]),
        ("高电流", [5.5, 4.5, 3.0, 0.20, 0.20]),
    ]

    print("  不同电流策略的效果:")
    for name, theta in test_cases:
        result = sim.evaluate(np.array(theta))
        if result['feasible']:
            obj = result['raw_objectives']
            print(f"    {name}: time={obj[0]/60:.1f}min, ΔT={obj[1]:.1f}K, aging={obj[2]:.4f}%")
        else:
            print(f"    {name}: 不可行 - {result['violation']}")

    print("\n✓ 参数敏感性测试通过\n")


def main():
    """运行所有集成测试"""
    print("\n")
    print("#" * 60)
    print("# 集成测试 - PyBaMM 仿真器与系统兼容性验证")
    print("#" * 60)
    print()

    try:
        test_config_compatibility()
        test_optimizer_import()
        test_simulator_with_config()
        test_batch_evaluation()
        test_parameter_sensitivity()

        print("=" * 60)
        print("所有集成测试通过！✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"集成测试失败：{e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
